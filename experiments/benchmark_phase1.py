"""Phase-1 benchmark + 4 ablations (decision/002 §"Required ablations").

Runs four ablations on a fixed toy scene + 4 target images + 32 candidate
viewpoints. Saves results to ``results/phase1_<timestamp>.json`` with the
M1 (mosaic SSIM gain per view) trace for every condition.

The four ablations, in order, are:

    1. Random viewpoint baseline. If M1 doesn't beat random we don't ship
       (decision/002).
    2. Sinkhorn vs. Hungarian on identical viewpoints. Confirms the
       matching change is the source of M1 movement, not a confound.
    3. Saliency on/off. Confirms mosaicraft's saliency reuse contributes.
    4. Oklab on/off. Same.

The Hungarian baseline calls ``scipy.optimize.linear_sum_assignment``
directly — this is *not* a fallback in the matching layer (decision/003
forbids that). It is a separate comparison path, only constructed inside
this benchmark harness, and never exported as a public API.

Compute budget: target is < 1 h on a single GPU (we use CPU here);
Phase-1 keeps tile counts small enough that a full sweep runs in
< 5 min on a laptop CPU.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2  # used inside _build_cost via cv2.cvtColor
import numpy as np
from scipy.optimize import linear_sum_assignment

from mosaicraft_active_vision.cost import (
    bgr_to_oklab,
    compute_cost_matrix,
    compute_grid_oklab_means,
    compute_saliency_weights,
    compute_tile_oklab_means,
    extract_features,
)
from mosaicraft_active_vision.matching import argmax_assignment, sinkhorn_ot
from mosaicraft_active_vision.metrics import mosaic_ssim_gain
from mosaicraft_active_vision.nbv import (
    MosaicSsimGainStrategy,
    RandomStrategy,
    run_nbv_loop,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Phase-1 dimensions (small but representative)
# ---------------------------------------------------------------------------
TILE_SIZE = 16
GRID_COLS = 6
GRID_ROWS = 6
N_CELLS = GRID_COLS * GRID_ROWS  # 36
IMG_W = TILE_SIZE * GRID_COLS  # 96
IMG_H = TILE_SIZE * GRID_ROWS  # 96
N_VIEWS = 32
TILES_PER_VIEW = 4  # 32 * 4 = 128 tiles total — exceeds 36 cells comfortably
N_TARGETS = 4
MAX_NBV_STEPS = 16
EPSILON = 0.05
SINKHORN_MAX_ITER = 100


# ---------------------------------------------------------------------------
# Toy scene & synthetic data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToyScene:
    """A deterministic synthetic scene with view-specific palettes.

    Each viewpoint has a fixed base colour (drawn from a wide HSV ring)
    and emits ``tiles_per_view`` tiles that vary mildly around that
    base colour. Targets are built from a small subset of view palettes
    (``target_view_subset``) — therefore only views *whose base colour
    matches the target's palette* are useful. This makes the strategy
    axis non-trivial: a uniform-random strategy wastes calls on
    irrelevant views, while a saliency-biased strategy that prefers
    views close to the target's palette gets to high SSIM faster.
    """

    rng_seed: int = 0
    n_views: int = N_VIEWS
    tiles_per_view: int = TILES_PER_VIEW
    tile_size: int = TILE_SIZE

    def view_base_color(self, idx: int) -> NDArray:
        """Return the (3,) uint8 base BGR colour of viewpoint ``idx``."""
        # Evenly spaced hue ring so the palette covers Oklab space well.
        if not 0 <= idx < self.n_views:
            raise ValueError(f"viewpoint {idx} out of range [0, {self.n_views})")
        hue = int(180 * idx / self.n_views)  # OpenCV HSV hue is [0, 180)
        hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr[0, 0]

    def view_image(self, idx: int) -> NDArray:
        rng = np.random.default_rng(seed=self.rng_seed * 1000 + idx)
        base = self.view_base_color(idx).astype(np.int16)
        side = self.tile_size * self.tiles_per_view
        img = np.empty((self.tile_size, side, 3), dtype=np.uint8)
        for t in range(self.tiles_per_view):
            tile = np.tile(base, (self.tile_size, self.tile_size, 1))
            tile += rng.integers(-15, 16, size=tile.shape, dtype=np.int16)
            img[:, t * self.tile_size : (t + 1) * self.tile_size] = np.clip(tile, 0, 255).astype(
                np.uint8
            )
        return img

    def tiles_from_observations(self, observations: list[NDArray]) -> list[NDArray]:
        tiles: list[NDArray] = []
        for obs in observations:
            n_tiles_in_obs = obs.shape[1] // self.tile_size
            for t in range(n_tiles_in_obs):
                tiles.append(obs[:, t * self.tile_size : (t + 1) * self.tile_size].copy())
        return tiles

    def palette_distance_to_target(self, target_bgr: NDArray) -> NDArray:
        """Return ``(n_views,)`` Oklab distance from each view's base
        colour to the target's mean Oklab colour.

        Lower distance = more useful view. The saliency_biased strategy
        uses ``1 / distance`` as its per-view sampling weight.
        """
        target_ok_mean = bgr_to_oklab(target_bgr).reshape(-1, 3).mean(axis=0)
        bases = np.stack(
            [
                bgr_to_oklab(self.view_base_color(i).reshape(1, 1, 3)).reshape(3)
                for i in range(self.n_views)
            ]
        )
        return np.linalg.norm(bases - target_ok_mean[None, :], axis=1)


def make_targets(rng_seed: int, scene: ToyScene, n: int = N_TARGETS) -> list[NDArray]:
    """Build ``n`` targets, each from a small subset of view palettes.

    Each target picks ``subset_size`` view indices uniformly, then for
    every grid cell samples one of those palettes (perturbed). So
    perfectly matching the target needs at most ``subset_size`` views;
    seeing the other ``n_views - subset_size`` views is wasted.
    """
    rng = np.random.default_rng(seed=rng_seed)
    subset_size = max(2, scene.n_views // 8)  # 32/8 = 4 informative views
    targets: list[NDArray] = []
    for _ in range(n):
        chosen = rng.choice(scene.n_views, size=subset_size, replace=False)
        target = np.empty((IMG_H, IMG_W, 3), dtype=np.uint8)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                view_idx = int(chosen[rng.integers(0, subset_size)])
                base = scene.view_base_color(view_idx).astype(np.int16)
                perturb = rng.integers(-15, 16, size=3, dtype=np.int16)
                cell_color = np.clip(base + perturb, 0, 255).astype(np.uint8)
                target[
                    r * TILE_SIZE : (r + 1) * TILE_SIZE,
                    c * TILE_SIZE : (c + 1) * TILE_SIZE,
                ] = cell_color
        targets.append(target)
    return targets


# ---------------------------------------------------------------------------
# Mosaic builders (one per matching algorithm)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuilderConfig:
    """Toggles for ablations 3 (saliency) and 4 (Oklab)."""

    use_saliency: bool = True
    oklab_weight: float = 0.20


def _build_cost(
    tiles: list[NDArray],
    target_bgr: NDArray,
    cfg: BuilderConfig,
) -> NDArray:
    """Shared cost-matrix construction used by both matchers."""
    # mosaicraft.features.extract_features takes CIELAB float32 + tile_size.
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    grid_feats = np.empty((N_CELLS, 191), dtype=np.float64)
    grid_oklab = compute_grid_oklab_means(
        target_bgr, grid_cols=GRID_COLS, grid_rows=GRID_ROWS, tile_size=TILE_SIZE
    )
    cell_idx = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            patch_lab = target_lab[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ]
            grid_feats[cell_idx] = extract_features(patch_lab, TILE_SIZE)
            cell_idx += 1

    tile_lab_list = [cv2.cvtColor(t, cv2.COLOR_BGR2LAB).astype(np.float32) for t in tiles]
    tile_feats = np.stack([extract_features(t_lab, TILE_SIZE) for t_lab in tile_lab_list], axis=0)
    tile_oklab = compute_tile_oklab_means(tiles)

    saliency = None
    if cfg.use_saliency:
        # Real mosaicraft.saliency.compute_saliency_weights (edges +
        # Laplacian energy + HSV saturation + center bias). The
        # ablation A3 isolates this term.
        target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        sal_grid = compute_saliency_weights(
            target_gray,
            target_bgr,
            grid_cols=GRID_COLS,
            grid_rows=GRID_ROWS,
            tile_size=TILE_SIZE,
        )
        saliency = sal_grid.reshape(-1).astype(np.float64)

    return compute_cost_matrix(
        grid_feats,
        tile_feats,
        grid_oklab,
        tile_oklab,
        saliency_weights=saliency,
        oklab_weight=cfg.oklab_weight,
        normalize=True,
    )


def _paste_tiles(tiles: list[NDArray], assignment: NDArray) -> NDArray:
    out = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    for cell_idx, tile_idx in enumerate(assignment):
        r, c = divmod(cell_idx, GRID_COLS)
        out[
            r * TILE_SIZE : (r + 1) * TILE_SIZE,
            c * TILE_SIZE : (c + 1) * TILE_SIZE,
        ] = tiles[int(tile_idx)]
    return out


def build_mosaic_sinkhorn(tiles: list[NDArray], target_bgr: NDArray, cfg: BuilderConfig) -> NDArray:
    cost = _build_cost(tiles, target_bgr, cfg)
    n, m = cost.shape
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(m, 1.0 / m, dtype=np.float64)
    res = sinkhorn_ot(cost, a=a, b=b, epsilon=EPSILON, max_iter=SINKHORN_MAX_ITER, tol=1e-7)
    assignment = argmax_assignment(res.plan, enforce_unique=False)
    return _paste_tiles(tiles, assignment)


def build_mosaic_hungarian(
    tiles: list[NDArray], target_bgr: NDArray, cfg: BuilderConfig
) -> NDArray:
    """Ablation-only Hungarian baseline. Not exported anywhere else."""
    cost = _build_cost(tiles, target_bgr, cfg)
    n, m = cost.shape
    if m < n:
        # Pad with high-cost columns so Hungarian has enough tiles.
        pad = np.full((n, n - m), float(cost.max()) + 1.0)
        cost_padded = np.concatenate([cost, pad], axis=1)
        _, cols = linear_sum_assignment(cost_padded)
        # Cells whose Hungarian choice is a padding column re-pick best real tile.
        assignment = np.where(cols < m, cols, np.argmin(cost, axis=1))
    else:
        _, cols = linear_sum_assignment(cost)
        assignment = cols
    return _paste_tiles(tiles, np.asarray(assignment, dtype=np.int64))


# ---------------------------------------------------------------------------
# One condition = (strategy, matcher, builder config)
# ---------------------------------------------------------------------------
@dataclass
class ConditionResult:
    name: str
    matcher: str
    strategy: str
    use_saliency: bool
    oklab_weight: float
    target_idx: int
    ssim_scores: list[float] = field(default_factory=list)
    ssim_gain: list[float] = field(default_factory=list)
    view_indices: list[int] = field(default_factory=list)
    final_ssim: float = 0.0
    sum_gain: float = 0.0
    elapsed_seconds: float = 0.0


def run_condition(
    *,
    name: str,
    matcher: str,
    strategy_name: str,
    target_idx: int,
    target_bgr: NDArray,
    scene: ToyScene,
    cfg: BuilderConfig,
    rng_seed: int,
    max_steps: int = MAX_NBV_STEPS,
) -> ConditionResult:
    if matcher == "sinkhorn":

        def builder(tiles: list[NDArray], tgt: NDArray) -> NDArray:
            return build_mosaic_sinkhorn(tiles, tgt, cfg)

    elif matcher == "hungarian":

        def builder(tiles: list[NDArray], tgt: NDArray) -> NDArray:
            return build_mosaic_hungarian(tiles, tgt, cfg)

    else:
        raise ValueError(f"unknown matcher {matcher!r}")

    if strategy_name == "random":
        strategy = RandomStrategy()
    elif strategy_name == "saliency_biased":
        # Target-conditioned per-view utility: views whose base palette
        # is closer (in Oklab) to the target's mean palette are
        # exponentially more likely to be picked. This is a cheap proxy
        # for the saliency-weighted-transport surrogate that the
        # eventual MosaicSsimGainStrategy will compute exactly.
        dist = scene.palette_distance_to_target(target_bgr)
        per_view = np.exp(-dist / max(float(dist.std()), 1e-3))
        strategy = MosaicSsimGainStrategy(saliency_per_view=per_view)
    else:
        raise ValueError(f"unknown strategy {strategy_name!r}")

    t0 = time.perf_counter()
    out = run_nbv_loop(
        target_bgr=target_bgr,
        candidate_view_indices=list(range(scene.n_views)),
        view_sampler=scene.view_image,
        tile_extractor=scene.tiles_from_observations,
        mosaic_builder=builder,
        strategy=strategy,
        max_steps=max_steps,
        rng=np.random.default_rng(seed=rng_seed),
    )
    elapsed = time.perf_counter() - t0

    gain = mosaic_ssim_gain(out.history).tolist()
    scores = list(out.history.ssim_scores)
    return ConditionResult(
        name=name,
        matcher=matcher,
        strategy=strategy_name,
        use_saliency=cfg.use_saliency,
        oklab_weight=cfg.oklab_weight,
        target_idx=target_idx,
        ssim_scores=scores,
        ssim_gain=gain,
        view_indices=list(out.history.view_indices),
        final_ssim=scores[-1],
        sum_gain=float(sum(gain)),
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Ablation matrix
# ---------------------------------------------------------------------------
def ablation_matrix() -> list[dict]:
    """Return the (name, matcher, strategy, cfg) tuples to run.

    The matrix is chosen so that each ablation question can be answered
    by holding all other axes fixed and varying one:

      A1 (strategy):    random          vs  saliency_biased  — fix matcher=sinkhorn, sal=on, ok=0.2
      A2 (matcher):     sinkhorn        vs  hungarian        — fix strategy=saliency_biased, sal=on, ok=0.2
      A3 (saliency):    sal_on          vs  sal_off          — fix matcher=sinkhorn, strategy=saliency_biased, ok=0.2
      A4 (oklab):       ok=0.2          vs  ok=0.0           — fix matcher=sinkhorn, strategy=saliency_biased, sal=on

    Common reference run is "sinkhorn+saliency_biased+sal_on+ok=0.2".
    """
    full_cfg = BuilderConfig(use_saliency=True, oklab_weight=0.20)
    no_sal_cfg = BuilderConfig(use_saliency=False, oklab_weight=0.20)
    no_ok_cfg = BuilderConfig(use_saliency=True, oklab_weight=0.0)

    return [
        {"name": "A1_random", "matcher": "sinkhorn", "strategy": "random", "cfg": full_cfg},
        {
            "name": "A1_saliency",
            "matcher": "sinkhorn",
            "strategy": "saliency_biased",
            "cfg": full_cfg,
        },
        {
            "name": "A2_hungarian",
            "matcher": "hungarian",
            "strategy": "saliency_biased",
            "cfg": full_cfg,
        },
        {
            "name": "A3_saliency_off",
            "matcher": "sinkhorn",
            "strategy": "saliency_biased",
            "cfg": no_sal_cfg,
        },
        {
            "name": "A4_oklab_off",
            "matcher": "sinkhorn",
            "strategy": "saliency_biased",
            "cfg": no_ok_cfg,
        },
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_phase1(
    *,
    output_dir: Path,
    scene_seed: int = 0,
    target_seed: int = 1,
    nbv_seed: int = 7,
    n_targets: int = N_TARGETS,
    max_nbv_steps: int = MAX_NBV_STEPS,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = ToyScene(rng_seed=scene_seed)
    targets = make_targets(target_seed, scene, n=n_targets)
    matrix = ablation_matrix()

    results: list[ConditionResult] = []
    for cond in matrix:
        print(f"\n=== {cond['name']} ===")
        for t_idx, target in enumerate(targets):
            r = run_condition(
                name=cond["name"],
                matcher=cond["matcher"],
                strategy_name=cond["strategy"],
                target_idx=t_idx,
                target_bgr=target,
                scene=scene,
                cfg=cond["cfg"],
                rng_seed=nbv_seed + 31 * t_idx,
                max_steps=max_nbv_steps,
            )
            results.append(r)
            print(
                f"  target={t_idx} "
                f"final_ssim={r.final_ssim:.4f} "
                f"sum_gain={r.sum_gain:+.4f} "
                f"t={r.elapsed_seconds:.1f}s"
            )

    # Aggregate per condition
    summary = summarize(results)

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_path = output_dir / f"phase1_{ts}.json"
    payload = {
        "schema_version": 1,
        "scene_seed": scene_seed,
        "target_seed": target_seed,
        "nbv_seed": nbv_seed,
        "n_targets": n_targets,
        "n_views": N_VIEWS,
        "max_nbv_steps": max_nbv_steps,
        "sinkhorn_epsilon": EPSILON,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print_summary(summary)
    print(f"\nFull JSON: {out_path}")
    return out_path


def summarize(results: Iterable[ConditionResult]) -> dict[str, dict[str, float]]:
    by_name: dict[str, list[ConditionResult]] = {}
    for r in results:
        by_name.setdefault(r.name, []).append(r)

    summary: dict[str, dict[str, float]] = {}
    for name, runs in by_name.items():
        finals = np.array([r.final_ssim for r in runs])
        gains = np.array([r.sum_gain for r in runs])
        elapsed = np.array([r.elapsed_seconds for r in runs])
        summary[name] = {
            "mean_final_ssim": float(finals.mean()),
            "std_final_ssim": float(finals.std()),
            "mean_sum_gain": float(gains.mean()),
            "mean_elapsed_seconds": float(elapsed.mean()),
            "n_targets": len(runs),
        }
    return summary


def print_summary(summary: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 70)
    print(f"{'condition':<22} {'final_ssim':>14} {'sum_gain':>12} {'t/run (s)':>12}")
    print("-" * 70)
    for name in sorted(summary):
        s = summary[name]
        print(
            f"{name:<22} "
            f"{s['mean_final_ssim']:>10.4f}±{s['std_final_ssim']:.3f} "
            f"{s['mean_sum_gain']:>+12.4f} "
            f"{s['mean_elapsed_seconds']:>12.2f}"
        )
    print("=" * 70)
    print("\nAblation reading guide (decision/002 §Required ablations):")
    print("  A1: A1_saliency MUST beat A1_random (else don't ship).")
    print("  A2: A1_saliency vs A2_hungarian shows the matcher's contribution.")
    print("  A3: A1_saliency vs A3_saliency_off shows the saliency contribution.")
    print("  A4: A1_saliency vs A4_oklab_off    shows the Oklab contribution.")


def main() -> None:
    import os

    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / "experiments" / "results"
    if os.environ.get("MOSAICRAFT_AV_BENCH_SMOKE"):
        # CI smoke run: 1 target, 2 NBV steps. Verifies the harness
        # imports + runs end-to-end without burning the CI minute budget.
        run_phase1(output_dir=out, n_targets=1, max_nbv_steps=2)
    else:
        run_phase1(output_dir=out)


if __name__ == "__main__":
    main()
