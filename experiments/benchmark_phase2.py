"""Phase-2 benchmark — decision/004 §"Phase 2 backlog".

Phase 1 found that Hungarian outperformed Sinkhorn-OT on the toy
scene (`decision/004`). The two leading hypotheses that decision/004
records are:

    (H1) Sinkhorn's ε=0.05 was a single point, never swept; with
         a tighter ε the entropic plan becomes harder and may
         match Hungarian.
    (H2) Saliency was applied as a row-scale on the cost matrix,
         which is mosaicraft's Hungarian-era hack; the actual OT
         formulation (decision/003 §Algorithm sketch) treats
         saliency as the **source marginal**, not as a cost
         multiplier. Switching to the marginal framing is the
         change Sinkhorn was designed for.

This file exercises both hypotheses on N=8 seeds (vs. the 4 used in
Phase 1) so the mean differences carry a credible interval. Each
condition reports mean ± std *and* a bootstrap 95% CI for the
difference vs. the Hungarian baseline, so the "did we move the
needle?" question has a statistical, not eyeballed, answer.

The Hungarian baseline is built exactly as in
``benchmark_phase1.py`` so the two phases are comparable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Reuse the Phase-1 toy scene unchanged so the comparison is apples-to-apples.
from experiments.benchmark_phase1 import (
    GRID_COLS,
    GRID_ROWS,
    MAX_NBV_STEPS,
    N_CELLS,
    N_TARGETS,
    N_VIEWS,
    SINKHORN_MAX_ITER,
    TILE_SIZE,
    ToyScene,
    _paste_tiles,
    make_targets,
)
from mosaicraft_active_vision.cost import (
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
    run_nbv_loop,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Phase-2 sweep dimensions
# ---------------------------------------------------------------------------
N_SEEDS = 8  # Phase 1 used 1; we sweep over 8 to get a real std.
EPSILONS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2)


# ---------------------------------------------------------------------------
# Cost matrix variants
# ---------------------------------------------------------------------------
def _build_features(
    target_bgr: NDArray,
    tiles: list[NDArray],
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return (grid_feats, tile_feats, grid_oklab, tile_oklab)."""
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
    return grid_feats, tile_feats, grid_oklab, tile_oklab


def _saliency_vec(target_bgr: NDArray) -> NDArray:
    """Saliency per grid cell (the (N_CELLS,) flattened vector)."""
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    sal_grid = compute_saliency_weights(
        target_gray,
        target_bgr,
        grid_cols=GRID_COLS,
        grid_rows=GRID_ROWS,
        tile_size=TILE_SIZE,
    )
    return sal_grid.reshape(-1).astype(np.float64)


def build_mosaic_rowscale(
    tiles: list[NDArray],
    target_bgr: NDArray,
    *,
    epsilon: float,
    enforce_unique: bool,
    oklab_weight: float = 0.20,
) -> NDArray:
    """Phase-1 framing: saliency multiplied row-wise on the cost matrix.

    Source marginal ``a`` is uniform. This is the framing that lost
    to Hungarian in decision/004.
    """
    grid_feats, tile_feats, grid_oklab, tile_oklab = _build_features(target_bgr, tiles)
    saliency = _saliency_vec(target_bgr)
    cost = compute_cost_matrix(
        grid_feats,
        tile_feats,
        grid_oklab,
        tile_oklab,
        saliency_weights=saliency,
        oklab_weight=oklab_weight,
        normalize=True,
    )
    n, m = cost.shape
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(m, 1.0 / m, dtype=np.float64)
    res = sinkhorn_ot(cost, a=a, b=b, epsilon=epsilon, max_iter=SINKHORN_MAX_ITER, tol=1e-7)
    assignment = argmax_assignment(res.plan, enforce_unique=enforce_unique)
    return _paste_tiles(tiles, assignment)


def build_mosaic_marginal(
    tiles: list[NDArray],
    target_bgr: NDArray,
    *,
    epsilon: float,
    enforce_unique: bool,
    oklab_weight: float = 0.20,
) -> NDArray:
    """Decision/003 framing: saliency IS the OT source marginal.

    Cost is built WITHOUT row-scaling. The OT problem becomes:
    given the cell-side mass distribution (saliency-weighted), how
    do tiles map onto cells to minimise transport cost?
    """
    grid_feats, tile_feats, grid_oklab, tile_oklab = _build_features(target_bgr, tiles)
    cost = compute_cost_matrix(
        grid_feats,
        tile_feats,
        grid_oklab,
        tile_oklab,
        saliency_weights=None,  # critical: NO row-scaling
        oklab_weight=oklab_weight,
        normalize=True,
    )
    saliency = _saliency_vec(target_bgr)
    a = saliency / saliency.sum()  # NORMALISED, used as marginal
    _, m = cost.shape
    b = np.full(m, 1.0 / m, dtype=np.float64)
    res = sinkhorn_ot(cost, a=a, b=b, epsilon=epsilon, max_iter=SINKHORN_MAX_ITER, tol=1e-7)
    assignment = argmax_assignment(res.plan, enforce_unique=enforce_unique)
    return _paste_tiles(tiles, assignment)


def build_mosaic_hungarian(
    tiles: list[NDArray],
    target_bgr: NDArray,
    *,
    oklab_weight: float = 0.20,
) -> NDArray:
    """Phase-1 Hungarian baseline. Reproduced here so multi-seed runs
    use identical code paths.
    """
    grid_feats, tile_feats, grid_oklab, tile_oklab = _build_features(target_bgr, tiles)
    saliency = _saliency_vec(target_bgr)
    cost = compute_cost_matrix(
        grid_feats,
        tile_feats,
        grid_oklab,
        tile_oklab,
        saliency_weights=saliency,
        oklab_weight=oklab_weight,
        normalize=True,
    )
    n, m = cost.shape
    if m < n:
        pad = np.full((n, n - m), float(cost.max()) + 1.0)
        cost_padded = np.concatenate([cost, pad], axis=1)
        _, cols = linear_sum_assignment(cost_padded)
        assignment = np.where(cols < m, cols, np.argmin(cost, axis=1))
    else:
        _, cols = linear_sum_assignment(cost)
        assignment = cols
    return _paste_tiles(tiles, np.asarray(assignment, dtype=np.int64))


# ---------------------------------------------------------------------------
# One condition x one (seed, target) run
# ---------------------------------------------------------------------------
@dataclass
class P2Result:
    name: str
    epsilon: float | None
    enforce_unique: bool
    seed: int
    target_idx: int
    final_ssim: float
    sum_gain: float
    elapsed_seconds: float
    ssim_scores: list[float] = field(default_factory=list)


def _strategy(scene: ToyScene, target_bgr: NDArray) -> MosaicSsimGainStrategy:
    dist = scene.palette_distance_to_target(target_bgr)
    per_view = np.exp(-dist / max(float(dist.std()), 1e-3))
    return MosaicSsimGainStrategy(saliency_per_view=per_view)


def _run_one(
    *,
    name: str,
    epsilon: float | None,
    enforce_unique: bool,
    seed: int,
    target_idx: int,
    target_bgr: NDArray,
    scene: ToyScene,
    builder_kind: str,
    max_steps: int = MAX_NBV_STEPS,
) -> P2Result:
    if builder_kind == "rowscale":
        assert epsilon is not None

        def builder(tiles: list[NDArray], tgt: NDArray) -> NDArray:
            return build_mosaic_rowscale(tiles, tgt, epsilon=epsilon, enforce_unique=enforce_unique)

    elif builder_kind == "marginal":
        assert epsilon is not None

        def builder(tiles: list[NDArray], tgt: NDArray) -> NDArray:
            return build_mosaic_marginal(tiles, tgt, epsilon=epsilon, enforce_unique=enforce_unique)

    elif builder_kind == "hungarian":

        def builder(tiles: list[NDArray], tgt: NDArray) -> NDArray:
            return build_mosaic_hungarian(tiles, tgt)

    else:
        raise ValueError(builder_kind)

    t0 = time.perf_counter()
    out = run_nbv_loop(
        target_bgr=target_bgr,
        candidate_view_indices=list(range(scene.n_views)),
        view_sampler=scene.view_image,
        tile_extractor=scene.tiles_from_observations,
        mosaic_builder=builder,
        strategy=_strategy(scene, target_bgr),
        max_steps=max_steps,
        rng=np.random.default_rng(seed=seed),
    )
    elapsed = time.perf_counter() - t0
    gain = mosaic_ssim_gain(out.history).tolist()
    scores = list(out.history.ssim_scores)
    return P2Result(
        name=name,
        epsilon=epsilon,
        enforce_unique=enforce_unique,
        seed=seed,
        target_idx=target_idx,
        final_ssim=scores[-1],
        sum_gain=float(sum(gain)),
        elapsed_seconds=elapsed,
        ssim_scores=scores,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI for the mean difference (paired by (seed, target))
# ---------------------------------------------------------------------------
def bootstrap_ci_diff(
    *,
    x: NDArray,
    y: NDArray,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    rng_seed: int = 17,
) -> tuple[float, float, float]:
    """Return (mean_diff, lo, hi) of x - y, paired indices.

    x and y must have identical length; element i of each comes from
    the same (seed, target) pair.
    """
    assert x.shape == y.shape and x.ndim == 1
    diff = x - y
    rng = np.random.default_rng(rng_seed)
    n = diff.shape[0]
    means: NDArray = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return float(diff.mean()), lo, hi


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_phase2(
    *,
    output_dir: Path,
    scene_seed: int = 0,
    target_seed: int = 1,
    n_seeds: int = N_SEEDS,
    n_targets: int = N_TARGETS,
    epsilons: tuple[float, ...] = EPSILONS,
    max_nbv_steps: int = MAX_NBV_STEPS,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = ToyScene(rng_seed=scene_seed)
    targets = make_targets(target_seed, scene, n=n_targets)
    results: list[P2Result] = []

    # --- Hungarian baseline (single matcher, no ε sweep needed)
    print("\n=== Hungarian baseline ===")
    for sd in range(n_seeds):
        for t_idx, target in enumerate(targets):
            r = _run_one(
                name="hungarian",
                epsilon=None,
                enforce_unique=False,
                seed=7 + 31 * sd + 91 * t_idx,
                target_idx=t_idx,
                target_bgr=target,
                scene=scene,
                builder_kind="hungarian",
                max_steps=max_nbv_steps,
            )
            results.append(r)

    # --- Sinkhorn variants x ε sweep x enforce_unique on/off
    for builder_kind, label in [("rowscale", "rowscale"), ("marginal", "marginal")]:
        for eps in epsilons:
            for enforce_unique in (False, True):
                name = f"{label}_eps={eps:.3g}_unique={int(enforce_unique)}"
                print(f"\n=== {name} ===")
                for sd in range(n_seeds):
                    for t_idx, target in enumerate(targets):
                        r = _run_one(
                            name=name,
                            epsilon=eps,
                            enforce_unique=enforce_unique,
                            seed=7 + 31 * sd + 91 * t_idx,
                            target_idx=t_idx,
                            target_bgr=target,
                            scene=scene,
                            builder_kind=builder_kind,
                            max_steps=max_nbv_steps,
                        )
                        results.append(r)
                # one progress dot per condition so smoke runs stay quiet
                finals = np.array(
                    [r.final_ssim for r in results if r.name == name], dtype=np.float64
                )
                print(f"  {name}: mean_final_ssim = {finals.mean():.4f} ± {finals.std():.4f}")

    # --- Aggregate
    summary = summarize_p2(results)
    diffs = compare_to_hungarian(results)

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_path = output_dir / f"phase2_{ts}.json"
    payload = {
        "schema_version": 2,
        "scene_seed": scene_seed,
        "target_seed": target_seed,
        "n_seeds": n_seeds,
        "n_targets": n_targets,
        "n_views": N_VIEWS,
        "max_nbv_steps": max_nbv_steps,
        "epsilons": list(epsilons),
        "summary": summary,
        "vs_hungarian_bootstrap_ci": diffs,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print_p2_summary(summary, diffs)
    print(f"\nFull JSON: {out_path}")
    return out_path


def summarize_p2(results: list[P2Result]) -> dict[str, dict[str, float]]:
    by_name: dict[str, list[P2Result]] = {}
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
            "n_runs": len(runs),
        }
    return summary


def compare_to_hungarian(
    results: list[P2Result],
) -> dict[str, dict[str, float]]:
    """For every non-Hungarian condition, bootstrap CI of
    (mean_final_ssim - Hungarian_mean_final_ssim) paired by (seed, target).
    """
    hungarian_runs = sorted(
        [r for r in results if r.name == "hungarian"],
        key=lambda r: (r.target_idx, r.seed),
    )
    if not hungarian_runs:
        return {}
    hungarian_finals = np.array([r.final_ssim for r in hungarian_runs], dtype=np.float64)
    keyed = {(r.seed, r.target_idx): r for r in hungarian_runs}

    out: dict[str, dict[str, float]] = {}
    by_name: dict[str, list[P2Result]] = {}
    for r in results:
        if r.name == "hungarian":
            continue
        by_name.setdefault(r.name, []).append(r)

    for name, runs in by_name.items():
        # Align runs to hungarian baseline by (seed, target_idx).
        ours = []
        theirs = []
        for r in runs:
            key = (r.seed, r.target_idx)
            if key not in keyed:
                continue
            ours.append(r.final_ssim)
            theirs.append(keyed[key].final_ssim)
        if len(ours) < 2:
            continue
        x = np.array(ours, dtype=np.float64)
        y = np.array(theirs, dtype=np.float64)
        diff, lo, hi = bootstrap_ci_diff(x=x, y=y)
        out[name] = {
            "mean_diff_vs_hungarian": diff,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_pairs": int(x.size),
            "beats_hungarian_95ci": bool(lo > 0.0),
        }
    # Unused — keep import explicit in case callers want the raw vector
    _ = hungarian_finals
    return out


def print_p2_summary(
    summary: dict[str, dict[str, float]],
    diffs: dict[str, dict[str, float]],
) -> None:
    print("\n" + "=" * 88)
    print(f"{'condition':<40} {'final_ssim':>14} {'vs Hungarian (95% CI)':>30}")
    print("-" * 88)
    # Print hungarian first
    if "hungarian" in summary:
        s = summary["hungarian"]
        print(f"{'hungarian':<40} {s['mean_final_ssim']:>10.4f}±{s['std_final_ssim']:.3f}")
    for name in sorted(summary):
        if name == "hungarian":
            continue
        s = summary[name]
        d = diffs.get(name, {})
        diff_s = ""
        if d:
            mark = "✓ BEATS" if d["beats_hungarian_95ci"] else "—"
            diff_s = (
                f"Δ={d['mean_diff_vs_hungarian']:+.4f} "
                f"[{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}] {mark}"
            )
        print(f"{name:<40} {s['mean_final_ssim']:>10.4f}±{s['std_final_ssim']:.3f} {diff_s:>30}")
    print("=" * 88)
    print("\nReading guide:")
    print("  - rows with `✓ BEATS` have a 95% bootstrap CI strictly above 0,")
    print("    so on this toy at N=8x4 paired runs they outperform Hungarian.")
    print("  - rows with `—` are statistically tied with Hungarian or behind it.")


def main() -> None:
    import os

    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / "experiments" / "results"
    if os.environ.get("MOSAICRAFT_AV_BENCH_SMOKE"):
        # CI smoke: 2 seeds x 1 target x 2 ε x 2 unique flags x 2 builders
        # + Hungarian (2 seeds x 1 target) = ~18 runs * 2 NBV steps.
        run_phase2(
            output_dir=out,
            n_seeds=2,
            n_targets=1,
            epsilons=(0.01, 0.1),
            max_nbv_steps=2,
        )
    else:
        run_phase2(output_dir=out)


if __name__ == "__main__":
    main()
