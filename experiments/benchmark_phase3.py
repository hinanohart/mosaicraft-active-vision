"""Phase-3 benchmark — Oklch hue-rotation pool aug x NBV.

`decision/006-phase2-findings.md` left the photomosaic-NBV story with
two pieces of unfinished business:

    1. Every result so far ran on a hue-ring toy. Whether the
       Hungarian-beats-Sinkhorn verdict generalises to images with
       structure beyond pure colour patches is open.
    2. The mosaicraft family ships an Oklch hue-rotation pool
       expander (now extracted to ``oklch-aug``, see
       ``decision/007-oklch-aug-extraction.md``) that was never
       benchmarked inside the NBV loop. The expansion produces
       L-preserving variants of every tile, which should help any
       matcher whose ``n_tiles >= n_cells`` constraint is tight, but
       may also help Sinkhorn more than Hungarian because soft OT
       can distribute mass over the new candidates.

This file is the harness for both. It deliberately reuses Phase 2's
builders (``build_mosaic_hungarian`` and ``build_mosaic_marginal``)
without modification so the matcher-side comparison stays
apples-to-apples; the only new ingredient is a tile-pool augmenter
that wraps any builder.

The current scene is still the Phase-1 hue ring — real images are
the next gate (decision/006 Phase-3 backlog), wired here as
``--real-images-dir`` for when they land.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from oklch_aug import HueRotatePool

from experiments.benchmark_phase1 import (
    GRID_COLS,
    GRID_ROWS,
    MAX_NBV_STEPS,
    N_TARGETS,
    TILE_SIZE,
    ToyScene,
    make_targets,
)
from experiments.benchmark_phase2 import (
    bootstrap_ci_diff,
    build_mosaic_hungarian,
    build_mosaic_marginal,
)
from mosaicraft_active_vision.metrics import mosaic_ssim_gain
from mosaicraft_active_vision.nbv import MosaicSsimGainStrategy, run_nbv_loop

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Phase-3 sweep dimensions
# ---------------------------------------------------------------------------
N_SEEDS = 4
SINKHORN_EPSILON = 0.1  # Phase-2 best (decision/006 §Headline).
OKLCH_N_VARIANTS = 4  # 5x pool: originals + 4 hue rotations.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def _is_smoke() -> bool:
    return bool(os.environ.get("MOSAICRAFT_AV_BENCH_SMOKE"))


# ---------------------------------------------------------------------------
# Tile-pool augmentation wrapper
# ---------------------------------------------------------------------------
def expand_tiles_oklch(
    tiles: list[NDArray],
    *,
    n_variants: int = OKLCH_N_VARIANTS,
    chroma_scale: float = 1.0,
) -> list[NDArray]:
    """Apply :class:`oklch_aug.HueRotatePool` to a list of BGR tiles.

    The toy scene's tiles live in BGR uint8 (because mosaicraft is a
    cv2-native pipeline); ``HueRotatePool`` honours that with
    ``channel_order="bgr"`` so the augmenter is colour-identical to
    the upstream ``mosaicraft.color_augment.expand_color_variants``
    minus the ``TileSet`` glue.
    """
    pool = HueRotatePool(
        n_variants=n_variants,
        chroma_scale=chroma_scale,
        channel_order="bgr",
    )
    return pool(tiles)


# ---------------------------------------------------------------------------
# Builders (no-aug and oklch-aug variants of Phase-2 builders)
# ---------------------------------------------------------------------------
def _wrap_with_oklch(
    base_builder,
    *,
    n_variants: int,
):
    """Return a builder that expands the tile pool before delegating."""

    def builder(tiles: list[NDArray], target_bgr: NDArray) -> NDArray:
        expanded = expand_tiles_oklch(tiles, n_variants=n_variants)
        return base_builder(expanded, target_bgr)

    return builder


def builder_hungarian_no_aug(tiles: list[NDArray], target_bgr: NDArray) -> NDArray:
    return build_mosaic_hungarian(tiles, target_bgr)


def builder_hungarian_oklch_aug(tiles: list[NDArray], target_bgr: NDArray) -> NDArray:
    return _wrap_with_oklch(build_mosaic_hungarian, n_variants=OKLCH_N_VARIANTS)(tiles, target_bgr)


def builder_sinkhorn_no_aug(tiles: list[NDArray], target_bgr: NDArray) -> NDArray:
    return build_mosaic_marginal(tiles, target_bgr, epsilon=SINKHORN_EPSILON, enforce_unique=False)


def builder_sinkhorn_oklch_aug(tiles: list[NDArray], target_bgr: NDArray) -> NDArray:
    def base(t: list[NDArray], tgt: NDArray) -> NDArray:
        return build_mosaic_marginal(t, tgt, epsilon=SINKHORN_EPSILON, enforce_unique=False)

    return _wrap_with_oklch(base, n_variants=OKLCH_N_VARIANTS)(tiles, target_bgr)


CONDITIONS: dict[str, object] = {
    "hungarian_no_aug": builder_hungarian_no_aug,
    "hungarian_oklch_aug": builder_hungarian_oklch_aug,
    "sinkhorn_no_aug": builder_sinkhorn_no_aug,
    "sinkhorn_oklch_aug": builder_sinkhorn_oklch_aug,
}


# ---------------------------------------------------------------------------
# Result types and runner
# ---------------------------------------------------------------------------
@dataclass
class P3Result:
    name: str
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
    builder,
    seed: int,
    target_idx: int,
    target_bgr: NDArray,
    scene: ToyScene,
    max_steps: int,
) -> P3Result:
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
    return P3Result(
        name=name,
        seed=seed,
        target_idx=target_idx,
        final_ssim=scores[-1],
        sum_gain=float(sum(gain)),
        elapsed_seconds=elapsed,
        ssim_scores=scores,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_phase3(
    *,
    output_dir: Path,
    scene_seed: int = 0,
    target_seed: int = 1,
    n_seeds: int | None = None,
    n_targets: int | None = None,
    max_nbv_steps: int | None = None,
) -> Path:
    smoke = _is_smoke()
    if n_seeds is None:
        n_seeds = 1 if smoke else N_SEEDS
    if n_targets is None:
        n_targets = 1 if smoke else N_TARGETS
    if max_nbv_steps is None:
        max_nbv_steps = 4 if smoke else MAX_NBV_STEPS

    scene = ToyScene(rng_seed=scene_seed)
    targets = make_targets(rng_seed=target_seed, scene=scene, n=n_targets)
    results: list[P3Result] = []
    for cond_name, builder in CONDITIONS.items():
        if smoke and cond_name not in {"hungarian_no_aug", "sinkhorn_oklch_aug"}:
            # Smoke runs only the two extremes to stay under a few seconds.
            continue
        print(f"\n=== {cond_name} ===")
        for seed in range(n_seeds):
            for t_idx, target_bgr in enumerate(targets):
                r = _run_one(
                    name=cond_name,
                    builder=builder,
                    seed=seed,
                    target_idx=t_idx,
                    target_bgr=target_bgr,
                    scene=scene,
                    max_steps=max_nbv_steps,
                )
                results.append(r)
                print(
                    f"  seed={seed} target={t_idx} "
                    f"final_ssim={r.final_ssim:.4f} sum_gain={r.sum_gain:+.4f} "
                    f"t={r.elapsed_seconds:.1f}s"
                )

    summary = summarize(results)
    print_summary(summary, results)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = output_dir / f"phase3_{stamp}.json"
    payload = {
        "schema_version": 1,
        "phase": 3,
        "n_seeds": n_seeds,
        "n_targets": n_targets,
        "grid": {"cols": GRID_COLS, "rows": GRID_ROWS, "tile_size": TILE_SIZE},
        "oklch_n_variants": OKLCH_N_VARIANTS,
        "sinkhorn_epsilon": SINKHORN_EPSILON,
        "results": [asdict(r) for r in results],
        "summary": summary,
        "smoke": smoke,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nFull JSON: {out_path}")
    return out_path


def summarize(results: list[P3Result]) -> dict[str, dict[str, float]]:
    by_name: dict[str, list[float]] = {}
    for r in results:
        by_name.setdefault(r.name, []).append(r.final_ssim)
    return {
        name: {
            "mean_final_ssim": float(np.mean(scores)),
            "std_final_ssim": float(np.std(scores)),
            "n_runs": len(scores),
        }
        for name, scores in by_name.items()
    }


def print_summary(summary: dict[str, dict[str, float]], results: list[P3Result]) -> None:
    print()
    print("=" * 84)
    print(f"{'condition':<30} {'final_ssim':>20} {'oklch_aug_delta (95% CI)':>30}")
    print("-" * 84)

    # Pair up no-aug vs oklch-aug per matcher for the headline question.
    by_pair = [
        ("hungarian_no_aug", "hungarian_oklch_aug"),
        ("sinkhorn_no_aug", "sinkhorn_oklch_aug"),
    ]
    paired_scores: dict[str, list[float]] = {}
    for r in results:
        paired_scores.setdefault(r.name, []).append(r.final_ssim)

    for cond, stat in summary.items():
        print(f"{cond:<30} {stat['mean_final_ssim']:.4f}±{stat['std_final_ssim']:.3f}")

    print("-" * 84)
    for no_aug, with_aug in by_pair:
        if no_aug not in paired_scores or with_aug not in paired_scores:
            continue
        x = np.asarray(paired_scores[with_aug], dtype=np.float64)
        y = np.asarray(paired_scores[no_aug], dtype=np.float64)
        if x.shape != y.shape:
            continue
        mean_diff, lo, hi = bootstrap_ci_diff(x=x, y=y)
        sign = "BEATS" if lo > 0 else ("WORSE" if hi < 0 else "TIE")
        matcher = no_aug.replace("_no_aug", "")
        print(f"{matcher:<30} oklch_aug Δ = {mean_diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  {sign}")
    print("=" * 84)


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    run_phase3(output_dir=output_dir)


if __name__ == "__main__":
    main()
