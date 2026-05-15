"""Next-best-view loop driven by photomosaic reconstruction loss.

This is the core scientific contribution of the repo, per
``decision/000-charter.md``: a viewpoint planner whose objective is the
**photomosaic SSIM gain** rather than the standard NBV objective of
voxel coverage.

The loop is intentionally simulator-agnostic. It depends only on three
caller-provided callables:

    1. ``view_sampler(view_index)``    -- returns the BGR image observed
                                          from candidate viewpoint ``view_index``.
    2. ``tile_extractor(observations)`` -- given the set of observations
                                          so far, returns the current
                                          tile pool (list of BGR tiles).
    3. ``mosaic_builder(tiles, target)`` -- builds a photomosaic of
                                            ``target`` using ``tiles``,
                                            using the Sinkhorn-OT
                                            assignment from this repo.

Two strategies are provided:

    * ``RandomStrategy``         -- the ablation baseline (decision/002 #1).
    * ``MosaicSsimGainStrategy`` -- our proposal: pick the view that
                                    *predicts* the highest SSIM gain via
                                    a cheap surrogate (saliency-weighted
                                    transport-cost-reduction estimate).

The strategy interface is deliberately tiny so adding more baselines
(coverage-greedy, view-entropy, etc.) does not require touching the
loop itself.

This module deliberately does **not** import torch. Differentiable
viewpoint selection lives in a future module under ``extras=["gpu"]``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .metrics import MosaicHistory, mosaic_ssim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "MosaicBuilder",
    "MosaicSsimGainStrategy",
    "NbvLoopResult",
    "RandomStrategy",
    "TileExtractor",
    "ViewSampler",
    "ViewStrategy",
    "run_nbv_loop",
]


# ---------------------------------------------------------------------------
# Callable protocols (so the loop is simulator-agnostic)
# ---------------------------------------------------------------------------
class ViewSampler(Protocol):
    """Return the BGR image observed from candidate viewpoint ``index``."""

    def __call__(self, index: int) -> NDArray: ...


class TileExtractor(Protocol):
    """Build the current tile pool from accumulated observations."""

    def __call__(self, observations: Sequence[NDArray]) -> list[NDArray]: ...


class MosaicBuilder(Protocol):
    """Assemble a photomosaic of ``target_bgr`` using ``tiles``.

    Must internally use Sinkhorn-OT (decision/003) — the loop assumes
    the assignment respects the saliency-marginal coupling.
    """

    def __call__(self, tiles: list[NDArray], target_bgr: NDArray) -> NDArray: ...


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
class ViewStrategy(abc.ABC):
    """Pick the next viewpoint index from a set of remaining candidates."""

    @abc.abstractmethod
    def select(
        self,
        *,
        remaining: list[int],
        observations: list[NDArray],
        ssim_history: list[float],
        rng: np.random.Generator,
    ) -> int:
        """Return one element of ``remaining``."""


@dataclass(frozen=True)
class RandomStrategy(ViewStrategy):
    """Uniform random selection — decision/002 ablation #1 baseline."""

    def select(
        self,
        *,
        remaining: list[int],
        observations: list[NDArray],
        ssim_history: list[float],
        rng: np.random.Generator,
    ) -> int:
        if not remaining:
            raise ValueError("RandomStrategy called with no remaining viewpoints")
        return int(rng.choice(remaining))


@dataclass(frozen=True)
class MosaicSsimGainStrategy(ViewStrategy):
    """Pick the view whose predicted SSIM gain is largest.

    The predictor is intentionally cheap: we score each candidate by the
    expected reduction in saliency-weighted Oklab variance of the
    target after that view's tiles are added. We deliberately do not
    actually run Sinkhorn-OT inside the predictor — that would defeat
    the whole point of "active vision" (think before you look).

    Phase 1 uses a stand-in surrogate (random with a saliency bias) so
    the loop is end-to-end runnable before the predictor is fully
    derived. Replacing this with the proper predictor is a future PR
    and is gated by decision/002 ablation #1 producing signal first.
    """

    saliency_per_view: NDArray | None = None
    """Optional ``(num_views,)`` per-view saliency score. If provided,
    views with higher saliency are more likely to be selected — a
    saliency-biased random baseline that should already beat plain
    random in decision/002 ablation #3."""

    def select(
        self,
        *,
        remaining: list[int],
        observations: list[NDArray],
        ssim_history: list[float],
        rng: np.random.Generator,
    ) -> int:
        if not remaining:
            raise ValueError("MosaicSsimGainStrategy called with no remaining viewpoints")
        if self.saliency_per_view is None:
            return int(rng.choice(remaining))
        weights = self.saliency_per_view[np.asarray(remaining, dtype=np.int64)]
        weights = np.maximum(weights, 1e-12)
        weights = weights / weights.sum()
        return int(rng.choice(remaining, p=weights))


# ---------------------------------------------------------------------------
# Loop result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NbvLoopResult:
    """Output of one NBV run."""

    history: MosaicHistory
    """SSIM trace + view indices."""

    mosaics: list[NDArray] = field(default_factory=list)
    """One photomosaic per step. Optional; populated only if requested."""

    strategy_name: str = ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_nbv_loop(
    *,
    target_bgr: NDArray,
    candidate_view_indices: Sequence[int],
    view_sampler: ViewSampler,
    tile_extractor: TileExtractor,
    mosaic_builder: MosaicBuilder,
    strategy: ViewStrategy,
    max_steps: int,
    initial_view: int | None = None,
    keep_mosaics: bool = False,
    rng: np.random.Generator | None = None,
) -> NbvLoopResult:
    """Run an active-vision loop and return the M1 trace.

    Parameters
    ----------
    target_bgr : np.ndarray
        The image the photomosaic is trying to reproduce.
    candidate_view_indices : sequence of int
        Pool of viewpoint indices the strategy can choose from.
    view_sampler, tile_extractor, mosaic_builder : Protocols
        See module docstring.
    strategy : ViewStrategy
    max_steps : int
        Number of next-best-views to take after the initial one.
    initial_view : int or None
        The starting viewpoint. If ``None``, picks ``candidate_view_indices[0]``.
    keep_mosaics : bool
        Materialize and store every photomosaic (memory-heavy for big
        scenes; default off).
    rng : np.random.Generator or None
        Source of randomness for random/biased strategies. ``None`` =
        ``np.random.default_rng()``.

    Returns
    -------
    NbvLoopResult
    """
    if max_steps < 0:
        raise ValueError(f"max_steps must be non-negative; got {max_steps}")
    if not candidate_view_indices:
        raise ValueError("candidate_view_indices is empty")

    rng = rng if rng is not None else np.random.default_rng()
    remaining = list(candidate_view_indices)

    first = initial_view if initial_view is not None else remaining[0]
    if first not in remaining:
        raise ValueError(f"initial_view {first} not in candidate_view_indices")
    remaining.remove(first)

    observations: list[NDArray] = [view_sampler(first)]
    tiles = tile_extractor(observations)
    mosaic0 = mosaic_builder(tiles, target_bgr)
    ssim0 = mosaic_ssim(mosaic0, target_bgr)

    ssim_trace: list[float] = [ssim0]
    view_trace: list[int] = [first]
    mosaics: list[NDArray] = [mosaic0] if keep_mosaics else []

    for _ in range(max_steps):
        if not remaining:
            break
        next_view = strategy.select(
            remaining=remaining,
            observations=observations,
            ssim_history=ssim_trace,
            rng=rng,
        )
        if next_view not in remaining:
            raise ValueError(
                f"strategy {type(strategy).__name__} returned {next_view}, "
                f"which is not in remaining={remaining[:8]}..."
            )
        remaining.remove(next_view)
        observations.append(view_sampler(next_view))
        tiles = tile_extractor(observations)
        mosaic_k = mosaic_builder(tiles, target_bgr)
        ssim_k = mosaic_ssim(mosaic_k, target_bgr)
        ssim_trace.append(ssim_k)
        view_trace.append(next_view)
        if keep_mosaics:
            mosaics.append(mosaic_k)

    history = MosaicHistory(
        ssim_scores=tuple(ssim_trace),
        view_indices=tuple(view_trace),
    )
    return NbvLoopResult(
        history=history,
        mosaics=mosaics,
        strategy_name=type(strategy).__name__,
    )
