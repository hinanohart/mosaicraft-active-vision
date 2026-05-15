"""Primary and diagnostic metrics for the active-vision loop.

Per ``decision/002-evaluation-metric.md``:

* **M1 — Mosaic-SSIM-gain per view (PRIMARY).**
  ``delta_ssim_k = ssim(mosaic_k, target) - ssim(mosaic_{k-1}, target)``.
  Sample efficiency is the smallest ``k`` such that ``ssim(mosaic_k, target)``
  exceeds a user-set threshold.

* **M2 — View coverage (DIAGNOSTIC, sanity-check vs. GenNBV).**
  Fraction of scene voxels that have been observed at least once.

These are the only metrics the repo reports as headline numbers.
Anything else is opt-in diagnostic.

References
----------
Wang, Z. et al. (2004). *Image quality assessment: from error visibility
    to structural similarity.* IEEE Trans. Image Process., 13(4).
GenNBV (CVPR 2024) — the NBV baseline whose coverage metric we report
    as M2 so reviewers can sanity-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from skimage.metrics import structural_similarity as _skimage_ssim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "MosaicHistory",
    "mosaic_ssim",
    "mosaic_ssim_gain",
    "sample_efficiency",
    "view_coverage",
]


# ---------------------------------------------------------------------------
# SSIM (mosaic vs. target)
# ---------------------------------------------------------------------------
def mosaic_ssim(
    mosaic_bgr: NDArray,
    target_bgr: NDArray,
    *,
    data_range: float | None = None,
    channel_axis: int | None = -1,
) -> float:
    """Structural similarity between a photomosaic and its target.

    Parameters
    ----------
    mosaic_bgr : np.ndarray
        Photomosaic image, BGR or grayscale.
    target_bgr : np.ndarray
        Target image, **same shape and dtype** as ``mosaic_bgr``.
    data_range : float or None
        Maximum value of the data type. Defaults to ``255`` for uint8 /
        ``1.0`` for floats.
    channel_axis : int or None, default -1
        Per skimage 0.22 API. ``None`` for grayscale, ``-1`` for HxWx3.

    Returns
    -------
    float
        SSIM in ``[-1, 1]``. Higher is better.
    """
    if mosaic_bgr.shape != target_bgr.shape:
        raise ValueError(f"shape mismatch: mosaic {mosaic_bgr.shape} vs target {target_bgr.shape}")
    if data_range is None:
        if mosaic_bgr.dtype == np.uint8:
            data_range = 255.0
        elif np.issubdtype(mosaic_bgr.dtype, np.floating):
            data_range = 1.0
        else:
            raise ValueError(
                f"data_range required for dtype {mosaic_bgr.dtype!r}; pass explicitly."
            )
    # skimage SSIM needs at least win_size pixels per axis (default 7).
    # For tiny test images we shrink win_size to fit, but skimage requires
    # win_size <= min(image side), so reject anything below 3 up-front.
    min_side = int(min(mosaic_bgr.shape[:2]))
    if min_side < 3:
        raise ValueError(f"image too small for SSIM: {mosaic_bgr.shape}. Need min side >= 3.")
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    return float(
        _skimage_ssim(
            mosaic_bgr,
            target_bgr,
            data_range=data_range,
            channel_axis=channel_axis,
            win_size=win_size,
        )
    )


# ---------------------------------------------------------------------------
# History container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MosaicHistory:
    """A sequence of photomosaic snapshots and their SSIM scores.

    Element ``k`` is the photomosaic produced **after** the k-th
    next-best-view has been visited (k = 0 corresponds to the initial
    observation). The active-vision loop appends one snapshot per
    NBV iteration.
    """

    ssim_scores: tuple[float, ...]
    """SSIM(mosaic_k, target) for k = 0, 1, ..., K. Length K+1."""

    view_indices: tuple[int, ...]
    """Index of the view chosen at step k (matches ssim_scores length)."""

    def __post_init__(self) -> None:
        if len(self.ssim_scores) != len(self.view_indices):
            raise ValueError(
                "ssim_scores and view_indices must have the same length; "
                f"got {len(self.ssim_scores)} and {len(self.view_indices)}"
            )


# ---------------------------------------------------------------------------
# M1 primary: SSIM gain per view
# ---------------------------------------------------------------------------
def mosaic_ssim_gain(history: MosaicHistory) -> NDArray:
    """Return ``delta_ssim_k`` per view as a 1-D array.

    Parameters
    ----------
    history : MosaicHistory
        Output of an NBV run.

    Returns
    -------
    np.ndarray, shape ``(K,)``
        ``out[k-1] = ssim_scores[k] - ssim_scores[k-1]`` for k = 1..K.
        Length is ``len(history.ssim_scores) - 1``.
    """
    scores = np.asarray(history.ssim_scores, dtype=np.float64)
    if scores.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(scores)


def sample_efficiency(
    history: MosaicHistory,
    *,
    threshold: float,
) -> int | None:
    """Smallest ``k`` such that ``ssim_scores[k] >= threshold``.

    Returns ``None`` if the threshold is never reached. The primary use
    is comparing strategies: random vs. saliency-Sinkhorn vs. baseline
    NBV should produce different ``k`` values for the same threshold.

    Parameters
    ----------
    history : MosaicHistory
    threshold : float
        Target SSIM value.

    Returns
    -------
    int or None
    """
    scores = np.asarray(history.ssim_scores, dtype=np.float64)
    above = np.where(scores >= threshold)[0]
    if above.size == 0:
        return None
    return int(above[0])


# ---------------------------------------------------------------------------
# M2 diagnostic: view coverage
# ---------------------------------------------------------------------------
def view_coverage(
    observed_masks: Sequence[NDArray],
) -> float:
    """Fraction of scene voxels (or pixels) observed at least once.

    Parameters
    ----------
    observed_masks : sequence of np.ndarray of bool
        Each element is the binary observation mask of a single view.
        All masks must share the same shape.

    Returns
    -------
    float in [0, 1]
        Union-area over total-area.
    """
    if not observed_masks:
        return 0.0
    union = np.zeros_like(observed_masks[0], dtype=bool)
    shape = observed_masks[0].shape
    for i, mask in enumerate(observed_masks):
        if mask.shape != shape:
            raise ValueError(f"mask {i} has shape {mask.shape}; expected {shape}")
        union |= mask.astype(bool)
    return float(union.sum() / union.size)
