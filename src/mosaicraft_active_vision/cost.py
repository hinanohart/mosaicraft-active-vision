"""Cost-matrix construction reusing mosaicraft's perceptual primitives.

This module is the bridge between the upstream `mosaicraft` codebase
(imported via the `external/mosaicraft` submodule) and this repo's
Sinkhorn-OT matcher in :mod:`mosaicraft_active_vision.matching`.

Three primitives are pulled from mosaicraft as-is:

    * ``mosaicraft.features.extract_features``      -- 191-dim color+texture
    * ``mosaicraft.saliency.compute_saliency_weights`` -- per-cell weights
    * ``mosaicraft.color.bgr_to_oklab``             -- Oklab transform [Ottosson 2020]

The only logic we **re-derive** here is the per-cell Oklab perceptual
distance: mosaicraft fuses Oklab distance with a top-K Hungarian
acceleration in ``placement.compute_cost_matrix`` (lines 98-103), so a
clean extraction is needed for the Sinkhorn-OT setting that does **not**
have a top-K restriction.

References
----------
Ottosson, B. (2020). *A perceptual color space for image processing.*
    https://bottosson.github.io/posts/oklab/
mosaicraft/placement.py L98-103 (commit 2918137) — the Hungarian-coupled
    Oklab fusion logic this module replaces.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Submodule import path bootstrap
# ---------------------------------------------------------------------------
# mosaicraft is intentionally not pip-installed (it depends on
# `opencv-python` while this repo pins `opencv-python-headless`; both
# expose `cv2` and conflict if both are installed). Instead we put the
# submodule's `src/` on `sys.path` so `import mosaicraft.features` works.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MOSAICRAFT_SRC = _REPO_ROOT / "external" / "mosaicraft" / "src"
if _MOSAICRAFT_SRC.is_dir():
    sys.path.insert(0, str(_MOSAICRAFT_SRC))
else:  # pragma: no cover - missing submodule is an installation error
    raise ImportError(
        f"mosaicraft submodule not found at {_MOSAICRAFT_SRC}. "
        "Run `git submodule update --init --recursive` from the repo root."
    )

from mosaicraft.color import bgr_to_oklab  # noqa: E402
from mosaicraft.features import FEATURE_DIM, extract_features  # noqa: E402
from mosaicraft.saliency import compute_saliency_weights  # noqa: E402

__all__ = [
    "FEATURE_DIM",
    "bgr_to_oklab",
    "compute_cost_matrix",
    "compute_grid_oklab_means",
    "compute_saliency_weights",
    "compute_tile_oklab_means",
    "extract_features",
    "oklab_distance",
]


# ---------------------------------------------------------------------------
# Oklab means
# ---------------------------------------------------------------------------
def compute_grid_oklab_means(
    target_bgr: NDArray,
    grid_cols: int,
    grid_rows: int,
    tile_size: int,
) -> NDArray:
    """Compute the per-cell mean Oklab color of a target image.

    Parameters
    ----------
    target_bgr : np.ndarray
        Target image in OpenCV BGR uint8, shape ``(H, W, 3)``.
    grid_cols, grid_rows : int
        Number of mosaic cells horizontally and vertically.
    tile_size : int
        Pixel side length of one cell.

    Returns
    -------
    np.ndarray
        Per-cell Oklab means, shape ``(n_cells, 3)``, float64,
        flattened in row-major order to match the cost matrix.
    """
    h, w = grid_rows * tile_size, grid_cols * tile_size
    oklab = bgr_to_oklab(target_bgr[:h, :w])
    # (grid_rows, tile_size, grid_cols, tile_size, 3) -> mean over tile pixels
    reshaped = oklab.reshape(grid_rows, tile_size, grid_cols, tile_size, 3)
    means = reshaped.mean(axis=(1, 3))  # (grid_rows, grid_cols, 3)
    return means.reshape(grid_rows * grid_cols, 3).astype(np.float64)


def compute_tile_oklab_means(tile_bgr_list: list[NDArray]) -> NDArray:
    """Compute the mean Oklab color of each tile.

    Parameters
    ----------
    tile_bgr_list : list of np.ndarray
        Tile images in BGR uint8.

    Returns
    -------
    np.ndarray
        Shape ``(n_tiles, 3)``, float64.
    """
    if not tile_bgr_list:
        return np.empty((0, 3), dtype=np.float64)
    means: NDArray = np.empty((len(tile_bgr_list), 3), dtype=np.float64)
    for i, tile in enumerate(tile_bgr_list):
        means[i] = bgr_to_oklab(tile).mean(axis=(0, 1))
    return means


# ---------------------------------------------------------------------------
# Oklab distance
# ---------------------------------------------------------------------------
def oklab_distance(grid_means: NDArray, tile_means: NDArray) -> NDArray:
    """Pairwise Euclidean Oklab distance between cells and tiles.

    Unlike ``mosaicraft.placement.compute_cost_matrix``, this computes
    the **full** ``(n_cells, n_tiles)`` matrix without a top-K
    restriction, because Sinkhorn-OT operates on the whole transport
    plan rather than on per-cell candidate shortlists.

    Parameters
    ----------
    grid_means : np.ndarray, shape ``(n_cells, 3)``
    tile_means : np.ndarray, shape ``(n_tiles, 3)``

    Returns
    -------
    np.ndarray, shape ``(n_cells, n_tiles)``, float64
        ``out[i, j] = || grid_means[i] - tile_means[j] ||_2`` in Oklab.
    """
    if grid_means.ndim != 2 or grid_means.shape[1] != 3:
        raise ValueError(f"grid_means must be (n, 3); got {grid_means.shape}")
    if tile_means.ndim != 2 or tile_means.shape[1] != 3:
        raise ValueError(f"tile_means must be (m, 3); got {tile_means.shape}")
    diff = grid_means[:, None, :] - tile_means[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2)).astype(np.float64)


# ---------------------------------------------------------------------------
# Full cost matrix
# ---------------------------------------------------------------------------
def compute_cost_matrix(
    grid_features: NDArray,
    tile_features: NDArray,
    grid_oklab_means: NDArray,
    tile_oklab_means: NDArray,
    saliency_weights: NDArray | None,
    *,
    oklab_weight: float = 0.20,
    normalize: bool = True,
) -> NDArray:
    """Build the full cost matrix for Sinkhorn-OT assignment.

    The cost is a convex combination of:

        * **Feature L2** in the 191-dim mosaicraft feature space
          (color quadrants + LAB histograms + gradients + LBP).
        * **Oklab perceptual distance** between per-cell and per-tile
          Oklab means, weighted by ``oklab_weight``.

    Saliency weights, when provided, multiplicatively scale each row
    so that perceptually important cells get larger cost penalties for
    mismatch (and therefore better tiles in the OT solution).

    The function deliberately omits the top-K mask present in
    mosaicraft's Hungarian-coupled version: Sinkhorn-OT must see the
    full cost matrix to spread mass via entropic regularization.

    Parameters
    ----------
    grid_features : np.ndarray, shape ``(n_cells, FEATURE_DIM)``
    tile_features : np.ndarray, shape ``(n_tiles, FEATURE_DIM)``
    grid_oklab_means : np.ndarray, shape ``(n_cells, 3)``
    tile_oklab_means : np.ndarray, shape ``(n_tiles, 3)``
    saliency_weights : np.ndarray or None
        Shape ``(grid_rows, grid_cols)`` or ``(n_cells,)``. If ``None``,
        cells are weighted uniformly.
    oklab_weight : float, default 0.20
        Mix weight for the Oklab term. Same default as
        ``mosaicraft.placement.compute_cost_matrix``.
    normalize : bool, default True
        Divide the feature term by its global max so the two terms are
        on comparable scales. Set ``False`` only if you have already
        normalized externally.

    Returns
    -------
    np.ndarray, shape ``(n_cells, n_tiles)``, float64
        Cost matrix ready to feed into ``matching.sinkhorn_ot``.
    """
    if grid_features.shape[1] != tile_features.shape[1]:
        raise ValueError(
            f"feature dim mismatch: {grid_features.shape[1]} vs {tile_features.shape[1]}"
        )

    # --- feature L2 ---
    # Vectorized squared Euclidean: ||a||^2 + ||b||^2 - 2 a.b
    gf = grid_features.astype(np.float64)
    tf = tile_features.astype(np.float64)
    gsq = np.sum(gf * gf, axis=1, keepdims=True)
    tsq = np.sum(tf * tf, axis=1, keepdims=True).T
    feat_l2 = gsq + tsq - 2.0 * (gf @ tf.T)
    np.maximum(feat_l2, 0.0, out=feat_l2)  # clip negatives from FP error

    if normalize:
        denom = float(feat_l2.max()) + 1e-12
        feat_l2 = feat_l2 / denom

    # --- Oklab perceptual distance, full matrix ---
    ok_dist = oklab_distance(grid_oklab_means, tile_oklab_means)
    cost = feat_l2 + oklab_weight * ok_dist

    # --- saliency row-scaling ---
    if saliency_weights is not None:
        sal_flat = np.asarray(saliency_weights, dtype=np.float64).reshape(-1)
        if sal_flat.shape[0] != cost.shape[0]:
            raise ValueError(f"saliency has {sal_flat.shape[0]} cells but cost has {cost.shape[0]}")
        cost = cost * sal_flat[:, None]

    return cost.astype(np.float64)
