"""Sanity tests for ``mosaicraft_active_vision.cost``.

The mosaicraft submodule provides the heavy primitives (features,
saliency, Oklab); these tests verify that the wrappers in ``cost.py``
preserve shapes, weighting, and rejection of malformed input.
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaicraft_active_vision.cost import (
    compute_cost_matrix,
    compute_grid_oklab_means,
    compute_tile_oklab_means,
    oklab_distance,
)


# ---------------------------------------------------------------------------
# Oklab means
# ---------------------------------------------------------------------------
def test_grid_oklab_means_shape() -> None:
    rng = np.random.default_rng(seed=0)
    target = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    means = compute_grid_oklab_means(target, grid_cols=4, grid_rows=4, tile_size=8)
    assert means.shape == (16, 3)
    assert means.dtype == np.float64


def test_grid_oklab_means_constant_image_constant_mean() -> None:
    target = np.full((32, 32, 3), 128, dtype=np.uint8)
    means = compute_grid_oklab_means(target, grid_cols=4, grid_rows=4, tile_size=8)
    # All cells share the same Oklab mean for a constant image
    np.testing.assert_allclose(means - means[0:1], 0.0, atol=1e-9)


def test_tile_oklab_means_empty_input() -> None:
    out = compute_tile_oklab_means([])
    assert out.shape == (0, 3)


def test_tile_oklab_means_shape() -> None:
    tiles = [
        np.full((8, 8, 3), 30, dtype=np.uint8),
        np.full((8, 8, 3), 220, dtype=np.uint8),
    ]
    out = compute_tile_oklab_means(tiles)
    assert out.shape == (2, 3)
    # Brighter tile must have larger L coordinate
    assert out[1, 0] > out[0, 0]


# ---------------------------------------------------------------------------
# Oklab distance
# ---------------------------------------------------------------------------
def test_oklab_distance_zero_for_identical() -> None:
    means = np.array([[0.5, 0.0, 0.0], [0.7, 0.1, -0.1]])
    d = oklab_distance(means, means)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)


def test_oklab_distance_symmetric_full_matrix() -> None:
    rng = np.random.default_rng(seed=1)
    a = rng.uniform(0, 1, size=(5, 3))
    b = rng.uniform(0, 1, size=(7, 3))
    d_ab = oklab_distance(a, b)
    d_ba = oklab_distance(b, a)
    assert d_ab.shape == (5, 7)
    np.testing.assert_allclose(d_ab, d_ba.T, atol=1e-12)


def test_oklab_distance_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        oklab_distance(np.zeros((4, 2)), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Full cost matrix
# ---------------------------------------------------------------------------
def _make_features(n: int, dim: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(0, 1, size=(n, dim)).astype(np.float64)


def test_compute_cost_matrix_shape() -> None:
    n, m, dim = 6, 9, 191
    gf = _make_features(n, dim, seed=0)
    tf = _make_features(m, dim, seed=1)
    g_ok = _make_features(n, 3, seed=2)
    t_ok = _make_features(m, 3, seed=3)
    cost = compute_cost_matrix(gf, tf, g_ok, t_ok, saliency_weights=None)
    assert cost.shape == (n, m)
    assert cost.dtype == np.float64
    assert np.all(np.isfinite(cost))
    assert np.all(cost >= 0)


def test_compute_cost_matrix_saliency_scales_rows() -> None:
    """Doubling saliency on row i must roughly double row i of the cost."""
    n, m, dim = 4, 5, 191
    gf = _make_features(n, dim, seed=10)
    tf = _make_features(m, dim, seed=11)
    g_ok = _make_features(n, 3, seed=12)
    t_ok = _make_features(m, 3, seed=13)

    base = compute_cost_matrix(gf, tf, g_ok, t_ok, saliency_weights=None)
    sal = np.ones(n)
    sal[2] = 2.0
    scaled = compute_cost_matrix(gf, tf, g_ok, t_ok, saliency_weights=sal)

    np.testing.assert_allclose(scaled[0], base[0])
    np.testing.assert_allclose(scaled[1], base[1])
    np.testing.assert_allclose(scaled[2], 2.0 * base[2])
    np.testing.assert_allclose(scaled[3], base[3])


def test_compute_cost_matrix_rejects_feature_dim_mismatch() -> None:
    gf = _make_features(3, 10, seed=0)
    tf = _make_features(4, 12, seed=1)
    g_ok = _make_features(3, 3, seed=2)
    t_ok = _make_features(4, 3, seed=3)
    with pytest.raises(ValueError, match="feature dim mismatch"):
        compute_cost_matrix(gf, tf, g_ok, t_ok, saliency_weights=None)


def test_compute_cost_matrix_rejects_saliency_length_mismatch() -> None:
    n, m, dim = 3, 4, 191
    gf = _make_features(n, dim, seed=0)
    tf = _make_features(m, dim, seed=1)
    g_ok = _make_features(n, 3, seed=2)
    t_ok = _make_features(m, 3, seed=3)
    with pytest.raises(ValueError, match="saliency"):
        compute_cost_matrix(gf, tf, g_ok, t_ok, saliency_weights=np.ones(n + 1))


def test_compute_cost_matrix_oklab_weight_zero_isolates_feature_term() -> None:
    """With oklab_weight=0 and no saliency, cost depends only on
    features — proves the convex combination structure."""
    n, m, dim = 3, 4, 191
    gf = _make_features(n, dim, seed=20)
    tf = _make_features(m, dim, seed=21)
    g_ok1 = _make_features(n, 3, seed=22)
    g_ok2 = _make_features(n, 3, seed=222)
    t_ok = _make_features(m, 3, seed=23)

    c1 = compute_cost_matrix(gf, tf, g_ok1, t_ok, None, oklab_weight=0.0)
    c2 = compute_cost_matrix(gf, tf, g_ok2, t_ok, None, oklab_weight=0.0)
    np.testing.assert_allclose(c1, c2, atol=1e-12)
