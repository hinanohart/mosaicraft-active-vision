"""Tests for ``mosaicraft_active_vision.metrics``.

Two flavours of tests:

1. **Property tests** — invariants that hold for *any* valid input
   (length, sign, idempotence). These guard the math.
2. **Golden hash** — a deterministic numerical fingerprint on a fixed
   tiny scene, so silent numerical drift in upstream dependencies
   (skimage, numpy) is caught immediately.

The golden hash is computed at test time from a fixed RNG seed, then
asserted against a checked-in value. To re-bless after an intentional
change, run the test once with ``REGENERATE_GOLDEN=1`` and copy the
printed digest into this file.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from mosaicraft_active_vision.metrics import (
    MosaicHistory,
    mosaic_ssim,
    mosaic_ssim_gain,
    sample_efficiency,
    view_coverage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_tiny_scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mosaic, target)`` as 32x32 BGR uint8 from a fixed seed.

    Small enough that SSIM runs fast in CI but big enough that
    skimage's default win_size of 7 is usable.
    """
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    noise = rng.integers(-32, 33, size=(32, 32, 3), dtype=np.int16)
    mosaic = np.clip(target.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return mosaic, target


# ---------------------------------------------------------------------------
# mosaic_ssim — bounds + identity
# ---------------------------------------------------------------------------
def test_mosaic_ssim_identity_is_one() -> None:
    """SSIM(x, x) == 1 for any image."""
    rng = np.random.default_rng(seed=1)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    assert mosaic_ssim(img, img) == pytest.approx(1.0, abs=1e-9)


def test_mosaic_ssim_in_bounds() -> None:
    mosaic, target = _make_tiny_scene()
    s = mosaic_ssim(mosaic, target)
    assert -1.0 <= s <= 1.0


def test_mosaic_ssim_rejects_shape_mismatch() -> None:
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.zeros((10, 12, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        mosaic_ssim(a, b)


def test_mosaic_ssim_grayscale_path() -> None:
    rng = np.random.default_rng(seed=2)
    img = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    assert mosaic_ssim(img, img, channel_axis=None) == pytest.approx(1.0, abs=1e-9)


def test_mosaic_ssim_tiny_image_shrinks_window() -> None:
    """SSIM on a 5x5 image should not crash."""
    a = np.full((5, 5, 3), 100, dtype=np.uint8)
    b = np.full((5, 5, 3), 100, dtype=np.uint8)
    s = mosaic_ssim(a, b)
    assert s == pytest.approx(1.0, abs=1e-9)


def test_mosaic_ssim_rejects_too_small() -> None:
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        mosaic_ssim(a, a)


def test_mosaic_ssim_float_dtype_default_range() -> None:
    rng = np.random.default_rng(seed=3)
    a = rng.uniform(0, 1, size=(16, 16, 3)).astype(np.float32)
    assert mosaic_ssim(a, a) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# mosaic_ssim — golden hash
# ---------------------------------------------------------------------------
# Re-bless: REGENERATE_GOLDEN=1 pytest tests/test_metrics.py -k golden -s
GOLDEN_SSIM_DIGEST_SEED0 = "869bdbdb"


def test_mosaic_ssim_golden_hash() -> None:
    mosaic, target = _make_tiny_scene(seed=0)
    s = mosaic_ssim(mosaic, target)
    digest = hashlib.sha256(f"{s:.10f}".encode()).hexdigest()[:8]
    if os.environ.get("REGENERATE_GOLDEN"):
        print(f"\nNEW GOLDEN SSIM DIGEST (seed=0): {digest} (ssim={s:.10f})")
        return
    assert digest == GOLDEN_SSIM_DIGEST_SEED0, (
        f"SSIM golden digest drifted: got {digest}, expected "
        f"{GOLDEN_SSIM_DIGEST_SEED0}. Re-bless with REGENERATE_GOLDEN=1."
    )


# ---------------------------------------------------------------------------
# MosaicHistory + mosaic_ssim_gain
# ---------------------------------------------------------------------------
def test_mosaic_history_validates_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        MosaicHistory(ssim_scores=(0.1, 0.2), view_indices=(0,))


def test_mosaic_ssim_gain_is_diff() -> None:
    h = MosaicHistory(ssim_scores=(0.10, 0.30, 0.45, 0.50), view_indices=(0, 1, 2, 3))
    gain = mosaic_ssim_gain(h)
    np.testing.assert_allclose(gain, [0.20, 0.15, 0.05])


def test_mosaic_ssim_gain_empty_when_single_step() -> None:
    h = MosaicHistory(ssim_scores=(0.1,), view_indices=(0,))
    gain = mosaic_ssim_gain(h)
    assert gain.shape == (0,)


def test_mosaic_ssim_gain_sums_to_total_delta() -> None:
    """Telescoping: sum of per-view gains == final - initial."""
    rng = np.random.default_rng(seed=4)
    scores = tuple(np.sort(rng.uniform(0.2, 0.9, size=8)).tolist())
    h = MosaicHistory(ssim_scores=scores, view_indices=tuple(range(8)))
    gain = mosaic_ssim_gain(h)
    assert gain.sum() == pytest.approx(scores[-1] - scores[0])


# ---------------------------------------------------------------------------
# sample_efficiency
# ---------------------------------------------------------------------------
def test_sample_efficiency_threshold_reached() -> None:
    h = MosaicHistory(ssim_scores=(0.1, 0.4, 0.6, 0.8), view_indices=(0, 1, 2, 3))
    assert sample_efficiency(h, threshold=0.5) == 2


def test_sample_efficiency_threshold_unreached() -> None:
    h = MosaicHistory(ssim_scores=(0.1, 0.2, 0.3), view_indices=(0, 1, 2))
    assert sample_efficiency(h, threshold=0.9) is None


def test_sample_efficiency_initial_already_above() -> None:
    h = MosaicHistory(ssim_scores=(0.95, 0.96), view_indices=(0, 1))
    assert sample_efficiency(h, threshold=0.5) == 0


# ---------------------------------------------------------------------------
# view_coverage (M2)
# ---------------------------------------------------------------------------
def test_view_coverage_empty() -> None:
    assert view_coverage([]) == 0.0


def test_view_coverage_full_union() -> None:
    a = np.array([[True, False], [False, True]])
    b = np.array([[False, True], [True, False]])
    assert view_coverage([a, b]) == pytest.approx(1.0)


def test_view_coverage_partial() -> None:
    a = np.array([[True, False, False, False]])
    b = np.array([[False, True, False, False]])
    assert view_coverage([a, b]) == pytest.approx(0.5)


def test_view_coverage_rejects_shape_mismatch() -> None:
    a = np.ones((2, 2), dtype=bool)
    b = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        view_coverage([a, b])
