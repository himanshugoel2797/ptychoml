"""Tests for ``ptychoml.testing`` and demonstrations of how to use it.

These exercises also serve as integration checks for the preprocessing
utilities: a known synthetic pattern goes through ``resize_diffraction_patterns``
and we verify the output is still centered, oriented, and not silently flipped.
"""
import numpy as np

from ptychoml.preprocess import crop_to_roi, resize_diffraction_patterns
from tests.synthetic import (
    make_diffraction_pattern,
    make_test_object,
    make_test_pattern,
)


# ----- generator behavior ---------------------------------------------------

def test_make_test_object_shape_and_dtype():
    obj = make_test_object("asymmetric_L", size=64)
    assert obj.shape == (64, 64)
    assert obj.dtype == np.float32


def test_make_diffraction_pattern_dc_at_center_when_shifted():
    """For a non-negative amplitude, |FFT| has its peak at DC."""
    obj = make_test_object("centered_disk", size=64)
    pattern = make_diffraction_pattern(obj, fftshift=True)
    peak_y, peak_x = np.unravel_index(np.argmax(pattern), pattern.shape)
    assert (peak_y, peak_x) == (32, 32)  # geometric center for even N


def test_make_diffraction_pattern_dc_at_corner_when_unshifted():
    """Without fftshift, DC sits at index (0, 0) — what PtychoViT was trained on."""
    obj = make_test_object("centered_disk", size=64)
    pattern = make_diffraction_pattern(obj, fftshift=False)
    peak_y, peak_x = np.unravel_index(np.argmax(pattern), pattern.shape)
    assert (peak_y, peak_x) == (0, 0)


def test_make_test_pattern_returns_object_and_pattern():
    obj, pattern = make_test_pattern("asymmetric_L", size=64)
    assert obj.shape == pattern.shape == (64, 64)
    assert pattern.dtype == np.float32


def test_make_test_object_unknown_name_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown test object"):
        make_test_object("not_a_real_name")


# ----- orientation: real objects are centro-symmetric, but mirror-asymmetric -----

def test_real_object_pattern_is_centrosymmetric():
    """Real, non-negative samples have |F(-k)| = |F(k)| (Friedel)."""
    _, pattern = make_test_pattern("asymmetric_L", size=64)
    # Skip the (0,0) row/col so the slice is exactly centro-symmetric.
    p = pattern[1:, 1:]
    np.testing.assert_allclose(p, p[::-1, ::-1], rtol=1e-4, atol=1e-4)


def test_asymmetric_L_pattern_breaks_mirror_symmetry():
    """Asymmetric shape ⇒ pattern is NOT mirror-symmetric — so axis flips are
    detectable even though the pattern is centro-symmetric."""
    _, pattern = make_test_pattern("asymmetric_L", size=64)
    p = pattern[1:, 1:]
    diff_x = np.abs(p - p[:, ::-1]).max()
    diff_y = np.abs(p - p[::-1, :]).max()
    assert diff_x > 1e-3 * pattern.max()
    assert diff_y > 1e-3 * pattern.max()


# ----- integration: resize_diffraction_patterns on synthetic input ----------

def test_resize_centers_synthetic_pattern():
    """Pad a centered 64×64 pattern into a 384 frame, then crop back to 256.

    The peak should land on the new center (128, 128) — proves the per-frame
    argmax centering is finding DC correctly.
    """
    _, pattern = make_test_pattern("centered_disk", size=64, fftshift=True)
    # Place the 64×64 pattern at an off-center location inside a larger frame
    # so the crop has to *find* it (not just slice the middle).
    big = np.zeros((384, 384), dtype=np.float32)
    py, px = 100, 220
    big[py:py + 64, px:px + 64] = pattern

    out = resize_diffraction_patterns([big], target_n=256)

    assert out.shape == (1, 256, 256)
    new_peak = np.unravel_index(np.argmax(out[0]), out[0].shape)
    assert new_peak == (128, 128)


def test_resize_preserves_orientation():
    """Resize must not silently transpose or flip the pattern.

    Compare a marker offset from the peak before and after the crop:
    its (dy, dx) relative to the peak should be unchanged.
    """
    _, pattern = make_test_pattern("centered_disk", size=64, fftshift=True)
    # Pattern's peak is at (32, 32). Plant a marker at a known offset, kept
    # well below the DC peak so it doesn't itself become the argmax.
    peak_value = float(pattern.max())
    marker_dy, marker_dx = 5, -7
    marker_value = peak_value * 0.5
    pattern[32 + marker_dy, 32 + marker_dx] = marker_value

    big = np.zeros((384, 384), dtype=np.float32)
    big[100:164, 220:284] = pattern

    out = resize_diffraction_patterns([big], target_n=256)[0]

    peak = np.unravel_index(np.argmax(out), out.shape)
    assert peak == (128, 128)
    assert out[peak[0] + marker_dy, peak[1] + marker_dx] == marker_value
    # The flipped/transposed positions must NOT carry the marker — guards
    # against an accidental vertical flip, horizontal flip, or transpose.
    assert out[peak[0] - marker_dy, peak[1] - marker_dx] != marker_value
    assert out[peak[0] + marker_dx, peak[1] + marker_dy] != marker_value


def test_crop_to_roi_round_trip_with_synthetic_pattern():
    """ROI crop on a centered pattern keeps the peak at the new center."""
    _, pattern = make_test_pattern("centered_disk", size=128, fftshift=True)
    # Pattern peak is at (64, 64). A symmetric ROI [32:96, 32:96] re-centers
    # the peak at (32, 32) of the cropped output.
    roi = [[32, 96], [32, 96]]
    out = crop_to_roi(pattern, roi)
    assert out.shape == (64, 64)
    peak = np.unravel_index(np.argmax(out), out.shape)
    assert peak == (32, 32)
