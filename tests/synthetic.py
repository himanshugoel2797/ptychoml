"""Synthetic test patterns for verifying diffraction-pattern preprocessing.

Generates known real-space objects and their far-field |FFT| patterns so
callers can detect whether a preprocessing pipeline (centering, cropping,
fftshift, axis order) is wired up correctly.

Pattern convention
------------------
A "diffraction pattern" here is ``|FFT2(object)|`` (or ``|FFT2|**2`` for
intensity), treating the test image as a real-space transmission function
illuminated by a uniform probe. The default objects are real and
non-negative, so the DC component is the brightest pixel — its location
after ``fftshift=True`` is exactly the geometric center, a useful
invariant for centering tests.

Detecting axis flips
--------------------
Real, non-negative objects produce centro-symmetric magnitude patterns
(Friedel's law: ``|F(-k)| = |F(k)|``), so a 180° rotation of the pattern
is invisible. Mirror flips, transposes, and arbitrary shifts ARE
detectable as long as the object's shape isn't itself mirror-symmetric:
the included ``"asymmetric_L"`` object is not symmetric under any single
reflection, so a flipped/transposed pattern differs from the original.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def make_test_object(name: str = "asymmetric_L", size: int = 64) -> np.ndarray:
    """Return a known synthetic real-space object.

    All built-in objects are 2D ``float32`` arrays of shape ``(size, size)``
    with non-negative values, so the DC component of their FFT is the
    global magnitude max.

    Available names:
      * ``"asymmetric_L"`` — an "L" shape that is not symmetric under any
        single reflection, so a flipped or transposed pattern is detectable
        even though the |FFT| magnitude is centro-symmetric.
      * ``"centered_disk"`` — a small uniform disk at the geometric center;
        useful for centering tests because the diffraction pattern is a
        radially-symmetric Airy-like envelope with a sharp central peak.
      * ``"two_dots"`` — two delta peaks at known offsets; produces a
        cosine-fringe pattern whose orientation tracks the dot separation.
    """
    if name == "asymmetric_L":
        obj = np.zeros((size, size), dtype=np.float32)
        v_lo, v_hi = size // 4, 3 * size // 4
        col_lo, col_hi = size // 4, size // 4 + max(size // 8, 1)
        obj[v_lo:v_hi, col_lo:col_hi] = 1.0  # vertical bar
        row_lo, row_hi = 3 * size // 4 - max(size // 8, 1), 3 * size // 4
        obj[row_lo:row_hi, col_lo:size // 2] = 1.0  # short horizontal foot
        return obj

    if name == "centered_disk":
        radius = max(size // 16, 2)
        cy = cx = size // 2
        y, x = np.ogrid[:size, :size]
        return ((y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2).astype(np.float32)

    if name == "two_dots":
        obj = np.zeros((size, size), dtype=np.float32)
        obj[size // 4, size // 4] = 1.0
        obj[size // 4, 3 * size // 4] = 1.0
        return obj

    raise ValueError(f"Unknown test object name: {name!r}")


def make_diffraction_pattern(
    obj: np.ndarray,
    *,
    fftshift: bool = True,
    intensity: bool = False,
) -> np.ndarray:
    """Compute a far-field diffraction pattern from a real-space object.

    Parameters
    ----------
    obj : ndarray
        2D complex (or real) object in real space.
    fftshift : bool
        If True (default), move the DC component to the geometric center.
        Matches the ``data_is_shifted=True`` convention used by
        ``PtychoViTInference``.
    intensity : bool
        If True, return ``|FFT|**2``; otherwise return ``|FFT|`` (amplitude),
        which matches the ``diffamp`` dataset key in holoptycho's HDF5
        format.

    Returns
    -------
    ndarray
        Float32 pattern with the same shape as ``obj``.
    """
    f = np.fft.fft2(obj)
    if fftshift:
        f = np.fft.fftshift(f)
    pattern = np.abs(f)
    if intensity:
        pattern = pattern ** 2
    return pattern.astype(np.float32)


def make_test_pattern(
    name: str = "asymmetric_L",
    size: int = 64,
    *,
    fftshift: bool = True,
    intensity: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: return ``(object, pattern)`` for a named test image.

    Equivalent to::

        obj = make_test_object(name, size)
        pattern = make_diffraction_pattern(obj, fftshift=..., intensity=...)
    """
    obj = make_test_object(name, size)
    pattern = make_diffraction_pattern(obj, fftshift=fftshift, intensity=intensity)
    return obj, pattern
