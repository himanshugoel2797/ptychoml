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

from typing import Dict, Tuple

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


def make_test_probe(size: int = 64, *, sigma: float | None = None) -> np.ndarray:
    """Return a complex Gaussian probe centered in a ``(size, size)`` grid.

    Default sigma is ``size / 8`` — wide enough to give the probe meaningful
    spatial extent (so the exit wave depends on the object) but tight enough
    that it doesn't wrap around the grid.
    """
    if sigma is None:
        sigma = size / 8
    cy = cx = size // 2
    y, x = np.ogrid[:size, :size]
    amp = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))
    return amp.astype(np.complex64)


# Pattern transforms tried by ``check_orientation``. Includes the identity
# plus every reflection / 90° rotation — the dihedral group D4. A correctly
# oriented triple matches ``identity``; any other best-fit transform points
# to a flip / transpose / rotation in the data path.
_ORIENTATION_TRANSFORMS = {
    "identity": lambda a: a,
    "flip_y": lambda a: a[::-1, :],
    "flip_x": lambda a: a[:, ::-1],
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "transpose": lambda a: a.T,
    "anti_transpose": lambda a: a[::-1, ::-1].T,
}


def _scale_invariant_residual(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 distance after fitting an optimal positive scale factor.

    Returns 0 for a perfect match (up to multiplicative scale), 1 for an
    orthogonal pattern. Diffraction intensities have arbitrary normalization
    in practice, so we have to factor that out before comparing.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    if aa == 0.0 or bb == 0.0:
        return float("inf")
    s = float(np.dot(a, b)) / bb
    return float(np.linalg.norm(a - s * b) / np.sqrt(aa))


def check_orientation(
    diffraction: np.ndarray,
    probe: np.ndarray,
    object_patch: np.ndarray,
    *,
    fftshift: bool = True,
    intensity: bool = False,
    tol: float = 0.05,
) -> Dict[str, object]:
    """Check that ``(diffraction, probe, object_patch)`` is mutually consistent.

    Computes the forward-model prediction
    ``|FFT(probe * object_patch)|`` (or ``|FFT|**2`` if ``intensity=True``)
    and compares it to the supplied diffraction pattern under every D4
    symmetry transform (identity, flips, transposes, 90° rotations). The
    transform that best matches reveals where the orientation bug is — if
    any.

    All three arrays must have the same 2D shape — they live on the same
    grid in the forward model. Scale is fitted out per-transform, since
    diffraction intensities have arbitrary detector normalization.

    Parameters
    ----------
    diffraction : ndarray
        Measured/observed pattern, ``(H, W)`` non-negative.
    probe : ndarray
        Complex probe on the same grid.
    object_patch : ndarray
        Complex object patch on the same grid (same scan position the
        diffraction pattern was acquired at).
    fftshift, intensity : bool
        Forward-model conventions; pass the same values used to interpret
        the diffraction data elsewhere in the pipeline.
    tol : float
        Maximum identity residual (scale-invariant relative L2) at which
        the triple is judged consistent. Default 0.05 (5 %).

    Returns
    -------
    dict
        ``{"consistent": bool, "best_transform": str, "best_residual": float,
        "residuals": {transform_name: residual}}``. ``consistent`` is True iff
        the identity transform matches within ``tol``. ``best_transform``
        names the closest match — if it isn't ``"identity"``, the data path
        likely contains the inverse of that transform somewhere.
    """
    if diffraction.shape != probe.shape or probe.shape != object_patch.shape:
        raise ValueError(
            f"Shape mismatch: diffraction={diffraction.shape}, "
            f"probe={probe.shape}, object_patch={object_patch.shape}"
        )
    if diffraction.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {diffraction.ndim}D")

    expected = make_diffraction_pattern(
        probe * object_patch, fftshift=fftshift, intensity=intensity
    )

    residuals = {
        name: _scale_invariant_residual(transform(diffraction), expected)
        for name, transform in _ORIENTATION_TRANSFORMS.items()
    }
    best = min(residuals, key=residuals.__getitem__)
    return {
        "consistent": residuals["identity"] < tol,
        "best_transform": best,
        "best_residual": residuals[best],
        "residuals": residuals,
    }
