#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fractal Curve Generators
=========================

This module implements various fractal curve generation algorithms:

- Fractional Brownian Motion (FBM) curves
- Weierstrass-Mandelbrot (WM) function curves
- Takagi (Blancmange) function curves
"""

import numpy as np
from typing import Tuple, Optional


def generate_fbm_curve(
    dimension: float, length: int = 1024, method: str = "daviesharte", seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Generate a Fractional Brownian Motion (FBM) curve with specified fractal dimension.

    Parameters
    ----------
    dimension : float
        Fractal dimension, must be in range (1, 2)
    length : int, optional
        Number of points in the curve, default is 1024
    method : str, optional
        Generation method. Only 'daviesharte' is implemented (default).
    seed : int, optional
        Random seed for reproducibility. None (default) draws fresh randomness
        each call, matching the previous behaviour.

    Returns
    -------
    curve : np.ndarray
        Generated FBM curve
    actual_dimension : float
        The actual fractal dimension (same as input dimension)

    Examples
    --------
    >>> from fracDimPy import generate_fbm_curve
    >>> curve, D = generate_fbm_curve(dimension=1.5, length=1024)
    >>> print(f"Length: {len(curve)}, Dimension: {D:.2f}")

    Notes
    -----
    Uses the Davies-Harte algorithm to generate FBM curves.
    The Hurst exponent H is related to dimension D by: H = 2 - D

    Reference:
        Davies, R.B. and Harte, D.S. (1987). Tests for Hurst effect.
        Biometrika, 74(1), 95-101.
    """
    if not (1 < dimension < 2):
        raise ValueError("Dimension must be in range (1, 2)")

    if method != "daviesharte":
        raise ValueError(
            f"Unknown method: {method!r}. Only 'daviesharte' is implemented."
        )

    H = 2 - dimension  # Convert to Hurst exponent
    curve = _generate_fbm_daviesharte(length, H, seed)

    return curve, dimension


def generate_wm_curve(
    dimension: float, length: int = 1024, gamma: float = 2.0, lambda_param: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Weierstrass-Mandelbrot function curve.

    Parameters
    ----------
    dimension : float
        Fractal dimension, must be in range (1, 2)
    length : int, optional
        Number of points, default is 1024
    gamma : float, optional
        Scaling parameter, default is 2.0
    lambda_param : float, optional
        Frequency scaling parameter, default is 2.0

    Returns
    -------
    x : np.ndarray
        x-coordinates of the curve
    y : np.ndarray
        y-coordinates of the Weierstrass-Mandelbrot curve

    Examples
    --------
    >>> from fracDimPy import generate_wm_curve
    >>> x, y = generate_wm_curve(dimension=1.5)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y)
    >>> plt.show()

    Notes
    -----
    The Weierstrass-Mandelbrot function is defined as:
    W(x) = sum[gamma^(-nH) * sin(lambda^n * x)] where H = 2 - D
    This function generates deterministic fractal curves with precise control
    over the fractal dimension D.
    """
    if not (1 < dimension < 2):
        raise ValueError("Dimension must be in range (1, 2)")

    # Convert to Hurst exponent
    H = 2 - dimension

    x = np.linspace(0, 1, length)
    y = np.zeros(length)

    # Calculate number of iterations based on resolution
    n_max = int(np.log(length) / np.log(lambda_param))

    for n in range(n_max):
        amplitude = gamma ** (-n * H)  # Amplitude decreases with Hurst exponent
        frequency = lambda_param**n
        y += amplitude * np.sin(2 * np.pi * frequency * x)

    return x, y


def generate_takagi_curve(
    dimension: float = 1.5, level: int = 10, length: int = 1024, b: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Takagi (Blancmange) fractal curve.

    Parameters
    ----------
    dimension : float, optional
        Fractal dimension in range (1, 2), default is 1.5
    level : int, optional
        Number of iterations, default is 10
    length : int, optional
        Number of points, default is 1024
    b : float, optional
        Base parameter, must be > 1, default is 2.0

    Returns
    -------
    x : np.ndarray
        x-coordinates of the curve
    y : np.ndarray
        y-coordinates of the Takagi curve

    Examples
    --------
    >>> from fracDimPy import generate_takagi_curve
    >>> x, y = generate_takagi_curve(dimension=1.5, level=10)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y)
    >>> plt.show()

    Notes
    -----
    The Takagi function is constructed iteratively.
    The parameter a is computed from the desired dimension D: a = b^D / 4
    This ensures that the constraint a*b > 1 is satisfied.

    The Takagi function is: Takagi(t) = sum[a^k * phi(b^k * t)]
    where phi(t) = |bt - round(bt)|
    """
    if not (1 < dimension < 2):
        raise ValueError("Dimension must be in range (1, 2)")

    if b <= 1:
        raise ValueError("Base parameter b must be > 1")

    # Compute amplitude parameter a from dimension
    # D = log(4*a) / log(b)  =>  a = b^D / 4
    a = b**dimension / 4

    x = np.linspace(0, 1, length)
    y = np.zeros(length)

    def phi(t):
        """Takagi sawtooth function: phi(t) = |bt - round(bt)|"""
        return np.abs(b * t - np.round(b * t))

    # Iteratively build the Takagi curve
    for k in range(level):
        y += (a**k) * phi((b**k) * x)

    return x, y


def _generate_fbm_daviesharte(n: int, H: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate Fractional Brownian Motion using the Davies-Harte algorithm.

    Parameters
    ----------
    n : int
        Number of points to generate
    H : float
        Hurst exponent, must be in range (0, 1)

    Returns
    -------
    fbm : np.ndarray
        Generated FBM sequence

    Reference
    ---------
    Davies, R.B. and Harte, D.S. (1987). Tests for Hurst effect.
    Biometrika, 74(1), 95-101.
    """
    if not (0 < H < 1):
        raise ValueError("Hurst exponent must be in range (0, 1)")

    # Define autocovariance function for FGN (Fractional Gaussian Noise)
    def _autocovariance(k, hurst):
        """Autocovariance function for fractional Gaussian noise"""
        return 0.5 * (
            np.abs(k - 1) ** (2 * hurst)
            - 2 * np.abs(k) ** (2 * hurst)
            + np.abs(k + 1) ** (2 * hurst)
        )

    # Determine FFT size
    # For FGN of length n, we need FFT of size at least 2*n
    # for efficient circular embedding
    n_fft = 2 ** int(np.ceil(np.log2(2 * n)))

    # Compute autocovariance sequence
    gamma = np.array([_autocovariance(i, H) for i in range(n_fft)])

    # Build circulant embedding vector for Toeplitz matrix
    # r = [gamma[0], gamma[1], ..., gamma[n-1], 0, gamma[n-1], ..., gamma[1]]
    r = np.concatenate([gamma[:n], np.array([0]), gamma[1:n][::-1]])

    # Compute eigenvalues via FFT
    # The eigenvalues of the circulant matrix are obtained via FFT
    eigenvalues = np.fft.fft(r).real

    # Ensure non-negative eigenvalues (numerical stability)
    if np.any(eigenvalues < 0):
        eigenvalues = np.maximum(eigenvalues, 0)

    # Generate random Gaussian variables
    # Z1, Z2 are independent standard normal
    rng = np.random.default_rng(seed)
    randn = rng.standard_normal(n_fft // 2 + 1) + 1j * rng.standard_normal(n_fft // 2 + 1)
    randn[0] = randn[0].real * np.sqrt(2)  # DC component (real)
    randn[-1] = randn[-1].real * np.sqrt(2)  # Nyquist frequency (real)

    # Generate FGN in frequency domain
    # w = sqrt(eigenvalues) * Z
    w = np.sqrt(eigenvalues[: n_fft // 2 + 1] / n_fft) * randn

    # Transform back to time domain via IFFT
    # Construct full symmetric spectrum
    w_full = np.concatenate([w, np.conj(w[-2:0:-1])])
    fgn = np.fft.ifft(w_full).real[:n]

    # Integrate FGN to get FBM
    fbm = np.cumsum(fgn)

    return fbm
