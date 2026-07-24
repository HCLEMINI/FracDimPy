#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fractal Surface Generators
===========================

This module implements various fractal surface generation algorithms:

- Fractional Brownian Motion (FBM) surfaces
- Weierstrass-Mandelbrot surfaces
- Takagi surfaces
"""

import numpy as np
from typing import Optional


def generate_fbm_surface(
    dimension: float, size: int = 256, method: str = "fft", seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a Fractional Brownian Motion (FBM) surface.

    Parameters
    ----------
    dimension : float
        Fractal dimension, must be in range (2, 3)
    size : int, optional
        Surface resolution (size x size), default is 256
    method : str, optional
        Generation method. Only 'fft' is implemented (default).
    seed : int, optional
        Random seed for reproducibility. None (default) draws fresh randomness.

    Returns
    -------
    surface : np.ndarray
        Generated FBM surface as (size x size) array

    Examples
    --------
    >>> from fracDimPy import generate_fbm_surface
    >>> surface = generate_fbm_surface(dimension=2.5, size=256)
    >>> print(f"Shape: {surface.shape}")

    Notes
    -----
    Uses FFT-based method to generate FBM surfaces.
    The Hurst exponent H is related to dimension D by: H = 3 - D
    """
    if not (2 < dimension < 3):
        raise ValueError("Dimension must be in range (2, 3)")

    if method == "fft":
        return _generate_fbm_surface_fft(dimension, size, seed)
    elif method == "midpoint":
        # Not yet implemented; kept distinct from 'fft' for forward compatibility.
        raise ValueError("method='midpoint' is not implemented; use method='fft' (default).")
    else:
        raise ValueError(f"Unknown method: {method}")


def _generate_fbm_surface_fft(dimension: float, size: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate FBM surface using FFT-based spectral synthesis.

    Generates 2D fractional Brownian motion surfaces via power spectrum.

    Reference:
        - Peitgen, H.O. & Saupe, D. (1988). The Science of Fractal Images.
        - Wood, A.T.A. & Chan, G. (1994). Simulation of stationary Gaussian
          processes in [0,1]^d. Journal of computational and graphical statistics.
    """
    H = 3 - dimension

    # Generate 2D power spectrum
    # FFT approach works well for surfaces of size x size
    n = size

    # Create 2D frequency grid
    # kx, ky are frequency coordinates
    kx = np.fft.fftfreq(n, d=1.0 / n)
    ky = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, ky)

    # Compute radial frequency squared
    # k^2 = kx^2 + ky^2
    K2 = KX**2 + KY**2
    K2[0, 0] = 1  # Avoid division by zero

    # Power spectrum for FBM
    # S(k) ~ k^(-2H-2) for 2D
    power_spectrum = K2 ** (-(H + 1))
    power_spectrum[0, 0] = 0  # Remove DC component

    # Amplitude spectrum
    amplitude = np.sqrt(power_spectrum)

    # Generate random phase
    # Complex Gaussian random field
    rng = np.random.default_rng(seed)
    phase = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

    # Enforce Hermitian symmetry for real-valued IFFT
    # This ensures IFFT output is real
    phase[0, 0] = 0  # DC must be real

    # Fourier field
    fourier_field = amplitude * phase

    # Inverse FFT to get spatial field
    surface = np.fft.ifft2(fourier_field).real

    # Center around zero mean
    surface = surface - np.mean(surface)

    return surface


def _generate_fbm_surface_midpoint(dimension: float, size: int) -> np.ndarray:
    """
    Generate FBM surface using midpoint displacement method.
    """
    # TODO: Implement midpoint displacement algorithm
    raise NotImplementedError("Midpoint displacement not yet implemented")


def generate_wm_surface(
    dimension: float,
    size: int = 256,
    level: int = 10,
    lambda_param: float = 1.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Weierstrass-Mandelbrot function surface.

    Parameters
    ----------
    dimension : float
        Fractal dimension in range (2, 3)
    size : int, optional
        Surface resolution, default is 256
    level : int, optional
        Number of iterations, default is 10
    lambda_param : float, optional
        Frequency scaling parameter, must be > 1, default is 1.5

    Returns
    -------
    surface : np.ndarray
        Generated WM surface as (size x size) array

    Examples
    --------
    >>> from fracDimPy import generate_wm_surface
    >>> surface = generate_wm_surface(dimension=2.5, level=10)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(surface, cmap='terrain')
    >>> plt.show()

    Notes
    -----
    The Weierstrass-Mandelbrot function for surfaces is defined as:

    Z(x,y) = sum[C[n] * lambda^((D-3)*n) * sin(lambda^n * (x*cos(B[n]) + y*sin(B[n])) + A[n])]

    Parameters:
    - C[n]: Random amplitude coefficients
    - A[n]: Random phase angles in [0, 2π]
    - B[n]: Random direction angles in [0, 2π]
    - D: Fractal dimension in range (2, 3)
    - lambda: Frequency scaling factor > 1
    """
    if not (2 < dimension < 3):
        raise ValueError("Dimension must be in range (2, 3)")

    if lambda_param <= 1:
        raise ValueError("Lambda parameter must be > 1")

    # Generate random parameters (local RNG; no global-state mutation)
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 1, level)  # Amplitude coefficients
    A = rng.uniform(0, 2 * np.pi, level)  # Phase angles
    B = rng.uniform(0, 2 * np.pi, level)  # Direction angles

    # Create coordinate grid
    x = np.linspace(0, 6, size)  # Domain [0, 6] for better visualization
    y = np.linspace(0, 6, size)
    X, Y = np.meshgrid(x, y)

    # Initialize surface
    Z = np.zeros((size, size))

    # Sum Weierstrass-Mandelbrot terms
    for n in range(1, level + 1):
        # Amplitude decreases with frequency
        amplitude = C[n - 1] * (lambda_param ** ((dimension - 3) * n))

        # Frequency increases exponentially
        frequency = lambda_param**n

        # Project onto random direction: x*cos(B) + y*sin(B)
        projection = X * np.cos(B[n - 1]) + Y * np.sin(B[n - 1])

        # Add oscillatory term
        Z += amplitude * np.sin(frequency * projection + A[n - 1])

    return Z


def generate_takagi_surface(dimension: float = 2.5, level: int = 10, size: int = 256) -> np.ndarray:
    """
    Generate a Takagi (Blancmange) function surface.

    Parameters
    ----------
    dimension : float, optional
        Fractal dimension in range (2, 3), default is 2.5
    level : int, optional
        Number of iterations, default is 10
    size : int, optional
        Surface resolution, default is 256

    Returns
    -------
    surface : np.ndarray
        Generated Takagi surface as (size x size) array

    Examples
    --------
    >>> from fracDimPy import generate_takagi_surface
    >>> surface = generate_takagi_surface(dimension=2.5, level=10)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(surface, cmap='terrain')
    >>> plt.show()

    Notes
    -----
    The Takagi surface is a 2D extension of the Takagi curve.
    The parameter b is computed from the desired dimension D: D = log(8*b) / log(2)
    This gives: b = 2^D / 8, which must satisfy 0.5 < b < 1

    The Takagi function is: Z(x,y) = sum[b^n * phi(2^(n-1)*x, 2^(n-1)*y)]
    where phi(x,y) = |2x - floor(2x)| * |2y - floor(2y)|
    """
    if not (2 < dimension < 3):
        raise ValueError("Dimension must be in range (2, 3)")

    # Compute parameter b from dimension
    # D = log(8*b) / log(2)  =>  b = 2^D / 8
    b = 2**dimension / 8

    if not (0.5 < b < 1):
        raise ValueError(f"Computed b={b:.4f} is outside valid range (0.5, 1)")

    # Create coordinate grid in [0, 1] x [0, 1]
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)

    # Initialize surface
    Z = np.zeros((size, size))

    # Define Takagi basis function
    def phi(x, y):
        """
        2D Takagi sawtooth function
        phi(x,y) = |2x - floor(2x)| * |2y - floor(2y)|
        """
        return np.abs(2 * x - np.floor(2 * x)) * np.abs(2 * y - np.floor(2 * y))

    # Iteratively build Takagi surface
    for n in range(1, level + 1):
        scale = 2 ** (n - 1)
        Z += (b**n) * phi(scale * X, scale * Y)

    return Z
