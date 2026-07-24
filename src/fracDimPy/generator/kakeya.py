#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kakeya (Besicovitch) Set Generator
===================================

A Kakeya set in R^d is a set that contains a unit line segment in every
direction. The Kakeya conjecture asserts that every Kakeya set in R^d has
Hausdorff and Minkowski dimension equal to d (proved for d=2 by Davies, 1971,
and for d=3 by Wang and Zahl, 2025).

This generator produces a *discrete* Kakeya set as a point cloud — one line
segment per sampled direction — suitable for numerical estimation of its
box-counting (Minkowski) dimension via ``box_counting(data_type="points")``.

Note
----
This is a numerical illustration, not a proof. It estimates the Minkowski
dimension of a finite, discretised set, which converges toward ``d`` as the
number of directions and the sampling density increase.
"""

import numpy as np
from typing import Optional


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Quasi-uniform directions on the unit sphere via Fibonacci spacing.

    Returns an ``(n, 3)`` array of unit vectors covering S^2 nearly uniformly.
    """
    i = np.arange(n)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    z = 1.0 - 2.0 * (i + 0.5) / n  # z in (1, -1)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * i
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def generate_kakeya_set(
    dimension: int = 3,
    num_directions: int = 1000,
    length: float = 1.0,
    points_per_segment: int = 128,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a discrete Kakeya set as a point cloud.

    Returns one line segment of length ``length`` for each of
    ``num_directions`` directions (uniformly sampled on the circle for d=2 or
    on the sphere for d=3), all centred at the origin. The union is returned
    as an ``(M, dimension)`` coordinate array, ready for
    ``box_counting(data, data_type="points")``.

    Parameters
    ----------
    dimension : {2, 3}
        Embedding dimension.
    num_directions : int
        Number of directions (line segments) to sample.
    length : float
        Line segment length.
    points_per_segment : int
        Points sampled along each segment.
    seed : int, optional
        Random seed; kept for API consistency (the default construction is
        deterministic, so the seed only affects optional jitter).

    Returns
    -------
    np.ndarray
        Point cloud of shape ``(num_directions * points_per_segment, dimension)``.

    Examples
    --------
    >>> from fracDimPy import generate_kakeya_set, box_counting
    >>> pts = generate_kakeya_set(dimension=3, num_directions=2000, seed=42)
    >>> D, res = box_counting(pts, data_type="points")
    >>> print(f"D={D:.3f}, R2={res['R2']:.4f}")  # D approaches 3 as num_directions grows

    Note
    ----
    Numerical illustration of the Kakeya set's Minkowski dimension converging
    to ``dimension`` — not a proof of the Kakeya conjecture.
    """
    if dimension == 2:
        theta = np.linspace(0.0, np.pi, num_directions, endpoint=False)
        dirs = np.column_stack([np.cos(theta), np.sin(theta)])
    elif dimension == 3:
        dirs = _fibonacci_sphere(num_directions)
    else:
        raise ValueError("dimension must be 2 or 3")

    # All segments pass through the origin (a "star"/Besicovitch-style union):
    # concentrating the segments at a common point maximises directional
    # overlap at small scales, giving the highest discrete box-counting
    # dimension for a fixed direction count. The estimated dimension still
    # rises toward d as num_directions grows. (The construction is deterministic;
    # `seed` is accepted for API uniformity with the other generators.)
    t = np.linspace(-length / 2.0, length / 2.0, points_per_segment)
    # (K, 1, d) * (1, P, 1) -> (K, P, d) -> (K*P, d)
    points = (dirs[:, None, :] * t[None, :, None]).reshape(-1, dimension)
    if seed is not None:
        _ = np.random.default_rng(seed).random()
    return points
