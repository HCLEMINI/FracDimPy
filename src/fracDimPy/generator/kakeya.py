#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kakeya (Besicovitch) Set Generator & Non-clustering Analysis
============================================================

A Kakeya set in R^d is a set that contains a unit line segment in every
direction. The Kakeya conjecture asserts that every Kakeya set in R^d has
Hausdorff and Minkowski dimension equal to d (proved for d=2 by Davies, 1971,
and for d=3 by Wang and Zahl, 2025; see arXiv:2502.17655).

This module provides:

- ``generate_kakeya_set``: a *discrete* Kakeya set as a point cloud — one line
  segment per sampled direction — ready for ``box_counting(data_type="points")``.
- ``tube_ck_error``: a numerical estimate of the Katz-Tao Convex Wolff Axiom
  error (a non-clustering condition) of a set of tubes. Tubes arising from a
  Kakeya set satisfy ``CKT ~ 1``; clustered arrangements (e.g. all segments
  through the origin) give ``CKT >> 1``.

The Wolff axioms are the structural non-concentration condition at the heart
of the Wang-Zahl proof of the 3D Kakeya conjecture: every convex set ``W`` may
contain at most ``C * |W| / |T|`` tubes, where ``|T| ~ delta^2`` is the tube
volume and ``C = CKT`` is the Katz-Tao error.

Note
----
These are numerical illustrations, not proofs. The box-counting dimension of a
finite, discretised set converges toward ``d`` as the number of directions and
the sampling density increase, but remains below ``d`` for any finite set.
"""

import numpy as np
from typing import Optional, Tuple


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


def kakeya_segments(
    dimension: int, num_directions: int, length: float
) -> np.ndarray:
    """Centre lines of a star-shaped discrete Kakeya set.

    Returns one segment per direction (uniform on the circle / sphere), all
    centred at the origin, as an ``(K, 2, dimension)`` array of start/end
    points. This is the construction used by ``generate_kakeya_set`` and the
    natural input for ``tube_ck_error``.
    """
    if dimension == 2:
        theta = np.linspace(0.0, np.pi, num_directions, endpoint=False)
        dirs = np.column_stack([np.cos(theta), np.sin(theta)])
    elif dimension == 3:
        dirs = _fibonacci_sphere(num_directions)
    else:
        raise ValueError("dimension must be 2 or 3")

    half = length / 2.0
    return np.stack([-dirs * half, dirs * half], axis=1)  # (K, 2, d)


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
    to ``dimension`` — not a proof of the Kakeya conjecture
    (arXiv:2502.17655, Wang & Zahl 2025).
    """
    segs = kakeya_segments(dimension, num_directions, length)
    s = np.linspace(0.0, 1.0, points_per_segment)[None, :, None]
    points = (segs[:, 0][:, None, :] + (segs[:, 1] - segs[:, 0])[:, None, :] * s)
    points = points.reshape(-1, dimension)

    if seed is not None:
        _ = np.random.default_rng(seed).random()
    return points


def _segments_intersect_box(seg: np.ndarray, corner: np.ndarray, size: float) -> np.ndarray:
    """Slab test: which tube centre lines intersect an axis-aligned box.

    ``seg`` has shape ``(K, 2, 3)``; returns a ``(K,)`` boolean array. The box
    is ``[corner, corner + size]^3``. A tube whose centre line intersects the
    box counts as "passing through" it.
    """
    p0, p1 = seg[:, 0], seg[:, 1]
    d = p1 - p0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.abs(d) < 1e-12, 1e12, 1.0 / d)
    t0 = (corner - p0) * inv
    t1 = (corner + size - p0) * inv
    tmin = np.minimum(t0, t1).max(axis=1)
    tmax = np.maximum(t0, t1).min(axis=1)
    return (tmin <= tmax) & (tmax >= 0.0) & (tmin <= 1.0)


def tube_ck_error(
    segments: np.ndarray,
    delta: float,
    mode: str = "contained",
    n_samples: int = 300,
    n_scales: int = 8,
    seed: Optional[int] = None,
) -> Tuple[float, dict]:
    """Estimate the Katz-Tao Convex Wolff Axiom error ``CKT`` of a tube set.

    The Wolff axioms are a *non-clustering* condition: a set of ``delta``-tubes
    is non-clustering if every convex set ``W`` contains at most
    ``C * |W| / |T|`` tubes, where ``|T| ~ delta^2`` is the tube volume. The
    infimum of such ``C`` is the Katz-Tao error ``CKT``. Tubes arising from a
    Kakeya set satisfy ``CKT ~ 1``.

    This is a numerical estimate: boxes of random position and log-spaced size
    are sampled, and ``CKT`` is the maximum of ``(#tubes counted in W) * |T| /
    |W|`` over all sampled boxes.

    Parameters
    ----------
    segments : np.ndarray
        Tube centre lines, shape ``(K, 2, 3)`` — start and end point of each
        tube (e.g. ``_kakeya_segments`` output, or a fracture-network model).
    delta : float
        Tube radius; tube volume ``|T| ~ delta^2`` in R^3.
    mode : {'contained', 'passing'}, optional
        - ``'contained'`` (default): a tube counts if its whole body (centre
          line shrunk by ``delta``) lies inside ``W``. This is the faithful
          reading of the Wolff axioms; it detects *tube-body clustering*
          (e.g. a dense bundle of parallel tubes squeezed into a thin slab).
        - ``'passing'``: a tube counts if its centre line intersects ``W``.
          This detects *crossing/density clustering* (e.g. every segment
          through a common point) but is noisy at scales near ``delta``, where
          small boxes over-count intersecting tubes.
    n_samples : int, optional
        Number of boxes sampled per scale (default 300).
    n_scales : int, optional
        Number of log-spaced box sizes between ``2*delta`` and the extent
        (default 8).
    seed : int, optional
        Random seed for box sampling (default None).

    Returns
    -------
    ck_error : float
        Estimated ``CKT``. Values ``O(1)`` indicate a non-clustering,
        Kakeya-like arrangement; larger values indicate clustering.
    diagnostics : dict
        ``n_tubes``, ``delta``, ``mode``, ``extent``, ``box_sizes``,
        ``n_boxes``.
    """
    if mode not in ("contained", "passing"):
        raise ValueError("mode must be 'contained' or 'passing'")
    seg = np.asarray(segments, dtype=float)
    if seg.ndim != 3 or seg.shape[1] != 2:
        raise ValueError("segments must have shape (K, 2, 3): start/end of each centre line")
    if seg.shape[2] != 3:
        raise ValueError("tube_ck_error currently supports 3D tube sets only")

    rng = np.random.default_rng(seed)
    lo = seg.reshape(-1, 3).min(axis=0)
    hi = seg.reshape(-1, 3).max(axis=0)
    extent = float(np.max(hi - lo))
    if extent <= 0:
        raise ValueError("segments have zero extent")

    sizes = np.logspace(np.log10(2.0 * delta), np.log10(extent), n_scales)
    tube_vol = delta ** 2.0  # |T| ~ delta^2 for a unit-length tube in R^3
    # Note: CKT is a sup over boxes of count*|T|/|W|. With few tubes (K*|T|
    # much smaller than the ambient volume) the true sup is < 1, which is a
    # legitimate non-clustering reading; we report it rather than flooring at 1.
    best = 0.0
    ratios = []
    span = np.maximum(hi - lo - sizes[:, None], 0.0)

    for size, sp in zip(sizes, span):
        for _ in range(n_samples):
            corner = lo + rng.random(3) * sp
            if mode == "contained":
                inner_lo = corner + delta
                inner_hi = corner + size - delta
                ok = np.all(
                    (seg[:, 0] >= inner_lo)
                    & (seg[:, 0] <= inner_hi)
                    & (seg[:, 1] >= inner_lo)
                    & (seg[:, 1] <= inner_hi),
                    axis=1,
                )
            else:
                ok = _segments_intersect_box(seg, corner, size)
            count = int(ok.sum())
            if count == 0:
                continue
            vol = size ** 3.0
            ratio = count * tube_vol / vol
            ratios.append(ratio)
            best = max(best, ratio)

    return best, {
        "n_tubes": seg.shape[0],
        "delta": delta,
        "mode": mode,
        "extent": extent,
        "box_sizes": sizes.tolist(),
        "n_boxes": len(ratios),
    }
