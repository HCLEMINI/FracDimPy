#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minkowski (Dilation) Dimension via Monte-Carlo
==============================================

Estimates the fractal dimension of a point set (curve, network, cloud) from
the volume of its delta-neighbourhood (Minkowski dilation / sausage):

    V(delta) = |{x : dist(x, set) < delta}| ~ delta^{d - D}

so ``D = d - slope(log V vs log delta)``. The neighbourhood volume is estimated
by Monte-Carlo: sample random points in the bounding box and count the
fraction within ``delta`` of the set (KD-tree nearest-neighbour query).

This works on *unordered* point clouds (networks, branched structures),
2D/3D, with error controlled by the sample count. It complements the divider
method (which needs an ordered curve) for fracture / pore-throat networks.
"""

import numpy as np
from typing import Tuple, Optional

from ..utils.fitting import log_log_fit


def minkowski_dimension_mc(
    points: np.ndarray,
    n_scales: int = 16,
    n_samples: int = 40000,
    min_hit_frac: float = 0.005,
    seed: Optional[int] = 42,
    verbose: bool = False,
) -> Tuple[float, dict]:
    """Minkowski (dilation) dimension via Monte-Carlo sampling.

    Parameters
    ----------
    points : np.ndarray
        Point set, shape ``(M,)`` or ``(M, d)``. Unordered (networks OK);
        a ``(d, M)`` array is auto-transposed.
    n_scales : int, optional
        Number of dilation radii, log-spaced (default 16).
    n_samples : int, optional
        Monte-Carlo sample count per scale (default 40000).
    min_hit_frac : float, optional
        Drop scales whose hit fraction is below this (MC relative error too
        high) before fitting (default 0.005).
    seed : int, optional
        Random seed (default 42).
    verbose : bool, optional
        Print per-scale diagnostics (default False).

    Returns
    -------
    dimension : float
        Estimated Minkowski dimension.
    result : dict
        Keys: ``dimension``, ``delta_values``, ``V_values``, ``hit_fractions``,
        ``log_delta``, ``log_V``, ``R2``, ``coefficients``, ``method``,
        ``embedding_dim``, ``n_points``.

    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import generate_koch_curve, minkowski_dimension_mc
    >>> pts, _ = generate_koch_curve(level=6)
    >>> D, res = minkowski_dimension_mc(pts)
    >>> print(f"D={D:.3f}, R2={res['R2']:.4f}")  # Koch: true D ~ 1.262
    """
    from scipy.spatial import cKDTree

    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[0] < pts.shape[1]:
        pts = pts.T  # interpret as (M, d)
    n_pts, dim = pts.shape

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = hi - lo
    box_vol = float(np.prod(extent))
    if box_vol <= 0:
        raise ValueError("Point set has zero extent")

    # crude total length (for curves / networks) to set a sensible delta_min
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(seg.sum()) if seg.size else 0.0

    # delta_max: keep below ~1/8 of the smallest box side to avoid boundary
    # effects (where the dilation sausage spills outside the box).
    delta_max = float(np.min(extent)) / 8.0
    # delta_min: keep the MC hit fraction workable. For a curve of length L in
    # a box of volume V, hit_frac(delta) ~ c * L * delta^{d-1} / V, so
    # delta_min ~ (min_hit_frac * V / (c*L))^{1/(d-1)}.
    if L > 0 and dim > 1:
        delta_min = (min_hit_frac * box_vol / (L * 2.0)) ** (1.0 / (dim - 1))
    else:
        delta_min = (min_hit_frac * box_vol / max(n_pts, 1)) ** (1.0 / dim)
    delta_min = max(delta_min, delta_max * 1e-3)
    if delta_min >= delta_max:
        raise ValueError("Cannot establish a valid delta range for this point set")

    scales = np.logspace(np.log10(delta_min), np.log10(delta_max), n_scales)
    tree = cKDTree(pts)
    rng = np.random.default_rng(seed)

    Vs, fracs = [], []
    for delta in scales:
        q = rng.uniform(lo, hi, size=(n_samples, dim))
        dist, _ = tree.query(q, k=1)
        frac = float((dist < delta).mean())
        Vs.append(max(frac, 1e-9) * box_vol)
        fracs.append(frac)
        if verbose:
            print(f"delta={delta:.6g}  hit_frac={frac:.5f}  V={frac*box_vol:.6g}")

    Vs = np.asarray(Vs)
    fracs = np.asarray(fracs)
    scales = np.asarray(scales)
    keep = fracs >= min_hit_frac
    Vs, scales, fracs = Vs[keep], scales[keep], fracs[keep]
    if len(Vs) < 3:
        raise ValueError(
            "Insufficient scales with adequate MC hits; "
            "increase n_samples or widen the point set"
        )

    x_fit = np.log(scales)
    y_fit = np.log(Vs)
    slope, intercept, R2 = log_log_fit(x_fit, y_fit)
    dimension = dim - slope

    result = {
        "dimension": dimension,
        "delta_values": scales.tolist(),
        "V_values": Vs.tolist(),
        "hit_fractions": fracs.tolist(),
        "log_delta": x_fit,
        "log_V": y_fit,
        "R2": R2,
        "coefficients": np.array([dimension, intercept]),
        "method": f"Minkowski-MC (dim={dim})",
        "embedding_dim": dim,
        "n_points": n_pts,
    }
    return dimension, result
