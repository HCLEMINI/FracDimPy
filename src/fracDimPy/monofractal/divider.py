#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Divider (Richardson / Compass) Method
=====================================

Estimates the fractal dimension of an *ordered* curve by walking it with a
fixed chord (divider) length ``delta`` and counting the steps ``N(delta)``.
The scaling ``N(delta) ~ delta^{-D}`` gives the divider dimension ``D``.

Unlike box-counting, the divider method never touches an ambient grid, so it
is well-conditioned for sparse curves (measure-zero sets) in 2D and 3D —
box-counting on such sets is dominated by empty-box statistics and grid
alignment, and can even fail outright for sparse 3D lines.

Reference: Richardson (1961); Mandelbrot, "How long is the coast of Britain?"
(Science, 1967).
"""

import numpy as np
from typing import Tuple

from ..utils.fitting import log_log_fit


def divider_dimension(
    points: np.ndarray, n_scales: int = 20, verbose: bool = False
) -> Tuple[float, dict]:
    """Fractal dimension of an ordered curve via the Richardson divider method.

    Parameters
    ----------
    points : np.ndarray
        Ordered curve points, shape ``(N,)`` for 1D or ``(N, d)`` for
        d-dimensional. The points must follow the curve in order (start to end).
    n_scales : int, optional
        Number of divider lengths, log-spaced (default 20).
    verbose : bool, optional
        Print per-scale diagnostics (default False).

    Returns
    -------
    dimension : float
        Estimated divider dimension.
    result : dict
        Keys: ``dimension``, ``delta_values``, ``N_values``, ``log_delta``,
        ``log_N``, ``R2``, ``coefficients``, ``method``, ``n_points``,
        ``curve_length``.

    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import generate_koch_curve, divider_dimension
    >>> pts, _ = generate_koch_curve(level=6)
    >>> D, res = divider_dimension(pts)
    >>> print(f"D={D:.3f}, R2={res['R2']:.4f}")  # Koch: true D = log4/log3 ~ 1.262
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    n_pts = pts.shape[0]
    if n_pts < 3:
        raise ValueError("Need at least 3 ordered points")

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        raise ValueError("Curve has zero total length")

    mean_step = total / (n_pts - 1)
    delta_min = mean_step * 2.0  # at least span two sampling steps
    delta_max = total / 4.0
    if delta_min >= delta_max:
        raise ValueError("Curve too short or too few points for divider method")

    scales = np.logspace(np.log10(delta_min), np.log10(delta_max), n_scales)
    Nl, delta_l = [], []
    for delta in scales:
        steps, pos = 0, 0
        while pos < n_pts - 1:
            dist = np.linalg.norm(pts[pos + 1 :] - pts[pos], axis=1)
            hit = np.where(dist >= delta)[0]
            if hit.size == 0:
                break
            pos = pos + 1 + hit[0]
            steps += 1
        if steps < 2:
            continue
        Nl.append(steps)
        delta_l.append(float(delta))
        if verbose:
            print(f"delta={delta:.6g}  N={steps}")

    if len(Nl) < 3:
        raise ValueError("Insufficient valid scales for divider method")

    x_fit = np.log(np.array(delta_l))
    y_fit = np.log(np.array(Nl))
    slope, intercept, R2 = log_log_fit(x_fit, y_fit)
    dimension = -slope  # N ~ delta^{-D}  =>  log N = -D log delta

    result = {
        "dimension": dimension,
        "delta_values": delta_l,
        "N_values": Nl,
        "log_delta": x_fit,
        "log_N": y_fit,
        "R2": R2,
        "coefficients": np.array([dimension, intercept]),
        "method": "Divider (Richardson)",
        "n_points": n_pts,
        "curve_length": total,
    }
    return dimension, result
