#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correlation Dimension
=====================

Implements the Grassberger-Procaccia algorithm for computing correlation dimension.

References:
    Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness of 
    strange attractors. Physica D: Nonlinear Phenomena, 9(1-2), 189-208.
"""

import numpy as np

# type: ignore
from typing import Tuple, Dict, Optional

from ..utils.fitting import linear_fit


def correlation_dimension(
    data: np.ndarray,
    min_r: Optional[float] = None,
    max_r: Optional[float] = None,
    num_points: int = 20,
    max_samples: int = 5000,
    verbose: bool = False,
) -> Tuple[float, Dict]:
    """
    Correlation Dimension

    Grassberger-Procaccia

    Parameters
    ----------
    data : np.ndarray
        1D2D
        - 1D: (N,)
        - 2D: (N, d)
    min_r : float, optional

    max_r : float, optional

    num_points : int, optional
        20
    max_samples : int, optional
        5000

    Returns
    -------
    D_corr : float

    result : dict

        - 'dimension':
        - 'radii':
        - 'correlations':
        - 'log_radii': log()
        - 'log_correlations': log()
        - 'coeffs':  [=D_C, ]
        - 'r_squared':
        - 'fit_range':  (start, end)

    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import correlation_dimension
    >>> # Lorenz
    >>> from scipy.integrate import odeint
    >>> def lorenz(state, t):
    >>>     x, y, z = state
    >>>     return [10*(y-x), x*(28-z)-y, x*y-8/3*z]
    >>> t = np.linspace(0, 50, 10000)
    >>> trajectory = odeint(lorenz, [1, 1, 1], t)
    >>> D, result = correlation_dimension(trajectory)
    >>> print(f": {D:.4f}")

    Notes
    -----

    C(r) = lim(N) (1/N) *  H(r - |x_i - x_j|)
     H Heaviside


    D_C = lim(r0) log(C(r)) / log(r)


    - Grassberger-Procaccia1983
    -
    -
    -

    References
    ----------
    Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness of
    strange attractors. Physica D: Nonlinear Phenomena, 9(1-2), 189-208.
    """
    #
    if data.ndim == 1:
        # 1D
        data = data.flatten()
        #
        points = np.column_stack([data[:-1], data[1:]])
    elif data.ndim == 2:
        points = data
    else:
        raise ValueError("1D2D")

    if len(points) == 0:
        raise ValueError("")

    #
    if len(points) > max_samples:
        indices = np.random.choice(len(points), max_samples, replace=False)
        points = points[indices]

    N = len(points)

    #
    if verbose:
        print(f"Processing {N} points ...")

    #
    # distances[i, j] = ||points[i] - points[j]||
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))

    #
    distances = distances[np.triu_indices(N, k=1)]

    #
    if min_r is None:
        min_r = np.percentile(distances, 0.5)
    if max_r is None:
        max_r = np.percentile(distances, 30)

    #
    radii = np.logspace(np.log10(min_r), np.log10(max_r), num_points)

    correlations = []

    if verbose:
        print("Computing correlations...")
    for r in radii:
        # r
        C_r = np.sum(distances < r) / len(distances)
        correlations.append(C_r)

    radii = np.array(radii)  # type: ignore[assignment]
    correlations = np.array(correlations)  # type: ignore[assignment]

    # log(0)
    #
    valid = (correlations > 1e-5) & (correlations < 0.8)  # type: ignore[operator]
    radii = radii[valid]
    correlations = correlations[valid]

    if len(radii) < 3:
        raise ValueError("")

    # log-log
    log_r = np.log(radii)
    log_C = np.log(correlations)

    #
    #
    best_r2 = 0
    best_coeffs = None
    best_range = None

    #
    min_fit_points = max(5, len(log_r) // 2)  #
    for start in range(len(log_r) - min_fit_points + 1):
        for end in range(start + min_fit_points, len(log_r) + 1):
            _, _, r2_temp = linear_fit(log_r[start:end], log_C[start:end])
            coeffs_temp = np.polyfit(log_r[start:end], log_C[start:end], 1)

            if r2_temp > best_r2:
                best_r2 = r2_temp
                best_coeffs = coeffs_temp
                best_range = (start, end)

    #
    if best_coeffs is None:
        coeffs = np.polyfit(log_r, log_C, 1)
        r_squared = 0
        best_range = (0, len(log_r))
    else:
        coeffs = best_coeffs
        r_squared = best_r2

    D_corr = coeffs[0]  #

    result = {
        "dimension": D_corr,
        "radii": radii,
        "correlations": correlations,
        "log_radii": log_r,
        "log_correlations": log_C,
        "coeffs": coeffs,
        "r_squared": r_squared,
        "fit_range": best_range,  #
    }

    return D_corr, result
