#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Structural Function Method
==========================

Computes fractal dimension from the structural function:

S(τ) = <[Z(x+τ) - Z(x)]²>

The scaling relationship log(S(τ)) ~ m * log(τ) gives dimension D = (4-m)/2
"""

import numpy as np
from typing import Tuple

from ..utils.fitting import log_log_fit

# type: ignore


def structural_function(
    y_data: np.ndarray, x_interval: float = 1.0, max_tau: int = 30, verbose: bool = False
) -> Tuple[float, dict]:
    """


    Parameters
    ----------
    y_data : np.ndarray

    x_interval : float, optional
        1.0
    max_tau : int, optional
        3010%

    Returns
    -------
    dimension : float

    result : dict

        - 'dimension':
        - 'tau_values':
        - 'S_values': S()
        - 'R2': R
        - 'coefficients':

    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import structural_function
    >>> #
    >>> x = np.linspace(0, 10, 1000)
    >>> y = np.random.randn(1000)
    >>> D, result = structural_function(y, x_interval=0.01)
    >>> print(f": {D:.4f}")

    Notes
    -----

    max_tau5%-10%
    """
    # SFM.py
    Z = np.asarray(y_data).flatten()
    interval = x_interval
    point_number = len(Z)

    # SFM.pymax_tau30
    # 10%
    if max_tau > int(0.1 * point_number):
        if verbose:
            print(f"max_tau={max_tau} 10% ({int(0.1 * point_number)})")
        max_tau = min(max_tau, 30)

    if max_tau > point_number // 2:
        raise ValueError(f"max_tau={max_tau}  ({point_number // 2})")

    Sl = []  # S()
    taol = []  #

    #
    for n in range(2, max_tau):
        z_sum = 0.0
        for i in range(0, point_number - n):
            diff = Z[i + n] - Z[i]
            z_sum += float(diff * diff)  #

        #
        S_value = float(z_sum / (point_number - n))
        Sl.append(S_value)
        taol.append(float(n * interval))

    #
    # numpy
    taol = np.array(taol)  # type: ignore[assignment]
    Sl = np.array(Sl)  # type: ignore[assignment]

    #
    valid_mask = (taol > 0) & (Sl > 0)  # type: ignore[operator]
    taol = taol[valid_mask]
    Sl = Sl[valid_mask]

    if len(taol) < 2:
        raise ValueError("")

    log_tau = np.log(taol)
    log_S = np.log(Sl)

    m, intercept, R2 = log_log_fit(log_tau, log_S)
    coefficients = np.array([m, intercept])
    dimension = (4 - m) / 2

    result = {
        "dimension": dimension,
        "tau_values": taol.tolist(),  # type: ignore[union-attr]
        "S_values": Sl.tolist(),  # type: ignore[union-attr]
        "log_tau": log_tau,
        "log_S": log_S,
        "slope": m,
        "R2": R2,
        "coefficients": coefficients,
        "method": "Structural Function",
    }

    return dimension, result
