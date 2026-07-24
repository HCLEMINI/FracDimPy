#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multifractal Analysis for 2D Images
====================================

Implements multifractal analysis for image data using partition function method.
"""

import numpy as np
from numpy import polyfit
from typing import Tuple, List, Optional

# type: ignore

from ..utils.multifractal_common import (
    default_q_list,
    compute_partition,
    build_figure_data,
    build_metrics,
)
from ..utils.scales import power_of_two_scales
from ..utils.box_counting_core import count_boxes_fixed


def multifractal_image(
    image: np.ndarray, q_list: Optional[List[float]] = None, verbose: bool = False
) -> Tuple[dict, dict]:
    """


    Parameters
    ----------
    image : np.ndarray
        (H x W)
    q_list : list of float, optional
        q1000

    Returns
    -------
    metrics : dict

    figure_data : dict


    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import multifractal_image
    >>> #
    >>> img = np.random.randint(0, 256, (256, 256))
    >>> metrics, figure_data = multifractal_image(img)
    >>> print(f"D(0): {metrics['D(0)'][0]:.4f}")

    Notes
    -----

    """
    mt = image
    height, width = mt.shape
    if verbose:
        print(f"height,width: {height}, {width}")

    # q0,1,2
    if q_list is None:
        q_min = -10
        q_max = 10
        q_list = default_q_list(q_min, q_max)

    q_min = min(q_list)  # type: ignore
    q_max = max(q_list)  # type: ignore
    if verbose:
        print(f"q: {len(q_list)}, : [{q_min}, {q_max}]")

    #
    xl = []  #
    tl = []  #
    al = []  # Holder
    fl = []  #
    dl = []  #
    Pill = []  #

    #
    M = min(height, width)
    epsilonl = power_of_two_scales(M)
    if verbose:
        print(f"{epsilonl}")

    #
    for epsilon in epsilonl:
        Pill.append(_compute_probability_image(mt, epsilon))

    # q
    tl, al, fl, dl, xl = compute_partition(q_list, Pill, epsilonl)

    al = list(al)
    q_list = list(q_list)
    dl = list(dl)

    #  f = a*^2 + b* + c
    coeff = polyfit(al, fl, 2)
    if verbose:
        print(f"\nf- " f"\nf = {coeff[0]:.4f} + {coeff[1]:.4f} + {coeff[2]:.4f}")

    metrics = build_metrics(q_list, al, fl, dl, q_min, q_max, coeff=coeff)

    #
    figure_data = build_figure_data(q_list, tl, al, fl, dl, xl)

    if verbose:
        for key in ["D(0)", "D(1)", "D(2)", "H", "width_total", "width_left", "width_right"]:
            print(f"  {key}: {metrics[key][0]:.4f}")

    return metrics, figure_data


def _compute_probability_image(mt: np.ndarray, epsilon: int) -> np.ndarray:
    """


    Parameters
    ----------
    mt : np.ndarray

    epsilon : int


    Returns
    -------
    Pil : np.ndarray

    """
    #
    temp_mt = _box_counting_2d(mt, epsilon)
    temp_mt = temp_mt.flatten()

    #
    N_sum = np.sum(temp_mt)
    Pil = temp_mt / N_sum

    return Pil


def _box_counting_2d(MT: np.ndarray, EPSILON: int) -> np.ndarray:
    """Box counting for 2D data using shared utility."""
    return count_boxes_fixed(MT, EPSILON)
