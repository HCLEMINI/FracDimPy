#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multifractal Analysis for 1D Curves
====================================

Implements partition function method for multifractal analysis of curves.

Computes:
- Mass exponent τ(q)
- Hölder exponent α(q)
- Multifractal spectrum f(α)
- Generalized dimensions D(q)
"""

import numpy as np
from numpy import polyfit
import multiprocessing

# type: ignore
from typing import Tuple, List, Optional, Union
from .custom_epsilon import custom_epsilon, is_power_of_two
from ..utils.multifractal_common import (
    default_q_list,
    compute_partition,
    build_figure_data,
    build_metrics,
)
from ..utils.scales import power_of_two_scales
from ..utils.box_counting_core import count_boxes_fixed


def multifractal_curve(
    data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    q_list: Optional[List[float]] = None,
    epsilon_grid: Optional[List[int]] = None,
    epsilonl: Optional[List[float]] = None,
    use_multiprocessing: bool = True,
    data_type: str = "single",
    interpolate_to_power2: bool = True,
    remove_zeros: bool = True,
    verbose: bool = False,
) -> Tuple[dict, dict]:
    """


    Parameters
    ----------
    data : np.ndarray or tuple
        - data_type='single': (np.ndarray)
        - data_type='dual': (x, y)
    q_list : list of float, optional
        q[-10, -9, ..., 9, 10]
    epsilon_grid : list of int, optional
        [2, 4, 8, ..., len(data)//2]
    epsilonl : list of float, optional
        None
    use_multiprocessing : bool, optional
        True
    data_type : str, optional
        'single'()  'dual'(XY)'single'
    interpolate_to_power2 : bool, optional
        2True
    remove_zeros : bool, optional
        0dualTrue

    Returns
    -------
    metrics : dict

        - '(q=0)', '(q=1)', '(q=2)': q
        - 'f(q=0)', 'f(q=1)', 'f(q=2)': q
        - 'D(0)', 'D(1)', 'D(2)':
        - 'H': Hurst
        - '', '', '':
    figure_data : dict

        - 'q': q
        - '(q)':
        - '(q)':
        - 'f()':
        - 'D(q)':

    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import multifractal_curve
    >>> #
    >>> data = np.random.randn(1024)
    >>> metrics, figure_data = multifractal_curve(data, data_type='single')
    >>> print(f"D(0): {metrics['D(0)'][0]:.4f}")
    >>> print(f"D(1): {metrics['D(1)'][0]:.4f}")
    >>>
    >>> # XY
    >>> x = np.linspace(0, 10, 1024)
    >>> y = np.random.randn(1024)
    >>> metrics, figure_data = multifractal_curve((x, y), data_type='dual')
    >>> print(f"D(0): {metrics['D(0)'][0]:.4f}")

    Notes
    -----

    -f

    data_type
    - 'single':
    - 'dual': XY
    """
    #
    if data_type == "dual":
        if isinstance(data, tuple) and len(data) == 2:
            x_data, y_data = data

            if verbose:
                print(f": {len(y_data)}")

            # epsilonRun_CustomEpsilon.py
            # 2
            if remove_zeros:
                # 0
                non_zero_indices = np.where(y_data != 0)[0]
                current_len = len(non_zero_indices)
            else:
                current_len = len(y_data)

            if interpolate_to_power2:
                if is_power_of_two(current_len):
                    target_length = current_len
                else:
                    target_length = 2 ** (int(np.log2(current_len)) + 1)
            else:
                target_length = None

            # epsilon
            epsilon_physical, mt, x_grid = custom_epsilon(
                x_data,
                y_data,
                interpolate_to_power2=interpolate_to_power2,
                target_length=target_length,
                remove_zeros=remove_zeros,
                verbose=verbose,
            )

            if interpolate_to_power2:
                if verbose:
                    print(f": {len(mt)} (2)")

        else:
            raise ValueError(" (x, y) ")
    else:
        mt = data  # type: ignore[assignment]
        epsilon_physical = 1.0

    # qMF_BC1D.py
    if q_list is None:
        q_min = -10
        q_max = 10
        q_list = default_q_list(q_min, q_max)

    # Run_CustomEpsilon.pyepsilonl_grid
    if epsilon_grid is None:
        epsilon_grid = power_of_two_scales(len(mt))

    # epsilon_grid
    if epsilonl is None and data_type == "dual":
        epsilonl = [e * epsilon_physical for e in power_of_two_scales(len(mt))]
    elif epsilonl is None:
        epsilonl = epsilon_grid  # type: ignore[assignment]

    if verbose:
        print(f"{epsilon_grid}")
        print(f"q: {q_list}")

    q_min = min(q_list)  # type: ignore[assignment]
    q_max = max(q_list)  # type: ignore[assignment]

    #
    xl = []  #
    tl = []  #
    al = []  #
    fl = []  #
    dl = []  #

    Pill = []

    #
    if use_multiprocessing and __name__ != "__main__":
        try:
            pool = multiprocessing.Pool()
            result_list = []
            for epsilon in epsilon_grid:
                PACK = (epsilon, mt)
                result_list.append(pool.apply_async(_compute_probability, (PACK,)))
            pool.close()
            pool.join()
            for item_return in result_list:
                Pill.append(item_return.get())
        except Exception:
            #
            use_multiprocessing = False

    if not use_multiprocessing:
        for epsilon in epsilon_grid:
            PACK = (epsilon, mt)
            Pill.append(_compute_probability(PACK))

    # q
    tl, al, fl, dl, xl = compute_partition(q_list, Pill, epsilonl)

    #
    al = list(al)
    q_list = list(q_list)
    dl = list(dl)

    coeff = polyfit(al, fl, 2)
    if verbose:
        print(f"f- : f = {coeff[0]:.4f} + {coeff[1]:.4f} + {coeff[2]:.4f}")

    metrics = build_metrics(
        q_list,
        al,
        fl,
        dl,
        q_min,
        q_max,
        coeff=coeff,
        extra={"lenth": np.sum(mt), "mean": np.mean(mt)},
    )

    figure_data = build_figure_data(q_list, tl, al, fl, dl, xl)

    return metrics, figure_data


def _compute_probability(PACK: Tuple) -> np.ndarray:
    """


    Parameters
    ----------
    PACK : tuple
        (epsilon, data)

    Returns
    -------
    Pil : np.ndarray

    """
    epsilon, mt = PACK

    #
    temp_mt = _box_counting_1d(mt, epsilon)

    # 0
    temp_mt = np.delete(temp_mt, np.where(temp_mt == 0)[0])

    #
    N_sum = np.sum(temp_mt)
    Pil = temp_mt / N_sum

    return Pil


def _box_counting_1d(MT: np.ndarray, EPSILON: int) -> np.ndarray:
    """Box counting for 1D data using shared utility."""
    return count_boxes_fixed(MT, EPSILON)
