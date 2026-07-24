#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom Epsilon Selection
=========================

Utilities for selecting appropriate scale ranges (epsilon values) in 
multifractal analysis.
"""

import numpy as np
from scipy import interpolate

# type: ignore
from typing import Tuple, Optional


def advise_mtepsilon(x: np.ndarray) -> float:
    """
    epsilon

    Parameters
    ----------
    x : np.ndarray
        X

    Returns
    -------
    epsilon : float
        epsilon

    Notes
    -----
    epsilon = (x_max - x_min) / num_points

    """
    xl = np.max(x) - np.min(x)
    num = len(x)
    epsilon = xl / num
    return epsilon


def coordinate_to_matrix(x: np.ndarray, z: np.ndarray, epsilon: float) -> np.ndarray:
    """
    (X, Z)

    Parameters
    ----------
    x : np.ndarray
        X
    z : np.ndarray
        Z
    epsilon : float


    Returns
    -------
    matrix : np.ndarray
        -1

    Notes
    -----

    1. Xx' = round((x - x_min) / epsilon)
    2.
    3. -1

    Examples
    --------
    >>> x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> epsilon = 0.2
    >>> matrix = coordinate_to_matrix(x, z, epsilon)
    """
    # X
    x_indices = np.round((x - np.min(x)) / epsilon).astype(int)

    # -1
    matrix = np.zeros(np.max(x_indices) + 1) - 1

    #
    for idx in np.unique(x_indices):
        #
        xy_indices = np.argwhere(x_indices == idx)
        #
        z[xy_indices.flatten()] = np.sum(z[xy_indices]) / len(xy_indices)

    #
    matrix[x_indices] = z

    return matrix


def fill_vacancy(mt: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    -1

    Parameters
    ----------
    mt : np.ndarray
        -1

    Returns
    -------
    mt : np.ndarray


    Notes
    -----
    -1

    Examples
    --------
    >>> mt = np.array([1.0, -1, -1, 4.0, 5.0, -1, 7.0])
    >>> mt_filled = fill_vacancy(mt)
    >>> # -1
    """
    length = len(mt)
    # -1
    vacancy_indices = np.argwhere(mt == -1).flatten()
    num_vacancy = len(vacancy_indices)

    if num_vacancy > 0:
        if verbose:
            print(f": {num_vacancy}/{length}; : {num_vacancy/length*100:.4f}%")

        # -1
        valid_indices = np.argwhere(mt > -1).flatten()

        for i in vacancy_indices:
            # -1
            left_valid = valid_indices[valid_indices < i]
            # -1
            right_valid = valid_indices[valid_indices > i]

            if len(left_valid) > 0 and len(right_valid) > 0:
                a1 = mt[left_valid[-1]]
                a2 = mt[right_valid[0]]
                mt[i] = (a1 + a2) / 2
            elif len(left_valid) > 0:
                #
                mt[i] = mt[left_valid[-1]]
            elif len(right_valid) > 0:
                #
                mt[i] = mt[right_valid[0]]

    return mt


def cubic_interpolation(
    x_data: np.ndarray, y_data: np.ndarray, x_new: np.ndarray, kind: str = "linear"
) -> np.ndarray:
    """


    Parameters
    ----------
    x_data : np.ndarray
        X
    y_data : np.ndarray
        Y
    x_new : np.ndarray
        X
    kind : str, optional
        'linear', 'cubic''linear'

    Returns
    -------
    y_new : np.ndarray
        Y
    """
    f_interp = interpolate.interp1d(x_data, y_data, kind=kind)
    y_new = f_interp(x_new)
    return y_new


def custom_epsilon(
    x: np.ndarray,
    z: np.ndarray,
    interpolate_to_power2: bool = True,
    target_length: Optional[int] = None,
    remove_zeros: bool = False,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    epsilonCustomEpsilon.ce

    Parameters
    ----------
    x : np.ndarray
        X
    z : np.ndarray
        Z
    interpolate_to_power2 : bool, optional
        2True
    target_length : int, optional
        None
    remove_zeros : bool, optional
        z=0False

    Returns
    -------
    epsilon : float
        epsilon
    matrix : np.ndarray

    x_grid : np.ndarray
        X

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> z = np.sin(x) + np.random.randn(100) * 0.1
    >>> epsilon, matrix, x_grid = custom_epsilon(x, z, interpolate_to_power2=True)
    >>> print(f": {len(matrix)} (2)")
    """
    # 0
    if remove_zeros:
        non_zero_indices = np.where(z != 0)[0]
        x = x[non_zero_indices]
        z = z[non_zero_indices]

    # epsilon
    mt_epsilon_min = advise_mtepsilon(x)

    #
    matrix = coordinate_to_matrix(x, z, mt_epsilon_min)

    #
    matrix = fill_vacancy(matrix, verbose=verbose)

    # 2
    if interpolate_to_power2:
        x_min = np.min(x)

        #
        x1 = np.array([x_min + mt_epsilon_min * i for i in range(len(matrix))])

        # 2
        if target_length is None:
            current_len = len(matrix)
            # current_len2
            target_length = 2 ** (int(np.log2(current_len)) + 1)

        #
        x2 = np.linspace(x_min, np.max(x1), target_length)

        #
        matrix = cubic_interpolation(x1, matrix, x2, kind="linear")

        # epsilon
        epsilon = x2[1] - x2[0]
        return epsilon, matrix, x2
    else:
        #
        x_grid = np.array([np.min(x) + mt_epsilon_min * i for i in range(len(matrix))])
        return mt_epsilon_min, matrix, x_grid


def is_power_of_two(n: int) -> bool:
    """
    2

    Parameters
    ----------
    n : int


    Returns
    -------
    bool
        n2TrueFalse

    Examples
    --------
    >>> is_power_of_two(8)
    True
    >>> is_power_of_two(10)
    False
    """
    return n > 0 and (n & (n - 1)) == 0
