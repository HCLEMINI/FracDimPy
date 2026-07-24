#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared multifractal computation utilities.

Provides common functions for q-list generation, partition function computation,
metrics dictionary construction, and figure data assembly — replacing duplicated
code between mf_curve.py and mf_image.py.
"""

import numpy as np
from numpy.polynomial.polynomial import polyfit


def default_q_list(q_min=-10, q_max=10, n=1000):
    """Generate default q-value list for multifractal analysis.

    Parameters
    ----------
    q_min : float, optional
        Minimum q value (default -10).
    q_max : float, optional
        Maximum q value (default 10).
    n : int, optional
        Number of linearly spaced points (default 1000).

    Returns
    -------
    list
        Sorted unique q values, guaranteed to include 0, 1, 2.
    """
    q = np.unique(np.append(np.round(np.linspace(q_min, q_max, n), 2), [0, 1, 2]))
    return q.tolist()


def compute_partition(q_list, Pill, epsilonl):
    """Compute multifractal partition function over q values.

    Parameters
    ----------
    q_list : list
        q values to evaluate.
    Pill : list of numpy.ndarray
        Probability distributions at each scale.
    epsilonl : list
        Scale (epsilon) values.

    Returns
    -------
    tl : list
        tau(q) values.
    al : list
        alpha(q) values.
    fl : list
        f(q) values.
    dl : list
        D(q) values.
    xl : list
        [log(epsilon), log(partition_sum), q] for each q.
    """
    tl, al, fl, dl, xl = [], [], [], [], []

    for q in q_list:
        xl_t = []
        xl_a = []
        xl_d = []

        for Pil in Pill:
            temp = np.power(Pil, q)
            X_t = np.sum(temp)
            xl_t.append(X_t)
            xl_a.append(np.sum(temp / X_t * np.log(Pil)))

            if q == 1:
                xl_d.append(np.sum(Pil * np.log(Pil)))

        t = polyfit(np.log(epsilonl), np.log(xl_t), 1)[0]
        X = [np.log(epsilonl), np.log(xl_t), q]
        a = polyfit(np.log(epsilonl), xl_a, 1)[0]
        f = q * a - t

        if q == 1:
            D = polyfit(np.log(epsilonl), xl_d, 1)[0]
        else:
            D = t / (q - 1)

        tl.append(t)
        xl.append(X)
        al.append(a)
        fl.append(f)
        dl.append(D)

    return tl, al, fl, dl, xl


def build_metrics(q_list, al, fl, dl, q_min_val, q_max_val, coeff=None, extra=None):
    """Build the standardized multifractal metrics dictionary.

    Produces pure-ASCII keys whose values are wrapped in single-element lists,
    e.g. ``metrics["D(0)"][0]``. This matches the historical access pattern of
    ``multifractal_curve`` / ``multifractal_image`` while retiring the legacy
    leading-space and CJK-bracket key names.

    Parameters
    ----------
    q_list : list
        q values.
    al, fl, dl : list
        alpha, f, and D values at each q.
    q_min_val : float
        Minimum q value.
    q_max_val : float
        Maximum q value.
    coeff : array-like, optional
        Quadratic f(alpha) = c2*alpha^2 + c1*alpha + c0 coefficients
        ``[c2, c1, c0]`` (numpy.polyfit descending order).
    extra : dict, optional
        Additional scalar entries; each is wrapped as ``[value]``.

    Returns
    -------
    dict
        Multifractal metrics dictionary.
    """
    metrics = {}
    q_arr = np.array(q_list)
    al_arr = np.array(al)
    fl_arr = np.array(fl)
    dl_arr = np.array(dl)

    def _idx(q_val):
        diffs = np.abs(q_arr - q_val)
        return int(np.argmin(diffs))

    i0, i1, i2 = _idx(0), _idx(1), _idx(2)
    imin, imax = _idx(q_min_val), _idx(q_max_val)

    # Holder exponent alpha(q)
    metrics["alpha(q=0)"] = [al_arr[i0]]
    metrics["alpha(q=1)"] = [al_arr[i1]]
    metrics["alpha(q=2)"] = [al_arr[i2]]
    metrics["alpha(q=min)"] = [al_arr[imin]]
    metrics["alpha(q=max)"] = [al_arr[imax]]
    metrics["alpha(q=min)-alpha(q=0)"] = [al_arr[imin] - al_arr[i0]]
    metrics["alpha(q=0)-alpha(q=max)"] = [al_arr[i0] - al_arr[imax]]
    metrics["alpha(q=min)-alpha(q=max)"] = [al_arr[imin] - al_arr[imax]]

    # Parabolic f(alpha) fit coefficients
    if coeff is not None:
        metrics["quad_coeff"] = [coeff[0]]
        metrics["linear_coeff"] = [coeff[1]]
        metrics["const_coeff"] = [coeff[2]]

    # f(alpha) values
    metrics["f(q=0)"] = [fl_arr[i0]]
    metrics["f(q=1)"] = [fl_arr[i1]]
    metrics["f(q=2)"] = [fl_arr[i2]]
    metrics["f(q=min)"] = [fl_arr[imin]]
    metrics["f(q=max)"] = [fl_arr[imax]]
    metrics["f(q=0)-f(q=1)"] = [fl_arr[i0] - fl_arr[i1]]
    metrics["f(q=min)-f(q=max)"] = [fl_arr[imin] - fl_arr[imax]]

    # Spectrum widths (derived from alpha)
    metrics["width_left"] = [al_arr[i0] - al_arr[imin]]
    metrics["width_right"] = [al_arr[imax] - al_arr[i0]]
    metrics["width_total"] = [al_arr[imax] - al_arr[imin]]

    # Generalized dimensions D(q) and Hurst exponent H = (1 + D(2)) / 2
    metrics["D(0)"] = [dl_arr[i0]]
    metrics["D(1)"] = [dl_arr[i1]]
    metrics["D(2)"] = [dl_arr[i2]]
    metrics["D(0)-D(1)"] = [dl_arr[i0] - dl_arr[i1]]
    metrics["D(min)"] = [dl_arr[imin]]
    metrics["D(max)"] = [dl_arr[imax]]
    metrics["D(min)-D(max)"] = [dl_arr[imin] - dl_arr[imax]]
    metrics["H"] = [(1 + dl_arr[i2]) / 2]

    if extra:
        for k, v in extra.items():
            metrics[k] = v if isinstance(v, list) else [v]

    return metrics


def build_figure_data(q_list, tl, al, fl, dl, xl, sample_every=20):
    """Build the standardized figure_data dictionary for plotting.

    Parameters
    ----------
    q_list : list
        q values.
    tl, al, fl, dl : list
        tau, alpha, f, D values at each q.
    xl : list
        [log(epsilon), log(partition_sum), q] for each q.
    sample_every : int, optional
        Sample every Nth q for X/r plot data (default 20).

    Returns
    -------
    dict
        Figure data dictionary with q-vectors and sampled plot data.
    """
    figure_data = {
        "q": q_list,
        "tau_q": tl,
        "alpha_q": al,
        "f()": fl,
        "D(q)": dl,
    }
    temp_q_n = max(1, int(len(q_list) / sample_every))
    for i, item in enumerate(q_list):
        if i != 0 and i % temp_q_n == 0:
            figure_data.update(
                {
                    f"q={item}_X": list(xl[i][1]),
                    f"q={item}_r": list(xl[i][0]),
                }
            )
    return figure_data
