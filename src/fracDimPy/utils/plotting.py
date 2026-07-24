#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting Utilities
==================

Visualization functions for fractal analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def plot_fractal_analysis(
    result: Dict[str, Any], save_path: Optional[str] = None, show: bool = True
) -> None:
    """
    Plot fractal dimension analysis results in log-log space.

    Parameters
    ----------
    result : dict
        Analysis results dictionary containing 'log_*' keys
    save_path : str, optional
        Path to save figure, if None does not save
    show : bool, optional
        Whether to display the plot, default is True

    Examples
    --------
    >>> from fracDimPy import higuchi_dimension
    >>> from fracDimPy.utils import plot_fractal_analysis
    >>> D, result = higuchi_dimension(data)
    >>> plot_fractal_analysis(result)
    """
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6))

    # Detect data type and set appropriate labels
    if "log_k" in result and "log_L" in result:
        # Higuchi method
        x = result["log_k"]
        y = result["log_L"]
        xlabel = "log(k)"
        ylabel = "log(L(k))"
    elif "log_r" in result and "log_rs" in result:
        # Hurst R/S method
        x = result["log_r"]
        y = result["log_rs"]
        xlabel = "log(r)"
        ylabel = "log(R/S)"
    elif "log_tau" in result and "log_S" in result:
        # Structural function
        x = result["log_tau"]
        y = result["log_S"]
        xlabel = "log(τ)"
        ylabel = "log(S(τ))"
    elif "log_lag" in result and "log_variogram" in result:
        # Variogram
        x = result["log_lag"]
        y = result["log_variogram"]
        xlabel = "log(h)"
        ylabel = "log(γ(h))"
    elif "log_inv_epsilon" in result and "log_N" in result:
        # Box-counting
        x = result["log_inv_epsilon"]
        y = result["log_N"]
        xlabel = "log(1/ε)"
        ylabel = "log(N)"
    elif "log_windows" in result and "log_fluctuations" in result:
        # DFA
        x = result["log_windows"]
        y = result["log_fluctuations"]
        xlabel = "log(n)"
        ylabel = "log(F(n))"
    elif "log_radii" in result and "log_correlations" in result:
        # Correlation dimension
        x = result["log_radii"]
        y = result["log_correlations"]
        xlabel = "log(r)"
        ylabel = "log(C(r))"
    else:
        raise ValueError("Unknown result format - cannot determine plot type")

    # Plot data points
    ax.plot(x, y, "ro", label="Data", markersize=8, alpha=0.6)

    # Plot regression line
    if "coefficients" in result:
        coeff = result["coefficients"]
        f = np.poly1d(coeff)
        ax.plot(x, f(x), "b-", linewidth=2, label="Linear Fit")

        # Add equation and statistics
        equation = f"y = {coeff[0]:.4f}x + {coeff[1]:.4f}"
        r2_text = f"R² = {result.get('R2', result.get('r_squared', 0)):.4f}"
        d_text = f"D = {result.get('dimension', 0):.4f}"

        text = f"{equation}\n{r2_text}\n{d_text}"
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f"Fractal Analysis - {result.get('method', '')}", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_multifractal_spectrum(
    figure_data: Dict[str, Any], save_path: Optional[str] = None, show: bool = True
) -> None:
    """
    Plot multifractal spectrum analysis results.

    Generates a 2x2 panel figure showing:
    - Mass exponent τ(q) vs q
    - Hölder exponent α(q) vs q
    - Multifractal spectrum f(α) vs α
    - Generalized dimension D(q) vs q

    Parameters
    ----------
    figure_data : dict
        Multifractal analysis results dictionary
    save_path : str, optional
        Path to save figure
    show : bool, optional
        Whether to display the plot, default is True

    Examples
    --------
    >>> from fracDimPy import multifractal_curve
    >>> from fracDimPy.utils import plot_multifractal_spectrum
    >>> metrics, figure_data = multifractal_curve(data)
    >>> plot_multifractal_spectrum(figure_data)
    """
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    q = figure_data["q"]
    tau = figure_data.get("tau_q", figure_data.get("τ(q)", figure_data.get("(q)", [])))
    alpha = figure_data.get("alpha_q", figure_data.get("α(q)", figure_data.get("(q)", [])))
    f = figure_data.get("f()", figure_data.get("f(α)", []))
    D = figure_data["D(q)"]

    # τ(q) vs q - Mass exponent
    axes[0, 0].plot(q, tau, "b-o", linewidth=2, markersize=6)
    axes[0, 0].set_xlabel("q", fontsize=12)
    axes[0, 0].set_ylabel("τ(q)", fontsize=12)
    axes[0, 0].set_title("Mass Exponent", fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # α(q) vs q - Hölder exponent
    axes[0, 1].plot(q, alpha, "r-o", linewidth=2, markersize=6)
    axes[0, 1].set_xlabel("q", fontsize=12)
    axes[0, 1].set_ylabel("α(q)", fontsize=12)
    axes[0, 1].set_title("Hölder Exponent", fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # f(α) vs α - Multifractal spectrum
    axes[1, 0].plot(alpha, f, "g-o", linewidth=2, markersize=6)
    axes[1, 0].set_xlabel("α", fontsize=12)
    axes[1, 0].set_ylabel("f(α)", fontsize=12)
    axes[1, 0].set_title("Multifractal Spectrum", fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # D(q) vs q - Generalized dimensions
    axes[1, 1].plot(q, D, "m-o", linewidth=2, markersize=6)
    axes[1, 1].set_xlabel("q", fontsize=12)
    axes[1, 1].set_ylabel("D(q)", fontsize=12)
    axes[1, 1].set_title("Generalized Dimensions", fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
