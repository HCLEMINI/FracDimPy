#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FracDimPy - A Comprehensive Python Package for Fractal Dimension Calculation and Multifractal Analysis
=====================================================================================================

FracDimPy provides researchers with a unified toolkit for analyzing complex, self-similar patterns in data.
This package includes:

Monofractal Analysis:
- Hurst exponent calculation (R/S analysis)
- Box-counting dimension
- Information dimension
- Correlation dimension
- Structure function method
- Variogram method
- Sandbox method
- Detrended Fluctuation Analysis (DFA)

Multifractal Analysis:
- Multifractal spectrum analysis for 1D curves
- Multifractal analysis for 2D images
- Multifractal Detrended Fluctuation Analysis (MF-DFA)

Fractal Generation:
- Classic fractals: Cantor set, Sierpinski triangle/carpet, Koch curve
- Stochastic fractals: Brownian motion, Lévy flight, DLA
- Mathematical fractals: FBM curves/surfaces, Weierstrass-Mandelbrot functions

Author: Zhile Han
Link: https://www.zhihu.com/people/xiao-xue-sheng-ye-xiang-xie-shu/posts
"""

__version__ = "0.1.6"
__author__ = "Zhile Han"

# Import main modules
from . import monofractal
from . import multifractal
from . import generator
from . import utils

# Import monofractal analysis functions
from .monofractal import (
    hurst_dimension,
    structural_function,
    variogram_method,
    box_counting,
    sandbox_method,
    information_dimension,
    correlation_dimension,
    dfa,
)

# Import multifractal analysis functions
from .multifractal import multifractal_curve, multifractal_image, mf_dfa

# Import fractal generator functions
from .generator import (
    # 1D fractal curves
    generate_fbm_curve,
    generate_wm_curve,
    generate_takagi_curve,
    # 2D fractal surfaces
    generate_fbm_surface,
    generate_wm_surface,
    generate_takagi_surface,
    # Geometric fractals
    generate_cantor_set,
    generate_sierpinski,
    generate_sierpinski_carpet,
    generate_vicsek_fractal,
    generate_koch_curve,
    generate_koch_snowflake,
    # Random fractals
    generate_brownian_motion,
    generate_levy_flight,
    generate_self_avoiding_walk,
    generate_dla,
    generate_menger_sponge,
    generate_kakeya_set,
    kakeya_segments,
    tube_ck_error,
)

__all__ = [
    # Monofractal analysis functions
    "hurst_dimension",
    "structural_function",
    "variogram_method",
    "box_counting",
    "sandbox_method",
    "information_dimension",
    "correlation_dimension",
    "dfa",
    # Multifractal analysis functions
    "multifractal_curve",
    "multifractal_image",
    "mf_dfa",
    # Fractal generation functions - curves
    "generate_fbm_curve",
    "generate_wm_curve",
    "generate_takagi_curve",
    # Fractal generation functions - surfaces
    "generate_fbm_surface",
    "generate_wm_surface",
    "generate_takagi_surface",
    # Fractal generation functions - geometric
    "generate_cantor_set",
    "generate_sierpinski",
    "generate_sierpinski_carpet",
    "generate_vicsek_fractal",
    "generate_koch_curve",
    "generate_koch_snowflake",
    # Fractal generation functions - random
    "generate_brownian_motion",
    "generate_levy_flight",
    "generate_self_avoiding_walk",
    "generate_dla",
    "generate_menger_sponge",
    "generate_kakeya_set",
    "kakeya_segments",
    "tube_ck_error",
    # Main modules
    "monofractal",
    "multifractal",
    "generator",
    "utils",
]
