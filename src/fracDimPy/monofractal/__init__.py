#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monofractal Analysis Module
===========================

This module provides various methods for calculating monofractal dimensions
from different types of data. Each method has its own strengths and is suitable
for different types of fractal patterns.

Available Methods:
- hurst_dimension: Hurst exponent calculation using R/S analysis
- structural_function: Structure function method for self-affine curves
- variogram_method: Variogram method from geostatistics
- box_counting: Box-counting dimension for various data types
- sandbox_method: Sandbox method for local scaling analysis
- information_dimension: Information dimension based on entropy
- correlation_dimension: Correlation dimension using Grassberger-Procaccia algorithm
- dfa: Detrended Fluctuation Analysis for non-stationary time series

Each method is optimized for specific data types and analysis scenarios.
"""

from .hurst import hurst_dimension
from .structural_function import structural_function
from .variogram import variogram_method
from .box_counting import box_counting
from .sandbox import sandbox_method
from .information_dimension import information_dimension
from .correlation_dimension import correlation_dimension
from .dfa import dfa
from .divider import divider_dimension
from .minkowski_mc import minkowski_dimension_mc

__all__ = [
    "hurst_dimension",
    "structural_function",
    "variogram_method",
    "box_counting",
    "sandbox_method",
    "information_dimension",
    "correlation_dimension",
    "dfa",
    "divider_dimension",
    "minkowski_dimension_mc",
]
