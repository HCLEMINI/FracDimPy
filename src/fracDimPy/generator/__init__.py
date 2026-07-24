#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fractal Generation Module
========================

This module provides functions to generate various types of fractals for testing,
validation, and educational purposes.

Curve Generators (1D):
- FBM: Fractional Brownian Motion curves
- WM: Weierstrass-Mandelbrot curves
- Takagi: Takagi (Blancmange) function curves

Surface Generators (2D):
- FBM: Fractional Brownian Motion surfaces
- WM: Weierstrass-Mandelbrot surfaces
- Takagi: Takagi surfaces

Pattern Generators:
- Cantor: Cantor set
- Sierpinski: Triangle and carpet
- DLA: Diffusion-limited aggregation
- Menger: Menger sponge (3D)
"""

from .curves import generate_fbm_curve, generate_wm_curve, generate_takagi_curve

from .surfaces import generate_fbm_surface, generate_wm_surface, generate_takagi_surface

from .patterns import (
    generate_cantor_set,
    generate_sierpinski,
    generate_sierpinski_carpet,
    generate_vicsek_fractal,
    generate_koch_curve,
    generate_koch_snowflake,
    generate_brownian_motion,
    generate_levy_flight,
    generate_self_avoiding_walk,
    generate_dla,
    generate_menger_sponge,
)
from .kakeya import generate_kakeya_set

__all__ = [
    #
    "generate_fbm_curve",
    "generate_wm_curve",
    "generate_takagi_curve",
    #
    "generate_fbm_surface",
    "generate_wm_surface",
    "generate_takagi_surface",
    #
    "generate_cantor_set",
    "generate_sierpinski",
    "generate_sierpinski_carpet",
    "generate_vicsek_fractal",
    "generate_koch_curve",
    "generate_koch_snowflake",
    "generate_brownian_motion",
    "generate_levy_flight",
    "generate_self_avoiding_walk",
    "generate_dla",
    "generate_menger_sponge",
    "generate_kakeya_set",
]
