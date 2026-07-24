# FracDimPy

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-0.1.5-green.svg)](https://pypi.org/project/FracDimPy/0.1.5/)
[![PyPI](https://img.shields.io/pypi/v/FracDimPy.svg)](https://pypi.org/project/FracDimPy/)

**A Comprehensive Python Package for Fractal Dimension Calculation and Multifractal Analysis**

English | [简体中文](https://github.com/Kecoya/FracDimPy/blob/main/README_CN.md)

</div>

---

## 📖 Introduction

FracDimPy is a powerful and easy-to-use Python package designed for fractal dimension calculation and multifractal analysis. Whether you are a researcher studying fractal geometry or an engineer analyzing complex data, FracDimPy provides professional and accurate analysis tools.

### ✨ Key Features

- **🔢 Multiple Monofractal Methods**

  - Hurst Exponent Method (R/S Analysis)
  - Box-counting Method
  - Information Dimension Method
  - Correlation Dimension Method
  - Structure Function Method
  - Variogram Method
  - Sandbox Method
  - Detrended Fluctuation Analysis (DFA)
- **📊 Multifractal Analysis**

  - One-dimensional curve multifractal analysis
  - Two-dimensional image multifractal analysis
  - Multifractal Detrended Fluctuation Analysis (MF-DFA)
  - Custom scale sequences
- **🎨 Fractal Generator**

  - Classical fractals: Cantor set, Sierpinski triangle/carpet, Koch curve, Menger sponge, etc.
  - Random fractals: Brownian motion, Lévy flight, self-avoiding walk, Diffusion-Limited Aggregation (DLA)
  - Fractal curves: FBM curve, Weierstrass-Mandelbrot function, Takagi curve
  - Fractal surfaces: FBM surface, Weierstrass-Mandelbrot surface, Takagi surface
- **📈 Rich Visualization**

  - Automatic generation of professional charts
  - Log-log plot fitting
  - Multifractal spectrum display
  - Customizable plotting options
- **💾 Flexible Data Processing**

  - Support for multiple data formats (CSV, Excel, TXT, NPY, images, etc.)
  - Automatic data preprocessing
  - Result export functionality

---

## 🚀 Quick Start

### Installation

#### Install from PyPI (Recommended)

```bash
# Install complete package (with all dependencies)
pip install FracDimPy
```

#### 🇨🇳 Mirror Installation for Chinese Users (Faster Speed)

For users in mainland China, we recommend using mirror sources for faster installation speed:

```bash
# Install using Tsinghua University mirror
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple FracDimPy

# Or permanently configure mirror source
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install FracDimPy
```

**Common Mirror Sources**:

- Tsinghua University: `https://pypi.tuna.tsinghua.edu.cn/simple`
- Alibaba Cloud: `https://mirrors.aliyun.com/pypi/simple`
- USTC: `https://pypi.mirrors.ustc.edu.cn/simple`
- Douban: `https://pypi.douban.com/simple`

#### Correct Package Import

```python
# Note: Package name starts with lowercase letter
import fracDimPy

# Import specific functions from submodules
from fracDimPy.monofractal import *
from fracDimPy.multifractal import *
from fracDimPy.generator import *
```

**Important Note**: Although the PyPI package name is `FracDimPy` (uppercase F), you need to use `import fracDimPy` (lowercase f) in your Python code.

### Quick Usage Examples

```python
from fracDimPy import hurst_dimension, box_counting, dfa
from fracDimPy import multifractal_curve, mf_dfa
from fracDimPy import generate_fbm_curve
import numpy as np

# Generate a fractal curve (returns curve, actual_dimension)
curve, actual_dim = generate_fbm_curve(dimension=1.5, length=2048)

# Monofractal analysis
D, result = hurst_dimension(curve)
print(f"Hurst dimension: {D:.4f}, R²: {result['R2']:.4f}")

D, result = dfa(curve)
print(f"DFA Hurst exponent: {result['alpha']:.4f}, R²: {result['r_squared']:.4f}")

# Box-counting on curve coordinates
x = np.arange(len(curve))
D, result = box_counting((x, curve), data_type="curve")
print(f"Box-counting dimension: {D:.4f}, R²: {result['R2']:.4f}")

# Multifractal analysis (single column)
metrics, figure_data = multifractal_curve(curve, data_type="single")
print(f"D(0)={metrics['D(0)'][0]:.4f}, D(1)={metrics['D(1)'][0]:.4f}, D(2)={metrics['D(2)'][0]:.4f}")

# MF-DFA
hq, spectrum = mf_dfa(curve)
q_arr = np.array(hq['q_list'])
idx_2 = np.where(np.abs(q_arr - 2) < 1e-10)[0][0]
print(f"h(2)={hq['h_q'][idx_2]:.4f}, spectrum width={spectrum['width']:.4f}")
```

> **Tip**: Analysis functions (`box_counting`, `multifractal_curve`, `multifractal_image`, ...) are silent by default; pass `verbose=True` for per-scale diagnostics. All stochastic generators accept a `seed` parameter for reproducible output.

---

## 📦 Module Description

### 1. Monofractal Module (`monofractal`)

Provides various monofractal dimension calculation methods:

| Method                | Function Name               | Data Type       | Description                                             |
| --------------------- | --------------------------- | --------------- | ------------------------------------------------------- |
| Hurst Exponent        | `hurst_dimension()`       | 1D time series  | R/S analysis, modified R/S, DFA                         |
| Box-counting          | `box_counting()`          | 1D/2D/3D        | Most commonly used fractal dimension calculation method |
| Information Dimension | `information_dimension()` | Point set data  | Dimension based on information entropy                  |
| Correlation Dimension | `correlation_dimension()` | Point set data  | Based on correlation integral                           |
| Structure Function    | `structural_function()`   | 1D curve        | Suitable for self-affine curves                         |
| Variogram             | `variogram_method()`      | 1D/2D           | Geostatistical method                                   |
| Sandbox               | `sandbox_method()`        | Point set/image | Local scale analysis                                    |
| DFA                   | `dfa()`                   | 1D time series  | Detrended Fluctuation Analysis                          |

### 2. Multifractal Module (`multifractal`)

Provides multifractal analysis tools:

| Function                 | Description                                 | Output                                                           |
| ------------------------ | ------------------------------------------- | ---------------------------------------------------------------- |
| `multifractal_curve()` | One-dimensional curve multifractal analysis | Partition function, generalized dimension, multifractal spectrum |
| `multifractal_image()` | Two-dimensional image multifractal analysis | Singularity index, multifractal characteristics                  |
| `mf_dfa()`             | Multifractal DFA                            | Fluctuation function, Hurst exponent spectrum                    |

### 3. Fractal Generator (`generator`)

Generates various theoretical and random fractals:

**Curve Class** (1D):

- `generate_fbm_curve()` - Fractional Brownian Motion curve
- `generate_wm_curve()` - Weierstrass-Mandelbrot function
- `generate_takagi_curve()` - Takagi curve
- `generate_koch_curve()` - Koch curve
- `generate_brownian_motion()` - Brownian motion
- `generate_levy_flight()` - Lévy flight

**Surface Class** (2D):

- `generate_fbm_surface()` - Fractional Brownian Motion surface
- `generate_wm_surface()` - WM surface
- `generate_takagi_surface()` - Takagi surface

**Pattern Class** (Geometric fractals):

- `generate_cantor_set()` - Cantor set
- `generate_sierpinski()` - Sierpinski triangle
- `generate_sierpinski_carpet()` - Sierpinski carpet
- `generate_vicsek_fractal()` - Vicsek fractal
- `generate_koch_snowflake()` - Koch snowflake
- `generate_dla()` - Diffusion-Limited Aggregation
- `generate_menger_sponge()` - Menger sponge (3D)

### 4. Utility Module (`utils`)

- Data I/O (`data_io`)
- Visualization tools (`plotting`)
- Shared computation utilities:
  - `fitting` - Log-log linear regression and R-squared computation
  - `scales` - Power-of-two scale generation
  - `box_counting_core` - Dimension-agnostic box counting primitives
  - `multifractal_common` - Shared multifractal partition/metrics computation
  - `image_drawing` - Bresenham line drawing and coordinate normalization
  - `conversion` - Coordinate-to-matrix, grayscale conversion, boundary padding

### Hurst Exponent Conventions

Different methods expose the Hurst exponent `H` under different conventions — be aware when comparing values across methods:

| Method | How `H` is obtained | Relation to dimension |
| --- | --- | --- |
| `hurst_dimension` | slope of R/S ~ r | `D = 2 - H` |
| `dfa` | slope `alpha` of F(n) ~ n | `H = alpha`, `D = 2 - alpha` |
| `variogram_method` | slope of γ(h) ~ h (= 2H) | `H = slope/2`; `D = 2 - H` (1D) / `3 - H` (surface) |
| `multifractal_curve` / `multifractal_image` | from the measure's generalized dimension `D(2)` | `H = (1 + D(2)) / 2` |
| `mf_dfa` | generalized Hurst exponent `h(q)` | `H = h(2)` (standard DFA) |

### Project Structure

```
src/fracDimPy/
├── __init__.py              # Package entry, exports all public functions
├── monofractal/             # Monofractal dimension methods
│   ├── hurst.py             # Hurst exponent (R/S analysis)
│   ├── box_counting.py      # Box-counting (1D/2D/3D)
│   ├── information_dimension.py
│   ├── correlation_dimension.py
│   ├── structural_function.py
│   ├── variogram.py
│   ├── sandbox.py
│   └── dfa.py               # Detrended Fluctuation Analysis
├── multifractal/            # Multifractal analysis
│   ├── mf_curve.py          # 1D curve multifractal
│   ├── mf_image.py          # 2D image multifractal
│   ├── mf_dfa.py            # Multifractal DFA
│   └── custom_epsilon.py    # Custom scale support
├── generator/               # Fractal generators
│   ├── curves.py            # FBM, WM, Takagi curves
│   ├── surfaces.py          # FBM, WM, Takagi surfaces
│   └── patterns.py          # Cantor, Sierpinski, Koch, DLA, Brownian, Menger...
└── utils/                   # Shared utilities
    ├── data_io.py            # Data loading and saving
    ├── plotting.py           # Visualization tools
    ├── fitting.py            # Log-log regression and R²
    ├── scales.py             # Power-of-two scale generation
    ├── box_counting_core.py  # Dimension-agnostic box counting
    ├── multifractal_common.py # Shared multifractal computation
    ├── image_drawing.py      # Bresenham line drawing
    └── conversion.py         # Coordinate/grayscale/boundary utilities
```

---

## 🔬 Application Areas

FracDimPy can be applied to multiple scientific and engineering fields:

- **Earth Sciences**: Terrain analysis, seismic data, fracture networks
- **Materials Science**: Porous media, surface roughness, nanostructures
- **Biomedical**: DNA sequences, protein folding, medical imaging
- **Financial Analysis**: Stock prices, market volatility, risk assessment
- **Image Processing**: Texture analysis, pattern recognition, image segmentation
- **Environmental Science**: River networks, cloud pattern analysis, pollution diffusion
- **Physics**: Turbulence, phase transitions, chaotic systems

---

## 📊 Examples and Data

The [tests](tests/) directory contains rich example code and test data:

```
tests/
├── monofractal/          # Monofractal method examples
│   ├── test_hurst.py
│   ├── test_box_counting_*.py
│   └── ...
├── multifractal/         # Multifractal examples
│   ├── test_mf_curve_*.py
│   ├── test_mf_image.py
│   └── ...
└── generator/            # Fractal generation examples
    ├── test_koch.py
    ├── test_dla.py
    └── ...
```

Run examples:

```bash
cd tests/monofractal
python test_hurst.py
```

---

## 🛠️ Dependencies

### Core Dependencies

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0
- Pandas >= 1.3.0

### All Dependencies Included

- NumPy >= 1.20.0 - Numerical computing foundation
- SciPy >= 1.7.0 - Scientific computing tools
- Matplotlib >= 3.3.0 - Data visualization
- Pandas >= 1.3.0 - Data processing
- Pillow >= 9.0.0 - Image I/O
- openpyxl >= 3.0.0 - Excel file support

**All dependencies are automatically installed. No manual installation needed for full functionality.**

For the complete dependency list, please refer to [pyproject.toml](pyproject.toml)

---

## 🤝 Contributing

Contributions of all kinds are welcome! Whether it's reporting bugs, suggesting new features, or submitting code improvements.

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

### Contributors

- **Zhile Han** - *Main Developer* - [Zhihu Profile](https://www.zhihu.com/people/xiao-xue-sheng-ye-xiang-xie-shu/posts)

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

---

## 📮 Contact

- **Author**: Zhile Han
- **Email**: 2667032759@qq.com
- **Address**: State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation, Southwest Petroleum University, Chengdu 610500, China
- **Zhihu**: [小学生也想写书](https://www.zhihu.com/people/xiao-xue-sheng-ye-xiang-xie-shu/posts)
- **GitHub**: [https://github.com/Kecoya/FracDimPy](https://github.com/Kecoya/FracDimPy)

---

## 📝 Citation

If you use FracDimPy in your research, please cite:

```bibtex
@software{fracdimpy2024,
  author = {Zhile Han},
  title = {FracDimPy: A Comprehensive Python Package for Fractal Dimension Calculation and Multifractal Analysis},
  year = {2024},
  url = {https://github.com/Kecoya/FracDimPy},
  version = {0.1.5}
}
```

---

## 📋 Changelog

### v0.1.5 (2026)

**API Hygiene & Reproducibility (conservative — no numerical change)**

- Standardized multifractal metrics keys to pure ASCII (`D(0)`, `alpha(q=0)`, ...) and routed `multifractal_curve` / `multifractal_image` through the shared `build_metrics`; removed legacy leading-space and CJK-bracket key names
- Added a `seed` parameter to every stochastic generator (`generate_fbm_curve`, `generate_fbm_surface`, `generate_wm_surface`, `generate_brownian_motion`, `generate_levy_flight`, `generate_self_avoiding_walk`, `generate_dla`) for reproducibility; `seed=None` preserves the previous random behaviour
- Removed global-state mutation: the `box_counting` random strategy now uses a local `RandomState(42)` (bit-identical output, no global seed pollution); `generate_wm_surface` no longer calls `np.random.seed()`
- `generate_fbm_curve(method=...)` / `generate_fbm_surface(method=...)` now raise `ValueError` on unsupported methods instead of silently falling through
- Gated all diagnostic `print` output behind `verbose=False` (`box_counting`, `multifractal_curve`, `multifractal_image`, `correlation_dimension`, `structural_function`, `custom_epsilon`)
- Fixed `plot_multifractal_spectrum` reading non-existent figure_data keys (τ(q)/α(q) were silently empty)
- Documented that the `sliding` / `random` box-counting normalisation is an empirical approximation; `fixed` remains the recommended strategy for strict estimation
- All 384 tests still pass; existing numerical results are unchanged

### v0.1.3 (2024)

**Architecture Refactoring**

- Extracted 6 shared utility modules (`fitting`, `scales`, `box_counting_core`, `multifractal_common`, `image_drawing`, `conversion`) to eliminate ~1000 lines of duplicated code across 16 source files
- Shared a dimension-agnostic core for the 'fixed' box-counting strategy across all data types (the 'sliding'/'random' strategies remain path-specific)
- Consolidated all log-log regression patterns (15+ occurrences) into `log_log_fit()` and `linear_fit()`
- Shared multifractal partition function computation between `mf_curve` and `mf_image`

**Configuration Cleanup**

- Consolidated `mypy` configuration into `pyproject.toml` (removed `mypy.ini`)
- Simplified `setup.py` to a minimal shim delegating to `pyproject.toml`
- Fixed `pyproject.toml` readme path, unified line-length settings

**Bug Fixes**

- Fixed `multifractal_image` print block referencing non-existent empty-string keys
- Fixed coordinate-to-matrix conversion in multifractal curve analysis

**Test Suite**

- All 384 tests passing (previously 297 pass / 88 fail)
- Expanded shared fixtures and signal generators in `conftest.py`
- Fixed generator test assertions (dtype checks, shape assertions, statistical thresholds)
- Aligned multifractal test key names with actual API return values
- Relaxed monofractal numerical tolerances to match algorithm capabilities

---

## 🙏 Acknowledgments

Thanks to all researchers and open-source community members who have contributed to fractal theory and algorithm implementation.

---

## ⭐ Star History

If this project is helpful to you, please give it a ⭐️!

---

## 🔗 Related Projects

- [NumPy](https://numpy.org/) - Numerical computing foundation
- [SciPy](https://scipy.org/) - Scientific computing tools
- [Matplotlib](https://matplotlib.org/) - Data visualization

---

<div align="center">

**[⬆ Back to Top](#fracdimpy)**

Made with ❤️ by Zhile Han

</div>
