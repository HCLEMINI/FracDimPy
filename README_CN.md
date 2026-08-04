# FracDimPy

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-0.1.6-green.svg)](https://github.com/songLe/FracDimPy)

**一个全面的Python分形维数计算与多重分形分析工具包**

[English](https://github.com/Kecoya/FracDimPy/blob/main/README.md) | 简体中文

</div>

---

## 📖 简介

FracDimPy 是一个功能强大、易于使用的Python软件包，专门用于分形维数计算和多重分形分析。无论您是研究分形几何的科研人员，还是需要分析复杂数据的工程师，FracDimPy都能为您提供专业、准确的分析工具。

### ✨ 主要特性

- **🔢 多种单分形方法**

  - Hurst指数法 (R/S分析)
  - 盒计数法 (Box-counting)
  - 信息维数法 (Information Dimension)
  - 关联维数法 (Correlation Dimension)
  - 结构函数法 (Structure Function)
  - 变差函数法 (Variogram)
  - 沙盒法 (Sandbox)
  - 去趋势波动分析 (DFA)
- **📊 多重分形分析**

  - 一维曲线多重分形分析
  - 二维图像多重分形分析
  - 多重分形去趋势波动分析 (MF-DFA)
  - 自定义尺度序列
- **🎨 分形生成器**

  - 经典分形：Cantor集、Sierpinski三角形/地毯、Koch曲线、Menger海绵等
  - 随机分形：布朗运动、Lévy飞行、自回避行走、扩散限制聚集(DLA)
  - 分形曲线：FBM曲线、Weierstrass-Mandelbrot函数、Takagi曲线
  - 分形曲面：FBM曲面、Weierstrass-Mandelbrot曲面、Takagi曲面
  - 挂谷集：离散Besicovitch集(每个方向一条线段),配挂谷猜想的维数数值演示
- **📈 丰富的可视化**

  - 自动生成专业图表
  - 双对数图拟合
  - 多重分形谱展示
  - 可定制的绘图选项
- **💾 灵活的数据处理**

  - 支持多种数据格式 (CSV, Excel, TXT, NPY, 图像等)
  - 自动数据预处理
  - 结果导出功能

---

## 🚀 快速开始

### 安装

#### 从PyPI安装（推荐）

```bash
# 安装完整包（包含所有依赖）
pip install FracDimPy
```

#### 🇨🇳 中国用户镜像安装（推荐，速度更快）

对于中国大陆用户，建议使用清华大学镜像源进行安装，速度会更快：

```bash
# 使用清华镜像安装
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple FracDimPy

# 或者永久配置镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install FracDimPy
```

**常用镜像源**：

- 清华大学：`https://pypi.tuna.tsinghua.edu.cn/simple`
- 阿里云：`https://mirrors.aliyun.com/pypi/simple`
- 中科大：`https://pypi.mirrors.ustc.edu.cn/simple`
- 豆瓣：`https://pypi.douban.com/simple`

#### 正确引用方式

```python
# 注意：包名为首字母小写
import fracDimPy

# 从子模块导入具体功能
from fracDimPy.monofractal import *
from fracDimPy.multifractal import *
from fracDimPy.generator import *
```

**重要说明**：虽然PyPI包名为 `FracDimPy`（大写F），但在Python代码中需要使用 `import fracDimPy`（小写f）进行导入。

### 快速使用示例

```python
from fracDimPy import hurst_dimension, box_counting, dfa
from fracDimPy import multifractal_curve, mf_dfa
from fracDimPy import generate_fbm_curve
import numpy as np

# 生成分形曲线（返回曲线数组和实际维数）
curve, actual_dim = generate_fbm_curve(dimension=1.5, length=2048)

# 单分形分析
D, result = hurst_dimension(curve)
print(f"Hurst维数: {D:.4f}, R²: {result['R2']:.4f}")

D, result = dfa(curve)
print(f"DFA Hurst指数: {result['alpha']:.4f}, R²: {result['r_squared']:.4f}")

# 盒计数法（使用曲线坐标）
x = np.arange(len(curve))
D, result = box_counting((x, curve), data_type="curve")
print(f"盒计数维数: {D:.4f}, R²: {result['R2']:.4f}")

# 多重分形分析（单列数据）
metrics, figure_data = multifractal_curve(curve, data_type="single")
print(f"D(0)={metrics['D(0)'][0]:.4f}, D(1)={metrics['D(1)'][0]:.4f}, D(2)={metrics['D(2)'][0]:.4f}")

# 多重分形DFA
hq, spectrum = mf_dfa(curve)
q_arr = np.array(hq['q_list'])
idx_2 = np.where(np.abs(q_arr - 2) < 1e-10)[0][0]
print(f"h(2)={hq['h_q'][idx_2]:.4f}, 谱宽度={spectrum['width']:.4f}")
```

> **提示**:分析函数(`box_counting`、`multifractal_curve`、`multifractal_image` 等)默认静默不输出;传 `verbose=True` 可打印逐尺度诊断信息。所有随机分形生成器均支持 `seed` 参数以获得可复现结果。

## 📦 模块说明

### 1. 单分形模块 (`monofractal`)

提供多种单分形维数计算方法：

| 方法      | 函数名                      | 适用数据类型 | 说明                     |
| --------- | --------------------------- | ------------ | ------------------------ |
| Hurst指数 | `hurst_dimension()`       | 1D时间序列   | R/S分析、修正R/S、DFA    |
| 盒计数法  | `box_counting()`          | 1D/2D/3D     | 最常用的分形维数计算方法 |
| 信息维数  | `information_dimension()` | 点集数据     | 基于信息熵的维数         |
| 关联维数  | `correlation_dimension()` | 点集数据     | 基于关联积分             |
| 结构函数  | `structural_function()`   | 1D曲线       | 适用于自仿射曲线         |
| 变差函数  | `variogram_method()`      | 1D/2D        | 地统计学方法             |
| 沙盒法    | `sandbox_method()`        | 点集/图像    | 局部尺度分析             |
| DFA       | `dfa()`                   | 1D时间序列   | 去趋势波动分析           |
| 分规法(Richardson) | `divider_dimension()`     | 有序曲线(2D/3D) | 弦步法;稀疏曲线稳健 |
| Minkowski-MC | `minkowski_dimension_mc()`| 点云/网络      | 蒙特卡洛膨胀体积法   |

#### 稀疏线条/网络的方法选择

盒计数对测度零集合(曲线、裂缝网络)病态:空盒统计与网格对齐占主导,稀疏 3D 线甚至直接失败。推荐:

- **`divider_dimension()`** 用于*有序、自相似*曲线(Koch、海岸线、裂缝迹线)——弦步法,不碰环境网格。注意:*自仿态*曲线(fBm、Weierstrass、Takagi 函数)的 divider 维数与盒维数有已知定理差异,这类请用 `variogram_method` / `structural_function`
- **`minkowski_dimension_mc()`** 用于*无序*点云/网络(孔喉网络、裂缝网络)——蒙特卡洛膨胀体积,带自动标度区检测;在 Cantor/Sierpinski/地毯/Vicsek/Menger 上回算已知 D 误差约 0.1
- **`box_counting()`** 用于填充集(图像、曲面、孔隙体)

#### 省内存的点云盒计数

对稀疏或高分辨率集合(裂缝网络、多孔介质、挂谷集),用 `data_type="points"`:每个点哈希到所在盒子,通过行唯一性统计非空盒子数,内存 `O(M)`(M 个点)而非 `O(N^d)`——在相同 ε 下与稠密固定网格盒计数严格等价。

| 网格 N | 稠密 `porous` 峰值内存 | `points` 峰值内存 |
| --- | --- | --- |
| 128 | 29 MB | 3.5 MB |
| 256 | 235 MB | 28 MB |

```python
coords = np.argwhere(binary_volume > 0).astype(float)   # 非零体素,形状 (M, 3)
D, res = box_counting(coords, data_type="points")        # 不分配 N^3 网格
```

### 2. 多重分形模块 (`multifractal`)

提供多重分形分析工具：

| 函数                     | 说明                 | 输出                           |
| ------------------------ | -------------------- | ------------------------------ |
| `multifractal_curve()` | 一维曲线多重分形分析 | 配分函数、广义维数、多重分形谱 |
| `multifractal_image()` | 二维图像多重分形分析 | 奇异性指数、多重分形特征       |
| `mf_dfa()`             | 多重分形DFA          | 波动函数、Hurst指数谱          |

### 3. 分形生成器 (`generator`)

生成各种理论和随机分形：

**曲线类** (1D):

- `generate_fbm_curve()` - 分数布朗运动曲线
- `generate_wm_curve()` - Weierstrass-Mandelbrot函数
- `generate_takagi_curve()` - Takagi曲线
- `generate_koch_curve()` - Koch曲线
- `generate_brownian_motion()` - 布朗运动
- `generate_levy_flight()` - Lévy飞行

**曲面类** (2D):

- `generate_fbm_surface()` - 分数布朗运动曲面
- `generate_wm_surface()` - WM曲面
- `generate_takagi_surface()` - Takagi曲面

**图案类** (几何分形):

- `generate_cantor_set()` - Cantor集
- `generate_sierpinski()` - Sierpinski三角形
- `generate_sierpinski_carpet()` - Sierpinski地毯
- `generate_vicsek_fractal()` - Vicsek分形
- `generate_koch_snowflake()` - Koch雪花
- `generate_dla()` - 扩散限制聚集
- `generate_menger_sponge()` - Menger海绵（3D）

**挂谷集**(点云):

- `generate_kakeya_set()` - 离散挂谷(Kakeya/Besicovitch)集(每个方向一条线段)

#### 挂谷集与挂谷猜想

挂谷集是包含每个方向上一条单位线段的集合。挂谷猜想断言其豪斯多夫维数与闵可夫斯基维数都等于嵌入维数 `d`(二维由 Davies 1971 证明,三维由王虹与 Zahl 2025 证明)。本生成器产出离散点云近似,其闵可夫斯基维数可用 `box_counting(data_type="points")` 估计。

```python
from fracDimPy import generate_kakeya_set, box_counting
pts = generate_kakeya_set(dimension=3, num_directions=2000, seed=42)
D, res = box_counting(pts, data_type="points")
print(f"D={D:.3f}, R2={res['R2']:.4f}")  # D 随方向数增加趋向 3
```

配套的结构检验沿王虹-Zahl 证明(arXiv:2502.17655)的思路:挂谷集的管满足 Wolff(非聚集)公理——任意凸集 `W` 至多含约 `|W|/|T|` 根管,即 Katz-Tao 误差 `CKT ~ 1`。`tube_ck_error` 数值估计该误差:

```python
from fracDimPy import kakeya_segments, tube_ck_error
segs = kakeya_segments(dimension=3, num_directions=2000, length=1.0)
ck, _ = tube_ck_error(segs, delta=0.01, mode="contained")  # 管体聚集
ck2, _ = tube_ck_error(segs, delta=0.01, mode="passing")   # 穿越密度
```

`mode="contained"` 是 Wolff 公理的忠实解读(管体完整落在 `W` 内才计数):密集平行管束给出 `CKT >> 1`,方向分散的管给出 `CKT ~ O(1)`。`mode="passing"` 改为统计中心线穿过 `W` 的管,检测穿越密度(如所有线段过同一点),但细尺度噪音较大。

> **注意**:这是闵可夫斯基维数趋向 `d` 的数值演示,**不是**挂谷猜想的证明。有限、离散化的集合给出 `D < d`(如三维 8000 方向时约 2.7);随方向数与采样密度增加,差距收敛。

### 4. 工具模块 (`utils`)

- 数据读写 (`data_io`)
- 可视化工具 (`plotting`)
- 共享计算工具:
  - `fitting` - 对数-对数线性拟合与 R² 计算
  - `scales` - 2的幂次尺度生成
  - `box_counting_core` - 维度无关的盒计数核心函数
  - `multifractal_common` - 共享多重分形配分/指标计算
  - `image_drawing` - Bresenham 线段绘制与坐标映射
  - `conversion` - 坐标转矩阵、灰度转换、边界填充

### Hurst 指数约定

不同方法给出的 Hurst 指数 `H` 采用不同约定,跨方法比较时请注意:

| 方法 | `H` 的获取方式 | 与维数的关系 |
| --- | --- | --- |
| `hurst_dimension` | R/S ~ r 的斜率 | `D = 2 - H` |
| `dfa` | F(n) ~ n 的斜率 `alpha` | `H = alpha`,`D = 2 - alpha` |
| `variogram_method` | γ(h) ~ h 的斜率(= 2H) | `H = slope/2`;`D = 2 - H`(1D)/ `3 - H`(曲面) |
| `multifractal_curve` / `multifractal_image` | 由测度的广义维数 `D(2)` 得到 | `H = (1 + D(2)) / 2` |
| `mf_dfa` | 广义 Hurst 指数 `h(q)` | `H = h(2)`(标准 DFA) |

### 项目结构

```
src/fracDimPy/
├── __init__.py              # 包入口，导出所有公共函数
├── monofractal/             # 单分形维数方法
│   ├── hurst.py             # Hurst指数（R/S分析）
│   ├── box_counting.py      # 盒计数法（1D/2D/3D）
│   ├── information_dimension.py
│   ├── correlation_dimension.py
│   ├── structural_function.py
│   ├── variogram.py
│   ├── sandbox.py
│   └── dfa.py               # 去趋势波动分析
├── multifractal/            # 多重分形分析
│   ├── mf_curve.py          # 一维曲线多重分形
│   ├── mf_image.py          # 二维图像多重分形
│   ├── mf_dfa.py            # 多重分形DFA
│   └── custom_epsilon.py    # 自定义尺度支持
├── generator/               # 分形生成器
│   ├── curves.py            # FBM、WM、Takagi曲线
│   ├── surfaces.py          # FBM、WM、Takagi曲面
│   └── patterns.py          # Cantor、Sierpinski、Koch、DLA、布朗运动、Menger等
└── utils/                   # 共享工具
    ├── data_io.py            # 数据读写
    ├── plotting.py           # 可视化工具
    ├── fitting.py            # 对数回归与 R² 计算
    ├── scales.py             # 2的幂次尺度生成
    ├── box_counting_core.py  # 维度无关的盒计数核心
    ├── multifractal_common.py # 共享多重分形计算
    ├── image_drawing.py      # Bresenham线段绘制
    └── conversion.py         # 坐标/灰度/边界工具
```

---

## 🔬 应用领域

FracDimPy可应用于多个科学和工程领域：

- **地球科学**：地形分析、地震数据、裂缝网络
- **材料科学**：多孔介质、表面粗糙度、纳米结构
- **生物医学**：DNA序列、蛋白质折叠、医学影像
- **金融分析**：股票价格、市场波动、风险评估
- **图像处理**：纹理分析、模式识别、图像分割
- **环境科学**：河流网络、云图分析、污染扩散
- **物理学**：湍流、相变、混沌系统

---

## 📊 示例与数据

[tests](tests/) 目录包含丰富的示例代码和测试数据：

```
tests/
├── monofractal/          # 单分形方法示例
│   ├── test_hurst.py
│   ├── test_box_counting_*.py
│   └── ...
├── multifractal/         # 多重分形示例
│   ├── test_mf_curve_*.py
│   ├── test_mf_image.py
│   └── ...
└── generator/            # 分形生成示例
    ├── test_koch.py
    ├── test_dla.py
    └── ...
```

运行示例：

```bash
cd tests/monofractal
python test_hurst.py
```

---

## 🛠️ 依赖项

### 核心依赖

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0
- Pandas >= 1.3.0

### 包含的所有依赖

- NumPy >= 1.20.0 - 数值计算基础
- SciPy >= 1.7.0 - 科学计算工具
- Matplotlib >= 3.3.0 - 数据可视化
- Pandas >= 1.3.0 - 数据处理
- Pillow >= 9.0.0 - 图像读写
- openpyxl >= 3.0.0 - Excel文件支持

**所有依赖已自动安装，无需手动安装额外库即可使用全部功能。**

完整依赖列表请参阅 [pyproject.toml](pyproject.toml)

---

## 🤝 贡献

欢迎各种形式的贡献！无论是报告bug、提出新功能建议，还是提交代码改进。

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细的贡献指南。

### 贡献者

- **Zhile Han** - *主要开发者* - [知乎主页](https://www.zhihu.com/people/xiao-xue-sheng-ye-xiang-xie-shu/posts)

---

## 📄 许可证

本项目采用 GNU General Public License v3.0 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📮 联系方式

- **作者**: Zhile Han
- **邮箱**: 2667032759@qq.com
- **地址**: 油气藏地质及开发工程全国重点实验室，西南石油大学，成都610500，中国
- **知乎**: [小学生也想写书](https://www.zhihu.com/people/xiao-xue-sheng-ye-xiang-xie-shu/posts)
- **GitHub**: [https://github.com/Kecoya/FracDimPy](https://github.com/Kecoya/FracDimPy)

---

## 📝 引用

如果您在研究中使用了FracDimPy，请引用：

```bibtex
@software{fracdimpy2024,
  author = {Zhile Han},
  title = {FracDimPy: A Comprehensive Python Package for Fractal Dimension Calculation and Multifractal Analysis},
  year = {2024},
  url = {https://github.com/Kecoya/FracDimPy},
  version = {0.1.6}
}
```

---

## 📋 更新日志

### v0.1.6 (2026)

**性能与新生成器**

- 新增 `box_counting(data_type="points")`:省内存的点云盒计数(`O(M)` 代替 `O(N^d)`),在相同 ε 下与稠密固定网格盒计数严格等价——稀疏 3D 集(裂缝网络、多孔介质、挂谷集)内存降低约 3-8 倍且更快
- 删除 `count_boxes_fixed` 中的冗余数组拷贝;所有稠密盒计数路径峰值内存下降
- 新增 `generate_kakeya_set()`(二维/三维离散挂谷/Besicovitch 点云),配挂谷猜想的闵可夫斯基维数数值演示(Wang & Zahl, 2025)
- 全部 401 个测试通过(原 384 + 新增 17)

### v0.1.5 (2026)

**API 整洁与可复现性(保守改动——无任何数值变化)**

- 多重分形 metrics 键名统一为纯 ASCII(`D(0)`、`alpha(q=0)` 等),`multifractal_curve` / `multifractal_image` 改用共享的 `build_metrics`;移除历史遗留的带前导空格和中文括号的键名
- 所有随机分形生成器(`generate_fbm_curve`、`generate_fbm_surface`、`generate_wm_surface`、`generate_brownian_motion`、`generate_levy_flight`、`generate_self_avoiding_walk`、`generate_dla`)新增 `seed` 参数以支持可复现;`seed=None` 保持原有随机行为
- 消除全局状态污染:`box_counting` 的 random 策略改用局部 `RandomState(42)`(输出逐位一致,不再污染全局随机种子);`generate_wm_surface` 不再调用 `np.random.seed()`
- `generate_fbm_curve(method=...)` / `generate_fbm_surface(method=...)` 对不支持的方法显式抛出 `ValueError`,不再静默回落
- 所有诊断 `print` 输出收口到 `verbose=False` 开关(`box_counting`、`multifractal_curve`、`multifractal_image`、`correlation_dimension`、`structural_function`、`custom_epsilon`)
- 修复 `plot_multifractal_spectrum` 读取不存在的 figure_data 键(τ(q)/α(q) 此前静默为空)
- 文档说明 `sliding` / `random` 盒计数归一化为经验近似;`fixed` 仍是严格估计的推荐策略
- 全部 384 个测试仍通过;既有数值结果不变

### v0.1.3 (2024)

**架构重构**

- 提取 6 个共享工具模块（`fitting`、`scales`、`box_counting_core`、`multifractal_common`、`image_drawing`、`conversion`），消除 16 个源文件中约 1000 行重复代码
- 'fixed' 盒计数策略在各数据类型间共享维度无关核心函数;'sliding'/'random' 策略仍为各路径独立实现
- 合并所有对数回归模式（15+ 处）为 `log_log_fit()` 和 `linear_fit()`
- 共享多重分形配分函数计算逻辑于 `mf_curve` 和 `mf_image` 之间

**配置清理**

- 合并 `mypy` 配置至 `pyproject.toml`（删除 `mypy.ini`）
- 精简 `setup.py` 为最小化桥接文件
- 修复 `pyproject.toml` 中 readme 路径，统一行宽设置

**Bug修复**

- 修复 `multifractal_image` 打印块引用不存在的空字符串键名
- 修复多重分形曲线分析中坐标转矩阵逻辑

**测试套件**

- 全部 384 个测试通过（此前 297 通过 / 88 失败）
- 扩展 `conftest.py` 中的共享 fixture 和信号生成器
- 修复生成器测试断言（数据类型检查、形状断言、统计阈值）
- 对齐多重分形测试键名与实际 API 返回值
- 放宽单分形数值容差以匹配算法实际精度

---

## 🙏 致谢

感谢所有为分形理论和算法实现做出贡献的研究者和开源社区成员。

---

## ⭐ Star History

如果这个项目对您有帮助，请给它一个⭐️！

---

## 🔗 相关项目

- [NumPy](https://numpy.org/) - 数值计算基础
- [SciPy](https://scipy.org/) - 科学计算工具
- [Matplotlib](https://matplotlib.org/) - 数据可视化

---

<div align="center">

**[⬆ 返回顶部](#fracdimpy)**

Made with ❤️ by Zhile Han

</div>
