# 恶劣天气图像复原系统

基于扩散模型的恶劣天气图像复原系统，支持雨、雾、雪三类退化场景的复原与评测。

**课题目标**：在雨、雾、雪等恶劣天气条件下，对采集的图像进行处理，消除或减弱天气退化效应，恢复图像清晰度和目标细节。

**技术路线**：采用 WeatherDiffusion（Özdenizci & Legenstein, TPAMI 2023）作为主方案，这是一种 patch-based 条件扩散模型，能够在复原过程中生成更丰富的纹理细节，适合处理真实场景中多种天气退化叠加的问题。

---

## 1. 快速开始

### 1.1 环境要求

- **Python 3.10**（推荐，3.8+ 可用）
- **操作系统**：Windows 10/11、Linux
- **GPU**：NVIDIA 显卡（可选，扩散模型推理强烈建议 GPU）

### 1.2 安装步骤

#### 第一步：创建 conda 虚拟环境

```bash
conda create -n graduation_project python=3.10 -y
conda activate graduation_project
```

#### 第二步：安装 PyTorch（按 CUDA 版本选择）

```bash
# CUDA 12.1（推荐，RTX 30/40 系列显卡）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8（老显卡）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 仅 CPU（无 NVIDIA 显卡时使用，扩散模型会很慢）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

验证安装：

```bash
python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### 第三步：安装其他依赖

```bash
pip install -r requirements.txt
```

依赖分类说明：

| 类别 | 包 | 用途 |
|---|---|---|
| 核心 | numpy, pillow, pyyaml | 图像处理、配置加载 |
| 界面 | PyQt5 | 可视化界面 |
| 指标 | scikit-image | PSNR / SSIM |
| 深度学习 | torch, torchvision | 扩散模型推理 |
| 扩展指标 | lpips, pyiqa, pytorch-fid | LPIPS / NIQE / BRISQUE / FID |
| 工具 | pyqtgraph, tqdm, requests | 柱状图、进度条、权重下载 |

#### 第四步：下载预训练权重

```bash
python scripts/download_weights.py
```

权重文件约 1.2 GB，下载后自动校验完整性。如遇网络问题，脚本支持断点续传。

#### 第五步：环境自检

```bash
python scripts/smoke_test.py
```

自检通过会打印：配置发现、传统方法全流程、DCP 算法、指标计算、批量评测、界面依赖、权重状态、GPU 状态。

### 1.3 运行系统

#### 命令行演示（无需界面）

```bash
# 生成合成样本数据
python scripts/make_samples.py

# 一键演示：出指标对比表 + 三联对比图
python scripts/demo.py

# 指定方法（传统方法 + 扩散模型）
python scripts/demo.py --methods classical,dcp,rain --limit 4
```

#### 可视化界面

```bash
python app.py
```

界面操作：**载入示例 → 选方法 → 开始复原**，右侧显示 PSNR/SSIM/耗时，中间可拖动中缝做擦除对比。

**快捷键**：
- `Ctrl+O` 打开图片
- `Ctrl+E` 载入示例
- `Ctrl+S` 保存结果
- `Ctrl+D` 切换对比模式（擦除/并排）
- `Esc` 取消

**环境变量**：
- `AWR_THEME=light` 切换浅色主题（默认深色）
- `AWR_DEVICE=cpu` 强制使用 CPU

```bash
# Windows PowerShell
$env:AWR_THEME="light"; python app.py

# Linux / macOS
AWR_THEME=light python app.py
```

---

## 2. 目录结构

```
.
├── app.py                    界面启动入口
├── requirements.txt          依赖清单
├── configs/                  每种天气/方法一个 yaml，新增方法不用改代码
│   ├── rain.yaml             去雨（扩散模型，需权重）
│   ├── haze.yaml             去雾（扩散模型，需权重）
│   ├── snow.yaml             去雪（扩散模型，需权重）
│   ├── allweather.yaml       统一模型（扩散模型，需权重）
│   ├── classical.yaml        传统方法（无需权重，雨/雾/雪自动判别）
│   └── dcp.yaml              暗通道先验去雾（对照基线）
├── weights/                  权重目录（不入库，见下文）
├── data/
│   └── samples/              make_samples.py 生成的合成样本（不入库）
├── core/                     核心层：与界面完全解耦
│   ├── base.py               接口契约 BaseRestorer / RestoreResult
│   ├── registry.py           模型注册表，按配置动态构造
│   ├── config.py             yaml 加载与路径解析
│   ├── image_utils.py        图像 IO、预处理、滤波算子
│   ├── synth.py              合成样本生成（清晰场景 + 雨/雾/雪退化模型）
│   ├── metrics.py            PSNR/SSIM/LPIPS/NIQE/BRISQUE/FID
│   ├── weatherdiff/          扩散模型实现
│   │   ├── restorer.py       权重加载/EMA/预处理/OOM 重试
│   │   ├── unet.py           DiffusionUNet（72M 参数）
│   │   └── sampling.py       patch-based DDIM 采样
│   └── baseline/
│       ├── classical.py      传统方法：自动判别雨/雾/雪 + 分层处理
│       └── dcp.py            暗通道先验去雾
├── engine/                   编排层：命令行与界面共用
│   ├── inference.py          ModelCache（权重缓存）、单图/批量推理
│   └── evaluate.py           批量评测、汇总、CSV 导出
├── ui/                       PyQt5 界面层
│   ├── theme.py              统一主题：调色盘 + 全局 QSS（深/浅两套）
│   ├── main_window.py        三个 Tab：单图复原 / 批量评测 / 环境自检
│   ├── compare_view.py       对比控件（擦除+并排、可导出）
│   ├── panels.py             参数面板 + 指标面板
│   ├── worker.py             QThread 后台推理线程
│   └── qt_utils.py           numpy <-> QImage 转换
└── scripts/
    ├── smoke_test.py         全链路自检（第一件事跑这个）
    ├── make_samples.py       生成合成样本数据
    ├── demo.py               一键演示：指标对比表 + 三联对比图
    ├── run_eval.py           命令行批量评测，支持多方法对比
    ├── ui_snapshot.py        把三个页面渲染成 PNG（答辩配图用）
    ├── download_weights.py   权重下载（断点续传 + 完整性校验）
    ├── check_weights.py      checkpoint 结构体检
    ├── tune_classical.py     传统方法参数网格搜索
    └── check_detect.py       退化类型判别准确率检查
```

---

## 3. 核心约定

1. 图像在内存中一律是 **RGB uint8 (H, W, 3)** 的 numpy 数组
2. 所有复原方法都继承 `core/base.py` 的 `BaseRestorer`，实现 `load_model()` 与 `restore()`
3. 上层统一调用 `restore_with_stats()`（自带计时、尺寸校验），不要直接调 `restore()`
4. `restore()` 输出尺寸必须与输入完全一致
5. 进度用 `progress_cb(step, total, preview)`，取消用 `cancel_cb() -> bool`，被取消时抛 `RestoreCancelled`
6. 界面只 import `core`、`engine`，**不允许** import `core.weatherdiff.*`
7. `AWR_DEVICE=cpu` 可强制用 CPU（默认自动选 GPU/CPU）

---

## 4. 模型与权重

### 4.1 主方案：WeatherDiffusion

- **论文**：Özdenizci & Legenstein, "Learning to See by Converting Noise to Clear Images," TPAMI 2023
- **架构**：patch-based 条件扩散模型，72M 参数 UNet
- **特点**：官方只公开统一的多天气模型，rain / haze / snow 三个配置共用同一份权重
- **推理**：DDIM 采样，默认 25 步，patch 大小 64×64，滑窗步长 16

### 4.2 权重下载

```bash
# 下载默认权重（WeatherDiff64）
python scripts/download_weights.py

# 查看权重状态
python scripts/download_weights.py --list

# 校验本地文件完整性
python scripts/download_weights.py --verify

# 强制重新下载
python scripts/download_weights.py --force
```

权重直链（TU Graz 服务器）：

| 文件名 | 大小 | 说明 |
|---|---|---|
| WeatherDiff64.pth.tar | 1.24 GB | patch=64，configs 默认用它 |
| WeatherDiff128.pth.tar | 更大 | patch=128，需改 `data.image_size` 为 128 |

### 4.3 备选方案

| 方案 | 仓库 | 特点 |
|---|---|---|
| DiffUIR | https://github.com/iSEE-Laboratory/DiffUIR | CVPR2024 统一复原 |
| Restormer | https://github.com/swz30/Restormer | 非扩散对照 |
| TransWeather | https://github.com/jeya-maria-jose/TransWeather | 多天气单模型 |
| DehazeFormer | https://github.com/IDKiro/DehazeFormer | 纯去雾兜底 |

---

## 5. 数据集

### 5.1 目录组织

批量评测要求"退化图目录"与"GT 目录"分开存放，文件名主干一致：

```
data/
├── raindrop/            # 雨滴 (RainDrop test_a)
│   ├── input/
│   └── gt/
├── rainfog/             # 雨+雾 (Outdoor-Rain test1)
│   ├── input/
│   └── gt/
├── snow/                # 雪 (Snow100K 测试子集)
│   ├── input/
│   └── gt/
├── haze/                # 纯雾 (RESIDE SOTS-outdoor 子集)
│   ├── input/
│   └── gt/
└── real/                # 真实网络图，无 GT，只算无参考指标
    └── input/
```

文件名可以带后缀，程序会自动配对（如 `0_rain.png` ↔ `0_clean.png`）。

### 5.2 数据集来源

| 天气 | 数据集 | 地址 |
|---|---|---|
| 雨滴 | RainDrop | https://github.com/rui1996/DeRaindrop |
| 雨+雾 | Outdoor-Rain | https://github.com/liruoteng/HeavyRainRemoval |
| 雪 | Snow100K | https://sites.google.com/view/yunfuliu/desnownet |
| 纯雾 | RESIDE | https://sites.google.com/view/reside-dehaze-datasets |
| 三合一 | AllWeather | https://github.com/jeya-maria-jose/TransWeather |

每类**只取 50~200 对**测试图即可，FID 需要每个目录至少 50 张。

### 5.3 没有真实数据时

`scripts/make_samples.py` 能生成合成样本（雨/雾/雪退化图 + GT），用于开发和测试：

```bash
python scripts/make_samples.py --count 8
```

---

## 6. 常用命令

```bash
# 环境自检
python scripts/smoke_test.py

# 生成样本数据
python scripts/make_samples.py --count 8 --force

# 一键演示
python scripts/demo.py
python scripts/demo.py --methods classical,dcp,rain --limit 4

# 可视化界面
python app.py                                    # 深色主题
$env:AWR_THEME="light"; python app.py            # 浅色主题（Windows）
$env:AWR_DEVICE="cpu"; python app.py             # 强制 CPU

# 界面截图（答辩配图）
python scripts/ui_snapshot.py
python scripts/ui_snapshot.py --theme light

# 命令行批量评测
python scripts/run_eval.py --config classical --input data/samples/rain/input --gt data/samples/rain/gt
python scripts/run_eval.py --config classical,dcp --input data/samples/snow/input --gt data/samples/snow/gt

# 只算无参考指标（真实图片无 GT）
python scripts/run_eval.py --config classical --input data/samples/real/input --metrics niqe,brisque

# 传统方法调参 / 退化类型判别检查
python scripts/tune_classical.py --weather rain
python scripts/check_detect.py
```

---

## 7. 常见问题

### 7.1 `ModuleNotFoundError: No module named 'numpy'`

**原因**：使用了系统 Python 而非 conda 环境。

**解决**：

```bash
# 先激活 conda 环境
conda activate graduation_project

# 或用完整路径
E:\miniconda\envs\graduation_project\python.exe app.py
```

注意：Windows 上 `py` 启动器会绕过 conda，请用 `python` 而非 `py`。

### 7.2 `numpy.dtype size changed, may indicate binary incompatibility`

**原因**：numpy 2.x 与 scikit-image 等库二进制不兼容。

**解决**：

```bash
pip install "numpy<2" --force-reinstall
```

### 7.3 `torchvision::nms does not exist`

**原因**：torch 与 torchvision 版本不匹配。

**解决**：

```bash
# 重新安装匹配的 torchvision
pip install torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 7.4 `PytorchStreamReader failed reading zip archive`

**原因**：权重文件下载不完整或损坏。

**解决**：

```bash
# 校验并重新下载
python scripts/download_weights.py --verify
python scripts/download_weights.py --force
```

### 7.5 扩散模型推理很慢

**原因**：CPU 推理或分辨率过大。

**解决**：
- 使用 GPU：确保 torch 安装了 CUDA 版本，且 NVIDIA 驱动正常
- 减小分辨率：配置里 `data.max_side` 改小（如 256）
- 减少采样步数：`sampling.timesteps` 改为 10-15（质量会下降）
- 减小 patch 批处理：`sampling.patch_batch_size` 改小（显存不足时自动减半）

### 7.6 显存不足 (CUDA out of memory)

系统会自动减半 `patch_batch_size` 重试。如果仍然 OOM：
- 减小 `data.max_side`（如 256）
- 减小 `sampling.grid_r`（增加 patch 重叠，但更慢）

### 7.7 界面启动闪退或报错

先跑 `python scripts/smoke_test.py` 看哪项失败，常见原因：
- PyQt5 未安装：`pip install PyQt5`
- 依赖版本冲突：重建 conda 环境

---

## 8. 扩展选做

- 模型轻量化（剪枝/量化/蒸馏）
- 视频时序增强
- 局部放大镜、滚轮缩放等高级交互
- pyqtgraph 柱状图展示批量评测结果
