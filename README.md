# LensOracle

**English** | [中文](#lensoracle-中文)

---

## LensOracle (English)

LensOracle is an offline deep learning research framework for **lens-table-guided defocus image restoration**. It introduces a physically motivated pipeline that conditions a restoration network on native PSF-SFR lens tables — calibrated optical characterizations stored as `[64, 48, 67]` tensors in polar coordinates — rather than pixel-level blur maps derived from ad-hoc preprocessing.

The project is built around the [BlindCAC / OmniLens++](https://github.com/zju-jiangqi/BlindCAC) dataset (NTIRE 2026), which provides paired defocus images and ground-truth PSF-SFR lens tables across 3,240 distinct lenses.

> **Note:** This repository contains the offline training and evaluation pipeline only. It does not include any hardware communication, camera control, or online closed-loop inference modules.

---

### Architecture

The active pipeline follows a three-stage lens-table fusion design:

```
Blur Image
  → Stage 1: BlindPriorEstimator
  → pred_psf_sfr  [B, 64, 48, 67]
  → Stage 2: OpticalDegradationNetwork (ODN)   ← physics closed-loop
  → Stage 3: LensTableEncoder + CoordGateNAFNetRestoration
  → Restored Image
```

**Stage 1 — BlindPriorEstimator:** Predicts the native PSF-SFR lens table from a single blurry image. Supervised directly by the calibrated `gt_psf_sfr` tensor; no pixel-level PSF maps are used.

**Stage 2 — OpticalDegradationNetwork (ODN):** Given a sharp image and the predicted lens table, simulates the degraded (blurry) image. Validates the physical consistency of the predicted prior in a forward-modeling closed loop. The stage is frozen after training and is not used at inference time.

**Stage 3 — LensTableEncoder + Restoration:** Encodes the multi-scale optical features from the lens table and injects them into the restoration backbone via coordinate-aware cross-attention. Key design decisions include:

- **CircularConv2d** — applies circular padding along the angular (θ) dimension and zero-padding along the radial (r) dimension of the lens table, respecting the periodic boundary of azimuthal angle while preserving the non-periodic nature of the radial field.
- **CrossAttentionRouter** — routes lens-table tokens to image feature tokens using Fourier-encoded polar coordinates `(r, θ)`, enabling physically grounded spatial conditioning.
- **CoordGateNAFBlock** — a NAFBlock variant whose depthwise convolution branch is gated by a coordinate-derived spatial mask, making each feature transformation position-aware.

---

### Repository Structure

```
LensOracle/
├── config/
│   ├── __init__.py          # Strongly-typed config dataclasses + YAML loader
│   └── default.yaml         # Default hyperparameters and data paths
├── models/
│   ├── __init__.py
│   ├── coordgate.py         # CoordGate, CoordGateNAFBlock, polar coord builder
│   ├── cross_attention_router.py  # CrossAttentionRouter, Fourier coord encoding
│   ├── degradation_simulator.py   # OpticalDegradationNetwork (Stage 2)
│   ├── lens_table_encoder.py      # CircularConv2d, LensTableEncoder
│   ├── losses.py            # CharbonnierLoss, MSSSIMLoss, VGGPerceptualLoss
│   ├── nafblock.py          # LayerNorm2d, SimpleGate, NAFBlock
│   ├── prior_estimator.py   # BlindPriorEstimator (Stage 1)
│   ├── restoration_backbone.py    # CoordGateNAFNetRestoration (Stage 3)
│   └── swin_block.py        # WindowAttention, SwinTransformerBlock, RSTB
├── scripts/
│   ├── __init__.py
│   └── check_omnilens_integrity.py  # Data integrity checker
├── utils/
│   ├── __init__.py
│   ├── checkpoint_sanitizer.py  # Legacy key removal from checkpoints
│   ├── coord_utils.py           # Polar coordinate map computation
│   ├── evaluation_datasets.py   # Test-time dataset loaders
│   ├── metrics.py               # PSNR, SSIM, LPIPS, Stage 1/2 evaluators
│   ├── model_builder.py         # Centralised model/trainer/dataloader factory
│   ├── omnilens_dataset.py      # MixLibDataset, LensGroupedBatchSampler
│   └── visualize.py             # Lens-table, attention, and restoration plots
├── result/                  # Experimental results (CSV + JSON)
├── train.py                 # Three-stage training entry point
├── trainer.py               # ThreeStageTrainer implementation
├── test.py                  # Inference and evaluation entry point
└── requirements.txt
```

---

### Installation

```bash
# Clone the repository
git clone https://github.com/RUA1027/LensOracle.git
cd LensOracle

# Install dependencies (Python 3.9+ recommended)
pip install -r requirements.txt
```

**Core dependencies:** `torch`, `torchvision`, `numpy`, `pyyaml`, `pillow`, `tqdm`, `matplotlib`, `tensorboard`, `lpips`, `thop`, `tabulate`, `pytest`.

---

### Data Preparation

LensOracle is designed for the **BlindCAC / OmniLens++ MixLib** dataset. Download the dataset from the [official BlindCAC repository](https://github.com/zju-jiangqi/BlindCAC) (Hugging Face link provided there) and organise it as follows:

```
<data_root>/
├── OmniLens++/
│   ├── split_MixLib/
│   │   └── MixLib_32401l40i/
│   │       └── hybrid/
│   │           ├── ab/        ← blurry images (1920×1280)
│   │           ├── gt/        ← sharp ground-truth images
│   │           └── label/     ← per-image lens profile .txt files
│   └── AODLibpro_lens/
│       └── psf_sfr/           ← PSF-SFR tensors, shape [64, 48, 67]
```

Update the paths in `config/default.yaml` under the `omnilens2` section to match your local layout.

**Verify data integrity before training:**

```bash
python scripts/check_omnilens_integrity.py \
  --ab-dir    /path/to/hybrid/ab \
  --gt-dir    /path/to/hybrid/gt \
  --label-dir /path/to/hybrid/label \
  --psf-sfr-dir /path/to/psf_sfr \
  --output    integrity_report.json \
  --verify-images \
  --verify-psf-sfr
```

---

### Training

#### Full three-stage training (Linux / server)

```bash
python train.py --config config/default.yaml --stage all
```

#### Train a single stage

```bash
# Stage 1: BlindPriorEstimator only
python train.py --config config/default.yaml --stage 1

# Stage 2: ODN only (Stage 1 frozen)
python train.py --config config/default.yaml --stage 2

# Stage 3: LensTableEncoder + Restoration (Stages 1 and 2 frozen)
python train.py --config config/default.yaml --stage 3
```

#### Override config values at runtime

```bash
python train.py --config config/default.yaml --stage 1 \
  --override training.stage_schedule.stage1_iterations=50000 \
  --override data.num_workers=4
```

#### Windows (local development)

```powershell
D:\anaconda\python.exe train.py --config config/default.yaml --stage 1
```

**Default iteration budget** (from `config/default.yaml`):

| Stage | Iterations | Batch Size |
|-------|-----------|------------|
| Stage 1 | 200,000 | 16 |
| Stage 2 | 200,000 | 8  |
| Stage 3 | 400,000 | 8  |

---

### Evaluation / Testing

```bash
python test.py \
  --checkpoint /path/to/checkpoint.pt \
  --config config/default.yaml \
  --dataset-type omnilens_mixlib
```

**Available dataset types:**

| `--dataset-type` | Description | Has GT |
|-----------------|-------------|--------|
| `omnilens_mixlib` | OmniLens++ MixLib (lens-level split) | ✓ |
| `dpdd` / `dpdd_canon` | DPDD Canon test set | ✓ |
| `dpdd_pixel` | DPDD Pixel test set | ✓ |
| `realdof` | RealDOF dataset | ✓ |
| `extreme` | Extreme defocus set | ✓ |
| `cuhk` | CUHK (no GT, inference only) | ✗ |

**Optional flags:**

```bash
--save-images       # Save blur / restored / GT side-by-side comparisons
--save-restored     # Save restored images individually
--export-visuals    # Export lens-table plots and attention maps
--visual-limit 8    # Maximum number of visual exports
```

Results are written to `<output_dir>/test_results.json` and `test_results.csv`.

---

### Configuration

All hyperparameters live in `config/default.yaml`. The configuration is loaded into strongly-typed Python dataclasses (see `config/__init__.py`), so typos raise errors at startup rather than silently failing during training.

Key configuration sections:

| Section | Purpose |
|---------|---------|
| `prior_estimator` | Stage 1 encoder depth and latent dimension |
| `lens_table_encoder` | Stage 3 encoder channels and block counts |
| `cross_attention` | Shared attention heads and Fourier frequencies |
| `odn` | Stage 2 ODN capacity and loss weight |
| `lens_split` | Lens-level train / val / test manifest |
| `training` | Optimiser, scheduler, AMP, gradient clipping |
| `restoration.losses` | Charbonnier / perceptual / MS-SSIM weights |
| `checkpoint` | Per-stage best-model metric selection |

---

### Outputs and Checkpoints

Training writes to `experiment.output_dir / experiment.name / run_name_timestamp/`.

**Checkpoint files** are saved to a fixed path `experiment.output_dir / experiment.name / checkpoints/`:

| Filename | Contents |
|----------|---------|
| `latest.pt` | Most recent overall checkpoint |
| `latest_stage{1,2,3}.pt` | Most recent per-stage checkpoint |
| `best.pt` | Best overall checkpoint |
| `best_stage{1,2,3}.pt` | Best per-stage checkpoint |
| `final_model.pt` | End-of-training checkpoint |
| `best_performance.json` | Per-stage best metric tracking |

---

### Key Metrics

**Stage 1:** `Val_PSF_SFR_L1`, `Val_PSF_SFR_MSSSIM`, `Val_LensIdentifiability`, `Val_BatchStd`

**Stage 2:** `Val_ODN_L1`, `Val_ODN_PSNR`

**Stage 3 / Test:** `PSNR`, `SSIM`, `MAE`, `LPIPS`

---

### Data Integrity Check

```bash
python scripts/check_omnilens_integrity.py \
  --ab-dir    /path/to/ab \
  --gt-dir    /path/to/gt \
  --label-dir /path/to/label \
  --psf-sfr-dir /path/to/psf_sfr \
  --output    integrity_report.json \
  --verify-images \
  --verify-psf-sfr
```

---

### Tests

```bash
python -m pytest tests -q
```

On Windows, if PyTorch import fails with `WinError 10106`, run configuration and contract tests first; full model smoke tests require a working CUDA environment.

---

### Acknowledgements

- **BlindCAC / OmniLens++** dataset and challenge: [ZJU Jiangqi Group](https://github.com/zju-jiangqi/BlindCAC)
- Restoration backbone inspired by [NAFNet](https://github.com/megvii-research/NAFNet)
- Transformer blocks adapted from [SwinIR](https://github.com/JingyunLiang/SwinIR)

---

---

# LensOracle 中文

LensOracle 是一个用于**镜头表格引导的离焦图像复原**的离线深度学习研究框架。与依赖像素级模糊核的传统方案不同，本项目直接以极坐标格式的原生 PSF-SFR 镜头表格（形状为 `[64, 48, 67]` 的标定张量）作为物理先验，通过多尺度跨注意力机制注入图像复原网络，实现具有镜头感知能力的离焦复原。

数据集采用 [BlindCAC / OmniLens++](https://github.com/zju-jiangqi/BlindCAC)（NTIRE 2026 挑战赛数据集），提供 3,240 个不同镜头的配对离焦图像与 PSF-SFR 物理标定张量。

> **说明：** 本仓库仅包含离线训练与评估流程，不涉及任何硬件通信、相机控制或在线闭环推理模块。

---

### 架构设计

当前流程为三阶段原生镜头表格融合 pipeline：

```
模糊图像
  → 阶段一：BlindPriorEstimator（盲先验估计器）
  → pred_psf_sfr  [B, 64, 48, 67]
  → 阶段二：OpticalDegradationNetwork（光学退化网络，物理闭环验证）
  → 阶段三：LensTableEncoder + CoordGateNAFNetRestoration
  → 复原图像
```

**阶段一 — BlindPriorEstimator：** 从单张模糊图像中预测原生 PSF-SFR 镜头表格，直接以标定的 `gt_psf_sfr` 张量为监督目标，不使用像素展开的 PSF 图。

**阶段二 — OpticalDegradationNetwork（ODN）：** 以清晰图像和预测的镜头表格为输入，模拟对应的模糊图像，在前向建模闭环中验证预测先验的物理自洽性。阶段二在训练结束后冻结，推理时不参与计算。

**阶段三 — LensTableEncoder + 复原网络：** 编码镜头表格的多尺度物理特征，通过坐标感知的跨注意力注入复原主干。核心设计选择如下：

- **CircularConv2d** — 在镜头表格的角度（θ）维使用循环 padding，在径向（r）维使用零 padding，尊重方位角的周期性边界同时保持径向的非周期性。
- **CrossAttentionRouter** — 利用 Fourier 编码的极坐标 `(r, θ)` 将镜头表格 token 路由到图像特征 token，实现基于物理坐标的空间条件化。
- **CoordGateNAFBlock** — NAFBlock 的变体，深度卷积分支通过坐标导出的空间门控进行调制，使每个特征变换对图像位置具备显式感知能力。

---

### 仓库结构

```
LensOracle/
├── config/
│   ├── __init__.py          # 强类型配置 dataclass + YAML 加载器
│   └── default.yaml         # 默认超参数与数据路径
├── models/
│   ├── __init__.py
│   ├── coordgate.py         # CoordGate、CoordGateNAFBlock、极坐标生成器
│   ├── cross_attention_router.py  # CrossAttentionRouter、Fourier 坐标编码
│   ├── degradation_simulator.py   # OpticalDegradationNetwork（阶段二）
│   ├── lens_table_encoder.py      # CircularConv2d、LensTableEncoder
│   ├── losses.py            # CharbonnierLoss、MSSSIMLoss、VGGPerceptualLoss
│   ├── nafblock.py          # LayerNorm2d、SimpleGate、NAFBlock
│   ├── prior_estimator.py   # BlindPriorEstimator（阶段一）
│   ├── restoration_backbone.py    # CoordGateNAFNetRestoration（阶段三）
│   └── swin_block.py        # WindowAttention、SwinTransformerBlock、RSTB
├── scripts/
│   ├── __init__.py
│   └── check_omnilens_integrity.py  # 数据完整性核验脚本
├── utils/
│   ├── __init__.py
│   ├── checkpoint_sanitizer.py  # checkpoint 历史键清理工具
│   ├── coord_utils.py           # 极坐标图计算工具
│   ├── evaluation_datasets.py   # 测试阶段数据集加载器
│   ├── metrics.py               # PSNR/SSIM/LPIPS 及各阶段评估器
│   ├── model_builder.py         # 模型/Trainer/DataLoader 集中构建入口
│   ├── omnilens_dataset.py      # MixLibDataset、LensGroupedBatchSampler
│   └── visualize.py             # 镜头表格、注意力权重、复原结果可视化
├── result/                  # 实验结果（CSV + JSON）
├── train.py                 # 三阶段训练入口
├── trainer.py               # ThreeStageTrainer 实现
├── test.py                  # 推理与评估入口
└── requirements.txt
```

---

### 安装

```bash
# 克隆仓库
git clone https://github.com/RUA1027/LensOracle.git
cd LensOracle

# 安装依赖（推荐 Python 3.9+）
pip install -r requirements.txt
```

**主要依赖：** `torch`、`torchvision`、`numpy`、`pyyaml`、`pillow`、`tqdm`、`matplotlib`、`tensorboard`、`lpips`、`thop`、`tabulate`、`pytest`。

---

### 数据准备

本仓库为 **BlindCAC / OmniLens++ MixLib** 数据集设计。请从 [BlindCAC 官方仓库](https://github.com/zju-jiangqi/BlindCAC) 下载数据集，并按以下结构组织：

```
<数据根目录>/
├── OmniLens++/
│   ├── split_MixLib/
│   │   └── MixLib_32401l40i/
│   │       └── hybrid/
│   │           ├── ab/        ← 模糊图像（1920×1280）
│   │           ├── gt/        ← 对应清晰图像
│   │           └── label/     ← 每张图的镜头 profile .txt 文件
│   └── AODLibpro_lens/
│       └── psf_sfr/           ← PSF-SFR 张量，形状 [64, 48, 67]
```

下载完成后，在 `config/default.yaml` 的 `omnilens2` 节下更新对应的本地路径。

**训练前请先核验数据完整性：**

```bash
python scripts/check_omnilens_integrity.py \
  --ab-dir    /path/to/hybrid/ab \
  --gt-dir    /path/to/hybrid/gt \
  --label-dir /path/to/hybrid/label \
  --psf-sfr-dir /path/to/psf_sfr \
  --output    integrity_report.json \
  --verify-images \
  --verify-psf-sfr
```

---

### 训练

#### 完整三阶段训练（Linux / 服务器）

```bash
python train.py --config config/default.yaml --stage all
```

#### 单阶段训练

```bash
# 阶段一：仅训练 BlindPriorEstimator
python train.py --config config/default.yaml --stage 1

# 阶段二：冻结阶段一，仅训练 ODN
python train.py --config config/default.yaml --stage 2

# 阶段三：冻结阶段一/二，仅训练 LensTableEncoder + Restoration
python train.py --config config/default.yaml --stage 3
```

#### 运行时覆盖配置项

```bash
python train.py --config config/default.yaml --stage 1 \
  --override training.stage_schedule.stage1_iterations=50000 \
  --override data.num_workers=4
```

#### Windows 本机开发

```powershell
D:\anaconda\python.exe train.py --config config/default.yaml --stage 1
```

**默认迭代预算**（来自 `config/default.yaml`）：

| 阶段 | 迭代次数 | Batch Size |
|------|---------|------------|
| 阶段一 | 200,000 | 16 |
| 阶段二 | 200,000 | 8  |
| 阶段三 | 400,000 | 8  |

---

### 测试与评估

```bash
python test.py \
  --checkpoint /path/to/checkpoint.pt \
  --config config/default.yaml \
  --dataset-type omnilens_mixlib
```

**可用数据集类型：**

| `--dataset-type` | 说明 | 有 GT |
|-----------------|------|-------|
| `omnilens_mixlib` | OmniLens++ MixLib（镜头级划分） | ✓ |
| `dpdd` / `dpdd_canon` | DPDD Canon 测试集 | ✓ |
| `dpdd_pixel` | DPDD Pixel 测试集 | ✓ |
| `realdof` | RealDOF 数据集 | ✓ |
| `extreme` | 极端离焦测试集 | ✓ |
| `cuhk` | CUHK（无 GT，仅推理） | ✗ |

**可选参数：**

```bash
--save-images       # 保存 blur / 复原 / GT 三联对比图
--save-restored     # 单独保存复原图像
--export-visuals    # 导出镜头表格图与注意力权重图
--visual-limit 8    # 最多导出 8 个可视化样本
```

测试结果写入 `<output_dir>/test_results.json` 与 `test_results.csv`。

---

### 配置系统说明

所有超参数集中于 `config/default.yaml`，加载时转换为强类型 Python dataclass（见 `config/__init__.py`），字段错误在启动时即可捕获。

**主要配置节：**

| 配置节 | 作用 |
|-------|------|
| `prior_estimator` | 阶段一编码器深度与隐向量维度 |
| `lens_table_encoder` | 阶段三编码器通道数与残差块数量 |
| `cross_attention` | 跨注意力头数与 Fourier 频率数 |
| `odn` | 阶段二 ODN 容量与损失权重 |
| `lens_split` | 镜头级 train/val/test 划分清单 |
| `training` | 优化器、调度器、AMP、梯度裁剪 |
| `restoration.losses` | Charbonnier / 感知 / MS-SSIM 损失权重 |
| `checkpoint` | 各阶段 best 模型的判定指标 |

---

### 输出与 Checkpoint 管理

训练输出写入 `experiment.output_dir / experiment.name / run_name_timestamp/`。

**固定路径 checkpoint** 保存于 `experiment.output_dir / experiment.name / checkpoints/`：

| 文件名 | 内容 |
|--------|------|
| `latest.pt` | 最新全局 checkpoint |
| `latest_stage{1,2,3}.pt` | 最新各阶段 checkpoint |
| `best.pt` | 全局最优 checkpoint |
| `best_stage{1,2,3}.pt` | 各阶段最优 checkpoint |
| `final_model.pt` | 训练结束 checkpoint |
| `best_performance.json` | 各阶段最优指标跟踪记录 |

---

### 评估指标

**阶段一：** `Val_PSF_SFR_L1`、`Val_PSF_SFR_MSSSIM`、`Val_LensIdentifiability`、`Val_BatchStd`

**阶段二：** `Val_ODN_L1`、`Val_ODN_PSNR`

**阶段三 / 测试：** `PSNR`、`SSIM`、`MAE`、`LPIPS`

---

### 单元测试

```bash
python -m pytest tests -q
```

Windows 环境下若 PyTorch 导入触发 `WinError 10106`，可先只运行不依赖 PyTorch 的静态/配置测试；完整模型测试请在 CUDA 环境正常的训练服务器上执行。

---

### 致谢

- **BlindCAC / OmniLens++** 数据集与挑战赛：[浙大江琦课题组](https://github.com/zju-jiangqi/BlindCAC)
- 复原主干设计参考 [NAFNet](https://github.com/megvii-research/NAFNet)
- Transformer 积木改编自 [SwinIR](https://github.com/JingyunLiang/SwinIR)
