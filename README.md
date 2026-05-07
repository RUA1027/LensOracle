# LensOracle

**English** | [中文](#lensoracle-中文)

---

## LensOracle (English)

LensOracle is an offline deep learning research framework for **lens-table-guided defocus image restoration**. It conditions a restoration network directly on native PSF-SFR lens tables — calibrated optical characterizations stored as `[64, 48, 67]` tensors in polar coordinates — read from the [BlindCAC / OmniLens++](https://github.com/zju-jiangqi/BlindCAC) dataset (NTIRE 2026), which provides paired defocus images and ground-truth PSF-SFR lens tables across 3,240 distinct lenses.

> **Note:** This repository contains the offline training and evaluation pipeline only. It does not include any hardware communication, camera control, or online closed-loop inference modules.

---

### Architecture

```
Blur Image + GT PSF-SFR [B, 64, 48, 67] (from BlindCAC dataset)
  → LensTableEncoder → {F1, F2, F3}
  → CrossAttentionRouter (injects lens features at 3 scales)
  → CoordGateNAFNetRestoration → Restored Image
```

The pipeline directly reads the ground-truth PSF-SFR lens table from the dataset and encodes it into multi-scale physical features via the `LensTableEncoder`. These features are injected into the restoration backbone at multiple resolutions through `CrossAttentionRouter` modules that use Fourier-encoded polar coordinates `(r, θ)` for physically grounded spatial conditioning.

Key design decisions:

- **CircularConv2d** — applies circular padding along the angular (θ) dimension and zero-padding along the radial (r) dimension of the lens table, respecting the periodic boundary of azimuthal angle while preserving the non-periodic nature of the radial field.
- **CrossAttentionRouter** — routes lens-table tokens to image feature tokens using Fourier-encoded polar coordinates, enabling physically grounded spatial conditioning.
- **CoordGateNAFBlock** — a NAFBlock variant whose depthwise convolution branch is gated by a coordinate-derived spatial mask, making each feature transformation position-aware.

---

### Repository Structure

```
LensOracle/
├── config/
│   ├── __init__.py              # Strongly-typed config dataclasses + YAML loader
│   ├── default.yaml             # Default hyperparameters and data paths
│   └── ablations/               # Ablation experiment configs
├── models/
│   ├── __init__.py
│   ├── coordgate.py             # CoordGate, CoordGateNAFBlock, polar coord builder
│   ├── cross_attention_router.py  # CrossAttentionRouter, Fourier coord encoding
│   ├── lens_table_encoder.py    # CircularConv2d, LensTableEncoder
│   ├── losses.py                # CharbonnierLoss, MSSSIMLoss, VGGPerceptualLoss
│   ├── nafblock.py              # LayerNorm2d, SimpleGate, NAFBlock
│   ├── restoration_backbone.py  # CoordGateNAFNetRestoration
│   └── swin_block.py            # WindowAttention, SwinTransformerBlock, RSTB
├── scripts/
│   ├── __init__.py
│   ├── check_omnilens_integrity.py    # Data integrity checker
│   └── evaluate_ablation_suite.py     # Ablation suite evaluator
├── utils/
│   ├── __init__.py
│   ├── checkpoint_sanitizer.py  # Legacy key removal from checkpoints
│   ├── coord_utils.py           # Polar coordinate map computation
│   ├── evaluation_datasets.py   # Test-time dataset loaders
│   ├── metrics.py               # PSNR, SSIM, LPIPS evaluators
│   ├── model_builder.py         # Centralised model/trainer/dataloader factory
│   ├── omnilens_dataset.py      # MixLibDataset
│   └── visualize.py             # Lens-table, attention, and restoration plots
├── train.py                     # Training entry point
├── trainer.py                   # LensOracleTrainer implementation
├── test.py                      # Inference and evaluation entry point
└── requirements.txt
```

---

### Installation

```bash
git clone https://github.com/RUA1027/LensOracle.git
cd LensOracle
pip install -r requirements.txt
```

**Core dependencies:** `torch`, `torchvision`, `numpy`, `pyyaml`, `pillow`, `tqdm`, `matplotlib`, `tensorboard`, `lpips`, `thop`, `tabulate`, `pytest`.

---

### Data Preparation

Download the BlindCAC / OmniLens++ MixLib dataset from the [official repository](https://github.com/zju-jiangqi/BlindCAC) and organise it as:

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

Update paths in `config/default.yaml` under the `omnilens2` section.

**Verify data integrity:**

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

```bash
python train.py --config config/default.yaml
```

Override config values at runtime:

```bash
python train.py --config config/default.yaml \
  --override training.stage_schedule.stage3_iterations=100000 \
  --override data.num_workers=4
```

Resume from checkpoint:

```bash
python train.py --config config/default.yaml --resume /path/to/checkpoint.pt
```

---

### Evaluation

```bash
python test.py \
  --checkpoint /path/to/checkpoint.pt \
  --config config/default.yaml \
  --dataset-type omnilens_mixlib
```

**Available dataset types:**

| `--dataset-type`        | Description                          | Has GT |
| ------------------------- | ------------------------------------ | ------ |
| `omnilens_mixlib`       | OmniLens++ MixLib (lens-level split) | ✓     |
| `dpdd` / `dpdd_canon` | DPDD Canon test set                  | ✓     |
| `dpdd_pixel`            | DPDD Pixel test set                  | ✓     |
| `realdof`               | RealDOF dataset                      | ✓     |
| `extreme`               | Extreme defocus set                  | ✓     |
| `cuhk`                  | CUHK (no GT, inference only)         | ✗     |

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

All hyperparameters live in `config/default.yaml`, loaded into strongly-typed Python dataclasses (`config/__init__.py`).

Key configuration sections:

| Section                | Purpose                                      |
| ---------------------- | -------------------------------------------- |
| `lens_table_encoder` | Encoder channels and block counts            |
| `cross_attention`    | Attention heads and Fourier frequencies      |
| `restoration`        | Restoration backbone and loss weights        |
| `lens_split`         | Lens-level train / val / test manifest       |
| `training`           | Optimiser, scheduler, AMP, gradient clipping |
| `checkpoint`         | Best-model metric selection                  |

---

### Checkpoints

Training writes to `experiment.output_dir / experiment.name / run_name_timestamp/`. Fixed-path checkpoints are saved to `experiment.output_dir / experiment.name / checkpoints/`:

| Filename                  | Contents                    |
| ------------------------- | --------------------------- |
| `latest.pt`             | Most recent checkpoint      |
| `best.pt`               | Best checkpoint (by PSNR)   |
| `final_model.pt`        | End-of-training checkpoint  |
| `best_performance.json` | Best metric tracking record |

---

### Key Metrics

**Restoration:** `PSNR`, `SSIM`, `MAE`, `LPIPS`

---

### Ablation Studies

Ablation configs in `config/ablations/`:

| Config                        | Purpose                                                                    |
| ----------------------------- | -------------------------------------------------------------------------- |
| `restoration_only.yaml`     | Baseline without lens-table conditioning (`lens_encoder_enabled: false`) |
| `zero_padding_encoder.yaml` | Standard zero-padding instead of circular padding                          |

Evaluate an ablation suite:

```bash
python scripts/evaluate_ablation_suite.py --suite path/to/suite.yaml
```

---

### Tests

```bash
python -m pytest tests -q
```

---

### Acknowledgements

- **BlindCAC / OmniLens++** dataset: [ZJU Jiangqi Group](https://github.com/zju-jiangqi/BlindCAC)
- Restoration backbone inspired by [NAFNet](https://github.com/megvii-research/NAFNet)
- Transformer blocks adapted from [SwinIR](https://github.com/JingyunLiang/SwinIR)

---

---

# LensOracle 中文

LensOracle 是一个用于**镜头表格引导的离焦图像复原**的离线深度学习研究框架。本项目直接从 [BlindCAC / OmniLens++](https://github.com/zju-jiangqi/BlindCAC) 数据集读取标定的原生 PSF-SFR 镜头表格（形状为 `[64, 48, 67]` 的张量），经 `LensTableEncoder` 编码后通过坐标感知的 `CrossAttentionRouter` 注入 `CoordGateNAFNetRestoration` 复原网络，实现具有镜头感知能力的离焦复原。

> **说明：** 本仓库仅包含离线训练与评估流程，不涉及任何硬件通信、相机控制或在线闭环推理模块。

---

### 架构设计

```
模糊图像 + GT PSF-SFR [B, 64, 48, 67]（来自 BlindCAC 数据集）
  → LensTableEncoder → {F1, F2, F3}
  → CrossAttentionRouter（在 3 个尺度注入镜头特征）
  → CoordGateNAFNetRestoration → 复原图像
```

核心设计：

- **CircularConv2d** — 在镜头表格的角度（θ）维使用循环 padding，在径向（r）维使用零 padding，尊重方位角的周期性边界。
- **CrossAttentionRouter** — 利用 Fourier 编码的极坐标 `(r, θ)` 将镜头表格 token 路由到图像特征 token，实现基于物理坐标的空间条件化。
- **CoordGateNAFBlock** — NAFBlock 的变体，深度卷积分支通过坐标导出的空间门控进行调制。

---

### 仓库结构

```
LensOracle/
├── config/
│   ├── __init__.py              # 强类型配置 dataclass + YAML 加载器
│   ├── default.yaml             # 默认超参数与数据路径
│   └── ablations/               # 消融实验配置
├── models/
│   ├── __init__.py
│   ├── coordgate.py             # CoordGate、CoordGateNAFBlock、极坐标生成器
│   ├── cross_attention_router.py  # CrossAttentionRouter、Fourier 坐标编码
│   ├── lens_table_encoder.py    # CircularConv2d、LensTableEncoder
│   ├── losses.py                # CharbonnierLoss、MSSSIMLoss、VGGPerceptualLoss
│   ├── nafblock.py              # LayerNorm2d、SimpleGate、NAFBlock
│   ├── restoration_backbone.py  # CoordGateNAFNetRestoration
│   └── swin_block.py            # WindowAttention、SwinTransformerBlock、RSTB
├── scripts/
│   ├── __init__.py
│   ├── check_omnilens_integrity.py    # 数据完整性核验脚本
│   └── evaluate_ablation_suite.py     # 消融套件评估脚本
├── utils/
│   ├── __init__.py
│   ├── checkpoint_sanitizer.py  # checkpoint 历史键清理工具
│   ├── coord_utils.py           # 极坐标图计算工具
│   ├── evaluation_datasets.py   # 测试阶段数据集加载器
│   ├── metrics.py               # PSNR/SSIM/LPIPS 评估器
│   ├── model_builder.py         # 模型/Trainer/DataLoader 集中构建入口
│   ├── omnilens_dataset.py      # MixLibDataset
│   └── visualize.py             # 镜头表格、注意力权重、复原结果可视化
├── train.py                     # 训练入口
├── trainer.py                   # LensOracleTrainer 实现
├── test.py                      # 推理与评估入口
└── requirements.txt
```

---

### 安装

```bash
git clone https://github.com/RUA1027/LensOracle.git
cd LensOracle
pip install -r requirements.txt
```

---

### 训练

```bash
python train.py --config config/default.yaml
```

训练输出写入 `experiment.output_dir / experiment.name / run_name_timestamp/`。

---

### 测试与评估

```bash
python test.py \
  --checkpoint /path/to/checkpoint.pt \
  --config config/default.yaml \
  --dataset-type omnilens_mixlib
```

测试结果写入 `<output_dir>/test_results.json` 与 `test_results.csv`。

---

### 评估指标

**复原质量：** `PSNR`、`SSIM`、`MAE`、`LPIPS`

---

### 消融实验

`config/ablations/` 目录下的消融配置：

| 配置                          | 用途                                                      |
| ----------------------------- | --------------------------------------------------------- |
| `restoration_only.yaml`     | 无镜头表格条件化的基线（`lens_encoder_enabled: false`） |
| `zero_padding_encoder.yaml` | 使用标准零 padding 替代循环 padding                       |

---

### 致谢

- **BlindCAC / OmniLens++** 数据集与挑战赛：[浙大江琦课题组](https://github.com/zju-jiangqi/BlindCAC)
- 复原主干设计参考 [NAFNet](https://github.com/megvii-research/NAFNet)
- Transformer 积木改编自 [SwinIR](https://github.com/JingyunLiang/SwinIR)
