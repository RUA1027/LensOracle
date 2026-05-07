"""单阶段 LensOracle 训练器。

LensOracleTrainer 联合训练 LensTableEncoder 和 Restoration 网络，
直接使用 ground-truth PSF SFR 表格作为先验信息。所有 loss 在单阶段内
统一计算，支持 Charbonnier、perceptual 和 MS-SSIM 损失的可选组合。
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from utils.checkpoint_sanitizer import sanitize_legacy_checkpoint


def lens_table_tv_loss(table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 lens-table 在 r 和 theta 维的 TV loss。

    table 形状为 `[B, 64, 48, 67]`。r 方向是普通相邻差分；theta 方向是
    circular 差分，最后一个角度会和第一个角度相连。
    """

    tv_r = torch.abs(table[:, 1:, :, :] - table[:, :-1, :, :]).mean()
    tv_theta = torch.abs(torch.roll(table, shifts=-1, dims=2) - table).mean()
    return tv_r + tv_theta, tv_r, tv_theta


class LensOracleTrainer:
    """单阶段训练器。

    职责边界：
    1) 管理 lens_table_encoder + restoration 的可训练性；
    2) 管理两路优化器/调度器与梯度累积；
    3) 管理 AMP、梯度裁剪与 non-finite 防护；
    4) 维护 checkpoint 与 tensorboard 相关状态。
    """

    def __init__(
        self,
        lens_table_encoder: nn.Module,
        restoration_net: nn.Module,
        lr_lens_encoder: float,
        lr_restoration: float,
        optimizer_type: str,
        weight_decay: float,
        grad_clip_lens_encoder: float,
        grad_clip_restoration: float,
        total_iterations: int,
        use_amp: bool,
        amp_dtype: str,
        accumulation_steps: int,
        device: str,
        tensorboard_dir: Optional[str],
        perceptual_weight: float = 0.0,
        perceptual_warmup_iterations: int = 0,
        perceptual_enabled: bool = True,
        perceptual_loss: Optional[nn.Module] = None,
        perceptual_loss_builder: Optional[Callable[[], nn.Module]] = None,
        charbonnier_loss: Optional[nn.Module] = None,
        charbonnier_enabled: bool = True,
        ms_ssim_loss: Optional[nn.Module] = None,
        ms_ssim_loss_builder: Optional[Callable[[], nn.Module]] = None,
        ms_ssim_weight: float = 0.0,
        ms_ssim_enabled: bool = False,
        tv_weight: float = 0.01,
        lens_encoder_enabled: bool = True,
        ablation: Optional[Dict[str, Any]] = None,
        nonfinite_patience: int = 3,
        nonfinite_backoff_factor: float = 0.5,
        nonfinite_min_lr: float = 1.0e-6,
    ):
        self.lens_table_encoder = lens_table_encoder
        self.restoration_net = restoration_net
        self.device = device
        self.total_iterations = total_iterations
        self.accumulation_steps = max(1, int(accumulation_steps))

        self.grad_clip_lens_encoder = float(grad_clip_lens_encoder)
        self.grad_clip_restoration = float(grad_clip_restoration)
        self.tv_weight = float(tv_weight)
        self.lens_encoder_enabled = bool(lens_encoder_enabled)
        self.ablation = dict(ablation or {})
        self.perceptual_weight = float(perceptual_weight)
        self.perceptual_warmup_iterations = max(0, int(perceptual_warmup_iterations))
        self.perceptual_enabled = bool(perceptual_enabled)
        self.perceptual_loss = perceptual_loss
        self.perceptual_loss_builder = perceptual_loss_builder
        self.charbonnier_loss = charbonnier_loss
        self.charbonnier_enabled = bool(charbonnier_enabled)
        self.ms_ssim_loss = ms_ssim_loss
        self.ms_ssim_loss_builder = ms_ssim_loss_builder
        self.ms_ssim_weight = float(ms_ssim_weight)
        self.ms_ssim_enabled = bool(ms_ssim_enabled)
        self.nonfinite_patience = max(1, int(nonfinite_patience))
        self.nonfinite_backoff_factor = float(nonfinite_backoff_factor)
        self.nonfinite_min_lr = float(nonfinite_min_lr)

        self.use_amp = bool(use_amp and device.startswith("cuda") and torch.cuda.is_available())
        self.amp_dtype = torch.float16 if str(amp_dtype).lower() == "float16" else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.optimizer_lens_encoder = (
            self._build_optimizer(
                self.lens_table_encoder.parameters(),
                optimizer_type,
                lr_lens_encoder,
                weight_decay,
            )
            if self.lens_encoder_enabled
            else None
        )
        self.optimizer_restoration = self._build_optimizer(
            self.restoration_net.parameters(), optimizer_type, lr_restoration, weight_decay
        )
        self.scheduler_lens_encoder = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_lens_encoder,
                T_max=max(1, self.total_iterations),
                eta_min=self.nonfinite_min_lr,
            )
            if self.optimizer_lens_encoder is not None
            else None
        )
        self.scheduler_restoration = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_restoration,
                T_max=max(1, self.total_iterations),
                eta_min=self.nonfinite_min_lr,
            )
            if self.optimizer_restoration is not None
            else None
        )

        # ---- 固定可训练性：lens_encoder（若启用）+ restoration 始终可训练 ----
        self._set_requires_grad(self.lens_table_encoder, self.lens_encoder_enabled)
        self._set_requires_grad(self.restoration_net, True)
        self.lens_table_encoder.train(self.lens_encoder_enabled)
        self.restoration_net.train()

        self.writer = None
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard.writer import SummaryWriter

                self.writer = SummaryWriter(log_dir=tensorboard_dir)
            except Exception:
                self.writer = None

        self.best_metrics: Dict[str, float] = {
            "psnr": float("-inf"),
            "mae": float("inf"),
        }
        self.lens_encoder_optimizer_steps = 0
        self.restoration_optimizer_steps = 0
        self._accum_step = 0
        self._last_step_nonfinite = False
        self.nonfinite_streak = 0
        self.lr_backoff_events = 0

    @property
    def pending_accumulation_steps(self) -> int:
        """当前尚未 flush 的累积步数。"""

        return int(self._accum_step)

    @staticmethod
    def _build_optimizer(params, optimizer_type: str, lr: float, weight_decay: float):
        """按配置创建优化器（Adam 或 AdamW）。"""

        params = list(params)
        if not params:
            return None
        if str(optimizer_type).lower() == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        """批量设置模块参数的 requires_grad。"""

        for param in module.parameters():
            param.requires_grad = enabled

    def _zero_all_grad(self) -> None:
        """清空两路优化器梯度。"""

        for optimizer in (
            self.optimizer_lens_encoder,
            self.optimizer_restoration,
        ):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

    def _maybe_zero_grad(self) -> None:
        """仅在累积起点清梯度，兼容 gradient accumulation。"""

        if self._accum_step == 0:
            self._zero_all_grad()

    def _current_perceptual_weight(self) -> float:
        """计算当前 step 的感知损失权重（含 warmup）。"""

        if not self.perceptual_enabled or self.perceptual_weight <= 0.0:
            return 0.0
        if self.perceptual_warmup_iterations <= 0:
            return self.perceptual_weight
        progress = min(1.0, self.restoration_optimizer_steps / float(self.perceptual_warmup_iterations))
        return self.perceptual_weight * progress

    def _get_perceptual_loss(self) -> Optional[nn.Module]:
        """延迟构建 perceptual loss（首次需要时实例化）。"""

        if not self.perceptual_enabled or self.perceptual_weight <= 0.0:
            return None
        if self.perceptual_loss is not None:
            return self.perceptual_loss
        if self.perceptual_loss_builder is None:
            return None
        self.perceptual_loss = self.perceptual_loss_builder()
        self.perceptual_loss_builder = None
        return self.perceptual_loss

    def _get_ms_ssim_loss(self) -> Optional[nn.Module]:
        """延迟构建 MS-SSIM loss。"""

        if not self.ms_ssim_enabled or self.ms_ssim_weight <= 0.0:
            return None
        if self.ms_ssim_loss is not None:
            return self.ms_ssim_loss
        if self.ms_ssim_loss_builder is None:
            return None
        self.ms_ssim_loss = self.ms_ssim_loss_builder()
        self.ms_ssim_loss_builder = None
        return self.ms_ssim_loss

    @staticmethod
    def _is_finite_scalar(value: Any) -> bool:
        """判断标量或标量张量是否有限。"""

        if torch.is_tensor(value):
            return bool(value.numel() > 0 and torch.isfinite(value).all().item())
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _compute_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """单阶段损失：restoration 主损失 + 可选 perceptual/MS-SSIM。"""

        blur = batch["blur"].to(self.device, non_blocking=True)
        sharp = batch["sharp"].to(self.device, non_blocking=True)
        crop_info = batch.get("crop_info")
        if torch.is_tensor(crop_info):
            crop_info = crop_info.to(self.device, non_blocking=True)

        prior_table = batch["gt_psf_sfr"].to(self.device, non_blocking=True)
        features = self.lens_table_encoder(prior_table) if self.lens_encoder_enabled else None
        restored = self.restoration_net(blur, features, crop_info=crop_info)

        # base restoration loss（Charbonnier 优先，否则退化为 L1）。
        if self.charbonnier_enabled and self.charbonnier_loss is not None:
            restoration_loss = self.charbonnier_loss(restored, sharp)
        else:
            restoration_loss = F.l1_loss(restored, sharp)
        total = restoration_loss
        zero = total.new_zeros(())

        perc_weight = self._current_perceptual_weight()
        loss_perceptual = zero
        perceptual_module = self._get_perceptual_loss()
        if perceptual_module is not None and perc_weight > 0.0:
            loss_perceptual = perceptual_module(restored, sharp)
            total = total + perc_weight * loss_perceptual

        loss_ms_ssim = zero
        ms_ssim_module = self._get_ms_ssim_loss()
        if ms_ssim_module is not None and self.ms_ssim_weight > 0.0:
            loss_ms_ssim = ms_ssim_module(restored, sharp)
            total = total + self.ms_ssim_weight * loss_ms_ssim

        return {
            "loss": total,
            "loss_table_l1": zero,
            "loss_tv": zero,
            "loss_tv_r": zero,
            "loss_tv_theta": zero,
            "loss_table_recon": zero,
            "loss_odn": zero,
            "loss_odn_weighted": zero,
            "loss_restoration": restoration_loss,
            "loss_perceptual": loss_perceptual,
            "loss_ms_ssim": loss_ms_ssim,
            "perceptual_weight_current": torch.full_like(zero, perc_weight),
            "ms_ssim_weight_current": torch.full_like(
                zero, self.ms_ssim_weight if self.ms_ssim_enabled else 0.0
            ),
        }

    def _active_optimizers(self):
        """返回当前应参与 step 的优化器集合。"""

        active = []
        if (
            self.lens_encoder_enabled
            and self.optimizer_lens_encoder is not None
            and self.scheduler_lens_encoder is not None
        ):
            active.append(
                (
                    self.optimizer_lens_encoder,
                    self.scheduler_lens_encoder,
                    self.lens_table_encoder.parameters(),
                    self.grad_clip_lens_encoder,
                    "lens_encoder",
                )
            )
        if self.optimizer_restoration is not None and self.scheduler_restoration is not None:
            active.append(
                (
                    self.optimizer_restoration,
                    self.scheduler_restoration,
                    self.restoration_net.parameters(),
                    self.grad_clip_restoration,
                    "restoration",
                )
            )
        return active

    def _increment_step_counter(self, name: str) -> None:
        """更新各子模块优化器 step 计数器。"""

        if name == "lens_encoder":
            self.lens_encoder_optimizer_steps += 1
        elif name == "restoration":
            self.restoration_optimizer_steps += 1

    def _clip_and_step(self) -> bool:
        """执行一次“裁剪 + optimizer.step + scheduler.step”。

        返回值：
        - True：本次成功完成参数更新；
        - False：因 non-finite 等原因跳过更新。
        """

        self._last_step_nonfinite = False
        active = self._active_optimizers()
        if not active:
            return False

        if self.use_amp:
            # AMP 路径：先 unscale，再裁剪，再 step。
            for optimizer, _, _, _, _ in active:
                self.scaler.unscale_(optimizer)
            finite = True
            for _, _, params, clip_value, _ in active:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_value)
                finite = finite and self._is_finite_scalar(grad_norm)
            if not finite:
                self._last_step_nonfinite = True
                self._zero_all_grad()
                return False
            scale_before = float(self.scaler.get_scale())
            for optimizer, _, _, _, _ in active:
                self.scaler.step(optimizer)
            self.scaler.update()
            if float(self.scaler.get_scale()) < scale_before:
                self._last_step_nonfinite = True
                self._zero_all_grad()
                return False
        else:
            # 非 AMP 路径：直接裁剪 + step。
            for _, _, params, clip_value, _ in active:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_value)
                if not self._is_finite_scalar(grad_norm):
                    self._last_step_nonfinite = True
                    self._zero_all_grad()
                    return False
            for optimizer, _, _, _, _ in active:
                optimizer.step()

        for _, scheduler, _, _, name in active:
            scheduler.step()
            self._increment_step_counter(name)
        return True

    def _apply_lr_backoff(self) -> bool:
        """在连续 non-finite 后按比例衰减学习率。"""

        if self.nonfinite_backoff_factor <= 0.0 or self.nonfinite_backoff_factor >= 1.0:
            return False
        changed = False
        for optimizer, scheduler, _, _, _ in self._active_optimizers():
            for group in optimizer.param_groups:
                current = float(group.get("lr", 0.0))
                target = max(self.nonfinite_min_lr, current * self.nonfinite_backoff_factor)
                if target + 1.0e-15 < current:
                    group["lr"] = target
                    changed = True
            base_lrs = getattr(scheduler, "base_lrs", None)
            if isinstance(base_lrs, list):
                scheduler.base_lrs = [
                    max(self.nonfinite_min_lr, lr * self.nonfinite_backoff_factor) for lr in base_lrs
                ]
        if changed:
            self.lr_backoff_events += 1
        return changed

    def _handle_nonfinite_event(self) -> float:
        """处理 non-finite 事件并按 patience 决定是否触发 lr backoff。"""

        self.nonfinite_streak += 1
        if self.nonfinite_streak < self.nonfinite_patience:
            return 0.0
        self.nonfinite_streak = 0
        return 1.0 if self._apply_lr_backoff() else 0.0

    def train_step(
        self,
        batch: Dict[str, Any],
        epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        """执行单个训练 step（支持梯度累积）。

        返回字典除 loss 外还包含：
        - `optimizer_step`：本次是否发生参数更新；
        - `skipped_nonfinite`：是否因 non-finite 被跳过；
        - `lr_backoff_event`：是否触发学习率回退。
        """

        self._maybe_zero_grad()

        # 先走 AMP 前向；若 loss 非有限，再用全精度重算一次做兜底。
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            losses = self._compute_losses(batch)

        if self.use_amp and not self._is_finite_scalar(losses["loss"]):
            with torch.cuda.amp.autocast(enabled=False):
                losses = self._compute_losses(batch)

        if not self._is_finite_scalar(losses["loss"]):
            self._accum_step = 0
            self._zero_all_grad()
            output = {key: float(value.detach().item()) for key, value in losses.items()}
            output["optimizer_step"] = 0.0
            output["skipped_nonfinite"] = 1.0
            output["lr_backoff_event"] = self._handle_nonfinite_event()
            output["nonfinite_streak"] = float(self.nonfinite_streak)
            return output

        loss_for_backward = losses["loss"] / float(self.accumulation_steps)
        if self.use_amp:
            self.scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        optimizer_step = False
        self._accum_step += 1
        if self._accum_step >= self.accumulation_steps:
            optimizer_step = self._clip_and_step()
            self._accum_step = 0

        skipped_nonfinite = 0.0
        lr_backoff_event = 0.0
        if self._last_step_nonfinite:
            skipped_nonfinite = 1.0
            lr_backoff_event = self._handle_nonfinite_event()
        elif optimizer_step:
            self.nonfinite_streak = 0

        output = {key: float(value.detach().item()) for key, value in losses.items()}
        output["optimizer_step"] = 1.0 if optimizer_step else 0.0
        output["skipped_nonfinite"] = skipped_nonfinite
        output["lr_backoff_event"] = lr_backoff_event
        output["nonfinite_streak"] = float(self.nonfinite_streak)
        return output

    def flush_pending_gradients(self) -> bool:
        """强制消费残余累积梯度。"""

        if self._accum_step <= 0:
            return False
        stepped = self._clip_and_step()
        self._accum_step = 0
        return stepped

    def reset_after_oom(self) -> bool:
        """在 OOM/异常后重置累积状态并清梯度。"""

        had_pending = self._accum_step > 0
        self._accum_step = 0
        self._zero_all_grad()
        return had_pending

    def get_current_lr(self) -> Dict[str, float]:
        """返回两路优化器当前学习率快照。"""

        lrs: Dict[str, float] = {}
        for name, optimizer in (
            ("lens_encoder", self.optimizer_lens_encoder),
            ("restoration", self.optimizer_restoration),
        ):
            if optimizer is not None:
                lrs[name] = float(optimizer.param_groups[0]["lr"])
        return lrs

    def update_best_metrics(self, val_metrics: Dict[str, Any]) -> Dict[str, bool]:
        """更新内存中的 best 指标缓存并返回命中标记。"""

        flags = {"psnr": False, "mae": False}
        metric = float(val_metrics.get("PSNR", float("nan")))
        if math.isfinite(metric) and metric > self.best_metrics["psnr"]:
            self.best_metrics["psnr"] = metric
            flags["psnr"] = True
        mae = float(val_metrics.get("MAE", float("nan")))
        if math.isfinite(mae) and mae < self.best_metrics["mae"]:
            self.best_metrics["mae"] = mae
            flags["mae"] = True
        return flags

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_metrics: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
    ) -> None:
        """序列化保存完整训练状态。"""

        checkpoint = {
            "epoch": int(epoch),
            "ablation": self.ablation,
            "val_metrics": val_metrics if val_metrics is not None else {},
            "best_metrics": self.best_metrics,
            "accum_step": int(self._accum_step),
            "global_step": None if global_step is None else int(global_step),
            "lens_encoder_optimizer_steps": int(self.lens_encoder_optimizer_steps),
            "restoration_optimizer_steps": int(self.restoration_optimizer_steps),
            "lens_table_encoder": self.lens_table_encoder.state_dict(),
            "restoration_net": self.restoration_net.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
        }
        if self.optimizer_lens_encoder is not None:
            checkpoint["optimizer_lens_encoder"] = self.optimizer_lens_encoder.state_dict()
        if self.optimizer_restoration is not None:
            checkpoint["optimizer_restoration"] = self.optimizer_restoration.state_dict()
        if self.scheduler_lens_encoder is not None:
            checkpoint["scheduler_lens_encoder"] = self.scheduler_lens_encoder.state_dict()
        if self.scheduler_restoration is not None:
            checkpoint["scheduler_restoration"] = self.scheduler_restoration.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """加载 checkpoint 到当前 trainer。

        - `load_optimizer=False` 用于 warm-start；
        - `load_optimizer=True` 用于断点续训。
        """

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "restoration_net" not in checkpoint:
            raise ValueError("Expected a restoration checkpoint containing restoration_net.")
        checkpoint, sanitization_report = sanitize_legacy_checkpoint(checkpoint)

        if "lens_table_encoder" in checkpoint:
            self.lens_table_encoder.load_state_dict(
                checkpoint.get("lens_table_encoder", {}), strict=False
            )
        self.restoration_net.load_state_dict(checkpoint.get("restoration_net", {}), strict=False)
        self._accum_step = 0
        self._zero_all_grad()

        if load_optimizer:
            for name, optimizer in (
                ("optimizer_lens_encoder", self.optimizer_lens_encoder),
                ("optimizer_restoration", self.optimizer_restoration),
            ):
                if name in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint[name])
            for name, scheduler in (
                ("scheduler_lens_encoder", self.scheduler_lens_encoder),
                ("scheduler_restoration", self.scheduler_restoration),
            ):
                if name in checkpoint and scheduler is not None:
                    try:
                        scheduler.load_state_dict(checkpoint[name])
                    except Exception:
                        pass
            if self.use_amp and checkpoint.get("scaler") is not None:
                self.scaler.load_state_dict(checkpoint["scaler"])

        self.lens_encoder_optimizer_steps = int(checkpoint.get("lens_encoder_optimizer_steps", 0))
        self.restoration_optimizer_steps = int(checkpoint.get("restoration_optimizer_steps", 0))
        if isinstance(checkpoint.get("best_metrics"), dict):
            self.best_metrics.update(
                {k: v for k, v in checkpoint["best_metrics"].items() if k in self.best_metrics}
            )

        return {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "val_metrics": checkpoint.get("val_metrics", {}),
            "sanitization_report": sanitization_report,
        }

    def log_to_tensorboard(self, metrics: Dict[str, Any], step: int, prefix: str = "train") -> None:
        """把标量指标写入 TensorBoard。"""

        if self.writer is None:
            return
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            scalar = float(value)
            if math.isnan(scalar) or math.isinf(scalar):
                continue
            self.writer.add_scalar(f"{prefix}/{key}", scalar, step)

    def close_tensorboard(self) -> None:
        """关闭 TensorBoard writer。"""

        if self.writer is not None:
            self.writer.close()
            self.writer = None
