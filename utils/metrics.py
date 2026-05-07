"""LensOracle 评估指标集合。

评估目标：在图像空间评估恢复质量（PSNR/SSIM/MAE/LPIPS）。

本模块仅负责评估，不参与任何训练反向传播或参数更新。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def resolve_stage_metric_spec(metric_name: str, stage: str) -> Tuple[str, Tuple[str, ...], bool]:
    """Resolve a configured metric name to canonical keys and comparison direction."""

    normalized = str(metric_name).strip().lower().replace("-", "_")
    if normalized == "psnr":
        return "PSNR", ("PSNR", "psnr"), True
    if normalized == "ssim":
        return "SSIM", ("SSIM", "ssim"), True
    if normalized == "lpips":
        return "LPIPS", ("LPIPS", "lpips"), False
    if normalized in {"mae", "val_loss"}:
        return "MAE" if normalized == "mae" else "val_loss", ("val_loss", "MAE", "mae", "loss"), False
    return "PSNR", ("PSNR", "psnr"), True


def get_numeric_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    """Extract a finite numeric metric with case-insensitive key fallback."""

    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    lowered = key.lower()
    for metric_key, metric_value in metrics.items():
        if str(metric_key).lower() == lowered and isinstance(metric_value, (int, float)):
            scalar = float(metric_value)
            if math.isfinite(scalar):
                return scalar
    return None


def extract_stage_score(metrics: Dict[str, Any], metric_name: str, stage: str) -> Optional[Tuple[str, float, bool]]:
    """Extract the comparable score for a training stage."""

    display_name, candidates, maximize = resolve_stage_metric_spec(metric_name, stage)
    for candidate in candidates:
        value = get_numeric_metric(metrics, candidate)
        if value is not None:
            return display_name, value, maximize
    return None


class PerformanceEvaluator:
    """统一评估器。

    封装图像恢复评估的底层指标函数、模型模式切换、
    以及 full-resolution 逐样本统计逻辑。
    """

    def __init__(self, device: str = "cuda", ssim_window: int = 11, ssim_sigma: float = 1.5):
        self.device = device
        self.ssim_window = int(ssim_window)
        self.ssim_sigma = float(ssim_sigma)
        self._lpips = None
        self._lpips_available: bool | None = None

    @staticmethod
    def _psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0, eps: float = 1.0e-8) -> torch.Tensor:
        mse = F.mse_loss(x, y)
        return 10.0 * torch.log10(max_val**2 / (mse + eps))

    @staticmethod
    def _gaussian_window(size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
        kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.expand(channels, 1, size, size).contiguous()

    def _ssim(self, x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        channels = int(x.shape[1])
        window_size = min(self.ssim_window, int(x.shape[-1]), int(x.shape[-2]))
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 3:
            return torch.tensor(1.0, device=x.device, dtype=x.dtype) / (1.0 + F.l1_loss(x, y))
        window = self._gaussian_window(window_size, self.ssim_sigma, channels, x.device, x.dtype)
        padding = window_size // 2
        mu_x = F.conv2d(x, window, padding=padding, groups=channels)
        mu_y = F.conv2d(y, window, padding=padding, groups=channels)
        mu_x2 = mu_x.square()
        mu_y2 = mu_y.square()
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
        sigma_y = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
        )
        return ssim_map.mean()

    def _ensure_lpips_loaded(self) -> None:
        if self._lpips_available is not None:
            return
        try:
            import lpips

            self._lpips = lpips.LPIPS(net="alex").to(self.device)
            self._lpips_available = True
        except Exception:
            self._lpips = None
            self._lpips_available = False

    def _lpips_score(self, x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        if min(int(x.shape[-2]), int(x.shape[-1])) < 64:
            return None
        self._ensure_lpips_loaded()
        if not self._lpips_available or self._lpips is None:
            return None
        return self._lpips(x * 2.0 - 1.0, y * 2.0 - 1.0).mean()

    @staticmethod
    def _count_parameters(*models: Optional[nn.Module]) -> float:
        params = 0
        for model in models:
            if model is None:
                continue
            params += sum(int(p.numel()) for p in model.parameters())
        return float(params) / 1.0e6

    @staticmethod
    def _sanitize_image_tensor(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    @staticmethod
    def _compute_mae(x: torch.Tensor, y: torch.Tensor) -> float:
        return float(F.l1_loss(x, y).item())

    @staticmethod
    def _accumulate_if_finite(total: float, count: int, value: float) -> Tuple[float, int]:
        if math.isfinite(value):
            return total + value, count + 1
        return total, count

    def _aggregate_metric_list(self, values: Iterable[float]) -> float:
        total = 0.0
        count = 0
        for value in values:
            total, count = self._accumulate_if_finite(total, count, float(value))
        return total / count if count > 0 else float("nan")

    def aggregate_metric_list(self, values: Iterable[float]) -> float:
        return self._aggregate_metric_list(values)

    def _set_eval_mode(self, *models: Optional[nn.Module]) -> List[Tuple[nn.Module, bool]]:
        states: List[Tuple[nn.Module, bool]] = []
        for model in models:
            if model is None:
                continue
            states.append((model, bool(model.training)))
            model.eval()
        return states

    @staticmethod
    def _restore_train_mode(states: List[Tuple[nn.Module, bool]]) -> None:
        for model, was_training in states:
            model.train(was_training)

    @staticmethod
    def _extract_image_pair(batch: Dict[str, Any], device: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        blur = batch["blur"].to(device)
        sharp_raw = batch.get("sharp")
        if sharp_raw is None or not torch.is_tensor(sharp_raw):
            return blur, None
        return blur, sharp_raw.to(device)

    def _compute_image_metrics(self, restored: torch.Tensor, sharp: Optional[torch.Tensor]) -> Dict[str, float]:
        if sharp is None:
            return {"PSNR": float("nan"), "SSIM": float("nan"), "MAE": float("nan"), "LPIPS": float("nan")}
        restored_eval = self._sanitize_image_tensor(restored)
        sharp_eval = self._sanitize_image_tensor(sharp)
        lp = self._lpips_score(restored_eval, sharp_eval)
        return {
            "PSNR": float(self._psnr(restored_eval, sharp_eval).item()),
            "SSIM": float(self._ssim(restored_eval, sharp_eval).item()),
            "MAE": self._compute_mae(restored_eval, sharp_eval),
            "LPIPS": float(lp.item()) if lp is not None else float("nan"),
        }

    def compute_image_metrics(self, restored: torch.Tensor, sharp: Optional[torch.Tensor]) -> Dict[str, float]:
        return self._compute_image_metrics(restored, sharp)

    def evaluate(
        self,
        restoration_net: nn.Module,
        psf_net: Optional[nn.Module],
        val_loader,
        device: str,
        lens_table_encoder: Optional[nn.Module] = None,
        prior_mode: str = "correct_gt",
    ) -> Dict[str, float]:
        """评估图像恢复指标。

        直接从 batch 读取 GT PSF-SFR，经 lens_table_encoder 注入 restoration_net。
        """

        active_lens_encoder = lens_table_encoder
        states = self._set_eval_mode(restoration_net, active_lens_encoder)
        psnr_values: List[float] = []
        ssim_values: List[float] = []
        mae_values: List[float] = []
        lpips_values: List[float] = []
        try:
            with torch.no_grad():
                for batch in val_loader:
                    blur, sharp = self._extract_image_pair(batch, device)
                    crop_info = batch.get("crop_info")
                    if torch.is_tensor(crop_info):
                        crop_info = crop_info.to(device)

                    prior_table = batch.get("gt_psf_sfr")
                    if torch.is_tensor(prior_table):
                        prior_table = prior_table.to(device)
                    else:
                        prior_table = None

                    if active_lens_encoder is not None and prior_table is not None:
                        lens_features = active_lens_encoder(prior_table)
                    else:
                        lens_features = None

                    restored = restoration_net(blur, lens_features, crop_info=crop_info)
                    metrics = self._compute_image_metrics(restored, sharp)
                    psnr_values.append(metrics["PSNR"])
                    ssim_values.append(metrics["SSIM"])
                    mae_values.append(metrics["MAE"])
                    lpips_values.append(metrics["LPIPS"])
        finally:
            self._restore_train_mode(states)
        return {
            "PSNR": self._aggregate_metric_list(psnr_values),
            "SSIM": self._aggregate_metric_list(ssim_values),
            "MAE": self._aggregate_metric_list(mae_values),
            "LPIPS": self._aggregate_metric_list(lpips_values),
            "Params(M)": self._count_parameters(restoration_net, active_lens_encoder),
        }

    @staticmethod
    def evaluate_model(restoration_net: nn.Module, psf_net: Optional[nn.Module], val_loader, device: str) -> Dict[str, float]:
        evaluator = PerformanceEvaluator(device=device)
        return evaluator.evaluate(restoration_net=restoration_net, psf_net=psf_net, val_loader=val_loader, device=device)

    def evaluate_full_resolution(
        self,
        restoration_net: nn.Module,
        psf_net: Optional[nn.Module],
        test_loader,
        device: str,
        lens_table_encoder: Optional[nn.Module] = None,
        prior_mode: str = "correct_gt",
    ) -> Tuple[Dict[str, float], List[Dict[str, float | str]]]:
        """全分辨率评估并返回逐图明细。"""

        active_lens_encoder = lens_table_encoder
        states = self._set_eval_mode(restoration_net, active_lens_encoder)
        results: List[Dict[str, float | str]] = []
        psnr_values: List[float] = []
        ssim_values: List[float] = []
        mae_values: List[float] = []
        lpips_values: List[float] = []
        try:
            with torch.no_grad():
                for batch in test_loader:
                    blur, sharp = self._extract_image_pair(batch, device)
                    crop_info = batch.get("crop_info")
                    if torch.is_tensor(crop_info):
                        crop_info = crop_info.to(device)

                    prior_table = batch.get("gt_psf_sfr")
                    if torch.is_tensor(prior_table):
                        prior_table = prior_table.to(device)
                    else:
                        prior_table = None

                    if active_lens_encoder is not None and prior_table is not None:
                        lens_features = active_lens_encoder(prior_table)
                    else:
                        lens_features = None

                    restored = restoration_net(blur, lens_features, crop_info=crop_info)

                    for index in range(int(blur.shape[0])):
                        sample_metrics = self._compute_image_metrics(
                            restored[index : index + 1],
                            None if sharp is None else sharp[index : index + 1],
                        )
                        raw_filename = batch.get("filename", "sample.png")
                        filename = str(raw_filename[index] if isinstance(raw_filename, (list, tuple)) else raw_filename)
                        results.append({"filename": filename, **sample_metrics})
                        psnr_values.append(sample_metrics["PSNR"])
                        ssim_values.append(sample_metrics["SSIM"])
                        mae_values.append(sample_metrics["MAE"])
                        lpips_values.append(sample_metrics["LPIPS"])
        finally:
            self._restore_train_mode(states)
        average_metrics = {
            "PSNR": self._aggregate_metric_list(psnr_values),
            "SSIM": self._aggregate_metric_list(ssim_values),
            "MAE": self._aggregate_metric_list(mae_values),
            "LPIPS": self._aggregate_metric_list(lpips_values),
            "Num_Images": len(results),
            "Params(M)": self._count_parameters(restoration_net, active_lens_encoder),
        }
        return average_metrics, results

    def _build_injection_aware_benchmark_model(
        self,
        restoration_net: nn.Module,
        psf_net: Optional[nn.Module],
        device: str,
        lens_table_encoder: Optional[nn.Module] = None,
    ) -> nn.Module:
        """构建用于 FLOPs/速度测试的统一包装模型。"""

        class _PipelineWrapper(nn.Module):
            def __init__(self, restoration_model, lens_encoder):
                super().__init__()
                self.restoration_model = restoration_model
                self.lens_encoder = lens_encoder

            def forward(self, blur: torch.Tensor, prior_table: Optional[torch.Tensor] = None) -> torch.Tensor:
                if self.lens_encoder is None or prior_table is None:
                    return self.restoration_model(blur)
                features = self.lens_encoder(prior_table)
                return self.restoration_model(blur, features)

        wrapper = _PipelineWrapper(restoration_net, lens_table_encoder).to(device)
        wrapper.eval()
        return wrapper

    @staticmethod
    def _try_flops(model: nn.Module, device: str, input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256)) -> Optional[float]:
        try:
            from thop import profile
        except Exception:
            return None
        try:
            dummy = torch.randn(*input_shape, device=device)
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
        except Exception:
            return None
        return float(flops)
