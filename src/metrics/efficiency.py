"""Inference-efficiency helpers for paper-facing model profiling."""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict

import torch
from torch import nn


ForwardFn = Callable[[nn.Module, Any], Any]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device).strip().lower()
    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_detection_inputs(*, batch_size: int, image_size: int, device: torch.device) -> list[torch.Tensor]:
    """Create a torchvision-style detection batch as a list of images."""
    return [
        torch.zeros((3, image_size, image_size), dtype=torch.float32, device=device)
        for _ in range(max(int(batch_size), 1))
    ]


def build_tensor_inputs(*, batch_size: int, image_size: int, device: torch.device) -> torch.Tensor:
    """Create a dense tensor batch for standard tensor-first models."""
    return torch.zeros((max(int(batch_size), 1), 3, image_size, image_size), dtype=torch.float32, device=device)


def detection_forward(model: nn.Module, inputs: list[torch.Tensor]) -> Any:
    """Run inference for torchvision-style detection models."""
    return model(inputs)


def tensor_forward(model: nn.Module, inputs: torch.Tensor) -> Any:
    """Run inference for tensor-first models."""
    return model(inputs)


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _maybe_profile_flops(
    *,
    model: nn.Module,
    profile_inputs: Any,
    device: torch.device,
) -> tuple[float | None, float | None, str | None]:
    try:
        from thop import profile
    except ImportError:
        return None, None, "thop_not_installed"

    if device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        macs, params = profile(model, inputs=(profile_inputs,), verbose=False)
        return float(macs), float(params), None
    except Exception as exc:  # pragma: no cover - model-specific operator support varies
        return None, None, f"{type(exc).__name__}: {exc}"


def summarize_efficiency(
    model: nn.Module,
    *,
    input_builder: Callable[..., Any],
    forward_fn: ForwardFn,
    image_size: int = 1024,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    warmup_iterations: int = 10,
    timed_iterations: int = 30,
    include_flops: bool = True,
) -> Dict[str, float | int | None | str]:
    """Measure core inference-efficiency statistics for a model.

    Parameters
    ----------
    model:
        Model already constructed with the architecture of interest.
    input_builder:
        Callable that creates a dummy batch on the target device.
    forward_fn:
        Callable used to invoke model inference for the chosen input type.
    image_size:
        Square input size used to build the dummy batch.
    batch_size:
        Batch size used for latency and throughput measurements.
    device:
        Target device. Accepts values such as ``cpu``, ``cuda``, or ``0``.
    warmup_iterations / timed_iterations:
        Warmup and timed iteration counts for latency measurement.
    include_flops:
        Whether to attempt a THOP MAC/FLOP estimate.
    """

    target_device = _resolve_device(device)
    model = model.to(target_device)
    model.eval()

    total_params, trainable_params = _count_parameters(model)
    latency_samples_ms: list[float] = []
    flops_error: str | None = None
    macs: float | None = None
    params_from_profile: float | None = None

    with torch.inference_mode():
        profile_inputs = input_builder(batch_size=1, image_size=image_size, device=target_device)
        if include_flops:
            macs, params_from_profile, flops_error = _maybe_profile_flops(
                model=model,
                profile_inputs=profile_inputs,
                device=target_device,
            )

        latency_inputs = input_builder(batch_size=batch_size, image_size=image_size, device=target_device)

        if target_device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(target_device)

        for _ in range(max(int(warmup_iterations), 0)):
            forward_fn(model, latency_inputs)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)

        for _ in range(max(int(timed_iterations), 1)):
            start = time.perf_counter()
            forward_fn(model, latency_inputs)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latency_samples_ms.append(elapsed_ms)

    mean_latency_ms = statistics.mean(latency_samples_ms)
    std_latency_ms = statistics.stdev(latency_samples_ms) if len(latency_samples_ms) > 1 else 0.0
    throughput_images_per_sec = (float(batch_size) * 1000.0) / mean_latency_ms

    peak_memory_bytes = None
    if target_device.type == "cuda":
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(target_device))

    return {
        "device": str(target_device),
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "warmup_iterations": int(warmup_iterations),
        "timed_iterations": int(timed_iterations),
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "params_million": float(total_params) / 1_000_000.0,
        "trainable_params_million": float(trainable_params) / 1_000_000.0,
        "profile_params_million": None if params_from_profile is None else float(params_from_profile) / 1_000_000.0,
        "macs_giga": None if macs is None else float(macs) / 1_000_000_000.0,
        "flops_giga": None if macs is None else (float(macs) * 2.0) / 1_000_000_000.0,
        "latency_ms_mean": float(mean_latency_ms),
        "latency_ms_std": float(std_latency_ms),
        "throughput_images_per_sec": float(throughput_images_per_sec),
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gb": None if peak_memory_bytes is None else float(peak_memory_bytes) / (1024.0**3),
        "flops_error": flops_error,
    }
