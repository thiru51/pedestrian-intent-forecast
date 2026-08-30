"""One place that decides what hardware we run on and how.

Everything else in the package asks this module rather than calling
torch.cuda.* directly, so there is exactly one definition of "the device" and
one definition of "which reduced precision is safe here".
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    is_cuda: bool
    total_vram_gb: float
    free_vram_gb: float
    capability: tuple[int, int] | None
    bf16: bool
    amp_dtype: torch.dtype | None
    tf32: bool

    @property
    def needs_grad_scaler(self) -> bool:
        """GradScaler exists to stop fp16 gradients underflowing to zero. bf16
        has fp32's exponent range, so scaling it is pointless overhead."""
        return self.amp_dtype is torch.float16

    def summary(self) -> str:
        if not self.is_cuda:
            return f"device=cpu ({self.name}), amp={_dtype_name(self.amp_dtype)}"
        cap = f"sm_{self.capability[0]}{self.capability[1]}" if self.capability else "?"
        return (
            f"device=cuda:{self.device.index or 0} ({self.name}), {cap}, "
            f"{self.total_vram_gb:.1f} GiB total / {self.free_vram_gb:.1f} GiB free, "
            f"bf16={self.bf16}, amp={_dtype_name(self.amp_dtype)}, tf32={self.tf32}"
        )


def _dtype_name(dtype: torch.dtype | None) -> str:
    return "off" if dtype is None else str(dtype).replace("torch.", "")


def resolve_device(requested: str | None = None) -> torch.device:
    """`--device auto` (or nothing at all) means cuda when there is a cuda."""
    if requested in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was asked for but torch.cuda.is_available() is False. "
            "Run `python scripts/check_gpu.py` to see why."
        )
    return dev


def supports_bf16(device: torch.device) -> bool:
    if device.type != "cuda":
        # CPU autocast does support bfloat16, but it is slower than plain fp32
        # for models this small, so we never turn it on there.
        return False
    return torch.cuda.is_bf16_supported()


def setup(
    requested: str | None = None,
    amp: bool = True,
    tf32: bool = True,
    benchmark: bool = True,
    seed: int | None = None,
) -> DeviceInfo:
    device = resolve_device(requested)

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    tf32_on = False
    if device.type == "cuda":
        if tf32:
            # TF32 keeps fp32's exponent but truncates the mantissa to 10 bits,
            # which is plenty for this model and roughly triples matmul rate on
            # Ampere and later. Three separate switches control it.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            tf32_on = True
        # Fixed-shape windows, so letting cudnn autotune once pays off.
        torch.backends.cudnn.benchmark = benchmark

    bf16 = supports_bf16(device)
    if not amp or device.type != "cuda":
        amp_dtype: torch.dtype | None = None
    else:
        amp_dtype = torch.bfloat16 if bf16 else torch.float16

    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        return DeviceInfo(
            device=torch.device("cuda", idx),
            name=props.name,
            is_cuda=True,
            total_vram_gb=total_b / 2**30,
            free_vram_gb=free_b / 2**30,
            capability=(props.major, props.minor),
            bf16=bf16,
            amp_dtype=amp_dtype,
            tf32=tf32_on,
        )

    return DeviceInfo(
        device=device,
        name=_cpu_name(),
        is_cuda=False,
        total_vram_gb=0.0,
        free_vram_gb=0.0,
        capability=None,
        bf16=False,
        amp_dtype=None,
        tf32=False,
    )


def _cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "cpu"


# Batch size that comfortably fits alongside a desktop session, keyed by the
# smallest card it is meant for. The model is tiny; these are deliberately
# conservative and are meant to be overridden with --batch-size on real runs.
_VRAM_BATCH = ((24.0, 1024), (16.0, 512), (10.0, 256), (6.0, 128), (0.0, 64))


def auto_batch_size(info: DeviceInfo) -> int:
    if not info.is_cuda:
        return 64
    for floor, batch in _VRAM_BATCH:
        if info.total_vram_gb >= floor:
            return batch
    return 64


def auto_num_workers(requested: int | None = None) -> int:
    if requested is not None and requested >= 0:
        return requested
    cpus = os.cpu_count() or 2
    # Leave headroom for the main process and keep it sane on big machines:
    # this dataset is small tensors, not JPEG decoding, so more workers past a
    # point is just more IPC.
    return max(0, min(8, cpus - 2))


def peak_vram_gb(info: DeviceInfo) -> float:
    if not info.is_cuda:
        return 0.0
    return torch.cuda.max_memory_allocated(info.device) / 2**30


def reset_peak_vram(info: DeviceInfo) -> None:
    if info.is_cuda:
        torch.cuda.reset_peak_memory_stats(info.device)


def synchronize(info: DeviceInfo) -> None:
    """CUDA kernels are async, so any wall-clock measurement has to sync first
    or it just measures how fast Python can queue work."""
    if info.is_cuda:
        torch.cuda.synchronize(info.device)
