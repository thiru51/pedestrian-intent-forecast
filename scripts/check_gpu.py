"""Run this first. It tells you whether the GPU path is actually going to work.

Prints torch/CUDA versions, the device, VRAM, compute capability, whether bf16
is real on this card, whether TF32 got switched on, and a small matmul
benchmark so there is a number to compare against when something feels slow.

Exit code is 0 even on CPU-only — a CPU box can still run everything here, just
slower. It is non-zero only if torch itself is broken.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument(
        "--matmul-size", type=int, default=4096, help="N for the NxN benchmark"
    )
    args = parser.parse_args()

    print(f"python           {platform.python_version()}  ({sys.executable})")
    print(f"platform         {platform.platform()}")

    try:
        import torch
    except ImportError as exc:
        print(f"\ntorch is not importable: {exc}")
        print("install it: pixi install, or pip install -r requirements.txt")
        return 1

    print(f"torch            {torch.__version__}")
    print(f"torch cuda build {torch.version.cuda or 'cpu-only wheel'}")
    print(f"cudnn            {torch.backends.cudnn.version() or 'n/a'}")
    print(f"cuda available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device count     {torch.cuda.device_count()}")

    from pif.device import auto_batch_size, auto_num_workers, setup

    info = setup(args.device, amp=args.amp)
    print()
    print(f"resolved device  {info.device}")
    print(f"device name      {info.name}")
    if info.is_cuda:
        print(f"compute cap      sm_{info.capability[0]}{info.capability[1]}")
        print(f"total VRAM       {info.total_vram_gb:.2f} GiB")
        print(f"free VRAM        {info.free_vram_gb:.2f} GiB")
    print(f"bf16 supported   {info.bf16}")
    print(
        f"amp dtype        "
        f"{'off' if info.amp_dtype is None else str(info.amp_dtype).replace('torch.', '')}"
    )
    print(f"GradScaler       {'yes (fp16)' if info.needs_grad_scaler else 'no'}")
    print(f"TF32 matmul      {torch.backends.cuda.matmul.allow_tf32 if info.is_cuda else False}")
    print(f"TF32 cudnn       {torch.backends.cudnn.allow_tf32 if info.is_cuda else False}")
    print(f"float32 matmul   {torch.get_float32_matmul_precision()}")
    print(f"cudnn.benchmark  {torch.backends.cudnn.benchmark}")
    print()
    print(f"auto batch size  {auto_batch_size(info)}")
    print(f"auto num workers {auto_num_workers(None)}")

    n = args.matmul_size
    print()
    print(f"benchmark        {n}x{n} fp32 matmul")
    a = torch.randn(n, n, device=info.device)
    b = torch.randn(n, n, device=info.device)
    for _ in range(3):  # warm up: first call pays for kernel selection
        a @ b
    if info.is_cuda:
        torch.cuda.synchronize()
    iters = 20 if info.is_cuda else 3
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    if info.is_cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tflops = 2 * n**3 / dt / 1e12
    print(f"  {dt * 1e3:8.2f} ms/iter   {tflops:6.2f} TFLOP/s")

    if info.is_cuda and info.amp_dtype is not None:
        ah = a.to(info.amp_dtype)
        bh = b.to(info.amp_dtype)
        for _ in range(3):
            ah @ bh
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            ah @ bh
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        name = str(info.amp_dtype).replace("torch.", "")
        print(f"  {dt * 1e3:8.2f} ms/iter   {2 * n**3 / dt / 1e12:6.2f} TFLOP/s  ({name})")

    if info.is_cuda:
        print()
        print(f"peak VRAM this check  {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB")
    else:
        print()
        print("No CUDA device found. Everything in this repo still runs on CPU,")
        print("just slower. If you expected a GPU: check `nvidia-smi` works, and")
        print("that the installed torch is a CUDA build (torch cuda build above")
        print("should be a version, not 'cpu-only wheel').")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
