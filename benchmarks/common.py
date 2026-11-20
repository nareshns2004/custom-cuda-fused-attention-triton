"""Shared benchmarking helpers. GPU only.

Every timing goes through triton.testing.do_bench (warm-up, many repetitions, median), never
time.time(): the first Triton call compiles (and autotunes), and a single launch is dominated
by launch overhead.
"""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
from dataclasses import dataclass, field

import torch
import triton
from triton.testing import do_bench

try:
    import cuda_kernels
except ImportError:
    cuda_kernels = None

HAS_CUDA_EXT = cuda_kernels is not None


def median_ms(fn) -> float:
    return do_bench(fn, warmup=25, rep=100, return_mode="median")


@dataclass
class Table:
    title: str
    columns: list[str]
    rows: list[list] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, *row) -> None:
        self.rows.append(list(row))

    def to_markdown(self) -> str:
        def fmt(v):
            if isinstance(v, float):
                return f"{v:.1f}" if v >= 10 else f"{v:.3f}"
            return str(v)

        lines = [f"## {self.title}", ""]
        lines += [f"{n}" for n in self.notes] + ([""] if self.notes else [])
        lines.append("| " + " | ".join(self.columns) + " |")
        lines.append("|" + "|".join("---" for _ in self.columns) + "|")
        lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in self.rows]
        return "\n".join(lines) + "\n"


def _driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def environment() -> dict[str, str]:
    props = torch.cuda.get_device_properties(0)
    return {
        "Date (UTC)": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "GPU": f"{props.name} ({props.multi_processor_count} SMs, "
        f"{props.total_memory / 2**30:.0f} GiB, sm_{props.major}{props.minor})",
        "Driver": _driver_version(),
        "CUDA (PyTorch build)": str(torch.version.cuda),
        "PyTorch": torch.__version__,
        "Triton": triton.__version__,
        "Python": platform.python_version(),
        "CUDA extension built": "yes" if HAS_CUDA_EXT else "no",
    }


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Benchmarks need a CUDA GPU; nothing to measure on this machine.")
