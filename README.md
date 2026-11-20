# Kernel lab: CUDA and Triton

> **Status: kernel lab.** This repo holds a small learning set of GPU kernels (vector add,
> layernorm, tiled matmul) in both CUDA and Triton, with tests against PyTorch.
> **It does not contain a fused attention kernel yet**, despite the repo name. The Triton fused
> attention kernel (prefill forward, split-K decode, paged KV) is being built inside the
> disaggregated inference engine, where its KV block layout is the same unit the engine moves
> between GPUs over RDMA. This repo will link there once it exists.

## What exists

| Kernel    | Triton                                   | CUDA                                         | Op (`ops/`)          |
|-----------|------------------------------------------|----------------------------------------------|----------------------|
| add       | `kernels/triton/add_kernel.py`           | `kernels/cuda/add_kernel.cu`                 | `ops/add_op.py`      |
| layernorm | `kernels/triton/layernorm_kernel.py`     | `kernels/cuda/layernorm_kernel.cu`           | `ops/layernorm_op.py`|
| matmul    | `kernels/triton/matmul_kernel.py`        | `kernels/cuda/matmul_kernel.cu` (16×16 tiles)| `ops/matmul_op.py`   |

Each op takes `backend="triton" | "cuda" | "torch"`, so A/B comparisons are one argument.
The CUDA kernels are deliberately simple teaching baselines (no register blocking, no tensor
cores); the Triton ones are the main path.

Known bugs found by the audit, and their fix status, are tracked in
[`docs/audit.md`](docs/audit.md). Design decisions are in
[`docs/design-decisions.md`](docs/design-decisions.md).

## Layout

```text
kernels/triton/   Triton kernels and launchers
kernels/cuda/     CUDA kernels, Torch C++ extension bindings, CMakeLists.txt
ops/              Backend dispatch (triton / cuda / torch)
tests/            Correctness vs PyTorch; CPU via TRITON_INTERPRET=1, GPU-only tests marked `gpu`
benchmarks/       triton.testing.do_bench scripts; `make bench` generates RESULTS.md
docs/             Audit, design decisions
```

## Usage

```bash
pip install -e ".[dev]"

# Correctness on CPU (no GPU needed), via Triton's interpreter
make test                 # = TRITON_INTERPRET=1 pytest -m "not gpu"

# On a GPU host
make build-cuda           # build the CUDA extension (CMake, C++20)
make test-gpu             # all tests, including CUDA-extension tests
make racecheck            # compute-sanitizer racecheck on the CUDA layernorm
make bench                # writes benchmarks/RESULTS.md, stamped with versions
```

```python
import torch
from ops.layernorm_op import layernorm

x = torch.randn(8, 4096, device="cuda")
w, b = torch.ones(4096, device="cuda"), torch.zeros(4096, device="cuda")
y = layernorm(x, w, b, backend="triton")
```

## Performance

No performance numbers are written by hand in this repo. `make bench` generates
`benchmarks/RESULTS.md` on a GPU, recording the GPU, driver, CUDA, Triton and PyTorch versions.
Until that file exists, there are no measured numbers.

## Scope

In scope: the three learning-set kernels above, correct for any size and stride, tested and
benchmarked against PyTorch. Out of scope here: attention (lives in the inference engine), a
backward pass, and the broader CUDA curriculum (prefix sum, histogram, CUTLASS, TensorRT
plugins, MoE kernels).

## License

Apache-2.0.
