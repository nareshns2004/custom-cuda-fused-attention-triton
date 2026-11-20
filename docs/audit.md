# Audit findings and status

Findings C1–C12 come from the master booklet audit of commit `b485bbc` (20 Nov 2026).
C13 was found while writing the tests. Status is updated in the commit that changes it.

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| C1 | High | Triton layernorm silently wrong when hidden size > `BLOCK_SIZE` (1024): max error 3.1e35 at 4096 | Fixed (P2): block = next power of two up to 64 KiB of input, chunked loop above that |
| C2 | Medium | Triton layernorm ignores strides (`row_start = row_idx * hidden_size`) | Fixed (P2): row and column strides passed to the kernel |
| C3 | High | Race in CUDA layernorm `block_reduce_sum`: second call writes `shared[]` with no barrier while other warps may still read `shared[0]` from the first; also hard-codes 256 threads | Open |
| C4 | High | `bindings.cpp` doesn't compile: `at::cuda::getCurrentCUDAStream` needs `ATen/cuda/CUDAContext.h`; no device guard | Open |
| C5 | High | CMake targets C++17 (PyTorch needs C++20); `CUDA_ARCHITECTURES native` needs CMake ≥ 3.24, minimum declared 3.18 | Open |
| C6 | High | Repo name promises fused attention; there is none. README title and tree didn't match the repo | README fixed; repo rename is the owner's call |
| C7 | Medium | Packaging empty; `pyproject.py` instead of `pyproject.toml`; error message pointed at commands that didn't work | Fixed |
| C8 | Medium | No tests, benchmarks or profiling | Open |
| C9 | Low | FP32 `tl.dot` uses TF32 on Ampere+ by default, so GPU results differ from `torch.matmul` | Open |
| C10 | Low | CUDA layernorm reads each row three times; CUDA matmul has no register blocking or tensor cores; Triton matmul had no autotune | Accepted: CUDA kernels are labelled baselines |
| C11 | Medium | README was a 7-phase, 40+ kernel curriculum | Fixed |
| C12 | Low | `main` is one commit on a different root from the PR branches | Small PRs from now on |
| C13 | Medium | Triton add ignores strides: flat offsets on non-contiguous or mixed-layout inputs give wrong results (max error 5.4 for a transposed + contiguous pair) | Fixed (P2): op makes operands contiguous (copy is the documented cost); shape/dtype checks |
