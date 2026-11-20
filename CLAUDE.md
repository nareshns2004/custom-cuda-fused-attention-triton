# Kernel lab: guidance for Claude Code

## Goal
Correct, measured GPU kernels: add, layernorm, matmul (learning set, CUDA + Triton)
and fused attention in Triton (prefill forward, decode with split-K, paged KV).
Reference is always PyTorch (torch.nn.functional.scaled_dot_product_attention for
attention). Forward only. Read docs/design-decisions.md before changing kernels/.

The attention kernel is written by the repo owner, not by Claude Code. Claude Code writes
tests, benchmarks and harnesses, and reviews kernel code without rewriting it.

## Commands
- pip install -e ".[dev]"
- TRITON_INTERPRET=1 pytest -m "not gpu"   # CPU, must pass before every commit (= make test)
- pytest -m gpu                            # GPU host only, when asked
- make build-cuda                          # CUDA extension, GPU host only
- make bench                               # writes benchmarks/RESULTS.md

## Rules
- Every kernel change ships with a test against PyTorch, covering: sizes not
  divisible by the block size, sizes larger than one block, non-contiguous inputs,
  FP16/BF16 and FP32, and (attention) causal and ragged lengths.
- State tolerances per dtype in one place (tests/tolerances.py); never loosen a
  tolerance to make a test pass without explaining why in the PR.
- Never write performance numbers by hand. RESULTS.md is generated and records
  GPU, driver, CUDA, Triton and PyTorch versions.
- Benchmarks use triton.testing.do_bench (warm-up + median), never time.time().
- Explain every new kernel in docs/ with: what bounds it, the tiling, and one
  Nsight Compute observation.
- Track audit findings in docs/audit.md; update the status in the commit that changes it.
- Don't add kernels outside the scope above without asking.

## Safety
- Ask before running anything that changes GPU state (nvidia-smi -lgc/-pl, MIG).
