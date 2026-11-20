"""Small driver for `compute-sanitizer --tool racecheck` (see `make racecheck`).

Runs the CUDA layernorm at block sizes that exercise multi-warp reductions. Racecheck
reports shared-memory hazards; the audit's C3 race was between the two block reductions.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cuda_kernels


def main() -> None:
    for hidden in (33, 768, 4096):
        x = torch.randn(8, hidden, device="cuda")
        w = torch.randn(hidden, device="cuda")
        b = torch.randn(hidden, device="cuda")
        out = cuda_kernels.layernorm(x, w, b, 1e-5)
        ref = torch.nn.functional.layer_norm(x, (hidden,), w, b, 1e-5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    torch.cuda.synchronize()
    print("cuda layernorm OK")


if __name__ == "__main__":
    main()
