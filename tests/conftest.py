import os
import sys
from pathlib import Path

import pytest
import torch

# Make `kernels` and `ops` importable without an install, and let tests import tolerances.
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@pytest.fixture(scope="session")
def device() -> str:
    """Device for Triton kernel tests: CPU under the interpreter, else the GPU."""
    if INTERPRET:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("Triton tests need a GPU or TRITON_INTERPRET=1")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
