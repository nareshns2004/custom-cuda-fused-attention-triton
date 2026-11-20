PYTHON ?= python
BUILD_DIR ?= build/cuda

.PHONY: install test test-cpu test-gpu lint bench build-cuda racecheck

install:
	$(PYTHON) -m pip install -e ".[dev]"

# CPU, no GPU needed: Triton kernels run in the interpreter. Must pass before every commit.
test test-cpu:
	TRITON_INTERPRET=1 $(PYTHON) -m pytest -m "not gpu"

# GPU host only.
test-gpu:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

# GPU host only. Writes benchmarks/RESULTS.md.
bench:
	$(PYTHON) -m benchmarks.run_all

# Builds the CUDA learning-set extension and copies cuda_kernels*.so to the repo root,
# where ops/*.py imports it from.
build-cuda:
	cmake -S kernels/cuda -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release \
		-DPython3_EXECUTABLE=$$(which $(PYTHON)) \
		-DCMAKE_PREFIX_PATH="$$($(PYTHON) -c 'import torch; print(torch.utils.cmake_prefix_path)');$$($(PYTHON) -m pybind11 --cmakedir)"
	cmake --build $(BUILD_DIR) -j
	cp $(BUILD_DIR)/cuda_kernels*.so .

racecheck: build-cuda
	compute-sanitizer --tool racecheck --error-exitcode 1 $(PYTHON) tests/run_cuda_layernorm.py
