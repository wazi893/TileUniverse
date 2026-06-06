#!/bin/bash

# Exit on error
set -e

echo "=== Testing Rust Core ==="
PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64:/c/Users/waziz/.cargo/bin:$PATH" cargo test --lib deutsch_jozsa --no-fail-fast
PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64:/c/Users/waziz/.cargo/bin:$PATH" cargo test --lib grover --no-fail-fast

echo ""
echo "=== Testing Rust Integration ==="
PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64:/c/Users/waziz/.cargo/bin:$PATH" cargo test --test dj_bv_integration --no-fail-fast
PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64:/c/Users/waziz/.cargo/bin:$PATH" cargo test --test grover_integration --no-fail-fast

echo ""
echo "=== Building Python Package ==="
cd python
VIRTUAL_ENV="C:/Dev/logic-fabric/engine/python/.venv" \
PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64:/c/Users/waziz/.cargo/bin:$PATH:C:/Dev/logic-fabric/engine/python/.venv/Scripts" \
"C:/Dev/logic-fabric/engine/python/.venv/Scripts/python.exe" -m maturin develop --quiet

echo ""
echo "=== Testing Python API ==="
VIRTUAL_ENV="C:/Dev/logic-fabric/engine/python/.venv" \
PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64:/c/Users/waziz/.cargo/bin:$PATH:C:/Dev/logic-fabric/engine/python/.venv/Scripts" \
"C:/Dev/logic-fabric/engine/python/.venv/Scripts/python.exe" -m pytest tests/test_quantum_api.py tests/test_dj_bv.py -v

echo ""
echo "=== Running Demos ==="
VIRTUAL_ENV="C:/Dev/logic-fabric/engine/python/.venv" \
PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64:/c/Users/waziz/.cargo/bin:$PATH:C:/Dev/logic-fabric/engine/python/.venv/Scripts" \
"C:/Dev/logic-fabric/engine/python/.venv/Scripts/python.exe" examples/quantum_algorithms_demo.py

echo ""
echo "✅ All tests complete!"
