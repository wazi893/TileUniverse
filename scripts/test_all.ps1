# PowerShell validation script for TileUniverse quantum algorithms

$ErrorActionPreference = "Stop"

Write-Host "=== Testing Rust Core ===" -ForegroundColor Cyan
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64;C:\Users\waziz\.cargo\bin;$env:PATH"
cargo test --lib deutsch_jozsa --no-fail-fast
cargo test --lib grover --no-fail-fast

Write-Host "`n=== Testing Rust Integration ===" -ForegroundColor Cyan
cargo test --test dj_bv_integration --no-fail-fast
cargo test --test grover_integration --no-fail-fast

Write-Host "`n=== Building Python Package ===" -ForegroundColor Cyan
Push-Location python
$env:VIRTUAL_ENV = "C:\Dev\logic-fabric\engine\python\.venv"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64;C:\Users\waziz\.cargo\bin;C:\Dev\logic-fabric\engine\python\.venv\Scripts;$env:PATH"
& "C:\Dev\logic-fabric\engine\python\.venv\Scripts\python.exe" -m maturin develop --quiet

Write-Host "`n=== Testing Python API ===" -ForegroundColor Cyan
& "C:\Dev\logic-fabric\engine\python\.venv\Scripts\python.exe" -m pytest tests/test_quantum_api.py tests/test_dj_bv.py -v

Write-Host "`n=== Running Demos ===" -ForegroundColor Cyan
& "C:\Dev\logic-fabric\engine\python\.venv\Scripts\python.exe" examples/quantum_algorithms_demo.py

Pop-Location

Write-Host "`n✅ All tests complete!" -ForegroundColor Green
