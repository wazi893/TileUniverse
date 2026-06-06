# TileUniverse vs Qiskit Aer Reproducibility Script
# =================================================
#
# This script reproduces the benchmark comparison between TileUniverse and Qiskit Aer.
# Run from the engine directory: .\scripts\reproduce_benchmark.ps1
#
# Requirements:
# - Windows with NVIDIA GPU (RTX 5090 for reported results)
# - WSL2 with Ubuntu 24.04 and qiskit-aer-gpu installed
# - Rust toolchain with CUDA features

param(
    [switch]$SkipBuild,
    [switch]$TileUniverseOnly,
    [switch]$AerOnly
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$engineDir = Split-Path -Parent $scriptDir

Write-Host "=" * 70
Write-Host "  TILEUNIVERSE vs QISKIT AER REPRODUCIBILITY BENCHMARK"
Write-Host "=" * 70
Write-Host ""

# Check CUDA availability
Write-Host "Checking CUDA installation..."
$cudaPath = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue
if ($cudaPath) {
    Write-Host "  CUDA found: $($cudaPath[0].FullName)"
} else {
    Write-Host "  WARNING: CUDA not found in default location"
}

# Check WSL2 availability for Aer
Write-Host "Checking WSL2..."
$wslAvailable = $false
try {
    $wslList = wsl --list --quiet 2>$null
    if ($wslList -match "Ubuntu") {
        Write-Host "  WSL2 Ubuntu found"
        $wslAvailable = $true
    }
} catch {
    Write-Host "  WSL2 not available"
}

Write-Host ""

# =============================================
# TILEUNIVERSE BENCHMARK
# =============================================
if (-not $AerOnly) {
    Write-Host "=" * 70
    Write-Host "  TILEUNIVERSE BENCHMARK (PureMMA Kernel)"
    Write-Host "=" * 70
    Write-Host ""

    Push-Location $engineDir
    try {
        if (-not $SkipBuild) {
            Write-Host "Building TileUniverse benchmark..."
            cargo build --example rtx5090_benchmark --features cuda,perf-bench --release 2>&1 | Out-Host
            if ($LASTEXITCODE -ne 0) {
                throw "Build failed"
            }
        }

        Write-Host ""
        Write-Host "Running TileUniverse benchmark..."
        Write-Host ""

        $tuOutput = & cargo run --example rtx5090_benchmark --features cuda,perf-bench --release 2>&1
        $tuOutput | Out-Host

        # Save results
        $tuOutput | Out-File -FilePath "$engineDir\tileuniverse_benchmark_results.txt" -Encoding utf8
        Write-Host ""
        Write-Host "Results saved to: tileuniverse_benchmark_results.txt"

    } finally {
        Pop-Location
    }
    Write-Host ""
}

# =============================================
# QISKIT AER BENCHMARK (via WSL2)
# =============================================
if (-not $TileUniverseOnly -and $wslAvailable) {
    Write-Host "=" * 70
    Write-Host "  QISKIT AER BENCHMARK (cuStateVec via WSL2)"
    Write-Host "=" * 70
    Write-Host ""

    # Create a temporary Python script for WSL
    $aerBenchScript = @'
#!/usr/bin/env python3
import time
import sys

def benchmark():
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    print("Qiskit Aer GPU Benchmark (cuStateVec)")
    print("=" * 50)

    try:
        sim = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
        print(f"Simulator: {sim}")
    except Exception as e:
        print(f"GPU not available: {e}")
        print("Falling back to CPU...")
        sim = AerSimulator(method='statevector', device='CPU')

    configs = [(12, 1000), (16, 1000), (20, 1000), (24, 500), (26, 500), (28, 500)]

    print(f"\n{'Qubits':>8} | {'Depth':>8} | {'Time (s)':>10} | {'TCOPS':>10}")
    print("-" * 50)

    for n_qubits, depth in configs:
        try:
            qc = QuantumCircuit(n_qubits)
            for _ in range(depth):
                for q in range(n_qubits):
                    qc.h(q)
                for q in range(0, n_qubits - 1, 2):
                    qc.cx(q, q + 1)

            qc.save_statevector()

            # Warmup
            for _ in range(2):
                job = sim.run(qc, shots=1)
                _ = job.result().get_statevector()

            # Timed run
            start = time.perf_counter()
            job = sim.run(qc, shots=1)
            sv = job.result().get_statevector()
            elapsed = time.perf_counter() - start

            gates = qc.size()
            amp_ops = (2 ** n_qubits) * gates
            tcops = amp_ops / elapsed / 1e12

            print(f"{n_qubits:>8} | {depth:>8} | {elapsed:>10.4f} | {tcops:>10.3f}")
        except Exception as e:
            print(f"{n_qubits:>8} | {depth:>8} | ERROR: {e}")

if __name__ == "__main__":
    benchmark()
'@

    # Write script to temp location accessible by WSL
    $tempScript = "$env:TEMP\aer_bench_temp.py"
    $aerBenchScript | Out-File -FilePath $tempScript -Encoding utf8 -NoNewline

    # Convert Windows path to WSL path
    $wslTempPath = "/mnt/c" + ($tempScript.Substring(2) -replace '\\', '/')

    Write-Host "Running Aer benchmark in WSL2..."
    Write-Host ""

    # Run in WSL with qiskit environment
    $aerOutput = wsl -d Ubuntu-24.04 bash -c "source /opt/qiskit_env/bin/activate 2>/dev/null && python3 '$wslTempPath'" 2>&1
    $aerOutput | Out-Host

    # Save results
    $aerOutput | Out-File -FilePath "$engineDir\aer_benchmark_results.txt" -Encoding utf8
    Write-Host ""
    Write-Host "Results saved to: aer_benchmark_results.txt"

    # Cleanup
    Remove-Item $tempScript -ErrorAction SilentlyContinue

    Write-Host ""
}

# =============================================
# SUMMARY
# =============================================
Write-Host "=" * 70
Write-Host "  BENCHMARK COMPLETE"
Write-Host "=" * 70
Write-Host ""
Write-Host "Results files:"
if (-not $AerOnly) {
    Write-Host "  - tileuniverse_benchmark_results.txt"
}
if (-not $TileUniverseOnly -and $wslAvailable) {
    Write-Host "  - aer_benchmark_results.txt"
}
Write-Host ""
Write-Host "For full comparison, see: docs/VERIFIED_AER_COMPARISON.md"
Write-Host ""
