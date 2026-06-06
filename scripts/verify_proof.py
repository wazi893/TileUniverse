
import subprocess
import json
import sys
import os

def run_proof(mode):
    print(f"Running Proof Engine [{mode}]...")
    cmd = [
        "cargo", "run", "--example", "proof_engine", "--quiet",
        "--features", "proof-mode cuda",
        "--",
        "--scenario", "h_1million",
        "--output", f"{mode}.json"
    ]
    if mode == "naive":
        cmd.append("--disable-algebraic-reduction")
    
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print(f"Error running {mode} mode")
        sys.exit(1)

def compare():
    print("Comparing results...")
    try:
        with open("optimized.json") as f:
            opt = json.load(f)
        with open("naive.json") as f:
            naive = json.load(f)
    except FileNotFoundError:
        print("Error: JSON output files not found")
        sys.exit(1)

    t_opt = opt["exec_time_sec"]
    t_naive = naive["exec_time_sec"]
    
    print(f"Optimized Time: {t_opt:.6f} s")
    print(f"Naive Time:     {t_naive:.6f} s")
    if t_opt > 0:
        speedup = t_naive / t_opt 
        print(f"Speedup:        {speedup:.2f}x")
    else:
         print(f"Speedup:        Inf")

    # Verify correctness (First 8 amplitudes)
    # H^1M = I. |0> -> |0>. Amplitudes: [1, 0, 0...]
    match = True
    print("Checking amplitudes:")
    for i, (a1, a2) in enumerate(zip(opt["amplitudes"], naive["amplitudes"])):
        r1, i1 = a1
        r2, i2 = a2
        diff = abs(r1 - r2) + abs(i1 - i2)
        if diff > 1e-4:
            print(f"  MISMATCH at index {i}: Opt={a1}, Naive={a2}, Diff={diff}")
            match = False
        else:
            if i < 3: # Print first few matches
                 print(f"  Match at index {i}: Opt={a1}, Naive={a2}")
    
    if match:
        print("SUCCESS: State vectors match within tolerance.")
    else:
        print("FAILURE: State vectors diverge.")
        sys.exit(1)

if __name__ == "__main__":
    if not os.path.exists("target/debug/examples/proof_engine.exe") and not os.path.exists("target/release/examples/proof_engine.exe"):
         print("Warning: verification might fail if cargo build hasn't run.")
         
    run_proof("optimized")
    run_proof("naive")
    compare()
