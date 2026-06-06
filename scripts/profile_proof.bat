
@echo off
echo Starting Nsight Systems Profiling for Proof Engine...
echo Requirement: NVIDIA Nsight Systems must be installed and in PATH (nsys).

rem Profile the Optimized Logic (Should show near zero kernel launches)
echo Profiling Optimized Run...
nsys profile --trace=cuda,osrt,nvtx --output=proof_optimized --force-overwrite=true cargo run --example proof_engine --release --features "proof-mode cuda" -- --scenario h_1million --enable-algebraic-reduction

rem Profile the Naive Logic (Should show many kernel launches)
echo Profiling Naive Run (Reduction Disabled)...
nsys profile --trace=cuda,osrt,nvtx --output=proof_naive --force-overwrite=true cargo run --example proof_engine --release --features "proof-mode cuda" -- --scenario h_1million --disable-algebraic-reduction

echo Profiling Complete. Open proof_optimized.nsys-rep and proof_naive.nsys-rep in Nsight Systems to compare.
