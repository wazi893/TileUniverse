#!/bin/bash
# QASMBench Optimization Benchmark
# Runs TileUniverse optimizer on QASMBench circuits

OPTIMIZER="./target/release/opt_qasm.exe"
QASMBENCH_DIR="./QASMBench"
RESULTS_FILE="qasmbench_results.csv"

echo "benchmark,category,original_gates,optimized_gates,reduction_pct" > $RESULTS_FILE

total_original=0
total_optimized=0
count=0

# Function to run optimizer on a circuit
run_benchmark() {
    local qasm_file=$1
    local category=$2
    local name=$(basename $(dirname $qasm_file))

    # Create temp output file
    local out_file="/tmp/opt_${name}.qasm"

    # Run optimizer and capture output
    output=$($OPTIMIZER "$qasm_file" "$out_file" 2>&1)

    # Extract gate counts from output
    original=$(echo "$output" | grep "Original gates:" | awk '{print $3}')
    final=$(echo "$output" | grep "Final gates:" | awk '{print $3}')

    if [ -n "$original" ] && [ -n "$final" ] && [ "$original" -gt 0 ]; then
        reduction=$(echo "scale=2; 100 * (1 - $final / $original)" | bc)
        echo "$name,$category,$original,$final,$reduction" >> $RESULTS_FILE
        echo "  $name: $original -> $final ($reduction%)"

        # Accumulate totals
        total_original=$((total_original + original))
        total_optimized=$((total_optimized + final))
        count=$((count + 1))
    else
        echo "  $name: PARSE ERROR or 0 gates"
    fi

    # Cleanup
    rm -f "$out_file"
}

echo "========================================"
echo "QASMBench Optimization Benchmark"
echo "========================================"
echo ""

# Run on small benchmarks
echo "=== SMALL BENCHMARKS (2-10 qubits) ==="
for dir in $QASMBENCH_DIR/small/*/; do
    qasm_file=$(find "$dir" -name "*.qasm" -type f | head -1)
    if [ -n "$qasm_file" ]; then
        run_benchmark "$qasm_file" "small"
    fi
done

echo ""
echo "=== MEDIUM BENCHMARKS (11-27 qubits) ==="
for dir in $QASMBENCH_DIR/medium/*/; do
    qasm_file=$(find "$dir" -name "*.qasm" -type f | head -1)
    if [ -n "$qasm_file" ]; then
        run_benchmark "$qasm_file" "medium"
    fi
done

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Benchmarks run: $count"
echo "Total original gates: $total_original"
echo "Total optimized gates: $total_optimized"
if [ $total_original -gt 0 ]; then
    overall=$(echo "scale=2; 100 * (1 - $total_optimized / $total_original)" | bc)
    echo "Overall reduction: $overall%"
fi
echo ""
echo "Results saved to: $RESULTS_FILE"
