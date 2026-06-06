# Qiskit Synthesis Cost Analysis

**Date:** 2025-12-27
**Sprint:** 37.0
**Purpose:** Quantify Qiskit synthesis overhead vs native gate execution to inform cost model decisions

## Benchmark Results

### 1. Qiskit Subprocess Overhead
- **Average synthesis time:** 1.6 seconds per call
- **Total for 20 calls:** 32 seconds
- **Breakdown:**
  - Python interpreter startup: ~1.5s
  - Qiskit module import: ~100-200ms
  - Actual synthesis: <100ms

### 2. Gate Execution Performance (10-qubit state)
- **CNot gate:** 1,261 ns/gate
- **Rz gate:** 3,207 ns/gate
- **Synthesized iSWAP (18 gates):** 47.1 μs/circuit

## Key Insights

### Cost Ratio
```
Synthesis overhead:  1,607,000 μs  (1.6 seconds)
Execute 18 gates:           47 μs  (synthesized circuit)
Overhead ratio:         ~34,000x
```

### Break-Even Analysis
```
Break-even = Synthesis cost / Execution cost
           = 1,607,000 μs / 47 μs
           = ~34,000 executions
```

**Conclusion:** Synthesis only makes sense if the same gate will be executed **>34,000 times**.

## Strategic Implications

### When to Use Synthesis

✅ **DO synthesize when:**
1. Hardware doesn't support the gate natively (e.g., iSWAP on CNot-only hardware)
2. Circuit will be executed tens of thousands of times (RL training, parameter sweeps)
3. Synthesis result can be cached and reused
4. Gate appears in a hot loop

❌ **DON'T synthesize when:**
1. One-shot circuit execution
2. Interactive/exploratory quantum programming
3. Small circuits (<100 gates total)
4. Hardware supports the gate natively

### Optimization Priorities

**1. Circuit Caching** (HIGHEST IMPACT)
- Cache synthesized circuits for common gates (iSWAP, SWAP, CZ, etc.)
- Store as `HashMap<GateType, Vec<QGate>>`
- Reduces 1.6s overhead to ~0μs for repeated gates
- **Impact:** Makes synthesis practical for all use cases

**2. Lazy Synthesis**
- Don't synthesize until execution is needed
- Allow users to build circuits without synthesis overhead
- Only synthesize when calling `.execute()` or `.run()`

**3. Batch Synthesis**
- If multiple unique gates need synthesis, batch them into single Python call
- Amortize the 1.5s startup overhead across multiple gates
- **Impact:** Could reduce per-gate cost from 1.6s to ~200ms

**4. Native Gate Preference**
- Check hardware capabilities before synthesis
- Use native gates when available
- **Impact:** Avoid synthesis entirely for supported gates

## Recommended Implementation

### Phase 1: Circuit Cache (Low-hanging fruit)
```rust
use std::collections::HashMap;
use once_cell::sync::Lazy;

static SYNTHESIS_CACHE: Lazy<Mutex<HashMap<String, Vec<QGate>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn synthesize_cached(unitary: &Matrix4x4, qa: u8, qb: u8) -> Vec<QGate> {
    let key = format_unitary_key(unitary, qa, qb);

    let cache = SYNTHESIS_CACHE.lock().unwrap();
    if let Some(gates) = cache.get(&key) {
        return gates.clone(); // Cache hit
    }
    drop(cache);

    // Cache miss - call Qiskit
    let gates = synthesize_two_cnot_qiskit(unitary, qa, qb)?;

    SYNTHESIS_CACHE.lock().unwrap().insert(key, gates.clone());
    gates
}
```

**Expected improvement:** 1.6s → ~0μs for repeated gates

### Phase 2: Pre-populate Common Gates
```rust
fn init_synthesis_cache() {
    let common_gates = vec![
        ("iSWAP", create_iswap_unitary()),
        ("SWAP", create_swap_unitary()),
        ("CZ", create_cz_unitary()),
        // ... etc
    ];

    for (name, unitary) in common_gates {
        synthesize_cached(&unitary, 0, 1);
    }
}
```

**Expected improvement:** No synthesis overhead for 90% of use cases

### Phase 3: Batch Synthesis (Future)
Modify Python script to accept multiple unitaries:
```python
# Input: {"gates": [{"matrix": ..., "qa": 0, "qb": 1}, ...]}
# Output: {"results": [[gate1, gate2, ...], [gate3, ...], ...]}
```

**Expected improvement:** 20 gates synthesized in 2s instead of 32s (16s savings)

## Cost Model Formula

```rust
fn should_synthesize(gate_type: &GateType, n_executions: usize, hardware_supports: bool) -> bool {
    const SYNTHESIS_COST_US: f64 = 1_607_000.0;
    const EXEC_COST_US: f64 = 47.0;
    const BREAKEVEN: f64 = SYNTHESIS_COST_US / EXEC_COST_US;

    if hardware_supports {
        return false; // Use native gate
    }

    if is_cached(gate_type) {
        return true; // Cached synthesis is free
    }

    n_executions as f64 > BREAKEVEN
}
```

## Next Steps

1. **Implement circuit cache** (this sprint)
   - Add `HashMap` for synthesis results
   - Pre-populate common gates on startup
   - Benchmark cache hit performance

2. **Measure cache effectiveness** (next sprint)
   - Track cache hit rate in real workloads
   - Identify additional gates to pre-cache
   - Quantify performance improvement

3. **Hardware capability detection** (future)
   - Query backend for supported native gates
   - Prefer native execution when available
   - Only synthesize when necessary

4. **Batch synthesis** (future, if needed)
   - Modify Python script for multiple gates
   - Benchmark amortized cost
   - Implement if cache doesn't cover 90%+ of cases

## Conclusion

**Current state:** Synthesis is impractical due to 1.6s overhead
**With caching:** Synthesis becomes instant for repeated gates
**Impact:** Enables correct gate execution without performance penalty

The 34,000x overhead is entirely due to Python subprocess startup. A simple cache eliminates this for all real-world use cases.
