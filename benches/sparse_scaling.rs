use criterion::{Criterion, criterion_group, criterion_main};
use engine::hybrid_state::HybridQState;
use engine::quantum::QGate;

fn benchmark_60_qubit_sparse_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_scaling");

    // 60 qubits would be 18 Exabytes in dense storage.
    // If this runs, it proves our sparse implementation works.
    group.bench_function("hybrid_state_60_qubits_X_gate", |b| {
        b.iter(|| {
            let mut state = HybridQState::new(60);

            // X gate flips bits but preserves sparsity (1 non-zero amp becomes 1 non-zero amp)
            state.apply_gate(&QGate::X(59));
            state.apply_gate(&QGate::X(0));

            // Verify implicitly by not crashing and finishing
            assert_eq!(
                state.cpu_state.block_count(),
                1,
                "Should stay in 1 block for X gates on zero state"
            );
        })
    });

    // Test growing sparsity: H gate on a few qubits creates superposition
    // but only locally or creating a few blocks
    group.bench_function("hybrid_state_60_qubits_limited_superposition", |b| {
        b.iter(|| {
            let mut state = HybridQState::new(60);

            // H on qubit 0 (inside block 0) -> 1 block, 2 entries
            state.apply_gate(&QGate::H(0));

            // H on qubit 59 (far away) -> Creates a second block?
            // Actually H on 0 is |+...0>.
            // H on 59 makes it (|0>+|1>) x |+...0>.
            // Since we start at |0...0>, H(59) creates superposition at the highest bit.
            // This copies Block 0 to Block (1<<52ish).
            // This is the true test of the hash map storage.
            state.apply_gate(&QGate::H(59));

            // We expect 2 blocks now (or more depending on implementation details)
            // Block 0: ...0...0
            // Block N: ...1...0
            assert!(state.cpu_state.block_count() >= 2);
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_60_qubit_sparse_ops);
criterion_main!(benches);
