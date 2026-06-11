use engine::hybrid_state::HybridQState;
use engine::quantum::QGate;

#[test]
fn test_hybrid_cpu_flow() {
    let mut state = HybridQState::new(10); // 10 qubits

    // Apply H(0) on CPU (threshold is high, so it defaults to CPU)
    // state is |0...0> -> H(0)|0...0> = |+...0>
    state.apply_gate(&QGate::H(0));

    let cpu_state = &state.cpu_state;
    assert_eq!(
        cpu_state.block_count(),
        1,
        "Should have 1 block active (Block 0)"
    );

    // In block 0: |0...0> (idx 0) and |1...0> (idx 1).
    // H(0) on |0> -> (|0> + |1>) / sqrt(2)
    let block = cpu_state.get_block(0).unwrap();
    let amp0 = block.get(0);
    let amp1 = block.get(1);

    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    assert!((amp0.re.to_f32() - inv_sqrt2).abs() < 1e-4);
    assert!((amp1.re.to_f32() - inv_sqrt2).abs() < 1e-4);
}
