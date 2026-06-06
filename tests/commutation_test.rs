use engine::fusion::{FusedOp, FusionConfig, fuse_program_with_config};
use engine::quantum::QGate;

#[test]
fn test_commutation_missed_opportunity() {
    // Circuit: H(0) -> Z(1) -> H(0)
    // Theoretically: H(0) commutes with Z(1). So H(0) -> H(0) -> Z(1) => I -> Z(1) => Z(1)
    // Currently (without commutation): FusedOp::Single(H0), FusedOp::Single(Z1), FusedOp::Single(H0)

    let gates = vec![QGate::H(0), QGate::Z(1), QGate::H(0)];

    let mut config = FusionConfig::default();
    config.enable_algebraic_reduction = true;
    config.enable_mega_fusion = false; // Disable mega fusion to isolate algebraic reduction

    let fused = fuse_program_with_config(&gates, &config);

    println!("Fused Ops: {:?}", fused.ops);

    // WITH Active Commutation implemented, we now expect full reduction of H(0)-H(0).
    // The only remaining gate should be Z(1).
    assert_eq!(
        fused.ops.len(),
        1,
        "Active Commutation should reduce H(0)-Z(1)-H(0) to Z(1)"
    );
    match &fused.ops[0] {
        FusedOp::Single(QGate::Z(1)) => {}
        _ => panic!("Expected Z(1), got {:?}", fused.ops[0]),
    }
}
