/// This bypasses the broken Weyl decomposition by directly searching for angles.
/// Uses a multi-phase search with local refinement for efficiency.
fn synthesize_two_cnot_template(target: &Matrix4x4, qa: u8, qb: u8) -> Option<Vec<QGate>> {
    use std::f32::consts::PI;

    // Sparse grid for initial search
    let sparse: [f32; 8] = [0.0, PI/4.0, PI/2.0, 3.0*PI/4.0, PI, -3.0*PI/4.0, -PI/2.0, -PI/4.0];

    let mut best_fidelity = 0.0f32;
    let mut best_params: Option<([f32; 3], [f32; 3], [f32; 2], [f32; 2], [f32; 3], [f32; 3])> = None;

    // Helper to evaluate a parameter set
    let eval = |ba: [f32; 3], bb: [f32; 3], ma: [f32; 2], mb: [f32; 2], aa: [f32; 3], ab: [f32; 3]| {
        let u = build_two_cnot_template_unitary(ba, bb, ma, mb, aa, ab, qa, qb);
        unitary_fidelity(target, &u)
    };

    // Phase 1: Sparse search over all parameters simultaneously
    for &ba0 in &sparse {
        for &ba1 in &sparse {
            for &bb0 in &sparse {
                for &bb1 in &sparse {
                    for &m0 in &sparse {
                        for &m1 in &sparse {
                            let fid = eval(
                                [ba0, ba1, 0.0], [bb0, bb1, 0.0],
                                [m0, m1], [0.0, 0.0],
                                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                            );
                            if fid > best_fidelity {
                                best_fidelity = fid;
                                best_params = Some((
                                    [ba0, ba1, 0.0], [bb0, bb1, 0.0],
                                    [m0, m1], [0.0, 0.0],
                                    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 2: Local refinement around best candidate
    if best_fidelity > 0.1 {
        if let Some((ba, bb, ma, mb, aa, ab)) = best_params {
            let delta = PI / 8.0;
            let refine = [-delta, 0.0, delta];

            for &d_ba0 in &refine {
                for &d_ba1 in &refine {
                    for &d_bb0 in &refine {
                        for &d_bb1 in &refine {
                            for &d_m0 in &refine {
                                for &d_m1 in &refine {
                                    let fid = eval(
                                        [ba[0] + d_ba0, ba[1] + d_ba1, 0.0],
                                        [bb[0] + d_bb0, bb[1] + d_bb1, 0.0],
                                        [ma[0] + d_m0, ma[1] + d_m1],
                                        mb, aa, ab
                                    );
                                    if fid > best_fidelity {
                                        best_fidelity = fid;
                                        best_params = Some((
                                            [ba[0] + d_ba0, ba[1] + d_ba1, 0.0],
                                            [bb[0] + d_bb0, bb[1] + d_bb1, 0.0],
                                            [ma[0] + d_m0, ma[1] + d_m1],
                                            mb, aa, ab
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 3: Search middle_b and after gates
    if best_fidelity > 0.3 {
        if let Some((ba, bb, ma, _, _, _)) = best_params {
            for &mb0 in &sparse {
                for &mb1 in &sparse {
                    for &aa0 in &sparse {
                        for &aa1 in &sparse {
                            for &ab0 in &sparse {
                                for &ab1 in &sparse {
                                    let fid = eval(
                                        ba, bb, ma,
                                        [mb0, mb1],
                                        [aa0, aa1, 0.0], [ab0, ab1, 0.0]
                                    );
                                    if fid > best_fidelity {
                                        best_fidelity = fid;
                                        best_params = Some((
                                            ba, bb, ma,
                                            [mb0, mb1],
                                            [aa0, aa1, 0.0], [ab0, ab1, 0.0]
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 4: Final local refinement
    if best_fidelity > 0.5 {
        if let Some((ba, bb, ma, mb, aa, ab)) = best_params {
            let delta = PI / 16.0;
            let refine = [-delta, 0.0, delta];

            for &d0 in &refine {
                for &d1 in &refine {
                    for &d2 in &refine {
                        for &d3 in &refine {
                            let fid = eval(
                                ba, bb, ma, mb,
                                [aa[0] + d0, aa[1] + d1, 0.0],
                                [ab[0] + d2, ab[1] + d3, 0.0]
                            );
                            if fid > best_fidelity {
                                best_fidelity = fid;
                                best_params = Some((
                                    ba, bb, ma, mb,
                                    [aa[0] + d0, aa[1] + d1, 0.0],
                                    [ab[0] + d2, ab[1] + d3, 0.0]
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 5: Try with third Rz gates (full ZYZ)
    if best_fidelity > 0.7 && best_fidelity < 0.98 {
        if let Some((ba, bb, ma, mb, aa, ab)) = best_params {
            for &ba2 in &sparse {
                for &bb2 in &sparse {
                    for &aa2 in &sparse {
                        for &ab2 in &sparse {
                            let fid = eval(
                                [ba[0], ba[1], ba2],
                                [bb[0], bb[1], bb2],
                                ma, mb,
                                [aa[0], aa[1], aa2],
                                [ab[0], ab[1], ab2]
                            );
                            if fid > best_fidelity {
                                best_fidelity = fid;
                                best_params = Some((
                                    [ba[0], ba[1], ba2],
                                    [bb[0], bb[1], bb2],
                                    ma, mb,
                                    [aa[0], aa[1], aa2],
                                    [ab[0], ab[1], ab2]
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // Return circuit if good enough (threshold 0.98 for numerical tolerance)
    if best_fidelity > 0.98 {
        if let Some((before_a, before_b, mid_a, mid_b, after_a, after_b)) = best_params {
            let mut gates = Vec::new();

            if before_a[0].abs() > 1e-6 { gates.push(QGate::Rz(qa, before_a[0])); }
            if before_a[1].abs() > 1e-6 { gates.push(QGate::Ry(qa, before_a[1])); }
            if before_a[2].abs() > 1e-6 { gates.push(QGate::Rz(qa, before_a[2])); }
            if before_b[0].abs() > 1e-6 { gates.push(QGate::Rz(qb, before_b[0])); }
            if before_b[1].abs() > 1e-6 { gates.push(QGate::Ry(qb, before_b[1])); }
            if before_b[2].abs() > 1e-6 { gates.push(QGate::Rz(qb, before_b[2])); }

            gates.push(QGate::CNot(qa, qb));

            if mid_a[0].abs() > 1e-6 { gates.push(QGate::Rz(qa, mid_a[0])); }
            if mid_a[1].abs() > 1e-6 { gates.push(QGate::Ry(qa, mid_a[1])); }
            if mid_b[0].abs() > 1e-6 { gates.push(QGate::Rz(qb, mid_b[0])); }
            if mid_b[1].abs() > 1e-6 { gates.push(QGate::Ry(qb, mid_b[1])); }

            gates.push(QGate::CNot(qa, qb));

            if after_a[0].abs() > 1e-6 { gates.push(QGate::Rz(qa, after_a[0])); }
            if after_a[1].abs() > 1e-6 { gates.push(QGate::Ry(qa, after_a[1])); }
            if after_a[2].abs() > 1e-6 { gates.push(QGate::Rz(qa, after_a[2])); }
            if after_b[0].abs() > 1e-6 { gates.push(QGate::Rz(qb, after_b[0])); }
            if after_b[1].abs() > 1e-6 { gates.push(QGate::Ry(qb, after_b[1])); }
            if after_b[2].abs() > 1e-6 { gates.push(QGate::Rz(qb, after_b[2])); }

            return Some(gates);
        }
    }

    None
}
