//! Representative benchmark circuits and synth metric harness.
//!
//! This module is intentionally small and deterministic. It gives us a repeatable
//! baseline for measuring `optimize()` and `synthesize(... optimize=true)` on a
//! handful of circuits that exercise arithmetic, deep AND chains, and mux-heavy
//! control logic.

use super::mapping::evaluate_aig;
use super::{Aig, AigLit, CellLibrary, SynthConfig, synthesize};

/// One benchmark case in the synth regression suite.
#[derive(Clone, Copy, Debug)]
pub struct SynthBenchmarkSpec {
    pub name: &'static str,
    pub build: fn() -> Aig,
}

/// Summary metrics for one benchmark case.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthBenchmarkResult {
    pub name: &'static str,
    pub num_inputs: u32,
    pub num_outputs: usize,
    pub raw_and_nodes: usize,
    pub opt_and_nodes: usize,
    pub raw_depth: u32,
    pub opt_depth: u32,
    pub raw_mapped_gates: usize,
    pub opt_mapped_gates: usize,
}

impl std::fmt::Display for SynthBenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: AIG and {}->{}, depth {}->{}, mapped {}->{}",
            self.name,
            self.raw_and_nodes,
            self.opt_and_nodes,
            self.raw_depth,
            self.opt_depth,
            self.raw_mapped_gates,
            self.opt_mapped_gates,
        )
    }
}

/// The small benchmark suite used for synth hardening.
pub fn benchmark_specs() -> Vec<SynthBenchmarkSpec> {
    vec![
        SynthBenchmarkSpec {
            name: "adder4",
            build: build_4bit_adder,
        },
        SynthBenchmarkSpec {
            name: "eq8",
            build: build_8bit_eq_comparator,
        },
        SynthBenchmarkSpec {
            name: "adder8",
            build: build_8bit_adder,
        },
        SynthBenchmarkSpec {
            name: "mul4",
            build: build_4bit_multiplier,
        },
        SynthBenchmarkSpec {
            name: "mux_ctrl",
            build: build_mux_control_block,
        },
        SynthBenchmarkSpec {
            name: "adder16",
            build: build_16bit_adder,
        },
        SynthBenchmarkSpec {
            name: "decoder4to16",
            build: build_decoder4to16,
        },
        SynthBenchmarkSpec {
            name: "prienc8",
            build: build_priority_encoder8,
        },
    ]
}

/// Run the benchmark suite and collect before/after metrics.
pub fn run_synth_benchmarks(lib: &CellLibrary) -> Vec<SynthBenchmarkResult> {
    benchmark_specs()
        .into_iter()
        .map(|spec| {
            let aig = (spec.build)();
            let raw = synthesize(&aig, lib, &SynthConfig::default());
            let opt = synthesize(
                &aig,
                lib,
                &SynthConfig {
                    optimize: true,
                    ..SynthConfig::default()
                },
            );

            SynthBenchmarkResult {
                name: spec.name,
                num_inputs: raw.stats.aig_inputs,
                num_outputs: raw.stats.aig_outputs,
                raw_and_nodes: raw.stats.aig_and_nodes,
                opt_and_nodes: opt.stats.aig_and_nodes,
                raw_depth: raw.stats.aig_depth,
                opt_depth: opt.stats.aig_depth,
                raw_mapped_gates: raw.stats.mapped_gates,
                opt_mapped_gates: opt.stats.mapped_gates,
            }
        })
        .collect()
}

/// Golden baseline metrics for synth benchmark regression testing.
///
/// Format: (name, [inputs, outputs, raw_and, opt_and, raw_depth, opt_depth, raw_mapped, opt_mapped])
/// Captured at Sprint 192 with NPN4 multi-variable Shannon expansion (76% coverage).
pub const SYNTH_BENCHMARK_GOLDENS: [(&str, [usize; 8]); 8] = [
    ("adder4", [8, 5, 31, 31, 8, 8, 17, 17]),
    ("eq8", [16, 1, 31, 31, 9, 5, 15, 15]),
    ("adder8", [16, 9, 67, 67, 16, 16, 37, 37]),
    ("mul4", [8, 8, 104, 104, 21, 21, 64, 64]),
    ("mux_ctrl", [14, 2, 30, 30, 10, 10, 30, 30]),
    ("adder16", [32, 17, 139, 139, 32, 32, 77, 77]),
    ("decoder4to16", [4, 16, 28, 28, 3, 3, 28, 28]),
    ("prienc8", [8, 4, 28, 24, 8, 7, 25, 23]),
];

/// Validate benchmark results against golden baselines.
/// Returns a list of mismatch descriptions (empty = all pass).
pub fn validate_synth_benchmarks(results: &[SynthBenchmarkResult]) -> Vec<String> {
    let mut mismatches = Vec::new();
    for (name, expected) in &SYNTH_BENCHMARK_GOLDENS {
        if let Some(r) = results.iter().find(|r| r.name == *name) {
            let actual = [
                r.num_inputs as usize,
                r.num_outputs,
                r.raw_and_nodes,
                r.opt_and_nodes,
                r.raw_depth as usize,
                r.opt_depth as usize,
                r.raw_mapped_gates,
                r.opt_mapped_gates,
            ];
            if actual != *expected {
                mismatches.push(format!(
                    "{}: expected {:?}, got {:?}",
                    name, expected, actual
                ));
            }
        } else {
            mismatches.push(format!("{}: missing from results", name));
        }
    }
    mismatches
}

/// Verify that two AIGs are functionally equivalent.
///
/// Exhaustive for n <= 20 inputs. For larger circuits, uses deterministic
/// sampling (boundary + stride-based patterns, 10000 vectors).
pub fn verify_aig_equivalence(lhs: &Aig, rhs: &Aig) -> bool {
    let n = lhs.num_inputs() as usize;
    assert_eq!(
        lhs.num_inputs(),
        rhs.num_inputs(),
        "AIG equivalence check requires identical input counts"
    );
    assert_eq!(
        lhs.num_output_bits(),
        rhs.num_output_bits(),
        "AIG equivalence check requires identical output counts"
    );

    if n <= 20 {
        for assignment in 0..(1u32 << n) {
            let inputs: Vec<bool> = (0..n).map(|i| (assignment >> i) & 1 != 0).collect();
            let lhs_out = evaluate_aig(lhs, &inputs);
            let rhs_out = evaluate_aig(rhs, &inputs);
            if lhs_out != rhs_out {
                eprintln!(
                    "AIG mismatch at input {:0width$b}: lhs={:?} rhs={:?}",
                    assignment,
                    lhs_out,
                    rhs_out,
                    width = n,
                );
                return false;
            }
        }
    } else {
        // Deterministic sampling for large circuits
        let total_combos: u64 = 1u64 << n.min(63);
        let num_samples: u64 = 10_000;
        let stride = total_combos / num_samples;

        for sample in 0..num_samples {
            let assignment = sample * stride;
            let inputs: Vec<bool> = (0..n).map(|i| (assignment >> i) & 1 != 0).collect();
            let lhs_out = evaluate_aig(lhs, &inputs);
            let rhs_out = evaluate_aig(rhs, &inputs);
            if lhs_out != rhs_out {
                eprintln!(
                    "AIG mismatch at input {}: lhs={:?} rhs={:?}",
                    assignment, lhs_out, rhs_out,
                );
                return false;
            }
        }
    }
    true
}

pub fn build_4bit_adder() -> Aig {
    let mut aig = Aig::new();
    let a = aig.add_input_bus("a", 4);
    let b = aig.add_input_bus("b", 4);
    let mut carry = AigLit::FALSE;
    let mut sum_bits = Vec::new();

    for i in 0..4 {
        let axb = aig.xor(a[i], b[i]);
        let sum = aig.xor(axb, carry);
        let ab = aig.and(a[i], b[i]);
        let caxb = aig.and(carry, axb);
        carry = aig.or(ab, caxb);
        sum_bits.push(sum);
    }

    sum_bits.push(carry);
    aig.add_output_bus("sum", &sum_bits);
    aig
}

pub fn build_8bit_adder() -> Aig {
    let mut aig = Aig::new();
    let a = aig.add_input_bus("a", 8);
    let b = aig.add_input_bus("b", 8);
    let mut carry = AigLit::FALSE;
    let mut sum_bits = Vec::new();

    for i in 0..8 {
        let axb = aig.xor(a[i], b[i]);
        let sum = aig.xor(axb, carry);
        let ab = aig.and(a[i], b[i]);
        let caxb = aig.and(carry, axb);
        carry = aig.or(ab, caxb);
        sum_bits.push(sum);
    }

    sum_bits.push(carry);
    aig.add_output_bus("sum", &sum_bits);
    aig
}

/// Equality comparator with a deliberately left-skewed AND chain.
pub fn build_8bit_eq_comparator() -> Aig {
    let mut aig = Aig::new();
    let a = aig.add_input_bus("a", 8);
    let b = aig.add_input_bus("b", 8);
    let mut eq = AigLit::TRUE;

    for i in 0..8 {
        let diff = aig.xor(a[i], b[i]);
        let same = aig.not(diff);
        eq = aig.and(eq, same);
    }

    aig.add_output("eq", eq);
    aig
}

/// 4×4 → 8-bit unsigned multiplier using partial products and ripple-carry addition.
pub fn build_4bit_multiplier() -> Aig {
    let mut aig = Aig::new();
    let a = aig.add_input_bus("a", 4);
    let b = aig.add_input_bus("b", 4);

    // Partial products: pp[i][j] = a[j] & b[i]
    let mut pp = [[AigLit::FALSE; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            pp[i][j] = aig.and(a[j], b[i]);
        }
    }

    // Accumulate shifted partial product rows via ripple-carry addition.
    // Start with row 0 (pp[0], no shift).
    let mut accum = [AigLit::FALSE; 8];
    for j in 0..4 {
        accum[j] = pp[0][j];
    }

    // Add rows 1-3, each shifted left by their row index
    for row in 1..4usize {
        let mut carry = AigLit::FALSE;
        for bit in row..row + 4 {
            let axb = aig.xor(accum[bit], pp[row][bit - row]);
            let sum = aig.xor(axb, carry);
            let ab = aig.and(accum[bit], pp[row][bit - row]);
            let caxb = aig.and(carry, axb);
            carry = aig.or(ab, caxb);
            accum[bit] = sum;
        }
        // Carry out goes to the next bit (row 1→5, row 2→6, row 3→7)
        accum[row + 4] = aig.xor(accum[row + 4], carry);
    }

    aig.add_output_bus("product", &accum);
    aig
}

/// 16-bit ripple-carry adder (32 inputs, 17 outputs).
pub fn build_16bit_adder() -> Aig {
    let mut aig = Aig::new();
    let a = aig.add_input_bus("a", 16);
    let b = aig.add_input_bus("b", 16);
    let mut carry = AigLit::FALSE;
    let mut sum_bits = Vec::new();

    for i in 0..16 {
        let axb = aig.xor(a[i], b[i]);
        let sum = aig.xor(axb, carry);
        let ab = aig.and(a[i], b[i]);
        let caxb = aig.and(carry, axb);
        carry = aig.or(ab, caxb);
        sum_bits.push(sum);
    }

    sum_bits.push(carry);
    aig.add_output_bus("sum", &sum_bits);
    aig
}

/// 4-to-16 decoder: 4 inputs, 16 one-hot outputs.
/// o[i] = AND of s[j]/!s[j] per bit pattern of i.
pub fn build_decoder4to16() -> Aig {
    let mut aig = Aig::new();
    let s = aig.add_input_bus("s", 4);
    let mut outputs = Vec::new();

    for i in 0..16u32 {
        // output[i] = AND of (s[j] if bit j of i is 1, else !s[j])
        let mut term = AigLit::TRUE;
        for j in 0..4 {
            let input = if (i >> j) & 1 != 0 {
                s[j]
            } else {
                aig.not(s[j])
            };
            term = aig.and(term, input);
        }
        outputs.push(term);
    }

    aig.add_output_bus("out", &outputs);
    aig
}

/// 8-bit priority encoder: 8 inputs → 3 encode bits + valid.
/// valid = OR(all inputs). encode = index of highest set bit.
pub fn build_priority_encoder8() -> Aig {
    let mut aig = Aig::new();
    let inp = aig.add_input_bus("in", 8);

    // valid = OR of all inputs
    let mut valid = inp[0];
    for i in 1..8 {
        valid = aig.or(valid, inp[i]);
    }

    // Priority encode: highest set bit wins.
    // enc[2] = any of bits 4-7 set
    // enc[1] = any of bits 2-3 or 6-7 set (when not masked by higher)
    // enc[0] = any of bits 1,3,5,7 set (when not masked by higher)

    // Build "none above" masks for priority:
    // no_above[i] = none of inp[i+1..8] are set
    let mut no_above = [AigLit::TRUE; 8];
    for i in (0..7).rev() {
        // no_above[i] = !inp[i+1] & no_above[i+1]
        let not_next = aig.not(inp[i + 1]);
        no_above[i] = aig.and(not_next, no_above[i + 1]);
    }

    // winner[i] = inp[i] & no_above[i] (exactly one is true: the highest set bit)
    let mut winner = [AigLit::FALSE; 8];
    for i in 0..8 {
        winner[i] = aig.and(inp[i], no_above[i]);
    }

    // enc[0] = winner[1] | winner[3] | winner[5] | winner[7]
    let e0_a = aig.or(winner[1], winner[3]);
    let e0_b = aig.or(winner[5], winner[7]);
    let enc0 = aig.or(e0_a, e0_b);

    // enc[1] = winner[2] | winner[3] | winner[6] | winner[7]
    let e1_a = aig.or(winner[2], winner[3]);
    let e1_b = aig.or(winner[6], winner[7]);
    let enc1 = aig.or(e1_a, e1_b);

    // enc[2] = winner[4] | winner[5] | winner[6] | winner[7]
    let e2_a = aig.or(winner[4], winner[5]);
    let e2_b = aig.or(winner[6], winner[7]);
    let enc2 = aig.or(e2_a, e2_b);

    aig.add_output_bus("enc", &[enc0, enc1, enc2]);
    aig.add_output("valid", valid);
    aig
}

/// Mux-heavy control block for optimizer stress testing.
///
/// Exercises cascaded MUX structures that trigger non-involutory NPN4 permutations
/// in the AIG rewriter. Fixed in Sprint 190 (inv_perm fix in build_subgraph).
pub fn build_mux_control_block() -> Aig {
    let mut aig = Aig::new();
    let sel = aig.add_input_bus("sel", 3);
    let data = aig.add_input_bus("data", 8);
    let enable = aig.add_input("enable");
    let invert = aig.add_input("invert");
    let override_flag = aig.add_input("override");

    let mux01 = aig.mux(sel[0], data[1], data[0]);
    let mux23 = aig.mux(sel[0], data[3], data[2]);
    let mux45 = aig.mux(sel[0], data[5], data[4]);
    let mux67 = aig.mux(sel[0], data[7], data[6]);
    let low_half = aig.mux(sel[1], mux23, mux01);
    let high_half = aig.mux(sel[1], mux67, mux45);
    let selected = aig.mux(sel[2], high_half, low_half);

    let primary = aig.mux(enable, selected, data[0]);
    let fallback = aig.mux(invert, data[7], data[1]);
    let status = aig.mux(override_flag, fallback, primary);

    aig.add_output("primary", primary);
    aig.add_output("status", status);
    aig
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::mapping::verify_equivalence;
    use crate::synth::{OptConfig, optimize};

    #[test]
    fn benchmark_suite_preserves_equivalence_and_mapping() {
        let lib = CellLibrary::tile_native();

        for spec in benchmark_specs() {
            let aig = (spec.build)();
            let opt = optimize(&aig, &OptConfig::default());
            let opt_result = synthesize(
                &aig,
                &lib,
                &SynthConfig {
                    optimize: true,
                    ..SynthConfig::default()
                },
            );

            assert!(
                verify_aig_equivalence(&aig, &opt),
                "optimized AIG for '{}' is not equivalent to the source AIG",
                spec.name
            );
            assert!(
                verify_equivalence(&aig, &opt_result.netlist, &lib),
                "optimized mapping for '{}' is not equivalent to the source AIG",
                spec.name
            );
            assert!(
                opt.num_nodes() <= aig.num_nodes(),
                "optimize increased node count for '{}': {} -> {}",
                spec.name,
                aig.num_nodes(),
                opt.num_nodes()
            );
            assert!(
                opt.depth() <= aig.depth(),
                "optimize increased depth for '{}': {} -> {}",
                spec.name,
                aig.depth(),
                opt.depth()
            );
        }
    }

    #[test]
    fn mux_control_block_optimization_preserves_equivalence() {
        use crate::synth::rewrite;

        let aig = build_mux_control_block();

        // Rewrite zero-cost mode (the path that was broken pre-Sprint 190)
        let rw_zc = rewrite::rewrite(&aig, true).compact();
        assert!(
            verify_aig_equivalence(&aig, &rw_zc),
            "rewrite(true) broke equivalence on mux control block"
        );

        // Full optimize pipeline
        let opt = optimize(&aig, &OptConfig::default());
        assert!(
            verify_aig_equivalence(&aig, &opt),
            "optimize() broke equivalence on mux control block"
        );

        // Full synthesize with optimization
        let lib = CellLibrary::tile_native();
        let result = synthesize(
            &aig,
            &lib,
            &SynthConfig {
                optimize: true,
                ..SynthConfig::default()
            },
        );
        assert!(
            verify_equivalence(&aig, &result.netlist, &lib),
            "optimized synthesis broke equivalence on mux control block"
        );
    }

    #[test]
    fn rewrite_zero_cost_cascaded_muxes() {
        use crate::synth::rewrite;

        // Two cascaded MUXes sharing inputs — exercises non-involutory NPN4 permutations
        let mut aig = crate::synth::Aig::new();
        let s0 = aig.add_input("sel0");
        let s1 = aig.add_input("sel1");
        let d0 = aig.add_input("d0");
        let d1 = aig.add_input("d1");
        let d2 = aig.add_input("d2");
        let m0 = aig.mux(s0, d1, d0);
        let m1 = aig.mux(s0, d2, d1);
        let _y = aig.mux(s1, m1, m0);
        aig.add_output("y", _y);

        let rw = rewrite::rewrite(&aig, true).compact();
        assert!(
            verify_aig_equivalence(&aig, &rw),
            "rewrite(true) on cascaded muxes broke equivalence"
        );
    }

    #[test]
    fn benchmark_suite_reports_useful_metrics() {
        let results = run_synth_benchmarks(&CellLibrary::tile_native());
        assert_eq!(results.len(), 8);
        assert!(
            results
                .iter()
                .any(|r| r.opt_and_nodes < r.raw_and_nodes || r.opt_depth < r.raw_depth),
            "expected at least one benchmark to improve under optimize()"
        );

        let eq8 = results
            .iter()
            .find(|r| r.name == "eq8")
            .expect("missing eq8 benchmark");
        assert!(
            eq8.opt_depth < eq8.raw_depth,
            "eq8 should benefit from balancing: {} -> {}",
            eq8.raw_depth,
            eq8.opt_depth
        );
    }

    #[test]
    fn synth_benchmark_golden_regression() {
        let results = run_synth_benchmarks(&CellLibrary::tile_native());
        let mismatches = validate_synth_benchmarks(&results);
        assert!(
            mismatches.is_empty(),
            "Golden benchmark regression:\n{}",
            mismatches.join("\n")
        );
    }

    #[test]
    fn adder16_sample_correctness() {
        use crate::synth::mapping::evaluate_aig;

        let aig = build_16bit_adder();
        assert_eq!(aig.num_inputs(), 32);
        assert_eq!(aig.num_output_bits(), 17);

        // Sample check: boundary + random values
        let test_cases: Vec<(u32, u32)> = vec![
            (0, 0),
            (1, 0),
            (0, 1),
            (0xFFFF, 0),
            (0, 0xFFFF),
            (0xFFFF, 0xFFFF),
            (0xFFFF, 1),
            (1, 0xFFFF),
            (0x5555, 0xAAAA),
            (0x1234, 0x5678),
            (0x8000, 0x8000),
            (0x7FFF, 0x7FFF),
            (42, 58),
            (1000, 2000),
            (0xDEAD, 0xBEEF),
        ];

        for &(a_val, b_val) in &test_cases {
            let mut inputs = vec![false; 32];
            for i in 0..16 {
                inputs[i] = (a_val >> i) & 1 != 0;
                inputs[16 + i] = (b_val >> i) & 1 != 0;
            }
            let outputs = evaluate_aig(&aig, &inputs);
            let mut result = 0u32;
            for (i, &bit) in outputs.iter().enumerate() {
                if bit {
                    result |= 1 << i;
                }
            }
            let expected = a_val + b_val;
            assert_eq!(
                result, expected,
                "adder16({}, {}) = {} (expected {})",
                a_val, b_val, result, expected
            );
        }
    }

    #[test]
    fn decoder4to16_exhaustive() {
        use crate::synth::mapping::evaluate_aig;

        let aig = build_decoder4to16();
        assert_eq!(aig.num_inputs(), 4);
        assert_eq!(aig.num_output_bits(), 16);

        for combo in 0..16u32 {
            let inputs: Vec<bool> = (0..4).map(|i| (combo >> i) & 1 != 0).collect();
            let outputs = evaluate_aig(&aig, &inputs);

            // Exactly one output should be high (one-hot)
            let mut active_count = 0;
            let mut active_idx = 0;
            for (i, &bit) in outputs.iter().enumerate() {
                if bit {
                    active_count += 1;
                    active_idx = i;
                }
            }
            assert_eq!(
                active_count, 1,
                "decoder4to16 input {} should produce one-hot output, got {} active",
                combo, active_count
            );
            assert_eq!(
                active_idx, combo as usize,
                "decoder4to16 input {} should activate output {}, got {}",
                combo, combo, active_idx
            );
        }
    }

    #[test]
    fn priority_encoder8_exhaustive() {
        use crate::synth::mapping::evaluate_aig;

        let aig = build_priority_encoder8();
        assert_eq!(aig.num_inputs(), 8);
        assert_eq!(aig.num_output_bits(), 4); // 3 enc + 1 valid

        for combo in 0..256u32 {
            let inputs: Vec<bool> = (0..8).map(|i| (combo >> i) & 1 != 0).collect();
            let outputs = evaluate_aig(&aig, &inputs);

            let enc = (outputs[0] as u32) | ((outputs[1] as u32) << 1) | ((outputs[2] as u32) << 2);
            let valid = outputs[3];

            if combo == 0 {
                assert!(!valid, "prienc8: input 0 should produce valid=false");
            } else {
                assert!(valid, "prienc8: input {} should produce valid=true", combo);
                // Find highest set bit
                let highest = 31 - combo.leading_zeros();
                assert_eq!(
                    enc, highest,
                    "prienc8: input {} highest bit {} but enc={}",
                    combo, highest, enc
                );
            }
        }
    }

    #[test]
    fn multiplier_functional_correctness() {
        use crate::synth::mapping::evaluate_aig;

        let aig = build_4bit_multiplier();
        assert_eq!(aig.num_inputs(), 8);
        assert_eq!(aig.num_output_bits(), 8);

        // Exhaustive check: all 256 input pairs
        for a_val in 0..16u32 {
            for b_val in 0..16u32 {
                let mut inputs = vec![false; 8];
                for i in 0..4 {
                    inputs[i] = (a_val >> i) & 1 != 0;
                    inputs[4 + i] = (b_val >> i) & 1 != 0;
                }
                let outputs = evaluate_aig(&aig, &inputs);
                let mut result = 0u32;
                for (i, &bit) in outputs.iter().enumerate() {
                    if bit {
                        result |= 1 << i;
                    }
                }
                assert_eq!(
                    result,
                    a_val * b_val,
                    "mul4({}, {}) = {} (expected {})",
                    a_val,
                    b_val,
                    result,
                    a_val * b_val
                );
            }
        }
    }
}
