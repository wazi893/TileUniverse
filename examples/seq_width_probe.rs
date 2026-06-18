//! Post-router-fix probe: does the sequential-HLS envelope now extend past
//! width 5? Runs sum_to and gcd FSMs at widths 6 and 8 on real tiles; the
//! run-time AIG audit inside SeqDatapath::run makes any residual place&route
//! divergence a loud error instead of a wrong answer.

use engine::synth::hls::HlsConfig;
use engine::synth::hls_seq::{eval_func_ref, synthesize_seq_func};
use engine::tile_cpu::v2_parser::parse_program;

fn main() {
    let cases: &[(&str, &str, &[&[u64]])] = &[
        (
            "int sum_to(int n) { int s = 0; int i = 1; while (i <= n) { s = s + i; i = i + 1; } return s; }",
            "sum_to",
            &[&[7], &[10], &[15]],
        ),
        (
            "int gcd(int a, int b) { while (a != b) { if (a > b) { a = a - b; } else { b = b - a; } } return a; }",
            "gcd",
            &[&[60, 24], &[100, 75], &[210, 45]],
        ),
    ];

    for width in [6u32, 8] {
        println!("=== width {width} ===");
        for (src, name, arg_sets) in cases {
            let prog = parse_program(src).expect("parse");
            let func = prog.funcs.iter().find(|f| f.name == *name).unwrap();
            match synthesize_seq_func(func, &HlsConfig { width }) {
                Err(e) => println!("  {name}: SYNTH FAILED — {e}"),
                Ok(mut dp) => {
                    for args in *arg_sets {
                        // Skip arg sets that overflow the width.
                        if args.iter().any(|&a| a >= (1 << width)) {
                            continue;
                        }
                        let expect = eval_func_ref(func, args, width).unwrap();
                        match dp.run(args, 5_000) {
                            Ok(out) if out.result == expect => println!(
                                "  {name}{args:?} = {} OK ({} cycles)",
                                out.result, out.cycles
                            ),
                            Ok(out) => println!(
                                "  {name}{args:?}: WRONG — tiles {} != ref {expect}",
                                out.result
                            ),
                            Err(e) => println!("  {name}{args:?}: RUN FAILED — {e}"),
                        }
                    }
                }
            }
        }
    }
}
