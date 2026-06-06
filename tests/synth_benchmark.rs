use engine::synth::benchmark::{
    benchmark_specs, run_synth_benchmarks, validate_synth_benchmarks, verify_aig_equivalence,
};
use engine::synth::mapping::verify_equivalence;
use engine::synth::{CellLibrary, OptConfig, SynthConfig, optimize, synthesize};

#[test]
fn synth_benchmark_suite_preserves_equivalence_and_mapping() {
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
fn synth_benchmark_suite_reports_useful_metrics() {
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

    for result in results {
        eprintln!("{result}");
    }
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
