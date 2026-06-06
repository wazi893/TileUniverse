use engine::synth::benchmark::{benchmark_specs, run_synth_benchmarks, verify_aig_equivalence};
use engine::synth::mapping::verify_equivalence;
use engine::synth::{CellLibrary, OptConfig, SynthConfig, optimize, synthesize};

fn main() {
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

    for result in run_synth_benchmarks(&lib) {
        println!("{result}");
    }
}
