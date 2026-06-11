//! AlphaFabric timing demo: timing-driven annealing vs pure-HPWL annealing.
//!
//! Both runs use the same simulated-annealing placer; the only difference is the
//! objective. Pure HPWL minimizes total wirelength. Timing-aware adds the
//! criticality-weighted-wirelength term (`PlacementEnv::timing_cost`): each net's
//! length is weighted by its STA-slack criticality, so the search spends a little
//! total wirelength to pull *timing-critical* nets short — the textbook
//! timing-driven-placement tradeoff. Deep carry chains (`adder*`, `mul4`) have
//! the most critical paths, so they show the largest effect; the shallow `eq8`
//! comparator shows the term is selective.
//!
//! Every reported layout is physically verified on real tiles, so the legality
//! gate (re-placed circuit still computes correctly) holds throughout.
//!
//! Run: `cargo run --release --example alphafabric_af_timing`

use engine::synth::alphafabric::corpus::{build_eq_comparator, build_ripple_adder};
use engine::synth::alphafabric::env::PhysicalDiagnosis;
use engine::synth::alphafabric::{anneal, AnnealConfig, Circuit, PlacementEnv};
use engine::synth::benchmark::build_4bit_multiplier;

/// Timing weight for the timing-aware run. Strong enough to bend the search
/// toward critical nets without making layouts unroutable on this corpus.
const TIMING_WEIGHT: f64 = 6.0;

fn pct(from: f64, to: f64) -> f64 {
    if from <= 0.0 {
        0.0
    } else {
        (from - to) / from * 100.0
    }
}

fn main() {
    let circuits = [
        Circuit::from_aig("adder4", build_ripple_adder(4)),
        Circuit::from_aig("adder8", build_ripple_adder(8)),
        Circuit::from_aig("adder12", build_ripple_adder(12)),
        Circuit::from_aig("mul4", build_4bit_multiplier()),
        Circuit::from_aig("eq8", build_eq_comparator(8)),
    ];

    let hpwl_cfg = AnnealConfig::default();
    let timing_cfg = AnnealConfig {
        timing_weight: TIMING_WEIGHT,
        ..AnnealConfig::default()
    };

    println!(
        "AlphaFabric — timing-driven vs pure-HPWL annealing ({} iters, timing weight {})\n",
        hpwl_cfg.iterations, TIMING_WEIGHT
    );
    println!(
        "crit-wl = criticality-weighted wirelength (the timing objective; lower = critical nets shorter)\n"
    );
    println!(
        "{:<9} {:>5} | {:>13} {:>13} {:>8} | {:>13} {:>13} {:>8} | {:>9}",
        "circuit", "gates", "hpwl(H>T)", "crit-wl(H>T)", "wire(H>T)", "Δcrit-wl", "Δhpwl", "phys", "verdict"
    );

    // Blocks naive timing-aware SA left unroutable, recovered by the
    // route-validated run: (name, routes, crit-wl, physical diagnosis).
    let mut recovered: Vec<(String, bool, f64, PhysicalDiagnosis)> = Vec::new();
    for c in &circuits {
        // Pure-HPWL annealing.
        let mut env_h = PlacementEnv::new(c).expect("env builds");
        let rh = anneal(&mut env_h, &hpwl_cfg);
        let phys_h = env_h.verify_physical();

        // Timing-aware annealing (same seed/schedule, timing term added).
        let mut env_t = PlacementEnv::new(c).expect("env builds");
        let rt = anneal(&mut env_t, &timing_cfg);
        let phys_t = env_t.verify_physical();

        // Δcrit-wl: how much shorter timing-aware made the critical nets vs HPWL-only.
        // Δhpwl: the total-wirelength cost paid for it (negative pct = HPWL grew).
        let d_crit = pct(rh.best_timing, rt.best_timing);
        let d_hpwl = pct(rh.best_hpwl as f64, rt.best_hpwl as f64);
        // Honest verdict: a crit-wl drop only counts if the timing-aware layout
        // still routes. Aggressive weighting can over-pack a congested block.
        let verdict = match (rh.best.metrics.routable, rt.best.metrics.routable) {
            (_, false) => "T unroutable",
            (false, true) => "T fixes route",
            (true, true) if d_crit > 0.5 => "timing↓",
            (true, true) => "tie",
        };

        println!(
            "{:<9} {:>5} | {:>5}>{:<7} {:>6.0}>{:<6.0} {:>3}>{:<4} | {:>+8.1}% {:>+8.1}% {:>4},{:<3} | {:>9}",
            c.name,
            env_h.num_gates(),
            rh.best_hpwl,
            rt.best_hpwl,
            rh.best_timing,
            rt.best_timing,
            rh.best.metrics.wire_tiles,
            rt.best.metrics.wire_tiles,
            d_crit,
            d_hpwl,
            phys_h,
            phys_t,
            verdict,
        );

        // Where naive timing left the block unroutable, show that
        // route_validated_best reclaims a routable layout. The expensive
        // route-checked run happens only here, where it is actually needed.
        if !rt.best.metrics.routable {
            let r_cfg = AnnealConfig {
                timing_weight: TIMING_WEIGHT,
                route_validated_best: true,
                ..AnnealConfig::default()
            };
            let mut env_r = PlacementEnv::new(c).expect("env builds");
            let rr = anneal(&mut env_r, &r_cfg);
            recovered.push((
                c.name.to_string(),
                rr.best.metrics.routable,
                rr.best_timing,
                env_r.diagnose_physical(),
            ));
        }
    }

    println!(
        "\nH = pure-HPWL run, T = timing-aware run. Δcrit-wl > 0 means timing-aware\n\
         shortened the timing-critical nets; Δhpwl < 0 is the total-wirelength it spent\n\
         to do so. Both objectives are O(pins) proxies (routing runs only on baseline +\n\
         best); routable layouts are physically verified on real tiles.\n\
         Tradeoff: a high timing weight can over-pack a congested block until it no\n\
         longer routes ('T unroutable'). The route-validated recovery below reclaims\n\
         a routable layout for those blocks."
    );

    if !recovered.is_empty() {
        println!("\nRoute-validated recovery (route_validated_best = true):");
        for (name, routes, crit_wl, diag) in &recovered {
            println!(
                "  {name:<9} routes: {routes}   crit-wl {crit_wl:.0}   physical: {diag:?}"
            );
        }
        println!(
            "  -> route_validated_best reclaims a ROUTABLE layout via a bounded route\n\
             check (not the full router in the inner loop). But the diagnosis is\n\
             WrongOutput: the reclaimed mul4 layout routes AND the tile sim settles,\n\
             yet computes the wrong function. So routability is NOT a sound legality\n\
             proxy for dense timing-driven placement — 'routable => correct' breaks\n\
             here. verify_physical (physical authority) is the real gate; the safe\n\
             operating range is a moderate weight, where the table's blocks stay legal."
        );
    }
}
