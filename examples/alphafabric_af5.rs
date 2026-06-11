//! AlphaFabric AF5 demo: route-aware learned placer (valid one-shot, no repair).
//!
//! AF4 fit the slot policy to HPWL only, so its one-shot placement could
//! over-cluster and become unroutable on bigger blocks (it needed an SA repair).
//! AF5 fits the full policy (slot + gate order) against the *routed cost* — which
//! charges unroutable layouts — so the one-shot placement is itself routable and
//! physically correct.
//!
//! This compares both policies' ONE-SHOT output (no SA) on held-out circuits the
//! policies never saw, including the mul4 multiplier.
//!
//! Run: `cargo run --release --example alphafabric_af5`

use engine::synth::alphafabric::corpus::{build_eq_comparator, build_parity, build_ripple_adder};
use engine::synth::alphafabric::{
    place, place_policy, train, train_route_aware, Circuit, PlacementEnv, TrainConfig,
};
use engine::synth::benchmark::build_4bit_multiplier;
use std::time::Instant;

fn main() {
    // Identical training set for both policies. It includes medium circuits
    // (adder7/eq7) that actually congest, so the route-aware penalty has signal.
    let train_set = [
        Circuit::from_aig("adder3", build_ripple_adder(3)),
        Circuit::from_aig("adder5", build_ripple_adder(5)),
        Circuit::from_aig("adder7", build_ripple_adder(7)),
        Circuit::from_aig("eq5", build_eq_comparator(5)),
        Circuit::from_aig("eq7", build_eq_comparator(7)),
        Circuit::from_aig("parity6", build_parity(6)),
    ];
    let test_set = [
        Circuit::from_aig("adder4", build_ripple_adder(4)),
        Circuit::from_aig("adder8", build_ripple_adder(8)),
        Circuit::from_aig("eq8", build_eq_comparator(8)),
        Circuit::from_aig("parity8", build_parity(8)),
        Circuit::from_aig("mul4", build_4bit_multiplier()),
    ];

    println!("AlphaFabric AF5 — route-aware learned placer (one-shot validity)\n");

    let af4 = train(&train_set, &TrainConfig::default());
    let t0 = Instant::now();
    let af5 = train_route_aware(
        &train_set,
        &TrainConfig {
            iterations: 150,
            ..TrainConfig::default()
        },
    );
    let af5_train_time = t0.elapsed();

    println!("AF4 (HPWL)        {}", af4.weights);
    println!("AF5 (route-aware) {}", af5.policy);
    println!(
        "AF5 train: cost ratio {:.3} (start {:.3}) in {:?}\n",
        af5.train_cost_ratio, af5.hand_tuned_cost_ratio, af5_train_time
    );

    println!(
        "{:<8} {:>5} {:>18} {:>18}",
        "held-out", "gates", "AF4 1-shot", "AF5 1-shot"
    );
    println!(
        "{:<8} {:>5} {:>18} {:>18}",
        "", "", "hpwl%  phys_ok", "hpwl%  phys_ok"
    );

    let mut af4_valid = 0;
    let mut af5_valid = 0;

    for c in &test_set {
        // AF4 one-shot (slot-only policy, degree order).
        let mut e4 = PlacementEnv::new(c).expect("env builds");
        e4.reset();
        let rowmajor = e4.hpwl() as f64;
        place(&mut e4, &af4.weights);
        let h4 = e4.hpwl() as f64;
        let p4 = e4.score().metrics.routable && e4.verify_physical();
        af4_valid += p4 as usize;

        // AF5 one-shot (full route-aware policy).
        let mut e5 = PlacementEnv::new(c).expect("env builds");
        e5.reset();
        place_policy(&mut e5, &af5.policy);
        let h5 = e5.hpwl() as f64;
        let p5 = e5.score().metrics.routable && e5.verify_physical();
        af5_valid += p5 as usize;

        println!(
            "{:<8} {:>5} {:>11.0}% {:>6} {:>11.0}% {:>6}",
            c.name,
            e4.num_gates(),
            h4 / rowmajor * 100.0,
            p4,
            h5 / rowmajor * 100.0,
            p5,
        );
    }

    println!(
        "\none-shot physically-valid (no SA repair): AF4 {}/{}, AF5 {}/{}",
        af4_valid,
        test_set.len(),
        af5_valid,
        test_set.len()
    );
    println!(
        "AF5 trades a little wirelength for routability so the learned placement\n\
         is correct on the fabric in a single pass — no annealing repair needed.\n\
         Note: route-aware training only learns to spread when the TRAINING set\n\
         contains congesting circuits (adder7/eq7 here) — easy data gives no signal."
    );
}
