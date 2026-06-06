//! Quick diagnostic for QUBO brain scale issues

use engine::brain::qubo_brain::{DecisionBudget, QuboBrain};
use engine::classical_brain::Move;
use engine::qec::noise::SimpleRng;
use engine::quantum_brain::SensorReadings;

fn main() {
    let mut rng = SimpleRng::new(42);
    // Test with just 5 vars (action bits only, no auxiliary)
    let mut brain = QuboBrain::random(5, 2.0, &mut rng);

    // Zero out couplings to test pure bias behavior
    for c in &mut brain.couplings {
        *c = 0.0;
    }

    println!("Testing with 5 vars (action bits only), zero couplings:");

    // Check weight ranges
    let max_coupling = brain
        .couplings
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_coupling = brain
        .couplings
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let max_sensor = brain
        .sensor_weights
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_sensor = brain
        .sensor_weights
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);

    println!("=== QUBO Brain Weight Analysis ===");
    println!("Coupling range: [{:.3}, {:.3}]", min_coupling, max_coupling);
    println!(
        "Sensor weight range: [{:.3}, {:.3}]",
        min_sensor, max_sensor
    );
    println!("One-hot penalty: {}", brain.one_hot_penalty);
    println!();

    // Test with LOW sensors (before finding resources)
    let sensors = SensorReadings {
        heat_n: 5,
        heat_s: 3,
        heat_e: 4,
        heat_w: 2,
        heat_local: 3,
        charge_local: 1,
    };
    println!("Testing with LOW sensors: n=5, s=3, e=4, w=2, local=3");

    let action_names = ["Up", "Down", "Left", "Right", "Stay"];

    // Compute biases and print them
    brain.compute_biases(&sensors);
    let biases = brain.get_biases();
    println!("Computed biases for STRONG NORTH (heat_n=255, others~0):");
    for i in 0..5 {
        println!("  {}: {:+.2}", action_names[i], biases[i]);
    }
    println!();

    // Test decisions with more sweeps
    let budget = DecisionBudget {
        sweeps: 500,
        beta: 5.0,
    };
    // Manually build problem and run annealer to debug
    

    let problem = brain.build_problem();

    println!("Problem biases (after one-hot penalty h_i - lambda):");
    for i in 0..5 {
        println!("  {}: {:+.2}", action_names[i], problem.h[i]);
    }
    println!();

    // Test a few energies manually
    println!("Energy calculations:");
    let test_states = [
        ([1, 0, 0, 0, 0], "Up only"),
        ([0, 0, 0, 0, 1], "Stay only"),
        ([0, 0, 0, 0, 0], "None"),
    ];
    for (state, name) in &test_states {
        let e = problem.energy_binary(state);
        println!("  {}: E = {:.2}", name, e);
    }
    println!();

    println!("Decision test (10 decisions with same strong-north input):");
    let mut counts = [0u32; 5];
    for i in 0..10 {
        let mut test_rng = SimpleRng::new(i as u64);
        let action = brain.decide(&sensors, &mut test_rng, &budget);
        let idx = match action {
            Move::Up => 0,
            Move::Down => 1,
            Move::Left => 2,
            Move::Right => 3,
            Move::Stay => 4,
        };
        counts[idx] += 1;
    }

    for (i, name) in action_names.iter().enumerate() {
        println!("  {}: {} times", name, counts[i]);
    }
    println!();

    // Compare: what if we scale up sensor weights?
    println!("=== Testing with scaled sensor weights (10x) ===");
    let mut scaled_brain = QuboBrain::random(8, 2.0, &mut SimpleRng::new(42));
    for w in &mut scaled_brain.sensor_weights {
        *w *= 10.0;
    }

    println!("Decision test with scaled weights:");
    let mut counts = [0u32; 5];
    for i in 0..10 {
        let mut test_rng = SimpleRng::new(i as u64);
        let action = scaled_brain.decide(&sensors, &mut test_rng, &budget);
        let idx = match action {
            Move::Up => 0,
            Move::Down => 1,
            Move::Left => 2,
            Move::Right => 3,
            Move::Stay => 4,
        };
        counts[idx] += 1;
    }

    for (i, name) in action_names.iter().enumerate() {
        println!("  {}: {} times", name, counts[i]);
    }
}
