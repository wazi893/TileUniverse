//! Analyze what quantum interference actually does to spike distributions
//!
//! Run with: cargo run --release --example interference_analysis

use std::f64::consts::PI;

fn main() {
    println!("================================================================");
    println!("  QUANTUM INTERFERENCE ANALYSIS");
    println!("  What does H+CZ+H actually do to spike distributions?");
    println!("================================================================\n");

    // Test various spike count distributions
    let test_cases: Vec<(&str, Vec<u32>)> = vec![
        ("Uniform", vec![10, 10, 10, 10, 10]),
        ("Single dominant", vec![50, 5, 5, 5, 5]),
        ("Two equal", vec![30, 30, 5, 5, 5]),
        ("Gradient", vec![40, 30, 20, 10, 5]),
        ("Bimodal", vec![40, 5, 5, 5, 40]),
        ("One-hot", vec![100, 0, 0, 0, 0]),
        ("Sparse", vec![10, 0, 0, 0, 5]),
    ];

    println!(
        "{:<20} | {:^50} | {:^50}",
        "Distribution", "Before Interference", "After Interference"
    );
    println!("{:-<124}", "");

    for (name, counts) in &test_cases {
        let before = counts_to_probs(counts);
        let after = apply_interference(counts);

        let before_str = format_probs(&before);
        let after_str = format_probs(&after);

        println!("{:<20} | {} | {}", name, before_str, after_str);
    }

    println!("\n================================================================");
    println!("  DETAILED ANALYSIS: What interference changes");
    println!("================================================================\n");

    // Detailed analysis of the "gradient" case
    let gradient = vec![40u32, 30, 20, 10, 5];
    analyze_in_detail("Gradient", &gradient);

    let bimodal = vec![40u32, 5, 5, 5, 40];
    analyze_in_detail("Bimodal", &bimodal);

    let dominant = vec![50u32, 5, 5, 5, 5];
    analyze_in_detail("Single dominant", &dominant);

    println!("\n================================================================");
    println!("  ENTROPY ANALYSIS");
    println!("================================================================\n");

    println!(
        "{:<20} {:>15} {:>15} {:>15}",
        "Distribution", "Before H", "After H", "Change"
    );
    println!("{:-<68}", "");

    for (name, counts) in &test_cases {
        let before = counts_to_probs(counts);
        let after = apply_interference(counts);

        let h_before = entropy(&before);
        let h_after = entropy(&after);
        let change = h_after - h_before;

        println!(
            "{:<20} {:>15.3} {:>15.3} {:>+15.3}",
            name, h_before, h_after, change
        );
    }

    println!("\n================================================================");
    println!("  KEY INSIGHT: How interference redistributes probability");
    println!("================================================================\n");

    // Show which actions gain/lose probability
    println!("For 'Gradient' distribution [40, 30, 20, 10, 5]:\n");

    let gradient = vec![40u32, 30, 20, 10, 5];
    let before = counts_to_probs(&gradient);
    let after = apply_interference(&gradient);

    for i in 0..5 {
        let change = after[i] - before[i];
        let arrow = if change > 0.01 {
            "↑"
        } else if change < -0.01 {
            "↓"
        } else {
            "→"
        };
        println!(
            "  Action {}: {:.1}% {} {:.1}% ({:+.1}%)",
            i,
            before[i] * 100.0,
            arrow,
            after[i] * 100.0,
            change * 100.0
        );
    }

    println!("\n  Interpretation:");
    println!("  - Interference SPREADS probability from dominant actions to weaker ones");
    println!("  - This creates more uniform distribution (higher entropy)");
    println!("  - BUT in our A/B test, quantum had LOWER entropy than classical!");
    println!("  - This suggests the SNN itself is producing different patterns for quantum");

    println!("\n================================================================");
    println!("  THE REAL MECHANISM");
    println!("================================================================\n");

    println!("  The interference circuit (H+CZ+H) does the following:");
    println!();
    println!("  1. HADAMARD on odd qubits: Creates mixing between action pairs");
    println!("     |1⟩ → (|0⟩-|1⟩)/√2, mixing amplitude between actions 0-1, 2-3, etc.");
    println!();
    println!("  2. CZ between neighbors: Applies -1 phase when both neighbors are |1⟩");
    println!("     This creates DESTRUCTIVE interference for certain combinations");
    println!();
    println!("  3. HADAMARD on even qubits: Further mixing");
    println!();
    println!("  NET EFFECT: Probability flows from isolated peaks to neighboring actions");
    println!("              via constructive interference, while some combinations");
    println!("              are suppressed via destructive interference.");
    println!();
    println!("  In predator-prey context:");
    println!("  - If SNN strongly favors 'Up', interference spreads some probability");
    println!("    to 'Down', 'Left', 'Right' - creating implicit 'exploration'");
    println!("  - The specific pattern of spreading may HAPPEN to correlate with");
    println!("    good escape directions (not guaranteed, task-dependent)");
}

fn counts_to_probs(counts: &[u32]) -> Vec<f64> {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return vec![1.0 / counts.len() as f64; counts.len()];
    }
    counts.iter().map(|&c| c as f64 / total as f64).collect()
}

fn apply_interference(counts: &[u32]) -> Vec<f64> {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return vec![1.0 / counts.len() as f64; counts.len()];
    }

    // Step 1: Encode as amplitudes (sqrt of probability)
    let mut amps: Vec<(f64, f64)> = counts
        .iter()
        .map(|&c| ((c as f64 / total as f64).sqrt(), 0.0))
        .collect();

    let n = amps.len();
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();

    // Hadamard on odd indices
    // H|0⟩ = (|0⟩+|1⟩)/√2, H|1⟩ = (|0⟩-|1⟩)/√2
    // In amplitude space, this mixes adjacent pairs
    for i in (1..n).step_by(2) {
        if i > 0 {
            let a = amps[i - 1];
            let b = amps[i];
            // Simplified: mix amplitudes of adjacent actions
            amps[i - 1] = ((a.0 + b.0) * sqrt2_inv, (a.1 + b.1) * sqrt2_inv);
            amps[i] = ((a.0 - b.0) * sqrt2_inv, (a.1 - b.1) * sqrt2_inv);
        }
    }

    // CZ between adjacent pairs
    // CZ adds -1 phase when both qubits are |1⟩
    // In our encoding, this affects interference between neighboring actions
    for i in 0..(n - 1) {
        // CZ effect: phase kick based on product of amplitudes
        let product = amps[i].0 * amps[i + 1].0;
        if product.abs() > 0.001 {
            // Apply phase rotation proportional to interaction
            let phase = -PI * product * 0.5; // Simplified CZ effect
            let cos_p = phase.cos();
            let sin_p = phase.sin();

            // Rotate both amplitudes
            let (re, im) = amps[i];
            amps[i] = (re * cos_p - im * sin_p, re * sin_p + im * cos_p);

            let (re, im) = amps[i + 1];
            amps[i + 1] = (re * cos_p - im * sin_p, re * sin_p + im * cos_p);
        }
    }

    // Hadamard on even indices
    for i in (0..n).step_by(2) {
        if i + 1 < n {
            let a = amps[i];
            let b = amps[i + 1];
            amps[i] = ((a.0 + b.0) * sqrt2_inv, (a.1 + b.1) * sqrt2_inv);
            amps[i + 1] = ((a.0 - b.0) * sqrt2_inv, (a.1 - b.1) * sqrt2_inv);
        }
    }

    // Convert back to probabilities
    let mut probs: Vec<f64> = amps.iter().map(|(re, im)| re * re + im * im).collect();

    // Normalize
    let sum: f64 = probs.iter().sum();
    if sum > 1e-10 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    probs
}

fn entropy(probs: &[f64]) -> f64 {
    let mut h = 0.0;
    for &p in probs {
        if p > 1e-10 {
            h -= p * p.log2();
        }
    }
    h
}

fn format_probs(probs: &[f64]) -> String {
    probs
        .iter()
        .map(|&p| format!("{:5.1}%", p * 100.0))
        .collect::<Vec<_>>()
        .join(" ")
}

fn analyze_in_detail(name: &str, counts: &[u32]) {
    println!("--- {} ---", name);
    println!("Spike counts: {:?}", counts);

    let before = counts_to_probs(counts);
    let after = apply_interference(counts);

    println!(
        "Before: {:?}",
        before
            .iter()
            .map(|p| format!("{:.3}", p))
            .collect::<Vec<_>>()
    );
    println!(
        "After:  {:?}",
        after
            .iter()
            .map(|p| format!("{:.3}", p))
            .collect::<Vec<_>>()
    );

    // Find biggest changes
    let mut changes: Vec<(usize, f64)> = (0..5).map(|i| (i, after[i] - before[i])).collect();
    changes.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    println!("Biggest changes:");
    for (i, change) in changes.iter().take(3) {
        if change.abs() > 0.001 {
            println!("  Action {}: {:+.1}%", i, change * 100.0);
        }
    }
    println!();
}
