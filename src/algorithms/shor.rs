// =============================================================================
// Shor's Factoring Algorithm
// =============================================================================
//
// Implements Shor's quantum factoring algorithm for factoring composite numbers.
//
// Components:
// - Quantum Fourier Transform (QFT) synthesis via Qiskit
// - Modular exponentiation circuits for specific (N, a) pairs
// - Period finding using quantum phase estimation
// - Classical post-processing with continued fractions
//
// Currently supports:
// - N=15 (factors: 3, 5) with a ∈ {2, 7, 8}
// - N=21 (factors: 3, 7) with a ∈ {2, 4}
//
// =============================================================================

use crate::number_theory;
use crate::quantum::{GateOutcome, QGate, QRng, QState, apply_gate_scalar};
use std::io::Write;
use std::process::{Command, Stdio};

// =============================================================================
// Qiskit Synthesis Data Structures
// =============================================================================

/// Simple JSON deserialization structures (manual parsing to avoid serde dependency)
struct QiskitGate {
    gate_type: String,
    control: u8,
    target: u8,
    qubit: u8,
    angle: f32,
}

struct QiskitResult {
    gates: Vec<QiskitGate>,
}

/// Parse JSON output from Qiskit synthesis script
fn parse_qiskit_result(json: &str) -> Option<QiskitResult> {
    // Very simple JSON parser for our specific format
    // Format: {"gates": [{"type": "...", "control": N, "target": M, "qubit": K, "angle": A}, ...]}

    let mut gates = Vec::new();

    // Find the gates array
    let gates_start = json.find("[")? + 1;
    let gates_end = json.rfind("]")?;
    let gates_str = &json[gates_start..gates_end];

    // Split by objects (very simple - assumes no nested objects)
    for gate_obj in gates_str.split("},{") {
        let gate_obj = gate_obj.trim_matches(|c| c == '{' || c == '}' || c == ' ');

        let mut gate_type = String::new();
        let mut control = 0u8;
        let mut target = 0u8;
        let mut qubit = 0u8;
        let mut angle = 0.0f32;

        // Parse key-value pairs
        for pair in gate_obj.split(',') {
            if let Some(colon_pos) = pair.find(':') {
                let key = pair[..colon_pos].trim().trim_matches('"');
                let value = pair[colon_pos + 1..].trim().trim_matches('"');

                match key {
                    "type" => gate_type = value.to_string(),
                    "control" => control = value.parse().unwrap_or(0),
                    "target" => target = value.parse().unwrap_or(0),
                    "qubit" => qubit = value.parse().unwrap_or(0),
                    "angle" => angle = value.parse().unwrap_or(0.0),
                    _ => {}
                }
            }
        }

        gates.push(QiskitGate {
            gate_type,
            control,
            target,
            qubit,
            angle,
        });
    }

    Some(QiskitResult { gates })
}

// =============================================================================
// Qiskit Synthesis Helpers
// =============================================================================

/// Call Qiskit synthesis script for QFT, inverse QFT, or modular exponentiation
fn call_qiskit_shor_synth(
    operation: &str,
    n_qubits: Option<u8>,
    a: Option<u64>,
    n: Option<u64>,
    n_count: Option<u8>,
    n_target: Option<u8>,
) -> Option<Vec<QGate>> {
    // Build JSON input based on operation type
    let json_input = match operation {
        "qft" | "inverse_qft" => {
            format!(
                "{{\"operation\":\"{}\",\"n_qubits\":{}}}",
                operation,
                n_qubits.unwrap_or(0)
            )
        }
        "modular_exp" => {
            format!(
                "{{\"operation\":\"modular_exp\",\"a\":{},\"n\":{},\"n_count\":{},\"n_target\":{}}}",
                a.unwrap_or(0),
                n.unwrap_or(0),
                n_count.unwrap_or(0),
                n_target.unwrap_or(0)
            )
        }
        _ => return None,
    };

    let python_cmd = if cfg!(windows) { "python" } else { "python3" };

    // Try to find the script - works from both root dir and src/algorithms dir
    let script_path = if std::path::Path::new("scripts/qiskit_shor_synth.py").exists() {
        "scripts/qiskit_shor_synth.py"
    } else if std::path::Path::new("../../scripts/qiskit_shor_synth.py").exists() {
        "../../scripts/qiskit_shor_synth.py"
    } else {
        // Fallback - try root path anyway
        "scripts/qiskit_shor_synth.py"
    };

    let mut child = Command::new(python_cmd)
        .arg(script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(json_input.as_bytes()).ok()?;
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("[Shor Qiskit] Python script failed:");
        eprintln!("{}", stderr);
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let result = parse_qiskit_result(&stdout)?;

    // Parse gates
    let mut gates = Vec::new();
    for qgate in result.gates {
        match qgate.gate_type.as_str() {
            "CNot" => gates.push(QGate::CNot(qgate.control, qgate.target)),
            "Rz" => gates.push(QGate::Rz(qgate.qubit, qgate.angle)),
            "Ry" => gates.push(QGate::Ry(qgate.qubit, qgate.angle)),
            "Rx" => gates.push(QGate::Rx(qgate.qubit, qgate.angle)),
            "H" => gates.push(QGate::H(qgate.qubit)),
            "X" => gates.push(QGate::X(qgate.qubit)),
            "Y" => gates.push(QGate::Y(qgate.qubit)),
            "Z" => gates.push(QGate::Z(qgate.qubit)),
            "CRz" => gates.push(QGate::CRz(qgate.control, qgate.target, qgate.angle)),
            "CPhase" => gates.push(QGate::CPhase(qgate.control, qgate.target, qgate.angle)),
            _ => {
                eprintln!("[Shor Qiskit] Unknown gate type: {}", qgate.gate_type);
            }
        }
    }

    if gates.is_empty() {
        return None;
    }

    Some(gates)
}

// =============================================================================
// Circuit Builders
// =============================================================================

/// Synthesize Quantum Fourier Transform circuit.
///
/// The QFT transforms computational basis to Fourier basis:
/// |j⟩ → (1/√2^n) Σ_k e^{2πijk/2^n} |k⟩
///
/// Used in Shor's algorithm for period extraction.
///
/// # Arguments
/// * `n_qubits` - Number of qubits to transform
///
/// # Returns
/// * `Vec<QGate>` - QFT circuit gates
pub fn qft_circuit(n_qubits: u8) -> Vec<QGate> {
    call_qiskit_shor_synth("qft", Some(n_qubits), None, None, None, None)
        .unwrap_or_else(|| panic!("Failed to synthesize QFT for {} qubits", n_qubits))
}

/// Synthesize Inverse Quantum Fourier Transform circuit.
///
/// The inverse QFT transforms from Fourier basis back to computational basis.
///
/// # Arguments
/// * `n_qubits` - Number of qubits to transform
///
/// # Returns
/// * `Vec<QGate>` - Inverse QFT circuit gates
pub fn inverse_qft_circuit(n_qubits: u8) -> Vec<QGate> {
    call_qiskit_shor_synth("inverse_qft", Some(n_qubits), None, None, None, None)
        .unwrap_or_else(|| panic!("Failed to synthesize inverse QFT for {} qubits", n_qubits))
}

/// Synthesize modular exponentiation circuit.
///
/// Implements controlled-U^{2^k} where U|y⟩ = |ay mod N⟩
///
/// Currently supports:
/// - N=15: a ∈ {2, 7, 8}
/// - N=21: a ∈ {2, 4}
///
/// # Arguments
/// * `a` - Base for modular multiplication
/// * `n` - Modulus
/// * `n_count` - Number of counting register qubits
/// * `n_target` - Number of target register qubits
///
/// # Returns
/// * `Option<Vec<QGate>>` - Modular exp circuit, or None if unsupported (a, N)
pub fn modular_exp_circuit(a: u64, n: u64, n_count: u8, n_target: u8) -> Option<Vec<QGate>> {
    call_qiskit_shor_synth(
        "modular_exp",
        None,
        Some(a),
        Some(n),
        Some(n_count),
        Some(n_target),
    )
}

/// Build complete Shor's algorithm circuit for factoring N.
///
/// Circuit structure:
/// 1. Initialize counting register to |+⟩^n_count (superposition)
/// 2. Initialize target register to |1⟩
/// 3. Apply modular exponentiation (controlled on counting register)
/// 4. Apply inverse QFT to counting register
/// 5. Measure counting register to extract period
///
/// # Arguments
/// * `n` - Number to factor (composite, not prime, not prime power)
/// * `a` - Base for modular exponentiation (coprime to n)
///
/// # Returns
/// * `Vec<QGate>` - Complete Shor circuit
///
/// # Panics
/// * If (a, n) pair is not supported for modular exp synthesis
///
/// # Example
/// ```ignore
/// use engine::algorithms::shor_circuit;
/// // Factor 15 using base a=7
/// let circuit = shor_circuit(15, 7);
/// ```
pub fn shor_circuit(n: u64, a: u64) -> Vec<QGate> {
    // Calculate register sizes
    // Counting register: 2 * ceil(log2(N)) qubits for high success probability
    // Target register: ceil(log2(N)) qubits to hold values mod N
    let n_bits = (n as f64).log2().ceil() as u8;
    let n_count = 2 * n_bits; // More counting qubits → better precision
    let n_target = n_bits;

    let mut gates = Vec::new();

    // Step 1: Initialize counting register to |+⟩^n_count (uniform superposition)
    for q in 0..n_count {
        gates.push(QGate::H(q));
    }

    // Step 2: Initialize target register to |1⟩ (start state for modular exp)
    gates.push(QGate::X(n_count)); // Flip first target qubit to |1⟩

    // Step 3: Modular exponentiation U^{2^k} controlled by counting qubits
    let mod_exp_gates = modular_exp_circuit(a, n, n_count, n_target)
        .unwrap_or_else(|| panic!("Unsupported modular exponentiation: a={}, N={}", a, n));
    gates.extend(mod_exp_gates);

    // Step 4: Inverse QFT on counting register to extract period
    let iqft_gates = inverse_qft_circuit(n_count);
    gates.extend(iqft_gates);

    // Step 5: Measure counting register
    for q in 0..n_count {
        gates.push(QGate::Measure(q));
    }

    gates
}

// =============================================================================
// Quantum Runners
// =============================================================================

/// Run Shor's quantum circuit and return measurement result.
///
/// Executes the full quantum circuit and measures the counting register.
/// The measured value encodes information about the period r of a^x mod N.
///
/// # Arguments
/// * `n` - Number to factor
/// * `a` - Base for modular exponentiation
/// * `seed` - RNG seed for deterministic measurement
///
/// # Returns
/// * `u64` - Measurement result from counting register
pub fn run_shor_quantum(n: u64, a: u64, seed: u64) -> u64 {
    let n_bits = (n as f64).log2().ceil() as u8;
    let n_count = 2 * n_bits;
    let n_target = n_bits;
    let n_total = n_count + n_target;

    // Build circuit
    let circuit = shor_circuit(n, a);

    // Initialize quantum state
    let mut state = QState::new_zero(n_total);
    let mut rng = QRng::new(seed);

    // Execute gates and collect measurement
    let mut measured = 0u64;

    for gate in circuit.iter() {
        let outcome = apply_gate_scalar(&mut state, gate, &mut rng);

        // Collect measurement bits
        if let GateOutcome::Measured { qubit, bit } = outcome {
            if bit != 0 {
                measured |= 1u64 << qubit;
            }
        }
    }

    measured
}

// =============================================================================
// Period Extraction and Classical Post-Processing
// =============================================================================

/// Extract period from Shor's measurement using continued fractions.
///
/// The measured value s from the counting register satisfies:
/// s/2^n_count ≈ k/r  (for some integer k < r)
///
/// where r is the period we're looking for.
///
/// Uses continued fraction algorithm to find best rational approximation.
///
/// # Arguments
/// * `measured` - Measurement result from counting register
/// * `register_size` - Size of counting register (2^n_count)
/// * `n` - The number being factored
///
/// # Returns
/// * `Vec<u64>` - Candidate periods (denominators of convergents)
fn extract_period_candidates(measured: u64, register_size: u64, n: u64) -> Vec<u64> {
    if measured == 0 {
        return vec![];
    }

    // Get period candidates from continued fraction expansion
    let mut candidates = number_theory::continued_fraction_candidates(measured, register_size);

    // Filter to valid periods (must divide into powers cleanly)
    candidates.retain(|&r| r > 0 && r < n);

    candidates
}

/// Orchestrator function: Factor N using Shor's algorithm.
///
/// Attempts to find non-trivial factors of N by:
/// 1. Checking for even numbers and prime powers (classical shortcuts)
/// 2. Trying multiple bases a
/// 3. Running quantum period finding circuit
/// 4. Extracting period candidates using continued fractions
/// 5. Computing factors from period using Shor's classical post-processing
///
/// # Arguments
/// * `n` - Composite number to factor (not prime, not prime power, not even)
///
/// # Returns
/// * `Some((p, q))` - Non-trivial factors where n = p * q
/// * `None` - If factorization fails (need more trials or unsupported n)
///
/// # Example
/// ```ignore
/// use engine::algorithms::factor_with_shor;
/// let factors = factor_with_shor(15);
/// assert!(factors == Some((3, 5)) || factors == Some((5, 3)));
/// ```
pub fn factor_with_shor(n: u64) -> Option<(u64, u64)> {
    // Classical shortcuts
    if n.is_multiple_of(2) {
        return Some((2, n / 2));
    }

    if let Some((p, _k)) = number_theory::is_prime_power(n) {
        return Some((p, n / p));
    }

    if number_theory::is_prime(n) {
        return None; // Can't factor primes
    }

    // Calculate register size
    let n_bits = (n as f64).log2().ceil() as u8;
    let register_size = 1u64 << (2 * n_bits);

    // Try different bases
    let bases = match n {
        15 => vec![7, 2, 8, 11, 13],
        21 => vec![2, 4, 5, 8, 10],
        _ => vec![2, 3, 5, 7, 11, 13],
    };

    for a in bases {
        // Skip if a and n share a factor
        if number_theory::gcd(a, n) > 1 {
            continue;
        }

        // Try multiple quantum trials (probabilistic)
        for trial in 0..20 {
            let seed = (a * 1000 + trial) as u64;
            let measured = run_shor_quantum(n, a, seed);

            // Extract period candidates
            let candidates = extract_period_candidates(measured, register_size, n);

            for r in candidates {
                // Try to extract factors from this period
                if let Some(factors) = number_theory::factors_from_period(n, a, r) {
                    return Some(factors);
                }
            }
        }
    }

    None // Failed to factor
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires Python + numpy + qiskit
    fn test_qft_synthesis() {
        // Test that QFT synthesis works
        let gates = qft_circuit(3);
        assert!(
            !gates.is_empty(),
            "QFT should synthesize to non-empty circuit"
        );
        println!("QFT(3 qubits) synthesized to {} gates", gates.len());
    }

    #[test]
    #[ignore] // Requires Python + numpy + qiskit
    fn test_inverse_qft_synthesis() {
        // Test that inverse QFT synthesis works
        let gates = inverse_qft_circuit(3);
        assert!(
            !gates.is_empty(),
            "Inverse QFT should synthesize to non-empty circuit"
        );
        println!("Inverse QFT(3 qubits) synthesized to {} gates", gates.len());
    }

    #[test]
    #[ignore] // Requires Python + numpy + qiskit
    fn test_modular_exp_synthesis() {
        // Test modular exp for supported (a, N) pairs
        let gates = modular_exp_circuit(7, 15, 4, 4);
        assert!(
            gates.is_some(),
            "Modular exp (a=7, N=15) should be supported"
        );

        if let Some(g) = gates {
            println!("Modular exp (a=7, N=15) synthesized to {} gates", g.len());
        }
    }

    #[test]
    #[ignore] // Requires Python + numpy + qiskit
    fn test_shor_circuit_structure() {
        // Test that Shor circuit can be built
        let circuit = shor_circuit(15, 7);
        assert!(!circuit.is_empty(), "Shor circuit should not be empty");

        // Count gate types
        let h_count = circuit.iter().filter(|g| matches!(g, QGate::H(_))).count();
        let measure_count = circuit
            .iter()
            .filter(|g| matches!(g, QGate::Measure(_)))
            .count();

        println!("Shor circuit for N=15, a=7:");
        println!("  Total gates: {}", circuit.len());
        println!("  Hadamards: {}", h_count);
        println!("  Measurements: {}", measure_count);

        assert!(h_count > 0, "Should have Hadamards for superposition");
        assert!(measure_count > 0, "Should have measurements");
    }

    #[test]
    fn test_period_extraction() {
        // Test continued fraction period extraction
        let candidates = extract_period_candidates(85, 256, 15);
        assert!(!candidates.is_empty(), "Should find period candidates");
        println!(
            "Period candidates from measurement 85/256: {:?}",
            candidates
        );
    }

    #[test]
    #[ignore] // This is a full integration test - may take time
    fn test_factor_15() {
        // Full factorization of 15
        let result = factor_with_shor(15);
        assert!(result.is_some(), "Should be able to factor 15");

        if let Some((p, q)) = result {
            println!("Factored 15 = {} × {}", p, q);
            assert!(p * q == 15, "Factors should multiply to 15");
            assert!(p > 1 && q > 1, "Factors should be non-trivial");
        }
    }

    #[test]
    #[ignore] // This is a full integration test - may take time
    fn test_factor_21() {
        // Full factorization of 21
        let result = factor_with_shor(21);
        assert!(result.is_some(), "Should be able to factor 21");

        if let Some((p, q)) = result {
            println!("Factored 21 = {} × {}", p, q);
            assert!(p * q == 21, "Factors should multiply to 21");
            assert!(p > 1 && q > 1, "Factors should be non-trivial");
        }
    }
}
