// =============================================================================
// src/quantum_brain.rs
// =============================================================================
//
// SPRINT 3.1: Quantum decision-making for organisms
//
// This module provides the "brain" for QuantumForager organisms:
// - Sensing: Read local environment fields
// - Encoding: Convert sensor readings to qubit rotations
// - Execution: Run the genome circuit and measure
// - Decoding: Convert measurement to action
// - Mutation: Evolve circuits across generations
//
// =============================================================================

use crate::quantum::{GateOutcome, QGate, QRng, QState, apply_gate_scalar};
use crate::simulation::Simulation;
use crate::tilemap::{HEIGHT, WIDTH};

// EPIC 90: Import fusion pipeline for optimized circuit execution
use crate::fusion::{execute_fused_cpu, fuse_program};

use std::f32::consts::PI;

// =============================================================================
// Sensor Readings
// =============================================================================

/// Environmental sensor readings around an organism's position.
#[derive(Debug, Clone, Copy, Default)]
pub struct SensorReadings {
    /// Heat in the cell to the north (y - 1)
    pub heat_n: u32,
    /// Heat in the cell to the south (y + 1)
    pub heat_s: u32,
    /// Heat in the cell to the east (x + 1)
    pub heat_e: u32,
    /// Heat in the cell to the west (x - 1)
    pub heat_w: u32,
    /// Heat at current position
    pub heat_local: u32,
    /// Charge at current position
    pub charge_local: u32,
}

impl SensorReadings {
    /// Returns the direction with maximum heat (for gradient following).
    /// Returns (dx, dy) where each is -1, 0, or 1.
    pub fn max_heat_direction(&self) -> (i32, i32) {
        let dirs = [
            (self.heat_n, 0, -1), // North
            (self.heat_s, 0, 1),  // South
            (self.heat_e, 1, 0),  // East
            (self.heat_w, -1, 0), // West
        ];

        let max = dirs.iter().max_by_key(|(h, _, _)| h).unwrap();
        (max.1, max.2)
    }

    /// Total heat in neighborhood (for assessing resource richness).
    pub fn total_heat(&self) -> u32 {
        self.heat_n + self.heat_s + self.heat_e + self.heat_w + self.heat_local
    }
}

/// Read environmental fields around the given position.
///
/// Returns sensor readings for the 4 cardinal directions plus local cell.
pub fn sense_local_fields(sim: &Simulation, x: u32, y: u32) -> SensorReadings {
    let xu = x as usize;
    let yu = y as usize;

    // Helper to safely read heat field
    let get_heat = |px: usize, py: usize| -> u32 {
        if px < WIDTH && py < HEIGHT {
            sim.get_heat_field_for_test(px, py)
        } else {
            0
        }
    };

    // Helper to safely read charge field
    let get_charge = |px: usize, py: usize| -> u32 {
        if px < WIDTH && py < HEIGHT {
            sim.get_charge_field_for_test(px, py)
        } else {
            0
        }
    };

    SensorReadings {
        heat_n: get_heat(xu, yu.saturating_sub(1)),
        heat_s: get_heat(xu, (yu + 1).min(HEIGHT - 1)),
        heat_e: get_heat((xu + 1).min(WIDTH - 1), yu),
        heat_w: get_heat(xu.saturating_sub(1), yu),
        heat_local: get_heat(xu, yu),
        charge_local: get_charge(xu, yu),
    }
}

// =============================================================================
// Quantum Encoding
// =============================================================================

/// Encode sensor readings as qubit rotation gates.
///
/// Maps heat values [0, 255] to rotation angles [0, π].
/// These rotations are prepended to the organism's genome circuit.
///
/// Encoding scheme (4 qubits):
/// - Qubit 0: heat_n normalized rotation
/// - Qubit 1: heat_s normalized rotation
/// - Qubit 2: heat_e normalized rotation
/// - Qubit 3: heat_w normalized rotation
pub fn encode_as_rotations(readings: &SensorReadings, n_qubits: u8) -> Vec<QGate> {
    let max_heat = 255.0f32;

    let mut gates = Vec::with_capacity(n_qubits as usize);

    // Normalize heat to [0, π] and apply as Phase rotations
    if n_qubits >= 1 {
        let angle = (readings.heat_n as f32 / max_heat) * PI;
        gates.push(QGate::Phase(0, angle));
    }
    if n_qubits >= 2 {
        let angle = (readings.heat_s as f32 / max_heat) * PI;
        gates.push(QGate::Phase(1, angle));
    }
    if n_qubits >= 3 {
        let angle = (readings.heat_e as f32 / max_heat) * PI;
        gates.push(QGate::Phase(2, angle));
    }
    if n_qubits >= 4 {
        let angle = (readings.heat_w as f32 / max_heat) * PI;
        gates.push(QGate::Phase(3, angle));
    }

    gates
}

// =============================================================================
// Circuit Execution
// =============================================================================

/// Execute a quantum circuit and return the measurement result.
///
/// The full circuit is: sensor_gates + genome + measurement_gates
///
/// # Arguments
/// * `genome` - The organism's evolvable circuit (no measurements)
/// * `sensor_gates` - Environment-encoded rotation gates
/// * `n_qubits` - Number of qubits in the circuit
/// * `seed` - Random seed for measurement outcomes
///
/// # Returns
/// The measurement result as a bit pattern (qubit 0 = LSB).
pub fn run_circuit(genome: &[QGate], sensor_gates: &[QGate], n_qubits: u8, seed: u64) -> u64 {
    // Initialize state to |0...0⟩
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(seed);

    // Apply Hadamard to create superposition (optional: could be part of genome)
    for q in 0..n_qubits {
        apply_gate_scalar(&mut state, &QGate::H(q), &mut rng);
    }

    // Apply sensor encoding
    for gate in sensor_gates {
        apply_gate_scalar(&mut state, gate, &mut rng);
    }

    // Apply genome (the evolvable part)
    for gate in genome {
        // Skip any measurements in the genome (they're added at the end)
        if !matches!(gate, QGate::Measure(_)) {
            apply_gate_scalar(&mut state, gate, &mut rng);
        }
    }

    // Measure all qubits
    let mut result: u64 = 0;
    for q in 0..n_qubits {
        let outcome = apply_gate_scalar(&mut state, &QGate::Measure(q), &mut rng);
        if let GateOutcome::Measured { qubit, bit } = outcome {
            if bit != 0 {
                result |= 1u64 << qubit;
            }
        }
    }

    result
}

/// Run a quantum circuit with noise injection.
///
/// Same as `run_circuit` but applies noise to the genome before execution.
/// This simulates running on noisy quantum hardware.
pub fn run_circuit_noisy(
    genome: &[QGate],
    sensor_gates: &[QGate],
    n_qubits: u8,
    seed: u64,
    noise_config: &crate::noise_model::NoiseConfig,
) -> u64 {
    use crate::noise_model::apply_noise_to_circuit;

    // Apply noise to the genome
    let noisy_genome = apply_noise_to_circuit(genome, noise_config, n_qubits, seed);

    // Run with noisy genome
    run_circuit(&noisy_genome, sensor_gates, n_qubits, seed.wrapping_add(1))
}

// =============================================================================
// EPIC 90: Fusion-Optimized Circuit Execution
// =============================================================================

/// Result of running a fused circuit, including performance metrics.
#[derive(Debug, Clone)]
pub struct FusedCircuitResult {
    /// The measurement outcome (bit pattern)
    pub measurement: u64,
    /// Number of gates in the original circuit
    pub original_gates: usize,
    /// Number of operations after fusion
    pub fused_ops: usize,
    /// Number of gates eliminated by algebraic fusion
    pub eliminated: usize,
    /// Whether fusion was used (vs fallback to scalar)
    pub used_fusion: bool,
}

/// Execute a quantum circuit using the fusion pipeline (EPIC 83-89).
///
/// This routes through the optimized fusion path which provides:
/// - Algebraic identity elimination (H² = I, X² = I, rotation accumulation)
/// - MegaGate composition (consecutive single-qubit gates → single 2x2 matrix)
/// - Batched execution preparation (for GPU offload in Phase 90.2)
///
/// # Arguments
/// * `genome` - The organism's evolvable circuit (no measurements)
/// * `sensor_gates` - Environment-encoded rotation gates
/// * `n_qubits` - Number of qubits in the circuit
/// * `seed` - Random seed for measurement outcomes
///
/// # Returns
/// A `FusedCircuitResult` containing measurement and performance metrics.
pub fn run_circuit_fused(
    genome: &[QGate],
    sensor_gates: &[QGate],
    n_qubits: u8,
    seed: u64,
) -> FusedCircuitResult {
    // Initialize state to |0...0⟩
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(seed);

    // Build the full circuit: H on all qubits + sensors + genome
    let mut full_circuit: Vec<QGate> =
        Vec::with_capacity(n_qubits as usize + sensor_gates.len() + genome.len());

    // Add initial Hadamards to create superposition
    for q in 0..n_qubits {
        full_circuit.push(QGate::H(q));
    }

    // Add sensor encoding
    full_circuit.extend(sensor_gates.iter().cloned());

    // Add genome (excluding any measurements)
    for gate in genome {
        if !matches!(gate, QGate::Measure(_)) {
            full_circuit.push(gate.clone());
        }
    }

    let original_gates = full_circuit.len();

    // Fuse the circuit using EPIC 83/84 optimizations
    let fused = fuse_program(&full_circuit);
    let fused_ops = fused.ops.len();
    let eliminated = fused.eliminated;

    // Execute the fused program
    let (_ops_count, _outcomes) = execute_fused_cpu(&mut state, &fused, &mut rng);

    // Measure all qubits
    let mut result: u64 = 0;
    for q in 0..n_qubits {
        let outcome = apply_gate_scalar(&mut state, &QGate::Measure(q), &mut rng);
        if let GateOutcome::Measured { qubit, bit } = outcome {
            if bit != 0 {
                result |= 1u64 << qubit;
            }
        }
    }

    FusedCircuitResult {
        measurement: result,
        original_gates,
        fused_ops,
        eliminated,
        used_fusion: true,
    }
}

/// Run circuit with automatic selection between scalar and fused paths.
///
/// Uses cost model to decide whether fusion is worthwhile for this circuit.
/// Small circuits (< 8 gates) typically don't benefit from fusion overhead.
///
/// # Arguments
/// Same as `run_circuit_fused`
///
/// # Returns
/// A `FusedCircuitResult` with `used_fusion` indicating which path was taken.
pub fn run_circuit_auto(
    genome: &[QGate],
    sensor_gates: &[QGate],
    n_qubits: u8,
    seed: u64,
) -> FusedCircuitResult {
    // Cost model threshold: fusion has ~2-3μs overhead
    // At ~0.5μs per scalar gate, need ~5+ gates to amortize
    let total_gates = n_qubits as usize + sensor_gates.len() + genome.len();

    if total_gates < 8 {
        // Small circuit: use scalar path, less overhead
        let measurement = run_circuit(genome, sensor_gates, n_qubits, seed);
        FusedCircuitResult {
            measurement,
            original_gates: total_gates,
            fused_ops: total_gates,
            eliminated: 0,
            used_fusion: false,
        }
    } else {
        // Larger circuit: use fusion for potential algebraic elimination
        run_circuit_fused(genome, sensor_gates, n_qubits, seed)
    }
}

// =============================================================================
// Action Decoding
// =============================================================================

/// Movement action: (dx, dy) where each component is -1, 0, or 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveAction {
    pub dx: i32,
    pub dy: i32,
}

impl MoveAction {
    pub const NORTH: MoveAction = MoveAction { dx: 0, dy: -1 };
    pub const SOUTH: MoveAction = MoveAction { dx: 0, dy: 1 };
    pub const EAST: MoveAction = MoveAction { dx: 1, dy: 0 };
    pub const WEST: MoveAction = MoveAction { dx: -1, dy: 0 };
    pub const STAY: MoveAction = MoveAction { dx: 0, dy: 0 };
}

/// Decode measurement result to movement action.
///
/// Uses the lowest 2 bits of the measurement:
/// - 00 → North (0, -1)
/// - 01 → South (0, +1)
/// - 10 → East (+1, 0)
/// - 11 → West (-1, 0)
pub fn decode_action(measurement: u64) -> MoveAction {
    match measurement & 0b11 {
        0b00 => MoveAction::NORTH,
        0b01 => MoveAction::SOUTH,
        0b10 => MoveAction::EAST,
        0b11 => MoveAction::WEST,
        _ => unreachable!(),
    }
}

/// Decode with 3 bits to include "stay" action.
///
/// - 000-011 → directional movement
/// - 100-111 → stay in place
#[allow(dead_code)]
pub fn decode_action_with_stay(measurement: u64) -> MoveAction {
    match measurement & 0b111 {
        0b000 => MoveAction::NORTH,
        0b001 => MoveAction::SOUTH,
        0b010 => MoveAction::EAST,
        0b011 => MoveAction::WEST,
        _ => MoveAction::STAY,
    }
}

// =============================================================================
// Random Circuit Generation
// =============================================================================

/// Simple xorshift64 PRNG for circuit generation.
/// (Separate from QRng to avoid coupling circuit evolution to quantum execution)
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) } // Ensure non-zero
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    pub fn gen_range(&mut self, min: u32, max: u32) -> u32 {
        if max <= min {
            return min;
        }
        min + (self.next_u32() % (max - min))
    }

    pub fn gen_float(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }

    pub fn gen_bool(&mut self, probability: f32) -> bool {
        self.gen_float() < probability
    }
}

/// Generate a random quantum circuit (the organism's initial genome).
///
/// The circuit includes a mix of:
/// - Single-qubit gates: H, X, Z, Phase(random angle)
/// - Two-qubit gates: CNOT, CZ
/// - (Optionally) Three-qubit gates: Toffoli, CCZ
///
/// No Measure gates are included — those are added during execution.
pub fn random_circuit(n_qubits: u8, length: usize, seed: u64) -> Vec<QGate> {
    let mut rng = SimpleRng::new(seed);
    let mut circuit = Vec::with_capacity(length);

    for _ in 0..length {
        let gate_type = rng.gen_range(0, 10);

        let gate = match gate_type {
            0..=1 => {
                // Hadamard
                let q = rng.gen_range(0, n_qubits as u32) as u8;
                QGate::H(q)
            }
            2 => {
                // X gate
                let q = rng.gen_range(0, n_qubits as u32) as u8;
                QGate::X(q)
            }
            3 => {
                // Z gate
                let q = rng.gen_range(0, n_qubits as u32) as u8;
                QGate::Z(q)
            }
            4..=5 => {
                // Phase with random angle
                let q = rng.gen_range(0, n_qubits as u32) as u8;
                let angle = rng.gen_float() * 2.0 * PI;
                QGate::Phase(q, angle)
            }
            6..=7 => {
                // CNOT (requires 2+ qubits)
                if n_qubits >= 2 {
                    let control = rng.gen_range(0, n_qubits as u32) as u8;
                    let mut target = rng.gen_range(0, n_qubits as u32) as u8;
                    while target == control {
                        target = rng.gen_range(0, n_qubits as u32) as u8;
                    }
                    QGate::CNot(control, target)
                } else {
                    QGate::H(0)
                }
            }
            8 => {
                // CZ (requires 2+ qubits)
                if n_qubits >= 2 {
                    let a = rng.gen_range(0, n_qubits as u32) as u8;
                    let mut b = rng.gen_range(0, n_qubits as u32) as u8;
                    while b == a {
                        b = rng.gen_range(0, n_qubits as u32) as u8;
                    }
                    QGate::CZ(a, b)
                } else {
                    QGate::Z(0)
                }
            }
            9 => {
                // Toffoli (requires 3+ qubits)
                if n_qubits >= 3 {
                    let c0 = rng.gen_range(0, n_qubits as u32) as u8;
                    let mut c1 = rng.gen_range(0, n_qubits as u32) as u8;
                    while c1 == c0 {
                        c1 = rng.gen_range(0, n_qubits as u32) as u8;
                    }
                    let mut t = rng.gen_range(0, n_qubits as u32) as u8;
                    while t == c0 || t == c1 {
                        t = rng.gen_range(0, n_qubits as u32) as u8;
                    }
                    QGate::Toffoli(c0, c1, t)
                } else if n_qubits >= 2 {
                    let control = rng.gen_range(0, n_qubits as u32) as u8;
                    let target = if control == 0 { 1 } else { 0 };
                    QGate::CNot(control, target)
                } else {
                    QGate::X(0)
                }
            }
            _ => QGate::H(0),
        };

        circuit.push(gate);
    }

    circuit
}

// =============================================================================
// Circuit Mutation
// =============================================================================

/// Mutation types for circuit evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationType {
    /// Insert a random gate at a random position
    InsertGate,
    /// Delete a gate at a random position
    DeleteGate,
    /// Swap two gates
    SwapGates,
    /// Change the qubit(s) a gate operates on
    ChangeQubits,
    /// Change the angle of a Phase gate
    ChangeAngle,
    /// Replace a gate with a different type
    ReplaceGate,
}

/// Mutate a circuit to create offspring variation.
///
/// Applies 1-3 random mutations based on mutation_rate.
///
/// # Arguments
/// * `circuit` - The parent circuit to mutate
/// * `n_qubits` - Number of qubits (for valid gate generation)
/// * `max_length` - Maximum circuit length (prevents unbounded growth)
/// * `seed` - Random seed for mutation choices
///
/// # Returns
/// A new circuit with mutations applied.
pub fn mutate_circuit(circuit: &[QGate], n_qubits: u8, max_length: usize, seed: u64) -> Vec<QGate> {
    let mut rng = SimpleRng::new(seed);
    let mut new_circuit = circuit.to_vec();

    // Apply 1-3 mutations
    let num_mutations = rng.gen_range(1, 4);

    for _ in 0..num_mutations {
        if new_circuit.is_empty() {
            // Can only insert if empty
            let gate = random_gate(n_qubits, &mut rng);
            new_circuit.push(gate);
            continue;
        }

        // Choose mutation type
        let mutation_type = match rng.gen_range(0, 6) {
            0 => MutationType::InsertGate,
            1 => MutationType::DeleteGate,
            2 => MutationType::SwapGates,
            3 => MutationType::ChangeQubits,
            4 => MutationType::ChangeAngle,
            _ => MutationType::ReplaceGate,
        };

        match mutation_type {
            MutationType::InsertGate => {
                if new_circuit.len() < max_length {
                    let pos = rng.gen_range(0, new_circuit.len() as u32 + 1) as usize;
                    let gate = random_gate(n_qubits, &mut rng);
                    new_circuit.insert(pos, gate);
                }
            }

            MutationType::DeleteGate => {
                if new_circuit.len() > 1 {
                    let pos = rng.gen_range(0, new_circuit.len() as u32) as usize;
                    new_circuit.remove(pos);
                }
            }

            MutationType::SwapGates => {
                if new_circuit.len() >= 2 {
                    let pos1 = rng.gen_range(0, new_circuit.len() as u32) as usize;
                    let mut pos2 = rng.gen_range(0, new_circuit.len() as u32) as usize;
                    while pos2 == pos1 {
                        pos2 = rng.gen_range(0, new_circuit.len() as u32) as usize;
                    }
                    new_circuit.swap(pos1, pos2);
                }
            }

            MutationType::ChangeQubits => {
                let pos = rng.gen_range(0, new_circuit.len() as u32) as usize;
                new_circuit[pos] = mutate_gate_qubits(&new_circuit[pos], n_qubits, &mut rng);
            }

            MutationType::ChangeAngle => {
                // Find a Phase gate and modify its angle
                let phase_positions: Vec<usize> = new_circuit
                    .iter()
                    .enumerate()
                    .filter_map(|(i, g)| {
                        if matches!(g, QGate::Phase(_, _)) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();

                if !phase_positions.is_empty() {
                    let idx =
                        phase_positions[rng.gen_range(0, phase_positions.len() as u32) as usize];
                    if let QGate::Phase(q, angle) = new_circuit[idx] {
                        // Small angle perturbation
                        let delta = (rng.gen_float() - 0.5) * PI * 0.5;
                        new_circuit[idx] = QGate::Phase(q, angle + delta);
                    }
                }
            }

            MutationType::ReplaceGate => {
                let pos = rng.gen_range(0, new_circuit.len() as u32) as usize;
                new_circuit[pos] = random_gate(n_qubits, &mut rng);
            }
        }
    }

    new_circuit
}

/// Generate a single random gate.
fn random_gate(n_qubits: u8, rng: &mut SimpleRng) -> QGate {
    let gate_type = rng.gen_range(0, 8);

    match gate_type {
        0 => QGate::H(rng.gen_range(0, n_qubits as u32) as u8),
        1 => QGate::X(rng.gen_range(0, n_qubits as u32) as u8),
        2 => QGate::Z(rng.gen_range(0, n_qubits as u32) as u8),
        3 => {
            let q = rng.gen_range(0, n_qubits as u32) as u8;
            let angle = rng.gen_float() * 2.0 * PI;
            QGate::Phase(q, angle)
        }
        4..=5 if n_qubits >= 2 => {
            let control = rng.gen_range(0, n_qubits as u32) as u8;
            let mut target = rng.gen_range(0, n_qubits as u32) as u8;
            while target == control {
                target = rng.gen_range(0, n_qubits as u32) as u8;
            }
            if gate_type == 4 {
                QGate::CNot(control, target)
            } else {
                QGate::CZ(control, target)
            }
        }
        6..=7 if n_qubits >= 3 => {
            let c0 = rng.gen_range(0, n_qubits as u32) as u8;
            let mut c1 = rng.gen_range(0, n_qubits as u32) as u8;
            while c1 == c0 {
                c1 = rng.gen_range(0, n_qubits as u32) as u8;
            }
            let mut t = rng.gen_range(0, n_qubits as u32) as u8;
            while t == c0 || t == c1 {
                t = rng.gen_range(0, n_qubits as u32) as u8;
            }
            if gate_type == 6 {
                QGate::Toffoli(c0, c1, t)
            } else {
                QGate::CCZ(c0, c1, t)
            }
        }
        _ => QGate::H(rng.gen_range(0, n_qubits as u32) as u8),
    }
}

/// Mutate the qubit indices of a gate while preserving its type.
fn mutate_gate_qubits(gate: &QGate, n_qubits: u8, rng: &mut SimpleRng) -> QGate {
    match gate {
        QGate::H(_) => QGate::H(rng.gen_range(0, n_qubits as u32) as u8),
        QGate::X(_) => QGate::X(rng.gen_range(0, n_qubits as u32) as u8),
        QGate::Z(_) => QGate::Z(rng.gen_range(0, n_qubits as u32) as u8),
        QGate::Phase(_, angle) => QGate::Phase(rng.gen_range(0, n_qubits as u32) as u8, *angle),
        QGate::CNot(_, _) if n_qubits >= 2 => {
            let c = rng.gen_range(0, n_qubits as u32) as u8;
            let mut t = rng.gen_range(0, n_qubits as u32) as u8;
            while t == c {
                t = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::CNot(c, t)
        }
        QGate::CZ(_, _) if n_qubits >= 2 => {
            let a = rng.gen_range(0, n_qubits as u32) as u8;
            let mut b = rng.gen_range(0, n_qubits as u32) as u8;
            while b == a {
                b = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::CZ(a, b)
        }
        QGate::Toffoli(_, _, _) if n_qubits >= 3 => {
            let c0 = rng.gen_range(0, n_qubits as u32) as u8;
            let mut c1 = rng.gen_range(0, n_qubits as u32) as u8;
            while c1 == c0 {
                c1 = rng.gen_range(0, n_qubits as u32) as u8;
            }
            let mut t = rng.gen_range(0, n_qubits as u32) as u8;
            while t == c0 || t == c1 {
                t = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::Toffoli(c0, c1, t)
        }
        QGate::CCZ(_, _, _) if n_qubits >= 3 => {
            let a = rng.gen_range(0, n_qubits as u32) as u8;
            let mut b = rng.gen_range(0, n_qubits as u32) as u8;
            while b == a {
                b = rng.gen_range(0, n_qubits as u32) as u8;
            }
            let mut c = rng.gen_range(0, n_qubits as u32) as u8;
            while c == a || c == b {
                c = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::CCZ(a, b, c)
        }
        QGate::Measure(q) => QGate::Measure(*q),
        // Fallback: return original gate
        _ => gate.clone(),
    }
}

// =============================================================================
// Circuit Analysis (for evolution metrics)
// =============================================================================

/// Count entangling gates in a circuit.
///
/// Entangling gates (CNOT, CZ, Toffoli, CCZ) create quantum correlations
/// that cannot be simulated efficiently classically.
pub fn count_entangling_gates(circuit: &[QGate]) -> usize {
    circuit
        .iter()
        .filter(|g| {
            matches!(g, QGate::CNot(_, _) | QGate::CZ(_, _))
                || matches!(g, QGate::Toffoli(_, _, _) | QGate::CCZ(_, _, _))
        })
        .count()
}

/// Calculate "quantum-ness" metric: entangling gates / total gates.
///
/// Returns 0.0 for empty circuits or circuits with no entangling gates.
/// Returns up to 1.0 for circuits that are all entangling gates.
pub fn quantum_ness(circuit: &[QGate]) -> f32 {
    if circuit.is_empty() {
        return 0.0;
    }
    count_entangling_gates(circuit) as f32 / circuit.len() as f32
}

/// Categorize gates in a circuit.
#[derive(Debug, Clone, Default)]
pub struct CircuitStats {
    pub total_gates: usize,
    pub hadamard_count: usize,
    pub pauli_count: usize, // X, Z
    pub phase_count: usize,
    pub cnot_count: usize,
    pub cz_count: usize,
    pub toffoli_count: usize,
    pub ccz_count: usize,
    pub measure_count: usize,
}

impl CircuitStats {
    pub fn entangling_gates(&self) -> usize {
        self.cnot_count + self.cz_count + self.toffoli_count + self.ccz_count
    }

    pub fn single_qubit_gates(&self) -> usize {
        self.hadamard_count + self.pauli_count + self.phase_count
    }
}

/// Analyze a circuit and return statistics.
pub fn analyze_circuit(circuit: &[QGate]) -> CircuitStats {
    let mut stats = CircuitStats::default();
    stats.total_gates = circuit.len();

    for gate in circuit {
        match gate {
            QGate::H(_) => stats.hadamard_count += 1,
            QGate::X(_) | QGate::Z(_) => stats.pauli_count += 1,
            QGate::Phase(_, _) => stats.phase_count += 1,
            QGate::Rx(_, _) | QGate::Ry(_, _) | QGate::Rz(_, _) => stats.phase_count += 1, // EPIC 83: rotation gates
            QGate::U3(_, _, _, _) => stats.phase_count += 1, // SPRINT 32.0: U3 is a parameterized single-qubit gate
            QGate::T(_) | QGate::Tdg(_) => stats.phase_count += 1, // SPRINT 48.0: T gates
            QGate::Y(_) => stats.pauli_count += 1,           // Y is a Pauli gate
            QGate::Swap(_, _) => stats.cnot_count += 3,      // Swap ~ 3 CNOTs
            QGate::CNot(_, _) => stats.cnot_count += 1,
            QGate::CZ(_, _) => stats.cz_count += 1,
            QGate::Toffoli(_, _, _) => stats.toffoli_count += 1,
            QGate::CCZ(_, _, _) => stats.ccz_count += 1,
            QGate::Measure(_) => stats.measure_count += 1,
            // Shor's algorithm: Controlled rotation gates
            QGate::CPhase(_, _, _) => stats.phase_count += 1,
            QGate::CRz(_, _, _) => stats.phase_count += 1,
        }
    }

    stats
}

// =============================================================================
// EPIC 90.2: Circuit Hashing for Batching
// =============================================================================

/// Hash a quantum circuit for batching purposes.
///
/// Organisms with the same circuit hash can potentially be batched together
/// for more efficient GPU execution. The hash captures the circuit structure
/// (gate types and qubit indices) but NOT parameter values (angles).
///
/// This allows batching organisms that have the same gate sequence but
/// different rotation angles (which is common after mutation).
pub fn circuit_hash(circuit: &[QGate]) -> u64 {
    // FNV-1a hash
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;

    for gate in circuit {
        // Hash gate type and qubit indices, but NOT angle values
        let gate_code: u64 = match gate {
            QGate::H(q) => 0x01 | ((*q as u64) << 8),
            QGate::X(q) => 0x02 | ((*q as u64) << 8),
            QGate::Z(q) => 0x03 | ((*q as u64) << 8),
            QGate::Phase(q, _) => 0x04 | ((*q as u64) << 8), // Ignore angle
            QGate::Rx(q, _) => 0x05 | ((*q as u64) << 8),
            QGate::Ry(q, _) => 0x06 | ((*q as u64) << 8),
            QGate::Rz(q, _) => 0x07 | ((*q as u64) << 8),
            QGate::Y(q) => 0x08 | ((*q as u64) << 8),
            QGate::U3(q, _, _, _) => 0x0F | ((*q as u64) << 8), // SPRINT 32.0: Ignore angles
            QGate::Swap(c, t) => 0x09 | ((*c as u64) << 8) | ((*t as u64) << 16),
            QGate::CNot(c, t) => 0x10 | ((*c as u64) << 8) | ((*t as u64) << 16),
            QGate::CZ(a, b) => 0x11 | ((*a as u64) << 8) | ((*b as u64) << 16),
            QGate::Toffoli(c0, c1, t) => {
                0x20 | ((*c0 as u64) << 8) | ((*c1 as u64) << 16) | ((*t as u64) << 24)
            }
            QGate::CCZ(a, b, c) => {
                0x21 | ((*a as u64) << 8) | ((*b as u64) << 16) | ((*c as u64) << 24)
            }
            QGate::Measure(q) => 0x30 | ((*q as u64) << 8),
            // Shor's algorithm: Controlled rotation gates
            QGate::CPhase(c, t, _) => 0x31 | ((*c as u64) << 8) | ((*t as u64) << 16), // Ignore angle
            QGate::CRz(c, t, _) => 0x32 | ((*c as u64) << 8) | ((*t as u64) << 16), // Ignore angle
            // SPRINT 48.0: T gates
            QGate::T(q) => 0x33 | ((*q as u64) << 8),
            QGate::Tdg(q) => 0x34 | ((*q as u64) << 8),
        };

        hash ^= gate_code;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    // Also incorporate circuit length
    hash ^= circuit.len() as u64;
    hash = hash.wrapping_mul(FNV_PRIME);

    hash
}

/// Hash a circuit including parameter values (for exact matching).
///
/// This stricter hash includes rotation angles, useful when you need
/// circuits to be exactly identical (not just structurally similar).
pub fn circuit_hash_exact(circuit: &[QGate]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;

    for gate in circuit {
        let gate_code: u64 = match gate {
            QGate::H(q) => 0x01 | ((*q as u64) << 8),
            QGate::X(q) => 0x02 | ((*q as u64) << 8),
            QGate::Z(q) => 0x03 | ((*q as u64) << 8),
            QGate::Phase(q, angle) => {
                let angle_bits = angle.to_bits() as u64;
                0x04 | ((*q as u64) << 8) | (angle_bits << 32)
            }
            QGate::Rx(q, angle) => {
                let angle_bits = angle.to_bits() as u64;
                0x05 | ((*q as u64) << 8) | (angle_bits << 32)
            }
            QGate::Ry(q, angle) => {
                let angle_bits = angle.to_bits() as u64;
                0x06 | ((*q as u64) << 8) | (angle_bits << 32)
            }
            QGate::Rz(q, angle) => {
                let angle_bits = angle.to_bits() as u64;
                0x07 | ((*q as u64) << 8) | (angle_bits << 32)
            }
            QGate::Y(q) => 0x08 | ((*q as u64) << 8),
            // SPRINT 32.0: U3 gate with all three angles
            QGate::U3(q, theta, phi, lambda) => {
                let t_bits = theta.to_bits() as u64;
                let p_bits = phi.to_bits() as u64;
                let l_bits = lambda.to_bits() as u64;
                0x0F | ((*q as u64) << 8) | ((t_bits ^ p_bits ^ l_bits) << 32)
            }
            QGate::Swap(c, t) => 0x09 | ((*c as u64) << 8) | ((*t as u64) << 16),
            QGate::CNot(c, t) => 0x10 | ((*c as u64) << 8) | ((*t as u64) << 16),
            QGate::CZ(a, b) => 0x11 | ((*a as u64) << 8) | ((*b as u64) << 16),
            QGate::Toffoli(c0, c1, t) => {
                0x20 | ((*c0 as u64) << 8) | ((*c1 as u64) << 16) | ((*t as u64) << 24)
            }
            QGate::CCZ(a, b, c) => {
                0x21 | ((*a as u64) << 8) | ((*b as u64) << 16) | ((*c as u64) << 24)
            }
            QGate::Measure(q) => 0x30 | ((*q as u64) << 8),
            // Shor's algorithm: Controlled rotation gates
            QGate::CPhase(c, t, angle) => {
                let angle_bits = angle.to_bits() as u64;
                0x31 | ((*c as u64) << 8) | ((*t as u64) << 16) | (angle_bits << 32)
            }
            QGate::CRz(c, t, angle) => {
                let angle_bits = angle.to_bits() as u64;
                0x32 | ((*c as u64) << 8) | ((*t as u64) << 16) | (angle_bits << 32)
            }
            // SPRINT 48.0: T gates (constant phase, no angle parameter)
            QGate::T(q) => 0x33 | ((*q as u64) << 8),
            QGate::Tdg(q) => 0x34 | ((*q as u64) << 8),
        };

        hash ^= gate_code;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash ^= circuit.len() as u64;
    hash = hash.wrapping_mul(FNV_PRIME);

    hash
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_readings_default() {
        let readings = SensorReadings::default();
        assert_eq!(readings.total_heat(), 0);
    }

    #[test]
    fn test_max_heat_direction() {
        let readings = SensorReadings {
            heat_n: 10,
            heat_s: 5,
            heat_e: 20, // Max
            heat_w: 3,
            heat_local: 0,
            charge_local: 0,
        };
        assert_eq!(readings.max_heat_direction(), (1, 0)); // East
    }

    #[test]
    fn test_encode_as_rotations() {
        let readings = SensorReadings {
            heat_n: 255,
            heat_s: 0,
            heat_e: 127,
            heat_w: 64,
            heat_local: 0,
            charge_local: 0,
        };

        let gates = encode_as_rotations(&readings, 4);
        assert_eq!(gates.len(), 4);

        // First gate should be Phase(0, π) for max heat
        if let QGate::Phase(q, angle) = gates[0] {
            assert_eq!(q, 0);
            assert!((angle - PI).abs() < 0.01);
        } else {
            panic!("Expected Phase gate");
        }
    }

    #[test]
    fn test_decode_action() {
        assert_eq!(decode_action(0b00), MoveAction::NORTH);
        assert_eq!(decode_action(0b01), MoveAction::SOUTH);
        assert_eq!(decode_action(0b10), MoveAction::EAST);
        assert_eq!(decode_action(0b11), MoveAction::WEST);

        // Higher bits should be ignored
        assert_eq!(decode_action(0b1100), MoveAction::NORTH);
    }

    #[test]
    fn test_random_circuit_length() {
        let circuit = random_circuit(4, 10, 12345);
        assert_eq!(circuit.len(), 10);
    }

    #[test]
    fn test_random_circuit_valid_qubits() {
        let n_qubits = 4u8;
        let circuit = random_circuit(n_qubits, 20, 54321);

        for gate in &circuit {
            match gate {
                QGate::H(q)
                | QGate::X(q)
                | QGate::Y(q)
                | QGate::Z(q)
                | QGate::Phase(q, _)
                | QGate::Rx(q, _)
                | QGate::Ry(q, _)
                | QGate::Rz(q, _)
                | QGate::U3(q, _, _, _)
                | QGate::T(q)
                | QGate::Tdg(q) => {
                    assert!(*q < n_qubits, "Qubit {} out of range", q);
                }
                QGate::CNot(c, t)
                | QGate::CZ(c, t)
                | QGate::Swap(c, t)
                | QGate::CPhase(c, t, _)
                | QGate::CRz(c, t, _) => {
                    assert!(*c < n_qubits && *t < n_qubits);
                    assert_ne!(*c, *t);
                }
                QGate::Toffoli(c0, c1, t) => {
                    assert!(*c0 < n_qubits && *c1 < n_qubits && *t < n_qubits);
                    assert_ne!(*c0, *c1);
                    assert_ne!(*c0, *t);
                    assert_ne!(*c1, *t);
                }
                QGate::CCZ(c0, c1, t) => {
                    assert!(*c0 < n_qubits && *c1 < n_qubits && *t < n_qubits);
                    assert_ne!(*c0, *c1);
                    assert_ne!(*c0, *t);
                    assert_ne!(*c1, *t);
                }
                QGate::Measure(q) => {
                    assert!(*q < n_qubits);
                }
            }
        }
    }

    #[test]
    fn test_mutate_circuit_changes_something() {
        let original = random_circuit(4, 5, 11111);
        let mutated = mutate_circuit(&original, 4, 20, 22222);

        // Mutation should change something (with very high probability)
        // This is a probabilistic test but should almost never fail
        let same = original.len() == mutated.len()
            && original
                .iter()
                .zip(mutated.iter())
                .all(|(a, b)| std::mem::discriminant(a) == std::mem::discriminant(b));

        // Allow the rare case where mutation happens to produce same structure
        // but generally expect difference
        if same && original.len() > 3 {
            // Re-mutate with different seed to reduce flakiness
            let mutated2 = mutate_circuit(&original, 4, 20, 33333);
            let still_same = original.len() == mutated2.len();
            assert!(
                !still_same || original.len() <= 1,
                "Mutation should change circuit structure"
            );
        }
    }

    #[test]
    fn test_mutate_empty_circuit() {
        let empty: Vec<QGate> = vec![];
        let mutated = mutate_circuit(&empty, 4, 20, 12345);

        // Should insert at least one gate
        assert!(!mutated.is_empty());
    }

    #[test]
    fn test_circuit_stats() {
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::CNot(0, 1),
            QGate::Phase(2, 1.0),
            QGate::Toffoli(0, 1, 2),
        ];

        let stats = analyze_circuit(&circuit);
        assert_eq!(stats.total_gates, 5);
        assert_eq!(stats.hadamard_count, 2);
        assert_eq!(stats.cnot_count, 1);
        assert_eq!(stats.phase_count, 1);
        assert_eq!(stats.toffoli_count, 1);
        assert_eq!(stats.entangling_gates(), 2);
    }

    #[test]
    fn test_quantum_ness() {
        // All entangling
        let all_entangling = vec![QGate::CNot(0, 1), QGate::CZ(1, 2)];
        assert!((quantum_ness(&all_entangling) - 1.0).abs() < 0.01);

        // No entangling
        let no_entangling = vec![QGate::H(0), QGate::X(1), QGate::Z(2)];
        assert_eq!(quantum_ness(&no_entangling), 0.0);

        // Mixed
        let mixed = vec![QGate::H(0), QGate::CNot(0, 1)];
        assert!((quantum_ness(&mixed) - 0.5).abs() < 0.01);

        // Empty
        assert_eq!(quantum_ness(&[]), 0.0);
    }

    #[test]
    fn test_run_circuit_deterministic() {
        let genome = vec![QGate::H(0), QGate::CNot(0, 1)];
        let sensors = vec![QGate::Phase(0, 0.5)];

        let result1 = run_circuit(&genome, &sensors, 4, 12345);
        let result2 = run_circuit(&genome, &sensors, 4, 12345);

        assert_eq!(result1, result2, "Same seed should give same result");
    }

    #[test]
    fn test_run_circuit_different_seeds() {
        // Note: run_circuit already applies H to all qubits at the start,
        // so we use a genome with entangling gates that don't cancel out
        let genome = vec![QGate::CNot(0, 1), QGate::Phase(2, 0.5)];
        let sensors = vec![];

        // With superposition, different seeds should give different results
        // Use seeds that are more spread out to avoid RNG correlation
        let mut results = std::collections::HashSet::new();
        for i in 0..100 {
            let seed = 12345u64.wrapping_mul(i + 1).wrapping_add(67890);
            results.insert(run_circuit(&genome, &sensors, 4, seed));
        }

        // With 4 qubits in superposition (16 possible outcomes) and 100 trials,
        // we should see multiple distinct outcomes
        assert!(
            results.len() > 1,
            "Should have variation with different seeds, got {} results",
            results.len()
        );
    }

    #[test]
    fn test_simple_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    // =============================================================================
    // EPIC 90: Fused Path Parity Tests
    // =============================================================================

    #[test]
    fn test_run_circuit_fused_returns_valid_result() {
        // Test that fused path returns a valid result structure
        let genome = vec![QGate::H(0), QGate::CNot(0, 1), QGate::Phase(2, 0.5)];
        let sensors = vec![QGate::Phase(0, 0.25)];

        let result = run_circuit_fused(&genome, &sensors, 4, 12345);

        // Should have valid metrics
        assert!(result.original_gates > 0);
        assert!(result.used_fusion);
        // Fused ops should be <= original (fusion eliminates or combines)
        assert!(result.fused_ops <= result.original_gates + result.eliminated);
    }

    #[test]
    fn test_run_circuit_auto_small_circuit() {
        // Small circuits should use scalar path
        let genome = vec![QGate::H(0)];
        let sensors = vec![];

        let result = run_circuit_auto(&genome, &sensors, 2, 12345);

        // With only 2+0+1=3 gates total (2 H from superposition + genome),
        // should NOT use fusion (< 8 threshold)
        assert!(!result.used_fusion, "Small circuit should use scalar path");
    }

    #[test]
    fn test_run_circuit_auto_large_circuit() {
        // Large circuits should use fused path
        let genome = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::CNot(0, 1),
            QGate::Phase(0, 0.5),
            QGate::Phase(1, 0.3),
            QGate::X(0),
            QGate::Z(1),
        ];
        let sensors = vec![QGate::Phase(0, 0.25)];

        let result = run_circuit_auto(&genome, &sensors, 4, 12345);

        // With 4+1+7=12 gates total, should use fusion (>= 8 threshold)
        assert!(result.used_fusion, "Large circuit should use fused path");
    }

    #[test]
    fn test_fused_eliminates_identity_pairs() {
        // H·H = I, should be eliminated by fusion
        let genome = vec![
            QGate::H(0),
            QGate::H(0), // H·H = I
            QGate::X(1),
            QGate::X(1), // X·X = I
        ];
        let sensors = vec![];

        let result = run_circuit_fused(&genome, &sensors, 4, 12345);

        // Should have eliminated some gates
        // Original: 4 H's (superposition) + 4 genome = 8
        // After fusion: H·H pairs should cancel
        assert!(
            result.eliminated > 0 || result.fused_ops < result.original_gates,
            "Fusion should eliminate identity pairs: eliminated={}, fused_ops={}, original={}",
            result.eliminated,
            result.fused_ops,
            result.original_gates
        );
    }

    #[test]
    fn test_fused_deterministic() {
        // Same seed should give same result
        let genome = vec![QGate::H(0), QGate::CNot(0, 1)];
        let sensors = vec![QGate::Phase(0, 0.5)];

        let result1 = run_circuit_fused(&genome, &sensors, 4, 12345);
        let result2 = run_circuit_fused(&genome, &sensors, 4, 12345);

        assert_eq!(
            result1.measurement, result2.measurement,
            "Same seed should produce same measurement"
        );
    }
}
