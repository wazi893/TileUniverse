//! Unified brain abstraction for all organism types.
//!
//! EPIC 91: Three-way comparison - Full Quantum vs Separable Quantum vs Classical NN
//! CHUNGUS 4: Added SpikingNN for neuromorphic spiking neural networks
//! HYBRID: Added HybridExploration - NN exploitation + QUBO exploration

use crate::brain::hybrid_brain::HybridExplorationBrain;
use crate::brain::qubo_brain::{DecisionBudget, QuboBrain};
use crate::brain::spiking::SpikingBrain;
use crate::classical_brain::{ClassicalBrain, Move};
use crate::constrained_circuits::{
    CircuitConstraints, entanglement_fraction, mutate_circuit_constrained,
    random_circuit_constrained,
};
use crate::handicapped_classical_brain::HandicappedClassicalBrain;
use crate::qec::noise::SimpleRng as PBitRng;
use crate::quantum::{QGate, QRng};
use crate::quantum_brain::{
    MoveAction, SensorReadings, decode_action, encode_as_rotations, run_circuit,
};

/// Brain type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BrainType {
    /// Full quantum circuit with entangling gates allowed
    FullQuantum,
    /// Quantum circuit restricted to single-qubit gates only
    SeparableQuantum,
    /// Classical feedforward neural network
    ClassicalNN,
    /// Handicapped classical NN (matched capacity, poor init, discrete mutations)
    HandicappedClassicalNN,
    /// Spiking neural network (CHUNGUS 4: Neuromorphic)
    SpikingNN,
    /// QUBO brain using p-bit annealing for decision making
    QuboBrain,
    /// Hybrid brain: NN for exploitation, QUBO for exploration
    HybridExploration,
}

impl BrainType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::FullQuantum => "full_quantum",
            Self::SeparableQuantum => "separable_quantum",
            Self::ClassicalNN => "classical_nn",
            Self::HandicappedClassicalNN => "handicapped_nn",
            Self::SpikingNN => "spiking_nn",
            Self::QuboBrain => "qubo_brain",
            Self::HybridExploration => "hybrid_exploration",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::FullQuantum => "Full Quantum (Entangling)",
            Self::SeparableQuantum => "Separable Quantum (Non-Entangling)",
            Self::ClassicalNN => "Classical Neural Network",
            Self::HandicappedClassicalNN => "Handicapped Classical NN (61 params)",
            Self::SpikingNN => "Spiking Neural Network (Neuromorphic)",
            Self::QuboBrain => "QUBO Brain (P-bit Annealing)",
            Self::HybridExploration => "Hybrid NN+QUBO (Exploit+Explore)",
        }
    }

    /// Get circuit constraints for quantum brain types
    pub fn to_constraints(&self) -> Option<CircuitConstraints> {
        match self {
            Self::FullQuantum => Some(CircuitConstraints::full_quantum()),
            Self::SeparableQuantum => Some(CircuitConstraints::classical_only()),
            Self::ClassicalNN => None,
            Self::HandicappedClassicalNN => None,
            Self::SpikingNN => None,
            Self::QuboBrain => None,
            Self::HybridExploration => None,
        }
    }
}

/// Unified brain that can be any of the seven types
#[derive(Clone, Debug)]
pub enum Brain {
    FullQuantum {
        genome: Vec<QGate>,
        n_qubits: u8,
    },
    SeparableQuantum {
        genome: Vec<QGate>,
        n_qubits: u8,
    },
    ClassicalNN(ClassicalBrain),
    HandicappedClassicalNN(HandicappedClassicalBrain),
    SpikingNN(SpikingBrain),
    QuboBrain {
        brain: QuboBrain,
        budget: DecisionBudget,
    },
    HybridExploration(HybridExplorationBrain),
}

impl Brain {
    /// Create a new random brain of the specified type
    pub fn random(
        brain_type: BrainType,
        n_qubits: u8,
        circuit_length: usize,
        rng: &mut QRng,
    ) -> Self {
        match brain_type {
            BrainType::FullQuantum => {
                let constraints = CircuitConstraints::full_quantum();
                let seed = (rng.next_f32() * u64::MAX as f32) as u64;
                let genome =
                    random_circuit_constrained(n_qubits, circuit_length, &constraints, seed);
                Brain::FullQuantum { genome, n_qubits }
            }
            BrainType::SeparableQuantum => {
                let constraints = CircuitConstraints::classical_only();
                let seed = (rng.next_f32() * u64::MAX as f32) as u64;
                let genome =
                    random_circuit_constrained(n_qubits, circuit_length, &constraints, seed);
                Brain::SeparableQuantum { genome, n_qubits }
            }
            BrainType::ClassicalNN => Brain::ClassicalNN(ClassicalBrain::random(rng)),
            BrainType::HandicappedClassicalNN => {
                Brain::HandicappedClassicalNN(HandicappedClassicalBrain::random(rng))
            }
            BrainType::SpikingNN => Brain::SpikingNN(SpikingBrain::random(rng)),
            BrainType::QuboBrain => {
                let seed = (rng.next_f32() * u64::MAX as f32) as u64;
                let mut pbit_rng = PBitRng::new(seed);
                // 5 variables: just action bits, no auxiliary
                // Penalty = 10.0 - moderate constraint enforcement
                // Note: too high penalty suppresses all actions; too low allows invalid states
                let brain = QuboBrain::random(5, 10.0, &mut pbit_rng);
                Brain::QuboBrain {
                    brain,
                    budget: DecisionBudget::default(),
                }
            }
            BrainType::HybridExploration => {
                Brain::HybridExploration(HybridExplorationBrain::random(rng))
            }
        }
    }

    /// Get the brain type
    pub fn brain_type(&self) -> BrainType {
        match self {
            Brain::FullQuantum { .. } => BrainType::FullQuantum,
            Brain::SeparableQuantum { .. } => BrainType::SeparableQuantum,
            Brain::ClassicalNN(_) => BrainType::ClassicalNN,
            Brain::HandicappedClassicalNN(_) => BrainType::HandicappedClassicalNN,
            Brain::SpikingNN(_) => BrainType::SpikingNN,
            Brain::QuboBrain { .. } => BrainType::QuboBrain,
            Brain::HybridExploration(_) => BrainType::HybridExploration,
        }
    }

    /// Execute brain and return movement decision
    ///
    /// For quantum brains, converts SensorReadings to quantum gates and runs circuit.
    /// For classical NN, converts SensorReadings to 8-element float array.
    /// For spiking NN, uses rate coding for input and winner-take-all for output.
    /// For QUBO brain, uses p-bit annealing to find minimum energy action.
    pub fn decide(&mut self, sensors: &SensorReadings, rng: &mut QRng) -> Move {
        match self {
            Brain::FullQuantum { genome, n_qubits }
            | Brain::SeparableQuantum { genome, n_qubits } => {
                // Encode sensors as rotation gates
                let sensor_gates = encode_as_rotations(sensors, *n_qubits);

                // Run quantum circuit
                let seed = (rng.next_f32() * u64::MAX as f32) as u64;
                let measurement = run_circuit(genome, &sensor_gates, *n_qubits, seed);

                // Decode to movement action (using 2-bit encoding)
                let action = decode_action(measurement);

                // Convert MoveAction to Move
                if action == MoveAction::NORTH {
                    Move::Up
                } else if action == MoveAction::SOUTH {
                    Move::Down
                } else if action == MoveAction::EAST {
                    Move::Right
                } else if action == MoveAction::WEST {
                    Move::Left
                } else {
                    Move::Stay
                }
            }
            Brain::ClassicalNN(nn) => {
                // Convert sensor readings to normalized float array
                let inputs = sensors_to_array(sensors);
                nn.forward(&inputs, rng)
            }
            Brain::HandicappedClassicalNN(nn) => {
                // Convert sensor readings to normalized float array
                let inputs = sensors_to_array(sensors);
                nn.forward(&inputs, rng)
            }
            Brain::SpikingNN(snn) => {
                // Spiking NN uses rate coding
                snn.decide(sensors)
            }
            Brain::QuboBrain { brain, budget } => {
                // QUBO brain uses p-bit annealing
                let seed = (rng.next_f32() * u64::MAX as f32) as u64;
                let mut pbit_rng = PBitRng::new(seed);
                brain.decide(sensors, &mut pbit_rng, budget)
            }
            Brain::HybridExploration(hybrid) => {
                // Hybrid brain: NN for exploitation, QUBO for exploration
                hybrid.decide(sensors, rng)
            }
        }
    }

    /// Create mutated offspring
    pub fn reproduce(
        &self,
        rng: &mut QRng,
        mutation_rate: f32,
        _max_circuit_length: usize,
    ) -> Self {
        match self {
            Brain::FullQuantum { genome, n_qubits } => {
                let constraints = CircuitConstraints::full_quantum();
                let seed = (rng.next_f32() * u64::MAX as f32) as u64;
                let mutated_genome =
                    mutate_circuit_constrained(genome, *n_qubits, &constraints, seed);
                Brain::FullQuantum {
                    genome: mutated_genome,
                    n_qubits: *n_qubits,
                }
            }
            Brain::SeparableQuantum { genome, n_qubits } => {
                let constraints = CircuitConstraints::classical_only();
                let seed = (rng.next_f32() * u64::MAX as f32) as u64;
                let mutated_genome =
                    mutate_circuit_constrained(genome, *n_qubits, &constraints, seed);
                Brain::SeparableQuantum {
                    genome: mutated_genome,
                    n_qubits: *n_qubits,
                }
            }
            Brain::ClassicalNN(nn) => Brain::ClassicalNN(nn.reproduce(rng, mutation_rate, 0.1)),
            Brain::HandicappedClassicalNN(nn) => {
                Brain::HandicappedClassicalNN(nn.reproduce(rng, mutation_rate, 0.1))
            }
            Brain::SpikingNN(snn) => Brain::SpikingNN(snn.reproduce(rng, mutation_rate)),
            Brain::QuboBrain { brain, budget } => {
                let seed = (rng.next_f32() * u64::MAX as f32) as u64;
                let mut pbit_rng = PBitRng::new(seed);
                let mutated_brain = brain.reproduce(&mut pbit_rng, mutation_rate, 0.1);
                Brain::QuboBrain {
                    brain: mutated_brain,
                    budget: *budget,
                }
            }
            Brain::HybridExploration(hybrid) => {
                Brain::HybridExploration(hybrid.reproduce(rng, mutation_rate, 0.1))
            }
        }
    }

    /// Get parameter count (for comparison)
    pub fn parameter_count(&self) -> usize {
        match self {
            Brain::FullQuantum { genome, .. } | Brain::SeparableQuantum { genome, .. } => {
                // Rough estimate: 1 parameter per parameterized gate
                genome
                    .iter()
                    .filter(|g| matches!(g, QGate::Phase(_, _)))
                    .count()
            }
            Brain::ClassicalNN(nn) => nn.parameter_count(),
            Brain::HandicappedClassicalNN(nn) => nn.parameter_count(),
            Brain::SpikingNN(snn) => snn.parameter_count(),
            Brain::QuboBrain { brain, .. } => brain.parameter_count(),
            Brain::HybridExploration(hybrid) => hybrid.parameter_count(),
        }
    }

    /// Get "quantumness" metric (entanglement fraction for quantum, 0.0 for classical)
    pub fn quantumness(&self) -> f32 {
        match self {
            Brain::FullQuantum { genome, .. } => entanglement_fraction(genome),
            Brain::SeparableQuantum { .. } => 0.0, // No entanglement possible
            Brain::ClassicalNN(_) => 0.0,          // Not quantum at all
            Brain::HandicappedClassicalNN(_) => 0.0, // Not quantum at all
            Brain::SpikingNN(_) => 0.0,            // Neuromorphic, not quantum
            Brain::QuboBrain { .. } => 0.0,        // P-bit based, not quantum
            Brain::HybridExploration(_) => 0.0,    // Hybrid, but not quantum
        }
    }

    /// Get genome length (for quantum brains)
    pub fn genome_length(&self) -> usize {
        match self {
            Brain::FullQuantum { genome, .. } | Brain::SeparableQuantum { genome, .. } => {
                genome.len()
            }
            Brain::ClassicalNN(_) => 0,
            Brain::HandicappedClassicalNN(_) => 0,
            Brain::SpikingNN(_) => 0, // SNNs don't have a genome
            Brain::QuboBrain { brain, .. } => brain.n_vars, // Number of QUBO variables
            Brain::HybridExploration(hybrid) => hybrid.qubo.n_vars, // QUBO component vars
        }
    }
}

/// Convert SensorReadings to normalized 8-element float array for classical NN.
///
/// Maps heat and charge values [0, 255] to [0.0, 1.0].
fn sensors_to_array(sensors: &SensorReadings) -> [f32; 8] {
    let normalize = |v: u32| (v as f32) / 255.0;
    [
        normalize(sensors.heat_n),
        normalize(sensors.heat_s),
        normalize(sensors.heat_e),
        normalize(sensors.heat_w),
        normalize(sensors.heat_local),
        normalize(sensors.charge_local),
        0.0, // Padding to match 8 inputs
        0.0, // Padding to match 8 inputs
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_type_names() {
        assert_eq!(BrainType::FullQuantum.name(), "full_quantum");
        assert_eq!(BrainType::SeparableQuantum.name(), "separable_quantum");
        assert_eq!(BrainType::ClassicalNN.name(), "classical_nn");
    }

    #[test]
    fn test_brain_creation() {
        let mut rng = QRng::new(42);

        let fq = Brain::random(BrainType::FullQuantum, 4, 10, &mut rng);
        assert_eq!(fq.brain_type(), BrainType::FullQuantum);

        let sq = Brain::random(BrainType::SeparableQuantum, 4, 10, &mut rng);
        assert_eq!(sq.brain_type(), BrainType::SeparableQuantum);

        let cn = Brain::random(BrainType::ClassicalNN, 4, 10, &mut rng);
        assert_eq!(cn.brain_type(), BrainType::ClassicalNN);
    }

    #[test]
    fn test_classical_nn_parameter_count() {
        let mut rng = QRng::new(42);
        let brain = Brain::random(BrainType::ClassicalNN, 4, 10, &mut rng);
        assert_eq!(brain.parameter_count(), 117);
    }

    #[test]
    fn test_quantumness() {
        let mut rng = QRng::new(42);

        // Classical NN should have 0 quantumness
        let cn = Brain::random(BrainType::ClassicalNN, 4, 10, &mut rng);
        assert_eq!(cn.quantumness(), 0.0);

        // Separable quantum should have 0 quantumness (no entanglement)
        let sq = Brain::random(BrainType::SeparableQuantum, 4, 10, &mut rng);
        assert_eq!(sq.quantumness(), 0.0);
    }
}
