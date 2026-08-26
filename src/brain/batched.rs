//! EPIC 93.1: Batched CPU Forward Pass
//!
//! Optimizes classical neural network inference by processing all organisms
//! through each layer before moving to the next layer, improving cache locality.
//!
//! Key Insight:
//! Instead of:  for organism { layer1(); layer2(); }
//! We do:       for organism { layer1(); } then for organism { layer2(); }
//!
//! This keeps weights hot in cache and enables SIMD opportunities.
//!
//! Target: 3-5× speedup over per-organism loop

use crate::classical_brain::{ClassicalBrain, Move};

/// Extracted weights from ClassicalBrain for batch processing
#[derive(Clone, Debug)]
pub struct BrainWeights {
    pub weights_ih: [[f32; 8]; 8], // Input→Hidden: [hidden][input]
    pub bias_h: [f32; 8],          // Hidden bias
    pub weights_ho: [[f32; 5]; 8], // Hidden→Output: [hidden][output]
    pub bias_o: [f32; 5],          // Output bias
}

impl From<&ClassicalBrain> for BrainWeights {
    fn from(brain: &ClassicalBrain) -> Self {
        Self {
            weights_ih: brain.weights_ih,
            bias_h: brain.bias_h,
            weights_ho: brain.weights_ho,
            bias_o: brain.bias_o,
        }
    }
}

impl BrainWeights {
    /// Create new random weights (for testing and benchmarking)
    pub fn random(seed: u64) -> Self {
        use crate::quantum_brain::SimpleRng;
        let mut rng = SimpleRng::new(seed);

        let gen_f32 = |rng: &mut SimpleRng| -> f32 {
            let u = rng.gen_range(0, 1000000) as f32 / 1000000.0;
            u * 2.0 - 1.0 // [-1, 1)
        };

        let mut weights_ih = [[0.0; 8]; 8];
        let mut weights_ho = [[0.0; 5]; 8];
        let mut bias_h = [0.0; 8];
        let mut bias_o = [0.0; 5];

        for i in 0..8 {
            for j in 0..8 {
                weights_ih[i][j] = gen_f32(&mut rng);
            }
            bias_h[i] = gen_f32(&mut rng);
        }

        for i in 0..8 {
            for j in 0..5 {
                weights_ho[i][j] = gen_f32(&mut rng);
            }
        }

        for i in 0..5 {
            bias_o[i] = gen_f32(&mut rng);
        }

        Self {
            weights_ih,
            bias_h,
            weights_ho,
            bias_o,
        }
    }
}

/// Batched forward pass for heterogeneous population (different weights per organism)
///
/// # Arguments
/// * `sensors` - Input sensors [n_organisms][8]
/// * `weights` - Brain weights for each organism [n_organisms]
///
/// # Returns
/// Vector of moves for each organism
///
/// # Performance
/// Target: 3-5× speedup over per-organism loop by improving cache locality
pub fn forward_batch(sensors: &[[f32; 8]], weights: &[BrainWeights]) -> Vec<Move> {
    assert_eq!(
        sensors.len(),
        weights.len(),
        "Sensors and weights must match"
    );

    let n = sensors.len();
    if n == 0 {
        return Vec::new();
    }

    // Allocate intermediate buffers
    let mut hidden = vec![[0.0f32; 8]; n];
    let mut output = vec![[0.0f32; 5]; n];

    // Layer 1: Input → Hidden (with activation)
    // Process all organisms through this layer before moving to next
    for (i, (sensor, w)) in sensors.iter().zip(weights.iter()).enumerate() {
        for h in 0..8 {
            let mut sum = w.bias_h[h];
            for inp in 0..8 {
                sum += sensor[inp] * w.weights_ih[h][inp];
            }
            hidden[i][h] = tanh_approx(sum); // tanh activation
        }
    }

    // Layer 2: Hidden → Output (with activation)
    for (i, w) in weights.iter().enumerate() {
        for o in 0..5 {
            let mut sum = w.bias_o[o];
            for h in 0..8 {
                sum += hidden[i][h] * w.weights_ho[h][o];
            }
            output[i][o] = tanh_approx(sum); // tanh activation
        }
    }

    // Decode outputs to moves
    output.iter().map(decode_move).collect()
}

/// Optimized batched forward pass for homogeneous population (shared weights)
///
/// When all organisms have the same brain (early generation), this is more efficient
/// than the heterogeneous version because weights stay hot in cache.
///
/// # Performance
/// Expected to be faster than `forward_batch` for homogeneous populations
pub fn forward_batch_shared_weights(sensors: &[[f32; 8]], weights: &BrainWeights) -> Vec<Move> {
    let n = sensors.len();
    if n == 0 {
        return Vec::new();
    }

    let mut hidden = vec![[0.0f32; 8]; n];
    let mut output = vec![[0.0f32; 5]; n];

    // Layer 1: Weights are shared, so they stay hot in L1 cache
    for (i, sensor) in sensors.iter().enumerate() {
        for h in 0..8 {
            let mut sum = weights.bias_h[h];
            for inp in 0..8 {
                sum += sensor[inp] * weights.weights_ih[h][inp];
            }
            hidden[i][h] = tanh_approx(sum);
        }
    }

    // Layer 2: Same shared weights
    for i in 0..n {
        for o in 0..5 {
            let mut sum = weights.bias_o[o];
            for h in 0..8 {
                sum += hidden[i][h] * weights.weights_ho[h][o];
            }
            output[i][o] = tanh_approx(sum);
        }
    }

    output.iter().map(decode_move).collect()
}

/// Fast tanh approximation (matches ClassicalBrain implementation)
#[inline(always)]
fn tanh_approx(x: f32) -> f32 {
    if x > 3.0 {
        1.0
    } else if x < -3.0 {
        -1.0
    } else {
        x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
    }
}

/// Decode output activations to Move (matches ClassicalBrain implementation)
/// Uses softmax to select discrete direction
fn decode_move(output: &[f32; 5]) -> Move {
    // Find max activation (argmax of softmax)
    let mut max_idx = 0;
    let mut max_val = output[0];

    for i in 1..5 {
        if output[i] > max_val {
            max_val = output[i];
            max_idx = i;
        }
    }

    Move::from_index(max_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_brain::SimpleRng;

    /// Generate random sensors for testing
    fn gen_sensors(n: usize, seed: u64) -> Vec<[f32; 8]> {
        let mut rng = SimpleRng::new(seed);
        let mut sensors = Vec::with_capacity(n);

        for _ in 0..n {
            let mut sensor = [0.0; 8];
            for j in 0..8 {
                let u = rng.gen_range(0, 1000000) as f32 / 1000000.0;
                sensor[j] = u * 2.0 - 1.0; // [-1, 1)
            }
            sensors.push(sensor);
        }

        sensors
    }

    #[test]
    fn test_batch_matches_individual() {
        // Test that batched forward matches individual forward passes
        let n = 100;
        let sensors = gen_sensors(n, 12345);
        let weights: Vec<BrainWeights> = (0..n).map(|i| BrainWeights::random(i as u64)).collect();

        // Compute batched
        let batch_moves = forward_batch(&sensors, &weights);

        // Compute individual (simulate what ClassicalBrain would do)
        let individual_moves: Vec<Move> = sensors
            .iter()
            .zip(weights.iter())
            .map(|(sensor, w)| {
                // Layer 1
                let mut hidden = [0.0f32; 8];
                for h in 0..8 {
                    let mut sum = w.bias_h[h];
                    for inp in 0..8 {
                        sum += sensor[inp] * w.weights_ih[h][inp];
                    }
                    hidden[h] = tanh_approx(sum);
                }

                // Layer 2
                let mut output = [0.0f32; 5];
                for o in 0..5 {
                    let mut sum = w.bias_o[o];
                    for h in 0..8 {
                        sum += hidden[h] * w.weights_ho[h][o];
                    }
                    output[o] = tanh_approx(sum);
                }

                decode_move(&output)
            })
            .collect();

        // Compare results
        assert_eq!(batch_moves.len(), individual_moves.len());
        for (i, (batch, individual)) in batch_moves.iter().zip(individual_moves.iter()).enumerate()
        {
            assert_eq!(
                batch, individual,
                "Organism {} move mismatch: batch={:?}, individual={:?}",
                i, batch, individual
            );
        }
    }

    #[test]
    fn test_shared_weights_matches() {
        // Test that shared weights optimization matches heterogeneous batch
        let n = 100;
        let sensors = gen_sensors(n, 54321);
        let weights = BrainWeights::random(99999);

        // Create heterogeneous weights (all same)
        let weights_vec = vec![weights.clone(); n];

        // Compute both ways
        let hetero_moves = forward_batch(&sensors, &weights_vec);
        let shared_moves = forward_batch_shared_weights(&sensors, &weights);

        // Compare
        assert_eq!(hetero_moves.len(), shared_moves.len());
        for (i, (hetero, shared)) in hetero_moves.iter().zip(shared_moves.iter()).enumerate() {
            assert_eq!(
                hetero, shared,
                "Organism {} move mismatch: hetero={:?}, shared={:?}",
                i, hetero, shared
            );
        }
    }

    #[test]
    fn test_edge_cases() {
        // Empty input
        let empty_sensors: Vec<[f32; 8]> = vec![];
        let empty_weights: Vec<BrainWeights> = vec![];
        assert_eq!(forward_batch(&empty_sensors, &empty_weights).len(), 0);

        // Single organism
        let single_sensors = gen_sensors(1, 11111);
        let single_weights = vec![BrainWeights::random(22222)];
        let result = forward_batch(&single_sensors, &single_weights);
        assert_eq!(result.len(), 1);

        // Large batch (10k organisms)
        let large_sensors = gen_sensors(10_000, 33333);
        let large_weights: Vec<BrainWeights> = (0..10_000)
            .map(|i| BrainWeights::random(i as u64 + 44444))
            .collect();
        let result = forward_batch(&large_sensors, &large_weights);
        assert_eq!(result.len(), 10_000);
    }

    #[test]
    fn test_move_validity() {
        // Ensure all moves are valid enum variants
        let n = 1000;
        let sensors = gen_sensors(n, 77777);
        let weights: Vec<BrainWeights> = (0..n)
            .map(|i| BrainWeights::random(i as u64 + 88888))
            .collect();

        let moves = forward_batch(&sensors, &weights);

        // Check that we got valid moves (no panics, correct count)
        assert_eq!(moves.len(), n);

        // Count distribution of moves
        let mut counts = [0; 5];
        for mv in &moves {
            counts[mv.to_index()] += 1;
        }

        // At least some variety expected with random weights/sensors
        let non_zero = counts.iter().filter(|&&c| c > 0).count();
        assert!(
            non_zero >= 2,
            "Expected some variety in moves, got distribution: {:?}",
            counts
        );
    }
}
