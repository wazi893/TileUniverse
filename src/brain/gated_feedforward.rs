//! GatedFeedforward Neural Network
//!
//! Winner of EPIC 101.5: Outperforms LSTM by 39% on memory tasks
//! while being 3× smaller and having no recurrence.
//!
//! Architecture: input(8) → gated_hidden(24) → output(5)
//! Parameters: 557 total
//!
//! Mechanism:
//!   gate      = sigmoid(Wg · input + bg)
//!   candidate = tanh(Wh · input + bh)
//!   hidden    = gate ⊙ candidate
//!   output    = tanh(Wo · hidden + bo)

use std::fmt;

/// GatedFeedforward weights for a single organism
///
/// Total parameters: 557
/// - Wg: 8×24 = 192
/// - bg: 24
/// - Wh: 8×24 = 192
/// - bh: 24
/// - Wo: 24×5 = 120
/// - bo: 5
#[derive(Clone)]
pub struct GatedFeedforwardWeights {
    pub wg: Vec<f32>, // [8, 24] Gate weights
    pub bg: Vec<f32>, // [24] Gate biases
    pub wh: Vec<f32>, // [8, 24] Hidden weights
    pub bh: Vec<f32>, // [24] Hidden biases
    pub wo: Vec<f32>, // [24, 5] Output weights
    pub bo: Vec<f32>, // [5] Output biases
}

impl GatedFeedforwardWeights {
    pub const INPUT_SIZE: usize = 8;
    pub const HIDDEN_SIZE: usize = 24;
    pub const OUTPUT_SIZE: usize = 5;
    pub const PARAM_COUNT: usize = 8 * 24 + 24 + 8 * 24 + 24 + 24 * 5 + 5; // 557

    /// Create new weights initialized to zero
    pub fn new() -> Self {
        Self {
            wg: vec![0.0; Self::INPUT_SIZE * Self::HIDDEN_SIZE],
            bg: vec![0.0; Self::HIDDEN_SIZE],
            wh: vec![0.0; Self::INPUT_SIZE * Self::HIDDEN_SIZE],
            bh: vec![0.0; Self::HIDDEN_SIZE],
            wo: vec![0.0; Self::HIDDEN_SIZE * Self::OUTPUT_SIZE],
            bo: vec![0.0; Self::OUTPUT_SIZE],
        }
    }

    /// Create new weights with random initialization (Xavier)
    pub fn new_random(seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);

        // Xavier initialization scales
        let xavier_input_hidden = (6.0 / (Self::INPUT_SIZE + Self::HIDDEN_SIZE) as f32).sqrt();
        let xavier_hidden_output = (6.0 / (Self::HIDDEN_SIZE + Self::OUTPUT_SIZE) as f32).sqrt();

        Self {
            wg: (0..Self::INPUT_SIZE * Self::HIDDEN_SIZE)
                .map(|_| rng.next_f32_range(-xavier_input_hidden, xavier_input_hidden))
                .collect(),
            bg: vec![0.0; Self::HIDDEN_SIZE],
            wh: (0..Self::INPUT_SIZE * Self::HIDDEN_SIZE)
                .map(|_| rng.next_f32_range(-xavier_input_hidden, xavier_input_hidden))
                .collect(),
            bh: vec![0.0; Self::HIDDEN_SIZE],
            wo: (0..Self::HIDDEN_SIZE * Self::OUTPUT_SIZE)
                .map(|_| rng.next_f32_range(-xavier_hidden_output, xavier_hidden_output))
                .collect(),
            bo: vec![0.0; Self::OUTPUT_SIZE],
        }
    }

    /// Forward pass: sensors → action scores
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), Self::INPUT_SIZE);

        let h = Self::HIDDEN_SIZE;
        let o = Self::OUTPUT_SIZE;
        let i = Self::INPUT_SIZE;

        // Gate: sigmoid(Wg · input + bg)
        let mut gate = vec![0.0f32; h];
        for j in 0..h {
            let mut sum = self.bg[j];
            for k in 0..i {
                sum += input[k] * self.wg[k * h + j];
            }
            gate[j] = sigmoid(sum);
        }

        // Candidate: tanh(Wh · input + bh)
        let mut candidate = vec![0.0f32; h];
        for j in 0..h {
            let mut sum = self.bh[j];
            for k in 0..i {
                sum += input[k] * self.wh[k * h + j];
            }
            candidate[j] = sum.tanh();
        }

        // Gated hidden: gate ⊙ candidate
        let mut hidden = vec![0.0f32; h];
        for j in 0..h {
            hidden[j] = gate[j] * candidate[j];
        }

        // Output: tanh(Wo · hidden + bo)
        let mut output = vec![0.0f32; o];
        for k in 0..o {
            let mut sum = self.bo[k];
            for j in 0..h {
                sum += hidden[j] * self.wo[j * o + k];
            }
            output[k] = sum.tanh();
        }

        output
    }

    /// Decide action (argmax of forward pass)
    pub fn decide(&self, input: &[f32]) -> usize {
        let output = self.forward(input);
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Mutate weights in place
    pub fn mutate(&mut self, rate: f32, magnitude: f32, seed: u64) {
        let mut rng = SimpleRng::new(seed);

        for w in self
            .wg
            .iter_mut()
            .chain(self.bg.iter_mut())
            .chain(self.wh.iter_mut())
            .chain(self.bh.iter_mut())
            .chain(self.wo.iter_mut())
            .chain(self.bo.iter_mut())
        {
            if rng.next_f32() < rate {
                *w += rng.next_f32_range(-magnitude, magnitude);
            }
        }
    }

    /// Crossover two parents to create child
    pub fn crossover(parent_a: &Self, parent_b: &Self, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);

        fn cross_vec(a: &[f32], b: &[f32], rng: &mut SimpleRng) -> Vec<f32> {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| if rng.next_f32() < 0.5 { *x } else { *y })
                .collect()
        }

        Self {
            wg: cross_vec(&parent_a.wg, &parent_b.wg, &mut rng),
            bg: cross_vec(&parent_a.bg, &parent_b.bg, &mut rng),
            wh: cross_vec(&parent_a.wh, &parent_b.wh, &mut rng),
            bh: cross_vec(&parent_a.bh, &parent_b.bh, &mut rng),
            wo: cross_vec(&parent_a.wo, &parent_b.wo, &mut rng),
            bo: cross_vec(&parent_a.bo, &parent_b.bo, &mut rng),
        }
    }

    /// Get all weights as a flat vector (for GPU transfer)
    pub fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(Self::PARAM_COUNT);
        flat.extend(&self.wg);
        flat.extend(&self.bg);
        flat.extend(&self.wh);
        flat.extend(&self.bh);
        flat.extend(&self.wo);
        flat.extend(&self.bo);
        flat
    }

    /// Create from flat vector
    pub fn from_flat(flat: &[f32]) -> Self {
        assert_eq!(flat.len(), Self::PARAM_COUNT);

        let mut offset = 0;
        let wg_size = Self::INPUT_SIZE * Self::HIDDEN_SIZE;
        let bg_size = Self::HIDDEN_SIZE;
        let wh_size = Self::INPUT_SIZE * Self::HIDDEN_SIZE;
        let bh_size = Self::HIDDEN_SIZE;
        let wo_size = Self::HIDDEN_SIZE * Self::OUTPUT_SIZE;
        let bo_size = Self::OUTPUT_SIZE;

        let wg = flat[offset..offset + wg_size].to_vec();
        offset += wg_size;
        let bg = flat[offset..offset + bg_size].to_vec();
        offset += bg_size;
        let wh = flat[offset..offset + wh_size].to_vec();
        offset += wh_size;
        let bh = flat[offset..offset + bh_size].to_vec();
        offset += bh_size;
        let wo = flat[offset..offset + wo_size].to_vec();
        offset += wo_size;
        let bo = flat[offset..offset + bo_size].to_vec();

        Self {
            wg,
            bg,
            wh,
            bh,
            wo,
            bo,
        }
    }
}

impl Default for GatedFeedforwardWeights {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for GatedFeedforwardWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GatedFeedforwardWeights {{ params: {} }}",
            Self::PARAM_COUNT
        )
    }
}

// Simple sigmoid function
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Simple PRNG for deterministic weight initialization
pub struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        // Scale 32-bit value to [0, 1)
        (self.next_u64() >> 32) as f32 / (u32::MAX as f32 + 1.0)
    }

    fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

/// Population of GatedFeedforward weights for batch operations
#[derive(Clone)]
pub struct GatedFeedforwardPopulation {
    pub organisms: Vec<GatedFeedforwardWeights>,
}

impl GatedFeedforwardPopulation {
    /// Create population with random weights
    pub fn new_random(size: usize, base_seed: u64) -> Self {
        let organisms = (0..size)
            .map(|i| GatedFeedforwardWeights::new_random(base_seed + i as u64))
            .collect();
        Self { organisms }
    }

    /// Get all weights as contiguous arrays for GPU transfer
    /// Returns (wg, bg, wh, bh, wo, bo) each as [N, ...] flattened
    pub fn to_gpu_arrays(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.organisms.len();
        let h = GatedFeedforwardWeights::HIDDEN_SIZE;
        let i = GatedFeedforwardWeights::INPUT_SIZE;
        let o = GatedFeedforwardWeights::OUTPUT_SIZE;

        let mut wg = Vec::with_capacity(n * i * h);
        let mut bg = Vec::with_capacity(n * h);
        let mut wh = Vec::with_capacity(n * i * h);
        let mut bh = Vec::with_capacity(n * h);
        let mut wo = Vec::with_capacity(n * h * o);
        let mut bo = Vec::with_capacity(n * o);

        for org in &self.organisms {
            wg.extend(&org.wg);
            bg.extend(&org.bg);
            wh.extend(&org.wh);
            bh.extend(&org.bh);
            wo.extend(&org.wo);
            bo.extend(&org.bo);
        }

        (wg, bg, wh, bh, wo, bo)
    }

    /// Population size
    pub fn len(&self) -> usize {
        self.organisms.len()
    }

    /// Is population empty
    pub fn is_empty(&self) -> bool {
        self.organisms.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_count() {
        let w = GatedFeedforwardWeights::new();
        let flat = w.to_flat();
        assert_eq!(flat.len(), GatedFeedforwardWeights::PARAM_COUNT);
        assert_eq!(flat.len(), 557);
    }

    #[test]
    fn test_forward_shape() {
        let w = GatedFeedforwardWeights::new_random(42);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let output = w.forward(&input);
        assert_eq!(output.len(), 5);

        // Output should be in [-1, 1] due to tanh
        for &v in &output {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_decide() {
        let w = GatedFeedforwardWeights::new_random(42);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let action = w.decide(&input);
        assert!(action < 5);
    }

    #[test]
    fn test_deterministic() {
        let w1 = GatedFeedforwardWeights::new_random(12345);
        let w2 = GatedFeedforwardWeights::new_random(12345);

        let input = vec![0.5; 8];
        let out1 = w1.forward(&input);
        let out2 = w2.forward(&input);

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_flat_roundtrip() {
        let original = GatedFeedforwardWeights::new_random(999);
        let flat = original.to_flat();
        let restored = GatedFeedforwardWeights::from_flat(&flat);

        let input = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let out1 = original.forward(&input);
        let out2 = restored.forward(&input);

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mutate() {
        let mut w = GatedFeedforwardWeights::new_random(42);
        let before = w.to_flat();

        w.mutate(0.1, 0.1, 123); // 10% mutation rate

        let after = w.to_flat();

        // Some weights should have changed (with 10% rate and 557 params, expect ~55)
        let changed = before
            .iter()
            .zip(after.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-9)
            .count();
        assert!(changed > 0, "At least some weights should change");
        // With 10% mutation rate, very unlikely all 557 change
        assert!(
            changed < GatedFeedforwardWeights::PARAM_COUNT,
            "Not all weights should change with 10% rate"
        );
    }

    #[test]
    fn test_crossover() {
        let a = GatedFeedforwardWeights::new_random(100);
        let b = GatedFeedforwardWeights::new_random(200);
        let child = GatedFeedforwardWeights::crossover(&a, &b, 300);

        let a_flat = a.to_flat();
        let b_flat = b.to_flat();
        let c_flat = child.to_flat();

        // Child should have mix of parent weights
        // Note: from_a + from_b can exceed PARAM_COUNT if a and b have same values by chance
        let from_a = c_flat
            .iter()
            .zip(a_flat.iter())
            .filter(|(c, a)| (*c - *a).abs() < 1e-9)
            .count();
        let from_b = c_flat
            .iter()
            .zip(b_flat.iter())
            .filter(|(c, b)| (*c - *b).abs() < 1e-9)
            .count();

        // With 50% probability for each parent, expect ~278 from each (but varies)
        // Just ensure both parents contributed significantly
        assert!(from_a > 100, "Parent A should contribute many weights");
        assert!(from_b > 100, "Parent B should contribute many weights");
    }

    #[test]
    fn test_population_gpu_arrays() {
        let pop = GatedFeedforwardPopulation::new_random(10, 42);
        let (wg, bg, wh, bh, wo, bo) = pop.to_gpu_arrays();

        assert_eq!(wg.len(), 10 * 8 * 24);
        assert_eq!(bg.len(), 10 * 24);
        assert_eq!(wh.len(), 10 * 8 * 24);
        assert_eq!(bh.len(), 10 * 24);
        assert_eq!(wo.len(), 10 * 24 * 5);
        assert_eq!(bo.len(), 10 * 5);
    }
}
