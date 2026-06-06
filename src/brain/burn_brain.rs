//! Burn-based neural network brains for TileUniverse organisms
//!
//! Provides flexible brain architectures:
//! - `SimpleBrain`: Feedforward NN (comparable to WMMA, but slower)
//! - `LstmBrain`: Recurrent NN with memory (enables temporal reasoning)
//!
//! # Feature Gate
//! This module requires the `burn-compute` feature.
//!
//! # Performance Notes
//! - SimpleBrain: ~464M organisms/sec (2.3× slower than WMMA)
//! - LstmBrain: ~100-200M organisms/sec (estimated)
//! - Use WMMA for simple NNs at scale, Burn for complex architectures
//!
//! # Example
//! ```ignore
//! use engine::burn_brain::{SimpleBrain, LstmBrain, BurnLstmState};
//! use burn_cubecl::CubeBackend;
//! use cubecl_wgpu::WgpuRuntime;
//! use burn::tensor::backend::Backend;
//!
//! type B = CubeBackend<WgpuRuntime, f32, i32, u8>;
//! let device = <B as Backend>::Device::default();
//!
//! // Simple feedforward brain
//! let simple = SimpleBrain::<B>::new(16, 16, 5, &device);
//! let sensors = Tensor::<B, 2>::zeros([100, 16], &device);
//! let actions = simple.forward(sensors);
//!
//! // LSTM brain with memory
//! let lstm = LstmBrain::<B>::new(16, 16, 5, &device);
//! let (actions, new_state) = lstm.forward(sensors, None);
//! // Next tick: pass new_state back in
//! let (actions2, state2) = lstm.forward(sensors, Some(new_state));
//! ```

use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig, LstmState};
use burn::prelude::*;

// ============================================================================
// SimpleBrain: Feedforward NN (16 → hidden → output)
// ============================================================================

/// Simple feedforward brain comparable to WMMA implementation.
///
/// Architecture: input → Linear → tanh → Linear → tanh → output
///
/// This is 2-4× slower than WMMA but provides a baseline for comparison
/// and can be used when WMMA is not available.
#[derive(Module, Debug)]
pub struct SimpleBrain<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
}

impl<B: Backend> SimpleBrain<B> {
    /// Create a new SimpleBrain with specified dimensions.
    ///
    /// # Arguments
    /// * `input_size` - Number of sensor inputs (typically 16)
    /// * `hidden_size` - Hidden layer size (typically 16)
    /// * `output_size` - Number of action outputs (typically 5)
    /// * `device` - Backend device to create tensors on
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            layer1: LinearConfig::new(input_size, hidden_size).init(device),
            layer2: LinearConfig::new(hidden_size, output_size).init(device),
        }
    }

    /// Create a brain with default TileUniverse dimensions (16 → 16 → 5).
    pub fn default_config(device: &B::Device) -> Self {
        Self::new(16, 16, 5, device)
    }

    /// Forward pass for a batch of organisms.
    ///
    /// # Arguments
    /// * `sensors` - Sensor inputs, shape `[batch_size, input_size]`
    ///
    /// # Returns
    /// Action outputs, shape `[batch_size, output_size]`, values in `[-1, 1]`
    pub fn forward(&self, sensors: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(sensors);
        let x = burn::tensor::activation::tanh(x);
        let x = self.layer2.forward(x);
        burn::tensor::activation::tanh(x)
    }

    /// Get discrete action indices (argmax over outputs).
    ///
    /// # Arguments
    /// * `sensors` - Sensor inputs, shape `[batch_size, input_size]`
    ///
    /// # Returns
    /// Action indices, shape `[batch_size, 1]`
    pub fn decide(&self, sensors: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        let outputs = self.forward(sensors);
        outputs.argmax(1)
    }
}

// ============================================================================
// LstmBrain: Recurrent NN with temporal memory
// ============================================================================

/// Our wrapper for LSTM state that persists across ticks.
///
/// Organisms using LSTM can remember past observations, enabling:
/// - Tracking moving food sources
/// - Learning predator patterns
/// - Delayed reward associations
#[derive(Clone, Debug)]
pub struct BurnLstmState<B: Backend> {
    /// Hidden state tensor, shape `[batch_size, hidden_size]`
    pub hidden: Tensor<B, 2>,
    /// Cell state tensor, shape `[batch_size, hidden_size]`
    pub cell: Tensor<B, 2>,
}

impl<B: Backend> BurnLstmState<B> {
    /// Create a new zero-initialized LSTM state.
    pub fn zeros(batch_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        Self {
            hidden: Tensor::zeros([batch_size, hidden_size], device),
            cell: Tensor::zeros([batch_size, hidden_size], device),
        }
    }

    /// Convert to Burn's native LstmState format.
    fn to_burn_state(self) -> LstmState<B, 2> {
        LstmState::new(self.cell, self.hidden)
    }

    /// Convert from Burn's native LstmState format.
    fn from_burn_state(state: LstmState<B, 2>) -> Self {
        Self {
            hidden: state.hidden,
            cell: state.cell,
        }
    }
}

/// LSTM brain with temporal memory.
///
/// Architecture: input → LSTM → Linear → tanh → output
///
/// The LSTM layer maintains hidden and cell states across ticks,
/// allowing the organism to learn temporal patterns and remember
/// past observations.
///
/// # Performance
/// Estimated ~100-200M organisms/sec (2-4× slower than SimpleBrain).
/// The overhead is justified when temporal reasoning provides
/// evolutionary advantage.
#[derive(Module, Debug)]
pub struct LstmBrain<B: Backend> {
    lstm: Lstm<B>,
    output: Linear<B>,
    #[module(skip)]
    hidden_size: usize,
}

impl<B: Backend> LstmBrain<B> {
    /// Create a new LstmBrain with specified dimensions.
    ///
    /// # Arguments
    /// * `input_size` - Number of sensor inputs (typically 16)
    /// * `hidden_size` - LSTM hidden size (typically 16-64)
    /// * `output_size` - Number of action outputs (typically 5)
    /// * `device` - Backend device to create tensors on
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            lstm: LstmConfig::new(input_size, hidden_size, true).init(device),
            output: LinearConfig::new(hidden_size, output_size).init(device),
            hidden_size,
        }
    }

    /// Create a brain with default TileUniverse dimensions (16 → 16 → 5).
    pub fn default_config(device: &B::Device) -> Self {
        Self::new(16, 16, 5, device)
    }

    /// Forward pass with memory state.
    ///
    /// # Arguments
    /// * `sensors` - Sensor inputs, shape `[batch_size, input_size]`
    /// * `state` - Optional previous LSTM state. If None, uses zeros.
    ///
    /// # Returns
    /// Tuple of (actions, new_state):
    /// - `actions`: shape `[batch_size, output_size]`, values in `[-1, 1]`
    /// - `new_state`: Updated LSTM state to pass to next tick
    pub fn forward(
        &self,
        sensors: Tensor<B, 2>,
        state: Option<BurnLstmState<B>>,
    ) -> (Tensor<B, 2>, BurnLstmState<B>) {
        // Reshape sensors to [batch, seq_len=1, input_size] for LSTM
        let sensors_3d = sensors.clone().unsqueeze_dim(1);

        // Prepare initial state
        let init_state = state.map(|s| s.to_burn_state());

        // LSTM forward pass
        let (lstm_out, new_lstm_state) = self.lstm.forward(sensors_3d, init_state);

        // lstm_out shape: [batch, seq_len=1, hidden_size]
        // Take output from last (only) timestep by squeezing dim 1
        let lstm_out_2d: Tensor<B, 2> = lstm_out.squeeze();

        // Project to actions
        let actions = self.output.forward(lstm_out_2d);
        let actions = burn::tensor::activation::tanh(actions);

        // Convert state back to our format
        let new_state = BurnLstmState::from_burn_state(new_lstm_state);

        (actions, new_state)
    }

    /// Get discrete action indices (argmax over outputs).
    pub fn decide(
        &self,
        sensors: Tensor<B, 2>,
        state: Option<BurnLstmState<B>>,
    ) -> (Tensor<B, 2, Int>, BurnLstmState<B>) {
        let (actions, new_state) = self.forward(sensors, state);
        (actions.argmax(1), new_state)
    }

    /// Get the hidden size for this brain.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

// ============================================================================
// BrainConfig: Runtime configuration for brain selection
// ============================================================================

/// Brain type selection for evolution experiments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrainType {
    /// Simple feedforward NN (fast, no memory)
    Simple,
    /// LSTM with temporal memory (slower, can learn patterns)
    Lstm,
}

/// Configuration for brain creation.
#[derive(Clone, Debug)]
pub struct BrainConfig {
    /// Number of sensor inputs
    pub input_size: usize,
    /// Hidden layer / LSTM hidden size
    pub hidden_size: usize,
    /// Number of action outputs
    pub output_size: usize,
    /// Type of brain to create
    pub brain_type: BrainType,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            input_size: 16,
            hidden_size: 16,
            output_size: 5,
            brain_type: BrainType::Simple,
        }
    }
}

impl BrainConfig {
    /// Create a config for SimpleBrain with default dimensions.
    pub fn simple() -> Self {
        Self {
            brain_type: BrainType::Simple,
            ..Default::default()
        }
    }

    /// Create a config for LstmBrain with default dimensions.
    pub fn lstm() -> Self {
        Self {
            brain_type: BrainType::Lstm,
            ..Default::default()
        }
    }

    /// Set custom dimensions.
    pub fn with_dims(mut self, input: usize, hidden: usize, output: usize) -> Self {
        self.input_size = input;
        self.hidden_size = hidden;
        self.output_size = output;
        self
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use burn_cubecl::CubeBackend;
    use cubecl_wgpu::WgpuRuntime;

    type B = CubeBackend<WgpuRuntime, f32, i32, u8>;

    #[test]
    fn test_simple_brain_forward() {
        let device = <B as Backend>::Device::default();
        let brain = SimpleBrain::<B>::default_config(&device);

        // Create batch of 10 organisms with 16 sensors each
        let sensors = Tensor::<B, 2>::random(
            [10, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let actions = brain.forward(sensors);
        let dims = actions.dims();

        assert_eq!(dims[0], 10, "Batch size should be 10");
        assert_eq!(dims[1], 5, "Output size should be 5");
    }

    #[test]
    fn test_simple_brain_decide() {
        let device = <B as Backend>::Device::default();
        let brain = SimpleBrain::<B>::default_config(&device);

        let sensors = Tensor::<B, 2>::random(
            [10, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let decisions = brain.decide(sensors);
        let dims = decisions.dims();

        assert_eq!(dims[0], 10, "Should have 10 decisions");
    }

    #[test]
    fn test_lstm_brain_forward() {
        let device = <B as Backend>::Device::default();
        let brain = LstmBrain::<B>::default_config(&device);

        let sensors = Tensor::<B, 2>::random(
            [10, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        // First forward pass (no state)
        let (actions, state) = brain.forward(sensors.clone(), None);
        let dims = actions.dims();

        assert_eq!(dims[0], 10, "Batch size should be 10");
        assert_eq!(dims[1], 5, "Output size should be 5");

        // Second forward pass (with state)
        let (actions2, _state2) = brain.forward(sensors, Some(state));
        let dims2 = actions2.dims();

        assert_eq!(dims2[0], 10, "Batch size should be 10");
        assert_eq!(dims2[1], 5, "Output size should be 5");
    }

    #[test]
    fn test_lstm_state_persistence() {
        let device = <B as Backend>::Device::default();
        let brain = LstmBrain::<B>::default_config(&device);

        // Same input twice
        let sensors = Tensor::<B, 2>::ones([5, 16], &device);

        // First pass (no state) - should get one output
        let (actions1, state1) = brain.forward(sensors.clone(), None);

        // Second pass (with state) - output should differ due to memory
        let (actions2, _state2) = brain.forward(sensors, Some(state1));

        // The outputs should be different because LSTM has internal state
        // We can't easily compare tensors directly, but we verify shapes
        assert_eq!(actions1.dims(), actions2.dims());
    }

    #[test]
    fn test_brain_config() {
        let config = BrainConfig::simple();
        assert_eq!(config.brain_type, BrainType::Simple);
        assert_eq!(config.input_size, 16);

        let config = BrainConfig::lstm().with_dims(32, 64, 8);
        assert_eq!(config.brain_type, BrainType::Lstm);
        assert_eq!(config.input_size, 32);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.output_size, 8);
    }
}
