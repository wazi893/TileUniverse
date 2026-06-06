//! Quantum vs Classical: Adversarial Predator-Prey Task
//!
//! Tests whether quantum decision-making provides advantage against
//! an adaptive predator that learns prey movement patterns.
//!
//! The predator uses a Markov model to predict prey's next move based
//! on recent history. Quantum's interference patterns should be harder
//! to predict than classical softmax sampling.
//!
//! Run with: cargo run --release --example quantum_predator_prey

use engine::classical_brain::Move;
use engine::snn::{NeuronConfig, QuantumSNN, QuantumSNNConfig, SNNConfig, STDPConfig};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

// ============================================================================
// Adaptive Predator - Learns prey patterns using Markov model
// ============================================================================

/// Predator that learns prey movement patterns
struct AdaptivePredator {
    x: i32,
    y: i32,
    /// Transition counts: (last_move, current_move) -> count
    transition_counts: HashMap<(usize, usize), u32>,
    /// Total counts per last_move for normalization
    total_counts: [u32; 5],
    /// History of prey moves (for context)
    prey_history: Vec<usize>,
    /// Prediction accuracy tracking
    correct_predictions: u32,
    total_predictions: u32,
    /// Movement speed (tiles per turn)
    speed: i32,
}

impl AdaptivePredator {
    fn new(x: i32, y: i32) -> Self {
        Self {
            x,
            y,
            transition_counts: HashMap::new(),
            total_counts: [1; 5], // Start with 1 to avoid division by zero
            prey_history: Vec::new(),
            correct_predictions: 0,
            total_predictions: 0,
            speed: 1,
        }
    }

    /// Update model with observed prey move
    fn observe_prey_move(&mut self, prey_move: usize) {
        if let Some(&last_move) = self.prey_history.last() {
            // Update transition counts
            *self
                .transition_counts
                .entry((last_move, prey_move))
                .or_insert(0) += 1;
            self.total_counts[last_move] += 1;
        }
        self.prey_history.push(prey_move);

        // Keep history bounded
        if self.prey_history.len() > 100 {
            self.prey_history.remove(0);
        }
    }

    /// Predict prey's next move based on learned patterns
    fn predict_prey_move(&self) -> usize {
        if let Some(&last_move) = self.prey_history.last() {
            // Find most likely next move given last move
            let mut best_move = 0;
            let mut best_prob = 0.0;

            for next_move in 0..5 {
                let count = self
                    .transition_counts
                    .get(&(last_move, next_move))
                    .copied()
                    .unwrap_or(0) as f64;
                let prob = count / self.total_counts[last_move] as f64;

                if prob > best_prob {
                    best_prob = prob;
                    best_move = next_move;
                }
            }
            best_move
        } else {
            // No history - predict Stay
            4
        }
    }

    /// Move toward where we predict the prey will be
    fn hunt(&mut self, prey_x: i32, prey_y: i32, width: i32, height: i32) {
        let predicted_move = self.predict_prey_move();

        // Calculate predicted prey position
        let (dx, dy) = match predicted_move {
            0 => (0, -1), // Up
            1 => (0, 1),  // Down
            2 => (-1, 0), // Left
            3 => (1, 0),  // Right
            _ => (0, 0),  // Stay
        };

        let predicted_x = (prey_x + dx + width) % width;
        let predicted_y = (prey_y + dy + height) % height;

        // Move toward predicted position
        let move_x = (predicted_x - self.x).signum() * self.speed;
        let move_y = (predicted_y - self.y).signum() * self.speed;

        self.x = (self.x + move_x + width) % width;
        self.y = (self.y + move_y + height) % height;
    }

    /// Check if we caught the prey and update prediction accuracy
    fn check_catch(&mut self, prey_x: i32, prey_y: i32, prey_move: usize) -> bool {
        // Track prediction accuracy
        let prediction = self.predict_prey_move();
        self.total_predictions += 1;
        if prediction == prey_move {
            self.correct_predictions += 1;
        }

        // Check if caught (within 1 tile)
        let dx = (self.x - prey_x).abs();
        let dy = (self.y - prey_y).abs();
        dx <= 1 && dy <= 1
    }

    fn prediction_accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            self.correct_predictions as f64 / self.total_predictions as f64
        }
    }

    fn reset_position(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }
}

// ============================================================================
// Neural Network Predator - Learns complex patterns via gradient descent
// ============================================================================

/// Neural network predator with online learning
/// Uses a 2-layer MLP to predict prey's next move from recent history
struct NeuralPredator {
    x: i32,
    y: i32,
    /// Input: last N moves one-hot encoded (N * 5 inputs)
    /// Hidden: 32 neurons with ReLU
    /// Output: 5 softmax probabilities
    history_length: usize,
    weights_ih: Vec<Vec<f64>>, // input -> hidden
    weights_ho: Vec<Vec<f64>>, // hidden -> output
    bias_h: Vec<f64>,
    bias_o: Vec<f64>,
    /// Recent prey moves for input
    prey_history: Vec<usize>,
    /// Learning rate
    lr: f64,
    /// Prediction tracking
    correct_predictions: u32,
    total_predictions: u32,
    /// Last hidden activations (for backprop)
    last_hidden: Vec<f64>,
    /// Last output probabilities
    last_output: Vec<f64>,
    speed: i32,
}

impl NeuralPredator {
    fn new(x: i32, y: i32) -> Self {
        // Smarter configuration:
        // - Longer history to detect more complex patterns
        // - Larger hidden layer for more capacity
        let history_length = 10; // Look at last 10 moves (was 5)
        let input_size = history_length * 5; // One-hot encoded
        let hidden_size = 64; // 64 hidden neurons (was 32)
        let output_size = 5;

        // Xavier initialization
        let mut rng_state = 42u64;
        let mut rand_f64 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
        };

        let scale_ih = (2.0 / input_size as f64).sqrt();
        let scale_ho = (2.0 / hidden_size as f64).sqrt();

        let weights_ih: Vec<Vec<f64>> = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rand_f64() * scale_ih).collect())
            .collect();

        let weights_ho: Vec<Vec<f64>> = (0..output_size)
            .map(|_| (0..hidden_size).map(|_| rand_f64() * scale_ho).collect())
            .collect();

        let bias_h = vec![0.0; hidden_size];
        let bias_o = vec![0.0; output_size];

        Self {
            x,
            y,
            history_length,
            weights_ih,
            weights_ho,
            bias_h,
            bias_o,
            prey_history: Vec::new(),
            lr: 0.05, // Lower learning rate for more stable learning
            correct_predictions: 0,
            total_predictions: 0,
            last_hidden: vec![0.0; hidden_size],
            last_output: vec![0.2; output_size],
            speed: 1,
        }
    }

    /// One-hot encode recent history
    fn encode_input(&self) -> Vec<f64> {
        let mut input = vec![0.0; self.history_length * 5];

        for (i, &mv) in self
            .prey_history
            .iter()
            .rev()
            .take(self.history_length)
            .enumerate()
        {
            if mv < 5 {
                input[i * 5 + mv] = 1.0;
            }
        }

        input
    }

    /// Forward pass with ReLU hidden and softmax output
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        // Hidden layer with ReLU
        self.last_hidden = self
            .weights_ih
            .iter()
            .zip(self.bias_h.iter())
            .map(|(weights, &bias)| {
                let sum: f64 = weights
                    .iter()
                    .zip(input.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>()
                    + bias;
                sum.max(0.0) // ReLU
            })
            .collect();

        // Output layer
        let raw_output: Vec<f64> = self
            .weights_ho
            .iter()
            .zip(self.bias_o.iter())
            .map(|(weights, &bias)| {
                weights
                    .iter()
                    .zip(self.last_hidden.iter())
                    .map(|(&w, &h)| w * h)
                    .sum::<f64>()
                    + bias
            })
            .collect();

        // Softmax
        let max_val = raw_output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = raw_output.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();

        self.last_output = exp_vals.iter().map(|&e| e / sum).collect();
        self.last_output.clone()
    }

    /// Backpropagation with cross-entropy loss
    fn backward(&mut self, input: &[f64], target: usize) {
        // Output gradient: softmax - one_hot (cross-entropy derivative)
        let mut output_grad: Vec<f64> = self.last_output.clone();
        output_grad[target] -= 1.0;

        // Hidden gradient
        let hidden_grad: Vec<f64> = (0..self.last_hidden.len())
            .map(|j| {
                let sum: f64 = self
                    .weights_ho
                    .iter()
                    .zip(output_grad.iter())
                    .map(|(weights, &og)| weights[j] * og)
                    .sum();
                if self.last_hidden[j] > 0.0 { sum } else { 0.0 } // ReLU derivative
            })
            .collect();

        // Update weights_ho
        for (i, weights) in self.weights_ho.iter_mut().enumerate() {
            for (j, w) in weights.iter_mut().enumerate() {
                *w -= self.lr * output_grad[i] * self.last_hidden[j];
            }
            self.bias_o[i] -= self.lr * output_grad[i];
        }

        // Update weights_ih
        for (j, weights) in self.weights_ih.iter_mut().enumerate() {
            for (k, w) in weights.iter_mut().enumerate() {
                *w -= self.lr * hidden_grad[j] * input[k];
            }
            self.bias_h[j] -= self.lr * hidden_grad[j];
        }
    }

    /// Observe prey move and learn
    fn observe_prey_move(&mut self, prey_move: usize) {
        if self.prey_history.len() >= self.history_length {
            // We have enough history to learn
            let input = self.encode_input();
            self.forward(&input);
            self.backward(&input, prey_move);
        }

        self.prey_history.push(prey_move);
        if self.prey_history.len() > 100 {
            self.prey_history.remove(0);
        }
    }

    /// Predict prey's next move
    fn predict_prey_move(&mut self) -> usize {
        if self.prey_history.len() < self.history_length {
            return 4; // Default: predict Stay
        }

        let input = self.encode_input();
        let probs = self.forward(&input);

        // Return argmax
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(4)
    }

    /// Hunt toward predicted prey position
    fn hunt(&mut self, prey_x: i32, prey_y: i32, width: i32, height: i32) {
        let predicted_move = self.predict_prey_move();

        let (dx, dy) = match predicted_move {
            0 => (0, -1),
            1 => (0, 1),
            2 => (-1, 0),
            3 => (1, 0),
            _ => (0, 0),
        };

        let predicted_x = (prey_x + dx + width) % width;
        let predicted_y = (prey_y + dy + height) % height;

        let move_x = (predicted_x - self.x).signum() * self.speed;
        let move_y = (predicted_y - self.y).signum() * self.speed;

        self.x = (self.x + move_x + width) % width;
        self.y = (self.y + move_y + height) % height;
    }

    /// Check catch and track accuracy
    fn check_catch(&mut self, prey_x: i32, prey_y: i32, prey_move: usize) -> bool {
        if self.prey_history.len() >= self.history_length {
            let prediction = self.predict_prey_move();
            self.total_predictions += 1;
            if prediction == prey_move {
                self.correct_predictions += 1;
            }
        }

        let dx = (self.x - prey_x).abs();
        let dy = (self.y - prey_y).abs();
        dx <= 1 && dy <= 1
    }

    fn prediction_accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            self.correct_predictions as f64 / self.total_predictions as f64
        }
    }
}

// ============================================================================
// RL Predator - Q-Learning with function approximation
// ============================================================================

/// Reinforcement Learning predator using Q-learning
/// Unlike prediction-based predators, this learns a POLICY for catching prey
/// by receiving rewards when successful.
struct RLPredator {
    x: i32,
    y: i32,
    /// Q-table: state_hash -> [Q-values for 5 actions]
    q_table: HashMap<u64, [f64; 5]>,
    /// Learning rate
    alpha: f64,
    /// Discount factor
    gamma: f64,
    /// Exploration rate (epsilon-greedy)
    epsilon: f64,
    /// Epsilon decay
    epsilon_decay: f64,
    /// Minimum epsilon
    epsilon_min: f64,
    /// History of prey moves for state encoding
    prey_history: Vec<usize>,
    /// Last state-action for TD update
    last_state: Option<u64>,
    last_action: Option<usize>,
    /// Performance tracking
    catches: u32,
    steps: u32,
    /// RNG state
    rng: u64,
    speed: i32,
}

impl RLPredator {
    fn new(x: i32, y: i32) -> Self {
        Self {
            x,
            y,
            q_table: HashMap::new(),
            alpha: 0.1,   // Learning rate
            gamma: 0.95,  // High discount - value future catches
            epsilon: 0.3, // Start with 30% exploration
            epsilon_decay: 0.999,
            epsilon_min: 0.05,
            prey_history: Vec::new(),
            last_state: None,
            last_action: None,
            catches: 0,
            steps: 0,
            rng: 12345,
            speed: 1,
        }
    }

    fn rand(&mut self) -> f64 {
        self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.rng >> 11) as f64) / (1u64 << 53) as f64
    }

    /// Encode state as hash of: relative position + recent prey history
    fn encode_state(&self, prey_x: i32, prey_y: i32, _width: i32, _height: i32) -> u64 {
        // Discretize relative position into bins
        let dx = prey_x - self.x;
        let dy = prey_y - self.y;

        // Bin relative position: -far, -near, same, +near, +far
        let dx_bin = match dx {
            d if d < -5 => 0,
            d if d < -1 => 1,
            d if d <= 1 => 2,
            d if d <= 5 => 3,
            _ => 4,
        };
        let dy_bin = match dy {
            d if d < -5 => 0,
            d if d < -1 => 1,
            d if d <= 1 => 2,
            d if d <= 5 => 3,
            _ => 4,
        };

        // Include last 3 prey moves in state
        let mut hasher = DefaultHasher::new();
        (dx_bin, dy_bin).hash(&mut hasher);

        for &mv in self.prey_history.iter().rev().take(3) {
            mv.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Get Q-values for a state (initialize if new)
    fn get_q_values(&mut self, state: u64) -> [f64; 5] {
        *self.q_table.entry(state).or_insert([0.0; 5])
    }

    /// Select action using epsilon-greedy
    fn select_action(&mut self, state: u64) -> usize {
        if self.rand() < self.epsilon {
            // Explore: random action
            (self.rand() * 5.0) as usize
        } else {
            // Exploit: best Q-value
            let q = self.get_q_values(state);
            q.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(4)
        }
    }

    /// Update Q-value using TD(0)
    fn update(&mut self, reward: f64, next_state: u64, done: bool) {
        if let (Some(state), Some(action)) = (self.last_state, self.last_action) {
            let current_q = self.get_q_values(state)[action];

            let next_max_q = if done {
                0.0
            } else {
                let next_q = self.get_q_values(next_state);
                next_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            };

            let td_target = reward + self.gamma * next_max_q;
            let td_error = td_target - current_q;

            // Update Q-value
            let q_values = self.q_table.entry(state).or_insert([0.0; 5]);
            q_values[action] += self.alpha * td_error;
        }

        // Decay epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    /// Observe prey move (for state encoding)
    fn observe_prey_move(&mut self, prey_move: usize) {
        self.prey_history.push(prey_move);
        if self.prey_history.len() > 10 {
            self.prey_history.remove(0);
        }
    }

    /// Hunt using learned policy
    fn hunt(&mut self, prey_x: i32, prey_y: i32, width: i32, height: i32) {
        let state = self.encode_state(prey_x, prey_y, width, height);
        let action = self.select_action(state);

        // Store for next update
        self.last_state = Some(state);
        self.last_action = Some(action);

        // Execute action
        let (dx, dy) = match action {
            0 => (0, -self.speed), // Up
            1 => (0, self.speed),  // Down
            2 => (-self.speed, 0), // Left
            3 => (self.speed, 0),  // Right
            _ => (0, 0),           // Stay
        };

        self.x = (self.x + dx + width) % width;
        self.y = (self.y + dy + height) % height;
        self.steps += 1;
    }

    /// Check if caught prey and update Q-values
    fn check_catch(&mut self, prey_x: i32, prey_y: i32, _prey_action: usize) -> bool {
        let dx = (self.x - prey_x).abs();
        let dy = (self.y - prey_y).abs();
        let caught = dx <= 1 && dy <= 1;

        // Get next state for TD update
        let next_state = self.encode_state(prey_x, prey_y, 30, 30);

        if caught {
            // Big reward for catching
            self.update(100.0, next_state, true);
            self.catches += 1;
        } else {
            // Small negative reward per step (encourages faster catches)
            self.update(-0.1, next_state, false);
        }

        caught
    }

    fn prediction_accuracy(&self) -> f64 {
        // For RL predator, report catch rate instead
        if self.steps == 0 {
            0.0
        } else {
            self.catches as f64 / (self.steps as f64 / 100.0).max(1.0)
        }
    }

    /// Report learning stats
    fn learning_stats(&self) -> (usize, f64) {
        (self.q_table.len(), self.epsilon)
    }
}

// ============================================================================
// Prey Brains - Quantum and Classical
// ============================================================================

struct QuantumPrey {
    qsnn: QuantumSNN,
    x: i32,
    y: i32,
}

impl QuantumPrey {
    fn new(seed: u64) -> Self {
        let config = QuantumSNNConfig {
            n_inputs: 8, // predator direction (4) + wall proximity (4)
            hidden_layers: vec![16, 8],
            n_outputs: 5,          // Up, Down, Left, Right, Stay
            interference_depth: 2, // More interference for unpredictability
            ticks_per_decision: 10,
            learning_enabled: true,
            ..Default::default()
        };

        let qsnn = QuantumSNN::with_seed(config, seed);
        Self { qsnn, x: 0, y: 0 }
    }

    fn get_sensors(&self, predator_x: i32, predator_y: i32, width: i32, height: i32) -> Vec<u8> {
        // Predator direction sensors (0-255 intensity)
        let dx = predator_x - self.x;
        let dy = predator_y - self.y;

        let pred_north = if dy < 0 {
            ((-dy).min(10) * 25) as u8
        } else {
            0
        };
        let pred_south = if dy > 0 { (dy.min(10) * 25) as u8 } else { 0 };
        let pred_west = if dx < 0 {
            ((-dx).min(10) * 25) as u8
        } else {
            0
        };
        let pred_east = if dx > 0 { (dx.min(10) * 25) as u8 } else { 0 };

        // Wall proximity (encourage staying away from edges where escape is harder)
        let wall_north = ((height - self.y).min(5) * 50) as u8;
        let wall_south = (self.y.min(5) * 50) as u8;
        let wall_west = (self.x.min(5) * 50) as u8;
        let wall_east = ((width - self.x).min(5) * 50) as u8;

        vec![
            pred_north, pred_south, pred_west, pred_east, wall_north, wall_south, wall_west,
            wall_east,
        ]
    }

    fn decide(
        &mut self,
        predator_x: i32,
        predator_y: i32,
        width: i32,
        height: i32,
    ) -> (Move, usize) {
        let sensors = self.get_sensors(predator_x, predator_y, width, height);
        self.qsnn.set_inputs(&sensors);
        let action = self.qsnn.decide();

        let mv = match action {
            0 => Move::Up,
            1 => Move::Down,
            2 => Move::Left,
            3 => Move::Right,
            _ => Move::Stay,
        };

        (mv, action)
    }

    fn apply_move(&mut self, mv: Move, width: i32, height: i32) {
        let (dx, dy) = match mv {
            Move::Up => (0, -1),
            Move::Down => (0, 1),
            Move::Left => (-1, 0),
            Move::Right => (1, 0),
            Move::Stay => (0, 0),
        };

        self.x = (self.x + dx + width) % width;
        self.y = (self.y + dy + height) % height;
    }

    fn apply_reward(&mut self, reward: f32) {
        let reward_i8 = (reward * 10.0).clamp(-127.0, 127.0) as i8;
        self.qsnn.set_reward(reward_i8);
    }

    fn decision_entropy(&self) -> f64 {
        let probs = self.qsnn.last_probabilities();
        if probs.is_empty() {
            return 0.0;
        }
        let mut entropy = 0.0;
        for &p in probs {
            if p > 1e-10 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

struct ClassicalPrey {
    snn: engine::snn::SNNNetwork,
    x: i32,
    y: i32,
    rng_state: u64,
}

impl ClassicalPrey {
    fn new(seed: u64) -> Self {
        let config = SNNConfig {
            n_inputs: 8,
            hidden_layers: vec![16, 8],
            n_outputs: 5,
            connection_prob: 0.5,
            neurons_per_cpu: 32,
            recurrent: false,
            neuron_config: NeuronConfig::default(),
            stdp_config: STDPConfig::default(),
            use_reward_modulation: true,
            duplicate_inputs: false,
        };

        let mut snn = engine::snn::SNNNetwork::new(config);
        snn.build_connectivity(seed);
        snn.randomize_weights(seed.wrapping_add(12345));

        Self {
            snn,
            x: 0,
            y: 0,
            rng_state: seed,
        }
    }

    fn get_sensors(&self, predator_x: i32, predator_y: i32, width: i32, height: i32) -> Vec<u8> {
        let dx = predator_x - self.x;
        let dy = predator_y - self.y;

        let pred_north = if dy < 0 {
            ((-dy).min(10) * 25) as u8
        } else {
            0
        };
        let pred_south = if dy > 0 { (dy.min(10) * 25) as u8 } else { 0 };
        let pred_west = if dx < 0 {
            ((-dx).min(10) * 25) as u8
        } else {
            0
        };
        let pred_east = if dx > 0 { (dx.min(10) * 25) as u8 } else { 0 };

        let wall_north = ((height - self.y).min(5) * 50) as u8;
        let wall_south = (self.y.min(5) * 50) as u8;
        let wall_west = (self.x.min(5) * 50) as u8;
        let wall_east = ((width - self.x).min(5) * 50) as u8;

        vec![
            pred_north, pred_south, pred_west, pred_east, wall_north, wall_south, wall_west,
            wall_east,
        ]
    }

    fn decide(
        &mut self,
        predator_x: i32,
        predator_y: i32,
        width: i32,
        height: i32,
    ) -> (Move, usize) {
        let sensors = self.get_sensors(predator_x, predator_y, width, height);
        self.snn.set_inputs(&sensors);

        self.snn.reset_output_counts();
        for _ in 0..10 {
            self.snn.step();
        }

        // Softmax sampling (same as evolution example)
        let counts = &self.snn.output_counts;
        let total: u32 = counts.iter().sum();

        let action = if total == 0 {
            self.rng_state = self
                .rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            ((self.rng_state >> 32) as usize) % 5
        } else {
            self.rng_state = self
                .rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let r = ((self.rng_state >> 33) as f64) / (1u64 << 31) as f64;

            let mut cumulative = 0.0;
            let mut selected = 4;
            for (i, &count) in counts.iter().enumerate() {
                cumulative += count as f64 / total as f64;
                if r < cumulative {
                    selected = i;
                    break;
                }
            }
            selected
        };

        let mv = match action {
            0 => Move::Up,
            1 => Move::Down,
            2 => Move::Left,
            3 => Move::Right,
            _ => Move::Stay,
        };

        (mv, action)
    }

    fn apply_move(&mut self, mv: Move, width: i32, height: i32) {
        let (dx, dy) = match mv {
            Move::Up => (0, -1),
            Move::Down => (0, 1),
            Move::Left => (-1, 0),
            Move::Right => (1, 0),
            Move::Stay => (0, 0),
        };

        self.x = (self.x + dx + width) % width;
        self.y = (self.y + dy + height) % height;
    }

    fn apply_reward(&mut self, reward: f32) {
        let reward_i8 = (reward * 10.0).clamp(-127.0, 127.0) as i8;
        self.snn.set_reward(reward_i8);
    }
}

/// Deterministic prey (always picks highest spike count) - baseline
struct DeterministicPrey {
    snn: engine::snn::SNNNetwork,
    x: i32,
    y: i32,
}

impl DeterministicPrey {
    fn new(seed: u64) -> Self {
        let config = SNNConfig {
            n_inputs: 8,
            hidden_layers: vec![16, 8],
            n_outputs: 5,
            connection_prob: 0.5,
            neurons_per_cpu: 32,
            recurrent: false,
            neuron_config: NeuronConfig::default(),
            stdp_config: STDPConfig::default(),
            use_reward_modulation: true,
            duplicate_inputs: false,
        };

        let mut snn = engine::snn::SNNNetwork::new(config);
        snn.build_connectivity(seed);
        snn.randomize_weights(seed.wrapping_add(12345));

        Self { snn, x: 0, y: 0 }
    }

    fn get_sensors(&self, predator_x: i32, predator_y: i32, width: i32, height: i32) -> Vec<u8> {
        let dx = predator_x - self.x;
        let dy = predator_y - self.y;

        let pred_north = if dy < 0 {
            ((-dy).min(10) * 25) as u8
        } else {
            0
        };
        let pred_south = if dy > 0 { (dy.min(10) * 25) as u8 } else { 0 };
        let pred_west = if dx < 0 {
            ((-dx).min(10) * 25) as u8
        } else {
            0
        };
        let pred_east = if dx > 0 { (dx.min(10) * 25) as u8 } else { 0 };

        let wall_north = ((height - self.y).min(5) * 50) as u8;
        let wall_south = (self.y.min(5) * 50) as u8;
        let wall_west = (self.x.min(5) * 50) as u8;
        let wall_east = ((width - self.x).min(5) * 50) as u8;

        vec![
            pred_north, pred_south, pred_west, pred_east, wall_north, wall_south, wall_west,
            wall_east,
        ]
    }

    fn decide(
        &mut self,
        predator_x: i32,
        predator_y: i32,
        width: i32,
        height: i32,
    ) -> (Move, usize) {
        let sensors = self.get_sensors(predator_x, predator_y, width, height);
        self.snn.set_inputs(&sensors);

        self.snn.reset_output_counts();
        for _ in 0..10 {
            self.snn.step();
        }

        // Deterministic: always pick highest count
        let action = self.snn.get_action();

        let mv = match action {
            0 => Move::Up,
            1 => Move::Down,
            2 => Move::Left,
            3 => Move::Right,
            _ => Move::Stay,
        };

        (mv, action)
    }

    fn apply_move(&mut self, mv: Move, width: i32, height: i32) {
        let (dx, dy) = match mv {
            Move::Up => (0, -1),
            Move::Down => (0, 1),
            Move::Left => (-1, 0),
            Move::Right => (1, 0),
            Move::Stay => (0, 0),
        };

        self.x = (self.x + dx + width) % width;
        self.y = (self.y + dy + height) % height;
    }

    fn apply_reward(&mut self, reward: f32) {
        let reward_i8 = (reward * 10.0).clamp(-127.0, 127.0) as i8;
        self.snn.set_reward(reward_i8);
    }
}

// ============================================================================
// Simulation
// ============================================================================

struct SimulationResult {
    survival_time: u32,
    predator_accuracy: f64,
    prey_entropy: f64,
}

fn run_quantum_trial(seed: u64, width: i32, height: i32, max_steps: u32) -> SimulationResult {
    let mut prey = QuantumPrey::new(seed);
    let mut predator = AdaptivePredator::new(width - 5, height - 5);

    prey.x = 5;
    prey.y = 5;

    let mut total_entropy = 0.0;
    let mut entropy_count = 0;

    for step in 0..max_steps {
        // Prey decides and moves
        let (mv, action) = prey.decide(predator.x, predator.y, width, height);
        prey.apply_move(mv, width, height);

        // Track entropy
        total_entropy += prey.decision_entropy();
        entropy_count += 1;

        // Predator observes and hunts
        if predator.check_catch(prey.x, prey.y, action) {
            prey.apply_reward(-10.0); // Caught!
            return SimulationResult {
                survival_time: step,
                predator_accuracy: predator.prediction_accuracy(),
                prey_entropy: total_entropy / entropy_count as f64,
            };
        }

        predator.observe_prey_move(action);
        predator.hunt(prey.x, prey.y, width, height);

        // Reward for surviving
        prey.apply_reward(0.1);
    }

    SimulationResult {
        survival_time: max_steps,
        predator_accuracy: predator.prediction_accuracy(),
        prey_entropy: total_entropy / entropy_count.max(1) as f64,
    }
}

fn run_classical_trial(seed: u64, width: i32, height: i32, max_steps: u32) -> SimulationResult {
    let mut prey = ClassicalPrey::new(seed);
    let mut predator = AdaptivePredator::new(width - 5, height - 5);

    prey.x = 5;
    prey.y = 5;

    for step in 0..max_steps {
        let (mv, action) = prey.decide(predator.x, predator.y, width, height);
        prey.apply_move(mv, width, height);

        if predator.check_catch(prey.x, prey.y, action) {
            prey.apply_reward(-10.0);
            return SimulationResult {
                survival_time: step,
                predator_accuracy: predator.prediction_accuracy(),
                prey_entropy: 0.0, // Not tracked for classical
            };
        }

        predator.observe_prey_move(action);
        predator.hunt(prey.x, prey.y, width, height);

        prey.apply_reward(0.1);
    }

    SimulationResult {
        survival_time: max_steps,
        predator_accuracy: predator.prediction_accuracy(),
        prey_entropy: 0.0,
    }
}

fn run_deterministic_trial(seed: u64, width: i32, height: i32, max_steps: u32) -> SimulationResult {
    let mut prey = DeterministicPrey::new(seed);
    let mut predator = AdaptivePredator::new(width - 5, height - 5);

    prey.x = 5;
    prey.y = 5;

    for step in 0..max_steps {
        let (mv, action) = prey.decide(predator.x, predator.y, width, height);
        prey.apply_move(mv, width, height);

        if predator.check_catch(prey.x, prey.y, action) {
            prey.apply_reward(-10.0);
            return SimulationResult {
                survival_time: step,
                predator_accuracy: predator.prediction_accuracy(),
                prey_entropy: 0.0,
            };
        }

        predator.observe_prey_move(action);
        predator.hunt(prey.x, prey.y, width, height);

        prey.apply_reward(0.1);
    }

    SimulationResult {
        survival_time: max_steps,
        predator_accuracy: predator.prediction_accuracy(),
        prey_entropy: 0.0,
    }
}

// ============================================================================
// Neural Network Predator Trials
// ============================================================================

fn run_quantum_vs_neural(seed: u64, width: i32, height: i32, max_steps: u32) -> SimulationResult {
    let mut prey = QuantumPrey::new(seed);
    let mut predator = NeuralPredator::new(width - 5, height - 5);

    prey.x = 5;
    prey.y = 5;

    let mut total_entropy = 0.0;
    let mut entropy_count = 0;

    for step in 0..max_steps {
        let (mv, action) = prey.decide(predator.x, predator.y, width, height);
        prey.apply_move(mv, width, height);

        total_entropy += prey.decision_entropy();
        entropy_count += 1;

        if predator.check_catch(prey.x, prey.y, action) {
            prey.apply_reward(-10.0);
            return SimulationResult {
                survival_time: step,
                predator_accuracy: predator.prediction_accuracy(),
                prey_entropy: total_entropy / entropy_count as f64,
            };
        }

        predator.observe_prey_move(action);
        predator.hunt(prey.x, prey.y, width, height);

        prey.apply_reward(0.1);
    }

    SimulationResult {
        survival_time: max_steps,
        predator_accuracy: predator.prediction_accuracy(),
        prey_entropy: total_entropy / entropy_count.max(1) as f64,
    }
}

fn run_classical_vs_neural(seed: u64, width: i32, height: i32, max_steps: u32) -> SimulationResult {
    let mut prey = ClassicalPrey::new(seed);
    let mut predator = NeuralPredator::new(width - 5, height - 5);

    prey.x = 5;
    prey.y = 5;

    for step in 0..max_steps {
        let (mv, action) = prey.decide(predator.x, predator.y, width, height);
        prey.apply_move(mv, width, height);

        if predator.check_catch(prey.x, prey.y, action) {
            prey.apply_reward(-10.0);
            return SimulationResult {
                survival_time: step,
                predator_accuracy: predator.prediction_accuracy(),
                prey_entropy: 0.0,
            };
        }

        predator.observe_prey_move(action);
        predator.hunt(prey.x, prey.y, width, height);

        prey.apply_reward(0.1);
    }

    SimulationResult {
        survival_time: max_steps,
        predator_accuracy: predator.prediction_accuracy(),
        prey_entropy: 0.0,
    }
}

fn run_deterministic_vs_neural(
    seed: u64,
    width: i32,
    height: i32,
    max_steps: u32,
) -> SimulationResult {
    let mut prey = DeterministicPrey::new(seed);
    let mut predator = NeuralPredator::new(width - 5, height - 5);

    prey.x = 5;
    prey.y = 5;

    for step in 0..max_steps {
        let (mv, action) = prey.decide(predator.x, predator.y, width, height);
        prey.apply_move(mv, width, height);

        if predator.check_catch(prey.x, prey.y, action) {
            prey.apply_reward(-10.0);
            return SimulationResult {
                survival_time: step,
                predator_accuracy: predator.prediction_accuracy(),
                prey_entropy: 0.0,
            };
        }

        predator.observe_prey_move(action);
        predator.hunt(prey.x, prey.y, width, height);

        prey.apply_reward(0.1);
    }

    SimulationResult {
        survival_time: max_steps,
        predator_accuracy: predator.prediction_accuracy(),
        prey_entropy: 0.0,
    }
}

// ============================================================================
// RL Predator Trials
// ============================================================================

/// Pre-train RL predator against quantum prey, then test
fn run_quantum_vs_rl(seed: u64, width: i32, height: i32, max_steps: u32) -> SimulationResult {
    let mut predator = RLPredator::new(width - 5, height - 5);

    // Pre-training phase: 200 episodes to learn a stable policy
    for ep in 0..200 {
        let mut prey = QuantumPrey::new(seed.wrapping_add(ep as u64 * 1000));
        prey.x = 5;
        prey.y = 5;
        predator.x = width - 5;
        predator.y = height - 5;
        predator.prey_history.clear();
        predator.last_state = None;
        predator.last_action = None;

        for _ in 0..200 {
            // Short episodes for faster training
            let (mv, action) = prey.decide(predator.x, predator.y, width, height);
            prey.apply_move(mv, width, height);

            if predator.check_catch(prey.x, prey.y, action) {
                break;
            }

            predator.observe_prey_move(action);
            predator.hunt(prey.x, prey.y, width, height);
        }
    }

    // Lock in exploitation (reduce epsilon)
    predator.epsilon = 0.05;

    // Now the actual test
    let mut prey = QuantumPrey::new(seed);
    prey.x = 5;
    prey.y = 5;
    predator.x = width - 5;
    predator.y = height - 5;
    predator.catches = 0;
    predator.steps = 0;
    predator.prey_history.clear();

    for step in 0..max_steps {
        let (mv, action) = prey.decide(predator.x, predator.y, width, height);
        prey.apply_move(mv, width, height);

        if predator.check_catch(prey.x, prey.y, action) {
            prey.apply_reward(-10.0);
            return SimulationResult {
                survival_time: step,
                predator_accuracy: predator.prediction_accuracy(),
                prey_entropy: 0.0,
            };
        }

        predator.observe_prey_move(action);
        predator.hunt(prey.x, prey.y, width, height);

        prey.apply_reward(0.1);
    }

    SimulationResult {
        survival_time: max_steps,
        predator_accuracy: predator.prediction_accuracy(),
        prey_entropy: 0.0,
    }
}

/// Pre-train RL predator against classical prey, then test
fn run_classical_vs_rl(seed: u64, width: i32, height: i32, max_steps: u32) -> SimulationResult {
    let mut predator = RLPredator::new(width - 5, height - 5);

    // Pre-training phase: 200 episodes to learn a stable policy
    for ep in 0..200 {
        let mut prey = ClassicalPrey::new(seed.wrapping_add(ep as u64 * 1000));
        prey.x = 5;
        prey.y = 5;
        predator.x = width - 5;
        predator.y = height - 5;
        predator.prey_history.clear();
        predator.last_state = None;
        predator.last_action = None;

        for _ in 0..200 {
            let (mv, action) = prey.decide(predator.x, predator.y, width, height);
            prey.apply_move(mv, width, height);

            if predator.check_catch(prey.x, prey.y, action) {
                break;
            }

            predator.observe_prey_move(action);
            predator.hunt(prey.x, prey.y, width, height);
        }
    }

    predator.epsilon = 0.05;

    // Test phase
    let mut prey = ClassicalPrey::new(seed);
    prey.x = 5;
    prey.y = 5;
    predator.x = width - 5;
    predator.y = height - 5;
    predator.catches = 0;
    predator.steps = 0;
    predator.prey_history.clear();

    for step in 0..max_steps {
        let (mv, action) = prey.decide(predator.x, predator.y, width, height);
        prey.apply_move(mv, width, height);

        if predator.check_catch(prey.x, prey.y, action) {
            prey.apply_reward(-10.0);
            return SimulationResult {
                survival_time: step,
                predator_accuracy: predator.prediction_accuracy(),
                prey_entropy: 0.0,
            };
        }

        predator.observe_prey_move(action);
        predator.hunt(prey.x, prey.y, width, height);

        prey.apply_reward(0.1);
    }

    SimulationResult {
        survival_time: max_steps,
        predator_accuracy: predator.prediction_accuracy(),
        prey_entropy: 0.0,
    }
}

/// Pre-train RL predator against deterministic prey, then test
fn run_deterministic_vs_rl(seed: u64, width: i32, height: i32, max_steps: u32) -> SimulationResult {
    let mut predator = RLPredator::new(width - 5, height - 5);

    // Pre-training phase: 200 episodes to learn a stable policy
    for ep in 0..200 {
        let mut prey = DeterministicPrey::new(seed.wrapping_add(ep as u64 * 1000));
        prey.x = 5;
        prey.y = 5;
        predator.x = width - 5;
        predator.y = height - 5;
        predator.prey_history.clear();
        predator.last_state = None;
        predator.last_action = None;

        for _ in 0..200 {
            let (mv, action) = prey.decide(predator.x, predator.y, width, height);
            prey.apply_move(mv, width, height);

            if predator.check_catch(prey.x, prey.y, action) {
                break;
            }

            predator.observe_prey_move(action);
            predator.hunt(prey.x, prey.y, width, height);
        }
    }

    predator.epsilon = 0.05;

    // Test phase
    let mut prey = DeterministicPrey::new(seed);
    prey.x = 5;
    prey.y = 5;
    predator.x = width - 5;
    predator.y = height - 5;
    predator.catches = 0;
    predator.steps = 0;
    predator.prey_history.clear();

    for step in 0..max_steps {
        let (mv, action) = prey.decide(predator.x, predator.y, width, height);
        prey.apply_move(mv, width, height);

        if predator.check_catch(prey.x, prey.y, action) {
            prey.apply_reward(-10.0);
            return SimulationResult {
                survival_time: step,
                predator_accuracy: predator.prediction_accuracy(),
                prey_entropy: 0.0,
            };
        }

        predator.observe_prey_move(action);
        predator.hunt(prey.x, prey.y, width, height);

        prey.apply_reward(0.1);
    }

    SimulationResult {
        survival_time: max_steps,
        predator_accuracy: predator.prediction_accuracy(),
        prey_entropy: 0.0,
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("================================================================");
    println!("  ADVERSARIAL PREDATOR-PREY: QUANTUM VS CLASSICAL");
    println!("  Can quantum unpredictability evade learning predators?");
    println!("================================================================\n");

    let width = 30;
    let height = 30;
    let max_steps = 500;
    let n_trials = 20;

    println!("Configuration:");
    println!("  Arena: {}x{}", width, height);
    println!("  Max steps: {}", max_steps);
    println!("  Trials: {}", n_trials);
    println!();

    // Use time-based seed for variance between runs
    let base_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // ========================================================================
    // PART 1: MARKOV PREDATOR (Simple pattern learning)
    // ========================================================================
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  PART 1: MARKOV PREDATOR (Transition probability learning)    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let mut quantum_markov: Vec<SimulationResult> = Vec::new();
    let mut classical_markov: Vec<SimulationResult> = Vec::new();
    let mut deterministic_markov: Vec<SimulationResult> = Vec::new();

    let start1 = Instant::now();
    for trial in 0..n_trials {
        let seed = base_seed.wrapping_add(trial as u64 * 137);
        quantum_markov.push(run_quantum_trial(seed, width, height, max_steps));
        classical_markov.push(run_classical_trial(seed, width, height, max_steps));
        deterministic_markov.push(run_deterministic_trial(seed, width, height, max_steps));
    }
    let elapsed1 = start1.elapsed();

    print_results(
        "MARKOV",
        &quantum_markov,
        &classical_markov,
        &deterministic_markov,
        n_trials,
        elapsed1,
    );

    // ========================================================================
    // PART 2: NEURAL NETWORK PREDATOR (Deep learning with backprop)
    // ========================================================================
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  PART 2: NEURAL NETWORK PREDATOR (2-layer MLP + online SGD)   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let mut quantum_neural: Vec<SimulationResult> = Vec::new();
    let mut classical_neural: Vec<SimulationResult> = Vec::new();
    let mut deterministic_neural: Vec<SimulationResult> = Vec::new();

    let start2 = Instant::now();
    for trial in 0..n_trials {
        let seed = base_seed.wrapping_add(trial as u64 * 137);
        quantum_neural.push(run_quantum_vs_neural(seed, width, height, max_steps));
        classical_neural.push(run_classical_vs_neural(seed, width, height, max_steps));
        deterministic_neural.push(run_deterministic_vs_neural(seed, width, height, max_steps));
    }
    let elapsed2 = start2.elapsed();

    print_results(
        "NEURAL",
        &quantum_neural,
        &classical_neural,
        &deterministic_neural,
        n_trials,
        elapsed2,
    );

    // ========================================================================
    // PART 3: RL PREDATOR (Q-Learning - learns to CATCH, not predict)
    // ========================================================================
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  PART 3: RL PREDATOR (Q-Learning - optimizes for catching)    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let mut quantum_rl: Vec<SimulationResult> = Vec::new();
    let mut classical_rl: Vec<SimulationResult> = Vec::new();
    let mut deterministic_rl: Vec<SimulationResult> = Vec::new();

    let start3 = Instant::now();
    for trial in 0..n_trials {
        let seed = base_seed.wrapping_add(trial as u64 * 137);
        quantum_rl.push(run_quantum_vs_rl(seed, width, height, max_steps));
        classical_rl.push(run_classical_vs_rl(seed, width, height, max_steps));
        deterministic_rl.push(run_deterministic_vs_rl(seed, width, height, max_steps));
    }
    let elapsed3 = start3.elapsed();

    print_results(
        "RL",
        &quantum_rl,
        &classical_rl,
        &deterministic_rl,
        n_trials,
        elapsed3,
    );

    // ========================================================================
    // FINAL COMPARISON: All three predators
    // ========================================================================
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  FINAL COMPARISON: ALL PREDATOR TYPES                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Compute quantum advantage for each predator type
    let q_markov_avg: f64 = quantum_markov
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;
    let c_markov_avg: f64 = classical_markov
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;
    let q_neural_avg: f64 = quantum_neural
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;
    let c_neural_avg: f64 = classical_neural
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;
    let q_rl_avg: f64 = quantum_rl
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;
    let c_rl_avg: f64 = classical_rl
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;

    let q_advantage_markov = q_markov_avg - c_markov_avg;
    let q_advantage_neural = q_neural_avg - c_neural_avg;
    let q_advantage_rl = q_rl_avg - c_rl_avg;

    println!("  Quantum advantage (survival steps over classical):");
    println!(
        "    vs Markov (pattern prediction):     {:+.1} steps",
        q_advantage_markov
    );
    println!(
        "    vs Neural (supervised learning):   {:+.1} steps",
        q_advantage_neural
    );
    println!(
        "    vs RL (reinforcement learning):    {:+.1} steps",
        q_advantage_rl
    );

    // Determine which predator is most dangerous
    let d_markov_avg: f64 = deterministic_markov
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;
    let d_neural_avg: f64 = deterministic_neural
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;
    let d_rl_avg: f64 = deterministic_rl
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;

    println!("\n  Prey survival times (lower = more dangerous predator):");
    println!(
        "    {:>12} {:>12} {:>12} {:>12}",
        "Predator", "Quantum", "Classical", "Deterministic"
    );
    println!(
        "    {:>12} {:>12.1} {:>12.1} {:>12.1}",
        "Markov", q_markov_avg, c_markov_avg, d_markov_avg
    );
    println!(
        "    {:>12} {:>12.1} {:>12.1} {:>12.1}",
        "Neural", q_neural_avg, c_neural_avg, d_neural_avg
    );
    println!(
        "    {:>12} {:>12.1} {:>12.1} {:>12.1}",
        "RL", q_rl_avg, c_rl_avg, d_rl_avg
    );

    // Final verdict
    println!("\n  ═══════════════════════════════════════════════════════════════");
    let quantum_wins = (if q_advantage_markov > 0.0 { 1 } else { 0 })
        + (if q_advantage_neural > 0.0 { 1 } else { 0 })
        + (if q_advantage_rl > 0.0 { 1 } else { 0 });

    if quantum_wins == 3 {
        println!("  🏆 QUANTUM WINS AGAINST ALL THREE LEARNING PARADIGMS!");
        println!("  Quantum interference is fundamentally hard to exploit.");
    } else if quantum_wins >= 2 {
        println!("  ✓ QUANTUM WINS AGAINST {}/3 PREDATORS", quantum_wins);
        if q_advantage_rl <= 0.0 {
            println!("  ⚠ RL predator cracked the quantum patterns!");
        }
    } else if quantum_wins == 1 {
        println!("  ⚠ QUANTUM ADVANTAGE IS LIMITED (1/3 predators)");
    } else {
        println!("  ✗ CLASSICAL WINS - No quantum advantage detected");
    }

    // Most dangerous predator?
    let avg_survival_markov = (q_markov_avg + c_markov_avg + d_markov_avg) / 3.0;
    let avg_survival_neural = (q_neural_avg + c_neural_avg + d_neural_avg) / 3.0;
    let avg_survival_rl = (q_rl_avg + c_rl_avg + d_rl_avg) / 3.0;

    let (most_dangerous, survival) =
        if avg_survival_rl < avg_survival_markov && avg_survival_rl < avg_survival_neural {
            ("RL (Q-Learning)", avg_survival_rl)
        } else if avg_survival_neural < avg_survival_markov {
            ("Neural Network", avg_survival_neural)
        } else {
            ("Markov", avg_survival_markov)
        };

    println!(
        "\n  Most dangerous predator: {} (avg survival: {:.1} steps)",
        most_dangerous, survival
    );
    println!("  ═══════════════════════════════════════════════════════════════");

    println!("\n================================================================");
    println!("  PREDATOR-PREY SIMULATION COMPLETE");
    println!("================================================================");
}

fn print_results(
    predator_name: &str,
    quantum_results: &[SimulationResult],
    classical_results: &[SimulationResult],
    deterministic_results: &[SimulationResult],
    n_trials: usize,
    elapsed: std::time::Duration,
) {
    let avg_quantum_survival: f64 = quantum_results
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;
    let avg_classical_survival: f64 = classical_results
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;
    let avg_deterministic_survival: f64 = deterministic_results
        .iter()
        .map(|r| r.survival_time as f64)
        .sum::<f64>()
        / n_trials as f64;

    let avg_quantum_pred_acc: f64 = quantum_results
        .iter()
        .map(|r| r.predator_accuracy)
        .sum::<f64>()
        / n_trials as f64;
    let avg_classical_pred_acc: f64 = classical_results
        .iter()
        .map(|r| r.predator_accuracy)
        .sum::<f64>()
        / n_trials as f64;
    let avg_deterministic_pred_acc: f64 = deterministic_results
        .iter()
        .map(|r| r.predator_accuracy)
        .sum::<f64>()
        / n_trials as f64;

    println!(
        "{:<20} {:>15} {:>20}",
        "Prey Type", "Avg Survival", "Predator Accuracy"
    );
    println!("{:-<60}", "");

    println!(
        "{:<20} {:>15.1} {:>19.1}%",
        "Deterministic",
        avg_deterministic_survival,
        avg_deterministic_pred_acc * 100.0
    );
    println!(
        "{:<20} {:>15.1} {:>19.1}%",
        "Classical Softmax",
        avg_classical_survival,
        avg_classical_pred_acc * 100.0
    );
    println!(
        "{:<20} {:>15.1} {:>19.1}%",
        "Quantum Hybrid",
        avg_quantum_survival,
        avg_quantum_pred_acc * 100.0
    );

    println!();

    let quantum_vs_classical = avg_quantum_survival - avg_classical_survival;

    if quantum_vs_classical > 0.0 {
        println!(
            "  Quantum survives {:.1} steps longer than Classical ({:.1}% improvement)",
            quantum_vs_classical,
            quantum_vs_classical / avg_classical_survival * 100.0
        );
    } else if quantum_vs_classical < 0.0 {
        println!(
            "  Classical survives {:.1} steps longer than Quantum ({:.1}% improvement)",
            -quantum_vs_classical,
            -quantum_vs_classical / avg_quantum_survival * 100.0
        );
    } else {
        println!("  Quantum and Classical have equal survival times");
    }

    // Winner for this predator
    if avg_quantum_survival > avg_classical_survival
        && avg_quantum_survival > avg_deterministic_survival
    {
        println!("  -> {} PREDATOR: QUANTUM WINS!", predator_name);
    } else if avg_classical_survival > avg_quantum_survival
        && avg_classical_survival > avg_deterministic_survival
    {
        println!("  -> {} PREDATOR: CLASSICAL WINS!", predator_name);
    } else if avg_deterministic_survival > avg_quantum_survival {
        println!("  -> {} PREDATOR: DETERMINISTIC WINS!", predator_name);
    } else {
        println!("  -> {} PREDATOR: TIE", predator_name);
    }

    println!(
        "\n  Completed {} trials in {:.2}s",
        n_trials * 3,
        elapsed.as_secs_f64()
    );
}
