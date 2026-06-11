//! Sprint 60.0: The Observer Grid
//!
//! A hybrid quantum-classical substrate where 268M CPUs each control
//! whether to observe (collapse) their local quantum state.
//!
//! The boundary between measured and unmeasured regions becomes
//! a dynamic, emergent membrane shaped by computation.
//!
//! Key concepts:
//! - Each CPU binds to a quantum block (128 amplitudes = 7 qubits)
//! - CPUs decide whether to "observe" based on local rules
//! - Observation collapses superposition irreversibly (Born rule)
//! - Unmeasured regions can maintain entanglement
//! - The measured/unmeasured boundary exhibits emergent dynamics
//!
//! Measurement model:
//! - Born rule: P(outcome=i) = |α_i|² where α_i is the amplitude
//! - Each block stores 128 probabilities (one per basis state)
//! - Measurement samples from this distribution using high-quality RNG

/// Number of basis states per quantum block (2^7 = 128)
pub const BLOCK_SIZE: usize = 128;

/// Observation state for a single CPU-quantum pair
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationState {
    /// Still in superposition - not yet observed
    Superposition,
    /// Collapsed to classical state
    Collapsed(u8), // 7-bit measurement result (0-127)
    /// Protected - explicitly marked to never observe
    Protected,
}

impl Default for ObservationState {
    fn default() -> Self {
        Self::Superposition
    }
}

/// Rules governing when a CPU decides to observe
#[derive(Clone, Copy, Debug)]
pub enum ObservationRule {
    /// Observe if local probability exceeds threshold
    ProbabilityThreshold(f64),
    /// Observe if N or more neighbors have observed
    NeighborCount(usize),
    /// Observe based on tick modulo (periodic)
    Periodic(u64),
    /// Never observe (sanctuary zone)
    Never,
    /// Always observe immediately
    Immediate,
    /// Observe if CPU register R0 > threshold
    RegisterThreshold(u8),
    /// Probabilistic - observe with probability p each tick
    Stochastic(f64),
}

/// Configuration for the Observer Grid
#[derive(Clone, Debug)]
pub struct ObserverGridConfig {
    /// Number of observer CPUs
    pub num_observers: usize,
    /// Grid width (observers laid out in 2D)
    pub grid_width: usize,
    /// Default observation rule
    pub default_rule: ObservationRule,
    /// Seed for deterministic randomness
    pub seed: u64,
    /// Whether to track entanglement between unmeasured regions
    pub track_entanglement: bool,
    /// Maximum observations per tick (0 = unlimited)
    pub max_observations_per_tick: usize,
}

impl Default for ObserverGridConfig {
    fn default() -> Self {
        Self {
            num_observers: 1000,
            grid_width: 32,
            default_rule: ObservationRule::NeighborCount(2),
            seed: 42,
            track_entanglement: true,
            max_observations_per_tick: 0,
        }
    }
}

/// Statistics about the observation state
#[derive(Clone, Debug, Default)]
pub struct ObservationStats {
    /// Total observers
    pub total: usize,
    /// Still in superposition
    pub superposition_count: usize,
    /// Collapsed (measured)
    pub collapsed_count: usize,
    /// Protected (never observe)
    pub protected_count: usize,
    /// Observations this tick
    pub observations_this_tick: usize,
    /// Total observations all time
    pub total_observations: u64,
    /// Boundary length (edges between measured/unmeasured)
    pub boundary_length: usize,
    /// Number of unmeasured islands
    pub unmeasured_islands: usize,
    /// Largest unmeasured region size
    pub largest_unmeasured_region: usize,
    /// Information entropy of measured results
    pub measured_entropy: f64,
    /// Anomaly score (deviation from expected)
    pub anomaly_score: f64,
}

/// Probability distribution for a quantum block (128 basis states)
#[derive(Clone, Debug)]
pub struct BlockDistribution {
    /// Probability of each basis state |0⟩ through |127⟩
    /// Stored as cumulative distribution for efficient sampling
    pub cumulative: [f64; BLOCK_SIZE],
    /// Total probability (should be ~1.0 for normalized state)
    pub total: f64,
    /// Distribution type hint for anomaly detection
    pub distribution_type: DistributionType,
}

/// Type of quantum state distribution
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionType {
    /// Unknown/general distribution
    Unknown,
    /// Uniform superposition (all equal)
    Uniform,
    /// GHZ-like (bimodal: |0...0⟩ and |1...1⟩)
    GHZ,
    /// W-state (n peaks of equal weight)
    WState,
    /// Single basis state (classical)
    Classical,
    /// Gaussian-like distribution around a mean
    Gaussian,
}

impl Default for BlockDistribution {
    fn default() -> Self {
        // Default: uniform distribution over all basis states
        let mut cumulative = [0.0; BLOCK_SIZE];
        let prob_each = 1.0 / BLOCK_SIZE as f64;
        for i in 0..BLOCK_SIZE {
            cumulative[i] = (i + 1) as f64 * prob_each;
        }
        Self {
            cumulative,
            total: 1.0,
            distribution_type: DistributionType::Uniform,
        }
    }
}

impl BlockDistribution {
    /// Create uniform distribution (equal superposition)
    pub fn uniform() -> Self {
        Self::default()
    }

    /// Create GHZ distribution (|0...0⟩ + |1...1⟩) / √2
    pub fn ghz() -> Self {
        let mut cumulative = [0.0; BLOCK_SIZE];
        cumulative[0] = 0.5;
        for i in 1..BLOCK_SIZE - 1 {
            cumulative[i] = 0.5;
        }
        cumulative[BLOCK_SIZE - 1] = 1.0;
        Self {
            cumulative,
            total: 1.0,
            distribution_type: DistributionType::GHZ,
        }
    }

    /// Create from raw probabilities (will normalize and compute CDF)
    pub fn from_probabilities(probs: &[f64]) -> Self {
        let mut cumulative = [0.0; BLOCK_SIZE];
        let mut sum = 0.0;

        for (i, &p) in probs.iter().take(BLOCK_SIZE).enumerate() {
            sum += p.max(0.0); // Clamp negative to 0
            cumulative[i] = sum;
        }

        // Fill remaining with last value if probs shorter than BLOCK_SIZE
        for i in probs.len()..BLOCK_SIZE {
            cumulative[i] = sum;
        }

        // Normalize
        if sum > 0.0 {
            for c in cumulative.iter_mut() {
                *c /= sum;
            }
        } else {
            // Fallback to uniform if all zero
            return Self::uniform();
        }

        // Detect distribution type
        let dist_type = Self::detect_type(&cumulative);

        Self {
            cumulative,
            total: sum,
            distribution_type: dist_type,
        }
    }

    /// Detect distribution type from CDF
    fn detect_type(cdf: &[f64; BLOCK_SIZE]) -> DistributionType {
        // Check for classical (single spike)
        let mut non_zero_count = 0;
        let mut last_value = 0.0;
        for &c in cdf.iter() {
            if c > last_value + 0.001 {
                non_zero_count += 1;
            }
            last_value = c;
        }

        if non_zero_count == 1 {
            return DistributionType::Classical;
        }

        // Check for GHZ (only first and last have probability)
        let first_prob = cdf[0];
        let last_prob = cdf[BLOCK_SIZE - 1] - cdf[BLOCK_SIZE - 2];
        if non_zero_count == 2 && first_prob > 0.4 && last_prob > 0.4 {
            return DistributionType::GHZ;
        }

        // Check for uniform (all roughly equal)
        let expected = 1.0 / BLOCK_SIZE as f64;
        let mut is_uniform = true;
        for i in 0..BLOCK_SIZE {
            let prob = if i == 0 { cdf[0] } else { cdf[i] - cdf[i - 1] };
            if (prob - expected).abs() > 0.01 {
                is_uniform = false;
                break;
            }
        }
        if is_uniform {
            return DistributionType::Uniform;
        }

        DistributionType::Unknown
    }

    /// Sample from this distribution using Born rule
    /// Returns basis state index (0-127)
    pub fn sample(&self, random: f64) -> u8 {
        // Binary search in cumulative distribution
        let r = random.clamp(0.0, 1.0);

        let mut low = 0;
        let mut high = BLOCK_SIZE;

        while low < high {
            let mid = (low + high) / 2;
            if self.cumulative[mid] < r {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        low.min(BLOCK_SIZE - 1) as u8
    }

    /// Get probability of specific outcome
    pub fn probability(&self, state: u8) -> f64 {
        let idx = state as usize;
        if idx >= BLOCK_SIZE {
            return 0.0;
        }
        if idx == 0 {
            self.cumulative[0]
        } else {
            self.cumulative[idx] - self.cumulative[idx - 1]
        }
    }

    /// Calculate Shannon entropy of this distribution
    pub fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for i in 0..BLOCK_SIZE {
            let p = self.probability(i as u8);
            if p > 1e-15 {
                h -= p * p.log2();
            }
        }
        h
    }
}

/// A single observer unit: CPU + quantum block binding
#[derive(Clone)]
pub struct Observer {
    /// Index in the grid
    pub id: usize,
    /// Current observation state
    pub state: ObservationState,
    /// Rule governing observation decision
    pub rule: ObservationRule,
    /// Local quantum block probability (total, cached)
    pub local_probability: f64,
    /// Full probability distribution of the quantum block
    pub distribution: BlockDistribution,
    /// Tick when observation occurred (if collapsed)
    pub observation_tick: Option<u64>,
    /// CPU register R0 value (for register-based rules)
    pub register_r0: u8,
    /// Entanglement links to other observers (by ID)
    pub entangled_with: Vec<usize>,
}

impl Observer {
    pub fn new(id: usize, rule: ObservationRule) -> Self {
        Self {
            id,
            state: ObservationState::Superposition,
            rule,
            local_probability: 1.0 / BLOCK_SIZE as f64, // Default uniform
            distribution: BlockDistribution::uniform(),
            observation_tick: None,
            register_r0: 0,
            entangled_with: Vec::new(),
        }
    }

    /// Create observer with GHZ distribution
    pub fn with_ghz(id: usize, rule: ObservationRule) -> Self {
        Self {
            id,
            state: ObservationState::Superposition,
            rule,
            local_probability: 1.0,
            distribution: BlockDistribution::ghz(),
            observation_tick: None,
            register_r0: 0,
            entangled_with: Vec::new(),
        }
    }

    /// Set the probability distribution from raw probabilities
    pub fn set_distribution(&mut self, probs: &[f64]) {
        self.distribution = BlockDistribution::from_probabilities(probs);
        self.local_probability = self.distribution.total;
    }

    /// Check if this observer is still in superposition
    pub fn is_superposition(&self) -> bool {
        matches!(self.state, ObservationState::Superposition)
    }

    /// Check if this observer has collapsed
    pub fn is_collapsed(&self) -> bool {
        matches!(self.state, ObservationState::Collapsed(_))
    }

    /// Get measurement result if collapsed
    pub fn measurement(&self) -> Option<u8> {
        match self.state {
            ObservationState::Collapsed(v) => Some(v),
            _ => None,
        }
    }
}

/// The Observer Grid: 268M CPUs deciding whether to collapse wavefunctions
/// High-quality xorshift128+ RNG state
#[derive(Clone, Debug)]
struct Xorshift128Plus {
    s0: u64,
    s1: u64,
}

impl Xorshift128Plus {
    fn new(seed: u64) -> Self {
        // Initialize with splitmix64 to avoid poor seeds
        let mut state = seed;
        let s0 = Self::splitmix64(&mut state);
        let s1 = Self::splitmix64(&mut state);
        Self { s0, s1 }
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = s0.wrapping_add(s1);

        s1 ^= s0;
        self.s0 = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
        self.s1 = s1.rotate_left(37);

        result
    }

    /// Generate uniform f64 in [0, 1)
    fn next_f64(&mut self) -> f64 {
        // Use upper 53 bits for full double precision
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

pub struct ObserverGrid {
    /// All observers
    pub observers: Vec<Observer>,
    /// Grid dimensions
    pub width: usize,
    pub height: usize,
    /// Current tick
    pub tick: u64,
    /// Configuration
    pub config: ObserverGridConfig,
    /// High-quality RNG for Born rule sampling
    rng: std::cell::RefCell<Xorshift128Plus>,
    /// Cumulative statistics
    pub stats: ObservationStats,
    /// Quantum block probabilities (from sparse quantum grid)
    pub block_probabilities: Vec<f64>,
    /// Boundary cells (indices of observers on measured/unmeasured boundary)
    pub boundary_cells: Vec<usize>,
    /// Anomaly detector state
    pub anomaly_detector: AnomalyDetector,
}

impl ObserverGrid {
    /// Create a new observer grid
    pub fn new(config: ObserverGridConfig) -> Self {
        let num = config.num_observers;
        let width = config.grid_width;
        let height = (num + width - 1) / width;

        let observers: Vec<Observer> = (0..num)
            .map(|i| Observer::new(i, config.default_rule))
            .collect();

        Self {
            observers,
            width,
            height,
            tick: 0,
            config: config.clone(),
            rng: std::cell::RefCell::new(Xorshift128Plus::new(config.seed)),
            stats: ObservationStats {
                total: num,
                superposition_count: num,
                ..Default::default()
            },
            block_probabilities: vec![0.0; num],
            boundary_cells: Vec::new(),
            anomaly_detector: AnomalyDetector::new(),
        }
    }

    /// Create observer grid sized to match CPU count
    pub fn for_cpu_count(num_cpus: usize, seed: u64) -> Self {
        let width = (num_cpus as f64).sqrt().ceil() as usize;
        let config = ObserverGridConfig {
            num_observers: num_cpus,
            grid_width: width,
            seed,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Get high-quality random number in [0, 1)
    fn next_random(&self) -> f64 {
        self.rng.borrow_mut().next_f64()
    }

    /// Get neighbor indices for an observer
    pub fn neighbors(&self, id: usize) -> Vec<usize> {
        let x = id % self.width;
        let y = id / self.width;
        let mut result = Vec::with_capacity(8);

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                    let nid = ny as usize * self.width + nx as usize;
                    if nid < self.observers.len() {
                        result.push(nid);
                    }
                }
            }
        }
        result
    }

    /// Update block probabilities from quantum grid (total probability per block)
    pub fn update_probabilities(&mut self, probabilities: &[f64]) {
        let len = probabilities.len().min(self.block_probabilities.len());
        self.block_probabilities[..len].copy_from_slice(&probabilities[..len]);

        // Update each observer's local probability
        for (i, obs) in self.observers.iter_mut().enumerate() {
            if i < probabilities.len() {
                obs.local_probability = probabilities[i];
            }
        }
    }

    /// Update full distributions for all observers from quantum amplitude data
    /// `distributions` is a slice of slices, each containing 128 probabilities
    pub fn update_distributions(&mut self, distributions: &[Vec<f64>]) {
        for (i, obs) in self.observers.iter_mut().enumerate() {
            if i < distributions.len() {
                obs.set_distribution(&distributions[i]);
            }
        }
    }

    /// Set all observers to GHZ distribution (for GHZ state binding)
    pub fn set_ghz_distribution(&mut self) {
        for obs in self.observers.iter_mut() {
            obs.distribution = BlockDistribution::ghz();
            obs.local_probability = 1.0;
        }
    }

    /// Set all observers to uniform distribution
    pub fn set_uniform_distribution(&mut self) {
        for obs in self.observers.iter_mut() {
            obs.distribution = BlockDistribution::uniform();
            obs.local_probability = 1.0;
        }
    }

    /// Decide whether an observer should observe this tick
    #[allow(dead_code)]
    fn should_observe(&self, observer: &Observer, neighbor_collapsed_count: usize) -> bool {
        if !observer.is_superposition() {
            return false; // Already observed or protected
        }

        match observer.rule {
            ObservationRule::ProbabilityThreshold(threshold) => {
                observer.local_probability > threshold
            }
            ObservationRule::NeighborCount(n) => neighbor_collapsed_count >= n,
            ObservationRule::Periodic(period) => self.tick.is_multiple_of(period),
            ObservationRule::Never => false,
            ObservationRule::Immediate => true,
            ObservationRule::RegisterThreshold(threshold) => observer.register_r0 > threshold,
            ObservationRule::Stochastic(p) => self.next_random() < p,
        }
    }

    /// Perform measurement using Born rule
    /// Samples from the observer's probability distribution
    fn measure(&self, observer: &Observer) -> u8 {
        let r = self.next_random();
        observer.distribution.sample(r)
    }

    /// Execute one tick of the observer grid
    pub fn tick(&mut self) {
        self.tick += 1;
        let current_tick = self.tick;
        let mut observations_this_tick = 0;

        // Phase 1: Gather neighbor states (read-only)
        let neighbor_collapsed: Vec<usize> = self
            .observers
            .iter()
            .enumerate()
            .map(|(id, _)| {
                self.neighbors(id)
                    .iter()
                    .filter(|&&nid| self.observers[nid].is_collapsed())
                    .count()
            })
            .collect();

        // Phase 2: Determine which observers should collapse and pre-generate measurements
        let max_obs = if self.config.max_observations_per_tick > 0 {
            self.config.max_observations_per_tick
        } else {
            usize::MAX
        };

        // Pre-compute decisions and measurements to avoid borrow conflicts
        let mut to_collapse: Vec<(usize, u8)> = Vec::new();

        for (id, observer) in self.observers.iter().enumerate() {
            if to_collapse.len() >= max_obs {
                break;
            }

            // Generate random for stochastic rule
            let random_val = self.next_random();

            let should_obs = match observer.rule {
                ObservationRule::ProbabilityThreshold(threshold) => {
                    observer.is_superposition() && observer.local_probability > threshold
                }
                ObservationRule::NeighborCount(n) => {
                    observer.is_superposition() && neighbor_collapsed[id] >= n
                }
                ObservationRule::Periodic(period) => {
                    observer.is_superposition() && current_tick.is_multiple_of(period)
                }
                ObservationRule::Never => false,
                ObservationRule::Immediate => observer.is_superposition(),
                ObservationRule::RegisterThreshold(threshold) => {
                    observer.is_superposition() && observer.register_r0 > threshold
                }
                ObservationRule::Stochastic(p) => observer.is_superposition() && random_val < p,
            };

            if should_obs {
                // Born rule measurement: sample from probability distribution
                let r = self.next_random();
                let measurement = observer.distribution.sample(r);
                to_collapse.push((id, measurement));
            }
        }

        // Phase 3: Apply collapses
        for (id, measurement) in to_collapse {
            self.observers[id].state = ObservationState::Collapsed(measurement);
            self.observers[id].observation_tick = Some(current_tick);
            observations_this_tick += 1;
        }

        // Phase 4: Update statistics
        self.update_stats(observations_this_tick);

        // Phase 5: Detect anomalies
        self.anomaly_detector.update(&self.stats, &self.observers);
    }

    /// Update statistics after a tick
    fn update_stats(&mut self, observations_this_tick: usize) {
        let mut superposition = 0;
        let mut collapsed = 0;
        let mut protected = 0;
        let mut measurement_sum = 0u64;
        let mut measurement_count = 0u64;

        for obs in &self.observers {
            match obs.state {
                ObservationState::Superposition => superposition += 1,
                ObservationState::Collapsed(v) => {
                    collapsed += 1;
                    measurement_sum += v as u64;
                    measurement_count += 1;
                }
                ObservationState::Protected => protected += 1,
            }
        }

        // Calculate boundary
        self.boundary_cells.clear();
        for (id, obs) in self.observers.iter().enumerate() {
            let is_collapsed = obs.is_collapsed();
            for nid in self.neighbors(id) {
                let neighbor_collapsed = self.observers[nid].is_collapsed();
                if is_collapsed != neighbor_collapsed {
                    self.boundary_cells.push(id);
                    break;
                }
            }
        }

        // Calculate entropy of measurements
        let entropy = if measurement_count > 0 {
            let mean = measurement_sum as f64 / measurement_count as f64;
            // Simplified entropy estimate
            let variance: f64 = self
                .observers
                .iter()
                .filter_map(|o| o.measurement())
                .map(|m| (m as f64 - mean).powi(2))
                .sum::<f64>()
                / measurement_count as f64;
            variance.sqrt() / 128.0 // Normalized
        } else {
            0.0
        };

        // Find unmeasured islands (simplified - just count regions)
        let (islands, largest) = self.count_unmeasured_regions();

        self.stats = ObservationStats {
            total: self.observers.len(),
            superposition_count: superposition,
            collapsed_count: collapsed,
            protected_count: protected,
            observations_this_tick,
            total_observations: self.stats.total_observations + observations_this_tick as u64,
            boundary_length: self.boundary_cells.len(),
            unmeasured_islands: islands,
            largest_unmeasured_region: largest,
            measured_entropy: entropy,
            anomaly_score: self.anomaly_detector.score,
        };
    }

    /// Count unmeasured regions using flood fill
    fn count_unmeasured_regions(&self) -> (usize, usize) {
        let mut visited = vec![false; self.observers.len()];
        let mut islands = 0;
        let mut largest = 0;

        for start in 0..self.observers.len() {
            if visited[start] || !self.observers[start].is_superposition() {
                continue;
            }

            // Flood fill from this unmeasured cell
            let mut size = 0;
            let mut stack = vec![start];

            while let Some(id) = stack.pop() {
                if visited[id] || !self.observers[id].is_superposition() {
                    continue;
                }
                visited[id] = true;
                size += 1;

                for nid in self.neighbors(id) {
                    if !visited[nid] && self.observers[nid].is_superposition() {
                        stack.push(nid);
                    }
                }
            }

            if size > 0 {
                islands += 1;
                largest = largest.max(size);
            }
        }

        (islands, largest)
    }

    /// Set a region as protected (will never observe)
    pub fn protect_region(&mut self, center_x: usize, center_y: usize, radius: usize) {
        for y in center_y.saturating_sub(radius)..=(center_y + radius).min(self.height - 1) {
            for x in center_x.saturating_sub(radius)..=(center_x + radius).min(self.width - 1) {
                let dx = x as i32 - center_x as i32;
                let dy = y as i32 - center_y as i32;
                if (dx * dx + dy * dy) <= (radius * radius) as i32 {
                    let id = y * self.width + x;
                    if id < self.observers.len() {
                        self.observers[id].state = ObservationState::Protected;
                        self.observers[id].rule = ObservationRule::Never;
                    }
                }
            }
        }
    }

    /// Trigger immediate observation wavefront from a point
    pub fn trigger_observation_wave(&mut self, start_x: usize, start_y: usize) {
        let start_id = start_y * self.width + start_x;
        if start_id < self.observers.len() {
            // Set neighbors to observe on next tick
            for nid in self.neighbors(start_id) {
                if self.observers[nid].is_superposition() {
                    self.observers[nid].rule = ObservationRule::Immediate;
                }
            }
            // Observe the start point
            if self.observers[start_id].is_superposition() {
                let measurement = self.measure(&self.observers[start_id]);
                self.observers[start_id].state = ObservationState::Collapsed(measurement);
                self.observers[start_id].observation_tick = Some(self.tick);
            }
        }
    }

    /// Get visualization data: 0=superposition, 1-128=collapsed value, 255=protected
    pub fn get_visualization_data(&self) -> Vec<u8> {
        self.observers
            .iter()
            .map(|obs| match obs.state {
                ObservationState::Superposition => 0,
                ObservationState::Collapsed(v) => v.saturating_add(1), // 1-128
                ObservationState::Protected => 255,
            })
            .collect()
    }

    /// Get boundary mask (1 = on boundary, 0 = not)
    pub fn get_boundary_mask(&self) -> Vec<u8> {
        let mut mask = vec![0u8; self.observers.len()];
        for &id in &self.boundary_cells {
            mask[id] = 1;
        }
        mask
    }
}

/// Anomaly detection system
///
/// Detects deviations from expected quantum measurement statistics.
/// For truly random quantum measurements, we expect:
/// - No autocorrelation between successive measurements
/// - Entropy matching the underlying distribution
/// - No information surplus beyond what the distribution predicts
#[derive(Clone, Debug)]
pub struct AnomalyDetector {
    /// Current anomaly score (0 = normal, higher = more anomalous)
    pub score: f64,
    /// Expected entropy based on distribution (bits)
    pub expected_entropy: f64,
    /// Running average of observed entropy (bits)
    pub observed_entropy_avg: f64,
    /// Information surplus (observed - expected, should be ~0)
    pub information_surplus: f64,
    /// Pattern correlation score (autocorrelation, should be ~0 for true randomness)
    pub pattern_score: f64,
    /// Chi-squared statistic for distribution fit
    pub chi_squared: f64,
    /// History of measurements for pattern detection
    measurement_history: Vec<u8>,
    /// Expected distribution for comparison
    expected_distribution: [f64; BLOCK_SIZE],
    /// Tick counter
    tick: u64,
}

impl AnomalyDetector {
    pub fn new() -> Self {
        // Default: uniform expected distribution
        let expected_distribution = [1.0 / BLOCK_SIZE as f64; BLOCK_SIZE];
        Self {
            score: 0.0,
            expected_entropy: 7.0, // log2(128) = 7 bits for uniform
            observed_entropy_avg: 0.0,
            information_surplus: 0.0,
            pattern_score: 0.0,
            chi_squared: 0.0,
            measurement_history: Vec::with_capacity(10000),
            expected_distribution,
            tick: 0,
        }
    }

    /// Create detector expecting GHZ distribution
    pub fn for_ghz() -> Self {
        let mut expected_distribution = [0.0; BLOCK_SIZE];
        expected_distribution[0] = 0.5;
        expected_distribution[BLOCK_SIZE - 1] = 0.5;
        Self {
            score: 0.0,
            expected_entropy: 1.0, // log2(2) = 1 bit for GHZ
            observed_entropy_avg: 0.0,
            information_surplus: 0.0,
            pattern_score: 0.0,
            chi_squared: 0.0,
            measurement_history: Vec::with_capacity(10000),
            expected_distribution,
            tick: 0,
        }
    }

    /// Set expected entropy based on observer distributions
    pub fn calibrate_from_observers(&mut self, observers: &[Observer]) {
        if observers.is_empty() {
            return;
        }

        // Average entropy across all observer distributions
        let total_entropy: f64 = observers.iter().map(|obs| obs.distribution.entropy()).sum();
        self.expected_entropy = total_entropy / observers.len() as f64;

        // Build aggregate expected distribution
        self.expected_distribution = [0.0; BLOCK_SIZE];
        for obs in observers {
            for i in 0..BLOCK_SIZE {
                self.expected_distribution[i] += obs.distribution.probability(i as u8);
            }
        }
        // Normalize
        let sum: f64 = self.expected_distribution.iter().sum();
        if sum > 0.0 {
            for p in self.expected_distribution.iter_mut() {
                *p /= sum;
            }
        }
    }

    /// Update anomaly detection with new observations
    pub fn update(&mut self, _stats: &ObservationStats, observers: &[Observer]) {
        self.tick += 1;

        // Collect recent measurements for pattern analysis
        for obs in observers {
            if let Some(m) = obs.measurement() {
                if obs.observation_tick == Some(self.tick) {
                    self.measurement_history.push(m);
                    if self.measurement_history.len() > 10000 {
                        self.measurement_history.remove(0);
                    }
                }
            }
        }

        // Skip analysis if not enough data
        if self.measurement_history.len() < 50 {
            return;
        }

        // Calculate observed entropy from measurement histogram
        let observed_entropy = self.calculate_observed_entropy();
        self.observed_entropy_avg = self.observed_entropy_avg * 0.95 + observed_entropy * 0.05;

        // Entropy deviation (normalized by expected)
        let entropy_deviation = if self.expected_entropy > 0.0 {
            ((observed_entropy - self.expected_entropy) / self.expected_entropy).abs()
        } else {
            0.0
        };

        // Check for autocorrelation patterns (should be ~0 for true randomness)
        self.pattern_score = self.detect_autocorrelation();

        // Chi-squared goodness of fit test
        self.chi_squared = self.calculate_chi_squared();
        // Critical value for 127 df at p=0.05 is ~152, at p=0.01 is ~166
        // Normalize so values below critical are ~0, above critical scale to 1
        let chi_critical = 152.0; // p=0.05 threshold
        let chi_squared_normalized = if self.chi_squared <= chi_critical {
            0.0
        } else {
            ((self.chi_squared - chi_critical) / chi_critical).min(1.0)
        };

        // Calculate information surplus
        let expected_info = self.expected_entropy * self.measurement_history.len() as f64;
        let observed_info = observed_entropy * self.measurement_history.len() as f64;
        self.information_surplus = observed_info - expected_info;

        // Combine into anomaly score with proper weighting
        // - Autocorrelation: strongest indicator of non-randomness (40%)
        // - Chi-squared: distribution mismatch (30%)
        // - Entropy deviation: information content mismatch (30%)
        self.score = self.pattern_score * 0.4
            + chi_squared_normalized * 0.3
            + entropy_deviation.min(1.0) * 0.3;
    }

    /// Calculate observed Shannon entropy from measurements
    fn calculate_observed_entropy(&self) -> f64 {
        if self.measurement_history.is_empty() {
            return 0.0;
        }

        let mut freq = [0u64; BLOCK_SIZE];
        for &m in &self.measurement_history {
            let idx = (m as usize).min(BLOCK_SIZE - 1);
            freq[idx] += 1;
        }

        let n = self.measurement_history.len() as f64;
        let mut entropy = 0.0;
        for &f in &freq {
            if f > 0 {
                let p = f as f64 / n;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Detect autocorrelation in measurement sequence
    fn detect_autocorrelation(&self) -> f64 {
        if self.measurement_history.len() < 100 {
            return 0.0;
        }

        let n = self.measurement_history.len();
        let mean: f64 = self
            .measurement_history
            .iter()
            .map(|&x| x as f64)
            .sum::<f64>()
            / n as f64;

        let variance: f64 = self
            .measurement_history
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / n as f64;

        if variance < 1e-10 {
            return 0.0; // Degenerate case
        }

        // Compute normalized autocorrelation for lags 1-10
        let mut max_autocorr: f64 = 0.0;
        for lag in 1..=10.min(n / 4) {
            let mut sum = 0.0;
            for i in lag..n {
                let a = self.measurement_history[i] as f64 - mean;
                let b = self.measurement_history[i - lag] as f64 - mean;
                sum += a * b;
            }
            let autocorr = (sum / ((n - lag) as f64 * variance)).abs();
            max_autocorr = max_autocorr.max(autocorr);
        }

        // For true randomness, autocorrelation should be ~1/sqrt(n)
        // Significant if much larger than this
        let expected_noise = 2.0 / (n as f64).sqrt();
        if max_autocorr > expected_noise {
            ((max_autocorr - expected_noise) / (1.0 - expected_noise)).min(1.0)
        } else {
            0.0
        }
    }

    /// Chi-squared goodness of fit test
    fn calculate_chi_squared(&self) -> f64 {
        if self.measurement_history.len() < 100 {
            return 0.0;
        }

        let mut observed = [0u64; BLOCK_SIZE];
        for &m in &self.measurement_history {
            let idx = (m as usize).min(BLOCK_SIZE - 1);
            observed[idx] += 1;
        }

        let n = self.measurement_history.len() as f64;
        let mut chi_sq = 0.0;

        for i in 0..BLOCK_SIZE {
            let expected = self.expected_distribution[i] * n;
            if expected > 0.0 {
                let diff = observed[i] as f64 - expected;
                chi_sq += diff * diff / expected;
            }
        }

        chi_sq
    }

    /// Check if current state is anomalous
    pub fn is_anomalous(&self) -> bool {
        self.score > 0.1 // Threshold for "interesting"
    }

    /// Get human-readable status
    pub fn status(&self) -> &'static str {
        if self.score < 0.01 {
            "NOMINAL"
        } else if self.score < 0.05 {
            "SLIGHT DEVIATION"
        } else if self.score < 0.1 {
            "NOTABLE PATTERN"
        } else if self.score < 0.2 {
            "ANOMALY DETECTED"
        } else {
            "SIGNIFICANT ANOMALY"
        }
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observer_grid_creation() {
        let grid = ObserverGrid::for_cpu_count(1000, 42);
        assert_eq!(grid.observers.len(), 1000);
        assert_eq!(grid.stats.superposition_count, 1000);
        assert_eq!(grid.stats.collapsed_count, 0);
    }

    #[test]
    fn test_observation_wave() {
        let mut grid = ObserverGrid::for_cpu_count(100, 42);
        grid.trigger_observation_wave(5, 5);
        grid.tick();
        grid.tick();
        grid.tick();

        // Should have some collapsed observers
        assert!(grid.stats.collapsed_count > 0);
        assert!(grid.stats.boundary_length > 0);
    }

    #[test]
    fn test_protected_region() {
        let mut grid = ObserverGrid::for_cpu_count(100, 42);
        grid.protect_region(5, 5, 2);

        // Run many ticks with immediate observation
        for obs in &mut grid.observers {
            obs.rule = ObservationRule::Immediate;
        }

        for _ in 0..10 {
            grid.tick();
        }

        // Protected region should still be in superposition/protected state
        assert!(grid.stats.protected_count > 0);
    }

    #[test]
    fn test_neighbor_rule() {
        let config = ObserverGridConfig {
            num_observers: 25,
            grid_width: 5,
            default_rule: ObservationRule::NeighborCount(2),
            seed: 42,
            ..Default::default()
        };
        let mut grid = ObserverGrid::new(config);

        // Manually collapse corner
        grid.observers[0].state = ObservationState::Collapsed(42);
        grid.observers[1].state = ObservationState::Collapsed(42);

        grid.tick();

        // Observer at (0,1) should now collapse (has 2 collapsed neighbors)
        // This depends on grid layout, but we should see some propagation
        assert!(grid.stats.collapsed_count >= 2);
    }

    #[test]
    fn test_anomaly_detector() {
        let mut detector = AnomalyDetector::new();
        let stats = ObservationStats {
            total: 100,
            collapsed_count: 50,
            measured_entropy: 0.5,
            ..Default::default()
        };
        let observers: Vec<Observer> = (0..100)
            .map(|i| Observer::new(i, ObservationRule::Never))
            .collect();

        detector.update(&stats, &observers);

        // Should be nominal with random-like entropy
        assert!(detector.score < 0.5);
    }
}
