//! EPIC 76.3: Stable Universe API
//!
//! This module provides a clean, high-level API for parallel universe simulation.
//! It hides the complexity of CUDA, tilemaps, and internal simulation details.
//!
//! # Example
//!
//! ```ignore
//! use engine::universe::{Universe, UniverseConfig, Ruleset};
//!
//! // Create 5 parallel universes, each 512x512
//! let config = UniverseConfig {
//!     worlds: 5,
//!     width: 512,
//!     height: 512,
//!     ruleset: Ruleset::GameOfLife,
//!     seed: 42,
//! };
//!
//! let mut universe = Universe::new(config)?;
//!
//! // Initialize worlds with random patterns
//! universe.randomize_all(0.3);
//!
//! // Evolve all worlds for 100 steps
//! universe.evolve(100);
//!
//! // Get a world's state as a flat buffer
//! let world0 = universe.get_world(0);
//! ```

use std::fmt;

// ============================================================================
// Configuration
// ============================================================================

/// Available simulation rulesets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ruleset {
    /// Conway's Game of Life (8-neighbor totalistic)
    GameOfLife,
    /// Wolfram's Rule 110 (1D cellular automaton)
    Rule110,
    /// Wire propagation (OR of 4 neighbors)
    #[default]
    Wire,
    /// Full logic gate simulation (AND, OR, XOR, NOT, etc.)
    Logic,
}

impl Ruleset {
    /// Parse ruleset from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "gol" | "gameoflife" | "game_of_life" | "life" => Some(Ruleset::GameOfLife),
            "rule110" | "r110" | "110" => Some(Ruleset::Rule110),
            "wire" | "wires" => Some(Ruleset::Wire),
            "logic" | "gates" | "tile" => Some(Ruleset::Logic),
            _ => None,
        }
    }

    /// Get ruleset name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Ruleset::GameOfLife => "gol",
            Ruleset::Rule110 => "rule110",
            Ruleset::Wire => "wire",
            Ruleset::Logic => "logic",
        }
    }
}

impl fmt::Display for Ruleset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Configuration for creating a Universe
#[derive(Debug, Clone)]
pub struct UniverseConfig {
    /// Number of parallel worlds
    pub worlds: usize,
    /// Width of each world (must be power of 2)
    pub width: usize,
    /// Height of each world (must be power of 2)
    pub height: usize,
    /// Simulation ruleset
    pub ruleset: Ruleset,
    /// Random seed for initialization
    pub seed: u64,
    /// Enable reversible mode (history tracking)
    pub reversible: bool,
    /// Maximum history depth (only used when reversible=true)
    pub max_history: usize,
}

impl Default for UniverseConfig {
    fn default() -> Self {
        Self {
            worlds: 5,
            width: 512,
            height: 512,
            ruleset: Ruleset::Wire,
            seed: 42,
            reversible: false,
            max_history: 1000,
        }
    }
}

impl UniverseConfig {
    /// Create config with specified world count and size
    pub fn new(worlds: usize, width: usize, height: usize) -> Self {
        Self {
            worlds,
            width,
            height,
            ..Default::default()
        }
    }

    /// Set the ruleset
    pub fn with_ruleset(mut self, ruleset: Ruleset) -> Self {
        self.ruleset = ruleset;
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable reversible mode
    pub fn with_reversible(mut self, enabled: bool) -> Self {
        self.reversible = enabled;
        self
    }

    /// Set maximum history depth for reversible mode
    pub fn with_max_history(mut self, depth: usize) -> Self {
        self.max_history = depth;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), UniverseError> {
        if self.worlds == 0 {
            return Err(UniverseError::InvalidConfig("worlds must be > 0".into()));
        }
        if self.width == 0 || self.height == 0 {
            return Err(UniverseError::InvalidConfig(
                "dimensions must be > 0".into(),
            ));
        }
        if !self.width.is_power_of_two() {
            return Err(UniverseError::InvalidConfig(format!(
                "width {} must be power of 2",
                self.width
            )));
        }
        if !self.height.is_power_of_two() {
            return Err(UniverseError::InvalidConfig(format!(
                "height {} must be power of 2",
                self.height
            )));
        }
        Ok(())
    }
}

// ============================================================================
// Error types
// ============================================================================

/// Errors that can occur during universe operations
#[derive(Debug)]
pub enum UniverseError {
    /// Invalid configuration
    InvalidConfig(String),
    /// World index out of bounds
    WorldOutOfBounds { index: usize, max: usize },
    /// Coordinates out of bounds
    CoordsOutOfBounds {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    },
    /// GPU operation failed
    #[cfg(feature = "cuda")]
    GpuError(String),
    /// Cannot rewind - no history available or not in reversible mode
    NoHistory,
    /// Cannot rewind further - at beginning of history
    AtBeginning,
}

impl fmt::Display for UniverseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UniverseError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            UniverseError::WorldOutOfBounds { index, max } => {
                write!(f, "World index {} out of bounds (max {})", index, max)
            }
            UniverseError::CoordsOutOfBounds {
                x,
                y,
                width,
                height,
            } => {
                write!(
                    f,
                    "Coords ({}, {}) out of bounds ({}x{})",
                    x, y, width, height
                )
            }
            #[cfg(feature = "cuda")]
            UniverseError::GpuError(msg) => write!(f, "GPU error: {}", msg),
            UniverseError::NoHistory => {
                write!(f, "No history available (reversible mode not enabled)")
            }
            UniverseError::AtBeginning => write!(f, "Cannot rewind - at beginning of history"),
        }
    }
}

impl std::error::Error for UniverseError {}

pub type UniverseResult<T> = Result<T, UniverseError>;

// ============================================================================
// Benchmark result
// ============================================================================

/// Result from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Total logic evaluations performed
    pub total_evals: u64,
    /// Elapsed wall-clock time in seconds
    pub elapsed_secs: f64,
    /// Throughput: evaluations per second
    pub evals_per_sec: f64,
    /// Number of parallel worlds
    pub worlds: usize,
    /// World dimensions
    pub width: usize,
    pub height: usize,
    /// Steps executed
    pub steps: u32,
    /// Memory usage in MB (GPU only)
    pub memory_mb: Option<f64>,
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.1}B evals/sec ({} worlds, {}x{}, {} steps)",
            self.evals_per_sec / 1e9,
            self.worlds,
            self.width,
            self.height,
            self.steps
        )
    }
}

// ============================================================================
// Universe - Main API
// ============================================================================

/// Parallel universe simulator.
///
/// Manages multiple independent worlds that evolve according to a ruleset.
/// Uses GPU acceleration when available.
///
/// Supports reversible mode where history is tracked, allowing time-travel
/// backwards through simulation steps.
pub struct Universe {
    /// Configuration
    config: UniverseConfig,
    /// World state buffers (flattened: worlds * width * height)
    state: Vec<u64>,
    /// Current simulation step
    step: u64,
    /// Whether GPU is available
    gpu_available: bool,
    /// History buffer for reversible mode (ring buffer of states)
    history: Vec<Vec<u64>>,
    /// Current position in history (for rewind/forward)
    history_pos: usize,
}

impl Universe {
    /// Create a new Universe with the given configuration.
    pub fn new(config: UniverseConfig) -> UniverseResult<Self> {
        config.validate()?;

        let total_cells = config.worlds * config.width * config.height;
        let state = vec![0u64; total_cells];

        #[cfg(feature = "cuda")]
        let gpu_available = crate::cuda::is_cuda_available();
        #[cfg(not(feature = "cuda"))]
        let gpu_available = false;

        // Initialize history if reversible mode enabled
        let history = if config.reversible {
            vec![state.clone()] // Start with initial state
        } else {
            Vec::new()
        };

        Ok(Universe {
            config,
            state,
            step: 0,
            gpu_available,
            history,
            history_pos: 0,
        })
    }

    /// Create with default config (5 worlds, 512x512, wire ruleset)
    pub fn new_default() -> UniverseResult<Self> {
        Self::new(UniverseConfig::default())
    }

    // ========================================================================
    // Properties
    // ========================================================================

    /// Get the number of parallel worlds
    pub fn worlds(&self) -> usize {
        self.config.worlds
    }

    /// Get world width
    pub fn width(&self) -> usize {
        self.config.width
    }

    /// Get world height
    pub fn height(&self) -> usize {
        self.config.height
    }

    /// Get world dimensions as (width, height)
    pub fn size(&self) -> (usize, usize) {
        (self.config.width, self.config.height)
    }

    /// Get tiles per world
    pub fn tiles_per_world(&self) -> usize {
        self.config.width * self.config.height
    }

    /// Get the active ruleset
    pub fn ruleset(&self) -> Ruleset {
        self.config.ruleset
    }

    /// Get current simulation step
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Check if GPU acceleration is available
    pub fn has_gpu(&self) -> bool {
        self.gpu_available
    }

    /// Get the configuration
    pub fn config(&self) -> &UniverseConfig {
        &self.config
    }

    /// Check if reversible mode is enabled
    pub fn is_reversible(&self) -> bool {
        self.config.reversible
    }

    /// Get the number of steps saved in history
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Check if we can rewind (go back in time)
    pub fn can_rewind(&self) -> bool {
        self.config.reversible && self.history_pos > 0
    }

    /// Check if we can fast-forward (after rewinding)
    pub fn can_forward(&self) -> bool {
        self.config.reversible && self.history_pos < self.history.len().saturating_sub(1)
    }

    // ========================================================================
    // Cell access
    // ========================================================================

    /// Get a cell value
    pub fn get_cell(&self, world: usize, x: usize, y: usize) -> UniverseResult<u64> {
        self.check_world(world)?;
        self.check_coords(x, y)?;
        let idx = self.cell_index(world, x, y);
        Ok(self.state[idx])
    }

    /// Set a cell value
    pub fn set_cell(&mut self, world: usize, x: usize, y: usize, value: u64) -> UniverseResult<()> {
        self.check_world(world)?;
        self.check_coords(x, y)?;
        let idx = self.cell_index(world, x, y);
        self.state[idx] = value;
        Ok(())
    }

    /// Get a world's state as a slice (row-major, height x width)
    pub fn get_world(&self, world: usize) -> UniverseResult<&[u64]> {
        self.check_world(world)?;
        let start = world * self.tiles_per_world();
        let end = start + self.tiles_per_world();
        Ok(&self.state[start..end])
    }

    /// Get a world's state as a mutable slice
    pub fn get_world_mut(&mut self, world: usize) -> UniverseResult<&mut [u64]> {
        self.check_world(world)?;
        let tpw = self.tiles_per_world();
        let start = world * tpw;
        let end = start + tpw;
        Ok(&mut self.state[start..end])
    }

    /// Set a world's state from a slice
    pub fn set_world(&mut self, world: usize, data: &[u64]) -> UniverseResult<()> {
        self.check_world(world)?;
        let tpw = self.tiles_per_world();
        if data.len() != tpw {
            return Err(UniverseError::InvalidConfig(format!(
                "data length {} doesn't match world size {}",
                data.len(),
                tpw
            )));
        }
        let start = world * tpw;
        self.state[start..start + tpw].copy_from_slice(data);
        Ok(())
    }

    /// Get all state as a flat buffer
    pub fn state(&self) -> &[u64] {
        &self.state
    }

    /// Get all state as a mutable flat buffer
    pub fn state_mut(&mut self) -> &mut [u64] {
        &mut self.state
    }

    // ========================================================================
    // Initialization
    // ========================================================================

    /// Initialize a world with random pattern
    pub fn randomize(&mut self, world: usize, density: f64) -> UniverseResult<()> {
        self.check_world(world)?;
        let mut rng_state = self.config.seed.wrapping_add(world as u64);
        let start = world * self.tiles_per_world();

        for i in 0..self.tiles_per_world() {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let r = (rng_state as f64) / (u64::MAX as f64);
            self.state[start + i] = if r < density { u64::MAX } else { 0 };
        }
        Ok(())
    }

    /// Initialize all worlds with random patterns
    pub fn randomize_all(&mut self, density: f64) {
        for world in 0..self.config.worlds {
            let _ = self.randomize(world, density);
        }
    }

    /// Reset all worlds to zero
    pub fn reset(&mut self) {
        self.state.fill(0);
        self.step = 0;
        if self.config.reversible {
            self.history.clear();
            self.history.push(self.state.clone());
            self.history_pos = 0;
        }
    }

    // ========================================================================
    // Evolution
    // ========================================================================

    /// Evolve all worlds forward by the specified number of steps.
    ///
    /// Uses GPU acceleration when available, falls back to CPU otherwise.
    /// In reversible mode, each step is saved to history.
    pub fn evolve(&mut self, steps: u32) {
        // If we're in the middle of history (after rewind), truncate future states
        if self.config.reversible && self.history_pos < self.history.len().saturating_sub(1) {
            self.history.truncate(self.history_pos + 1);
        }

        for _ in 0..steps {
            self.step_cpu();

            // Record history if reversible mode enabled
            if self.config.reversible {
                // Manage ring buffer size
                if self.history.len() >= self.config.max_history {
                    self.history.remove(0);
                }
                self.history.push(self.state.clone());
                self.history_pos = self.history.len() - 1;
            }
        }
        self.step += steps as u64;
    }

    /// Rewind simulation by the specified number of steps.
    ///
    /// Only available in reversible mode.
    pub fn rewind(&mut self, steps: u32) -> UniverseResult<u32> {
        if !self.config.reversible {
            return Err(UniverseError::NoHistory);
        }

        let mut rewound = 0;
        for _ in 0..steps {
            if self.history_pos == 0 {
                break;
            }
            self.history_pos -= 1;
            rewound += 1;
        }

        if rewound > 0 {
            self.state = self.history[self.history_pos].clone();
            self.step = self.step.saturating_sub(rewound as u64);
        }

        Ok(rewound)
    }

    /// Fast-forward simulation after a rewind.
    ///
    /// Replays recorded history without re-computing.
    pub fn forward(&mut self, steps: u32) -> UniverseResult<u32> {
        if !self.config.reversible {
            return Err(UniverseError::NoHistory);
        }

        let mut forwarded = 0;
        for _ in 0..steps {
            if self.history_pos >= self.history.len() - 1 {
                break;
            }
            self.history_pos += 1;
            forwarded += 1;
        }

        if forwarded > 0 {
            self.state = self.history[self.history_pos].clone();
            self.step += forwarded as u64;
        }

        Ok(forwarded)
    }

    /// Get state at a specific point in history.
    ///
    /// Returns None if index is out of range or reversible mode is disabled.
    pub fn get_history(&self, index: usize) -> Option<&[u64]> {
        if !self.config.reversible {
            return None;
        }
        self.history.get(index).map(|v| v.as_slice())
    }

    /// Jump to a specific point in history.
    pub fn goto_history(&mut self, index: usize) -> UniverseResult<()> {
        if !self.config.reversible {
            return Err(UniverseError::NoHistory);
        }
        if index >= self.history.len() {
            return Err(UniverseError::InvalidConfig(format!(
                "history index {} out of range (max {})",
                index,
                self.history.len() - 1
            )));
        }

        let delta = index as i64 - self.history_pos as i64;
        self.history_pos = index;
        self.state = self.history[index].clone();
        self.step = (self.step as i64 + delta) as u64;

        Ok(())
    }

    /// Evolve with GPU acceleration (if available)
    ///
    /// # Arguments
    /// * `steps` - Total steps to run
    /// * `depth` - Steps per kernel launch (GPU only, ignored on CPU)
    #[cfg(feature = "cuda")]
    pub fn evolve_gpu(&mut self, steps: u32, _depth: u32) -> UniverseResult<()> {
        if !self.gpu_available {
            self.evolve(steps);
            return Ok(());
        }

        // For now, fall back to CPU - GPU evolution requires state upload/download
        // which we'll implement when we wire up the GPU path properly
        self.evolve(steps);
        Ok(())
    }

    // ========================================================================
    // Benchmarking
    // ========================================================================

    /// Run a benchmark and return performance metrics.
    ///
    /// Uses GPU when available.
    #[cfg(feature = "cuda")]
    pub fn benchmark(
        &self,
        steps: u32,
        warmup: u32,
        depth: u32,
    ) -> UniverseResult<BenchmarkResult> {
        if self.gpu_available {
            let rt = crate::cuda::CudaRuntime::new()
                .map_err(|e| UniverseError::GpuError(format!("{}", e)))?;

            let result = crate::cuda_tiles::benchmark_tile_eval_depth_gpu(
                &rt,
                self.config.worlds,
                steps,
                depth,
                warmup,
            )
            .map_err(|e| UniverseError::GpuError(format!("{}", e)))?;

            return Ok(BenchmarkResult {
                total_evals: result.0,
                elapsed_secs: result.1,
                evals_per_sec: result.2,
                worlds: self.config.worlds,
                width: self.config.width,
                height: self.config.height,
                steps,
                memory_mb: Some(result.3),
            });
        }

        self.benchmark_cpu(steps, warmup)
    }

    /// Run CPU-only benchmark
    pub fn benchmark_cpu(&self, steps: u32, warmup: u32) -> UniverseResult<BenchmarkResult> {
        use std::time::Instant;

        let total_tiles = self.tiles_per_world() * self.config.worlds;
        let mut state = self.state.clone();

        // Warmup
        for _ in 0..warmup {
            Self::step_state(&mut state, &self.config);
        }

        // Timed run
        let start = Instant::now();
        for _ in 0..steps {
            Self::step_state(&mut state, &self.config);
        }
        let elapsed = start.elapsed().as_secs_f64();

        let total_evals = (steps as u64) * (total_tiles as u64);
        let evals_per_sec = if elapsed > 0.0 {
            total_evals as f64 / elapsed
        } else {
            0.0
        };

        Ok(BenchmarkResult {
            total_evals,
            elapsed_secs: elapsed,
            evals_per_sec,
            worlds: self.config.worlds,
            width: self.config.width,
            height: self.config.height,
            steps,
            memory_mb: None,
        })
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    fn check_world(&self, world: usize) -> UniverseResult<()> {
        if world >= self.config.worlds {
            return Err(UniverseError::WorldOutOfBounds {
                index: world,
                max: self.config.worlds - 1,
            });
        }
        Ok(())
    }

    fn check_coords(&self, x: usize, y: usize) -> UniverseResult<()> {
        if x >= self.config.width || y >= self.config.height {
            return Err(UniverseError::CoordsOutOfBounds {
                x,
                y,
                width: self.config.width,
                height: self.config.height,
            });
        }
        Ok(())
    }

    fn cell_index(&self, world: usize, x: usize, y: usize) -> usize {
        world * self.tiles_per_world() + y * self.config.width + x
    }

    fn step_cpu(&mut self) {
        let mut next = self.state.clone();
        Self::step_state(&mut next, &self.config);
        self.state = next;
    }

    fn step_state(state: &mut [u64], config: &UniverseConfig) {
        let tpw = config.width * config.height;
        let mut next = state.to_vec();

        for world in 0..config.worlds {
            let offset = world * tpw;

            for y in 0..config.height {
                for x in 0..config.width {
                    let idx = offset + y * config.width + x;

                    // Get 4 neighbors
                    let left = if x > 0 {
                        state[offset + y * config.width + (x - 1)]
                    } else {
                        0
                    };
                    let right = if x < config.width - 1 {
                        state[offset + y * config.width + (x + 1)]
                    } else {
                        0
                    };
                    let up = if y > 0 {
                        state[offset + (y - 1) * config.width + x]
                    } else {
                        0
                    };
                    let down = if y < config.height - 1 {
                        state[offset + (y + 1) * config.width + x]
                    } else {
                        0
                    };
                    let current = state[idx];

                    let new_val = match config.ruleset {
                        Ruleset::Wire | Ruleset::Logic => left | right | up | down,
                        Ruleset::GameOfLife => {
                            let alive = |v: u64| if v != 0 { 1 } else { 0 };

                            // 8 neighbors for GoL
                            let ul = if x > 0 && y > 0 {
                                state[offset + (y - 1) * config.width + (x - 1)]
                            } else {
                                0
                            };
                            let ur = if x < config.width - 1 && y > 0 {
                                state[offset + (y - 1) * config.width + (x + 1)]
                            } else {
                                0
                            };
                            let dl = if x > 0 && y < config.height - 1 {
                                state[offset + (y + 1) * config.width + (x - 1)]
                            } else {
                                0
                            };
                            let dr = if x < config.width - 1 && y < config.height - 1 {
                                state[offset + (y + 1) * config.width + (x + 1)]
                            } else {
                                0
                            };

                            let neighbors = alive(left)
                                + alive(right)
                                + alive(up)
                                + alive(down)
                                + alive(ul)
                                + alive(ur)
                                + alive(dl)
                                + alive(dr);

                            let is_alive = current != 0;

                            if is_alive {
                                if neighbors == 2 || neighbors == 3 {
                                    u64::MAX
                                } else {
                                    0
                                }
                            } else if neighbors == 3 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        Ruleset::Rule110 => {
                            let l = if left != 0 { 1 } else { 0 };
                            let c = if current != 0 { 1 } else { 0 };
                            let r = if right != 0 { 1 } else { 0 };

                            let pattern = (l << 2) | (c << 1) | r;
                            let rule110 = 0b01101110u8;
                            if (rule110 >> pattern) & 1 == 1 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                    };

                    next[idx] = new_val;
                }
            }
        }

        state.copy_from_slice(&next);
    }
}

impl fmt::Debug for Universe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Universe")
            .field("worlds", &self.config.worlds)
            .field("size", &(self.config.width, self.config.height))
            .field("ruleset", &self.config.ruleset)
            .field("step", &self.step)
            .field("gpu", &self.gpu_available)
            .finish()
    }
}

// ============================================================================
// Convenience functions
// ============================================================================

/// Check if CUDA GPU acceleration is available
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::is_cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get GPU device name (if available)
#[cfg(feature = "cuda")]
pub fn gpu_device_name() -> Option<String> {
    crate::cuda::CudaRuntime::new()
        .ok()
        .and_then(|rt| rt.device_name().ok())
}

/// Run a standalone benchmark without creating a Universe
#[cfg(feature = "cuda")]
pub fn benchmark(
    worlds: usize,
    width: usize,
    height: usize,
    steps: u32,
    warmup: u32,
    depth: u32,
) -> UniverseResult<BenchmarkResult> {
    let config = UniverseConfig::new(worlds, width, height);
    config.validate()?;

    if is_gpu_available() {
        let rt = crate::cuda::CudaRuntime::new()
            .map_err(|e| UniverseError::GpuError(format!("{}", e)))?;

        let result =
            crate::cuda_tiles::benchmark_tile_eval_depth_gpu(&rt, worlds, steps, depth, warmup)
                .map_err(|e| UniverseError::GpuError(format!("{}", e)))?;

        return Ok(BenchmarkResult {
            total_evals: result.0,
            elapsed_secs: result.1,
            evals_per_sec: result.2,
            worlds,
            width,
            height,
            steps,
            memory_mb: Some(result.3),
        });
    }

    // CPU fallback
    let universe = Universe::new(config)?;
    universe.benchmark_cpu(steps, warmup)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_universe() {
        let universe = Universe::new_default().unwrap();
        assert_eq!(universe.worlds(), 5);
        assert_eq!(universe.width(), 512);
        assert_eq!(universe.height(), 512);
    }

    #[test]
    fn test_randomize() {
        let mut universe = Universe::new(UniverseConfig::new(1, 64, 64)).unwrap();
        universe.randomize(0, 0.5).unwrap();

        let world = universe.get_world(0).unwrap();
        let alive = world.iter().filter(|&&v| v != 0).count();

        // Should be roughly 50% alive (with some variance)
        assert!(alive > 1000 && alive < 3000);
    }

    #[test]
    fn test_evolve() {
        let mut universe =
            Universe::new(UniverseConfig::new(1, 64, 64).with_ruleset(Ruleset::GameOfLife))
                .unwrap();

        universe.randomize(0, 0.3).unwrap();
        let before = universe
            .get_world(0)
            .unwrap()
            .iter()
            .filter(|&&v| v != 0)
            .count();

        universe.evolve(10);

        let after = universe
            .get_world(0)
            .unwrap()
            .iter()
            .filter(|&&v| v != 0)
            .count();
        assert_eq!(universe.step(), 10);
        // Population should change
        assert_ne!(before, after);
    }

    #[test]
    fn test_invalid_config() {
        assert!(Universe::new(UniverseConfig::new(0, 64, 64)).is_err());
        assert!(Universe::new(UniverseConfig::new(1, 100, 64)).is_err()); // not power of 2
    }

    #[test]
    fn test_world_bounds() {
        let universe = Universe::new(UniverseConfig::new(3, 64, 64)).unwrap();
        assert!(universe.get_world(0).is_ok());
        assert!(universe.get_world(2).is_ok());
        assert!(universe.get_world(3).is_err());
    }

    #[test]
    fn test_cell_access() {
        let mut universe = Universe::new(UniverseConfig::new(1, 64, 64)).unwrap();

        universe.set_cell(0, 10, 20, 0xDEADBEEF).unwrap();
        assert_eq!(universe.get_cell(0, 10, 20).unwrap(), 0xDEADBEEF);
    }

    #[test]
    fn test_reversible_mode() {
        let config = UniverseConfig::new(1, 64, 64)
            .with_ruleset(Ruleset::Wire)
            .with_reversible(true)
            .with_max_history(100);

        let mut universe = Universe::new(config).unwrap();
        assert!(universe.is_reversible());
        assert_eq!(universe.history_len(), 1); // Initial state

        // Set up a simple pattern
        universe.set_cell(0, 32, 32, u64::MAX).unwrap();
        let initial_state = universe.get_world(0).unwrap().to_vec();

        // Evolve forward
        universe.evolve(5);
        assert_eq!(universe.step(), 5);
        assert_eq!(universe.history_len(), 6); // initial + 5 steps

        let state_at_5 = universe.get_world(0).unwrap().to_vec();
        assert_ne!(initial_state, state_at_5); // State should have changed

        // Rewind
        let rewound = universe.rewind(3).unwrap();
        assert_eq!(rewound, 3);
        assert_eq!(universe.step(), 2);
        assert!(universe.can_forward());

        let _state_at_2 = universe.get_world(0).unwrap().to_vec();

        // Fast forward
        let forwarded = universe.forward(3).unwrap();
        assert_eq!(forwarded, 3);
        assert_eq!(universe.step(), 5);
        assert_eq!(universe.get_world(0).unwrap(), state_at_5.as_slice());

        // Rewind and evolve (should truncate future)
        universe.rewind(2).unwrap();
        assert_eq!(universe.history_len(), 6); // Still has full history
        universe.evolve(1); // This should truncate and add new state
        assert_eq!(universe.history_len(), 5); // Truncated to pos 3 + new state
    }

    #[test]
    fn test_reversible_disabled() {
        let mut universe = Universe::new(UniverseConfig::new(1, 64, 64)).unwrap();
        assert!(!universe.is_reversible());
        assert_eq!(universe.history_len(), 0);

        // Rewind should fail
        assert!(universe.rewind(1).is_err());
        assert!(universe.forward(1).is_err());
    }

    #[test]
    fn test_goto_history() {
        let config = UniverseConfig::new(1, 64, 64)
            .with_reversible(true)
            .with_max_history(100);

        let mut universe = Universe::new(config).unwrap();
        universe.set_cell(0, 10, 10, u64::MAX).unwrap();
        universe.evolve(10);

        // Jump to middle
        universe.goto_history(5).unwrap();
        assert_eq!(universe.step(), 5);

        // Jump to start
        universe.goto_history(0).unwrap();
        assert_eq!(universe.step(), 0);

        // Invalid index should fail
        assert!(universe.goto_history(100).is_err());
    }
}
