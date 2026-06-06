//! TileUniverse Python Bindings
//!
//! EPIC 76.1: PyO3 bindings for the GPU-accelerated tile universe engine.
//!
//! Exposes:
//! - `Engine`: Main simulation controller
//! - `benchmark()`: Performance measurement
//! - `is_cuda_available()`: GPU detection
//! - `cuda_device_name()`: GPU name query
//! - SPRINT 2.2: Quantum API (`QuantumSimulation`, `QGate`, algorithms)
//! - Sprint 212: V2 CPU API (`V2Cpu`, `V2Trace`) and `synthesize()`

mod v2_cpu;
mod v2_seq;
mod v2_synth;

use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

// SPRINT 2.2: Quantum-classical hybrid imports
use engine::algorithms::grover::{
    grover_circuit as rust_grover_circuit, optimal_iterations as rust_optimal_iterations,
};
use engine::quantum::{QGate as RustQGate, QState};
use engine::simulation::Simulation;
// SPRINT 2.3: Deutsch-Jozsa and Bernstein-Vazirani imports
use engine::algorithms::deutsch_jozsa::{
    DJOracleType as RustDJOracleType, bernstein_vazirani_circuit as rust_bv_circuit,
    deutsch_jozsa_circuit as rust_dj_circuit,
};
use engine::tilemap::{HEIGHT, WIDTH};

// SNN bindings: Quantum-SNN hybrid with Triggered interference mode
use engine::snn::{
    InterferenceMode as RustInterferenceMode, QuantumSNN as RustQuantumSNN,
    QuantumSNNConfig as RustQuantumSNNConfig,
};

// EPIC 97-style Fused SNN bindings (GPU-resident, deterministic)
#[cfg(feature = "cuda")]
use engine::snn::{
    GpuFusedSNN as RustGpuFusedSNN, SynapseCSRData as RustSynapseCSRData,
    generate_columnar_csr as rust_generate_columnar_csr,
    generate_dense_feedforward_csr as rust_generate_dense_feedforward_csr,
    generate_direct_csr as rust_generate_direct_csr,
    generate_layered_csr as rust_generate_layered_csr,
};

// Milestone 5: Batch GPU SNN bindings (Phase 3)
#[cfg(feature = "cuda")]
use engine::snn::{
    GpuBatchSNN as RustGpuBatchSNN, discriminative_2layer_thresholds as rust_discrim_thresholds,
    generate_discriminative_2layer_csr_weighted as rust_gen_discrim_csr_weighted,
    generate_random_csr as rust_gen_random_csr,
};

// Sprint 59.0: Stabilizer-based quantum neural networks
use engine::snn::{
    BridgeGate as RustBridgeGate, ClusterBridge as RustClusterBridge,
    FeedbackMode as RustFeedbackMode, HybridNetworkConfig as RustHybridNetworkConfig,
    HybridQuantumNetwork as RustHybridQuantumNetwork, QuantumNeuralRecorder,
    StabilizerNetworkConfig as RustStabilizerNetworkConfig,
    StabilizerNeuronNetwork as RustStabilizerNeuronNetwork,
};

// Fast W state bindings (Vec-based, scales to billions of qubits)
use engine::tile8::{SparseQuantumGridVec, WStateVerificationVec as RustWStateVerification};

// Fast GHZ state bindings (O(1) sparse - only 2 amplitudes regardless of qubit count)
use engine::tile8::{GhzVerification as RustGhzVerification, SparseQuantumGrid};

// ============================================================================
// Ruleset enum (matches what we expose to Python)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ruleset {
    GameOfLife,
    Rule110,
    Wire,
    Logic,
}

impl Ruleset {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "gol" | "gameoflife" | "game_of_life" | "life" => Some(Ruleset::GameOfLife),
            "rule110" | "r110" | "110" => Some(Ruleset::Rule110),
            "wire" | "wires" => Some(Ruleset::Wire),
            "logic" | "gates" | "tile" => Some(Ruleset::Logic),
            _ => None,
        }
    }
}

// ============================================================================
// BenchmarkResult: Returned from benchmark()
// ============================================================================

/// Result from a benchmark run
#[pyclass]
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Total logic evaluations performed
    #[pyo3(get)]
    pub total_evals: u64,
    /// Elapsed wall-clock time in seconds
    #[pyo3(get)]
    pub elapsed_secs: f64,
    /// Throughput: evaluations per second
    #[pyo3(get)]
    pub evals_per_sec: f64,
    /// Number of parallel worlds
    #[pyo3(get)]
    pub worlds: usize,
    /// World width
    #[pyo3(get)]
    pub width: usize,
    /// World height
    #[pyo3(get)]
    pub height: usize,
    /// Steps executed
    #[pyo3(get)]
    pub steps: u32,
}

#[pymethods]
impl BenchmarkResult {
    fn __repr__(&self) -> String {
        format!(
            "BenchmarkResult(evals_per_sec={:.2e}, worlds={}, size={}x{}, steps={})",
            self.evals_per_sec, self.worlds, self.width, self.height, self.steps
        )
    }

    fn __str__(&self) -> String {
        format!(
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
// Engine: Main simulation interface
// ============================================================================

/// GPU-accelerated parallel universe simulator.
///
/// Args:
///     worlds: Number of parallel worlds to simulate (default: 5)
///     size: Tuple (width, height) for world dimensions (default: (512, 512))
///     ruleset: Simulation ruleset - "gol", "rule110", "wire", or "logic" (default: "logic")
///     seed: Random seed for initialization (default: 42)
///
/// Example:
///     >>> engine = Engine(worlds=5, size=(512, 512), ruleset="gol")
///     >>> engine.evolve(100)
///     >>> world = engine.get_world(0)
#[pyclass]
pub struct Engine {
    /// Number of parallel worlds
    worlds: usize,
    /// World width
    width: usize,
    /// World height
    height: usize,
    /// Active ruleset
    ruleset: Ruleset,
    /// Random seed
    seed: u64,
    /// Current simulation step
    step: u64,
    /// World state buffers (flattened: worlds * width * height)
    /// Each cell is a u64 logic value
    state: Vec<u64>,
    /// Whether CUDA is available (checked at init)
    cuda_available: bool,
    /// History buffer for reversible mode
    history: Vec<Vec<u64>>,
    /// Current position in history
    history_pos: usize,
    /// Whether reversible mode is enabled
    reversible: bool,
    /// Maximum history depth
    max_history: usize,
}

#[pymethods]
impl Engine {
    /// Create a new Engine instance.
    #[new]
    #[pyo3(signature = (worlds=5, size=(512, 512), ruleset="logic", seed=42, reversible=false, max_history=1000))]
    fn new(
        worlds: usize,
        size: (usize, usize),
        ruleset: &str,
        seed: u64,
        reversible: bool,
        max_history: usize,
    ) -> PyResult<Self> {
        let (width, height) = size;

        // Validate inputs
        if worlds == 0 {
            return Err(PyValueError::new_err("worlds must be > 0"));
        }
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("size dimensions must be > 0"));
        }
        if !width.is_power_of_two() || !height.is_power_of_two() {
            return Err(PyValueError::new_err(
                "width and height must be powers of 2 (e.g., 256, 512, 1024)",
            ));
        }

        let rs = Ruleset::from_str(ruleset).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown ruleset '{}'. Valid options: gol, rule110, wire, logic",
                ruleset
            ))
        })?;

        // Initialize state buffer (all zeros initially)
        let total_cells = worlds * width * height;
        let state = vec![0u64; total_cells];

        // Check CUDA availability
        #[cfg(feature = "cuda")]
        let cuda_available = engine::cuda::is_cuda_available();
        #[cfg(not(feature = "cuda"))]
        let cuda_available = false;

        // Initialize history if reversible mode
        let history = if reversible {
            vec![state.clone()]
        } else {
            Vec::new()
        };

        Ok(Engine {
            worlds,
            width,
            height,
            ruleset: rs,
            seed,
            step: 0,
            state,
            cuda_available,
            history,
            history_pos: 0,
            reversible,
            max_history,
        })
    }

    /// Get the number of parallel worlds.
    #[getter]
    fn worlds(&self) -> usize {
        self.worlds
    }

    /// Get the world dimensions as (width, height).
    #[getter]
    fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Get the active ruleset name.
    #[getter]
    fn ruleset(&self) -> &str {
        match self.ruleset {
            Ruleset::GameOfLife => "gol",
            Ruleset::Rule110 => "rule110",
            Ruleset::Wire => "wire",
            Ruleset::Logic => "logic",
        }
    }

    /// Get the current simulation step.
    #[getter]
    fn step(&self) -> u64 {
        self.step
    }

    /// Check if GPU acceleration is available.
    fn has_gpu(&self) -> bool {
        self.cuda_available
    }

    /// Check if reversible mode is enabled.
    #[getter]
    fn is_reversible(&self) -> bool {
        self.reversible
    }

    /// Get the number of steps saved in history.
    #[getter]
    fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Check if we can rewind (go back in time).
    fn can_rewind(&self) -> bool {
        self.reversible && self.history_pos > 0
    }

    /// Check if we can fast-forward (after rewinding).
    fn can_forward(&self) -> bool {
        self.reversible && self.history_pos < self.history.len().saturating_sub(1)
    }

    /// Rewind simulation by the specified number of steps.
    ///
    /// Only available in reversible mode. Returns the number of steps actually rewound.
    #[pyo3(signature = (steps=1))]
    fn rewind(&mut self, steps: u32) -> PyResult<u32> {
        if !self.reversible {
            return Err(PyRuntimeError::new_err("Reversible mode not enabled"));
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
    /// Replays recorded history without re-computing. Returns the number of steps forwarded.
    #[pyo3(signature = (steps=1))]
    fn forward(&mut self, steps: u32) -> PyResult<u32> {
        if !self.reversible {
            return Err(PyRuntimeError::new_err("Reversible mode not enabled"));
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

    /// Jump to a specific point in history (0 = initial state).
    fn goto_history(&mut self, index: usize) -> PyResult<()> {
        if !self.reversible {
            return Err(PyRuntimeError::new_err("Reversible mode not enabled"));
        }
        if index >= self.history.len() {
            return Err(PyValueError::new_err(format!(
                "History index {} out of range (max {})",
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

    /// Set a cell value in a specific world.
    ///
    /// Args:
    ///     world: World index (0 to worlds-1)
    ///     x: X coordinate
    ///     y: Y coordinate
    ///     value: Cell value (u64)
    fn set_cell(&mut self, world: usize, x: usize, y: usize, value: u64) -> PyResult<()> {
        if world >= self.worlds {
            return Err(PyValueError::new_err(format!(
                "world {} out of range (0..{})",
                world, self.worlds
            )));
        }
        if x >= self.width || y >= self.height {
            return Err(PyValueError::new_err(format!(
                "coordinates ({}, {}) out of range ({}x{})",
                x, y, self.width, self.height
            )));
        }

        let idx = world * self.width * self.height + y * self.width + x;
        self.state[idx] = value;
        Ok(())
    }

    /// Get a cell value from a specific world.
    fn get_cell(&self, world: usize, x: usize, y: usize) -> PyResult<u64> {
        if world >= self.worlds {
            return Err(PyValueError::new_err(format!(
                "world {} out of range (0..{})",
                world, self.worlds
            )));
        }
        if x >= self.width || y >= self.height {
            return Err(PyValueError::new_err(format!(
                "coordinates ({}, {}) out of range ({}x{})",
                x, y, self.width, self.height
            )));
        }

        let idx = world * self.width * self.height + y * self.width + x;
        Ok(self.state[idx])
    }

    /// Get a world state as a numpy array.
    ///
    /// Args:
    ///     world: World index (0 to worlds-1)
    ///
    /// Returns:
    ///     numpy.ndarray: 2D array of shape (height, width) with u64 values
    fn get_world<'py>(&self, py: Python<'py>, world: usize) -> PyResult<Bound<'py, PyArray2<u64>>> {
        if world >= self.worlds {
            return Err(PyValueError::new_err(format!(
                "world {} out of range (0..{})",
                world, self.worlds
            )));
        }

        let start = world * self.width * self.height;
        let end = start + self.width * self.height;
        let world_data: Vec<u64> = self.state[start..end].to_vec();

        // Reshape to 2D (height x width) for numpy convention
        let arr = ndarray::Array2::from_shape_vec((self.height, self.width), world_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Array reshape failed: {}", e)))?;

        Ok(arr.into_pyarray(py))
    }

    /// Set a world state from a numpy array.
    ///
    /// Args:
    ///     world: World index (0 to worlds-1)
    ///     data: numpy.ndarray of shape (height, width) with u64 values
    fn set_world(&mut self, world: usize, data: &Bound<'_, PyArray2<u64>>) -> PyResult<()> {
        if world >= self.worlds {
            return Err(PyValueError::new_err(format!(
                "world {} out of range (0..{})",
                world, self.worlds
            )));
        }

        let arr = unsafe { data.as_array() };
        let shape = arr.shape();

        if shape[0] != self.height || shape[1] != self.width {
            return Err(PyValueError::new_err(format!(
                "Array shape {:?} doesn't match world size ({}, {})",
                shape, self.height, self.width
            )));
        }

        let start = world * self.width * self.height;
        for (i, &val) in arr.iter().enumerate() {
            self.state[start + i] = val;
        }

        Ok(())
    }

    /// Evolve all worlds forward by the specified number of steps.
    ///
    /// Uses GPU acceleration when available, falls back to CPU otherwise.
    /// In reversible mode, each step is saved to history.
    ///
    /// Args:
    ///     steps: Number of simulation steps to run (default: 1)
    ///     depth: Steps per GPU kernel launch (default: 50, GPU only)
    #[pyo3(signature = (steps=1, depth=50))]
    #[allow(unused_variables)]
    fn evolve(&mut self, steps: u32, depth: u32) -> PyResult<()> {
        // If we're in the middle of history (after rewind), truncate future states
        if self.reversible && self.history_pos < self.history.len().saturating_sub(1) {
            self.history.truncate(self.history_pos + 1);
        }

        for _ in 0..steps {
            self.step_cpu();

            // Record history if reversible mode enabled
            if self.reversible {
                if self.history.len() >= self.max_history {
                    self.history.remove(0);
                }
                self.history.push(self.state.clone());
                self.history_pos = self.history.len() - 1;
            }
        }
        self.step += steps as u64;
        Ok(())
    }

    /// Run a benchmark and return performance metrics.
    ///
    /// Args:
    ///     steps: Number of steps to benchmark (default: 1000)
    ///     warmup: Warmup steps before timing (default: 100)
    ///     depth: Steps per kernel launch (default: 50)
    #[pyo3(signature = (steps=1000, warmup=100, depth=50))]
    fn benchmark(&self, steps: u32, warmup: u32, depth: u32) -> PyResult<BenchmarkResult> {
        #[cfg(feature = "cuda")]
        {
            if self.cuda_available {
                // Create a fresh CUDA runtime for the benchmark
                let rt = engine::cuda::CudaRuntime::new()
                    .map_err(|e| PyRuntimeError::new_err(format!("CUDA init failed: {}", e)))?;

                // Use the engine's GPU benchmark
                // Returns (total_evals, elapsed, evals_per_sec, memory_mb)
                let result = engine::cuda::benchmark_tile_eval_depth_gpu(
                    &rt,
                    self.worlds,
                    steps,
                    depth,
                    warmup,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("GPU benchmark failed: {}", e)))?;

                return Ok(BenchmarkResult {
                    total_evals: result.0,
                    elapsed_secs: result.1,
                    evals_per_sec: result.2,
                    worlds: self.worlds,
                    width: self.width,
                    height: self.height,
                    steps,
                });
            }
        }

        // CPU fallback benchmark
        use std::time::Instant;

        let tiles_per_world = self.width * self.height;
        let total_tiles = tiles_per_world * self.worlds;

        // Warmup
        let mut state = self.state.clone();
        for _ in 0..warmup {
            Self::step_state_cpu(
                &mut state,
                self.worlds,
                self.width,
                self.height,
                self.ruleset,
            );
        }

        // Timed run
        let start = Instant::now();
        for _ in 0..steps {
            Self::step_state_cpu(
                &mut state,
                self.worlds,
                self.width,
                self.height,
                self.ruleset,
            );
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
            worlds: self.worlds,
            width: self.width,
            height: self.height,
            steps,
        })
    }

    /// Initialize a world with a random pattern.
    ///
    /// Args:
    ///     world: World index (0 to worlds-1)
    ///     density: Probability of cell being alive (0.0 to 1.0, default: 0.3)
    #[pyo3(signature = (world, density=0.3))]
    fn randomize(&mut self, world: usize, density: f64) -> PyResult<()> {
        if world >= self.worlds {
            return Err(PyValueError::new_err(format!(
                "world {} out of range (0..{})",
                world, self.worlds
            )));
        }

        // Simple PRNG using seed + world + position
        let mut rng_state = self.seed.wrapping_add(world as u64);

        let start = world * self.width * self.height;
        for i in 0..(self.width * self.height) {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let r = (rng_state as f64) / (u64::MAX as f64);
            self.state[start + i] = if r < density { u64::MAX } else { 0 };
        }

        Ok(())
    }

    /// Reset all worlds to zero state.
    fn reset(&mut self) {
        self.state.fill(0);
        self.step = 0;
        if self.reversible {
            self.history.clear();
            self.history.push(self.state.clone());
            self.history_pos = 0;
        }
    }

    fn __repr__(&self) -> String {
        if self.reversible {
            format!(
                "Engine(worlds={}, size=({}, {}), ruleset='{}', step={}, gpu={}, reversible=true, history={})",
                self.worlds,
                self.width,
                self.height,
                self.ruleset(),
                self.step,
                self.has_gpu(),
                self.history.len()
            )
        } else {
            format!(
                "Engine(worlds={}, size=({}, {}), ruleset='{}', step={}, gpu={})",
                self.worlds,
                self.width,
                self.height,
                self.ruleset(),
                self.step,
                self.has_gpu()
            )
        }
    }
}

impl Engine {
    /// CPU single-step evolution (internal)
    fn step_cpu(&mut self) {
        let mut new_state = self.state.clone();
        Self::step_state_cpu(
            &mut new_state,
            self.worlds,
            self.width,
            self.height,
            self.ruleset,
        );
        self.state = new_state;
    }

    /// Static CPU step function (for benchmarking without &mut self)
    fn step_state_cpu(
        state: &mut [u64],
        worlds: usize,
        width: usize,
        height: usize,
        ruleset: Ruleset,
    ) {
        let tiles_per_world = width * height;
        let mut next = state.to_vec();

        for world in 0..worlds {
            let offset = world * tiles_per_world;

            for y in 0..height {
                for x in 0..width {
                    let idx = offset + y * width + x;

                    // Get neighbors (with wrapping for Game of Life style)
                    let left = if x > 0 {
                        state[offset + y * width + (x - 1)]
                    } else {
                        0
                    };
                    let right = if x < width - 1 {
                        state[offset + y * width + (x + 1)]
                    } else {
                        0
                    };
                    let up = if y > 0 {
                        state[offset + (y - 1) * width + x]
                    } else {
                        0
                    };
                    let down = if y < height - 1 {
                        state[offset + (y + 1) * width + x]
                    } else {
                        0
                    };
                    let current = state[idx];

                    let new_val = match ruleset {
                        Ruleset::Wire => {
                            // Wire: OR of all neighbors
                            left | right | up | down
                        }
                        Ruleset::Logic => {
                            // Logic: OR of all neighbors (like wire for now)
                            left | right | up | down
                        }
                        Ruleset::GameOfLife => {
                            // Game of Life: count live neighbors
                            // Using binary: alive = u64::MAX, dead = 0
                            let alive = |v: u64| if v != 0 { 1 } else { 0 };

                            // Get all 8 neighbors
                            let ul = if x > 0 && y > 0 {
                                state[offset + (y - 1) * width + (x - 1)]
                            } else {
                                0
                            };
                            let ur = if x < width - 1 && y > 0 {
                                state[offset + (y - 1) * width + (x + 1)]
                            } else {
                                0
                            };
                            let dl = if x > 0 && y < height - 1 {
                                state[offset + (y + 1) * width + (x - 1)]
                            } else {
                                0
                            };
                            let dr = if x < width - 1 && y < height - 1 {
                                state[offset + (y + 1) * width + (x + 1)]
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

                            // Conway's rules
                            if is_alive {
                                if neighbors == 2 || neighbors == 3 {
                                    u64::MAX
                                } else {
                                    0
                                }
                            } else {
                                if neighbors == 3 { u64::MAX } else { 0 }
                            }
                        }
                        Ruleset::Rule110 => {
                            // Rule 110: 1D cellular automaton
                            // For 2D grid, apply horizontally per row
                            let l = if left != 0 { 1 } else { 0 };
                            let c = if current != 0 { 1 } else { 0 };
                            let r = if right != 0 { 1 } else { 0 };

                            let pattern = (l << 2) | (c << 1) | r;

                            // Rule 110: 01101110 in binary
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

// ============================================================================
// Module-level functions
// ============================================================================

/// Check if CUDA GPU acceleration is available.
#[pyfunction]
fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        engine::cuda::is_cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get the name of the CUDA device.
#[pyfunction]
fn cuda_device_name() -> String {
    #[cfg(feature = "cuda")]
    {
        match engine::cuda::CudaRuntime::new() {
            Ok(rt) => rt
                .device_name()
                .unwrap_or_else(|_| "Unknown GPU".to_string()),
            Err(_) => "No GPU available".to_string(),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        "CUDA not compiled".to_string()
    }
}

/// Run a standalone benchmark without creating an Engine.
///
/// Args:
///     worlds: Number of parallel worlds (default: 5)
///     width: World width (default: 512)
///     height: World height (default: 512)
///     steps: Benchmark steps (default: 1000)
///     warmup: Warmup steps (default: 100)
///     depth: Steps per GPU kernel (default: 50)
///
/// Returns:
///     BenchmarkResult with performance metrics
#[pyfunction]
#[pyo3(signature = (worlds=5, width=512, height=512, steps=1000, warmup=100, depth=50))]
fn benchmark(
    worlds: usize,
    width: usize,
    height: usize,
    steps: u32,
    warmup: u32,
    depth: u32,
) -> PyResult<BenchmarkResult> {
    #[cfg(feature = "cuda")]
    {
        if engine::cuda::is_cuda_available() {
            let rt = engine::cuda::CudaRuntime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("CUDA init failed: {}", e)))?;

            // Returns (total_evals, elapsed, evals_per_sec, memory_mb)
            let result =
                engine::cuda::benchmark_tile_eval_depth_gpu(&rt, worlds, steps, depth, warmup)
                    .map_err(|e| PyRuntimeError::new_err(format!("GPU benchmark failed: {}", e)))?;

            return Ok(BenchmarkResult {
                total_evals: result.0,
                elapsed_secs: result.1,
                evals_per_sec: result.2,
                worlds,
                width,
                height,
                steps,
            });
        }
    }

    // CPU fallback
    let engine = Engine::new(worlds, (width, height), "logic", 42, false, 1000)?;
    engine.benchmark(steps, warmup, depth)
}

// ============================================================================
// Python module definition
// ============================================================================

/// Load an Engine from a YAML or JSON config file.
///
/// Args:
///     path: Path to config file (.yaml, .yml, or .json)
///
/// Returns:
///     Engine configured according to the file
///
/// Example config (YAML):
///     ```yaml
///     name: "my-scenario"
///     worlds: 5
///     width: 512
///     height: 512
///     ruleset: gol
///     seed: 42
///     initial_patterns:
///       - pattern: random
///         world: 0
///         density: 0.3
///     ```
#[pyfunction]
fn load_config(path: &str) -> PyResult<Engine> {
    // Simple config parsing without full serde dependency in Python crate
    let content = std::fs::read_to_string(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read config: {}", e)))?;

    // Detect format and parse minimally
    let _ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase());

    // Parse key fields with simple regex-like matching
    let worlds = parse_int_field(&content, "worlds").unwrap_or(5);
    let width = parse_int_field(&content, "width").unwrap_or(512);
    let height = parse_int_field(&content, "height").unwrap_or(512);
    let seed = parse_int_field(&content, "seed").unwrap_or(42) as u64;
    let ruleset = parse_string_field(&content, "ruleset").unwrap_or_else(|| "logic".to_string());
    let reversible = parse_bool_field(&content, "reversible").unwrap_or(false);
    let max_history = parse_int_field(&content, "max_history").unwrap_or(1000);

    Engine::new(
        worlds,
        (width, height),
        &ruleset,
        seed,
        reversible,
        max_history,
    )
}

// Simple field parsers for config files
fn parse_int_field(content: &str, field: &str) -> Option<usize> {
    for line in content.lines() {
        let line = line.trim();
        // Handle YAML style: "field: value" or JSON style: "\"field\": value"
        if line.starts_with(field) || line.starts_with(&format!("\"{}\"", field)) {
            if let Some(colon_pos) = line.find(':') {
                let value_part = line[colon_pos + 1..]
                    .trim()
                    .trim_matches(',')
                    .trim_matches('"');
                if let Ok(v) = value_part.parse() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn parse_string_field(content: &str, field: &str) -> Option<String> {
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with(field) || line.starts_with(&format!("\"{}\"", field)) {
            if let Some(colon_pos) = line.find(':') {
                let value_part = line[colon_pos + 1..]
                    .trim()
                    .trim_matches(',')
                    .trim_matches('"');
                if !value_part.is_empty() {
                    return Some(value_part.to_string());
                }
            }
        }
    }
    None
}

fn parse_bool_field(content: &str, field: &str) -> Option<bool> {
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with(field) || line.starts_with(&format!("\"{}\"", field)) {
            if let Some(colon_pos) = line.find(':') {
                let value_part = line[colon_pos + 1..]
                    .trim()
                    .trim_matches(',')
                    .trim_matches('"')
                    .to_lowercase();
                match value_part.as_str() {
                    "true" | "yes" | "1" => return Some(true),
                    "false" | "no" | "0" => return Some(false),
                    _ => {}
                }
            }
        }
    }
    None
}

// =============================================================================
// SPRINT 2.2: Quantum API Classes
// =============================================================================

/// Quantum gate for use in quantum circuits.
///
/// Create gates using static methods:
///     QGate.h(0)        # Hadamard on qubit 0
///     QGate.x(1)        # Pauli-X on qubit 1
///     QGate.cnot(0, 1)  # CNOT with control=0, target=1
///     QGate.measure(0)  # Measure qubit 0
#[pyclass(name = "QGate")]
#[derive(Clone, Debug)]
pub struct PyQGate {
    inner: RustQGate,
}

#[pymethods]
impl PyQGate {
    /// Hadamard gate: creates superposition
    /// H|0⟩ = (|0⟩ + |1⟩)/√2
    #[staticmethod]
    fn h(qubit: u8) -> Self {
        PyQGate {
            inner: RustQGate::H(qubit),
        }
    }

    /// Pauli-X gate: bit flip
    /// X|0⟩ = |1⟩, X|1⟩ = |0⟩
    #[staticmethod]
    fn x(qubit: u8) -> Self {
        PyQGate {
            inner: RustQGate::X(qubit),
        }
    }

    /// Pauli-Z gate: phase flip
    /// Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    #[staticmethod]
    fn z(qubit: u8) -> Self {
        PyQGate {
            inner: RustQGate::Z(qubit),
        }
    }

    /// Phase gate: arbitrary phase rotation
    /// P(θ)|1⟩ = e^(iθ)|1⟩
    #[staticmethod]
    fn phase(qubit: u8, theta: f32) -> Self {
        PyQGate {
            inner: RustQGate::Phase(qubit, theta),
        }
    }

    /// CNOT gate: controlled NOT
    /// Flips target if control is |1⟩
    #[staticmethod]
    fn cnot(control: u8, target: u8) -> Self {
        PyQGate {
            inner: RustQGate::CNot(control, target),
        }
    }

    /// Controlled-Z gate: controlled phase flip
    /// Applies -1 phase when both qubits are |1⟩
    #[staticmethod]
    fn cz(qubit0: u8, qubit1: u8) -> Self {
        PyQGate {
            inner: RustQGate::CZ(qubit0, qubit1),
        }
    }

    /// Toffoli gate (CCX): controlled-controlled NOT
    /// Flips target when both controls are |1⟩
    #[staticmethod]
    fn toffoli(ctrl0: u8, ctrl1: u8, target: u8) -> Self {
        PyQGate {
            inner: RustQGate::Toffoli(ctrl0, ctrl1, target),
        }
    }

    /// Alias for toffoli()
    #[staticmethod]
    fn ccx(ctrl0: u8, ctrl1: u8, target: u8) -> Self {
        PyQGate {
            inner: RustQGate::Toffoli(ctrl0, ctrl1, target),
        }
    }

    /// CCZ gate: controlled-controlled Z
    /// Applies -1 phase when all three qubits are |1⟩
    #[staticmethod]
    fn ccz(qubit0: u8, qubit1: u8, qubit2: u8) -> Self {
        PyQGate {
            inner: RustQGate::CCZ(qubit0, qubit1, qubit2),
        }
    }

    /// Measurement gate: collapses qubit to |0⟩ or |1⟩
    /// Result is stored in tile's logic output
    #[staticmethod]
    fn measure(qubit: u8) -> Self {
        PyQGate {
            inner: RustQGate::Measure(qubit),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        match &self.inner {
            RustQGate::H(q) => format!("H({})", q),
            RustQGate::X(q) => format!("X({})", q),
            RustQGate::Y(q) => format!("Y({})", q),
            RustQGate::Z(q) => format!("Z({})", q),
            RustQGate::Phase(q, theta) => format!("Phase({}, {:.3})", q, theta),
            RustQGate::CNot(c, t) => format!("CNOT({}, {})", c, t),
            RustQGate::Swap(a, b) => format!("Swap({}, {})", a, b),
            RustQGate::Measure(q) => format!("Measure({})", q),
            RustQGate::CZ(a, b) => format!("CZ({}, {})", a, b),
            RustQGate::Toffoli(c0, c1, t) => format!("Toffoli({}, {}, {})", c0, c1, t),
            RustQGate::CCZ(a, b, c) => format!("CCZ({}, {}, {})", a, b, c),
            RustQGate::CPhase(c, t, theta) => format!("CPhase({}, {}, {:.3})", c, t, theta),
            RustQGate::CRz(c, t, theta) => format!("CRz({}, {}, {:.3})", c, t, theta),
            RustQGate::Rx(q, theta) => format!("Rx({}, {:.3})", q, theta),
            RustQGate::Ry(q, theta) => format!("Ry({}, {:.3})", q, theta),
            RustQGate::Rz(q, theta) => format!("Rz({}, {:.3})", q, theta),
            RustQGate::U3(q, theta, phi, lambda) => {
                format!("U3({}, {:.3}, {:.3}, {:.3})", q, theta, phi, lambda)
            }
            RustQGate::T(q) => format!("T({})", q),
            RustQGate::Tdg(q) => format!("Tdg({})", q),
        }
    }
}

impl PyQGate {
    /// Convert to inner Rust gate (for internal use)
    pub fn to_rust(&self) -> RustQGate {
        self.inner.clone()
    }
}

/// Quantum-classical hybrid simulation.
///
/// This class wraps the core Simulation engine and provides
/// access to quantum tile registration and execution.
///
/// Example:
///     >>> from tileuniverse import QuantumSimulation, QGate
///     >>> sim = QuantumSimulation()
///     >>> sim.register_qdemo(50, 50, 2, [
///     ...     QGate.h(0),
///     ...     QGate.cnot(0, 1),
///     ...     QGate.measure(0),
///     ...     QGate.measure(1),
///     ... ], seed=42)
///     >>> sim.tick_n(4)
///     >>> result = sim.get_logic(50, 50)
///     >>> print(f"Measured: {result:02b}")
#[pyclass(name = "QuantumSimulation", unsendable)]
pub struct PyQuantumSimulation {
    inner: Simulation,
}

#[pymethods]
impl PyQuantumSimulation {
    /// Create a new quantum-classical simulation.
    ///
    /// The simulation uses the standard tilemap size (512x512 by default).
    #[new]
    fn new() -> Self {
        PyQuantumSimulation {
            inner: Simulation::new(),
        }
    }

    /// Register a quantum demo tile at the given position.
    ///
    /// Args:
    ///     x: X coordinate (0 to width-1)
    ///     y: Y coordinate (0 to height-1)
    ///     n_qubits: Number of qubits (1-12)
    ///     program: List of QGate objects defining the circuit
    ///     seed: Random seed for measurement outcomes
    ///
    /// Example:
    ///     >>> sim.register_qdemo(10, 10, 3, [
    ///     ...     QGate.h(0),
    ///     ...     QGate.h(1),
    ///     ...     QGate.h(2),
    ///     ...     QGate.measure(0),
    ///     ...     QGate.measure(1),
    ///     ...     QGate.measure(2),
    ///     ... ], seed=12345)
    #[pyo3(signature = (x, y, n_qubits, program, seed=0))]
    fn register_qdemo(
        &mut self,
        x: usize,
        y: usize,
        n_qubits: u8,
        program: Vec<PyQGate>,
        seed: u64,
    ) -> PyResult<()> {
        // Validate inputs
        if n_qubits == 0 || n_qubits > 12 {
            return Err(PyValueError::new_err("n_qubits must be 1-12"));
        }

        if x >= WIDTH || y >= HEIGHT {
            return Err(PyValueError::new_err(format!(
                "Position ({}, {}) out of bounds (max: {}, {})",
                x,
                y,
                WIDTH - 1,
                HEIGHT - 1
            )));
        }

        if program.is_empty() {
            return Err(PyValueError::new_err("Quantum program cannot be empty"));
        }

        // Convert Python gates to Rust gates
        let rust_program: Vec<RustQGate> = program.iter().map(|g| g.to_rust()).collect();

        // Create quantum state and register
        let state = QState::new_zero(n_qubits);
        self.inner
            .register_qdemo_tile(x, y, state, rust_program, seed);

        Ok(())
    }

    /// Advance the simulation by one tick.
    ///
    /// Each tick:
    /// 1. Toggles the global clock
    /// 2. Evaluates all dirty tiles until stable
    /// 3. Executes one gate per quantum tile
    fn tick(&mut self) {
        self.inner.tick();
    }

    /// Advance the simulation by n ticks.
    ///
    /// Args:
    ///     n: Number of ticks to execute
    fn tick_n(&mut self, n: usize) {
        for _ in 0..n {
            self.inner.tick();
        }
    }

    /// Get the logic value at a tile position.
    ///
    /// For quantum tiles, this returns the encoded measurement results:
    /// - Bit 0 = qubit 0 measurement (0 or 1)
    /// - Bit 1 = qubit 1 measurement (0 or 1)
    /// - etc.
    ///
    /// Args:
    ///     x: X coordinate
    ///     y: Y coordinate
    ///
    /// Returns:
    ///     64-bit logic value at the tile
    fn get_logic(&self, x: usize, y: usize) -> PyResult<u64> {
        if x >= WIDTH || y >= HEIGHT {
            return Err(PyValueError::new_err(format!(
                "Position ({}, {}) out of bounds",
                x, y
            )));
        }

        match self.inner.tilemap.get_tile(x, y) {
            Some(tile) => Ok(tile.logic.load(std::sync::atomic::Ordering::Relaxed)),
            None => Err(PyValueError::new_err(format!("No tile at ({}, {})", x, y))),
        }
    }

    /// Set the logic value at a tile position.
    ///
    /// Args:
    ///     x: X coordinate
    ///     y: Y coordinate
    ///     value: 64-bit logic value to set
    fn set_logic(&mut self, x: usize, y: usize, value: u64) -> PyResult<()> {
        if x >= WIDTH || y >= HEIGHT {
            return Err(PyValueError::new_err(format!(
                "Position ({}, {}) out of bounds",
                x, y
            )));
        }

        self.inner.set_logic_value(x, y, value);
        Ok(())
    }

    /// Set the tile type at a position.
    ///
    /// Args:
    ///     x: X coordinate
    ///     y: Y coordinate
    ///     tile_type: Tile type string ("wire", "and", "or", "xor", "not", etc.)
    fn set_tile(&mut self, x: usize, y: usize, tile_type: &str) -> PyResult<()> {
        if x >= WIDTH || y >= HEIGHT {
            return Err(PyValueError::new_err(format!(
                "Position ({}, {}) out of bounds",
                x, y
            )));
        }

        let tt = match tile_type.to_lowercase().as_str() {
            "wire" => engine::tile_meta::TileType::Wire,
            "and" => engine::tile_meta::TileType::And,
            "or" => engine::tile_meta::TileType::Or,
            "xor" => engine::tile_meta::TileType::Xor,
            "not" => engine::tile_meta::TileType::Not,
            "clock" | "clockglobal" => engine::tile_meta::TileType::ClockGlobal,
            "latch" => engine::tile_meta::TileType::Latch,
            "register" | "register8" => engine::tile_meta::TileType::Register8,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown tile type '{}'. Valid: wire, and, or, xor, not, clock, latch, register",
                    tile_type
                )));
            }
        };

        self.inner.set_tile(x, y, tt);
        Ok(())
    }

    /// Get the number of registered quantum tiles.
    #[getter]
    fn quantum_tile_count(&self) -> usize {
        self.inner.quantum_tile_count()
    }

    /// Get world dimensions.
    #[getter]
    fn size(&self) -> (usize, usize) {
        (WIDTH, HEIGHT)
    }

    /// Get the current global clock state.
    #[getter]
    fn clock(&self) -> bool {
        self.inner.global_clock
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumSimulation(size={}x{}, quantum_tiles={})",
            WIDTH,
            HEIGHT,
            self.quantum_tile_count()
        )
    }
}

// =============================================================================
// SPRINT 2.2: Grover Algorithm Helpers
// =============================================================================

/// Build a Grover search circuit.
///
/// Args:
///     n_qubits: Number of qubits (search space = 2^n_qubits)
///     marked_state: The state to find (as integer, e.g., 5 for |101⟩)
///     iterations: Number of Grover iterations (use optimal_iterations() to calculate)
///
/// Returns:
///     List of QGate objects implementing Grover's algorithm
///
/// Example:
///     >>> from tileuniverse import grover_circuit, optimal_iterations
///     >>> n = 3
///     >>> circuit = grover_circuit(n, 5, optimal_iterations(n))
///     >>> len(circuit)
///     42
#[pyfunction]
#[pyo3(signature = (n_qubits, marked_state, iterations=None))]
fn grover_circuit(
    n_qubits: u8,
    marked_state: u64,
    iterations: Option<usize>,
) -> PyResult<Vec<PyQGate>> {
    if n_qubits == 0 || n_qubits > 12 {
        return Err(PyValueError::new_err("n_qubits must be 1-12"));
    }

    let max_state = 1u64 << n_qubits;
    if marked_state >= max_state {
        return Err(PyValueError::new_err(format!(
            "marked_state {} out of range for {} qubits (max: {})",
            marked_state,
            n_qubits,
            max_state - 1
        )));
    }

    let iters = iterations.unwrap_or_else(|| rust_optimal_iterations(n_qubits));

    let rust_circuit = rust_grover_circuit(n_qubits, marked_state, iters);
    let py_circuit: Vec<PyQGate> = rust_circuit
        .into_iter()
        .map(|g| PyQGate { inner: g })
        .collect();

    Ok(py_circuit)
}

/// Calculate optimal number of Grover iterations.
///
/// For N = 2^n_qubits items with 1 marked state:
/// optimal iterations ≈ π/4 × √N
///
/// Args:
///     n_qubits: Number of qubits
///
/// Returns:
///     Optimal iteration count
///
/// Example:
///     >>> optimal_iterations(3)  # 8 items
///     2
///     >>> optimal_iterations(4)  # 16 items
///     3
#[pyfunction]
fn optimal_iterations(n_qubits: u8) -> PyResult<usize> {
    if n_qubits == 0 || n_qubits > 12 {
        return Err(PyValueError::new_err("n_qubits must be 1-12"));
    }

    Ok(rust_optimal_iterations(n_qubits))
}

// =============================================================================
// SPRINT 2.3: Deutsch-Jozsa and Bernstein-Vazirani
// =============================================================================

/// Oracle type for Deutsch-Jozsa algorithm.
///
/// Use the static methods to create oracle types:
///     DJOracle.constant_zero()  # f(x) = 0 for all x
///     DJOracle.constant_one()   # f(x) = 1 for all x
///     DJOracle.balanced(5)      # f(x) based on pattern 5
#[pyclass(name = "DJOracle")]
#[derive(Clone, Debug)]
pub struct PyDJOracle {
    inner: RustDJOracleType,
}

#[pymethods]
impl PyDJOracle {
    /// Constant oracle that always returns 0.
    #[staticmethod]
    fn constant_zero() -> Self {
        PyDJOracle {
            inner: RustDJOracleType::ConstantZero,
        }
    }

    /// Constant oracle that always returns 1.
    #[staticmethod]
    fn constant_one() -> Self {
        PyDJOracle {
            inner: RustDJOracleType::ConstantOne,
        }
    }

    /// Balanced oracle based on a bit pattern.
    ///
    /// The oracle computes f(x) = 1 if x·pattern is odd, else 0.
    /// This gives a balanced function (half inputs → 0, half → 1).
    #[staticmethod]
    fn balanced(pattern: u64) -> Self {
        PyDJOracle {
            inner: RustDJOracleType::Balanced(pattern),
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            RustDJOracleType::ConstantZero => "DJOracle.constant_zero()".to_string(),
            RustDJOracleType::ConstantOne => "DJOracle.constant_one()".to_string(),
            RustDJOracleType::Balanced(p) => format!("DJOracle.balanced({})", p),
        }
    }

    fn __str__(&self) -> String {
        match self.inner {
            RustDJOracleType::ConstantZero => "constant-zero".to_string(),
            RustDJOracleType::ConstantOne => "constant-one".to_string(),
            RustDJOracleType::Balanced(p) => format!("balanced({})", p),
        }
    }

    /// Check if this oracle is constant.
    fn is_constant(&self) -> bool {
        matches!(
            self.inner,
            RustDJOracleType::ConstantZero | RustDJOracleType::ConstantOne
        )
    }

    /// Check if this oracle is balanced.
    fn is_balanced(&self) -> bool {
        matches!(self.inner, RustDJOracleType::Balanced(_))
    }
}

/// Build a Deutsch-Jozsa circuit.
///
/// The Deutsch-Jozsa algorithm determines whether a function is constant
/// (always 0 or always 1) or balanced (half 0, half 1) with a single query.
///
/// Classical complexity: O(2^(n-1) + 1) queries worst case
/// Quantum complexity: O(1) query — exponential speedup!
///
/// Args:
///     n_qubits: Number of input qubits
///     oracle: Oracle type (use DJOracle.constant_zero(), etc.)
///
/// Returns:
///     List of QGate objects. After measurement:
///     - |0...0⟩ indicates constant oracle
///     - Anything else indicates balanced oracle
///
/// Example:
///     >>> circuit = deutsch_jozsa_circuit(3, DJOracle.balanced(5))
///     >>> # Run on QDemo tile, measure, check if result is 0
#[pyfunction]
fn deutsch_jozsa_circuit(n_qubits: u8, oracle: &PyDJOracle) -> PyResult<Vec<PyQGate>> {
    if n_qubits == 0 || n_qubits > 10 {
        return Err(PyValueError::new_err("n_qubits must be 1-10"));
    }

    let rust_circuit = rust_dj_circuit(n_qubits, oracle.inner);
    let py_circuit: Vec<PyQGate> = rust_circuit
        .into_iter()
        .map(|g| PyQGate { inner: g })
        .collect();

    Ok(py_circuit)
}

/// Build a Bernstein-Vazirani circuit.
///
/// The Bernstein-Vazirani algorithm finds a hidden string s where
/// the oracle computes f(x) = s·x mod 2 (bitwise dot product).
///
/// Classical complexity: O(n) queries (one per bit)
/// Quantum complexity: O(1) query — linear to constant speedup!
///
/// Args:
///     n_qubits: Number of bits in the hidden string
///     hidden_string: The secret string to encode in the oracle
///
/// Returns:
///     List of QGate objects. After measurement, result equals hidden_string.
///
/// Example:
///     >>> circuit = bernstein_vazirani_circuit(4, 0b1010)
///     >>> # Run on QDemo tile, measure, should get 0b1010 = 10
#[pyfunction]
fn bernstein_vazirani_circuit(n_qubits: u8, hidden_string: u64) -> PyResult<Vec<PyQGate>> {
    if n_qubits == 0 || n_qubits > 10 {
        return Err(PyValueError::new_err("n_qubits must be 1-10"));
    }

    let max_value = 1u64 << n_qubits;
    if hidden_string >= max_value {
        return Err(PyValueError::new_err(format!(
            "hidden_string {} out of range for {} qubits (max: {})",
            hidden_string,
            n_qubits,
            max_value - 1
        )));
    }

    let rust_circuit = rust_bv_circuit(n_qubits, hidden_string);
    let py_circuit: Vec<PyQGate> = rust_circuit
        .into_iter()
        .map(|g| PyQGate { inner: g })
        .collect();

    Ok(py_circuit)
}

// =============================================================================
// SNN BINDINGS: Quantum-SNN Hybrid for Reinforcement Learning
// =============================================================================

/// Interference mode for Quantum-SNN hybrid decision making.
///
/// Controls how quantum interference is applied to SNN spike patterns:
/// - none: Pure spike-proportional sampling (no quantum effects)
/// - soft_mix: Gentle blending between adjacent actions (exploration)
/// - grover: Amplitude amplification of dominant action (exploitation)
/// - anti_grover: Amplify least-preferred actions (curiosity/exploration)
/// - adaptive: Gradual transition from exploration to exploitation
/// - triggered: Explore until high reward, then exploit (BEST for sparse rewards)
/// - hadamard_cz: Original H+CZ circuit (not recommended)
#[pyclass(name = "InterferenceMode")]
#[derive(Clone, Debug)]
pub struct PyInterferenceMode {
    inner: RustInterferenceMode,
}

#[pymethods]
impl PyInterferenceMode {
    /// No interference - pure spike-proportional sampling
    #[staticmethod]
    fn none() -> Self {
        PyInterferenceMode {
            inner: RustInterferenceMode::None,
        }
    }

    /// Soft mixing - gentle blending for exploration while preserving spike info
    #[staticmethod]
    fn soft_mix() -> Self {
        PyInterferenceMode {
            inner: RustInterferenceMode::SoftMix,
        }
    }

    /// Grover-like amplitude amplification - enhances dominant action
    #[staticmethod]
    fn grover() -> Self {
        PyInterferenceMode {
            inner: RustInterferenceMode::Grover,
        }
    }

    /// Anti-Grover curiosity exploration - amplifies LEAST preferred actions
    /// Good for escaping local optima and exploring unexplored options
    #[staticmethod]
    fn anti_grover() -> Self {
        PyInterferenceMode {
            inner: RustInterferenceMode::AntiGrover,
        }
    }

    /// Adaptive: starts with AntiGrover (exploration), decays to Grover (exploitation)
    #[staticmethod]
    fn adaptive() -> Self {
        PyInterferenceMode {
            inner: RustInterferenceMode::Adaptive,
        }
    }

    /// Triggered: explore with AntiGrover until high reward, then switch to exploitation.
    /// BEST MODE for sparse reward problems (+39% over pure quantum, +69% over epsilon-greedy)
    #[staticmethod]
    fn triggered() -> Self {
        PyInterferenceMode {
            inner: RustInterferenceMode::Triggered,
        }
    }

    /// Original H+CZ circuit (not recommended - creates destructive interference)
    #[staticmethod]
    fn hadamard_cz() -> Self {
        PyInterferenceMode {
            inner: RustInterferenceMode::HadamardCZ,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            RustInterferenceMode::None => "InterferenceMode.none()".to_string(),
            RustInterferenceMode::SoftMix => "InterferenceMode.soft_mix()".to_string(),
            RustInterferenceMode::Grover => "InterferenceMode.grover()".to_string(),
            RustInterferenceMode::AntiGrover => "InterferenceMode.anti_grover()".to_string(),
            RustInterferenceMode::Adaptive => "InterferenceMode.adaptive()".to_string(),
            RustInterferenceMode::Triggered => "InterferenceMode.triggered()".to_string(),
            RustInterferenceMode::HadamardCZ => "InterferenceMode.hadamard_cz()".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self.inner {
            RustInterferenceMode::None => "none".to_string(),
            RustInterferenceMode::SoftMix => "soft_mix".to_string(),
            RustInterferenceMode::Grover => "grover".to_string(),
            RustInterferenceMode::AntiGrover => "anti_grover".to_string(),
            RustInterferenceMode::Adaptive => "adaptive".to_string(),
            RustInterferenceMode::Triggered => "triggered".to_string(),
            RustInterferenceMode::HadamardCZ => "hadamard_cz".to_string(),
        }
    }
}

/// Configuration for Quantum-SNN hybrid network.
///
/// Example:
///     >>> config = QuantumSNNConfig(
///     ...     n_inputs=8,
///     ...     hidden_layers=[32, 16],
///     ...     n_outputs=4,
///     ...     mode=InterferenceMode.triggered(),
///     ...     ticks_per_decision=20,
///     ... )
#[pyclass(name = "QuantumSNNConfig")]
#[derive(Clone, Debug)]
pub struct PyQuantumSNNConfig {
    /// Number of input neurons
    #[pyo3(get, set)]
    pub n_inputs: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Number of output neurons (actions)
    #[pyo3(get, set)]
    pub n_outputs: usize,
    /// Interference mode
    pub mode: RustInterferenceMode,
    /// Mixing angle for SoftMix mode (radians)
    #[pyo3(get, set)]
    pub mix_angle: f64,
    /// Ticks per decision (integration window)
    #[pyo3(get, set)]
    pub ticks_per_decision: usize,
    /// Enable R-STDP learning
    #[pyo3(get, set)]
    pub learning_enabled: bool,
    /// Reward threshold for Triggered mode
    #[pyo3(get, set)]
    pub trigger_threshold: i8,
    /// Consecutive high rewards required to trigger exploitation
    #[pyo3(get, set)]
    pub trigger_count: usize,
    /// Duplicate inputs for each output channel
    /// When true, creates n_inputs * n_outputs input neurons
    /// Enables channel-aware learning for context-dependent tasks (e.g., CartPole)
    #[pyo3(get, set)]
    pub duplicate_inputs: bool,
}

#[pymethods]
impl PyQuantumSNNConfig {
    /// Create a new configuration.
    ///
    /// Args:
    ///     n_inputs: Number of input neurons (sensor dimensions)
    ///     hidden_layers: List of hidden layer sizes (e.g., [32, 16])
    ///     n_outputs: Number of output neurons (actions)
    ///     mode: Interference mode (default: triggered)
    ///     mix_angle: Mixing angle for soft_mix mode (default: 0.1)
    ///     ticks_per_decision: SNN ticks per decision (default: 20)
    ///     learning_enabled: Enable R-STDP learning (default: True)
    ///     trigger_threshold: Reward threshold for triggered mode (default: 50)
    ///     trigger_count: Consecutive high rewards to trigger exploitation (default: 5)
    ///     duplicate_inputs: Duplicate inputs for each channel (default: False)
    #[new]
    #[pyo3(signature = (
        n_inputs=16,
        hidden_layers=None,
        n_outputs=4,
        mode=None,
        mix_angle=0.1,
        ticks_per_decision=20,
        learning_enabled=true,
        trigger_threshold=50,
        trigger_count=5,
        duplicate_inputs=false
    ))]
    fn new(
        n_inputs: usize,
        hidden_layers: Option<Vec<usize>>,
        n_outputs: usize,
        mode: Option<&PyInterferenceMode>,
        mix_angle: f64,
        ticks_per_decision: usize,
        learning_enabled: bool,
        trigger_threshold: i8,
        trigger_count: usize,
        duplicate_inputs: bool,
    ) -> Self {
        let hidden = hidden_layers.unwrap_or_else(|| vec![32, 16]);
        let interference_mode = mode
            .map(|m| m.inner)
            .unwrap_or(RustInterferenceMode::Triggered);

        PyQuantumSNNConfig {
            n_inputs,
            hidden_layers: hidden,
            n_outputs,
            mode: interference_mode,
            mix_angle,
            ticks_per_decision,
            learning_enabled,
            trigger_threshold,
            trigger_count,
            duplicate_inputs,
        }
    }

    /// Get hidden layer sizes
    #[getter]
    fn hidden_layers(&self) -> Vec<usize> {
        self.hidden_layers.clone()
    }

    /// Set hidden layer sizes
    #[setter]
    fn set_hidden_layers(&mut self, layers: Vec<usize>) {
        self.hidden_layers = layers;
    }

    /// Get interference mode
    #[getter]
    fn mode(&self) -> PyInterferenceMode {
        PyInterferenceMode { inner: self.mode }
    }

    /// Set interference mode
    #[setter]
    fn set_mode(&mut self, mode: &PyInterferenceMode) {
        self.mode = mode.inner;
    }

    /// Create a small config for testing
    #[staticmethod]
    fn small() -> Self {
        PyQuantumSNNConfig {
            n_inputs: 8,
            hidden_layers: vec![16],
            n_outputs: 4,
            mode: RustInterferenceMode::Triggered,
            mix_angle: 0.1,
            ticks_per_decision: 10,
            learning_enabled: true,
            trigger_threshold: 50,
            trigger_count: 5,
            duplicate_inputs: false,
        }
    }

    /// Create config optimized for sparse reward tasks
    #[staticmethod]
    fn sparse_reward(n_inputs: usize, n_outputs: usize) -> Self {
        PyQuantumSNNConfig {
            n_inputs,
            hidden_layers: vec![64, 32],
            n_outputs,
            mode: RustInterferenceMode::Triggered,
            mix_angle: 0.1,
            ticks_per_decision: 20,
            learning_enabled: true,
            trigger_threshold: 50,
            trigger_count: 5,
            duplicate_inputs: false,
        }
    }

    /// Total neurons in network (accounting for input duplication)
    fn total_neurons(&self) -> usize {
        let actual_inputs = if self.duplicate_inputs {
            self.n_inputs * self.n_outputs
        } else {
            self.n_inputs
        };
        actual_inputs + self.hidden_layers.iter().sum::<usize>() + self.n_outputs
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumSNNConfig(n_inputs={}, hidden={:?}, n_outputs={}, mode={:?})",
            self.n_inputs, self.hidden_layers, self.n_outputs, self.mode
        )
    }
}

impl PyQuantumSNNConfig {
    fn to_rust(&self) -> RustQuantumSNNConfig {
        RustQuantumSNNConfig {
            n_inputs: self.n_inputs,
            hidden_layers: self.hidden_layers.clone(),
            n_outputs: self.n_outputs,
            interference_mode: self.mode,
            interference_depth: 1,
            mix_angle: self.mix_angle,
            ticks_per_decision: self.ticks_per_decision,
            learning_enabled: self.learning_enabled,
            trigger_threshold: self.trigger_threshold,
            trigger_count: self.trigger_count,
            duplicate_inputs: self.duplicate_inputs,
        }
    }
}

/// Quantum-SNN Hybrid Neural Network.
///
/// Combines a classical Spiking Neural Network (SNN) for feature extraction
/// with a sparse quantum decision layer for action selection via interference.
///
/// Key features:
/// - LIF neurons with STDP/R-STDP learning
/// - Quantum interference for exploration/exploitation trade-off
/// - Triggered mode: explore until reward, then exploit (+69% over epsilon-greedy)
///
/// Example:
///     >>> from tileuniverse import QuantumSNN, QuantumSNNConfig, InterferenceMode
///     >>>
///     >>> # Create network for 4-action RL task
///     >>> config = QuantumSNNConfig(n_inputs=8, n_outputs=4, mode=InterferenceMode.triggered())
///     >>> brain = QuantumSNN(config, seed=42)
///     >>>
///     >>> # Decision loop
///     >>> for step in range(1000):
///     ...     inputs = get_sensor_readings()  # [0-255] per input
///     ...     action = brain.decide(inputs)
///     ...     reward = environment.step(action)
///     ...     brain.learn(reward)
#[pyclass(name = "QuantumSNN", unsendable)]
pub struct PyQuantumSNN {
    inner: RustQuantumSNN,
    config: PyQuantumSNNConfig,
}

#[pymethods]
impl PyQuantumSNN {
    /// Create a new Quantum-SNN hybrid network.
    ///
    /// Args:
    ///     config: Network configuration
    ///     seed: Random seed for reproducibility (default: 42)
    #[new]
    #[pyo3(signature = (config, seed=42))]
    fn new(config: PyQuantumSNNConfig, seed: u64) -> Self {
        let rust_config = config.to_rust();
        let inner = RustQuantumSNN::with_seed(rust_config, seed);
        PyQuantumSNN { inner, config }
    }

    /// Set input spike rates from sensor readings.
    ///
    /// Args:
    ///     inputs: List of input values [0-255] for each input neuron.
    ///             Higher values = higher spike probability.
    fn set_inputs(&mut self, inputs: Vec<u8>) {
        self.inner.set_inputs(&inputs);
    }

    /// Run one complete decision cycle and return the chosen action.
    ///
    /// This method:
    /// 1. Runs the SNN for ticks_per_decision steps
    /// 2. Encodes spike counts as quantum amplitudes
    /// 3. Applies quantum interference (based on mode)
    /// 4. Measures to get action
    ///
    /// Args:
    ///     inputs: Optional input values. If provided, calls set_inputs first.
    ///
    /// Returns:
    ///     int: The selected action index [0, n_outputs)
    #[pyo3(signature = (inputs=None))]
    fn decide(&mut self, inputs: Option<Vec<u8>>) -> usize {
        if let Some(inp) = inputs {
            self.inner.set_inputs(&inp);
        }
        self.inner.decide()
    }

    /// Get the classical winner-take-all action (for comparison).
    ///
    /// Returns the action with the most spikes, without quantum interference.
    fn classical_action(&self) -> usize {
        self.inner.classical_action()
    }

    /// Provide reward and apply R-STDP learning.
    ///
    /// For Triggered mode, this also checks if the reward should trigger
    /// the switch from exploration to exploitation.
    ///
    /// Args:
    ///     reward: Reward signal [-127 to 127]. Positive = reinforce, negative = punish.
    fn learn(&mut self, reward: i8) {
        self.inner.learn(reward);
    }

    /// Action-gated learning: only update synapses causally connected to chosen action.
    ///
    /// This implements proper credit assignment for RL:
    /// - Only synapses that contributed to the selected action receive reward
    /// - Prevents symmetric reward from reinforcing all actions equally
    /// - Essential for environments like CartPole where all actions get same reward
    ///
    /// Args:
    ///     reward: Reward signal [-127 to 127]. Positive = reinforce, negative = punish.
    ///     action: Optional action index. If None, uses last action from decide().
    #[pyo3(signature = (reward, action=None))]
    fn learn_gated(&mut self, reward: i8, action: Option<usize>) {
        self.inner.learn_gated(reward, action);
    }

    /// Competitive learning: strengthen chosen action, weaken others.
    ///
    /// This explicitly creates competition between output neurons by weakening
    /// synapses targeting non-chosen outputs when reinforcing the chosen one.
    /// Helps differentiate actions even with shared hidden representations.
    ///
    /// Args:
    ///     reward: Reward signal [-127 to 127].
    ///     action: Optional action index. If None, uses last action from decide().
    ///     weaken_percent: How much to weaken other actions (0-100). Default 30.
    #[pyo3(signature = (reward, action=None, weaken_percent=None))]
    fn learn_competitive(&mut self, reward: i8, action: Option<usize>, weaken_percent: Option<i8>) {
        self.inner.learn_competitive(reward, action, weaken_percent);
    }

    /// Deep gated learning: update output AND selective hidden layer synapses.
    ///
    /// This extends gated learning to also update input->hidden synapses,
    /// but ONLY for hidden neurons that have stronger connections to the
    /// chosen output than to other outputs. Enables context-dependent learning.
    ///
    /// Args:
    ///     reward: Reward signal [-127 to 127].
    ///     action: Optional action index. If None, uses last action from decide().
    #[pyo3(signature = (reward, action=None))]
    fn learn_deep_gated(&mut self, reward: i8, action: Option<usize>) {
        self.inner.learn_deep_gated(reward, action);
    }

    /// Channel-aware learning: respect the network's channel structure.
    ///
    /// The network is built with dedicated neuron channels per action.
    /// This method only updates synapses where BOTH source and target
    /// belong to the chosen action's channel, preventing learning interference.
    ///
    /// Best for context-dependent tasks where different inputs require
    /// different actions (e.g., contextual bandit, CartPole).
    ///
    /// Args:
    ///     reward: Reward signal [-127 to 127].
    ///     action: Optional action index. If None, uses last action from decide().
    #[pyo3(signature = (reward, action=None))]
    fn learn_channeled(&mut self, reward: i8, action: Option<usize>) {
        self.inner.learn_channeled(reward, action);
    }

    /// Set reward signal without applying learning yet.
    fn set_reward(&mut self, reward: i8) {
        self.inner.set_reward(reward);
    }

    /// Apply accumulated eligibility traces with current reward.
    fn apply_learning(&mut self) {
        self.inner.apply_learning();
    }

    /// Get the last action taken.
    #[getter]
    fn last_action(&self) -> usize {
        self.inner.last_action()
    }

    /// Get probability distribution from last decision.
    ///
    /// Returns:
    ///     List of probabilities for each action (sums to 1.0)
    #[getter]
    fn last_probabilities(&self) -> Vec<f64> {
        self.inner.last_probabilities().to_vec()
    }

    /// Get output spike counts from last decision period.
    #[getter]
    fn output_spike_counts(&self) -> Vec<u32> {
        self.inner.output_spike_counts().to_vec()
    }

    /// Get total ticks elapsed.
    #[getter]
    fn tick(&self) -> u64 {
        self.inner.tick()
    }

    /// Check if exploitation has been triggered (for Triggered mode).
    #[getter]
    fn is_exploitation_triggered(&self) -> bool {
        self.inner.is_exploitation_triggered()
    }

    /// Get consecutive high reward count (for Triggered mode).
    #[getter]
    fn consecutive_high_rewards(&self) -> usize {
        self.inner.consecutive_high_rewards()
    }

    /// Manually trigger exploitation mode.
    fn trigger_exploitation(&mut self) {
        self.inner.trigger_exploitation();
    }

    /// Reset to exploration mode.
    fn reset_trigger(&mut self) {
        self.inner.reset_trigger();
    }

    /// Get number of active quantum blocks (sparsity metric).
    #[getter]
    fn quantum_blocks(&self) -> usize {
        self.inner.quantum_blocks()
    }

    /// Get network configuration.
    #[getter]
    fn config(&self) -> PyQuantumSNNConfig {
        self.config.clone()
    }

    /// Reset the network state (clear spikes, reset trigger).
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Calculate exploration entropy (bits) from last decision.
    ///
    /// Higher entropy = more exploration, lower = more exploitation.
    fn exploration_entropy(&self) -> f64 {
        let probs = self.inner.last_probabilities();
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

    // =========================================================================
    // Curiosity-Driven Intrinsic Motivation (EPIC 121)
    // =========================================================================

    /// Enable novelty-based curiosity for intrinsic motivation.
    ///
    /// Provides intrinsic rewards for visiting novel states, encouraging
    /// exploration in sparse reward environments.
    ///
    /// Args:
    ///     reward_scale: Scale factor for intrinsic rewards (default: 0.5)
    ///     decay_rate: How fast to forget old visits, 0-1 (default: 0.0 = never)
    #[pyo3(signature = (reward_scale=0.5, decay_rate=0.0))]
    fn enable_novelty_curiosity(&mut self, reward_scale: f32, decay_rate: f32) {
        self.inner
            .enable_novelty_curiosity(reward_scale, decay_rate);
    }

    /// Enable prediction-error curiosity.
    ///
    /// Provides intrinsic rewards for surprising state transitions,
    /// encouraging exploration of unpredictable parts of the environment.
    ///
    /// Args:
    ///     learning_rate: How fast to update the prediction model (default: 0.01)
    ///     reward_scale: Scale factor for intrinsic rewards (default: 0.5)
    #[pyo3(signature = (learning_rate=0.01, reward_scale=0.5))]
    fn enable_prediction_curiosity(&mut self, learning_rate: f32, reward_scale: f32) {
        self.inner
            .enable_prediction_curiosity(learning_rate, reward_scale);
    }

    /// Enable combined curiosity (novelty + prediction error).
    ///
    /// Args:
    ///     reward_scale: Scale factor for intrinsic rewards (default: 0.5)
    #[pyo3(signature = (reward_scale=0.5))]
    fn enable_combined_curiosity(&mut self, reward_scale: f32) {
        self.inner.enable_combined_curiosity(reward_scale);
    }

    /// Disable curiosity-driven exploration.
    fn disable_curiosity(&mut self) {
        self.inner.disable_curiosity();
    }

    /// Check if curiosity is enabled.
    #[getter]
    fn has_curiosity(&self) -> bool {
        self.inner.has_curiosity()
    }

    /// Learn with curiosity-augmented reward.
    ///
    /// Combines extrinsic reward with intrinsic curiosity reward.
    ///
    /// Args:
    ///     reward: External reward from environment [-127 to 127]
    ///     current_state: Current observation as list of floats [0-1]
    ///
    /// Returns:
    ///     Total reward (extrinsic + intrinsic)
    fn learn_with_curiosity(&mut self, reward: i8, current_state: Vec<f32>) -> f32 {
        self.inner.learn_with_curiosity(reward, &current_state)
    }

    /// Apply curiosity decay (call at end of episode).
    ///
    /// For novelty-based curiosity, decays visit counts so old states
    /// become novel again over time.
    fn decay_curiosity(&mut self) {
        self.inner.decay_curiosity();
    }

    // =========================================================================
    // TD-Style Credit Assignment (EPIC 122)
    // =========================================================================

    /// Enable TD-style credit assignment for delayed rewards.
    ///
    /// Uses temporal difference learning to bootstrap value estimates,
    /// providing better credit assignment for delayed reward problems.
    ///
    /// Args:
    ///     alpha: Value function learning rate (default: 0.05)
    ///     gamma: Discount factor (default: 0.95)
    ///     td_scale: Scale factor for TD error signal (default: 50.0)
    #[pyo3(signature = (alpha=0.05, gamma=0.95, td_scale=50.0))]
    fn enable_td_credit(&mut self, alpha: f32, gamma: f32, td_scale: f32) {
        self.inner.enable_td_credit(alpha, gamma, td_scale);
    }

    /// Disable TD credit assignment.
    fn disable_td_credit(&mut self) {
        self.inner.disable_td_credit();
    }

    /// Check if TD credit is enabled.
    #[getter]
    fn has_td_credit(&self) -> bool {
        self.inner.has_td_credit()
    }

    /// Learn with TD-modulated credit assignment.
    ///
    /// Uses TD error instead of raw reward for R-STDP, providing better
    /// credit assignment for delayed rewards.
    ///
    /// Args:
    ///     reward: Immediate reward from environment
    ///     current_state: Current state as list of floats [0-1]
    ///     done: Whether episode ended (default: False)
    ///
    /// Returns:
    ///     The TD error (positive = better than expected)
    #[pyo3(signature = (reward, current_state, done=false))]
    fn learn_with_td(&mut self, reward: f32, current_state: Vec<f32>, done: bool) -> f32 {
        self.inner.learn_with_td(reward, &current_state, done)
    }

    /// Get estimated value of a state (requires TD credit enabled).
    ///
    /// Args:
    ///     state: State as list of floats [0-1]
    ///
    /// Returns:
    ///     Estimated value, or None if TD credit not enabled
    fn state_value(&self, state: Vec<f32>) -> Option<f32> {
        self.inner.state_value(&state)
    }

    /// Reset TD credit at episode boundary.
    fn reset_td_episode(&mut self) {
        self.inner.reset_td_episode();
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumSNN(neurons={}, tick={}, triggered={})",
            self.config.total_neurons(),
            self.inner.tick(),
            self.inner.is_exploitation_triggered()
        )
    }
}

// =============================================================================
// SPRINT 59.0: STABILIZER-BASED QUANTUM NEURAL NETWORKS
// =============================================================================

/// Feedback mode for hybrid quantum-neural networks.
///
/// Controls how quantum cluster states influence the stabilizer backbone.
#[pyclass(name = "FeedbackMode")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyFeedbackMode {
    /// No feedback (default)
    #[pyo3(name = "NONE")]
    None,
    /// Phase feedback via S gates
    #[pyo3(name = "PHASE")]
    Phase,
    /// Amplitude-based feedback
    #[pyo3(name = "AMPLITUDE")]
    Amplitude,
    /// Gated feedback (enable/disable synapses)
    #[pyo3(name = "GATED")]
    Gated,
}

impl PyFeedbackMode {
    fn to_rust(&self) -> RustFeedbackMode {
        match self {
            PyFeedbackMode::None => RustFeedbackMode::None,
            PyFeedbackMode::Phase => RustFeedbackMode::Phase,
            PyFeedbackMode::Amplitude => RustFeedbackMode::Amplitude,
            PyFeedbackMode::Gated => RustFeedbackMode::Gated,
        }
    }
}

/// Bridge gate type for inter-cluster quantum operations.
#[pyclass(name = "BridgeGate")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyBridgeGate {
    /// Teleport state from source to target
    #[pyo3(name = "TELEPORT")]
    Teleport,
    /// Swap states between clusters
    #[pyo3(name = "REMOTE_SWAP")]
    RemoteSwap,
    /// Distributed CZ gate (creates entanglement)
    #[pyo3(name = "DISTRIBUTED_CZ")]
    DistributedCZ,
}

impl PyBridgeGate {
    fn to_rust(&self) -> RustBridgeGate {
        match self {
            PyBridgeGate::Teleport => RustBridgeGate::Teleport,
            PyBridgeGate::RemoteSwap => RustBridgeGate::RemoteSwap,
            PyBridgeGate::DistributedCZ => RustBridgeGate::DistributedCZ,
        }
    }
}

/// Stabilizer-based Spiking Neural Network.
///
/// Uses the stabilizer formalism for efficient simulation of quantum neurons.
/// Supports O(n²) memory scaling for n neurons, enabling 10K+ quantum neurons.
///
/// Example:
///     >>> snn = StabilizerSNN(n_inputs=4, hidden=[16], n_outputs=2)
///     >>> action = snn.decide([200, 150, 100, 50])
///     >>> snn.learn(50)  # Positive reward
#[pyclass(name = "StabilizerSNN", unsendable)]
pub struct PyStabilizerSNN {
    inner: RustStabilizerNeuronNetwork,
}

#[pymethods]
impl PyStabilizerSNN {
    /// Create a new stabilizer-based SNN.
    ///
    /// Args:
    ///     n_inputs: Number of input neurons
    ///     hidden: List of hidden layer sizes (e.g., [32, 16])
    ///     n_outputs: Number of output neurons (actions)
    ///     connectivity: Connection probability between layers (default: 0.3)
    ///     seed: Random seed for reproducibility (default: 42)
    #[new]
    #[pyo3(signature = (n_inputs, hidden, n_outputs, connectivity=0.3, seed=42))]
    fn new(
        n_inputs: usize,
        hidden: Vec<usize>,
        n_outputs: usize,
        connectivity: f32,
        seed: u64,
    ) -> Self {
        let config = RustStabilizerNetworkConfig {
            n_inputs,
            hidden_layers: hidden,
            n_outputs,
            connectivity,
            ..Default::default()
        };
        let inner = RustStabilizerNeuronNetwork::new(config, seed);
        Self { inner }
    }

    /// Run one tick with input spike rates.
    ///
    /// Args:
    ///     inputs: List of input values [0-255] for each input neuron.
    ///
    /// Returns:
    ///     List of neuron indices that fired this tick.
    fn step(&mut self, inputs: Vec<u8>) -> Vec<usize> {
        self.inner.step(&inputs)
    }

    /// Make a decision based on output spike counts.
    ///
    /// Runs one tick and returns the output with most spikes.
    ///
    /// Args:
    ///     inputs: List of input values [0-255].
    ///
    /// Returns:
    ///     Action index (output neuron with most activity).
    fn decide(&mut self, inputs: Vec<u8>) -> usize {
        self.inner.decide(&inputs)
    }

    /// Apply reward-modulated learning.
    ///
    /// Args:
    ///     reward: Reward signal [-127 to 127]. Positive = reinforce.
    fn learn(&mut self, reward: i8) {
        self.inner.learn(reward);
    }

    /// Get output spike counts.
    #[getter]
    fn output_spike_counts(&self) -> Vec<u32> {
        self.inner.get_output_spike_counts()
    }

    /// Get current tick.
    #[getter]
    fn tick(&self) -> u64 {
        self.inner.tick()
    }

    /// Get total neuron count.
    #[getter]
    fn num_neurons(&self) -> usize {
        self.inner.num_neurons()
    }

    /// Get memory usage in bytes.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_usage()
    }

    /// Reset output spike counts.
    fn reset_output_counts(&mut self) {
        self.inner.reset_output_counts();
    }

    fn __repr__(&self) -> String {
        format!(
            "StabilizerSNN(neurons={}, tick={}, memory={}KB)",
            self.inner.num_neurons(),
            self.inner.tick(),
            self.inner.memory_usage() / 1024
        )
    }
}

/// Hybrid Quantum Neural Network.
///
/// Combines a stabilizer backbone (10K+ neurons) with small full-quantum
/// clusters (8-16 qubits each) for true quantum advantage via non-Clifford gates.
///
/// Features:
/// - Adaptive cluster activation based on input entropy
/// - Bidirectional feedback between backbone and clusters
/// - Cluster-to-cluster bridges for distributed quantum processing
/// - Magic state injection for T gates
///
/// Example:
///     >>> hybrid = HybridQuantumSNN(n_inputs=4, hidden=[32], n_outputs=3,
///     ...                           n_clusters=2, cluster_size=8)
///     >>> action = hybrid.decide([200, 150, 100, 50])
///     >>> hybrid.learn(50)
#[pyclass(name = "HybridQuantumSNN", unsendable)]
pub struct PyHybridQuantumSNN {
    inner: RustHybridQuantumNetwork,
    recorder: Option<QuantumNeuralRecorder>,
}

#[pymethods]
impl PyHybridQuantumSNN {
    /// Create a new hybrid quantum-neural network.
    ///
    /// Args:
    ///     n_inputs: Number of input neurons
    ///     hidden: List of hidden layer sizes
    ///     n_outputs: Number of output neurons (actions)
    ///     n_clusters: Number of full-quantum clusters (default: 1)
    ///     cluster_size: Qubits per cluster, 4-16 (default: 8)
    ///     feedback_mode: Feedback mode (default: NONE)
    ///     adaptive_threshold: Entropy threshold for adaptive activation (default: None)
    ///     seed: Random seed (default: 42)
    #[new]
    #[pyo3(signature = (n_inputs, hidden, n_outputs, n_clusters=1, cluster_size=8, feedback_mode=None, adaptive_threshold=None, seed=42))]
    fn new(
        n_inputs: usize,
        hidden: Vec<usize>,
        n_outputs: usize,
        n_clusters: usize,
        cluster_size: usize,
        feedback_mode: Option<PyFeedbackMode>,
        adaptive_threshold: Option<f64>,
        seed: u64,
    ) -> Self {
        let backbone_config = RustStabilizerNetworkConfig {
            n_inputs,
            hidden_layers: hidden,
            n_outputs,
            connectivity: 0.3,
            ..Default::default()
        };

        let mut config = RustHybridNetworkConfig {
            backbone: backbone_config,
            n_clusters,
            cluster_size,
            ..Default::default()
        };

        // Set feedback mode
        if let Some(mode) = feedback_mode {
            config.feedback_mode = mode.to_rust();
        }

        // Enable adaptive mode if threshold specified
        if let Some(threshold) = adaptive_threshold {
            config = config.with_adaptive(threshold);
        }

        let inner = RustHybridQuantumNetwork::new(config, seed);
        Self {
            inner,
            recorder: None,
        }
    }

    /// Run one tick with input spike rates.
    ///
    /// Args:
    ///     inputs: List of input values [0-255].
    ///
    /// Returns:
    ///     List of neuron indices that fired.
    fn step(&mut self, inputs: Vec<u8>) -> Vec<usize> {
        let fired = self.inner.step(&inputs);

        // Record if enabled
        if let Some(ref mut rec) = self.recorder {
            rec.record_with_spikes(&self.inner, &fired);
        }

        fired
    }

    /// Make a decision based on output spike counts.
    ///
    /// Args:
    ///     inputs: List of input values [0-255].
    ///
    /// Returns:
    ///     Action index.
    fn decide(&mut self, inputs: Vec<u8>) -> usize {
        self.inner.decide(&inputs)
    }

    /// Run multiple ticks and then decide.
    ///
    /// Args:
    ///     inputs: Input values
    ///     ticks: Number of ticks to run (default: 10)
    ///
    /// Returns:
    ///     Action index based on integrated spike counts.
    #[pyo3(signature = (inputs, ticks=10))]
    fn decide_integrated(&mut self, inputs: Vec<u8>, ticks: usize) -> usize {
        self.inner.decide_integrated(&inputs, ticks)
    }

    /// Apply reward-modulated learning.
    ///
    /// Args:
    ///     reward: Reward signal [-127 to 127].
    fn learn(&mut self, reward: i8) {
        self.inner.learn(reward);
    }

    /// Get output spike counts.
    #[getter]
    fn output_spike_counts(&self) -> Vec<u32> {
        self.inner.get_output_spike_counts()
    }

    /// Reset output spike counts.
    fn reset_output_counts(&mut self) {
        self.inner.reset_output_counts();
    }

    /// Get current tick.
    #[getter]
    fn tick(&self) -> u64 {
        self.inner.tick()
    }

    /// Get number of clusters.
    #[getter]
    fn n_clusters(&self) -> usize {
        self.inner.n_clusters()
    }

    /// Get number of active clusters.
    #[getter]
    fn active_clusters(&self) -> usize {
        self.inner.active_cluster_count()
    }

    /// Get memory usage in bytes.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Get entanglement entropy for a cluster.
    ///
    /// Args:
    ///     cluster_id: Cluster index
    ///
    /// Returns:
    ///     Entropy in bits (higher = more entanglement)
    fn cluster_entropy(&self, cluster_id: usize) -> f64 {
        self.inner.cluster_entropy(cluster_id)
    }

    /// Get total Z expectation across all clusters.
    ///
    /// Returns:
    ///     Average Z expectation. +1 = all |0⟩, -1 = all |1⟩.
    fn total_z_expectation(&self) -> f64 {
        self.inner.total_cluster_z_expectation()
    }

    // =========================================================================
    // CLUSTER BRIDGES (EPIC 139)
    // =========================================================================

    /// Add a bridge between two clusters.
    ///
    /// Args:
    ///     source_cluster: Source cluster ID
    ///     source_qubit: Source qubit index within cluster
    ///     target_cluster: Target cluster ID
    ///     target_qubit: Target qubit index within cluster
    ///     gate_type: Type of bridge gate (default: DISTRIBUTED_CZ)
    #[pyo3(signature = (source_cluster, source_qubit, target_cluster, target_qubit, gate_type=None))]
    fn add_bridge(
        &mut self,
        source_cluster: usize,
        source_qubit: usize,
        target_cluster: usize,
        target_qubit: usize,
        gate_type: Option<PyBridgeGate>,
    ) {
        let gate = gate_type.unwrap_or(PyBridgeGate::DistributedCZ).to_rust();
        let bridge = RustClusterBridge::new(
            source_cluster,
            source_qubit,
            target_cluster,
            target_qubit,
            gate,
        );
        self.inner.add_bridge(bridge);
    }

    /// Get number of bridges.
    #[getter]
    fn n_bridges(&self) -> usize {
        self.inner.n_bridges()
    }

    /// Apply all active bridges.
    fn apply_bridges(&mut self) {
        self.inner.apply_bridges();
    }

    /// Create chain topology connecting clusters.
    fn chain_clusters(&mut self) {
        self.inner.chain_clusters();
    }

    /// Create ring topology connecting clusters.
    fn ring_clusters(&mut self) {
        self.inner.ring_clusters();
    }

    /// Create fully connected mesh topology.
    fn mesh_clusters(&mut self) {
        self.inner.mesh_clusters();
    }

    /// Create star topology with center cluster.
    ///
    /// Args:
    ///     center: Index of center cluster
    fn star_clusters(&mut self, center: usize) {
        self.inner.star_clusters(center);
    }

    /// Clear all bridges.
    fn clear_bridges(&mut self) {
        self.inner.clear_bridges();
    }

    // =========================================================================
    // RECORDING (EPIC 136)
    // =========================================================================

    /// Enable recording of network snapshots.
    ///
    /// Args:
    ///     max_snapshots: Maximum snapshots to retain (default: 1000)
    #[pyo3(signature = (max_snapshots=1000))]
    fn enable_recording(&mut self, max_snapshots: usize) {
        self.recorder = Some(QuantumNeuralRecorder::new(max_snapshots));
    }

    /// Disable recording.
    fn disable_recording(&mut self) {
        self.recorder = None;
    }

    /// Check if recording is enabled.
    #[getter]
    fn is_recording(&self) -> bool {
        self.recorder.is_some()
    }

    /// Get entropy time series for a cluster.
    ///
    /// Args:
    ///     cluster_id: Cluster index
    ///
    /// Returns:
    ///     List of entropy values over time.
    fn entropy_series(&self, cluster_id: usize) -> Vec<f64> {
        self.recorder
            .as_ref()
            .map(|r| r.entropy_series(cluster_id))
            .unwrap_or_default()
    }

    /// Get average entropy time series across all clusters.
    fn average_entropy_series(&self) -> Vec<f64> {
        self.recorder
            .as_ref()
            .map(|r| r.average_entropy_series())
            .unwrap_or_default()
    }

    /// Get number of recorded snapshots.
    fn recording_length(&self) -> usize {
        self.recorder.as_ref().map(|r| r.len()).unwrap_or(0)
    }

    fn __repr__(&self) -> String {
        format!(
            "HybridQuantumSNN(clusters={}, active={}, tick={}, memory={}KB, bridges={})",
            self.inner.n_clusters(),
            self.inner.active_cluster_count(),
            self.inner.tick(),
            self.inner.memory_bytes() / 1024,
            self.inner.n_bridges()
        )
    }
}

// =============================================================================
// FAST W STATE BINDINGS: Vec-based sparse quantum states (billions of qubits)
// =============================================================================

/// Verification result for W state simulation.
///
/// Contains fidelity metrics and memory usage statistics.
#[pyclass(name = "WStateVerification")]
#[derive(Clone, Debug)]
pub struct PyWStateVerification {
    /// Number of qubits in the W state
    #[pyo3(get)]
    pub n_qubits: usize,
    /// Expected number of non-zero amplitudes (equals n_qubits for W state)
    #[pyo3(get)]
    pub expected_amplitudes: usize,
    /// Number of amplitudes with correct value (within tolerance)
    #[pyo3(get)]
    pub correct_amplitudes: usize,
    /// Expected amplitude value (1/√n)
    #[pyo3(get)]
    pub expected_amplitude: f64,
    /// Total probability (should be 1.0 for normalized state)
    #[pyo3(get)]
    pub total_probability: f64,
    /// Fidelity (overlap with ideal W state)
    #[pyo3(get)]
    pub fidelity: f64,
    /// Maximum amplitude error
    #[pyo3(get)]
    pub max_amplitude_error: f64,
    /// Number of active memory blocks
    #[pyo3(get)]
    pub active_blocks: usize,
    /// Memory usage in bytes
    #[pyo3(get)]
    pub memory_bytes: usize,
}

#[pymethods]
impl PyWStateVerification {
    /// Check if fidelity is within acceptable tolerance (> 0.999)
    fn is_valid(&self) -> bool {
        (self.fidelity - 1.0).abs() < 0.001
    }

    /// Get memory usage in megabytes
    fn memory_mb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get memory usage in gigabytes
    fn memory_gb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "WStateVerification(n_qubits={}, fidelity={:.6}, memory={:.1}MB)",
            self.n_qubits,
            self.fidelity,
            self.memory_mb()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "W|{}>: fidelity={:.6}, correct={}/{}, memory={:.1}MB",
            self.n_qubits,
            self.fidelity,
            self.correct_amplitudes,
            self.expected_amplitudes,
            self.memory_mb()
        )
    }
}

impl From<RustWStateVerification> for PyWStateVerification {
    fn from(v: RustWStateVerification) -> Self {
        PyWStateVerification {
            n_qubits: v.n_qubits,
            expected_amplitudes: v.expected_amplitudes,
            correct_amplitudes: v.correct_amplitudes,
            expected_amplitude: v.expected_amplitude,
            total_probability: v.total_probability,
            fidelity: v.fidelity,
            max_amplitude_error: v.max_amplitude_error,
            active_blocks: v.active_blocks,
            memory_bytes: v.memory_bytes,
        }
    }
}

/// Fast Vec-based W state simulation.
///
/// Enables simulation of W states with billions of qubits by using
/// direct Vec indexing instead of HashMap<BigInt> for block storage.
///
/// Performance comparison:
/// | Qubits | BigInt W State | Vec W State |
/// |--------|----------------|-------------|
/// | 1M     | ~30 seconds    | ~25 ms      |
/// | 100M   | impossible     | ~130 ms     |
/// | 1B     | impossible     | ~1.2 sec    |
/// | 4B     | impossible     | ~14 sec     |
///
/// Example:
///     >>> from tileuniverse import FastWState
///     >>>
///     >>> # Create 1 billion qubit W state
///     >>> w = FastWState(1_000_000_000)
///     >>> print(f"Created in {w.creation_time_ms:.1f}ms")
///     >>>
///     >>> # Verify fidelity
///     >>> v = w.verify()
///     >>> print(f"Fidelity: {v.fidelity:.6f}")
///     >>> print(f"Memory: {v.memory_gb():.1f} GB")
#[pyclass(name = "FastWState", unsendable)]
pub struct PyFastWState {
    inner: SparseQuantumGridVec,
    creation_time_us: u64,
}

#[pymethods]
impl PyFastWState {
    /// Create a new W state with the specified number of qubits.
    ///
    /// W state: |W_n⟩ = (|10...0⟩ + |01...0⟩ + ... + |00...1⟩) / √n
    ///
    /// Args:
    ///     n_qubits: Number of qubits (minimum 7, no practical maximum)
    ///     parallel: Use parallel creation for large states (default: auto)
    ///
    /// Returns:
    ///     FastWState object with the W state created
    #[new]
    #[pyo3(signature = (n_qubits, parallel=None))]
    fn new(n_qubits: usize, parallel: Option<bool>) -> PyResult<Self> {
        if n_qubits < 7 {
            return Err(PyValueError::new_err("n_qubits must be >= 7"));
        }

        let mut grid = SparseQuantumGridVec::new(n_qubits);

        // Auto-select parallel for large states
        let use_parallel = parallel.unwrap_or(n_qubits > 100_000);

        let creation_time_us = if use_parallel {
            grid.create_w_state_parallel()
        } else {
            grid.create_w_state()
        };

        Ok(PyFastWState {
            inner: grid,
            creation_time_us,
        })
    }

    /// Get the number of qubits
    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Get the number of memory blocks
    #[getter]
    fn n_blocks(&self) -> usize {
        self.inner.n_blocks()
    }

    /// Get the number of allocated blocks
    #[getter]
    fn allocated_blocks(&self) -> usize {
        self.inner.allocated_blocks()
    }

    /// Get memory usage in bytes
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Get memory usage in megabytes
    fn memory_mb(&self) -> f64 {
        self.inner.memory_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Get memory usage in gigabytes
    fn memory_gb(&self) -> f64 {
        self.inner.memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get creation time in microseconds
    #[getter]
    fn creation_time_us(&self) -> u64 {
        self.creation_time_us
    }

    /// Get creation time in milliseconds
    #[getter]
    fn creation_time_ms(&self) -> f64 {
        self.creation_time_us as f64 / 1000.0
    }

    /// Get creation time in seconds
    #[getter]
    fn creation_time_sec(&self) -> f64 {
        self.creation_time_us as f64 / 1_000_000.0
    }

    /// Verify the W state fidelity.
    ///
    /// Args:
    ///     parallel: Use parallel verification (default: auto for large states)
    ///
    /// Returns:
    ///     WStateVerification with fidelity metrics
    #[pyo3(signature = (parallel=None))]
    fn verify(&self, parallel: Option<bool>) -> PyWStateVerification {
        let use_parallel = parallel.unwrap_or(self.inner.n_qubits() > 100_000);

        let result = if use_parallel {
            self.inner.verify_w_fidelity_parallel()
        } else {
            self.inner.verify_w_fidelity()
        };

        result.into()
    }

    /// Get amplitude at a specific qubit position.
    ///
    /// For W state, qubit i has amplitude 1/√n at position |0...1_i...0⟩
    ///
    /// Args:
    ///     qubit: Qubit index (0 to n_qubits-1)
    ///
    /// Returns:
    ///     Tuple (real, imaginary) of the amplitude
    fn get_amplitude(&self, qubit: usize) -> PyResult<(f64, f64)> {
        if qubit >= self.inner.n_qubits() {
            return Err(PyValueError::new_err(format!(
                "qubit {} out of range (0..{})",
                qubit,
                self.inner.n_qubits()
            )));
        }
        let amp = self.inner.get_amplitude(qubit);
        Ok((amp.re, amp.im))
    }

    fn __repr__(&self) -> String {
        let mem = self.memory_mb();
        if mem > 1024.0 {
            format!(
                "FastWState(n_qubits={}, memory={:.1}GB, created_in={:.1}ms)",
                self.inner.n_qubits(),
                mem / 1024.0,
                self.creation_time_ms()
            )
        } else {
            format!(
                "FastWState(n_qubits={}, memory={:.1}MB, created_in={:.1}ms)",
                self.inner.n_qubits(),
                mem,
                self.creation_time_ms()
            )
        }
    }

    fn __str__(&self) -> String {
        let n = self.inner.n_qubits();
        let label = if n >= 1_000_000_000 {
            format!("{:.1}B", n as f64 / 1_000_000_000.0)
        } else if n >= 1_000_000 {
            format!("{:.1}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.1}K", n as f64 / 1_000.0)
        } else {
            format!("{}", n)
        };
        format!("|W_{}>", label)
    }
}

/// Create a fast W state (convenience function).
///
/// Args:
///     n_qubits: Number of qubits (minimum 7)
///     parallel: Use parallel creation (default: auto)
///
/// Returns:
///     FastWState object
///
/// Example:
///     >>> w = create_fast_w_state(1_000_000)
///     >>> print(w)
///     |W_1.0M⟩
#[pyfunction]
#[pyo3(signature = (n_qubits, parallel=None))]
fn create_fast_w_state(n_qubits: usize, parallel: Option<bool>) -> PyResult<PyFastWState> {
    PyFastWState::new(n_qubits, parallel)
}

// =============================================================================
// FAST GHZ STATE BINDINGS: O(1) sparse (only 2 amplitudes for any qubit count)
// =============================================================================

/// Verification result for GHZ state simulation.
///
/// GHZ state: (|00...0> + |11...1>) / sqrt(2)
/// Only 2 non-zero amplitudes regardless of qubit count!
#[pyclass(name = "GHZStateVerification")]
#[derive(Clone, Debug)]
pub struct PyGHZStateVerification {
    /// Number of qubits in the GHZ state
    #[pyo3(get)]
    pub n_qubits: usize,
    /// Amplitude of |00...0> state
    #[pyo3(get)]
    pub amp_zero_state: f64,
    /// Amplitude of |11...1> state
    #[pyo3(get)]
    pub amp_one_state: f64,
    /// Expected amplitude (1/sqrt(2) = 0.7071...)
    #[pyo3(get)]
    pub expected_amplitude: f64,
    /// Fidelity (should be 1.0 for perfect GHZ)
    #[pyo3(get)]
    pub fidelity: f64,
    /// Maximum spurious amplitude (should be 0.0)
    #[pyo3(get)]
    pub max_spurious: f64,
    /// Number of active memory blocks (should be 2)
    #[pyo3(get)]
    pub active_blocks: usize,
}

#[pymethods]
impl PyGHZStateVerification {
    /// Check if fidelity is within acceptable tolerance (> 0.999)
    fn is_valid(&self) -> bool {
        (self.fidelity - 1.0).abs() < 0.001
    }

    /// Check if state is pure GHZ (no spurious amplitudes)
    fn is_pure(&self) -> bool {
        self.max_spurious < 1e-10
    }

    fn __repr__(&self) -> String {
        format!(
            "GHZStateVerification(n_qubits={}, fidelity={:.6}, blocks={})",
            self.n_qubits, self.fidelity, self.active_blocks
        )
    }

    fn __str__(&self) -> String {
        format!(
            "GHZ|{}>: fidelity={:.6}, |0>={:.4}, |1>={:.4}, blocks={}",
            self.n_qubits,
            self.fidelity,
            self.amp_zero_state,
            self.amp_one_state,
            self.active_blocks
        )
    }
}

impl From<RustGhzVerification> for PyGHZStateVerification {
    fn from(v: RustGhzVerification) -> Self {
        PyGHZStateVerification {
            n_qubits: v.n_qubits,
            amp_zero_state: v.amp_zero_state,
            amp_one_state: v.amp_one_state,
            expected_amplitude: v.expected_amplitude,
            fidelity: v.fidelity,
            max_spurious: v.max_spurious,
            active_blocks: v.active_blocks,
        }
    }
}

/// Fast sparse GHZ state simulation.
///
/// GHZ state: (|00...0> + |11...1>) / sqrt(2)
///
/// Key insight: GHZ has only 2 non-zero amplitudes regardless of qubit count!
/// This means O(1) memory (always 4KB) and O(n) creation time for n qubits.
///
/// NOTE: Maximum 135 qubits due to u128 block ID limitation.
/// For larger GHZ states, use BigInt variant (future feature).
///
/// Performance:
/// | Qubits | Create Time | Memory |
/// |--------|-------------|--------|
/// | 10     | 0.02 ms     | 4 KB   |
/// | 100    | 0.06 ms     | 4 KB   |
/// | 135    | 0.08 ms     | 4 KB   |
///
/// Example:
///     >>> from tileuniverse import FastGHZState
///     >>>
///     >>> # Create 100 qubit GHZ state
///     >>> ghz = FastGHZState(100)
///     >>> print(f"Created in {ghz.creation_time_ms:.3f}ms")
///     >>> print(f"Memory: only {ghz.memory_bytes} bytes!")
///     >>>
///     >>> # Verify fidelity
///     >>> v = ghz.verify()
///     >>> print(f"Fidelity: {v.fidelity:.6f}")
#[pyclass(name = "FastGHZState", unsendable)]
pub struct PyFastGHZState {
    inner: SparseQuantumGrid,
    creation_time_us: u64,
}

#[pymethods]
impl PyFastGHZState {
    /// Create a new GHZ state with the specified number of qubits.
    ///
    /// GHZ state: (|00...0> + |11...1>) / sqrt(2)
    ///
    /// Args:
    ///     n_qubits: Number of qubits (7 to 135)
    ///
    /// Returns:
    ///     FastGHZState object with the GHZ state created
    ///
    /// Raises:
    ///     ValueError: If n_qubits < 7 or n_qubits > 135
    #[new]
    fn new(n_qubits: usize) -> PyResult<Self> {
        if n_qubits < 7 {
            return Err(PyValueError::new_err("n_qubits must be >= 7"));
        }
        if n_qubits > 135 {
            return Err(PyValueError::new_err(
                "n_qubits must be <= 135 (u128 block ID limit). Use BigInt variant for larger states.",
            ));
        }

        let mut grid = SparseQuantumGrid::new(n_qubits);
        let creation_time_us = grid.create_ghz_state();

        Ok(PyFastGHZState {
            inner: grid,
            creation_time_us,
        })
    }

    /// Get the number of qubits
    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Get the number of active memory blocks (should be 2 for GHZ)
    #[getter]
    fn active_blocks(&self) -> usize {
        self.inner.active_blocks()
    }

    /// Get memory usage in bytes (O(1) - constant regardless of qubit count!)
    #[getter]
    fn memory_bytes(&self) -> usize {
        // Each block is 128 * 16 bytes (re + im as f64) = 2KB
        // GHZ state uses exactly 2 blocks = 4KB
        self.inner.active_blocks() * 128 * 16
    }

    /// Get creation time in microseconds
    #[getter]
    fn creation_time_us(&self) -> u64 {
        self.creation_time_us
    }

    /// Get creation time in milliseconds
    #[getter]
    fn creation_time_ms(&self) -> f64 {
        self.creation_time_us as f64 / 1000.0
    }

    /// Get creation time in seconds
    #[getter]
    fn creation_time_sec(&self) -> f64 {
        self.creation_time_us as f64 / 1_000_000.0
    }

    /// Verify the GHZ state fidelity.
    ///
    /// Returns:
    ///     GHZStateVerification with fidelity metrics
    fn verify(&self) -> PyGHZStateVerification {
        self.inner.verify_ghz_fidelity().into()
    }

    /// Get the total probability (should be 1.0)
    fn total_probability(&self) -> f64 {
        self.inner.total_probability()
    }

    fn __repr__(&self) -> String {
        format!(
            "FastGHZState(n_qubits={}, memory={}B, created_in={:.3}ms)",
            self.inner.n_qubits(),
            self.memory_bytes(),
            self.creation_time_ms()
        )
    }

    fn __str__(&self) -> String {
        let n = self.inner.n_qubits();
        let label = if n >= 1_000_000_000 {
            format!("{:.1}B", n as f64 / 1_000_000_000.0)
        } else if n >= 1_000_000 {
            format!("{:.1}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.1}K", n as f64 / 1_000.0)
        } else {
            format!("{}", n)
        };
        format!("|GHZ_{}>", label)
    }
}

/// Create a fast GHZ state (convenience function).
///
/// GHZ state: (|00...0> + |11...1>) / sqrt(2)
///
/// Args:
///     n_qubits: Number of qubits (minimum 7)
///
/// Returns:
///     FastGHZState object
///
/// Example:
///     >>> ghz = create_fast_ghz_state(1_000_000_000)
///     >>> print(ghz)
///     |GHZ_1.0B>
///     >>> print(f"Memory: {ghz.memory_bytes} bytes")  # Only 4KB!
#[pyfunction]
fn create_fast_ghz_state(n_qubits: usize) -> PyResult<PyFastGHZState> {
    PyFastGHZState::new(n_qubits)
}

// ============================================================================
// EPIC 97-style Fused SNN: GPU-resident, deterministic, fire-and-forget
// ============================================================================

/// GPU-accelerated Fused SNN for high-throughput RL.
///
/// This implements the EPIC 97 architecture: all state GPU-resident,
/// zero PCIe transfers per tick, deterministic execution.
///
/// Key properties:
/// - Deterministic: same seed + inputs = identical outputs (bit-for-bit)
/// - GPU-resident: state stays on GPU between decisions
/// - Fire-and-forget: tick_many() queues work without CPU sync
///
/// Example:
///     >>> from tileuniverse import FusedSNN
///     >>> snn = FusedSNN(n_neurons=100000, n_inputs=10000, n_outputs=5, seed=42)
///     >>> snn.set_inputs([128] * 10000)  # spike rates [0-255]
///     >>> for episode in range(1000):
///     ...     obs = env.reset()
///     ...     snn.set_inputs(obs_to_spikes(obs))
///     ...     action = snn.decide(ticks=20)
///     ...     reward = env.step(action)
#[cfg(feature = "cuda")]
#[pyclass(name = "FusedSNN", unsendable)]
pub struct PyFusedSNN {
    inner: RustGpuFusedSNN,
    runtime: engine::cuda::CudaRuntime,
    n_neurons: usize,
    n_inputs: usize,
    n_outputs: usize,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl PyFusedSNN {
    /// Create a new GPU-resident fused SNN.
    ///
    /// Args:
    ///     n_neurons: Total number of neurons
    ///     n_inputs: Number of input neurons (rate-coded)
    ///     n_outputs: Number of output neurons (decision layer)
    ///     connectivity: "dense", "columnar", or "layered" (default: "columnar")
    ///                   - "dense": guaranteed spike propagation (best for small networks)
    ///                   - "columnar": GPU-friendly memory patterns (best for large networks)
    ///                   - "layered": balanced sparse connectivity
    ///     threshold: Firing threshold, Q8.8 format (default: 100)
    ///     leak: Leak factor, Q0.8 format (default: 230 ≈ 0.9)
    ///     seed: Random seed for reproducibility
    #[new]
    #[pyo3(signature = (n_neurons, n_inputs, n_outputs, connectivity="columnar", threshold=100, leak=230, seed=42))]
    fn new(
        n_neurons: usize,
        n_inputs: usize,
        n_outputs: usize,
        connectivity: &str,
        threshold: i16,
        leak: u8,
        seed: u64,
    ) -> PyResult<Self> {
        let runtime = engine::cuda::CudaRuntime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("CUDA init failed: {:?}", e)))?;

        let synapses = match connectivity {
            "dense" => rust_generate_dense_feedforward_csr(n_neurons, n_inputs, n_outputs, seed),
            "columnar" => rust_generate_columnar_csr(
                n_neurons, n_inputs, n_outputs, 256, // column_size = GPU block size
                0.1, // intra_column_density
                5,   // inter_column_synapses
                seed,
            ),
            "direct" => rust_generate_direct_csr(n_neurons, n_inputs, n_outputs, seed),
            "layered" | _ => rust_generate_layered_csr(n_neurons, n_inputs, n_outputs, 20, seed),
        };

        let inner = RustGpuFusedSNN::new(
            &runtime, n_neurons, n_inputs, n_outputs, threshold, leak, &synapses,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("SNN creation failed: {:?}", e)))?;

        Ok(Self {
            inner,
            runtime,
            n_neurons,
            n_inputs,
            n_outputs,
        })
    }

    /// Set input spike rates.
    ///
    /// Args:
    ///     rates: List of spike rates [0-255] for each input neuron.
    ///            Higher values = higher spike probability per tick.
    fn set_inputs(&mut self, rates: Vec<u8>) -> PyResult<()> {
        if rates.len() != self.n_inputs {
            return Err(PyValueError::new_err(format!(
                "Expected {} inputs, got {}",
                self.n_inputs,
                rates.len()
            )));
        }
        self.inner
            .set_input_rates(&self.runtime, &rates)
            .map_err(|e| PyRuntimeError::new_err(format!("set_inputs failed: {:?}", e)))
    }

    /// Run ticks and make a decision (winner-take-all).
    ///
    /// Args:
    ///     ticks: Number of simulation ticks per decision (default: 20)
    ///
    /// Returns:
    ///     int: Action index [0, n_outputs)
    #[pyo3(signature = (ticks=20))]
    fn decide(&mut self, ticks: usize) -> PyResult<usize> {
        self.inner
            .reset_output_counts(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset failed: {:?}", e)))?;

        self.inner
            .tick_many_with_counting(&self.runtime, ticks)
            .map_err(|e| PyRuntimeError::new_err(format!("tick failed: {:?}", e)))?;

        self.inner
            .synchronize(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("sync failed: {:?}", e)))?;

        self.inner
            .get_decision(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("get_decision failed: {:?}", e)))
    }

    /// Get output spike counts from last decision.
    ///
    /// Returns:
    ///     List[int]: Spike counts for each output neuron
    fn get_output_counts(&self) -> PyResult<Vec<u32>> {
        self.inner
            .get_output_counts(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("get_output_counts failed: {:?}", e)))
    }

    /// Run ticks without counting (pure fire-and-forget).
    ///
    /// Use this for maximum throughput when you don't need per-tick output counts.
    ///
    /// Args:
    ///     ticks: Number of ticks to run
    fn tick_many(&mut self, ticks: usize) -> PyResult<()> {
        self.inner
            .tick_many(&self.runtime, ticks)
            .map_err(|e| PyRuntimeError::new_err(format!("tick_many failed: {:?}", e)))
    }

    /// Run ticks with output counting.
    ///
    /// This accumulates output spike counts during the ticks.
    /// Slightly slower than tick_many() but required for decision making.
    ///
    /// Args:
    ///     ticks: Number of ticks to run
    fn tick_many_with_counting(&mut self, ticks: usize) -> PyResult<()> {
        self.inner
            .tick_many_with_counting(&self.runtime, ticks)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("tick_many_with_counting failed: {:?}", e))
            })
    }

    /// Synchronize GPU (wait for pending operations).
    fn sync(&self) -> PyResult<()> {
        self.inner
            .synchronize(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("sync failed: {:?}", e)))
    }

    /// Reset output spike counters.
    fn reset_outputs(&mut self) -> PyResult<()> {
        self.inner
            .reset_output_counts(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset failed: {:?}", e)))
    }

    /// Apply threshold asymmetry to output neurons.
    ///
    /// This breaks the uniform output distribution that results from
    /// symmetric random connectivity.
    ///
    /// Args:
    ///     spread: Range of threshold variation (±spread around base)
    ///     seed: Random seed for asymmetry pattern
    #[pyo3(signature = (spread=30, seed=12345))]
    fn apply_threshold_asymmetry(&mut self, spread: i16, seed: u64) -> PyResult<()> {
        self.inner
            .apply_output_threshold_asymmetry(&self.runtime, 100, spread, seed)
            .map_err(|e| PyRuntimeError::new_err(format!("threshold asymmetry failed: {:?}", e)))
    }

    /// Get network statistics.
    fn stats(&self) -> PyResult<(usize, usize, usize, u64)> {
        Ok((
            self.n_neurons,
            self.n_inputs,
            self.n_outputs,
            self.inner.current_tick(),
        ))
    }

    #[getter]
    fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    #[getter]
    fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    #[getter]
    fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    #[getter]
    fn tick(&self) -> u64 {
        self.inner.current_tick()
    }

    fn __repr__(&self) -> String {
        format!(
            "FusedSNN(neurons={}, inputs={}, outputs={}, tick={})",
            self.n_neurons,
            self.n_inputs,
            self.n_outputs,
            self.inner.current_tick()
        )
    }

    // ========================================================================
    // Gymnasium-compatible RL interface
    // ========================================================================

    /// Reset the network for a new episode.
    ///
    /// This resets neuron membrane potentials and output spike counters,
    /// preparing for a fresh episode.
    fn reset(&mut self) -> PyResult<()> {
        self.inner
            .reset_output_counts(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset failed: {:?}", e)))?;
        self.inner
            .reset_state(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset state failed: {:?}", e)))?;
        Ok(())
    }

    /// Process one RL step: observation -> action.
    ///
    /// This is the main interface for Gymnasium integration:
    /// 1. Set input spike rates from observation
    /// 2. Run the network for specified ticks
    /// 3. Return the winning output neuron as the action
    ///
    /// Args:
    ///     observation: List of floats (will be scaled to [0-255] spike rates)
    ///     ticks: Number of simulation ticks per decision (default: 20)
    ///
    /// Returns:
    ///     int: The action (output neuron with most spikes)
    #[pyo3(signature = (observation, ticks=20))]
    fn step(&mut self, observation: Vec<f32>, ticks: usize) -> PyResult<usize> {
        // Scale observation to spike rates [0, 255]
        // Assumes observation values are roughly in [-inf, inf], normalize to [0, 255]
        let rates: Vec<u8> = observation
            .iter()
            .map(|&v| {
                // Sigmoid-like scaling: map any float to [0, 255]
                let scaled = (v.tanh() + 1.0) * 127.5;
                scaled.clamp(0.0, 255.0) as u8
            })
            .collect();

        if rates.len() != self.n_inputs {
            return Err(PyValueError::new_err(format!(
                "Observation length {} doesn't match n_inputs {}",
                rates.len(),
                self.n_inputs
            )));
        }

        // Set inputs
        self.inner
            .set_input_rates(&self.runtime, &rates)
            .map_err(|e| PyRuntimeError::new_err(format!("set_inputs failed: {:?}", e)))?;

        // Reset counters and run with counting
        self.inner
            .reset_output_counts(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset_outputs failed: {:?}", e)))?;
        self.inner
            .tick_many_with_counting(&self.runtime, ticks)
            .map_err(|e| PyRuntimeError::new_err(format!("tick failed: {:?}", e)))?;
        self.inner
            .synchronize(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("sync failed: {:?}", e)))?;

        // Get decision
        let action = self
            .inner
            .get_decision(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("get_decision failed: {:?}", e)))?;
        Ok(action)
    }

    /// Batch step for vectorized environments.
    ///
    /// Process multiple observations in parallel (batch inference).
    /// Note: Currently runs sequentially - for full parallelism, use
    /// multiple FusedSNN instances.
    ///
    /// Args:
    ///     observations: List of observations, each a list of floats
    ///     ticks: Number of simulation ticks per decision
    ///
    /// Returns:
    ///     List[int]: Actions for each observation
    #[pyo3(signature = (observations, ticks=20))]
    fn step_batch(&mut self, observations: Vec<Vec<f32>>, ticks: usize) -> PyResult<Vec<usize>> {
        let mut actions = Vec::with_capacity(observations.len());
        for obs in observations {
            actions.push(self.step(obs, ticks)?);
        }
        Ok(actions)
    }

    // ========================================================================
    // R-STDP Learning Interface
    // ========================================================================

    /// Enable R-STDP learning.
    ///
    /// When learning is enabled:
    /// - Eligibility traces are updated after each tick
    /// - learn() applies reward-modulated weight updates
    fn enable_learning(&mut self) {
        self.inner.enable_learning();
    }

    /// Disable R-STDP learning.
    fn disable_learning(&mut self) {
        self.inner.disable_learning();
    }

    /// Check if learning is enabled.
    #[getter]
    fn learning_enabled(&self) -> bool {
        self.inner.is_learning_enabled()
    }

    /// Set R-STDP learning parameters.
    ///
    /// Args:
    ///     tau_decay: Eligibility decay factor per tick (default: 0.95)
    ///     ltp_rate: LTP increment when pre+post spike (default: 0.1)
    ///     ltd_rate: LTD decrement when pre without post (default: 0.05)
    ///     learning_rate: Weight update rate (default: 0.01)
    #[pyo3(signature = (tau_decay=0.95, ltp_rate=0.1, ltd_rate=0.05, learning_rate=0.01))]
    fn set_learning_params(
        &mut self,
        tau_decay: f32,
        ltp_rate: f32,
        ltd_rate: f32,
        learning_rate: f32,
    ) {
        self.inner
            .set_learning_params(tau_decay, ltp_rate, ltd_rate, learning_rate);
    }

    /// Enable timing-based STDP (three-factor learning).
    ///
    /// When enabled, eligibility traces depend on the relative timing of
    /// pre and post spikes, not just co-occurrence. This creates differentiated
    /// eligibility traces that can solve the "cold start" problem where all
    /// outputs fire uniformly.
    ///
    /// This is the recommended mode for learning from scratch.
    fn enable_timing_stdp(&mut self) {
        self.inner.enable_timing_stdp();
    }

    /// Disable timing-based STDP (use legacy co-occurrence rule).
    fn disable_timing_stdp(&mut self) {
        self.inner.disable_timing_stdp();
    }

    /// Check if timing-based STDP is enabled.
    fn is_timing_stdp_enabled(&self) -> bool {
        self.inner.is_timing_stdp_enabled()
    }

    /// Set STDP timing parameters.
    ///
    /// Args:
    ///     tau_plus: LTP time constant in ticks (default: 20.0)
    ///     tau_minus: LTD time constant in ticks (default: 20.0)
    ///     window: Maximum timing window in ticks (default: 30)
    #[pyo3(signature = (tau_plus=20.0, tau_minus=20.0, window=30))]
    fn set_stdp_timing_params(&mut self, tau_plus: f32, tau_minus: f32, window: u32) {
        self.inner
            .set_stdp_timing_params(tau_plus, tau_minus, window);
    }

    /// Reset spike times (call at episode start).
    fn reset_spike_times(&mut self) -> PyResult<()> {
        self.inner
            .reset_spike_times(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset_spike_times failed: {:?}", e)))
    }

    /// Apply reward signal to update weights (R-STDP).
    ///
    /// This applies reward-modulated weight updates using accumulated
    /// eligibility traces. Call after each step to provide feedback.
    ///
    /// Args:
    ///     reward: Scalar reward signal (typically -1 to 1, but any float works)
    fn learn(&mut self, reward: f32) -> PyResult<()> {
        self.inner
            .apply_reward(&self.runtime, reward)
            .map_err(|e| PyRuntimeError::new_err(format!("learn failed: {:?}", e)))
    }

    /// Apply reward only to synapses targeting the chosen action.
    ///
    /// This is action-specific learning: only updates weights for synapses
    /// that directly contributed to the chosen action. Much better for RL
    /// than learn() which updates all synapses uniformly.
    ///
    /// Args:
    ///     action: The chosen action (0 to n_outputs-1)
    ///     reward: Reward signal for the chosen action
    ///     anti_reward_factor: Factor for non-chosen outputs (default: -0.5)
    ///                         -0.5 means losers get opposite reward
    #[pyo3(signature = (action, reward, anti_reward_factor=-0.5))]
    fn learn_action(
        &mut self,
        action: usize,
        reward: f32,
        anti_reward_factor: f32,
    ) -> PyResult<()> {
        self.inner
            .apply_reward_action_specific(&self.runtime, action, reward, anti_reward_factor)
            .map_err(|e| PyRuntimeError::new_err(format!("learn_action failed: {:?}", e)))
    }

    /// Apply reward with pathway-based credit assignment.
    ///
    /// Updates all synapses but scales reward based on contribution:
    /// - Direct to winner: full reward
    /// - Direct to loser: -0.3 * reward
    /// - Hidden layer: 0.1 * reward
    ///
    /// This provides softer credit assignment than learn_action().
    fn learn_pathway(&mut self, action: usize, reward: f32) -> PyResult<()> {
        self.inner
            .apply_reward_pathway(&self.runtime, action, reward)
            .map_err(|e| PyRuntimeError::new_err(format!("learn_pathway failed: {:?}", e)))
    }

    /// Reset eligibility traces to zero.
    ///
    /// Call at the start of each episode.
    fn reset_eligibility(&mut self) -> PyResult<()> {
        self.inner
            .reset_eligibility(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset_eligibility failed: {:?}", e)))
    }

    /// Process one RL step with learning.
    ///
    /// Combines step() with eligibility accumulation. Use this instead of
    /// step() when learning is enabled for better performance.
    ///
    /// Args:
    ///     observation: List of floats (will be scaled to [0-255] spike rates)
    ///     ticks: Number of simulation ticks per decision (default: 20)
    ///
    /// Returns:
    ///     int: The action (output neuron with most spikes)
    #[pyo3(signature = (observation, ticks=20))]
    fn step_learn(&mut self, observation: Vec<f32>, ticks: usize) -> PyResult<usize> {
        // Scale observation to spike rates [0, 255]
        let rates: Vec<u8> = observation
            .iter()
            .map(|&v| {
                let scaled = (v.tanh() + 1.0) * 127.5;
                scaled.clamp(0.0, 255.0) as u8
            })
            .collect();

        if rates.len() != self.n_inputs {
            return Err(PyValueError::new_err(format!(
                "Observation length {} doesn't match n_inputs {}",
                rates.len(),
                self.n_inputs
            )));
        }

        // Set inputs
        self.inner
            .set_input_rates(&self.runtime, &rates)
            .map_err(|e| PyRuntimeError::new_err(format!("set_inputs failed: {:?}", e)))?;

        // Reset counters and run with counting + learning
        self.inner
            .reset_output_counts(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset_outputs failed: {:?}", e)))?;
        self.inner
            .tick_many_with_counting_and_learning(&self.runtime, ticks)
            .map_err(|e| PyRuntimeError::new_err(format!("tick failed: {:?}", e)))?;
        self.inner
            .synchronize(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("sync failed: {:?}", e)))?;

        // Get decision
        let action = self
            .inner
            .get_decision(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("get_decision failed: {:?}", e)))?;
        Ok(action)
    }

    /// Process one RL step with learning using pre-computed spike rates.
    ///
    /// Use this when you need precise control over input encoding (e.g., for
    /// MountainCar where velocity needs amplification).
    ///
    /// Args:
    ///     rates: List of spike rates [0-255] for each input
    ///     ticks: Number of simulation ticks per decision (default: 20)
    ///
    /// Returns:
    ///     int: The action (output neuron with most spikes)
    #[pyo3(signature = (rates, ticks=20))]
    fn step_learn_rates(&mut self, rates: Vec<u8>, ticks: usize) -> PyResult<usize> {
        if rates.len() != self.n_inputs {
            return Err(PyValueError::new_err(format!(
                "Rates length {} doesn't match n_inputs {}",
                rates.len(),
                self.n_inputs
            )));
        }

        // Set inputs
        self.inner
            .set_input_rates(&self.runtime, &rates)
            .map_err(|e| PyRuntimeError::new_err(format!("set_inputs failed: {:?}", e)))?;

        // Reset counters and run with counting + learning
        self.inner
            .reset_output_counts(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset_outputs failed: {:?}", e)))?;
        self.inner
            .tick_many_with_counting_and_learning(&self.runtime, ticks)
            .map_err(|e| PyRuntimeError::new_err(format!("tick failed: {:?}", e)))?;
        self.inner
            .synchronize(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("sync failed: {:?}", e)))?;

        // Get decision
        let action = self
            .inner
            .get_decision(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("get_decision failed: {:?}", e)))?;
        Ok(action)
    }
}

// =============================================================================
// Milestone 5 Phase 3: BatchSNN — GPU-batched SNN with ALIF + E-prop
// =============================================================================

/// GPU-batched spiking neural network that runs B independent instances in parallel.
///
/// Supports discriminative-channel initialization (Fisher LDA priors),
/// Hebbian STDP, ALIF adaptive thresholds, and GPU E-prop training.
///
/// Example:
///     >>> from tileuniverse import BatchSNN
///     >>> snn = BatchSNN.from_discriminative(10, 150, 32, d_norms, t_per_class, batch_size=256)
///     >>> snn.upload_input_batch(rates)  # flat list[int], len = batch_size × n_inputs
///     >>> snn.reset_output_counts()
///     >>> snn.tick_many_with_counting(100)
///     >>> snn.synchronize()
///     >>> counts = snn.get_output_counts()  # numpy array [batch_size, n_outputs]
#[cfg(feature = "cuda")]
#[pyclass(name = "BatchSNN", unsendable)]
pub struct PyBatchSNN {
    inner: RustGpuBatchSNN,
    runtime: engine::cuda::CudaRuntime,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl PyBatchSNN {
    // ── Constructors ────────────────────────────────────────────────────────

    /// Build a discriminative-channel BatchSNN (Fisher LDA weight initialisation).
    ///
    /// Network layout: [k_per_class × n_classes input neurons]
    ///               → [hidden_per_class × n_classes hidden/output neurons]
    ///
    /// Args:
    ///     n_classes:        Number of output classes (e.g. 10 for MNIST)
    ///     k_per_class:      Discriminative input pixels per class (e.g. 150)
    ///     hidden_per_class: Hidden neurons per class — also the readout layer (e.g. 32)
    ///     d_norms_flat:     Fisher LDA scores, flat list[float] len = n_classes × k_per_class
    ///     t_per_class:      Per-class hidden-layer threshold, list[int] len = n_classes
    ///     batch_size:       Parallel batch size B (default 256)
    ///     seed:             Random seed (default 42)
    ///     connection_prob:  Input→hidden connection probability (default 0.5)
    ///     w_max:            Max weight for discriminative synapses Q1.7 (default 64)
    ///     input_threshold:  Threshold for input neurons (default 32000)
    ///     leak:             Leak factor 0–255 applied to all neurons (default 230 ≈ 0.9)
    #[staticmethod]
    #[pyo3(signature = (
        n_classes, k_per_class, hidden_per_class,
        d_norms_flat, t_per_class,
        batch_size=256, seed=42,
        connection_prob=0.5, w_max=64,
        input_threshold=32000, leak=230
    ))]
    fn from_discriminative(
        n_classes: usize,
        k_per_class: usize,
        hidden_per_class: usize,
        d_norms_flat: Vec<f32>,
        t_per_class: Vec<i16>,
        batch_size: usize,
        seed: u64,
        connection_prob: f32,
        w_max: i8,
        input_threshold: i16,
        leak: u8,
    ) -> PyResult<Self> {
        let runtime = engine::cuda::CudaRuntime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("CUDA init failed: {:?}", e)))?;

        let synapses = rust_gen_discrim_csr_weighted(
            k_per_class,
            hidden_per_class,
            n_classes,
            connection_prob,
            w_max,
            &d_norms_flat,
            seed,
        );
        let thresholds = rust_discrim_thresholds(
            k_per_class,
            hidden_per_class,
            n_classes,
            &t_per_class,
            input_threshold,
        );

        let n_inputs = k_per_class * n_classes;
        let n_outputs = hidden_per_class * n_classes;
        let n_neurons = n_inputs + n_outputs;
        let leaks = vec![leak; n_neurons];

        let inner = RustGpuBatchSNN::new(
            &runtime,
            n_neurons,
            n_inputs,
            n_outputs,
            &thresholds,
            &leaks,
            &synapses,
            batch_size,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("BatchSNN creation failed: {:?}", e)))?;

        Ok(Self { inner, runtime })
    }

    /// Build a BatchSNN with random sparse connectivity.
    ///
    /// Args:
    ///     n_neurons:           Total neuron count
    ///     n_inputs:            Input neurons (rate-coded, first n_inputs neurons)
    ///     n_outputs:           Output neurons (readout, last n_outputs neurons)
    ///     synapses_per_neuron: Random fan-out per neuron (default 10)
    ///     batch_size:          Parallel batch size B (default 256)
    ///     seed:                Random seed (default 42)
    ///     threshold:           Firing threshold Q8.8 (default 100)
    ///     leak:                Leak factor 0–255 (default 230 ≈ 0.9)
    #[staticmethod]
    #[pyo3(signature = (
        n_neurons, n_inputs, n_outputs,
        synapses_per_neuron=10,
        batch_size=256, seed=42,
        threshold=100, leak=230
    ))]
    fn from_random(
        n_neurons: usize,
        n_inputs: usize,
        n_outputs: usize,
        synapses_per_neuron: usize,
        batch_size: usize,
        seed: u64,
        threshold: i16,
        leak: u8,
    ) -> PyResult<Self> {
        let runtime = engine::cuda::CudaRuntime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("CUDA init failed: {:?}", e)))?;

        let synapses = rust_gen_random_csr(n_neurons, synapses_per_neuron, seed);
        let thresholds = vec![threshold; n_neurons];
        let leaks = vec![leak; n_neurons];

        let inner = RustGpuBatchSNN::new(
            &runtime,
            n_neurons,
            n_inputs,
            n_outputs,
            &thresholds,
            &leaks,
            &synapses,
            batch_size,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("BatchSNN creation failed: {:?}", e)))?;

        Ok(Self { inner, runtime })
    }

    // ── Forward pass ────────────────────────────────────────────────────────

    /// Upload input spike rates for the current batch.
    ///
    /// Args:
    ///     rates: flat list[int] of length batch_size × n_inputs (row-major).
    ///            rates[b * n_inputs + i] is rate [0-255] for instance b, input i.
    fn upload_input_batch(&mut self, rates: Vec<u8>) -> PyResult<()> {
        self.inner
            .upload_input_batch(&self.runtime, &rates)
            .map_err(|e| PyRuntimeError::new_err(format!("upload_input_batch failed: {:?}", e)))
    }

    /// Run n_ticks without output counting (fastest path).
    fn tick_many(&mut self, n_ticks: usize) -> PyResult<()> {
        self.inner
            .tick_many(&self.runtime, n_ticks)
            .map_err(|e| PyRuntimeError::new_err(format!("tick_many failed: {:?}", e)))
    }

    /// Run n_ticks with per-batch output spike accumulation.
    fn tick_many_with_counting(&mut self, n_ticks: usize) -> PyResult<()> {
        self.inner
            .tick_many_with_counting(&self.runtime, n_ticks)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("tick_many_with_counting failed: {:?}", e))
            })
    }

    /// Reset neuron membrane potentials and refractory counters for all batch instances.
    fn reset_state(&self) -> PyResult<()> {
        self.inner
            .reset_state(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset_state failed: {:?}", e)))
    }

    /// Reset output spike counters to zero for all batch instances.
    fn reset_output_counts(&self) -> PyResult<()> {
        self.inner
            .reset_output_counts(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset_output_counts failed: {:?}", e)))
    }

    /// Block until all pending GPU kernels have completed.
    fn synchronize(&self) -> PyResult<()> {
        self.inner
            .synchronize(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("synchronize failed: {:?}", e)))
    }

    // ── Output ──────────────────────────────────────────────────────────────

    /// Return output spike counts as a numpy array of shape [batch_size, n_outputs].
    fn get_output_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let flat = self
            .inner
            .get_output_counts_flat(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("get_output_counts failed: {:?}", e)))?;
        let arr = ndarray::Array2::from_shape_vec(
            (self.inner.batch_size(), self.inner.n_outputs()),
            flat,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("reshape failed: {:?}", e)))?;
        Ok(arr.into_pyarray_bound(py))
    }

    // ── Hebbian STDP ────────────────────────────────────────────────────────

    /// Allocate full per-neuron spike-count buffers needed for Hebbian updates.
    fn enable_stdp(&mut self) -> PyResult<()> {
        self.inner
            .enable_stdp(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("enable_stdp failed: {:?}", e)))
    }

    /// Reset all STDP spike-count buffers (call before each episode).
    fn reset_all_counts(&self) -> PyResult<()> {
        self.inner
            .reset_all_counts(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset_all_counts failed: {:?}", e)))
    }

    /// Run n_ticks with full per-neuron spike counting (required before apply_hebbian_reward).
    fn tick_many_full_counting(&mut self, n_ticks: usize) -> PyResult<()> {
        self.inner
            .tick_many_full_counting(&self.runtime, n_ticks)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("tick_many_full_counting failed: {:?}", e))
            })
    }

    /// Apply reward-modulated Hebbian weight update.
    ///
    /// Args:
    ///     rewards:  list[float] of length batch_size (+1.0 correct, -1.0 wrong)
    ///     n_ticks:  episode tick count (used for firing-rate normalisation)
    ///     lr:       learning rate (e.g. 0.001)
    fn apply_hebbian_reward(&mut self, rewards: Vec<f32>, n_ticks: usize, lr: f32) -> PyResult<()> {
        self.inner
            .apply_hebbian_reward(&self.runtime, &rewards, n_ticks, lr)
            .map_err(|e| PyRuntimeError::new_err(format!("apply_hebbian_reward failed: {:?}", e)))
    }

    /// Download current synapse weights as a flat list of i8 values.
    fn get_weights(&self) -> PyResult<Vec<i8>> {
        self.inner
            .get_weights(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("get_weights failed: {:?}", e)))
    }

    // ── ALIF ─────────────────────────────────────────────────────────────────

    /// Enable ALIF adaptive threshold. Call once before inference or training.
    ///
    /// Args:
    ///     alpha_adapt: Spike-history decay coefficient (e.g. 0.967 = exp(-1/30))
    ///     beta_adapt:  Adaptive threshold scale (e.g. 0.1)
    fn enable_alif(&mut self, alpha_adapt: f32, beta_adapt: f32) -> PyResult<()> {
        self.inner
            .enable_alif(&self.runtime, alpha_adapt, beta_adapt)
            .map_err(|e| PyRuntimeError::new_err(format!("enable_alif failed: {:?}", e)))
    }

    /// Zero the ALIF adaptation state (call between episodes during training).
    fn reset_adapt_state(&mut self) -> PyResult<()> {
        self.inner
            .reset_adapt_state(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset_adapt_state failed: {:?}", e)))
    }

    // ── E-prop ───────────────────────────────────────────────────────────────

    /// Enable GPU E-prop eligibility traces. Call after enable_alif() if using ALIF.
    ///
    /// Args:
    ///     alpha_pre:  Pre-synaptic trace decay per tick (e.g. 0.95)
    ///     alpha_elig: Eligibility trace decay per tick (e.g. 0.95)
    ///     gamma:      Surrogate gradient sharpness — fast-sigmoid ψ = 1/(1+γ|v/θ-1|)² (e.g. 0.3)
    fn enable_eprop(&mut self, alpha_pre: f32, alpha_elig: f32, gamma: f32) -> PyResult<()> {
        self.inner
            .enable_eprop(&self.runtime, alpha_pre, alpha_elig, gamma)
            .map_err(|e| PyRuntimeError::new_err(format!("enable_eprop failed: {:?}", e)))
    }

    /// Zero pre-synaptic trace and eligibility buffers (call between episodes).
    fn reset_eprop_state(&mut self) -> PyResult<()> {
        self.inner
            .reset_eprop_state(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("reset_eprop_state failed: {:?}", e)))
    }

    /// Run n_ticks with LIF dynamics + output counting + pre-trace + eligibility update.
    ///
    /// Use this instead of tick_many_with_counting() during E-prop training episodes.
    fn tick_many_with_eprop(&mut self, n_ticks: usize) -> PyResult<()> {
        self.inner
            .tick_many_with_eprop(&self.runtime, n_ticks)
            .map_err(|e| PyRuntimeError::new_err(format!("tick_many_with_eprop failed: {:?}", e)))
    }

    /// Apply E-prop weight update at end of episode.
    ///
    /// Args:
    ///     learning_signals: list[float] of length batch_size (+1.0 correct, -1.0 wrong)
    ///     lr:               learning rate (e.g. 0.0005)
    fn apply_eprop_update(&mut self, learning_signals: Vec<f32>, lr: f32) -> PyResult<()> {
        self.inner
            .apply_eprop_update(&self.runtime, &learning_signals, lr)
            .map_err(|e| PyRuntimeError::new_err(format!("apply_eprop_update failed: {:?}", e)))
    }

    /// Return (mean_abs_eligibility, max_abs_eligibility) for convergence monitoring.
    ///
    /// Call every N epochs to verify eligibility traces are accumulating.
    fn get_eligibility_stats(&self) -> PyResult<(f32, f32)> {
        self.inner
            .get_eligibility_stats(&self.runtime)
            .map_err(|e| PyRuntimeError::new_err(format!("get_eligibility_stats failed: {:?}", e)))
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    #[getter]
    fn n_neurons(&self) -> usize {
        self.inner.n_neurons()
    }

    #[getter]
    fn n_inputs(&self) -> usize {
        self.inner.n_inputs()
    }

    #[getter]
    fn n_outputs(&self) -> usize {
        self.inner.n_outputs()
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchSNN(neurons={}, inputs={}, outputs={}, batch={})",
            self.inner.n_neurons(),
            self.inner.n_inputs(),
            self.inner.n_outputs(),
            self.inner.batch_size(),
        )
    }
}

// =============================================================================
// E-prop: Eligibility Propagation Learning (CPU-based, can learn from scratch)
// =============================================================================

use engine::snn::eprop::{
    EpropConfig as RustEpropConfig, EpropNetwork as RustEpropNetwork,
    SurrogateGradient as RustSurrogateGradient,
};

/// Surrogate gradient function for e-prop learning.
///
/// These functions approximate the derivative of the spike function
/// (which is technically undefined) to enable gradient-based learning.
#[pyclass(name = "SurrogateGradient")]
#[derive(Clone)]
pub struct PySurrogateGradient {
    inner: RustSurrogateGradient,
}

#[pymethods]
impl PySurrogateGradient {
    /// Create a piecewise linear surrogate gradient.
    ///
    /// Args:
    ///     gamma: Width of the gradient window (default: 0.3)
    #[staticmethod]
    #[pyo3(signature = (gamma=0.3))]
    fn piecewise_linear(gamma: f32) -> Self {
        Self {
            inner: RustSurrogateGradient::PiecewiseLinear { gamma },
        }
    }

    /// Create a fast sigmoid surrogate gradient.
    ///
    /// Args:
    ///     gamma: Steepness parameter (default: 0.3)
    #[staticmethod]
    #[pyo3(signature = (gamma=0.3))]
    fn fast_sigmoid(gamma: f32) -> Self {
        Self {
            inner: RustSurrogateGradient::FastSigmoid { gamma },
        }
    }

    /// Create an exponential surrogate gradient.
    ///
    /// Args:
    ///     gamma: Decay parameter (default: 0.3)
    #[staticmethod]
    #[pyo3(signature = (gamma=0.3))]
    fn exponential(gamma: f32) -> Self {
        Self {
            inner: RustSurrogateGradient::Exponential { gamma },
        }
    }
}

/// E-prop: Eligibility Propagation SNN
///
/// Unlike R-STDP, e-prop can learn from scratch without pre-training.
/// Based on Bellec et al. (2020) "A solution to the learning dilemma
/// for recurrent networks of spiking neurons."
///
/// Key differences from R-STDP:
/// - Uses surrogate gradients for credit assignment
/// - Eligibility traces track gradient information, not just spike correlations
/// - Can assign credit even when neurons don't spike (if they were close to threshold)
///
/// Example:
///     >>> from tileuniverse import EpropSNN
///     >>> snn = EpropSNN(n_inputs=4, hidden_layers=[32], n_outputs=3)
///     >>> action = snn.decide([200, 100, 50, 25], ticks=20)
///     >>> snn.learn(reward=1.0)  # Positive reinforcement
#[pyclass(name = "EpropSNN")]
pub struct PyEpropSNN {
    inner: RustEpropNetwork,
    n_inputs: usize,
    n_outputs: usize,
}

#[pymethods]
impl PyEpropSNN {
    /// Create a new e-prop SNN.
    ///
    /// Args:
    ///     n_inputs: Number of input neurons
    ///     hidden_layers: List of hidden layer sizes (e.g., [32, 16])
    ///     n_outputs: Number of output neurons (actions)
    ///     tau_mem: Membrane time constant in ms (default: 20.0)
    ///     tau_eligibility: Eligibility trace time constant in ms (default: 100.0)
    ///     learning_rate: Learning rate (default: 0.01)
    ///     adaptive_threshold: Use ALIF neurons with adaptive thresholds (default: False)
    ///     seed: Random seed for weight initialization
    #[new]
    #[pyo3(signature = (n_inputs, hidden_layers, n_outputs, tau_mem=20.0, tau_eligibility=100.0, learning_rate=0.01, adaptive_threshold=false, seed=42))]
    fn new(
        n_inputs: usize,
        hidden_layers: Vec<usize>,
        n_outputs: usize,
        tau_mem: f32,
        tau_eligibility: f32,
        learning_rate: f32,
        adaptive_threshold: bool,
        seed: u64,
    ) -> Self {
        let config = RustEpropConfig {
            n_inputs,
            hidden_layers,
            n_outputs,
            tau_mem,
            tau_eligibility,
            learning_rate,
            adaptive_threshold,
            ..Default::default()
        };

        let inner = RustEpropNetwork::with_seed(config, seed);

        Self {
            inner,
            n_inputs,
            n_outputs,
        }
    }

    /// Make a decision based on input spike rates.
    ///
    /// Args:
    ///     rates: List of spike rates [0-255] for each input
    ///     ticks: Number of simulation ticks (default: 20)
    ///
    /// Returns:
    ///     int: Action index (output with most spikes)
    #[pyo3(signature = (rates, ticks=20))]
    fn decide(&mut self, rates: Vec<u8>, ticks: usize) -> PyResult<usize> {
        if rates.len() != self.n_inputs {
            return Err(PyValueError::new_err(format!(
                "Expected {} inputs, got {}",
                self.n_inputs,
                rates.len()
            )));
        }
        Ok(self.inner.decide(&rates, ticks))
    }

    /// Apply learning with reward signal.
    ///
    /// Uses policy gradient: ΔW = η * reward * eligibility_trace
    ///
    /// Args:
    ///     reward: Reward signal (positive = reinforce, negative = punish)
    fn learn(&mut self, reward: f32) {
        self.inner.learn(reward);
    }

    /// Apply action-specific learning.
    ///
    /// Only updates weights connected to the specified action's output.
    ///
    /// Args:
    ///     action: The action to reinforce/punish
    ///     reward: Reward signal
    fn learn_action(&mut self, action: usize, reward: f32) {
        self.inner.learn_action(action, reward);
    }

    /// Apply supervised learning toward target action.
    ///
    /// Directly strengthens connection to target action based on hidden
    /// layer activity. More reliable than reward-based learning for
    /// classification tasks.
    ///
    /// Args:
    ///     target_action: The correct action to reinforce
    fn learn_supervised(&mut self, target_action: usize) {
        self.inner.learn_supervised(target_action);
    }

    /// Reset eligibility traces.
    ///
    /// Call this at the start of each episode.
    fn reset_eligibility(&mut self) {
        self.inner.reset_eligibility();
    }

    /// Reset network state (membrane potentials, spikes).
    ///
    /// Call this at the start of each episode.
    fn reset_state(&mut self) {
        self.inner.reset_state();
    }

    /// Full reset (state + eligibility).
    fn reset(&mut self) {
        self.inner.reset_state();
        self.inner.reset_eligibility();
    }

    /// Get output spike counts from last decision.
    fn get_output_counts(&self) -> Vec<u32> {
        self.inner.output_counts().to_vec()
    }

    #[getter]
    fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    #[getter]
    fn n_outputs(&self) -> usize {
        self.n_outputs
    }
}

/// TileUniverse: GPU-accelerated parallel universe simulator
/// with quantum-classical hybrid computation support.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classical parallel worlds
    m.add_class::<Engine>()?;
    m.add_class::<BenchmarkResult>()?;

    // SPRINT 2.2: Quantum-classical hybrid
    m.add_class::<PyQGate>()?;
    m.add_class::<PyQuantumSimulation>()?;

    // SPRINT 2.3: Deutsch-Jozsa and Bernstein-Vazirani
    m.add_class::<PyDJOracle>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(is_cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_device_name, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark, m)?)?;
    m.add_function(wrap_pyfunction!(load_config, m)?)?;

    // SPRINT 2.2: Quantum algorithm helpers
    m.add_function(wrap_pyfunction!(grover_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_iterations, m)?)?;

    // SPRINT 2.3: DJ and BV algorithm helpers
    m.add_function(wrap_pyfunction!(deutsch_jozsa_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(bernstein_vazirani_circuit, m)?)?;

    // SNN bindings: Quantum-SNN hybrid
    m.add_class::<PyInterferenceMode>()?;
    m.add_class::<PyQuantumSNNConfig>()?;
    m.add_class::<PyQuantumSNN>()?;

    // Sprint 59.0: Stabilizer-based quantum neural networks
    m.add_class::<PyFeedbackMode>()?;
    m.add_class::<PyBridgeGate>()?;
    m.add_class::<PyStabilizerSNN>()?;
    m.add_class::<PyHybridQuantumSNN>()?;

    // EPIC 97-style Fused SNN (GPU-resident, deterministic)
    #[cfg(feature = "cuda")]
    m.add_class::<PyFusedSNN>()?;

    // Milestone 5: Batch GPU SNN with ALIF + E-prop
    #[cfg(feature = "cuda")]
    m.add_class::<PyBatchSNN>()?;

    // E-prop: Eligibility Propagation (CPU-based, can learn from scratch)
    m.add_class::<PySurrogateGradient>()?;
    m.add_class::<PyEpropSNN>()?;

    // Fast W state bindings (Vec-based, scales to billions of qubits)
    m.add_class::<PyWStateVerification>()?;
    m.add_class::<PyFastWState>()?;
    m.add_function(wrap_pyfunction!(create_fast_w_state, m)?)?;

    // Fast GHZ state bindings (O(1) sparse - only 2 amplitudes)
    m.add_class::<PyGHZStateVerification>()?;
    m.add_class::<PyFastGHZState>()?;
    m.add_function(wrap_pyfunction!(create_fast_ghz_state, m)?)?;

    // Sprint 212: V2 CPU + Synth
    v2_cpu::register(m)?;
    v2_synth::register(m)?;
    // Sprint 214: Sequential circuits
    v2_seq::register(m)?;

    Ok(())
}
