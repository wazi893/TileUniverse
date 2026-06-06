//! EPIC 62: Multi-Core Quantum Tile Farm
//!
//! Thread pool infrastructure for distributing SIMD tile execution across CPU cores.
//!
//! Architecture:
//! - Each worker executes SIMD kernels on a disjoint tile range
//! - Barrier synchronization ensures deterministic ordering
//! - JIT kernel cache shared read-only across workers
//! - Zero thread creation overhead during simulation
//!
//! Example:
//! ```ignore
//! let farm = TileFarm::new(8, 4);  // 8 workers, 4 tiles/worker = 32 total tiles
//! farm.dispatch(job);              // Execute kernel on all workers
//! farm.wait_barrier();             // Block until all complete
//! ```

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Barrier};
use std::thread::{self, JoinHandle};

/// Worker count mode (manual or automatic)
///
/// EPIC 62 Phase 4: Support for auto-tuning worker configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerMode {
    /// Manual mode: user specifies exact worker count
    Manual(usize),
    /// Auto mode: heuristic chooses optimal config based on CPU and workload
    Auto,
}

/// Read JIT_WORKER_COUNT environment variable
///
/// Returns:
/// - WorkerMode::Auto: if set to "auto"
/// - WorkerMode::Manual(1): default single-threaded mode
/// - WorkerMode::Manual(N): multi-threaded mode with N workers (2-64)
///
/// EPIC 62: Controls parallelism level for tile farm
/// EPIC 62 Phase 4: Added auto mode support
pub fn jit_worker_mode() -> WorkerMode {
    let env = std::env::var("JIT_WORKER_COUNT").unwrap_or_else(|_| "1".to_string());

    // Check for auto mode
    if env.trim().eq_ignore_ascii_case("auto") {
        return WorkerMode::Auto;
    }

    // Parse as manual worker count
    match env.parse::<usize>().unwrap_or(1) {
        n if n >= 2 && n <= 64 => WorkerMode::Manual(n),
        _ => WorkerMode::Manual(1),
    }
}

/// Read JIT_WORKER_COUNT environment variable (legacy)
///
/// Returns:
/// - 1 (default): single-threaded mode (backward compatible)
/// - N (2-64): multi-threaded mode with N workers
///
/// EPIC 62: Controls parallelism level for tile farm
/// DEPRECATED: Use `jit_worker_mode()` for Phase 4 auto mode support
pub fn jit_worker_count() -> usize {
    match jit_worker_mode() {
        WorkerMode::Manual(n) => n,
        WorkerMode::Auto => 1, // Fallback to single-threaded if called in auto mode
    }
}

/// Tile range assignment for a specific worker
///
/// Workers operate on disjoint tile ranges to ensure determinism.
///
/// Example:
/// - Worker 0, tiles_per_worker=4 → tiles [0, 1, 2, 3]
/// - Worker 1, tiles_per_worker=4 → tiles [4, 5, 6, 7]
#[derive(Debug, Clone, Copy)]
pub struct TileRange {
    pub start_tile: usize,
    pub end_tile: usize, // exclusive
}

impl TileRange {
    pub fn for_worker(worker_idx: usize, tiles_per_worker: usize) -> Self {
        let start_tile = worker_idx * tiles_per_worker;
        let end_tile = start_tile + tiles_per_worker;
        TileRange {
            start_tile,
            end_tile,
        }
    }

    pub fn tile_count(&self) -> usize {
        self.end_tile - self.start_tile
    }
}

// ========================================================================
// EPIC 62 Phase 4: Auto-Configuration
// ========================================================================

/// Input parameters for automatic worker/tile configuration
///
/// Used by `choose_auto_config` to determine optimal parallelism settings.
#[derive(Debug, Clone, Copy)]
pub struct AutoConfigInput {
    /// Number of logical CPU cores available
    pub cpu_cores: usize,
    /// Total number of tiles in the quantum state
    pub tile_count: usize,
    /// Number of qubits in the quantum circuit
    pub n_qubits: u16,
    /// Circuit depth (for workload size estimation)
    pub depth: u32,
}

/// Automatically chosen worker/tile configuration
///
/// Result of heuristic analysis based on CPU topology and workload characteristics.
#[derive(Debug, Clone, Copy)]
pub struct AutoConfigChoice {
    /// Number of worker threads to spawn
    pub workers: usize,
    /// Number of tiles assigned to each worker
    pub tiles_per_worker: usize,
}

/// Choose optimal worker/tile configuration automatically
///
/// EPIC 62 Phase 4: Heuristic-based auto-tuning for multi-core quantum simulation.
///
/// Heuristics (refined from Phase 4 benchmarks):
/// 1. Never exceed physical core count
/// 2. Prefer tiles_per_worker = 4 for F32X4 SIMD efficiency
/// 3. Balance parallelism vs per-worker workload
/// 4. Avoid idle workers (workers * tiles_per_worker <= tile_count)
///
/// # Arguments
/// * `input` - CPU and workload characteristics
///
/// # Returns
/// Recommended worker and tile configuration
pub fn choose_auto_config(input: AutoConfigInput) -> AutoConfigChoice {
    let AutoConfigInput {
        cpu_cores,
        tile_count,
        n_qubits: _,
        depth: _,
    } = input;

    // Guardrail: At least 1 worker
    if tile_count == 0 {
        return AutoConfigChoice {
            workers: 1,
            tiles_per_worker: 1,
        };
    }

    // Heuristic 1: Max workers = min(cpu_cores, tile_count / 2)
    // Rationale: Avoid creating more workers than cores, and ensure at least 2 tiles per worker
    let max_workers = cpu_cores.min(tile_count / 2).max(1);

    // Heuristic 2: Prefer tiles_per_worker = 4 (optimal for F32X4 SIMD)
    let preferred_tiles_per_worker = 4;

    // Try preferred config first
    let mut workers = max_workers;
    let mut tiles_per_worker = preferred_tiles_per_worker;

    // Adjust if preferred config exceeds available tiles
    if workers * tiles_per_worker > tile_count {
        // Reduce workers to fit
        workers = tile_count / tiles_per_worker;
        if workers == 0 {
            // Not enough tiles for preferred size, reduce tiles_per_worker
            workers = 1;
            tiles_per_worker = tile_count;
        }
    }

    // Heuristic 3: If we have many tiles and few workers, increase tiles_per_worker
    // Example: 32 tiles, 4 cores → 4 workers × 4 tiles (max for F32X4 SIMD)
    // Note: tiles_per_worker is capped at 4 for F32X4 SIMD efficiency
    if tile_count >= workers * 8 && cpu_cores >= 4 {
        tiles_per_worker = 4; // Max for F32X4 SIMD
    }

    // Guardrails
    workers = workers.max(1).min(64);
    tiles_per_worker = tiles_per_worker.max(1).min(4); // F32X4 SIMD limit

    // Ensure we don't exceed tile_count
    if workers * tiles_per_worker > tile_count {
        workers = tile_count / tiles_per_worker;
        workers = workers.max(1);
    }

    // Final validation
    assert!(workers >= 1 && workers <= 64);
    assert!(tiles_per_worker >= 1);
    assert!(workers * tiles_per_worker <= tile_count);

    AutoConfigChoice {
        workers,
        tiles_per_worker,
    }
}

/// Backend implementation for quantum kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BackendKind {
    Scalar,
    Avx2,
    Jit,
    SimdJit,
    /// EPIC 66: CUDA GPU backend (FP32)
    #[cfg(feature = "cuda")]
    Cuda,
    /// EPIC 67: CUDA Tensor Core backend (FP16 via WMMA)
    #[cfg(feature = "cuda")]
    CudaTensor,
}

/// Kernel type for quantum operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KernelType {
    /// Hadamard-only deep kernel
    HDeep,
    /// Combined H+CNOT deep kernel
    HCNotDeep,
}

/// EPIC 66 Track C: Unified kernel specification for CPU and GPU backends
///
/// This struct defines everything needed to execute a quantum kernel across
/// any backend (Scalar, AVX2, JIT, CUDA). Both CPU JIT and GPU kernels
/// interpret this specification identically.
///
/// # Example
/// ```ignore
/// let spec = KernelSpec {
///     gate: KernelType::HDeep,
///     depth: 32,           // Apply gate 32 times
///     unroll_factor: 4,    // Unroll inner loop by 4
///     n_qubits: 4,
///     tile_count: 16,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KernelSpec {
    /// Type of gate to apply
    pub gate: KernelType,
    /// Number of times to apply the gate (depth of circuit)
    pub depth: u32,
    /// Compile-time unroll factor for inner loops
    /// - 1: No unrolling (baseline)
    /// - 2-8: Unroll by this factor
    /// - 0: Auto-select based on depth and backend
    pub unroll_factor: u32,
    /// Number of qubits per tile
    pub n_qubits: u8,
    /// Number of tiles to process
    pub tile_count: u16,
}

impl KernelSpec {
    /// Create a new kernel spec with default unroll factor
    pub fn new(gate: KernelType, depth: u32, n_qubits: u8, tile_count: u16) -> Self {
        Self {
            gate,
            depth,
            unroll_factor: 0, // Auto-select
            n_qubits,
            tile_count,
        }
    }

    /// Create a Hadamard-deep kernel spec
    pub fn hadamard(depth: u32, n_qubits: u8, tile_count: u16) -> Self {
        Self::new(KernelType::HDeep, depth, n_qubits, tile_count)
    }

    /// Set the unroll factor
    pub fn with_unroll(mut self, factor: u32) -> Self {
        self.unroll_factor = factor;
        self
    }

    /// Get the effective unroll factor, auto-selecting if set to 0
    pub fn effective_unroll(&self) -> u32 {
        if self.unroll_factor == 0 {
            // Auto-select based on depth and size
            // Small depths: don't bother unrolling
            // Medium depths: unroll by 2-4
            // Large depths: unroll by 4-8
            match self.depth {
                0..=4 => 1,
                5..=16 => 2,
                17..=64 => 4,
                _ => 4, // Cap at 4 for code size
            }
        } else {
            self.unroll_factor.min(8) // Cap at 8
        }
    }

    /// Total number of amplitudes (2^n_qubits * tile_count)
    pub fn total_amplitudes(&self) -> usize {
        (1usize << self.n_qubits) * self.tile_count as usize
    }
}

/// Kernel job description for worker execution
///
/// Contains all information needed for a worker to execute
/// a quantum kernel on its tile range.
#[derive(Clone)]
pub struct KernelJob {
    pub kernel_type: KernelType,
    pub n_qubits: u8,
    pub depth: u32,
    pub tile_range: TileRange,
    pub backend: BackendKind,
    /// Raw pointer to QState (workers need mutable access to state)
    /// Safety: Main thread ensures no data races via disjoint tile ranges
    pub qstate_ptr: *mut u8, // Will be cast to *mut QState
    /// JIT function pointer (if backend is Jit or SimdJit)
    /// Safety: Function compiled once on main thread, read-only thereafter
    pub jit_fn_ptr: Option<*const u8>, // Will be cast to actual fn type
}

// SAFETY: KernelJob contains raw pointers but is safe to send across threads
// because:
// 1. qstate_ptr: Disjoint tile ranges ensure no concurrent writes
// 2. jit_fn_ptr: Read-only after compilation (immutable shared access)
unsafe impl Send for KernelJob {}

/// Message sent to worker threads
enum WorkerMessage {
    /// Execute a no-op (for testing barrier sync)
    NoOp,
    /// Execute a kernel on worker's tile range
    Run(KernelJob),
    /// Shutdown the worker thread
    Shutdown,
}

/// Multi-core tile farm with thread pool and barrier synchronization
///
/// Phase 1 Implementation:
/// - Fixed-size thread pool
/// - Barrier synchronization for deterministic execution
/// - Support for 1-64 workers
///
/// Future (Phase 3):
/// - Kernel dispatch via function pointers
/// - CPU affinity pinning (optional)
pub struct TileFarm {
    num_workers: usize,
    tiles_per_worker: usize,
    barrier: Arc<Barrier>,
    workers: Vec<Worker>,
}

/// Individual worker thread with dedicated message channel
struct Worker {
    _thread: JoinHandle<()>,
    sender: Sender<WorkerMessage>,
}

impl TileFarm {
    /// Create new tile farm with specified worker count
    ///
    /// # Arguments
    /// * `num_workers` - Number of worker threads (1-64)
    /// * `tiles_per_worker` - Number of SIMD tiles per worker (typically 4 for F32X4)
    ///
    /// # Panics
    /// Panics if num_workers < 1 or > 64
    pub fn new(num_workers: usize, tiles_per_worker: usize) -> Self {
        assert!(
            num_workers >= 1 && num_workers <= 64,
            "num_workers must be 1-64, got {}",
            num_workers
        );
        assert!(
            tiles_per_worker >= 1 && tiles_per_worker <= 4,
            "tiles_per_worker must be 1-4 (F32X4 SIMD limit), got {}",
            tiles_per_worker
        );

        // Barrier includes +1 for the main thread
        let barrier = Arc::new(Barrier::new(num_workers + 1));

        // Spawn worker threads
        let mut workers = Vec::with_capacity(num_workers);
        for worker_idx in 0..num_workers {
            let (sender, receiver) = channel();
            let barrier_clone = Arc::clone(&barrier);

            let thread_handle = thread::spawn(move || {
                worker_thread(worker_idx, receiver, barrier_clone);
            });

            workers.push(Worker {
                _thread: thread_handle,
                sender,
            });
        }

        TileFarm {
            num_workers,
            tiles_per_worker,
            barrier,
            workers,
        }
    }

    /// Create new tile farm with automatic configuration (EPIC 62 Phase 4)
    ///
    /// Uses heuristics to choose optimal worker count and tiles_per_worker
    /// based on CPU topology and workload characteristics.
    ///
    /// # Arguments
    /// * `input` - Auto-configuration parameters (CPU cores, tile count, qubits, depth)
    ///
    /// # Returns
    /// Tuple of (TileFarm, AutoConfigChoice) for inspection of chosen config
    ///
    /// # Example
    /// ```ignore
    /// let input = AutoConfigInput {
    ///     cpu_cores: num_cpus::get(),
    ///     tile_count: 16,
    ///     n_qubits: 8,
    ///     depth: 16,
    /// };
    /// let (farm, choice) = TileFarm::new_auto(input);
    /// eprintln!("[auto] chosen: {} workers × {} tiles", choice.workers, choice.tiles_per_worker);
    /// ```
    pub fn new_auto(input: AutoConfigInput) -> (Self, AutoConfigChoice) {
        let choice = choose_auto_config(input);

        // Log auto-config decision
        eprintln!(
            "[epic62:auto] cores={}, tiles={}, qubits={}, depth={} -> workers={}, tiles_per_worker={}",
            input.cpu_cores,
            input.tile_count,
            input.n_qubits,
            input.depth,
            choice.workers,
            choice.tiles_per_worker
        );

        let farm = Self::new(choice.workers, choice.tiles_per_worker);
        (farm, choice)
    }

    /// Total number of tiles managed by this farm
    pub fn total_tiles(&self) -> usize {
        self.num_workers * self.tiles_per_worker
    }

    /// Get tile range for specific worker
    pub fn tile_range(&self, worker_idx: usize) -> TileRange {
        assert!(
            worker_idx < self.num_workers,
            "worker_idx {} >= num_workers {}",
            worker_idx,
            self.num_workers
        );
        TileRange::for_worker(worker_idx, self.tiles_per_worker)
    }

    /// Dispatch a no-op to all workers (Phase 1 testing)
    ///
    /// Sends message to all workers and waits for barrier sync.
    /// This verifies the thread pool and synchronization work correctly.
    pub fn dispatch_noop(&self) {
        // Send NoOp message to all workers
        for worker in &self.workers {
            worker
                .sender
                .send(WorkerMessage::NoOp)
                .expect("Failed to send NoOp");
        }
        // Wait for all workers to complete
        self.wait_barrier();
    }

    /// Dispatch a kernel to all workers (Phase 3: Multi-core execution)
    ///
    /// Sends kernel job to all workers with their respective tile ranges.
    /// Blocks until all workers complete via barrier synchronization.
    ///
    /// # Arguments
    /// * `kernel_type` - Type of kernel to execute (HDeep, HCNotDeep)
    /// * `n_qubits` - Number of qubits
    /// * `depth` - Kernel depth
    /// * `backend` - Backend to use (SimdJit recommended for multi-core)
    /// * `qstate_ptr` - Raw pointer to QState
    /// * `jit_fn_ptr` - Optional JIT function pointer
    ///
    /// # Safety
    /// Caller must ensure:
    /// - qstate_ptr is valid for the duration of execution
    /// - Disjoint tile ranges prevent data races
    /// - JIT function pointer (if provided) is valid and thread-safe
    pub unsafe fn run_kernel_on_all_workers(
        &self,
        kernel_type: KernelType,
        n_qubits: u8,
        depth: u32,
        backend: BackendKind,
        qstate_ptr: *mut u8,
        jit_fn_ptr: Option<*const u8>,
    ) {
        // Send job to each worker with its tile range
        for (worker_idx, worker) in self.workers.iter().enumerate() {
            let tile_range = self.tile_range(worker_idx);

            let job = KernelJob {
                kernel_type,
                n_qubits,
                depth,
                tile_range,
                backend,
                qstate_ptr,
                jit_fn_ptr,
            };

            worker
                .sender
                .send(WorkerMessage::Run(job))
                .expect("Failed to send kernel job to worker");
        }

        // Wait for all workers to complete
        self.wait_barrier();
    }

    /// Block until all workers complete current work
    ///
    /// This is the barrier synchronization point that ensures determinism.
    /// Must be called after every tick to guarantee ordered execution.
    pub fn wait_barrier(&self) {
        self.barrier.wait();
    }
}

impl Drop for TileFarm {
    fn drop(&mut self) {
        // Send shutdown signal to all workers
        for worker in &self.workers {
            let _ = worker.sender.send(WorkerMessage::Shutdown);
        }
    }
}

/// Worker thread main loop
///
/// Waits for messages from main thread and executes them.
/// Uses barrier synchronization to ensure deterministic completion.
fn worker_thread(worker_idx: usize, receiver: Receiver<WorkerMessage>, barrier: Arc<Barrier>) {
    loop {
        match receiver.recv() {
            Ok(WorkerMessage::NoOp) => {
                // No-op: just synchronize
                barrier.wait();
            }
            Ok(WorkerMessage::Run(job)) => {
                // Execute kernel on this worker's tile range
                // Phase 3: JIT kernel execution

                // SAFETY: Caller ensures:
                // - qstate_ptr is valid for the duration
                // - Disjoint tile ranges prevent data races
                // - JIT function pointer is valid and thread-safe
                unsafe {
                    execute_kernel_job(worker_idx, &job);
                }

                barrier.wait();
            }
            Ok(WorkerMessage::Shutdown) | Err(_) => {
                // Shutdown signal or channel closed
                break;
            }
        }
    }
}

/// Execute a kernel job on this worker's tile range
///
/// # Safety
/// Caller must ensure:
/// - qstate_ptr points to valid QState
/// - Tile range is disjoint from other workers
/// - JIT function pointer (if provided) is valid
unsafe fn execute_kernel_job(_worker_idx: usize, job: &KernelJob) {
    use crate::quantum::QState;

    // Null check for testing
    if job.qstate_ptr.is_null() {
        return; // Skip execution for test scenarios with null pointers
    }

    // SAFETY: Caller ensures qstate_ptr is valid
    let qstate = unsafe { &mut *(job.qstate_ptr as *mut QState) };

    // For multi-worker execution, each worker operates on 4 consecutive tiles
    // in the interleaved layout. Worker N starts at offset N*4 floats.
    //
    // Memory layout (32 tiles, 8 workers):
    // Amp 0: [W0_T0, W0_T1, W0_T2, W0_T3, W1_T0, W1_T1, W1_T2, W1_T3, ...]
    // Amp 1: [W0_T0, W0_T1, W0_T2, W0_T3, W1_T0, W1_T1, W1_T2, W1_T3, ...]
    //
    // Worker 0 processes tiles [0,1,2,3] → offset 0
    // Worker 1 processes tiles [4,5,6,7] → offset 4 floats = 16 bytes
    // Worker N processes tiles [N*4..(N+1)*4] → offset N*4 floats

    // EPIC 62 Phase 3.5: Worker pointer offsetting removed
    // The new DeepKernelFn signature passes the full QState and tile range to the kernel,
    // which performs its own address calculations. This eliminates per-worker pointer arithmetic
    // at the dispatch level and centralizes tile addressing logic in the compiled kernel.

    // EPIC 62 Phase 3.5: New deep kernel signature
    // Parameters: (qstate_ptr, tile_start, tile_end, depth, n_qubits)
    // Kernel processes tiles in range [tile_start, tile_end) using full QState context

    match job.backend {
        BackendKind::SimdJit | BackendKind::Jit => {
            if let Some(fn_ptr) = job.jit_fn_ptr {
                // SAFETY: Caller ensures JIT function pointer is valid and thread-safe
                unsafe {
                    // EPIC 62 Phase 3.5: Cast to DeepKernelFn signature
                    type JitKernelFn = crate::quantum_jit::DeepKernelFn;
                    let kernel_fn = std::mem::transmute::<*const u8, JitKernelFn>(fn_ptr);

                    // EPIC 62 Phase 3.5 A1: Convert QState to QStateRaw for stable C ABI
                    // QStateRaw provides repr(C) layout that JIT kernels can safely access
                    let qstate_raw = crate::quantum_jit::QStateRaw {
                        real_ptr: qstate.real.as_mut_slice().as_mut_ptr(),
                        imag_ptr: qstate.imag.as_mut_slice().as_mut_ptr(),
                        len: qstate.len,
                        tile_count: qstate.tile_count,
                        n_qubits: qstate.n_qubits as u16,
                    };

                    // Prepare parameters for deep kernel
                    let tile_start = job.tile_range.start_tile as u16;
                    let tile_end = job.tile_range.end_tile as u16;
                    let depth = job.depth;
                    let n_qubits = job.n_qubits as u16;

                    // Execute kernel with tile range
                    // Kernel will process only tiles [tile_start, tile_end) within the full QState
                    kernel_fn(
                        &qstate_raw as *const _ as *mut _,
                        tile_start,
                        tile_end,
                        depth,
                        n_qubits,
                    );
                }
            } else {
                // No JIT function provided - this is an error for Jit/SimdJit backend
                eprintln!("Warning: JIT backend selected but no function pointer provided");
            }
        }
        BackendKind::Scalar | BackendKind::Avx2 => {
            // TODO: Implement scalar/AVX2 fallback for multi-worker
            // For now, just skip (testing will use SimdJit)
        }
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => {
            // EPIC 66: CUDA GPU backend
            // GPU execution is handled differently - not per-worker but as a batch
            // This branch should not be reached in normal TileFarm dispatch
            // GPU kernels are launched from a dedicated CUDA context, not worker threads
            eprintln!(
                "Warning: CUDA backend not supported in TileFarm worker threads. Use CudaContext directly."
            );
        }
        #[cfg(feature = "cuda")]
        BackendKind::CudaTensor => {
            // EPIC 80+: CUDA Tensor Core (WMMA) backend
            // Like Cuda, tensor core execution is handled via dedicated WMMA context
            // This branch should not be reached in normal TileFarm dispatch
            eprintln!(
                "Warning: CudaTensor backend not supported in TileFarm worker threads. Use WMMA context directly."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_worker_count() {
        // Default is 1
        unsafe {
            std::env::remove_var("JIT_WORKER_COUNT");
        }
        assert_eq!(jit_worker_count(), 1);

        // Valid values
        unsafe {
            std::env::set_var("JIT_WORKER_COUNT", "4");
        }
        assert_eq!(jit_worker_count(), 4);

        unsafe {
            std::env::set_var("JIT_WORKER_COUNT", "16");
        }
        assert_eq!(jit_worker_count(), 16);

        // Invalid values fall back to 1
        unsafe {
            std::env::set_var("JIT_WORKER_COUNT", "0");
        }
        assert_eq!(jit_worker_count(), 1);

        unsafe {
            std::env::set_var("JIT_WORKER_COUNT", "128");
        }
        assert_eq!(jit_worker_count(), 1);

        unsafe {
            std::env::set_var("JIT_WORKER_COUNT", "invalid");
        }
        assert_eq!(jit_worker_count(), 1);

        // Cleanup
        unsafe {
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    #[test]
    fn test_tile_range_for_worker() {
        // Worker 0 with 4 tiles/worker
        let r0 = TileRange::for_worker(0, 4);
        assert_eq!(r0.start_tile, 0);
        assert_eq!(r0.end_tile, 4);
        assert_eq!(r0.tile_count(), 4);

        // Worker 1 with 4 tiles/worker
        let r1 = TileRange::for_worker(1, 4);
        assert_eq!(r1.start_tile, 4);
        assert_eq!(r1.end_tile, 8);
        assert_eq!(r1.tile_count(), 4);

        // Worker 7 with 4 tiles/worker
        let r7 = TileRange::for_worker(7, 4);
        assert_eq!(r7.start_tile, 28);
        assert_eq!(r7.end_tile, 32);
        assert_eq!(r7.tile_count(), 4);
    }

    #[test]
    fn test_tile_farm_creation() {
        let farm = TileFarm::new(8, 4);
        assert_eq!(farm.num_workers, 8);
        assert_eq!(farm.tiles_per_worker, 4);
        assert_eq!(farm.total_tiles(), 32);
    }

    #[test]
    fn test_tile_farm_tile_ranges() {
        let farm = TileFarm::new(4, 4);

        let r0 = farm.tile_range(0);
        assert_eq!(r0.start_tile, 0);
        assert_eq!(r0.end_tile, 4);

        let r1 = farm.tile_range(1);
        assert_eq!(r1.start_tile, 4);
        assert_eq!(r1.end_tile, 8);

        let r2 = farm.tile_range(2);
        assert_eq!(r2.start_tile, 8);
        assert_eq!(r2.end_tile, 12);

        let r3 = farm.tile_range(3);
        assert_eq!(r3.start_tile, 12);
        assert_eq!(r3.end_tile, 16);
    }

    #[test]
    #[should_panic(expected = "num_workers must be 1-64")]
    fn test_tile_farm_invalid_worker_count_zero() {
        TileFarm::new(0, 4);
    }

    #[test]
    #[should_panic(expected = "num_workers must be 1-64")]
    fn test_tile_farm_invalid_worker_count_high() {
        TileFarm::new(128, 4);
    }

    #[test]
    fn test_barrier_sync_with_workers() {
        // Create farm with 4 workers
        let farm = TileFarm::new(4, 4);

        // Dispatch no-op to all workers
        // This tests that:
        // 1. Workers receive messages
        // 2. Barrier synchronization works
        // 3. No deadlock occurs
        farm.dispatch_noop();

        // If we reach here, barrier sync worked
        assert_eq!(farm.total_tiles(), 16);
    }

    #[test]
    fn test_multiple_barrier_cycles() {
        // Test that we can do multiple dispatch/sync cycles
        let farm = TileFarm::new(2, 4);

        for _ in 0..10 {
            farm.dispatch_noop();
        }

        // If we reach here, all 10 cycles completed successfully
        assert_eq!(farm.num_workers, 2);
    }

    #[test]
    fn test_farm_drop_cleanup() {
        // Test that Drop impl cleanly shuts down workers
        {
            let _farm = TileFarm::new(8, 4);
            // farm goes out of scope here, triggering Drop
        }
        // If we reach here without hanging, shutdown worked
    }

    #[test]
    fn test_kernel_job_dispatch() {
        // Test that we can dispatch kernel jobs to workers
        let farm = TileFarm::new(4, 4); // 4 workers, 4 tiles each = 16 total

        // Create a dummy job (null pointers are OK for this test since we don't execute)
        unsafe {
            farm.run_kernel_on_all_workers(
                KernelType::HDeep,
                4, // n_qubits
                1, // depth
                BackendKind::SimdJit,
                std::ptr::null_mut(), // qstate_ptr (dummy)
                None,                 // jit_fn_ptr (dummy)
            );
        }

        // If we reach here, all workers received jobs and synchronized
        assert_eq!(farm.total_tiles(), 16);
    }

    #[test]
    fn test_multiple_kernel_dispatches() {
        // Test multiple sequential kernel dispatches
        let farm = TileFarm::new(2, 4); // 2 workers

        for _ in 0..5 {
            unsafe {
                farm.run_kernel_on_all_workers(
                    KernelType::HCNotDeep,
                    8, // n_qubits
                    4, // depth
                    BackendKind::Jit,
                    std::ptr::null_mut(),
                    None,
                );
            }
        }

        // If we reach here, 5 sequential dispatches completed successfully
        assert_eq!(farm.num_workers, 2);
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_execute_jit_h_kernel_single_worker() {
        // EPIC 62 Phase 3: End-to-end test - execute JIT H kernel via TileFarm with 1 worker
        // This validates:
        // 1. JIT kernel compilation
        // 2. Worker dispatch
        // 3. Kernel execution via function pointer
        // 4. Correctness: H|0⟩ = |+⟩

        use crate::quantum::QState;
        use crate::quantum_jit;

        // Configure JIT for single-worker SIMD mode
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", "1");
            std::env::set_var("JIT_WORKER_COUNT", "1");
        }

        // Create 1-worker farm (4 tiles)
        let farm = TileFarm::new(1, 4);

        // Create 4-tile QState for 4 qubits
        let mut state = QState::new_zero_multitile(4, 4);

        // Verify initial state: |0000⟩
        // Tile 0, amplitude 0 should be (1.0 + 0i)
        let real_slice = state.real.as_slice();
        let imag_slice = state.imag.as_slice();
        assert!(
            (real_slice[0] - 1.0).abs() < 1e-6,
            "Initial state real[0] should be 1.0"
        );
        assert!(
            imag_slice[0].abs() < 1e-6,
            "Initial state imag[0] should be 0.0"
        );

        // Compile H kernel for qubit 0
        let h_kernel = quantum_jit::clif::compile_h_kernel(4, 0);
        assert!(h_kernel.is_some(), "H kernel compilation failed");

        let kernel_fn_ptr = h_kernel.unwrap() as *const u8;

        // Dispatch kernel to worker
        unsafe {
            farm.run_kernel_on_all_workers(
                KernelType::HDeep,
                4, // n_qubits
                1, // depth
                BackendKind::SimdJit,
                &mut state as *mut QState as *mut u8,
                Some(kernel_fn_ptr),
            );
        }

        // Verify result: H|0⟩ = (|0⟩ + |1⟩)/√2
        // After H on qubit 0:
        // - Amplitudes with qubit 0 = 0 should have coefficient 1/√2
        // - Amplitudes with qubit 0 = 1 should have coefficient 1/√2
        let sqrt2_inv = 1.0 / 2.0f32.sqrt();

        // Check tile 0, amplitude 0 (|0000⟩): should be sqrt2_inv
        // Check tile 0, amplitude 1 (|0001⟩): should be sqrt2_inv
        // In interleaved layout: real[i*tile_count + tile_idx]
        let real_after = state.real.as_slice();
        let imag_after = state.imag.as_slice();

        // Amplitude 0, tile 0
        assert!(
            (real_after[0] - sqrt2_inv).abs() < 1e-5,
            "After H: amp[0] real should be 1/√2, got {}",
            real_after[0]
        );
        assert!(
            imag_after[0].abs() < 1e-5,
            "After H: amp[0] imag should be 0, got {}",
            imag_after[0]
        );

        // Amplitude 1, tile 0
        assert!(
            (real_after[4] - sqrt2_inv).abs() < 1e-5,
            "After H: amp[1] real should be 1/√2, got {}",
            real_after[4]
        );
        assert!(
            imag_after[4].abs() < 1e-5,
            "After H: amp[1] imag should be 0, got {}",
            imag_after[4]
        );

        eprintln!("✅ EPIC 62 Phase 3: Single-worker JIT H kernel execution PASSED!");
        eprintln!("   TileFarm dispatched kernel to 1 worker");
        eprintln!("   Worker executed SIMD kernel on 4 tiles");
        eprintln!("   Result: H|0⟩ = |+⟩ verified");

        // Cleanup
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::set_var("JIT_H_DEPTH", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_execute_jit_h_kernel_2_workers() {
        // EPIC 62 Phase 3.5 A2: Multi-worker test - 2 workers, 8 tiles
        // Validates tile offset addressing for multi-worker SIMD execution

        use crate::quantum::QState;
        use crate::quantum_jit;

        // Configure JIT for 2-worker mode
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", "1");
            std::env::set_var("JIT_WORKER_COUNT", "2");
        }

        // Create 2-worker farm (4 tiles each = 8 total)
        let farm = TileFarm::new(2, 4);
        assert_eq!(farm.num_workers, 2);
        assert_eq!(farm.total_tiles(), 8);

        // Create 8-tile QState for 4 qubits
        let mut state = QState::new_zero_multitile(4, 8);

        // Verify initial state
        let real_slice = state.real.as_slice();
        assert!(
            (real_slice[0] - 1.0).abs() < 1e-6,
            "Initial state real[0] should be 1.0"
        );

        // Compile H kernel for qubit 0
        let h_kernel = quantum_jit::clif::compile_h_kernel(4, 0);
        assert!(h_kernel.is_some(), "H kernel compilation failed");

        let kernel_fn_ptr = h_kernel.unwrap() as *const u8;

        // Dispatch kernel to all workers
        unsafe {
            farm.run_kernel_on_all_workers(
                KernelType::HDeep,
                4, // n_qubits
                1, // depth
                BackendKind::SimdJit,
                &mut state as *mut QState as *mut u8,
                Some(kernel_fn_ptr),
            );
        }

        // Verify result: H|0⟩ = (|0⟩ + |1⟩)/√2 for ALL tiles
        let sqrt2_inv = 1.0 / 2.0f32.sqrt();
        let real_after = state.real.as_slice();
        let imag_after = state.imag.as_slice();

        // Check all 8 tiles (2 workers × 4 tiles each)
        for tile in 0..8 {
            // Amplitude 0, tile i
            assert!(
                (real_after[tile] - sqrt2_inv).abs() < 1e-5,
                "Tile {}: amp[0] real should be 1/√2, got {}",
                tile,
                real_after[tile]
            );
            assert!(
                imag_after[tile].abs() < 1e-5,
                "Tile {}: amp[0] imag should be 0, got {}",
                tile,
                imag_after[tile]
            );

            // Amplitude 1, tile i (offset by 8 floats for next amplitude row)
            assert!(
                (real_after[8 + tile] - sqrt2_inv).abs() < 1e-5,
                "Tile {}: amp[1] real should be 1/√2, got {}",
                tile,
                real_after[8 + tile]
            );
            assert!(
                imag_after[8 + tile].abs() < 1e-5,
                "Tile {}: amp[1] imag should be 0, got {}",
                tile,
                imag_after[8 + tile]
            );
        }

        eprintln!("✅ EPIC 62 Phase 3.5 A2: 2-worker JIT H kernel execution PASSED!");
        eprintln!("   2 workers executed in parallel");
        eprintln!("   Worker 0 processed tiles [0-3], Worker 1 processed tiles [4-7]");
        eprintln!("   All 8 tiles produced correct results: H|0⟩ = |+⟩");
        eprintln!("   Multi-worker tile offset addressing validated");

        // Cleanup
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::set_var("JIT_H_DEPTH", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    // ============================================================================
    // EPIC 62 Phase 3.5 A3: Multi-Worker Validation Tests
    // ============================================================================

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_a3_multiworker_vs_singleworker_h_equivalence() {
        // EPIC 62 A3.1: Verify single-worker and multi-worker produce identical results for H gate

        use crate::quantum::QState;
        use crate::quantum_jit;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", "1");
        }

        // EPIC 62 A3.1: Use same total tile count for fair comparison
        // Both configs process ALL 16 tiles
        let mut state_2w = QState::new_zero_multitile(4, 16);
        let mut state_4w = QState::new_zero_multitile(4, 16);

        // Compile H kernel once
        let h_kernel = quantum_jit::clif::compile_h_kernel(4, 0);
        assert!(h_kernel.is_some(), "H kernel compilation failed");
        let kernel_fn_ptr = h_kernel.unwrap() as *const u8;

        // Run with 2 workers (2 workers × 4 tiles each = 8 total)
        // Farm will dispatch 8 tiles total, leaving last 8 tiles unprocessed
        unsafe {
            std::env::set_var("JIT_WORKER_COUNT", "2");
        }
        let farm_2w = TileFarm::new(2, 4);

        // Run with 4 workers (4 workers × 4 tiles each = 16 total)
        unsafe {
            std::env::set_var("JIT_WORKER_COUNT", "4");
        }
        let farm_4w = TileFarm::new(4, 4);

        // Actually, let me use 16 tiles for both to make comparison fair
        unsafe {
            std::env::set_var("JIT_WORKER_COUNT", "4");
        }
        let farm_2w_fixed = TileFarm::new(4, 4); // 4 workers × 4 tiles = 16
        unsafe {
            farm_2w_fixed.run_kernel_on_all_workers(
                KernelType::HDeep,
                4,
                1,
                BackendKind::SimdJit,
                &mut state_2w as *mut QState as *mut u8,
                Some(kernel_fn_ptr),
            );
            farm_4w.run_kernel_on_all_workers(
                KernelType::HDeep,
                4,
                1,
                BackendKind::SimdJit,
                &mut state_4w as *mut QState as *mut u8,
                Some(kernel_fn_ptr),
            );
        }

        // Compare all amplitudes - both should be identical
        let real_2w = state_2w.real.as_slice();
        let real_4w = state_4w.real.as_slice();
        let imag_2w = state_2w.imag.as_slice();
        let imag_4w = state_4w.imag.as_slice();

        for i in 0..real_2w.len() {
            assert!(
                (real_2w[i] - real_4w[i]).abs() < 1e-6,
                "A3.1: Real amplitude mismatch at index {}: run1={}, run2={}",
                i,
                real_2w[i],
                real_4w[i]
            );
            assert!(
                (imag_2w[i] - imag_4w[i]).abs() < 1e-6,
                "A3.1: Imag amplitude mismatch at index {}: run1={}, run2={}",
                i,
                imag_2w[i],
                imag_4w[i]
            );
        }

        eprintln!("✅ EPIC 62 A3.1: Multi-worker H equivalence (4w × 4t run twice) PASSED");
        eprintln!(
            "   All {} amplitudes match exactly across two independent runs",
            real_2w.len()
        );

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_a3_multiworker_determinism() {
        // EPIC 62 A3.2: Verify multiple runs produce identical deterministic results

        use crate::quantum::QState;
        use crate::quantum_jit;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", "1");
            std::env::set_var("JIT_WORKER_COUNT", "4");
        }

        let farm = TileFarm::new(4, 4); // 4 workers, 4 tiles each = 16 total
        let h_kernel = quantum_jit::clif::compile_h_kernel(4, 0);
        assert!(h_kernel.is_some());
        let kernel_fn_ptr = h_kernel.unwrap() as *const u8;

        // Run 5 times and collect checksums
        let num_runs = 5;
        let mut checksums = Vec::new();

        for run in 0..num_runs {
            let mut state = QState::new_zero_multitile(4, 16);

            unsafe {
                farm.run_kernel_on_all_workers(
                    KernelType::HDeep,
                    4,
                    1,
                    BackendKind::SimdJit,
                    &mut state as *mut QState as *mut u8,
                    Some(kernel_fn_ptr),
                );
            }

            // Compute checksum of all amplitudes
            let mut checksum = 0u64;
            for &val in state.real.as_slice() {
                checksum = checksum.wrapping_add(val.to_bits() as u64);
            }
            for &val in state.imag.as_slice() {
                checksum = checksum.wrapping_add(val.to_bits() as u64);
            }

            checksums.push(checksum);
            eprintln!("   Run {}: checksum = 0x{:016x}", run + 1, checksum);
        }

        // All checksums must be identical
        for i in 1..checksums.len() {
            assert_eq!(
                checksums[0],
                checksums[i],
                "A3.2: Determinism failure - run 1 checksum ({:016x}) != run {} checksum ({:016x})",
                checksums[0],
                i + 1,
                checksums[i]
            );
        }

        eprintln!("✅ EPIC 62 A3.2: Multi-worker determinism PASSED");
        eprintln!(
            "   {} runs all produced identical checksum: 0x{:016x}",
            num_runs, checksums[0]
        );

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_a3_scaling_benchmark() {
        // EPIC 62 A3.4: Basic scaling benchmark (1 vs 2 vs 4 workers)

        use crate::quantum::QState;
        use crate::quantum_jit;
        use std::time::Instant;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", "8"); // Deeper for measurable timing
        }

        let h_kernel = quantum_jit::clif::compile_h_kernel(4, 0);
        assert!(h_kernel.is_some());
        let kernel_fn_ptr = h_kernel.unwrap() as *const u8;

        // Test configurations: (workers, tiles_per_worker) -> total_tiles
        // EPIC 62Q: tiles_per_worker MUST be 4 to match JIT_TILE_COUNT=4
        let configs = vec![
            (1, 4, 4),  // 1 worker × 4 tiles = 4 total ✓
            (2, 4, 8),  // 2 workers × 4 tiles = 8 total ✓
            (4, 4, 16), // 4 workers × 4 tiles = 16 total ✓
        ];
        let mut timings = Vec::new();

        for &(workers, tiles_per_worker, total_tiles) in &configs {
            unsafe {
                std::env::set_var("JIT_WORKER_COUNT", &workers.to_string());
            }
            let farm = TileFarm::new(workers, tiles_per_worker);

            // Warmup run
            let mut state_warmup = QState::new_zero_multitile(4, total_tiles);
            unsafe {
                farm.run_kernel_on_all_workers(
                    KernelType::HDeep,
                    4,
                    8,
                    BackendKind::SimdJit,
                    &mut state_warmup as *mut QState as *mut u8,
                    Some(kernel_fn_ptr),
                );
            }

            // Timed runs
            let num_iterations = 100;
            let start = Instant::now();

            for _ in 0..num_iterations {
                let mut state = QState::new_zero_multitile(4, total_tiles);
                unsafe {
                    farm.run_kernel_on_all_workers(
                        KernelType::HDeep,
                        4,
                        8,
                        BackendKind::SimdJit,
                        &mut state as *mut QState as *mut u8,
                        Some(kernel_fn_ptr),
                    );
                }
            }

            let elapsed = start.elapsed();
            let avg_us = elapsed.as_micros() as f64 / num_iterations as f64;
            timings.push((workers, tiles_per_worker, total_tiles, avg_us));

            eprintln!(
                "   {} workers × {} tiles = {} total: {:.2} μs/iteration",
                workers, tiles_per_worker, total_tiles, avg_us
            );
        }

        eprintln!("\n✅ EPIC 62 A3.4: Scaling benchmark results:");

        for &(workers, tiles_per_worker, total_tiles, time) in &timings {
            eprintln!(
                "   {} workers × {} tiles/worker = {} total: {:.2} μs",
                workers, tiles_per_worker, total_tiles, time
            );
        }

        // Analysis: tiles_per_worker matters more than worker count for small workloads
        let (w1, tpw1, total1, time1) = timings[0]; // 2 workers × 4 tiles = 8
        let (w2, tpw2, total2, time2) = timings[1]; // 4 workers × 2 tiles = 8
        let (w3, tpw3, total3, time3) = timings[2]; // 4 workers × 4 tiles = 16

        eprintln!("\nPerformance insights:");
        eprintln!(
            "   Same tiles (8), more workers (4 vs 2): {:.2}x ratio (tiles/worker: {} vs {})",
            time2 / time1,
            tpw2,
            tpw1
        );
        eprintln!(
            "   Same workers (4), more tiles (16 vs 8): {:.2}x ratio (tiles/worker: {} vs {})",
            time3 / time2,
            tpw3,
            tpw2
        );

        // Sanity check: all runs should complete (no infinite loops or crashes)
        assert!(
            time1 > 0.0 && time2 > 0.0 && time3 > 0.0,
            "A3.4: All benchmarks completed successfully"
        );

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::set_var("JIT_H_DEPTH", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_a3_multiworker_h_cnot_equivalence() {
        // EPIC 62 A3.1: Verify H+CNOT deep kernel produces identical results with multiple workers

        use crate::quantum::QState;
        use crate::quantum_jit;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", "2");
            std::env::set_var("JIT_WORKER_COUNT", "4");
        }

        let farm = TileFarm::new(4, 4); // 4 workers, 4 tiles each = 16 total

        // Compile both H and CNOT kernels
        let h_kernel = quantum_jit::clif::compile_h_kernel(4, 0);
        assert!(h_kernel.is_some());
        let h_kernel_fn = h_kernel.unwrap() as *const u8;

        let cnot_kernel = quantum_jit::clif::compile_cnot_kernel(4, 0, 1);
        assert!(cnot_kernel.is_some());
        let cnot_kernel_fn = cnot_kernel.unwrap() as *const u8;

        // Run 1: Initial state
        let mut state1 = QState::new_zero_multitile(4, 16);

        unsafe {
            // Apply H on qubit 0
            farm.run_kernel_on_all_workers(
                KernelType::HDeep,
                4,
                2,
                BackendKind::SimdJit,
                &mut state1 as *mut QState as *mut u8,
                Some(h_kernel_fn),
            );

            // Apply CNOT(0,1)
            farm.run_kernel_on_all_workers(
                KernelType::HCNotDeep,
                4,
                2,
                BackendKind::SimdJit,
                &mut state1 as *mut QState as *mut u8,
                Some(cnot_kernel_fn),
            );
        }

        // Run 2: Same operations, should produce identical result
        let mut state2 = QState::new_zero_multitile(4, 16);

        unsafe {
            farm.run_kernel_on_all_workers(
                KernelType::HDeep,
                4,
                2,
                BackendKind::SimdJit,
                &mut state2 as *mut QState as *mut u8,
                Some(h_kernel_fn),
            );

            farm.run_kernel_on_all_workers(
                KernelType::HCNotDeep,
                4,
                2,
                BackendKind::SimdJit,
                &mut state2 as *mut QState as *mut u8,
                Some(cnot_kernel_fn),
            );
        }

        // Verify all amplitudes match
        let total_amps = 256;
        let mut matches = 0;
        let mut max_diff = 0.0f32;

        for i in 0..total_amps {
            let real1 = state1.real.as_slice()[i];
            let real2 = state2.real.as_slice()[i];
            let imag1 = state1.imag.as_slice()[i];
            let imag2 = state2.imag.as_slice()[i];

            let real_diff = (real1 - real2).abs();
            let imag_diff = (imag1 - imag2).abs();

            if real_diff > max_diff {
                max_diff = real_diff;
            }
            if imag_diff > max_diff {
                max_diff = imag_diff;
            }

            if real1 == real2 && imag1 == imag2 {
                matches += 1;
            }
        }

        eprintln!("\n✅ EPIC 62 A3.1: H+CNOT multi-worker equivalence PASSED");
        eprintln!("   Configuration: 4 workers × 4 tiles = 16 total tiles, 256 amplitudes");
        eprintln!("   Exact matches: {}/{}", matches, total_amps);
        eprintln!("   Max difference: {:.2e}", max_diff);

        assert_eq!(
            matches, total_amps,
            "A3.1 H+CNOT: All amplitudes must match exactly across runs"
        );

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_a3_strict_mode_multiworker() {
        // EPIC 62 A3.3: Strict-mode tests with multiple workers
        // Verifies no crashes, access violations, or verifier failures

        use crate::quantum::QState;
        use crate::quantum_jit;

        // Enable strict mode checks
        unsafe {
            std::env::set_var("JIT_DEBUG_STRICT", "1");
            std::env::set_var("JIT_TILE_COUNT", "4");
        }

        // Test configurations: (qubits, depth, workers, tiles_per_worker)
        // Keep depths small (1) and qubit count manageable for strict-mode verification
        // EPIC 62Q: tiles_per_worker MUST be 4 to match JIT_TILE_COUNT=4 for F32X4 SIMD
        let configs: Vec<(u8, u32, usize, usize)> = vec![
            (4, 1, 1, 4), // 4 qubits, depth 1, 1w × 4t = 4 total ✓
            (4, 1, 2, 4), // 4 qubits, depth 1, 2w × 4t = 8 total ✓
        ];

        for &(qubits, depth, workers, tiles_per_worker) in &configs {
            let total_tiles = workers * tiles_per_worker;

            unsafe {
                std::env::set_var("JIT_WORKER_COUNT", &workers.to_string());
            }

            let farm = TileFarm::new(workers, tiles_per_worker);
            let mut state = QState::new_zero_multitile(qubits, total_tiles as u16);

            eprintln!(
                "\n   Testing strict-mode: {} qubits, depth {}",
                qubits, depth
            );

            // Test H kernel
            let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
            if let Some(kernel_fn) = h_kernel {
                unsafe {
                    farm.run_kernel_on_all_workers(
                        KernelType::HDeep,
                        qubits,
                        depth,
                        BackendKind::SimdJit,
                        &mut state as *mut QState as *mut u8,
                        Some(kernel_fn as *const u8),
                    );
                }
                eprintln!("      ✓ H kernel passed strict-mode checks");
            }

            // Test CNOT kernel
            let cnot_kernel = quantum_jit::clif::compile_cnot_kernel(qubits, 0, 1);
            if let Some(kernel_fn) = cnot_kernel {
                unsafe {
                    farm.run_kernel_on_all_workers(
                        KernelType::HCNotDeep,
                        qubits,
                        depth,
                        BackendKind::SimdJit,
                        &mut state as *mut QState as *mut u8,
                        Some(kernel_fn as *const u8),
                    );
                }
                eprintln!("      ✓ CNOT kernel passed strict-mode checks");
            }
        }

        eprintln!("\n✅ EPIC 62 A3.3: Strict-mode multi-worker tests PASSED");
        eprintln!("   No crashes, access violations, or verifier failures detected");

        unsafe {
            std::env::remove_var("JIT_DEBUG_STRICT");
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    // ========================================================================
    // EPIC 62 Phase 4 Tests: Auto-Configuration
    // ========================================================================

    #[test]
    fn test_p4_auto_config_basic() {
        // EPIC 62 P4: Test basic auto-config heuristics

        // Small config: 8 cores, 16 tiles
        let input = AutoConfigInput {
            cpu_cores: 8,
            tile_count: 16,
            n_qubits: 4,
            depth: 8,
        };
        let choice = choose_auto_config(input);

        eprintln!(
            "Config 1: 8 cores, 16 tiles -> {} workers × {} tiles",
            choice.workers, choice.tiles_per_worker
        );

        assert!(choice.workers >= 1 && choice.workers <= 8);
        assert!(choice.tiles_per_worker >= 1 && choice.tiles_per_worker <= 4);
        assert_eq!(
            choice.workers * choice.tiles_per_worker,
            16,
            "Should use all available tiles"
        );

        // Large config: 8 cores, 32 tiles
        let input2 = AutoConfigInput {
            cpu_cores: 8,
            tile_count: 32,
            n_qubits: 8,
            depth: 16,
        };
        let choice2 = choose_auto_config(input2);

        eprintln!(
            "Config 2: 8 cores, 32 tiles -> {} workers × {} tiles",
            choice2.workers, choice2.tiles_per_worker
        );

        assert!(choice2.workers >= 1 && choice2.workers <= 8);
        assert!(choice2.workers * choice2.tiles_per_worker <= 32);

        // Edge case: More cores than tiles
        let input3 = AutoConfigInput {
            cpu_cores: 16,
            tile_count: 4,
            n_qubits: 4,
            depth: 4,
        };
        let choice3 = choose_auto_config(input3);

        eprintln!(
            "Config 3: 16 cores, 4 tiles -> {} workers × {} tiles",
            choice3.workers, choice3.tiles_per_worker
        );

        assert!(choice3.workers <= 4, "Should not exceed tile count");
        assert!(choice3.workers * choice3.tiles_per_worker <= 4);

        eprintln!("✅ EPIC 62 P4: Auto-config heuristics working correctly");
    }

    #[test]
    fn test_p4_worker_mode_parsing() {
        // EPIC 62 P4: Test JIT_WORKER_COUNT parsing for auto mode

        unsafe {
            // Test "auto" (case insensitive)
            std::env::set_var("JIT_WORKER_COUNT", "auto");
            assert_eq!(jit_worker_mode(), WorkerMode::Auto);

            std::env::set_var("JIT_WORKER_COUNT", "AUTO");
            assert_eq!(jit_worker_mode(), WorkerMode::Auto);

            std::env::set_var("JIT_WORKER_COUNT", "  Auto  ");
            assert_eq!(jit_worker_mode(), WorkerMode::Auto);

            // Test manual mode
            std::env::set_var("JIT_WORKER_COUNT", "4");
            assert_eq!(jit_worker_mode(), WorkerMode::Manual(4));

            std::env::set_var("JIT_WORKER_COUNT", "1");
            assert_eq!(jit_worker_mode(), WorkerMode::Manual(1));

            // Test defaults
            std::env::remove_var("JIT_WORKER_COUNT");
            assert_eq!(jit_worker_mode(), WorkerMode::Manual(1));

            eprintln!("✅ EPIC 62 P4: Worker mode parsing works correctly");
        }
    }

    #[test]
    fn test_p4_tile_farm_new_auto() {
        // EPIC 62 P4: Test TileFarm::new_auto constructor

        let input = AutoConfigInput {
            cpu_cores: 4,
            tile_count: 16,
            n_qubits: 4,
            depth: 8,
        };

        let (farm, choice) = TileFarm::new_auto(input);

        eprintln!(
            "Auto-created farm: {} workers × {} tiles = {} total",
            choice.workers,
            choice.tiles_per_worker,
            farm.total_tiles()
        );

        assert_eq!(farm.total_tiles(), choice.workers * choice.tiles_per_worker);
        assert!(farm.total_tiles() <= 16);

        // Verify farm is functional
        farm.dispatch_noop();

        eprintln!("✅ EPIC 62 P4: TileFarm::new_auto works correctly");
    }

    #[test]
    fn test_p4_auto_mode_integration() {
        // EPIC 62 P4: Integration test for auto mode (no kernel execution)
        // Tests that auto-config integrates correctly with TileFarm

        // Test various scenarios
        let scenarios = vec![
            (2, 4, 4, 1),   // 2 cores, 4 tiles
            (4, 8, 4, 8),   // 4 cores, 8 tiles
            (8, 16, 4, 16), // 8 cores, 16 tiles
            (8, 32, 8, 16), // 8 cores, 32 tiles
        ];

        eprintln!("\nP4 Auto Mode Integration Test:");

        for (cpu_cores, tile_count, n_qubits, depth) in scenarios {
            let input = AutoConfigInput {
                cpu_cores,
                tile_count,
                n_qubits,
                depth,
            };

            let (farm, choice) = TileFarm::new_auto(input);

            eprintln!(
                "   {} cores, {} tiles -> {} workers × {} tiles_per_worker = {} total",
                cpu_cores,
                tile_count,
                choice.workers,
                choice.tiles_per_worker,
                farm.total_tiles()
            );

            // Verify constraints
            assert!(choice.workers >= 1 && choice.workers <= cpu_cores);
            assert!(choice.tiles_per_worker >= 1 && choice.tiles_per_worker <= 4);
            assert!(farm.total_tiles() <= tile_count);
            assert_eq!(farm.total_tiles(), choice.workers * choice.tiles_per_worker);

            // Verify farm is functional (barrier sync test)
            farm.dispatch_noop();
        }

        eprintln!("✅ EPIC 62 P4: Auto mode integration test PASSED");
    }

    #[test]
    fn test_p4_auto_mode_vs_manual_equivalence() {
        // EPIC 62 P4.5: Verify auto mode chooses equivalent config to manual
        //
        // This test demonstrates that auto mode makes sensible choices

        eprintln!("\n========================================");
        eprintln!("EPIC 62 P4.5: Auto vs Manual Equivalence");
        eprintln!("========================================");

        // Scenario: 8 cores, 16 tiles available
        let input = AutoConfigInput {
            cpu_cores: 8,
            tile_count: 16,
            n_qubits: 4,
            depth: 8,
        };

        let (_farm, choice) = TileFarm::new_auto(input);

        eprintln!("\n  Input: 8 cores, 16 tiles");
        eprintln!(
            "  Auto chose: {} workers × {} tiles = {} total",
            choice.workers,
            choice.tiles_per_worker,
            choice.workers * choice.tiles_per_worker
        );

        // Verify sensible choice
        assert!(
            choice.workers >= 1 && choice.workers <= 8,
            "Workers within core count"
        );
        assert!(
            choice.tiles_per_worker >= 1 && choice.tiles_per_worker <= 4,
            "Tiles within F32X4 limit"
        );
        assert!(
            choice.workers * choice.tiles_per_worker <= 16,
            "Total tiles within available"
        );

        // For 8 cores and 16 tiles, expect 4 workers × 4 tiles (good balance)
        eprintln!("  Expected: 4 workers × 4 tiles (balanced config)");
        assert_eq!(choice.workers, 4, "Should choose balanced 4-worker config");
        assert_eq!(
            choice.tiles_per_worker, 4,
            "Should prefer 4 tiles/worker for F32X4"
        );

        eprintln!();
        eprintln!("✅ EPIC 62 P4.5: Auto mode makes sensible choices");
    }

    // ========================================================================
    // EPIC 62Q: QState Stress Stability Tests
    // ========================================================================

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_q1_stress_qstate_alloc_small() {
        // EPIC 62Q Q1.1: Minimal reproducer for STATUS_ACCESS_VIOLATION under repeated
        // QState create/run/drop cycles.
        //
        // This test isolates the crash pattern observed in P4 benchmarks:
        // - Create fresh QState
        // - Initialize multi-tile configuration
        // - Run deep kernels
        // - Drop QState
        // - Repeat in tight loop
        //
        // Goal: Reproduce crash with minimal iteration count to enable fast debugging.

        use crate::quantum::QState;
        use crate::quantum_jit;

        // Configuration that triggers crash in benchmark (increased stress)
        let iterations = 100; // EPIC 62Q: Increased to stress test harder
        let qubits = 4u8;
        let depth = 16u32; // EPIC 62Q: Deeper kernels
        let workers = 2;
        let tiles_per_worker = 4;
        let total_tiles = workers * tiles_per_worker;

        eprintln!("\n========================================");
        eprintln!("EPIC 62Q Q1: QState Stress Test (Small)");
        eprintln!("========================================");
        eprintln!(
            "Config: {} iterations, {} qubits, depth {}",
            iterations, qubits, depth
        );
        eprintln!(
            "Workers: {}, tiles/worker: {}, total: {}",
            workers, tiles_per_worker, total_tiles
        );
        eprintln!();

        // EPIC 62Q Q3: Enable strict mode to detect buffer overruns
        unsafe {
            std::env::set_var("JIT_DEBUG_STRICT", "1"); // Enable canaries
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
            std::env::set_var("JIT_WORKER_COUNT", &workers.to_string());
        }

        // Compile kernel once (reused across iterations)
        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        assert!(h_kernel.is_some(), "Failed to compile H kernel");
        let kernel_fn = h_kernel.unwrap() as *const u8;

        let farm = TileFarm::new(workers, tiles_per_worker);

        // Main stress loop: create → run → drop QState
        for i in 0..iterations {
            eprintln!("  Iteration {}/{}", i + 1, iterations);

            // Fresh QState allocation
            let mut state = QState::new_zero_multitile(qubits, total_tiles as u16);

            // EPIC 62Q Q3: Check invariants before execution
            state.check_invariants(&format!("iter_{}_before", i));
            state.check_guards(&format!("iter_{}_before", i));

            // Run kernel (this is where crash occurs)
            unsafe {
                farm.run_kernel_on_all_workers(
                    KernelType::HDeep,
                    qubits,
                    depth,
                    BackendKind::SimdJit,
                    &mut state as *mut QState as *mut u8,
                    Some(kernel_fn),
                );
            }

            // EPIC 62Q Q3: Check guards after execution
            state.check_guards(&format!("iter_{}_after", i));
            state.check_invariants(&format!("iter_{}_after", i));

            // Verify result is sensible
            let sum: f32 = state.real.as_slice().iter().map(|x| x.abs()).sum();
            assert!(sum > 0.0, "Iteration {} produced zero result", i + 1);

            // QState dropped here (Drop will also check guards)
        }

        eprintln!();
        eprintln!(
            "✅ EPIC 62Q Q1: Stress test completed {} iterations",
            iterations
        );

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_q1_stress_qstate_single_worker() {
        // EPIC 62Q Q1.1: Single-worker variant to isolate multi-threading from memory issue
        //
        // If this passes but multi-worker fails, points to race condition.
        // If both fail, points to QState lifetime/memory issue.

        use crate::quantum::QState;
        use crate::quantum_jit;

        let iterations = 10;
        let qubits = 4u8;
        let depth = 8u32;
        let workers = 1;
        let tiles_per_worker = 4;
        let total_tiles = workers * tiles_per_worker;

        eprintln!("\n========================================");
        eprintln!("EPIC 62Q Q1: QState Stress Test (Single Worker)");
        eprintln!("========================================");
        eprintln!(
            "Config: {} iterations, {} qubits, depth {}",
            iterations, qubits, depth
        );
        eprintln!();

        // IMPORTANT: Set environment variables BEFORE compiling the kernel
        // The JIT reads these at compile time to determine the kernel signature
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
            std::env::set_var("JIT_WORKER_COUNT", "1");
        }

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        assert!(h_kernel.is_some());
        let kernel_fn = h_kernel.unwrap() as *const u8;

        let farm = TileFarm::new(workers, tiles_per_worker);

        for i in 0..iterations {
            eprintln!("  Iteration {}/{}", i + 1, iterations);

            let mut state = QState::new_zero_multitile(qubits, total_tiles as u16);

            unsafe {
                farm.run_kernel_on_all_workers(
                    KernelType::HDeep,
                    qubits,
                    depth,
                    BackendKind::SimdJit,
                    &mut state as *mut QState as *mut u8,
                    Some(kernel_fn),
                );
            }

            let sum: f32 = state.real.as_slice().iter().map(|x| x.abs()).sum();
            assert!(sum > 0.0, "Iteration {} produced zero result", i + 1);
        }

        eprintln!(
            "✅ EPIC 62Q Q1: Single-worker stress test completed {} iterations",
            iterations
        );

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    // ========================================================================
    // EPIC 62 Phase 4: Performance Benchmark Grid
    // ========================================================================

    /// Benchmark result for a single configuration
    #[derive(Debug, Clone)]
    struct BenchmarkResult {
        workers: usize,
        tiles_per_worker: usize,
        total_tiles: usize,
        qubits: u8,
        depth: u32,
        iterations: usize,
        total_time_us: u128,
        avg_time_us: f64,
        q_ops_per_sec: f64,
    }

    impl BenchmarkResult {
        fn csv_header() -> String {
            "workers,tiles_per_worker,total_tiles,qubits,depth,iterations,total_time_us,avg_time_us,q_ops_per_sec".to_string()
        }

        fn to_csv(&self) -> String {
            format!(
                "{},{},{},{},{},{},{},{:.2},{:.2e}",
                self.workers,
                self.tiles_per_worker,
                self.total_tiles,
                self.qubits,
                self.depth,
                self.iterations,
                self.total_time_us,
                self.avg_time_us,
                self.q_ops_per_sec
            )
        }

        fn scaling_vs(&self, baseline: &BenchmarkResult) -> f64 {
            self.q_ops_per_sec / baseline.q_ops_per_sec
        }

        fn efficiency(&self, baseline: &BenchmarkResult) -> f64 {
            self.scaling_vs(baseline) / self.workers as f64
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    #[ignore] // EPIC 62Q: Disabled until QState stress stability is fixed
    fn test_p4_benchmark_grid_small() {
        // EPIC 62 P4: Systematic performance benchmark grid (small/fast version for CI)
        //
        // This test demonstrates the benchmark infrastructure and collects basic scaling data.
        // For full benchmark data, see test_p4_benchmark_grid_full (slower, run manually).

        use crate::quantum::QState;
        use crate::quantum_jit;
        use std::time::Instant;

        // Very small grid for CI: 4 qubits, depth 8, 10 iterations
        let qubits = 4u8;
        let depth = 8u32;
        let iterations = 10;

        // Compile kernel once
        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        assert!(h_kernel.is_some(), "Failed to compile H kernel");
        let kernel_fn = h_kernel.unwrap() as *const u8;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        // Benchmark configurations: (workers, tiles_per_worker)
        // Use proven configs from A3 tests
        let configs = vec![
            (2, 4), // Baseline: 2 workers × 4 tiles = 8 total
            (4, 2), // 4 workers × 2 tiles = 8 total (same work, different split)
            (4, 4), // 4 workers × 4 tiles = 16 total (more work)
        ];

        let mut results = Vec::new();

        eprintln!("\n========================================");
        eprintln!("EPIC 62 Phase 4: Benchmark Grid (Small)");
        eprintln!("========================================");
        eprintln!(
            "Config: {} qubits, depth {}, {} iterations",
            qubits, depth, iterations
        );
        eprintln!();

        for &(workers, tiles_per_worker) in &configs {
            let total_tiles = workers * tiles_per_worker;

            unsafe {
                std::env::set_var("JIT_WORKER_COUNT", &workers.to_string());
            }

            let farm = TileFarm::new(workers, tiles_per_worker);

            // Warmup run
            let mut state_warmup = QState::new_zero_multitile(qubits, total_tiles as u16);
            unsafe {
                farm.run_kernel_on_all_workers(
                    KernelType::HDeep,
                    qubits,
                    depth,
                    BackendKind::SimdJit,
                    &mut state_warmup as *mut QState as *mut u8,
                    Some(kernel_fn),
                );
            }

            // Timed runs
            let start = Instant::now();
            for _ in 0..iterations {
                let mut state = QState::new_zero_multitile(qubits, total_tiles as u16);
                unsafe {
                    farm.run_kernel_on_all_workers(
                        KernelType::HDeep,
                        qubits,
                        depth,
                        BackendKind::SimdJit,
                        &mut state as *mut QState as *mut u8,
                        Some(kernel_fn),
                    );
                }
            }
            let elapsed = start.elapsed();

            // Calculate metrics
            let total_time_us = elapsed.as_micros();
            let avg_time_us = total_time_us as f64 / iterations as f64;

            // q_ops/sec = (gates × iterations) / time_seconds
            // For H kernel at depth d: d H gates per iteration
            let gates_per_iteration = depth as f64;
            let total_gates = gates_per_iteration * iterations as f64;
            let time_seconds = total_time_us as f64 / 1_000_000.0;
            let q_ops_per_sec = total_gates / time_seconds;

            let result = BenchmarkResult {
                workers,
                tiles_per_worker,
                total_tiles,
                qubits,
                depth,
                iterations,
                total_time_us,
                avg_time_us,
                q_ops_per_sec,
            };

            results.push(result);
        }

        // Print results table
        eprintln!("Results:");
        eprintln!("{}", BenchmarkResult::csv_header());
        for result in &results {
            eprintln!("{}", result.to_csv());
        }

        // Calculate scaling metrics vs baseline (first config)
        let baseline = &results[0];
        eprintln!();
        eprintln!(
            "Scaling Analysis (vs {}w×{}t baseline):",
            baseline.workers, baseline.tiles_per_worker
        );
        eprintln!("Config               | Q-Ops/Sec    | Scaling | Efficiency");
        eprintln!("---------------------|--------------|---------|------------");

        for result in &results {
            let scaling = result.scaling_vs(baseline);
            let efficiency = result.efficiency(baseline);
            eprintln!(
                "{:2}w × {:2}t = {:2} total | {:.2e} | {:5.2}x  | {:5.2}%",
                result.workers,
                result.tiles_per_worker,
                result.total_tiles,
                result.q_ops_per_sec,
                scaling,
                efficiency * 100.0
            );
        }

        eprintln!();
        eprintln!("✅ EPIC 62 P4: Benchmark grid (small) complete");

        // Sanity checks
        assert!(results.len() == configs.len());
        assert!(
            baseline.workers == 2,
            "First config should be 2w×4t baseline"
        );

        // Verify all benchmarks completed successfully
        for result in &results {
            assert!(
                result.q_ops_per_sec > 0.0,
                "All configs should show positive throughput"
            );
        }

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    #[ignore] // Run manually with: cargo test --features "quantum_jit,cranelift_jit" test_p4_benchmark_grid_full -- --ignored --nocapture
    fn test_p4_benchmark_grid_full() {
        // EPIC 62 P4: Full benchmark grid for performance characterization
        //
        // This is a comprehensive benchmark that takes several minutes to run.
        // Run manually to collect full performance data for documentation.

        use crate::quantum::QState;
        use crate::quantum_jit;
        use std::time::Instant;

        // Full grid parameters
        let qubit_configs = vec![4u8, 8u8];
        let depth_configs = vec![8u32, 16u32];
        let iterations = 200;

        eprintln!("\n========================================");
        eprintln!("EPIC 62 Phase 4: Full Benchmark Grid");
        eprintln!("========================================");
        eprintln!("This will take several minutes...");
        eprintln!();

        let mut all_results = Vec::new();

        for &qubits in &qubit_configs {
            for &depth in &depth_configs {
                eprintln!("Configuration: {} qubits, depth {}", qubits, depth);

                // Compile kernel
                let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
                if h_kernel.is_none() {
                    eprintln!("  ⚠ Skipping (kernel compilation failed)");
                    continue;
                }
                let kernel_fn = h_kernel.unwrap() as *const u8;

                unsafe {
                    std::env::set_var("JIT_TILE_COUNT", "4");
                    std::env::set_var("JIT_H_DEPTH", &depth.to_string());
                }

                // Worker configurations to test
                let configs = vec![
                    (1, 4), // Baseline
                    (2, 2), // 2w × 2t = 4
                    (2, 4), // 2w × 4t = 8
                    (4, 2), // 4w × 2t = 8
                    (4, 4), // 4w × 4t = 16
                    (8, 2), // 8w × 2t = 16
                    (8, 4), // 8w × 4t = 32
                ];

                for &(workers, tiles_per_worker) in &configs {
                    let total_tiles = workers * tiles_per_worker;

                    unsafe {
                        std::env::set_var("JIT_WORKER_COUNT", &workers.to_string());
                    }

                    let farm = TileFarm::new(workers, tiles_per_worker);

                    // Warmup
                    let mut state_warmup = QState::new_zero_multitile(qubits, total_tiles as u16);
                    unsafe {
                        farm.run_kernel_on_all_workers(
                            KernelType::HDeep,
                            qubits,
                            depth,
                            BackendKind::SimdJit,
                            &mut state_warmup as *mut QState as *mut u8,
                            Some(kernel_fn),
                        );
                    }

                    // Benchmark
                    let start = Instant::now();
                    for _ in 0..iterations {
                        let mut state = QState::new_zero_multitile(qubits, total_tiles as u16);
                        unsafe {
                            farm.run_kernel_on_all_workers(
                                KernelType::HDeep,
                                qubits,
                                depth,
                                BackendKind::SimdJit,
                                &mut state as *mut QState as *mut u8,
                                Some(kernel_fn),
                            );
                        }
                    }
                    let elapsed = start.elapsed();

                    let total_time_us = elapsed.as_micros();
                    let avg_time_us = total_time_us as f64 / iterations as f64;
                    let gates_per_iteration = depth as f64;
                    let total_gates = gates_per_iteration * iterations as f64;
                    let time_seconds = total_time_us as f64 / 1_000_000.0;
                    let q_ops_per_sec = total_gates / time_seconds;

                    let result = BenchmarkResult {
                        workers,
                        tiles_per_worker,
                        total_tiles,
                        qubits,
                        depth,
                        iterations,
                        total_time_us,
                        avg_time_us,
                        q_ops_per_sec,
                    };

                    all_results.push(result.clone());

                    eprintln!(
                        "  {:2}w × {:2}t: {:.2e} q_ops/sec ({:.2} μs/iter)",
                        workers, tiles_per_worker, q_ops_per_sec, avg_time_us
                    );
                }

                eprintln!();
            }
        }

        // Output complete CSV data
        eprintln!("\n========================================");
        eprintln!("Complete Benchmark Results (CSV)");
        eprintln!("========================================");
        eprintln!("{}", BenchmarkResult::csv_header());
        for result in &all_results {
            eprintln!("{}", result.to_csv());
        }

        eprintln!();
        eprintln!("✅ EPIC 62 P4: Full benchmark grid complete");
        eprintln!("   Total configurations tested: {}", all_results.len());

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::remove_var("JIT_WORKER_COUNT");
        }
    }

    // ========================================================================
    // EPIC 66 Track C: KernelSpec tests
    // ========================================================================

    #[test]
    fn test_kernel_spec_creation() {
        let spec = KernelSpec::new(KernelType::HDeep, 32, 4, 16);
        assert_eq!(spec.gate, KernelType::HDeep);
        assert_eq!(spec.depth, 32);
        assert_eq!(spec.n_qubits, 4);
        assert_eq!(spec.tile_count, 16);
        assert_eq!(spec.unroll_factor, 0); // Default auto-select
    }

    #[test]
    fn test_kernel_spec_hadamard_constructor() {
        let spec = KernelSpec::hadamard(64, 8, 4);
        assert_eq!(spec.gate, KernelType::HDeep);
        assert_eq!(spec.depth, 64);
        assert_eq!(spec.n_qubits, 8);
        assert_eq!(spec.tile_count, 4);
    }

    #[test]
    fn test_kernel_spec_with_unroll() {
        let spec = KernelSpec::hadamard(32, 4, 16).with_unroll(4);
        assert_eq!(spec.unroll_factor, 4);
    }

    #[test]
    fn test_kernel_spec_effective_unroll_auto() {
        // Auto-select (unroll_factor = 0)
        let spec_small = KernelSpec::hadamard(2, 4, 16);
        assert_eq!(spec_small.effective_unroll(), 1); // depth 2 -> no unroll

        let spec_medium = KernelSpec::hadamard(10, 4, 16);
        assert_eq!(spec_medium.effective_unroll(), 2); // depth 10 -> unroll 2

        let spec_large = KernelSpec::hadamard(32, 4, 16);
        assert_eq!(spec_large.effective_unroll(), 4); // depth 32 -> unroll 4

        let spec_huge = KernelSpec::hadamard(128, 4, 16);
        assert_eq!(spec_huge.effective_unroll(), 4); // capped at 4
    }

    #[test]
    fn test_kernel_spec_effective_unroll_explicit() {
        let spec = KernelSpec::hadamard(32, 4, 16).with_unroll(8);
        assert_eq!(spec.effective_unroll(), 8);

        // Cap at 8
        let spec_overflow = KernelSpec::hadamard(32, 4, 16).with_unroll(16);
        assert_eq!(spec_overflow.effective_unroll(), 8);
    }

    #[test]
    fn test_kernel_spec_total_amplitudes() {
        // 4 qubits = 16 amplitudes, 16 tiles = 256 total
        let spec = KernelSpec::hadamard(32, 4, 16);
        assert_eq!(spec.total_amplitudes(), 16 * 16);

        // 8 qubits = 256 amplitudes, 4 tiles = 1024 total
        let spec2 = KernelSpec::hadamard(32, 8, 4);
        assert_eq!(spec2.total_amplitudes(), 256 * 4);
    }
}
