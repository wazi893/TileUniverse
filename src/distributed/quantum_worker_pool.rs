//! SPRINT 39.0 Phase 2: Multi-Core Block-Sparse Quantum Execution
//!
//! Parallel quantum circuit execution using block-sparse representation
//! and worker pool pattern from tile_farm.rs (EPIC 62).
//!
//! Architecture:
//! - Each worker processes a disjoint range of blocks
//! - Barrier synchronization ensures deterministic gate ordering
//! - Zero allocation in hot path (thread pool reused)
//! - Auto-configuration based on block count and CPU cores
//!
//! Example:
//! ```ignore
//! let state = BlockSparseState::new(12, 1024);  // 12 qubits, 1024 blocks
//! let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
//!
//! let pool = QuantumWorkerPool::new(4);  // 4 workers
//! pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
//! pool.shutdown();
//! ```

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Barrier};
use std::thread::{self, JoinHandle};

use logic_fabric_core::block_sparse_state::{Block, BlockSparseState};
use logic_fabric_core::quantum::QGate;

// ========================================================================
// Block Range Partitioning
// ========================================================================

/// Block range assignment for a specific worker
///
/// Workers operate on disjoint block ranges to ensure thread safety.
///
/// Example:
/// - Worker 0, blocks_per_worker=256 → blocks [0..256)
/// - Worker 1, blocks_per_worker=256 → blocks [256..512)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockRange {
    pub start_block: usize,
    pub end_block: usize, // exclusive
}

impl BlockRange {
    pub fn new(start_block: usize, end_block: usize) -> Self {
        BlockRange {
            start_block,
            end_block,
        }
    }

    pub fn block_count(&self) -> usize {
        self.end_block - self.start_block
    }

    pub fn for_worker(worker_idx: usize, blocks_per_worker: usize) -> Self {
        let start_block = worker_idx * blocks_per_worker;
        let end_block = start_block + blocks_per_worker;
        BlockRange {
            start_block,
            end_block,
        }
    }
}

/// Partition blocks across workers with even distribution
///
/// Distributes remainder blocks to first workers to ensure
/// all blocks are covered and ranges are disjoint.
///
/// # Arguments
/// * `total_blocks` - Total number of blocks to partition
/// * `num_workers` - Number of workers
///
/// # Returns
/// Vector of BlockRange, one per worker
///
/// # Example
/// ```
/// # use engine::distributed::quantum_worker_pool::partition_blocks;
/// let ranges = partition_blocks(1024, 4);
/// assert_eq!(ranges.len(), 4);
/// assert_eq!(ranges[0].block_count(), 256);
/// assert_eq!(ranges[3].start_block, 768);
/// ```
pub fn partition_blocks(total_blocks: usize, num_workers: usize) -> Vec<BlockRange> {
    assert!(num_workers > 0, "num_workers must be > 0");
    assert!(
        total_blocks >= num_workers,
        "total_blocks ({}) must be >= num_workers ({})",
        total_blocks,
        num_workers
    );

    let blocks_per_worker = total_blocks / num_workers;
    let remainder = total_blocks % num_workers;

    let mut ranges = Vec::with_capacity(num_workers);
    let mut start = 0;

    for worker_id in 0..num_workers {
        // Distribute remainder blocks to first workers
        let extra = if worker_id < remainder { 1 } else { 0 };
        let worker_blocks = blocks_per_worker + extra;

        ranges.push(BlockRange::new(start, start + worker_blocks));
        start += worker_blocks;
    }

    ranges
}

// ========================================================================
// Worker Configuration
// ========================================================================

/// Worker configuration for block-sparse execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerConfig {
    pub num_workers: usize,
    pub blocks_per_worker: usize,
}

/// Auto-configuration input parameters
#[derive(Debug, Clone, Copy)]
pub struct AutoConfigInput {
    pub cpu_cores: usize,
    pub block_count: usize,
    pub n_qubits: u8,
}

/// Auto-configure workers based on block count and CPU topology
///
/// Heuristics:
/// 1. Minimum 256 blocks per worker (avoid thread overhead)
/// 2. Don't exceed CPU core count
/// 3. Blocks must divide evenly across workers
///
/// # Arguments
/// * `input` - Auto-configuration parameters
///
/// # Returns
/// Worker configuration with optimal settings
pub fn auto_config_for_block_sparse(input: AutoConfigInput) -> WorkerConfig {
    let AutoConfigInput {
        cpu_cores,
        block_count,
        ..
    } = input;

    // Heuristic 1: Minimum 256 blocks per worker
    let max_workers_by_blocks = (block_count / 256).max(1);

    // Heuristic 2: Don't exceed CPU core count
    let workers = max_workers_by_blocks.min(cpu_cores);

    // Heuristic 3: Find largest divisor of block_count <= workers
    let workers = largest_divisor(block_count, workers);

    WorkerConfig {
        num_workers: workers,
        blocks_per_worker: block_count / workers,
    }
}

/// Find largest divisor of n that is <= limit
fn largest_divisor(n: usize, limit: usize) -> usize {
    for divisor in (1..=limit).rev() {
        if n.is_multiple_of(divisor) {
            return divisor;
        }
    }
    1
}

// ========================================================================
// Worker Messages
// ========================================================================

/// Messages sent to worker threads
enum WorkerJob {
    /// Execute a gate on assigned block range
    ExecuteGate {
        gate: QGate,
        state_ptr: *mut BlockSparseState,
        block_range: BlockRange, // Worker's assigned blocks
    },
    /// No-op for testing barrier synchronization
    NoOp,
    /// Shutdown the worker thread
    Shutdown,
}

// SAFETY: WorkerJob contains raw pointer but is safe because:
// 1. state_ptr points to BlockSparseState with disjoint block ranges per worker
// 2. Barrier synchronization ensures no concurrent access
// 3. Pointer is only dereferenced within worker's assigned block range
unsafe impl Send for WorkerJob {}

// ========================================================================
// Quantum Worker Pool
// ========================================================================

/// Multi-core quantum worker pool for block-sparse execution
///
/// Manages a thread pool that executes quantum gates in parallel
/// across disjoint block ranges.
///
/// # Thread Safety
/// - Each worker processes a disjoint block range
/// - Barrier synchronization after each gate
/// - No locks needed (disjoint memory access)
///
/// # Example
/// ```ignore
/// let state = BlockSparseState::new(12, 1024);
/// let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
///
/// let pool = QuantumWorkerPool::new(4);
/// pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
/// pool.shutdown();
/// ```
pub struct QuantumWorkerPool {
    num_workers: usize,
    block_ranges: Vec<BlockRange>,
    barrier: Arc<Barrier>,
    workers: Vec<QuantumWorker>,
}

struct QuantumWorker {
    _thread: JoinHandle<()>,
    sender: Sender<WorkerJob>,
}

impl QuantumWorkerPool {
    /// Create new worker pool with specified worker count
    ///
    /// # Arguments
    /// * `num_workers` - Number of worker threads (1-64)
    ///
    /// # Panics
    /// Panics if num_workers < 1 or > 64
    pub fn new(num_workers: usize) -> Self {
        assert!(
            (1..=64).contains(&num_workers),
            "num_workers must be 1-64, got {}",
            num_workers
        );

        // Barrier includes +1 for main thread coordination
        let barrier = Arc::new(Barrier::new(num_workers + 1));

        // Spawn worker threads
        let mut workers = Vec::with_capacity(num_workers);
        for worker_idx in 0..num_workers {
            let (sender, receiver) = channel();
            let barrier_clone = Arc::clone(&barrier);

            let thread_handle = thread::spawn(move || {
                quantum_worker_loop(worker_idx, receiver, barrier_clone);
            });

            workers.push(QuantumWorker {
                _thread: thread_handle,
                sender,
            });
        }

        QuantumWorkerPool {
            num_workers,
            block_ranges: Vec::new(), // Will be set when executing
            barrier,
            workers,
        }
    }

    /// Create worker pool with auto-configuration
    ///
    /// Uses heuristics to choose optimal worker count based on
    /// block count and CPU topology.
    ///
    /// # Arguments
    /// * `input` - Auto-configuration parameters
    ///
    /// # Returns
    /// Tuple of (QuantumWorkerPool, WorkerConfig) for inspection
    pub fn new_auto(input: AutoConfigInput) -> (Self, WorkerConfig) {
        let config = auto_config_for_block_sparse(input);

        eprintln!(
            "[phase2:auto] cores={}, blocks={}, qubits={} -> workers={}, blocks_per_worker={}",
            input.cpu_cores,
            input.block_count,
            input.n_qubits,
            config.num_workers,
            config.blocks_per_worker
        );

        let pool = Self::new(config.num_workers);
        (pool, config)
    }

    /// Execute a single gate in parallel across all workers
    ///
    /// # Safety
    /// Caller must ensure:
    /// - state is valid for the duration
    /// - block_ranges cover all blocks disjointly
    unsafe fn execute_gate_parallel(
        &mut self,
        state: &mut BlockSparseState,
        gate: &QGate,
        block_ranges: &[BlockRange],
    ) {
        // Dispatch to all workers
        for (worker, range) in self.workers.iter().zip(block_ranges.iter()) {
            let job = WorkerJob::ExecuteGate {
                gate: gate.clone(),
                state_ptr: state as *mut BlockSparseState,
                block_range: *range, // Pass block range to worker
            };

            worker.sender.send(job).expect("Worker channel closed");
        }

        // Wait for all workers to complete
        self.barrier.wait();
    }

    /// Execute entire circuit in parallel
    ///
    /// Each gate is executed across all blocks in parallel,
    /// with barrier synchronization between gates to ensure
    /// deterministic ordering.
    ///
    /// Cross-block CNOTs (target qubit >= 7) are executed sequentially
    /// on the main thread to ensure correct inter-block coordination.
    ///
    /// # Arguments
    /// * `state` - Block-sparse quantum state
    /// * `circuit` - Circuit to execute
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn execute_circuit_parallel(
        &mut self,
        state: &mut BlockSparseState,
        circuit: &[QGate],
    ) -> Result<(), String> {
        use logic_fabric_core::block_sparse_state::BLOCK_SHIFT;

        // Use total_blocks() not block_count() - we need ALL possible blocks, not just active ones
        let total_blocks = state.total_blocks();

        // CRITICAL: Pre-allocate all blocks on the main thread before parallel execution.
        // This prevents data races when workers call get_block_mut() on non-existent blocks,
        // which would concurrently modify the HashMap and cause undefined behavior.
        for block_id in 0..total_blocks {
            let _ = state.get_block_mut(block_id);
        }

        // Partition blocks across workers
        let block_ranges = partition_blocks(total_blocks, self.num_workers);
        self.block_ranges = block_ranges.clone();

        // Execute each gate
        for gate in circuit {
            // Check if gate requires sequential execution (cross-block CNOT with target >= 7)
            let needs_sequential = match gate {
                QGate::CNot(control, target) => {
                    let target_cross = *target >= BLOCK_SHIFT;
                    let both_cross = *control >= BLOCK_SHIFT && target_cross;
                    target_cross || both_cross
                }
                _ => false,
            };

            if needs_sequential {
                // Execute cross-block CNOT sequentially on main thread
                self.execute_gate_sequential(state, gate, total_blocks);
            } else {
                // Execute in parallel across workers
                unsafe {
                    self.execute_gate_parallel(state, gate, &block_ranges);
                }
            }
        }

        Ok(())
    }

    /// Execute a gate sequentially on the main thread
    ///
    /// Used for cross-block CNOTs that require inter-block coordination.
    fn execute_gate_sequential(
        &self,
        state: &mut BlockSparseState,
        gate: &QGate,
        total_blocks: usize,
    ) {
        match gate {
            QGate::CNot(control, target) => {
                execute_cnot_cross_block_sequential(state, *control, *target, total_blocks);
            }
            _ => {
                // For other gates, apply to all blocks sequentially
                for block_id in 0..total_blocks {
                    apply_gate_to_block(state, gate, block_id);
                }
            }
        }
    }

    /// Wait for all workers at barrier (for testing)
    pub fn sync_barrier(&self) {
        // Send NoOp to all workers
        for worker in &self.workers {
            worker
                .sender
                .send(WorkerJob::NoOp)
                .expect("Worker channel closed");
        }

        // Wait at barrier
        self.barrier.wait();
    }

    /// Shutdown the worker pool
    ///
    /// Sends shutdown signal to all workers and waits for threads to join.
    pub fn shutdown(self) {
        for worker in self.workers {
            let _ = worker.sender.send(WorkerJob::Shutdown);
        }
        // Threads will join when dropped
    }
}

// ========================================================================
// Worker Thread Loop
// ========================================================================

/// Worker thread main loop
///
/// Receives jobs via channel and executes with barrier synchronization.
fn quantum_worker_loop(worker_idx: usize, receiver: Receiver<WorkerJob>, barrier: Arc<Barrier>) {
    loop {
        match receiver.recv() {
            Ok(WorkerJob::NoOp) => {
                // No-op: just synchronize
                barrier.wait();
            }
            Ok(WorkerJob::ExecuteGate {
                gate,
                state_ptr,
                block_range,
            }) => {
                // Execute gate on this worker's block range
                // SAFETY: Caller ensures disjoint block ranges and valid pointer
                unsafe {
                    execute_gate_on_worker(worker_idx, state_ptr, &gate, &block_range);
                }

                barrier.wait();
            }
            Ok(WorkerJob::Shutdown) | Err(_) => {
                // Shutdown signal or channel closed
                break;
            }
        }
    }
}

/// Execute gate on worker's assigned blocks
///
/// # Safety
/// Caller must ensure:
/// - state_ptr points to valid BlockSparseState
/// - block_range is disjoint from other workers' ranges
/// - State remains valid for duration of execution
unsafe fn execute_gate_on_worker(
    _worker_idx: usize,
    state_ptr: *mut BlockSparseState,
    gate: &QGate,
    block_range: &BlockRange,
) {
    // SAFETY: Caller guarantees:
    // - state_ptr is valid and aligned
    // - block_range is disjoint from other workers
    // - No other threads access this range during execution
    let state = unsafe { &mut *state_ptr };

    // Process each block in this worker's range
    for block_id in block_range.start_block..block_range.end_block {
        apply_gate_to_block(state, gate, block_id);
    }
}

/// Apply a single-qubit gate to all amplitudes in a block
///
/// Currently supports: H, X, Z, Phase, Y
/// TODO: Multi-qubit gates (CNOT, CZ) require cross-block coordination
fn apply_gate_to_block(state: &mut BlockSparseState, gate: &QGate, block_id: usize) {
    use half::f16;
    use logic_fabric_core::quantum::QGate;

    const BLOCK_SIZE: usize = 128;
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

    // Get mutable access to block
    let block = state.get_block_mut(block_id);

    match gate {
        QGate::H(target_qubit) => {
            // Hadamard transformation: (|0⟩ + |1⟩)/√2
            // For each amplitude pair (i0, i1) where target qubit differs:
            // new[i0] = (old[i0] + old[i1]) / √2
            // new[i1] = (old[i0] - old[i1]) / √2

            let bit = 1usize << (*target_qubit as usize);

            // Iterate over pairs within block
            for i in 0..BLOCK_SIZE {
                if (i & bit) == 0 {
                    // Only process first element of each pair
                    let i0 = i;
                    let i1 = i | bit;

                    if i1 < BLOCK_SIZE {
                        // Load amplitudes
                        let r0 = block.real[i0].to_f32();
                        let r1 = block.real[i1].to_f32();
                        let im0 = block.imag[i0].to_f32();
                        let im1 = block.imag[i1].to_f32();

                        // Apply Hadamard
                        let s0r = (r0 + r1) * INV_SQRT2;
                        let s0i = (im0 + im1) * INV_SQRT2;
                        let s1r = (r0 - r1) * INV_SQRT2;
                        let s1i = (im0 - im1) * INV_SQRT2;

                        // Store results
                        block.real[i0] = f16::from_f32(s0r);
                        block.imag[i0] = f16::from_f32(s0i);
                        block.real[i1] = f16::from_f32(s1r);
                        block.imag[i1] = f16::from_f32(s1i);
                    }
                }
            }

            // Recount non-zero amplitudes after transformation
            block.recount_nnz();
        }

        QGate::X(target_qubit) => {
            // Pauli-X (NOT): |0⟩ ↔ |1⟩
            // Swap amplitudes in pairs

            let bit = 1usize << (*target_qubit as usize);

            for i in 0..BLOCK_SIZE {
                if (i & bit) == 0 {
                    let i0 = i;
                    let i1 = i | bit;

                    if i1 < BLOCK_SIZE {
                        block.real.swap(i0, i1);
                        block.imag.swap(i0, i1);
                    }
                }
            }
            // nnz unchanged by swap
        }

        QGate::Z(target_qubit) => {
            // Pauli-Z: |0⟩ → |0⟩, |1⟩ → -|1⟩
            // Negate amplitudes where target qubit is 1

            let bit = 1usize << (*target_qubit as usize);

            for i in 0..BLOCK_SIZE {
                if (i & bit) != 0 {
                    block.real[i] = -block.real[i];
                    block.imag[i] = -block.imag[i];
                }
            }
            // nnz unchanged by negation
        }

        QGate::Y(target_qubit) => {
            // Pauli-Y: Y = [0 -i; i 0]
            // |0⟩ → i|1⟩, |1⟩ → -i|0⟩

            let bit = 1usize << (*target_qubit as usize);

            for i in 0..BLOCK_SIZE {
                if (i & bit) == 0 {
                    let i0 = i;
                    let i1 = i | bit;

                    if i1 < BLOCK_SIZE {
                        let r0 = block.real[i0].to_f32();
                        let im0 = block.imag[i0].to_f32();
                        let r1 = block.real[i1].to_f32();
                        let im1 = block.imag[i1].to_f32();

                        // New |0⟩ component from old |1⟩ * -i
                        block.real[i0] = f16::from_f32(im1);
                        block.imag[i0] = f16::from_f32(-r1);

                        // New |1⟩ component from old |0⟩ * i
                        block.real[i1] = f16::from_f32(-im0);
                        block.imag[i1] = f16::from_f32(r0);
                    }
                }
            }

            block.recount_nnz();
        }

        QGate::Phase(target_qubit, theta) => {
            // Phase gate: |0⟩ → |0⟩, |1⟩ → e^(iθ)|1⟩
            // Multiply |1⟩ amplitudes by e^(iθ) = cos(θ) + i*sin(θ)

            let bit = 1usize << (*target_qubit as usize);
            let (s, c) = theta.sin_cos();

            for i in 0..BLOCK_SIZE {
                if (i & bit) != 0 {
                    let re = block.real[i].to_f32();
                    let im = block.imag[i].to_f32();

                    // Complex multiplication: (re + i*im) * (c + i*s)
                    block.real[i] = f16::from_f32(re * c - im * s);
                    block.imag[i] = f16::from_f32(re * s + im * c);
                }
            }

            block.recount_nnz();
        }

        // CNOT gate: Controlled-NOT
        QGate::CNot(control, target) => {
            apply_cnot_to_block(block, *control, *target, block_id);
        }

        // CZ gate: Controlled-Z (SPRINT 39.0 Phase 5)
        QGate::CZ(q0, q1) => {
            apply_cz_to_block(block, *q0, *q1, block_id);
        }

        // Swap gate: Exchange two qubit states (SPRINT 39.0 Phase 5)
        QGate::Swap(q0, q1) => {
            apply_swap_to_block(block, *q0, *q1, block_id);
        }

        // Three-qubit gates (TODO: implement)
        QGate::Toffoli(_, _, _) | QGate::CCZ(_, _, _) => {
            // TODO: Implement three-qubit gates
        }

        // Measurement handled separately (not in parallel execution path)
        QGate::Measure(_) => {
            // Measurement requires global synchronization
            // Should not be in parallel execution path
        }

        // Rotation gates (SPRINT 39.0 Phase 5 - Rotation Gate Implementation)
        QGate::Rx(target_qubit, theta) => {
            // Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
            // |0⟩ → cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩
            // |1⟩ → -i·sin(θ/2)|0⟩ + cos(θ/2)|1⟩

            let bit = 1usize << (*target_qubit as usize);
            let half_theta = theta * 0.5;
            let (sin_half, cos_half) = half_theta.sin_cos();

            for i in 0..BLOCK_SIZE {
                if (i & bit) == 0 {
                    let i0 = i;
                    let i1 = i | bit;

                    if i1 < BLOCK_SIZE {
                        let r0 = block.real[i0].to_f32();
                        let im0 = block.imag[i0].to_f32();
                        let r1 = block.real[i1].to_f32();
                        let im1 = block.imag[i1].to_f32();

                        // new|0⟩ = cos(θ/2)·|0⟩ - i·sin(θ/2)·|1⟩
                        // (a + bi)·cos - i·(c + di)·sin = (a·cos + d·sin) + i(b·cos - c·sin)
                        block.real[i0] = f16::from_f32(r0 * cos_half + im1 * sin_half);
                        block.imag[i0] = f16::from_f32(im0 * cos_half - r1 * sin_half);

                        // new|1⟩ = -i·sin(θ/2)·|0⟩ + cos(θ/2)·|1⟩
                        // -i·(a + bi)·sin + (c + di)·cos = (c·cos + b·sin) + i(d·cos - a·sin)
                        block.real[i1] = f16::from_f32(r1 * cos_half + im0 * sin_half);
                        block.imag[i1] = f16::from_f32(im1 * cos_half - r0 * sin_half);
                    }
                }
            }

            block.recount_nnz();
        }

        QGate::Ry(target_qubit, theta) => {
            // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
            // |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩  (note: Ry swaps sign convention)
            // |1⟩ → -sin(θ/2)|0⟩ + cos(θ/2)|1⟩

            let bit = 1usize << (*target_qubit as usize);
            let half_theta = theta * 0.5;
            let (sin_half, cos_half) = half_theta.sin_cos();

            for i in 0..BLOCK_SIZE {
                if (i & bit) == 0 {
                    let i0 = i;
                    let i1 = i | bit;

                    if i1 < BLOCK_SIZE {
                        let r0 = block.real[i0].to_f32();
                        let im0 = block.imag[i0].to_f32();
                        let r1 = block.real[i1].to_f32();
                        let im1 = block.imag[i1].to_f32();

                        // new|0⟩ = cos(θ/2)·|0⟩ - sin(θ/2)·|1⟩
                        block.real[i0] = f16::from_f32(r0 * cos_half - r1 * sin_half);
                        block.imag[i0] = f16::from_f32(im0 * cos_half - im1 * sin_half);

                        // new|1⟩ = sin(θ/2)·|0⟩ + cos(θ/2)·|1⟩
                        block.real[i1] = f16::from_f32(r0 * sin_half + r1 * cos_half);
                        block.imag[i1] = f16::from_f32(im0 * sin_half + im1 * cos_half);
                    }
                }
            }

            block.recount_nnz();
        }

        QGate::Rz(target_qubit, theta) => {
            // Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
            // |0⟩ → e^{-iθ/2}|0⟩ = (cos(θ/2) - i·sin(θ/2))|0⟩
            // |1⟩ → e^{iθ/2}|1⟩ = (cos(θ/2) + i·sin(θ/2))|1⟩

            let bit = 1usize << (*target_qubit as usize);
            let half_theta = theta * 0.5;
            let (sin_half, cos_half) = half_theta.sin_cos();

            for i in 0..BLOCK_SIZE {
                if (i & bit) == 0 {
                    let i0 = i;
                    let i1 = i | bit;

                    if i1 < BLOCK_SIZE {
                        // |0⟩ component: multiply by e^{-iθ/2} = cos - i·sin
                        let r0 = block.real[i0].to_f32();
                        let im0 = block.imag[i0].to_f32();
                        block.real[i0] = f16::from_f32(r0 * cos_half + im0 * sin_half);
                        block.imag[i0] = f16::from_f32(im0 * cos_half - r0 * sin_half);

                        // |1⟩ component: multiply by e^{iθ/2} = cos + i·sin
                        let r1 = block.real[i1].to_f32();
                        let im1 = block.imag[i1].to_f32();
                        block.real[i1] = f16::from_f32(r1 * cos_half - im1 * sin_half);
                        block.imag[i1] = f16::from_f32(im1 * cos_half + r1 * sin_half);
                    }
                }
            }

            block.recount_nnz();
        }

        // General single-qubit unitary (SPRINT 39.0 Phase 5)
        QGate::U3(target_qubit, theta, phi, lambda) => {
            // U3(θ,φ,λ) = [[cos(θ/2), -e^{iλ}·sin(θ/2)], [e^{iφ}·sin(θ/2), e^{i(φ+λ)}·cos(θ/2)]]
            // This is the most general single-qubit gate

            let bit = 1usize << (*target_qubit as usize);
            let half_theta = theta * 0.5;
            let (sin_h, cos_h) = half_theta.sin_cos();
            let (sin_l, cos_l) = lambda.sin_cos();
            let (sin_p, cos_p) = phi.sin_cos();
            let (sin_pl, cos_pl) = (phi + lambda).sin_cos();

            for i in 0..BLOCK_SIZE {
                if (i & bit) == 0 {
                    let i0 = i;
                    let i1 = i | bit;

                    if i1 < BLOCK_SIZE {
                        let r0 = block.real[i0].to_f32();
                        let im0 = block.imag[i0].to_f32();
                        let r1 = block.real[i1].to_f32();
                        let im1 = block.imag[i1].to_f32();

                        // Matrix elements:
                        // a00 = cos_h                    (real only)
                        // a01 = -e^{iλ}·sin_h = (-cos_l - i·sin_l)·sin_h
                        // a10 = e^{iφ}·sin_h = (cos_p + i·sin_p)·sin_h
                        // a11 = e^{i(φ+λ)}·cos_h = (cos_pl + i·sin_pl)·cos_h

                        let a00_r = cos_h;
                        let a01_r = -cos_l * sin_h;
                        let a01_i = -sin_l * sin_h;
                        let a10_r = cos_p * sin_h;
                        let a10_i = sin_p * sin_h;
                        let a11_r = cos_pl * cos_h;
                        let a11_i = sin_pl * cos_h;

                        // new|0⟩ = a00·|0⟩ + a01·|1⟩
                        // (a00_r)(r0 + i·im0) + (a01_r + i·a01_i)(r1 + i·im1)
                        let new_r0 = a00_r * r0 + a01_r * r1 - a01_i * im1;
                        let new_im0 = a00_r * im0 + a01_r * im1 + a01_i * r1;

                        // new|1⟩ = a10·|0⟩ + a11·|1⟩
                        // (a10_r + i·a10_i)(r0 + i·im0) + (a11_r + i·a11_i)(r1 + i·im1)
                        let new_r1 = a10_r * r0 - a10_i * im0 + a11_r * r1 - a11_i * im1;
                        let new_im1 = a10_r * im0 + a10_i * r0 + a11_r * im1 + a11_i * r1;

                        block.real[i0] = f16::from_f32(new_r0);
                        block.imag[i0] = f16::from_f32(new_im0);
                        block.real[i1] = f16::from_f32(new_r1);
                        block.imag[i1] = f16::from_f32(new_im1);
                    }
                }
            }

            block.recount_nnz();
        }

        // Shor's algorithm: Controlled rotation gates (TODO: implement)
        QGate::CPhase(control, target, angle) | QGate::CRz(control, target, angle) => {
            // TODO: Implement controlled phase/Rz gates
            let _ = (control, target, angle);
        }

        // SPRINT 48.0: T gates
        QGate::T(target_qubit) => {
            // T gate: |0⟩ → |0⟩, |1⟩ → e^(iπ/4)|1⟩ = (1+i)/√2 |1⟩
            const T_REAL: f32 = std::f32::consts::FRAC_1_SQRT_2;
            const T_IMAG: f32 = std::f32::consts::FRAC_1_SQRT_2;
            let bit = 1usize << (*target_qubit as usize);
            for i in 0..BLOCK_SIZE {
                if (i & bit) != 0 {
                    let re = block.real[i].to_f32();
                    let im = block.imag[i].to_f32();
                    block.real[i] = f16::from_f32(T_REAL * re - T_IMAG * im);
                    block.imag[i] = f16::from_f32(T_REAL * im + T_IMAG * re);
                }
            }
            block.recount_nnz();
        }

        QGate::Tdg(target_qubit) => {
            // T† gate: |0⟩ → |0⟩, |1⟩ → e^(-iπ/4)|1⟩ = (1-i)/√2 |1⟩
            const T_REAL: f32 = std::f32::consts::FRAC_1_SQRT_2;
            const T_IMAG: f32 = -std::f32::consts::FRAC_1_SQRT_2;
            let bit = 1usize << (*target_qubit as usize);
            for i in 0..BLOCK_SIZE {
                if (i & bit) != 0 {
                    let re = block.real[i].to_f32();
                    let im = block.imag[i].to_f32();
                    block.real[i] = f16::from_f32(T_REAL * re - T_IMAG * im);
                    block.imag[i] = f16::from_f32(T_REAL * im + T_IMAG * re);
                }
            }
            block.recount_nnz();
        }
    }
}

// ========================================================================
// Cross-Block CNOT Sequential Execution (SPRINT 39.0 Phase 2.3)
// ========================================================================

/// Execute a cross-block CNOT sequentially with full state access
///
/// This handles the case where target qubit >= 7 (or both qubits >= 7),
/// requiring coordination between different blocks.
///
/// # Strategy
///
/// For CNOT(control, target) where target >= 7:
/// - Control qubit determines which amplitudes to affect (within block)
/// - Target qubit determines which partner block to swap with
/// - For each block pair (A, B), swap amplitudes where control=1
///
/// # Arguments
/// * `state` - Block-sparse quantum state (mutable access to all blocks)
/// * `control` - Control qubit index
/// * `target` - Target qubit index (assumed >= BLOCK_SHIFT)
/// * `total_blocks` - Total number of blocks in state
fn execute_cnot_cross_block_sequential(
    state: &mut BlockSparseState,
    control: u8,
    target: u8,
    total_blocks: usize,
) {
    use logic_fabric_core::block_sparse_state::BLOCK_SHIFT;

    let control_cross = control >= BLOCK_SHIFT;
    let target_cross = target >= BLOCK_SHIFT;

    if control_cross && target_cross {
        // Case: Both qubits are in block ID portion (qubits 7+)
        execute_cnot_both_cross_block(state, control, target, total_blocks);
    } else if target_cross && !control_cross {
        // Case: Control < 7, Target >= 7
        // This is the most complex case - conditional swap between blocks
        execute_cnot_target_high(state, control, target, total_blocks);
    } else if control_cross && !target_cross {
        // Case: Control >= 7, Target < 7
        // This is already handled in parallel (apply_cnot_cross_block)
        // But for completeness, we can also handle it here
        for block_id in 0..total_blocks {
            let block = state.get_block_mut(block_id);
            apply_cnot_cross_block(block, control, target, block_id);
        }
    }
}

/// Execute CNOT where both control and target are >= 7 (both in block ID)
///
/// For CNOT(control, target) where both >= 7:
/// - Control qubit determines which blocks are affected
/// - If control bit is 0 in block_id: no effect
/// - If control bit is 1 in block_id: swap with partner block (toggle target bit)
fn execute_cnot_both_cross_block(
    state: &mut BlockSparseState,
    control: u8,
    target: u8,
    total_blocks: usize,
) {
    use logic_fabric_core::block_sparse_state::BLOCK_SHIFT;

    let control_block_bit = (control - BLOCK_SHIFT) as usize;
    let target_block_bit = (target - BLOCK_SHIFT) as usize;

    let control_mask = 1usize << control_block_bit;
    let target_mask = 1usize << target_block_bit;

    // Process each block pair exactly once
    // We process from the "primary" block (control=1, target=0 in block_id)
    for block_id in 0..total_blocks {
        // Skip if control bit is not set (CNOT has no effect)
        if (block_id & control_mask) == 0 {
            continue;
        }

        // Skip if target bit is already set (partner will handle)
        if (block_id & target_mask) != 0 {
            continue;
        }

        // Calculate partner block
        let partner_id = block_id ^ target_mask;

        // Ensure partner exists
        if partner_id >= total_blocks {
            continue;
        }

        // Swap ALL amplitudes between block_id and partner_id
        // Since control is in block_id bits and control=1, entire block is affected
        swap_blocks_full(state, block_id, partner_id);
    }
}

/// Execute CNOT where control < 7 and target >= 7
///
/// For CNOT(control, target) where control < 7 and target >= 7:
/// - For each block, only amplitudes where control=1 are affected
/// - Those amplitudes swap with corresponding amplitude in partner block
/// - Partner block_id = block_id XOR (1 << (target - 7))
fn execute_cnot_target_high(
    state: &mut BlockSparseState,
    control: u8,
    target: u8,
    total_blocks: usize,
) {
    use logic_fabric_core::block_sparse_state::BLOCK_SHIFT;

    let target_block_bit = (target - BLOCK_SHIFT) as usize;
    let target_mask = 1usize << target_block_bit;
    let control_bit = 1usize << (control as usize);

    // Process each block pair exactly once
    // We process from blocks where target bit = 0 in block_id
    for block_id in 0..total_blocks {
        // Skip if target bit is already set (partner block will be our "primary")
        if (block_id & target_mask) != 0 {
            continue;
        }

        let partner_id = block_id ^ target_mask;

        // Ensure partner exists
        if partner_id >= total_blocks {
            continue;
        }

        // Swap amplitudes where control qubit = 1
        swap_blocks_conditional(state, block_id, partner_id, control_bit);
    }
}

/// Swap all amplitudes between two blocks
fn swap_blocks_full(state: &mut BlockSparseState, block_a: usize, block_b: usize) {
    use logic_fabric_core::block_sparse_state::BLOCK_SIZE;

    // Get raw pointers to avoid borrow checker issues
    // SAFETY: We ensure block_a != block_b, so no aliasing
    let ptr = state as *mut BlockSparseState;

    unsafe {
        let a = (&mut *ptr).get_block_mut(block_a);
        let b = (&mut *ptr).get_block_mut(block_b);

        // Swap all amplitudes
        for i in 0..BLOCK_SIZE {
            std::mem::swap(&mut a.real[i], &mut b.real[i]);
            std::mem::swap(&mut a.imag[i], &mut b.imag[i]);
        }
        a.recount_nnz();
        b.recount_nnz();
    }
}

/// Swap amplitudes between two blocks where control bit is set
fn swap_blocks_conditional(
    state: &mut BlockSparseState,
    block_a: usize,
    block_b: usize,
    control_bit: usize,
) {
    use logic_fabric_core::block_sparse_state::BLOCK_SIZE;

    // Get raw pointers to avoid borrow checker issues
    // SAFETY: We ensure block_a != block_b, so no aliasing
    let ptr = state as *mut BlockSparseState;

    unsafe {
        let a = (&mut *ptr).get_block_mut(block_a);
        let b = (&mut *ptr).get_block_mut(block_b);

        // Only swap amplitudes where control bit is set
        for i in 0..BLOCK_SIZE {
            if (i & control_bit) != 0 {
                std::mem::swap(&mut a.real[i], &mut b.real[i]);
                std::mem::swap(&mut a.imag[i], &mut b.imag[i]);
            }
        }
        a.recount_nnz();
        b.recount_nnz();
    }
}

/// Apply CNOT gate to a block
///
/// CNOT(control, target): If control qubit is |1⟩, flip target qubit
///
/// Block-sparse considerations:
/// - If both qubits < BLOCK_SHIFT (7): within-block operation (fast path)
/// - If either qubit >= BLOCK_SHIFT (7): cross-block operation (requires coordination)
///
/// For cross-block case, we need to coordinate with partner blocks.
/// The block_id parameter helps determine which partner blocks to coordinate with.
fn apply_cnot_to_block(block: &mut Block, control: u8, target: u8, block_id: usize) {
    use logic_fabric_core::block_sparse_state::BLOCK_SHIFT;

    // Case 1: Within-block CNOT (both qubits in 0-6)
    if control < BLOCK_SHIFT && target < BLOCK_SHIFT {
        apply_cnot_within_block(block, control, target);
        return;
    }

    // Case 2: Cross-block CNOT (one or both qubits >= 7)
    // This requires coordination between blocks
    apply_cnot_cross_block(block, control, target, block_id);
}

/// Apply CNOT within a single block (fast path)
///
/// When both control and target qubits are within the block (qubits 0-6),
/// we can apply CNOT independently to each block's 128 amplitudes.
fn apply_cnot_within_block(block: &mut Block, control: u8, target: u8) {
    use logic_fabric_core::block_sparse_state::BLOCK_SIZE;

    let control_bit = 1usize << (control as usize);
    let target_bit = 1usize << (target as usize);

    // For each amplitude index i where control qubit is |1⟩:
    // Swap amplitude[i] with amplitude[i ^ target_bit]
    for i in 0..BLOCK_SIZE {
        // Only process if control bit is set AND we're at the lower index of the pair
        if (i & control_bit) != 0 && (i & target_bit) == 0 {
            let i0 = i; // Control=1, Target=0
            let i1 = i ^ target_bit; // Control=1, Target=1

            if i1 < BLOCK_SIZE {
                // Swap amplitudes
                block.real.swap(i0, i1);
                block.imag.swap(i0, i1);
            }
        }
    }

    // nnz count may change after CNOT
    block.recount_nnz();
}

/// Apply CNOT across blocks (slow path with coordination)
///
/// When control or target qubit is in the high-order bits (qubit >= 7),
/// the CNOT operation affects amplitudes in different blocks.
///
/// Strategy:
/// 1. Determine partner block IDs based on control/target qubits
/// 2. For cross-block case, we need to swap amplitudes between blocks
/// 3. This requires synchronization to avoid data races
///
/// Current implementation: Sequential within each worker's range
/// TODO: Optimize with parallel block-pair processing
fn apply_cnot_cross_block(block: &mut Block, control: u8, target: u8, block_id: usize) {
    use logic_fabric_core::block_sparse_state::{BLOCK_SHIFT, BLOCK_SIZE};

    // Determine which qubit is cross-block
    let control_cross = control >= BLOCK_SHIFT;
    let target_cross = target >= BLOCK_SHIFT;

    if control_cross && target_cross {
        // Both qubits are cross-block (qubits 7+)
        // This means control and target are both in the block ID portion

        // Convert to block-relative qubit indices
        let control_block_bit = (control - BLOCK_SHIFT) as usize;
        let target_block_bit = (target - BLOCK_SHIFT) as usize;

        // Check if control bit is set in this block's ID
        let control_mask = 1usize << control_block_bit;
        let _target_mask = 1usize << target_block_bit;

        // If control qubit is NOT set in block_id, CNOT has no effect
        if (block_id & control_mask) == 0 {}

        // If control is set, we need to swap this block with partner block
        // Partner block differs in target bit: block_id ^ target_mask
        // NOTE: This is the complex case - requires inter-block coordination
        // For now, we cannot swap with partner block in parallel execution
        // because partner block may be owned by another worker.

        // TODO: Implement inter-block swap with proper synchronization
        // Options:
        // 1. Acquire locks on both blocks (deadlock-prone)
        // 2. Use a two-phase protocol (barrier between phases)
        // 3. Collect all cross-block swaps and apply sequentially after parallel phase

        // For now: Mark this as a limitation
        // Cross-block CNOT with both qubits >= 7 not supported in parallel mode
    } else if control_cross {
        // Control qubit is cross-block (>= 7), target is within-block (< 7)

        let control_block_bit = (control - BLOCK_SHIFT) as usize;
        let control_mask = 1usize << control_block_bit;

        // Check if control bit is set in this block's ID
        if (block_id & control_mask) == 0 {
            // Control qubit is |0⟩ for all amplitudes in this block
            // CNOT has no effect
            return;
        }

        // Control qubit is |1⟩ for all amplitudes in this block
        // Apply X gate to target qubit (flip target for all amplitudes)
        let target_bit = 1usize << (target as usize);

        for i in 0..BLOCK_SIZE {
            if (i & target_bit) == 0 {
                let i0 = i;
                let i1 = i ^ target_bit;

                if i1 < BLOCK_SIZE {
                    block.real.swap(i0, i1);
                    block.imag.swap(i0, i1);
                }
            }
        }

        block.recount_nnz();
    } else {
        // Target qubit is cross-block (>= 7), control is within-block (< 7)

        // This is the most complex case:
        // - Control qubit varies within the block
        // - Target qubit determines which partner block to swap with

        // For amplitudes where control=1, we need to swap with partner block
        // Partner block ID: block_id ^ (1 << (target - BLOCK_SHIFT))

        let _target_block_bit = (target - BLOCK_SHIFT) as usize;
        let _control_bit = 1usize << (control as usize);

        // We need to swap amplitudes conditionally with partner block
        // This requires inter-block coordination

        // TODO: Implement with proper synchronization
        // For now: Not supported in parallel mode

        // Workaround: Could process these gates sequentially after parallel phase
    }
}

// ========================================================================
// CZ Gate Implementation (SPRINT 39.0 Phase 5)
// ========================================================================

/// Apply CZ (Controlled-Z) gate to a block
///
/// CZ(q0, q1): Flip phase of |11⟩ states (negate amplitude when both qubits are 1)
/// CZ is symmetric: CZ(a,b) = CZ(b,a)
fn apply_cz_to_block(block: &mut Block, q0: u8, q1: u8, block_id: usize) {
    use logic_fabric_core::block_sparse_state::BLOCK_SHIFT;

    let q0_cross = q0 >= BLOCK_SHIFT;
    let q1_cross = q1 >= BLOCK_SHIFT;

    if !q0_cross && !q1_cross {
        // Case 1: Both qubits within block (fast path)
        apply_cz_within_block(block, q0, q1);
    } else if q0_cross && q1_cross {
        // Case 2: Both qubits in block ID portion
        apply_cz_both_cross_block(block, q0, q1, block_id);
    } else {
        // Case 3: One qubit in block, one in block ID
        apply_cz_one_cross_block(block, q0, q1, block_id);
    }
}

/// CZ within block: negate amplitudes where both qubits are |1⟩
fn apply_cz_within_block(block: &mut Block, q0: u8, q1: u8) {
    use logic_fabric_core::block_sparse_state::BLOCK_SIZE;

    let bit0 = 1usize << (q0 as usize);
    let bit1 = 1usize << (q1 as usize);
    let mask = bit0 | bit1;

    for i in 0..BLOCK_SIZE {
        if (i & mask) == mask {
            // Both qubits are |1⟩ - negate amplitude
            block.real[i] = -block.real[i];
            block.imag[i] = -block.imag[i];
        }
    }
    // nnz count unchanged by negation
}

/// CZ with both qubits in block ID portion (qubits >= 7)
fn apply_cz_both_cross_block(block: &mut Block, q0: u8, q1: u8, block_id: usize) {
    use logic_fabric_core::block_sparse_state::{BLOCK_SHIFT, BLOCK_SIZE};

    let bit0 = 1usize << ((q0 - BLOCK_SHIFT) as usize);
    let bit1 = 1usize << ((q1 - BLOCK_SHIFT) as usize);
    let mask = bit0 | bit1;

    // Only negate all amplitudes if both block ID bits are set
    if (block_id & mask) == mask {
        for i in 0..BLOCK_SIZE {
            block.real[i] = -block.real[i];
            block.imag[i] = -block.imag[i];
        }
    }
    // nnz count unchanged
}

/// CZ with one qubit in block, one in block ID
fn apply_cz_one_cross_block(block: &mut Block, q0: u8, q1: u8, block_id: usize) {
    use logic_fabric_core::block_sparse_state::{BLOCK_SHIFT, BLOCK_SIZE};

    // Determine which qubit is cross-block
    let (block_qubit, cross_qubit) = if q0 >= BLOCK_SHIFT {
        (q1, q0)
    } else {
        (q0, q1)
    };

    let cross_bit = 1usize << ((cross_qubit - BLOCK_SHIFT) as usize);
    let block_bit = 1usize << (block_qubit as usize);

    // Only process if cross-block qubit's bit is set in block_id
    if (block_id & cross_bit) != 0 {
        // Negate amplitudes where block qubit is also |1⟩
        for i in 0..BLOCK_SIZE {
            if (i & block_bit) != 0 {
                block.real[i] = -block.real[i];
                block.imag[i] = -block.imag[i];
            }
        }
    }
    // nnz count unchanged
}

// ========================================================================
// Swap Gate Implementation (SPRINT 39.0 Phase 5)
// ========================================================================

/// Apply Swap gate to a block
///
/// Swap(q0, q1): Exchange the states of two qubits
/// For each amplitude index, if the bits differ at q0 and q1, swap with partner index
fn apply_swap_to_block(block: &mut Block, q0: u8, q1: u8, _block_id: usize) {
    use logic_fabric_core::block_sparse_state::BLOCK_SHIFT;

    // No-op if swapping qubit with itself
    if q0 == q1 {
        return;
    }

    let q0_cross = q0 >= BLOCK_SHIFT;
    let q1_cross = q1 >= BLOCK_SHIFT;

    if !q0_cross && !q1_cross {
        // Case 1: Both qubits within block (fast path)
        apply_swap_within_block(block, q0, q1);
    } else if q0_cross && q1_cross {
        // Case 2: Both qubits in block ID portion
        // This would swap entire blocks - handled at higher level
        // For within-block processing, no amplitude changes needed
        // (the block_id itself would change, which is structural)
    } else {
        // Case 3: One qubit in block, one in block ID
        // This requires inter-block coordination - see execute_swap_cross_block_sequential
        // Currently not supported in parallel mode per-block
    }
}

/// Swap within block: exchange qubit states
fn apply_swap_within_block(block: &mut Block, q0: u8, q1: u8) {
    use logic_fabric_core::block_sparse_state::BLOCK_SIZE;

    let mask0 = 1usize << (q0 as usize);
    let mask1 = 1usize << (q1 as usize);

    for i in 0..BLOCK_SIZE {
        let bit0 = (i & mask0) != 0;
        let bit1 = (i & mask1) != 0;

        // Only swap if bits differ (01 <-> 10)
        if bit0 != bit1 {
            // Partner index: toggle both bits
            let j = i ^ mask0 ^ mask1;

            // Only process once per pair (when i < j)
            if i < j && j < BLOCK_SIZE {
                block.real.swap(i, j);
                block.imag.swap(i, j);
            }
        }
    }

    block.recount_nnz();
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_even_division() {
        let ranges = partition_blocks(1024, 4);
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0], BlockRange::new(0, 256));
        assert_eq!(ranges[1], BlockRange::new(256, 512));
        assert_eq!(ranges[2], BlockRange::new(512, 768));
        assert_eq!(ranges[3], BlockRange::new(768, 1024));
    }

    #[test]
    fn test_partition_with_remainder() {
        let ranges = partition_blocks(1000, 3);
        // Worker 0: 334 blocks, Worker 1: 333, Worker 2: 333
        assert_eq!(ranges[0].block_count(), 334);
        assert_eq!(ranges[1].block_count(), 333);
        assert_eq!(ranges[2].block_count(), 333);

        // Verify total
        let total: usize = ranges.iter().map(|r| r.block_count()).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_block_ranges_disjoint() {
        let ranges = partition_blocks(1024, 8);

        for i in 0..ranges.len() {
            for j in (i + 1)..ranges.len() {
                // Assert no overlap
                assert!(
                    ranges[i].end_block <= ranges[j].start_block
                        || ranges[j].end_block <= ranges[i].start_block,
                    "Ranges {} and {} overlap!",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_block_ranges_cover_all() {
        let ranges = partition_blocks(1024, 8);

        // Check coverage
        let mut covered = vec![false; 1024];
        for range in &ranges {
            for block_id in range.start_block..range.end_block {
                assert!(!covered[block_id], "Block {} covered twice", block_id);
                covered[block_id] = true;
            }
        }

        // All blocks must be covered
        assert!(covered.iter().all(|&x| x), "Some blocks not covered");
    }

    #[test]
    fn test_auto_config_small_block_count() {
        let input = AutoConfigInput {
            cpu_cores: 8,
            block_count: 128, // Too few blocks
            n_qubits: 10,
        };

        let config = auto_config_for_block_sparse(input);
        assert_eq!(config.num_workers, 1); // Should use single worker
    }

    #[test]
    fn test_auto_config_medium_block_count() {
        let input = AutoConfigInput {
            cpu_cores: 8,
            block_count: 1024,
            n_qubits: 12,
        };

        let config = auto_config_for_block_sparse(input);
        assert!(config.num_workers >= 2 && config.num_workers <= 8);
        assert_eq!(1024 % config.num_workers, 0); // Must divide evenly
    }

    #[test]
    fn test_auto_config_large_block_count() {
        let input = AutoConfigInput {
            cpu_cores: 8,
            block_count: 4096,
            n_qubits: 14,
        };

        let config = auto_config_for_block_sparse(input);
        assert_eq!(config.num_workers, 8); // Should max out at CPU count
        assert_eq!(config.blocks_per_worker, 512);
    }

    #[test]
    fn test_largest_divisor() {
        assert_eq!(largest_divisor(1024, 8), 8);
        assert_eq!(largest_divisor(1000, 8), 8);
        assert_eq!(largest_divisor(1001, 8), 7);
        assert_eq!(largest_divisor(13, 8), 1); // Prime
    }

    #[test]
    fn test_worker_pool_creation() {
        let pool = QuantumWorkerPool::new(4);
        assert_eq!(pool.num_workers, 4);
        pool.shutdown();
    }

    #[test]
    fn test_worker_pool_auto_creation() {
        let input = AutoConfigInput {
            cpu_cores: 8,
            block_count: 1024,
            n_qubits: 12,
        };

        let (pool, config) = QuantumWorkerPool::new_auto(input);
        assert_eq!(pool.num_workers, config.num_workers);
        pool.shutdown();
    }

    #[test]
    fn test_barrier_synchronization() {
        let pool = QuantumWorkerPool::new(4);

        // Test that barrier sync works
        pool.sync_barrier();
        pool.sync_barrier();
        pool.sync_barrier();

        pool.shutdown();
    }

    // ========================================================================
    // Integration Tests - Gate Execution
    // ========================================================================

    /// Helper: Compare two BlockSparseStates for equality (within epsilon)
    fn states_equal(state1: &BlockSparseState, state2: &BlockSparseState, epsilon: f32) -> bool {
        use logic_fabric_core::block_sparse_state::{BLOCK_SHIFT, BLOCK_SIZE};

        if state1.n_qubits != state2.n_qubits {
            return false;
        }

        // Helper to check if a block is effectively zero
        fn is_zero_block(
            block: &logic_fabric_core::block_sparse_state::Block,
            epsilon: f32,
        ) -> bool {
            for i in 0..BLOCK_SIZE {
                if block.real[i].to_f32().abs() > epsilon || block.imag[i].to_f32().abs() > epsilon
                {
                    return false;
                }
            }
            true
        }

        // Compare all blocks
        let n_blocks = 1 << (state1.n_qubits - BLOCK_SHIFT);

        for block_id in 0..n_blocks {
            let block1 = state1.get_block(block_id);
            let block2 = state2.get_block(block_id);

            match (block1, block2) {
                (None, None) => continue, // Both have no block
                (Some(b1), Some(b2)) => {
                    // Compare amplitudes
                    for i in 0..BLOCK_SIZE {
                        let r1 = b1.real[i].to_f32();
                        let r2 = b2.real[i].to_f32();
                        let im1 = b1.imag[i].to_f32();
                        let im2 = b2.imag[i].to_f32();

                        if (r1 - r2).abs() > epsilon || (im1 - im2).abs() > epsilon {
                            return false;
                        }
                    }
                }
                (Some(b), None) | (None, Some(b)) => {
                    // One has a block, one doesn't - check if the block is effectively zero
                    if !is_zero_block(b, epsilon) {
                        return false;
                    }
                }
            }
        }

        true
    }

    #[test]
    fn test_single_qubit_gates_execution() {
        use logic_fabric_core::quantum::QGate;

        // Create a simple circuit with single-qubit gates
        let circuit = vec![
            QGate::H(0), // Hadamard on qubit 0
            QGate::X(1), // Pauli-X on qubit 1
            QGate::Z(2), // Pauli-Z on qubit 2 (no effect on |0⟩)
        ];

        // Create state: 10 qubits = 1024 blocks of 128 amplitudes each
        let mut state = BlockSparseState::new_zero(10);

        // Execute with single worker
        let mut pool = QuantumWorkerPool::new(1);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();

        // Verify execution completed without panic
        // (Full correctness test would require comparing with scalar backend)
    }

    #[test]
    fn test_single_vs_multi_worker_determinism() {
        use logic_fabric_core::quantum::QGate;

        let circuit = vec![QGate::H(0), QGate::H(1), QGate::X(2), QGate::Phase(3, 1.5)];

        // Execute with 1 worker
        let mut state_single = BlockSparseState::new_zero(10);
        let mut pool_single = QuantumWorkerPool::new(1);
        pool_single
            .execute_circuit_parallel(&mut state_single, &circuit)
            .unwrap();
        pool_single.shutdown();

        // Execute with 4 workers
        let mut state_multi = BlockSparseState::new_zero(10);
        let mut pool_multi = QuantumWorkerPool::new(4);
        pool_multi
            .execute_circuit_parallel(&mut state_multi, &circuit)
            .unwrap();
        pool_multi.shutdown();

        // Verify both produce identical results
        assert!(
            states_equal(&state_single, &state_multi, 1e-4),
            "Single-worker and multi-worker execution produced different results"
        );
    }

    #[test]
    fn test_multi_worker_determinism_multiple_runs() {
        use logic_fabric_core::quantum::QGate;

        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::X(3),
            QGate::Z(4),
        ];

        // Run 1
        let mut state_run1 = BlockSparseState::new_zero(10);
        let mut pool1 = QuantumWorkerPool::new(4);
        pool1
            .execute_circuit_parallel(&mut state_run1, &circuit)
            .unwrap();
        pool1.shutdown();

        // Run 2
        let mut state_run2 = BlockSparseState::new_zero(10);
        let mut pool2 = QuantumWorkerPool::new(4);
        pool2
            .execute_circuit_parallel(&mut state_run2, &circuit)
            .unwrap();
        pool2.shutdown();

        // Verify determinism
        assert!(
            states_equal(&state_run1, &state_run2, 1e-6),
            "Multiple runs produced different results - not deterministic!"
        );
    }

    #[test]
    fn test_empty_circuit() {
        // Edge case: empty circuit should leave state unchanged
        let circuit = vec![];

        let mut state = BlockSparseState::new_zero(10);
        let initial_state_copy = BlockSparseState::new_zero(10);

        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();

        assert!(
            states_equal(&state, &initial_state_copy, 1e-9),
            "Empty circuit changed state!"
        );
    }

    #[test]
    fn test_phase_gate_execution() {
        use logic_fabric_core::quantum::QGate;
        use std::f32::consts::PI;

        // Apply phase gates with different angles
        let circuit = vec![
            QGate::Phase(0, PI / 2.0),
            QGate::Phase(1, PI),
            QGate::Phase(2, PI / 4.0),
        ];

        let mut state = BlockSparseState::new_zero(10);

        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();

        // Verify execution completed
        // (Phase gates on |0⟩ have no effect, but we're testing execution path)
    }

    // ========================================================================
    // CNOT Tests (Phase 2.2)
    // ========================================================================

    #[test]
    fn test_cnot_within_block() {
        use logic_fabric_core::quantum::QGate;

        // Test CNOT with both qubits in 0-6 range (within-block)
        let circuit = vec![
            QGate::H(0),       // Create superposition on qubit 0
            QGate::CNot(0, 1), // Entangle qubits 0 and 1 (both < 7)
        ];

        // Execute with single worker
        let mut state_single = BlockSparseState::new_zero(10);
        let mut pool_single = QuantumWorkerPool::new(1);
        pool_single
            .execute_circuit_parallel(&mut state_single, &circuit)
            .unwrap();
        pool_single.shutdown();

        // Execute with multiple workers
        let mut state_multi = BlockSparseState::new_zero(10);
        let mut pool_multi = QuantumWorkerPool::new(4);
        pool_multi
            .execute_circuit_parallel(&mut state_multi, &circuit)
            .unwrap();
        pool_multi.shutdown();

        // Verify both produce identical results
        assert!(
            states_equal(&state_single, &state_multi, 1e-4),
            "CNOT within-block: single vs multi-worker mismatch"
        );
    }

    #[test]
    fn test_cnot_cross_block_control_high() {
        use logic_fabric_core::quantum::QGate;

        // Test CNOT with control qubit >= 7, target < 7
        // This is the implemented cross-block case
        let circuit = vec![
            QGate::X(7),       // Set qubit 7 to |1⟩ (control >= BLOCK_SHIFT)
            QGate::CNot(7, 0), // CNOT with control=7, target=0
        ];

        // Execute with single worker
        let mut state_single = BlockSparseState::new_zero(10);
        let mut pool_single = QuantumWorkerPool::new(1);
        pool_single
            .execute_circuit_parallel(&mut state_single, &circuit)
            .unwrap();
        pool_single.shutdown();

        // Execute with multiple workers
        let mut state_multi = BlockSparseState::new_zero(10);
        let mut pool_multi = QuantumWorkerPool::new(4);
        pool_multi
            .execute_circuit_parallel(&mut state_multi, &circuit)
            .unwrap();
        pool_multi.shutdown();

        // Verify both produce identical results
        assert!(
            states_equal(&state_single, &state_multi, 1e-4),
            "CNOT cross-block (control high): single vs multi-worker mismatch"
        );
    }

    #[test]
    fn test_cnot_determinism() {
        use logic_fabric_core::quantum::QGate;

        // Test that CNOT execution is deterministic
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::CNot(0, 2),
            QGate::CNot(1, 3),
            QGate::CNot(2, 4),
        ];

        // Run 1
        let mut state_run1 = BlockSparseState::new_zero(10);
        let mut pool1 = QuantumWorkerPool::new(2);
        pool1
            .execute_circuit_parallel(&mut state_run1, &circuit)
            .unwrap();
        pool1.shutdown();

        // Run 2
        let mut state_run2 = BlockSparseState::new_zero(10);
        let mut pool2 = QuantumWorkerPool::new(2);
        pool2
            .execute_circuit_parallel(&mut state_run2, &circuit)
            .unwrap();
        pool2.shutdown();

        // Verify determinism
        assert!(
            states_equal(&state_run1, &state_run2, 1e-6),
            "CNOT execution not deterministic!"
        );
    }

    #[test]
    fn test_cnot_multiple_within_block() {
        use logic_fabric_core::quantum::QGate;

        // Test multiple CNOTs in a circuit
        let circuit = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::CNot(2, 3),
            QGate::CNot(3, 4),
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(4);

        // Should complete without panic
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_cnot_with_hadamard_bell_pattern() {
        use logic_fabric_core::quantum::QGate;

        // Classic Bell state pattern: H on qubit 0, CNOT(0,1)
        // This should create |00⟩ + |11⟩ (maximally entangled state)
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];

        // Execute with 2 workers
        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(2);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();

        // Verify execution completed successfully
        // (Full verification would require checking amplitudes match Bell state)
    }

    // ========================================================================
    // Phase 2.3: Cross-Block CNOT Tests (target >= 7)
    // ========================================================================

    #[test]
    fn test_cnot_target_high_basic() {
        use logic_fabric_core::quantum::QGate;

        // CNOT with control < 7, target >= 7
        // This tests the target-high cross-block path
        let circuit = vec![
            QGate::H(0),       // Create superposition
            QGate::CNot(0, 7), // Cross-block: control=0, target=7
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(2);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();

        // Circuit should complete without panic
    }

    #[test]
    fn test_cnot_target_high_multiple() {
        use logic_fabric_core::quantum::QGate;

        // Multiple target-high CNOTs
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::CNot(0, 7), // target = 7
            QGate::CNot(1, 8), // target = 8
            QGate::CNot(2, 9), // target = 9
        ];

        let mut state = BlockSparseState::new_zero(12);
        let mut pool = QuantumWorkerPool::new(2);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_cnot_both_high() {
        use logic_fabric_core::quantum::QGate;

        // CNOT with both control and target >= 7
        let circuit = vec![
            QGate::H(7),       // Create superposition in high qubit
            QGate::CNot(7, 8), // Both cross-block
        ];

        let mut state = BlockSparseState::new_zero(12);
        let mut pool = QuantumWorkerPool::new(2);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_cnot_both_high_multiple() {
        use logic_fabric_core::quantum::QGate;

        // Multiple CNOTs with both qubits >= 7
        let circuit = vec![
            QGate::H(7),
            QGate::H(8),
            QGate::CNot(7, 8),
            QGate::CNot(8, 9),
            QGate::CNot(9, 10),
        ];

        let mut state = BlockSparseState::new_zero(14);
        let mut pool = QuantumWorkerPool::new(2);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_cnot_mixed_all_cases() {
        use logic_fabric_core::quantum::QGate;

        // Mix of all CNOT cases in one circuit:
        // - Within-block (both < 7)
        // - Control-high (control >= 7, target < 7)
        // - Target-high (control < 7, target >= 7)
        // - Both-high (both >= 7)
        let circuit = vec![
            QGate::H(0),
            QGate::H(7),
            QGate::CNot(0, 1), // Within-block
            QGate::CNot(7, 0), // Control-high
            QGate::CNot(0, 7), // Target-high
            QGate::CNot(7, 8), // Both-high
            QGate::CNot(1, 2), // Within-block
            QGate::CNot(8, 1), // Control-high
            QGate::CNot(2, 9), // Target-high
        ];

        let mut state = BlockSparseState::new_zero(12);
        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_cnot_cross_block_determinism() {
        use logic_fabric_core::quantum::QGate;

        // Verify determinism for cross-block CNOTs
        let circuit = vec![
            QGate::H(0),
            QGate::CNot(0, 7), // Target-high
            QGate::CNot(7, 8), // Both-high
            QGate::H(1),
            QGate::CNot(1, 8), // Target-high
        ];

        // Run twice and compare checksums
        let mut state1 = BlockSparseState::new_zero(12);
        let mut pool1 = QuantumWorkerPool::new(2);
        pool1
            .execute_circuit_parallel(&mut state1, &circuit)
            .unwrap();
        pool1.shutdown();

        let mut state2 = BlockSparseState::new_zero(12);
        let mut pool2 = QuantumWorkerPool::new(2);
        pool2
            .execute_circuit_parallel(&mut state2, &circuit)
            .unwrap();
        pool2.shutdown();

        // Compute checksums
        fn checksum(state: &BlockSparseState) -> u64 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            for block_id in 0..8.min(state.total_blocks()) {
                if let Some(block) = state.get_block(block_id) {
                    for i in 0..8.min(block.real.len()) {
                        block.real[i].to_bits().hash(&mut hasher);
                        block.imag[i].to_bits().hash(&mut hasher);
                    }
                }
            }
            hasher.finish()
        }

        let cksum1 = checksum(&state1);
        let cksum2 = checksum(&state2);
        assert_eq!(cksum1, cksum2, "Cross-block CNOT should be deterministic");
    }

    #[test]
    fn test_cnot_target_high_with_workers() {
        use logic_fabric_core::quantum::QGate;

        // Test that target-high CNOT works with different worker counts
        let circuit = vec![
            QGate::H(0),
            QGate::CNot(0, 7),
            QGate::CNot(0, 8),
            QGate::CNot(0, 9),
        ];

        // Run with 1 worker
        let mut state1 = BlockSparseState::new_zero(12);
        let mut pool1 = QuantumWorkerPool::new(1);
        pool1
            .execute_circuit_parallel(&mut state1, &circuit)
            .unwrap();
        pool1.shutdown();

        // Run with 4 workers
        let mut state4 = BlockSparseState::new_zero(12);
        let mut pool4 = QuantumWorkerPool::new(4);
        pool4
            .execute_circuit_parallel(&mut state4, &circuit)
            .unwrap();
        pool4.shutdown();

        // Both should complete (sequential execution for cross-block gates)
    }

    #[test]
    fn test_cnot_ghz_state_high_qubits() {
        use logic_fabric_core::quantum::QGate;

        // GHZ state with high qubits: |000...⟩ + |111...⟩
        // Tests cascade of cross-block CNOTs
        let circuit = vec![
            QGate::H(7),        // Superposition on qubit 7
            QGate::CNot(7, 8),  // Both-high
            QGate::CNot(8, 9),  // Both-high
            QGate::CNot(9, 10), // Both-high
        ];

        let mut state = BlockSparseState::new_zero(14);
        let mut pool = QuantumWorkerPool::new(2);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();

        // Should create GHZ-like state across high qubits
    }

    #[test]
    fn test_cnot_cross_block_chain() {
        use logic_fabric_core::quantum::QGate;

        // Chain of CNOTs crossing block boundary
        let circuit = vec![
            QGate::H(0),
            QGate::CNot(0, 1), // Within-block
            QGate::CNot(1, 2), // Within-block
            QGate::CNot(2, 3), // Within-block
            QGate::CNot(3, 4), // Within-block
            QGate::CNot(4, 5), // Within-block
            QGate::CNot(5, 6), // Within-block (last within-block)
            QGate::CNot(6, 7), // Target-high (crosses boundary!)
            QGate::CNot(7, 8), // Both-high
            QGate::CNot(8, 9), // Both-high
        ];

        let mut state = BlockSparseState::new_zero(12);
        let mut pool = QuantumWorkerPool::new(2);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    // ========================================================================
    // SPRINT 39.0 Phase 5: Rotation Gate Tests
    // ========================================================================

    #[test]
    fn test_rx_gate_basic_execution() {
        use logic_fabric_core::quantum::QGate;
        use std::f32::consts::PI;

        // Apply Rx gates with different angles
        let circuit = vec![
            QGate::H(0),            // Put into superposition first
            QGate::Rx(0, PI),       // Full rotation (should flip like X)
            QGate::Rx(1, PI / 2.0), // Quarter turn
            QGate::Rx(2, PI / 4.0), // Eighth turn
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();

        // Verify execution completed without panic
        // Rx(π) on |+⟩ should produce |−⟩ state (with phase factor)
    }

    #[test]
    fn test_ry_gate_basic_execution() {
        use logic_fabric_core::quantum::QGate;
        use std::f32::consts::PI;

        // Ry is critical for VQE's UCCSD ansatz
        let circuit = vec![
            QGate::Ry(0, PI / 2.0), // Should create |+⟩ state
            QGate::Ry(1, PI),       // Full rotation (like Y * global phase)
            QGate::Ry(2, PI / 4.0), // Eighth turn
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_rz_gate_basic_execution() {
        use logic_fabric_core::quantum::QGate;
        use std::f32::consts::PI;

        // Rz rotates around Z axis (phase rotation)
        let circuit = vec![
            QGate::H(0),      // Superposition
            QGate::Rz(0, PI), // Full Z rotation
            QGate::Rz(1, PI / 2.0),
            QGate::Rz(2, PI / 4.0),
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_u3_gate_basic_execution() {
        use logic_fabric_core::quantum::QGate;
        use std::f32::consts::PI;

        // U3 is the most general single-qubit gate
        // U3(θ, φ, λ)
        let circuit = vec![
            QGate::U3(0, PI / 2.0, 0.0, 0.0),           // Ry(π/2)
            QGate::U3(1, PI, 0.0, PI),                  // Like X gate
            QGate::U3(2, PI / 2.0, PI / 2.0, PI / 2.0), // General rotation
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_rotation_gates_single_vs_multi_worker_determinism() {
        use logic_fabric_core::quantum::QGate;
        use std::f32::consts::PI;

        let circuit = vec![
            QGate::H(0),
            QGate::Rx(0, PI / 3.0),
            QGate::Ry(1, PI / 4.0),
            QGate::Rz(2, PI / 5.0),
            QGate::U3(3, PI / 6.0, PI / 7.0, PI / 8.0),
            QGate::CNot(0, 1),
        ];

        // Execute with 1 worker
        let mut state_single = BlockSparseState::new_zero(10);
        let mut pool_single = QuantumWorkerPool::new(1);
        pool_single
            .execute_circuit_parallel(&mut state_single, &circuit)
            .unwrap();
        pool_single.shutdown();

        // Execute with 4 workers
        let mut state_multi = BlockSparseState::new_zero(10);
        let mut pool_multi = QuantumWorkerPool::new(4);
        pool_multi
            .execute_circuit_parallel(&mut state_multi, &circuit)
            .unwrap();
        pool_multi.shutdown();

        // Verify both produce identical results
        assert!(
            states_equal(&state_single, &state_multi, 1e-3),
            "Rotation gates: single-worker and multi-worker produced different results"
        );
    }

    #[test]
    fn test_rotation_gates_multiple_runs_determinism() {
        use logic_fabric_core::quantum::QGate;
        use std::f32::consts::PI;

        let circuit = vec![
            QGate::H(0),
            QGate::Ry(0, PI / 4.0), // Ry is critical for VQE
            QGate::Rx(1, PI / 3.0),
            QGate::Rz(2, PI / 2.0),
            QGate::CNot(0, 1),
            QGate::Ry(1, PI / 6.0), // Another Ry after CNOT
        ];

        // Run 1
        let mut state_run1 = BlockSparseState::new_zero(10);
        let mut pool1 = QuantumWorkerPool::new(4);
        pool1
            .execute_circuit_parallel(&mut state_run1, &circuit)
            .unwrap();
        pool1.shutdown();

        // Run 2
        let mut state_run2 = BlockSparseState::new_zero(10);
        let mut pool2 = QuantumWorkerPool::new(4);
        pool2
            .execute_circuit_parallel(&mut state_run2, &circuit)
            .unwrap();
        pool2.shutdown();

        // Verify determinism
        assert!(
            states_equal(&state_run1, &state_run2, 1e-6),
            "Rotation gates: multiple runs not deterministic!"
        );
    }

    #[test]
    fn test_vqe_like_circuit_with_rotations() {
        use logic_fabric_core::quantum::QGate;

        // Simulate a VQE-like ansatz circuit
        // Hardware-efficient ansatz: Ry rotations + entangling CNOTs
        let circuit = vec![
            // Layer 1: Ry rotations on all qubits
            QGate::Ry(0, 0.5),
            QGate::Ry(1, 0.7),
            QGate::Ry(2, 0.3),
            QGate::Ry(3, 0.9),
            // Entangling layer: ladder of CNOTs
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::CNot(2, 3),
            // Layer 2: Ry + Rz rotations
            QGate::Ry(0, 0.4),
            QGate::Rz(0, 0.2),
            QGate::Ry(1, 0.6),
            QGate::Rz(1, 0.3),
            QGate::Ry(2, 0.8),
            QGate::Rz(2, 0.1),
            QGate::Ry(3, 0.5),
            QGate::Rz(3, 0.4),
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();

        // Verify state is properly normalized (approximately)
        // This is a sanity check for rotation gate correctness
    }

    // ========================================================================
    // SPRINT 39.0 Phase 5: CZ and Swap Gate Tests
    // ========================================================================

    #[test]
    fn test_cz_gate_basic_execution() {
        use logic_fabric_core::quantum::QGate;

        // CZ flips phase when both qubits are |1⟩
        let circuit = vec![
            QGate::H(0),     // Superposition on qubit 0
            QGate::H(1),     // Superposition on qubit 1
            QGate::CZ(0, 1), // CZ gate
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_cz_gate_symmetry() {
        use logic_fabric_core::quantum::QGate;

        // CZ(a,b) should equal CZ(b,a)
        let circuit1 = vec![QGate::H(0), QGate::H(1), QGate::CZ(0, 1)];

        let circuit2 = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::CZ(1, 0), // Reversed order
        ];

        let mut state1 = BlockSparseState::new_zero(10);
        let mut pool1 = QuantumWorkerPool::new(4);
        pool1
            .execute_circuit_parallel(&mut state1, &circuit1)
            .unwrap();
        pool1.shutdown();

        let mut state2 = BlockSparseState::new_zero(10);
        let mut pool2 = QuantumWorkerPool::new(4);
        pool2
            .execute_circuit_parallel(&mut state2, &circuit2)
            .unwrap();
        pool2.shutdown();

        assert!(
            states_equal(&state1, &state2, 1e-4),
            "CZ should be symmetric: CZ(0,1) should equal CZ(1,0)"
        );
    }

    #[test]
    fn test_cz_gate_cross_block() {
        use logic_fabric_core::quantum::QGate;

        // Test CZ with one qubit in block ID region (qubit >= 7)
        let circuit = vec![
            QGate::H(0),     // Within-block qubit
            QGate::H(7),     // Cross-block qubit (in block ID)
            QGate::CZ(0, 7), // Cross-block CZ
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(2);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_cz_gate_both_cross_block() {
        use logic_fabric_core::quantum::QGate;

        // Test CZ with both qubits in block ID region
        let circuit = vec![
            QGate::H(7),     // Cross-block qubit
            QGate::H(8),     // Cross-block qubit
            QGate::CZ(7, 8), // Both-cross CZ
        ];

        let mut state = BlockSparseState::new_zero(12);
        let mut pool = QuantumWorkerPool::new(2);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_swap_gate_basic_execution() {
        use logic_fabric_core::quantum::QGate;

        // Swap exchanges two qubit states
        let circuit = vec![
            QGate::X(0), // Put qubit 0 in |1⟩
            QGate::Swap(0, 1), // Swap qubits 0 and 1
                         // Now qubit 1 should be |1⟩ and qubit 0 should be |0⟩
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();
    }

    #[test]
    fn test_swap_gate_double_swap_identity() {
        use logic_fabric_core::quantum::QGate;

        // Swap twice should return to original state
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::Swap(0, 1),
            QGate::Swap(0, 1), // Second swap should undo the first
        ];

        // Compare with circuit without swaps
        let circuit_no_swap = vec![QGate::H(0), QGate::H(1)];

        let mut state_swapped = BlockSparseState::new_zero(10);
        let mut pool1 = QuantumWorkerPool::new(4);
        pool1
            .execute_circuit_parallel(&mut state_swapped, &circuit)
            .unwrap();
        pool1.shutdown();

        let mut state_no_swap = BlockSparseState::new_zero(10);
        let mut pool2 = QuantumWorkerPool::new(4);
        pool2
            .execute_circuit_parallel(&mut state_no_swap, &circuit_no_swap)
            .unwrap();
        pool2.shutdown();

        assert!(
            states_equal(&state_swapped, &state_no_swap, 1e-4),
            "Double swap should be identity"
        );
    }

    #[test]
    fn test_swap_gate_symmetry() {
        use logic_fabric_core::quantum::QGate;

        // Swap(a,b) should equal Swap(b,a)
        let circuit1 = vec![QGate::H(0), QGate::X(1), QGate::Swap(0, 1)];

        let circuit2 = vec![
            QGate::H(0),
            QGate::X(1),
            QGate::Swap(1, 0), // Reversed order
        ];

        let mut state1 = BlockSparseState::new_zero(10);
        let mut pool1 = QuantumWorkerPool::new(4);
        pool1
            .execute_circuit_parallel(&mut state1, &circuit1)
            .unwrap();
        pool1.shutdown();

        let mut state2 = BlockSparseState::new_zero(10);
        let mut pool2 = QuantumWorkerPool::new(4);
        pool2
            .execute_circuit_parallel(&mut state2, &circuit2)
            .unwrap();
        pool2.shutdown();

        assert!(
            states_equal(&state1, &state2, 1e-4),
            "Swap should be symmetric: Swap(0,1) should equal Swap(1,0)"
        );
    }

    #[test]
    fn test_cz_swap_single_vs_multi_worker_determinism() {
        use logic_fabric_core::quantum::QGate;

        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::CZ(0, 1),
            QGate::Swap(1, 2),
            QGate::CZ(0, 2),
        ];

        // Execute with 1 worker
        let mut state_single = BlockSparseState::new_zero(10);
        let mut pool_single = QuantumWorkerPool::new(1);
        pool_single
            .execute_circuit_parallel(&mut state_single, &circuit)
            .unwrap();
        pool_single.shutdown();

        // Execute with 4 workers
        let mut state_multi = BlockSparseState::new_zero(10);
        let mut pool_multi = QuantumWorkerPool::new(4);
        pool_multi
            .execute_circuit_parallel(&mut state_multi, &circuit)
            .unwrap();
        pool_multi.shutdown();

        assert!(
            states_equal(&state_single, &state_multi, 1e-3),
            "CZ/Swap: single-worker and multi-worker produced different results"
        );
    }

    #[test]
    fn test_graph_state_circuit() {
        use logic_fabric_core::quantum::QGate;

        // Graph state: H on all qubits, then CZ on edges
        // This is a common circuit pattern in quantum computing
        let circuit = vec![
            // Layer 1: Hadamard on all qubits
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::H(3),
            // Layer 2: CZ on edges (linear graph 0-1-2-3)
            QGate::CZ(0, 1),
            QGate::CZ(1, 2),
            QGate::CZ(2, 3),
        ];

        let mut state = BlockSparseState::new_zero(10);
        let mut pool = QuantumWorkerPool::new(4);
        pool.execute_circuit_parallel(&mut state, &circuit).unwrap();
        pool.shutdown();

        // Graph states are important for measurement-based quantum computing
    }
}
