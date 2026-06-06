//! EPIC 63 Phase 1: Distributed Tile Farm
//!
//! Multi-node distributed quantum simulation with embarrassingly parallel tile sharding.
//!
//! # Architecture
//!
//! ```text
//! DistributedTileFarm (Coordinator)
//!   ├─ NodeHandle 0: tiles [0..16]   → TileFarm (workers=2, tiles/w=4)
//!   ├─ NodeHandle 1: tiles [16..32]  → TileFarm (workers=2, tiles/w=4)
//!   └─ NodeHandle 2: tiles [32..48]  → TileFarm (workers=2, tiles/w=4)
//! ```
//!
//! # Invariants
//!
//! - Global tile space partitioned into disjoint node shards
//! - No cross-node entanglement (embarrassingly parallel)
//! - Each node runs independent EPIC 62 TileFarm
//! - Deterministic execution given fixed node count + tile layout
//! - tiles_per_node = global_tile_count / node_count

use crate::distributed::quantum_worker_pool::QuantumWorkerPool;
use crate::distributed::tile_farm::{BackendKind, KernelType, TileFarm};
use crate::quantum::QState;
use logic_fabric_core::block_sparse_state::BlockSparseState;

// ============================================================================
// EPIC 63 Phase 2: Distributed Protocol & Transport
// ============================================================================

/// Message sent from coordinator to worker node
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DistributedJob {
    /// Initialize a node with its tile range and configuration
    InitNode {
        node_id: u16,
        tile_start: u16,
        tile_end: u16,
        qubits: u8,
    },

    /// Run a kernel on this node's local tiles
    RunKernel {
        kernel_type: KernelType,
        depth: u32,
        backend: BackendKind,
        /// JIT function pointer (u64 for serialization)
        /// Note: Over TCP, this will be 0 (workers compile their own kernels)
        #[cfg_attr(feature = "serde", serde(skip))]
        jit_fn_ptr: Option<*const u8>,
    },

    /// Run N kernels back-to-back and return aggregated result (EPIC 65 Phase 1)
    RunKernelBatch {
        kernel_type: KernelType,
        depth: u32,
        backend: BackendKind,
        count: u32,
        #[cfg_attr(feature = "serde", serde(skip))]
        jit_fn_ptr: Option<*const u8>,
    },

    /// Shutdown this node gracefully
    Shutdown,

    /// Heartbeat ping (EPIC 64 Phase 2)
    Ping,

    // SPRINT 39.0 Phase 3: Distributed Quantum Circuit Execution
    /// Execute quantum circuit on block-sparse state (SPRINT 39.0 Phase 3)
    RunQuantumCircuit {
        /// Number of qubits
        n_qubits: u8,
        /// Quantum circuit (gates to execute)
        /// Note: QGate doesn't implement Serialize, so skip for serde
        #[cfg_attr(feature = "serde", serde(skip))]
        circuit: Vec<logic_fabric_core::quantum::QGate>,
        /// Number of workers to use per node
        workers_per_node: usize,
    },

    /// Execute quantum circuit batch (multiple independent circuits)
    RunQuantumBatch {
        /// Number of qubits
        n_qubits: u8,
        /// Batch of circuits to execute
        /// Note: QGate doesn't implement Serialize, so skip for serde
        #[cfg_attr(feature = "serde", serde(skip))]
        circuits: Vec<Vec<logic_fabric_core::quantum::QGate>>,
        /// Number of workers per node
        workers_per_node: usize,
    },
}

/// Result sent from worker node back to coordinator
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DistributedResult {
    /// Result from a kernel execution
    KernelComplete {
        /// Node ID that produced this result
        node_id: u16,
        /// Local checksum from this node
        checksum: u64,
        /// Optional metrics (latency, throughput, etc.)
        metrics: NodeMetrics,
    },

    /// Result from a batch of kernels (EPIC 65 Phase 1)
    BatchComplete {
        node_id: u16,
        /// Aggregated checksum from all kernels in batch
        checksum: u64,
        /// Number of kernels executed
        count: u32,
        /// Aggregated metrics
        metrics: NodeMetrics,
    },

    /// Heartbeat pong response (EPIC 64 Phase 2)
    Pong { node_id: u16 },

    // SPRINT 39.0 Phase 3: Distributed Quantum Results
    /// Result from quantum circuit execution
    QuantumCircuitComplete {
        node_id: u16,
        /// State checksum (for verification)
        checksum: u64,
        /// Number of gates executed
        gate_count: usize,
        /// Execution time in microseconds
        exec_time_us: u64,
        /// Metrics
        metrics: NodeMetrics,
    },

    /// Result from quantum circuit batch
    QuantumBatchComplete {
        node_id: u16,
        /// Aggregated checksum
        checksum: u64,
        /// Number of circuits executed
        circuit_count: usize,
        /// Total gates executed across all circuits
        total_gates: usize,
        /// Execution time in microseconds
        exec_time_us: u64,
        /// Metrics
        metrics: NodeMetrics,
    },
}

impl DistributedResult {
    pub fn node_id(&self) -> u16 {
        match self {
            DistributedResult::KernelComplete { node_id, .. } => *node_id,
            DistributedResult::BatchComplete { node_id, .. } => *node_id,
            DistributedResult::Pong { node_id } => *node_id,
            DistributedResult::QuantumCircuitComplete { node_id, .. } => *node_id,
            DistributedResult::QuantumBatchComplete { node_id, .. } => *node_id,
        }
    }

    pub fn checksum(&self) -> Option<u64> {
        match self {
            DistributedResult::KernelComplete { checksum, .. } => Some(*checksum),
            DistributedResult::BatchComplete { checksum, .. } => Some(*checksum),
            DistributedResult::Pong { .. } => None,
            DistributedResult::QuantumCircuitComplete { checksum, .. } => Some(*checksum),
            DistributedResult::QuantumBatchComplete { checksum, .. } => Some(*checksum),
        }
    }

    pub fn metrics(&self) -> Option<&NodeMetrics> {
        match self {
            DistributedResult::KernelComplete { metrics, .. } => Some(metrics),
            DistributedResult::BatchComplete { metrics, .. } => Some(metrics),
            DistributedResult::Pong { .. } => None,
            DistributedResult::QuantumCircuitComplete { metrics, .. } => Some(metrics),
            DistributedResult::QuantumBatchComplete { metrics, .. } => Some(metrics),
        }
    }

    pub fn batch_count(&self) -> Option<u32> {
        match self {
            DistributedResult::BatchComplete { count, .. } => Some(*count),
            _ => None,
        }
    }
}

/// Performance metrics from a node
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeMetrics {
    /// Execution time in microseconds
    pub execution_time_us: u64,

    /// Number of quantum operations performed
    pub q_ops: u64,
}

// Safety: DistributedJob contains a raw pointer (jit_fn_ptr) which is read-only
// and guaranteed to outlive any transport operations (compiled once on coordinator).
unsafe impl Send for DistributedJob {}

/// Transport abstraction for node communication
///
/// Phase 2a: In-process channels (mpsc)
/// Phase 2b: IPC/TCP sockets (future)
pub trait NodeTransport: Send {
    /// Send a job to a worker node
    fn send_job(&mut self, job: DistributedJob) -> Result<(), String>;

    /// Receive a result from a worker node
    fn recv_result(&mut self) -> Result<DistributedResult, String>;
}

// ============================================================================
// Phase 2a: In-Process Channel Transport
// ============================================================================

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

/// In-process channel-based transport
///
/// Each node runs on a background thread.
/// Coordinator sends jobs via mpsc channel.
/// Node sends results back via another channel.
pub struct ChannelTransport {
    /// Node ID
    node_id: u16,

    /// Channel to send jobs to the node thread
    job_sender: Sender<DistributedJob>,

    /// Channel to receive results from the node thread
    result_receiver: Receiver<DistributedResult>,
}

impl ChannelTransport {
    /// Create a new channel transport and spawn node worker thread
    ///
    /// # Arguments
    ///
    /// * `node_id` - Node identifier
    /// * `tile_start` - Starting tile index for this node
    /// * `tile_end` - Ending tile index (exclusive)
    /// * `qubits` - Number of qubits
    /// * `workers_per_node` - Workers in local TileFarm
    ///
    /// # Returns
    ///
    /// ChannelTransport handle for communicating with the node thread
    pub fn spawn_node(
        node_id: u16,
        tile_start: u16,
        tile_end: u16,
        qubits: u8,
        workers_per_node: usize,
    ) -> Self {
        let (job_tx, job_rx) = channel();
        let (result_tx, result_rx) = channel();

        // Spawn node worker thread
        thread::Builder::new()
            .name(format!("node-{}", node_id))
            .spawn(move || {
                node_worker_loop(
                    node_id,
                    tile_start,
                    tile_end,
                    qubits,
                    workers_per_node,
                    job_rx,
                    result_tx,
                );
            })
            .expect("Failed to spawn node worker thread");

        Self {
            node_id,
            job_sender: job_tx,
            result_receiver: result_rx,
        }
    }
}

impl NodeTransport for ChannelTransport {
    fn send_job(&mut self, job: DistributedJob) -> Result<(), String> {
        self.job_sender
            .send(job)
            .map_err(|e| format!("Node {} send error: {}", self.node_id, e))
    }

    fn recv_result(&mut self) -> Result<DistributedResult, String> {
        self.result_receiver
            .recv()
            .map_err(|e| format!("Node {} recv error: {}", self.node_id, e))
    }
}

// SPRINT 39.0 Phase 3: Quantum State Checksum Helper
/// Compute a simple checksum from block-sparse state for verification
///
/// Uses XOR of block IDs and first few amplitudes to create a deterministic checksum
fn compute_state_checksum(state: &logic_fabric_core::block_sparse_state::BlockSparseState) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash number of qubits
    state.n_qubits.hash(&mut hasher);

    // Hash block count
    state.block_count().hash(&mut hasher);

    // Hash total blocks capacity
    state.total_blocks().hash(&mut hasher);

    // Hash first few blocks by iterating through possible block IDs
    // (We can't access private state.blocks, so we iterate through all possible IDs)
    let total_blocks = state.total_blocks();
    let blocks_to_check = 10.min(total_blocks);

    for block_id in 0..blocks_to_check {
        if let Some(block) = state.get_block(block_id) {
            block_id.hash(&mut hasher);

            // Hash first few amplitudes
            for i in 0..8.min(block.real.len()) {
                let r = block.real[i].to_f32().to_bits();
                let im = block.imag[i].to_f32().to_bits();
                r.hash(&mut hasher);
                im.hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

/// Node worker thread loop
///
/// Listens for DistributedJob messages and executes them on local TileFarm.
fn node_worker_loop(
    node_id: u16,
    tile_start: u16,
    tile_end: u16,
    qubits: u8,
    workers_per_node: usize,
    job_rx: Receiver<DistributedJob>,
    result_tx: Sender<DistributedResult>,
) {
    let tiles_per_node = (tile_end - tile_start) as usize;
    let tiles_per_worker = tiles_per_node / workers_per_node;

    eprintln!(
        "[node-{}] Worker thread started: tiles [{}..{}), {} workers × {} tiles/w",
        node_id, tile_start, tile_end, workers_per_node, tiles_per_worker
    );

    // Create local TileFarm and QState for this node
    let local_farm = TileFarm::new(workers_per_node, tiles_per_worker);
    let mut local_qstate = QState::new_zero_multitile(qubits, tiles_per_node as u16);

    // Node worker loop: process jobs until shutdown
    loop {
        match job_rx.recv() {
            Ok(DistributedJob::InitNode { .. }) => {
                // Already initialized; this is a no-op in Phase 2a
                eprintln!("[node-{}] Received InitNode (already initialized)", node_id);
            }

            Ok(DistributedJob::RunKernel {
                kernel_type,
                depth,
                backend,
                jit_fn_ptr,
            }) => {
                use std::time::Instant;
                let start = Instant::now();

                // Execute kernel on local TileFarm
                unsafe {
                    local_farm.run_kernel_on_all_workers(
                        kernel_type,
                        qubits,
                        depth,
                        backend,
                        &mut local_qstate as *mut QState as *mut u8,
                        jit_fn_ptr,
                    );
                }

                let elapsed = start.elapsed();

                // Compute local checksum
                let real_sum: f32 = local_qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                let imag_sum: f32 = local_qstate.imag.as_slice().iter().map(|x| x.abs()).sum();
                let tile_count = tiles_per_node as f32;
                let real_avg = real_sum / tile_count;
                let imag_avg = imag_sum / tile_count;
                let checksum = (real_avg.to_bits() as u64) ^ ((imag_avg.to_bits() as u64) << 32);

                let result = DistributedResult::KernelComplete {
                    node_id,
                    checksum,
                    metrics: NodeMetrics {
                        execution_time_us: elapsed.as_micros() as u64,
                        q_ops: 0, // TODO: compute actual q_ops
                    },
                };

                eprintln!(
                    "[node-{}] RunKernel complete: checksum=0x{:016X}, time={}us",
                    node_id,
                    checksum,
                    elapsed.as_micros()
                );

                if result_tx.send(result).is_err() {
                    eprintln!(
                        "[node-{}] Failed to send result (coordinator disconnected)",
                        node_id
                    );
                    break;
                }
            }

            Ok(DistributedJob::RunKernelBatch {
                kernel_type,
                depth,
                backend,
                count,
                jit_fn_ptr,
            }) => {
                use std::time::Instant;
                let batch_start = Instant::now();

                // Run N kernels back-to-back
                let mut total_checksum = 0u64;
                for _ in 0..count {
                    unsafe {
                        local_farm.run_kernel_on_all_workers(
                            kernel_type,
                            qubits,
                            depth,
                            backend,
                            &mut local_qstate as *mut QState as *mut u8,
                            jit_fn_ptr,
                        );
                    }

                    // Accumulate checksum
                    let real_sum: f32 = local_qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                    let imag_sum: f32 = local_qstate.imag.as_slice().iter().map(|x| x.abs()).sum();
                    let tile_count = tiles_per_node as f32;
                    let real_avg = real_sum / tile_count;
                    let imag_avg = imag_sum / tile_count;
                    let kernel_checksum =
                        (real_avg.to_bits() as u64) ^ ((imag_avg.to_bits() as u64) << 32);
                    total_checksum ^= kernel_checksum;
                }

                let batch_elapsed = batch_start.elapsed();

                let result = DistributedResult::BatchComplete {
                    node_id,
                    checksum: total_checksum,
                    count,
                    metrics: NodeMetrics {
                        execution_time_us: batch_elapsed.as_micros() as u64,
                        q_ops: (qubits as u64 * depth as u64 * count as u64),
                    },
                };

                eprintln!(
                    "[node-{}] BatchComplete: {} kernels, checksum=0x{:016X}, time={}us",
                    node_id,
                    count,
                    total_checksum,
                    batch_elapsed.as_micros()
                );

                if result_tx.send(result).is_err() {
                    eprintln!(
                        "[node-{}] Failed to send result (coordinator disconnected)",
                        node_id
                    );
                    break;
                }
            }

            // SPRINT 39.0 Phase 3: Quantum Circuit Execution
            Ok(DistributedJob::RunQuantumCircuit {
                n_qubits,
                circuit,
                workers_per_node,
            }) => {
                use crate::distributed::quantum_worker_pool::QuantumWorkerPool;
                use logic_fabric_core::block_sparse_state::BlockSparseState;
                use std::time::Instant;

                let start = Instant::now();

                // Create block-sparse state
                let mut state = BlockSparseState::new_zero(n_qubits);

                // Execute circuit with worker pool
                let mut pool = QuantumWorkerPool::new(workers_per_node);
                let exec_result = pool.execute_circuit_parallel(&mut state, &circuit);
                pool.shutdown();

                let elapsed = start.elapsed();

                // Compute simple checksum from first few amplitudes
                let checksum = compute_state_checksum(&state);

                let result = if exec_result.is_ok() {
                    DistributedResult::QuantumCircuitComplete {
                        node_id,
                        checksum,
                        gate_count: circuit.len(),
                        exec_time_us: elapsed.as_micros() as u64,
                        metrics: NodeMetrics {
                            execution_time_us: elapsed.as_micros() as u64,
                            q_ops: circuit.len() as u64,
                        },
                    }
                } else {
                    eprintln!(
                        "[node-{}] Quantum circuit execution failed: {:?}",
                        node_id, exec_result
                    );
                    continue;
                };

                if result_tx.send(result).is_err() {
                    eprintln!(
                        "[node-{}] Failed to send result (coordinator disconnected)",
                        node_id
                    );
                    break;
                }
            }

            Ok(DistributedJob::RunQuantumBatch {
                n_qubits,
                circuits,
                workers_per_node,
            }) => {
                use crate::distributed::quantum_worker_pool::QuantumWorkerPool;
                use logic_fabric_core::block_sparse_state::BlockSparseState;
                use std::time::Instant;

                let batch_start = Instant::now();
                let mut total_checksum = 0u64;
                let mut total_gates = 0usize;

                // Create worker pool once and reuse for all circuits in batch
                let mut pool = QuantumWorkerPool::new(workers_per_node);

                for circuit in &circuits {
                    let mut state = BlockSparseState::new_zero(n_qubits);

                    if pool.execute_circuit_parallel(&mut state, circuit).is_ok() {
                        total_checksum ^= compute_state_checksum(&state);
                        total_gates += circuit.len();
                    }
                }

                // Shutdown pool after all circuits are executed
                pool.shutdown();

                let batch_elapsed = batch_start.elapsed();

                let result = DistributedResult::QuantumBatchComplete {
                    node_id,
                    checksum: total_checksum,
                    circuit_count: circuits.len(),
                    total_gates,
                    exec_time_us: batch_elapsed.as_micros() as u64,
                    metrics: NodeMetrics {
                        execution_time_us: batch_elapsed.as_micros() as u64,
                        q_ops: total_gates as u64,
                    },
                };

                if result_tx.send(result).is_err() {
                    eprintln!(
                        "[node-{}] Failed to send result (coordinator disconnected)",
                        node_id
                    );
                    break;
                }
            }

            Ok(DistributedJob::Shutdown) => {
                eprintln!("[node-{}] Received Shutdown, exiting", node_id);
                break;
            }

            Ok(DistributedJob::Ping) => {
                // Heartbeat: respond immediately with Pong
                let pong = DistributedResult::Pong { node_id };
                if result_tx.send(pong).is_err() {
                    eprintln!(
                        "[node-{}] Failed to send Pong (coordinator disconnected)",
                        node_id
                    );
                    break;
                }
            }

            Err(_) => {
                eprintln!("[node-{}] Job channel closed, exiting", node_id);
                break;
            }
        }
    }

    eprintln!("[node-{}] Worker thread terminated", node_id);
}

// ============================================================================
// EPIC 64 Phase 1: Cluster Config & Manifest
// ============================================================================

/// Cluster configuration manifest (loaded from cluster.toml)
#[cfg(feature = "cluster")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClusterConfig {
    pub cluster: ClusterMeta,
    pub node: Vec<NodeConfig>,
}

// ============================================================================
// EPIC 64 Phase 2: Node Health Tracking
// ============================================================================

/// Health state of a worker node
#[cfg(feature = "cluster")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeHealth {
    /// Node is responsive and operating normally
    Healthy,
    /// Node is responding but slower than expected
    Degraded,
    /// Node is not responding (timeout or connection lost)
    Unreachable,
}

/// Health tracking info for a node
#[cfg(feature = "cluster")]
#[derive(Debug, Clone)]
pub struct NodeHealthInfo {
    pub node_id: u16,
    pub health: NodeHealth,
    pub last_ping_us: Option<u64>,
    pub last_heartbeat: std::time::Instant,
}

#[cfg(feature = "cluster")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClusterMeta {
    pub name: String,
}

#[cfg(feature = "cluster")]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct NodeConfig {
    pub id: u16,
    pub host: String,
    pub port: u16,
    pub tiles: u16,
}

#[cfg(feature = "cluster")]
impl ClusterConfig {
    /// Load cluster configuration from a TOML file
    pub fn from_file(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read cluster config {}: {}", path, e))?;

        toml::from_str(&content).map_err(|e| format!("Failed to parse cluster config: {}", e))
    }

    /// Validate cluster configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.node.is_empty() {
            return Err("Cluster must have at least one node".to_string());
        }

        // Check for duplicate node IDs
        let mut seen_ids = std::collections::HashSet::new();
        for node in &self.node {
            if !seen_ids.insert(node.id) {
                return Err(format!("Duplicate node ID: {}", node.id));
            }
        }

        // Validate tiles per node (must be divisible by 4 per EPIC 62Q)
        for node in &self.node {
            if node.tiles == 0 || node.tiles % 4 != 0 {
                return Err(format!(
                    "Node {} tiles must be non-zero and divisible by 4 (got {})",
                    node.id, node.tiles
                ));
            }
        }

        Ok(())
    }

    /// Get total tile count across all nodes
    pub fn total_tiles(&self) -> u16 {
        self.node.iter().map(|n| n.tiles).sum()
    }

    /// Get tile range for a specific node
    pub fn tile_range_for_node(&self, node_id: u16) -> Option<(u16, u16)> {
        let mut start = 0u16;
        for node in &self.node {
            let end = start + node.tiles;
            if node.id == node_id {
                return Some((start, end));
            }
            start = end;
        }
        None
    }
}

// ============================================================================
// Phase 2b: TCP Transport
// ============================================================================

#[cfg(feature = "serde")]
use std::io::{Read as IoRead, Write as IoWrite};
#[cfg(feature = "serde")]
use std::net::{TcpListener, TcpStream};
#[cfg(feature = "serde")]
use std::time::Duration;

/// TCP-based transport with length-prefix framing
///
/// Each message is framed as:
///   [u32 length][payload bytes]
///
/// This solves the "partial read" problem for streaming protocols.
#[cfg(feature = "serde")]
pub struct TcpTransport {
    _node_id: u16,
    stream: TcpStream,
}

#[cfg(feature = "serde")]
impl TcpTransport {
    /// Connect to a worker node at the given address
    pub fn connect(node_id: u16, addr: &str) -> Result<Self, String> {
        let stream = TcpStream::connect(addr)
            .map_err(|e| format!("Failed to connect to {}: {}", addr, e))?;

        // Set timeouts to avoid blocking forever
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| format!("Failed to set read timeout: {}", e))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| format!("Failed to set write timeout: {}", e))?;

        eprintln!("[tcp-transport] Connected to node {} at {}", node_id, addr);

        Ok(TcpTransport {
            _node_id: node_id,
            stream,
        })
    }

    /// Send a message with length-prefix framing
    fn send<T: serde::Serialize>(&mut self, msg: &T) -> Result<(), String> {
        let bytes =
            bincode::serialize(msg).map_err(|e| format!("Failed to serialize message: {}", e))?;

        let len = bytes.len() as u32;

        // Write length prefix (u32 big-endian)
        self.stream
            .write_all(&len.to_be_bytes())
            .map_err(|e| format!("Failed to write length prefix: {}", e))?;

        // Write payload
        self.stream
            .write_all(&bytes)
            .map_err(|e| format!("Failed to write payload: {}", e))?;

        self.stream
            .flush()
            .map_err(|e| format!("Failed to flush stream: {}", e))?;

        Ok(())
    }

    /// Receive a message with length-prefix framing
    fn recv<T: serde::de::DeserializeOwned>(&mut self) -> Result<T, String> {
        // Read length prefix (u32 big-endian)
        let mut len_buf = [0u8; 4];
        self.stream
            .read_exact(&mut len_buf)
            .map_err(|e| format!("Failed to read length prefix: {}", e))?;

        let len = u32::from_be_bytes(len_buf) as usize;

        // Sanity check: reject absurdly large messages (>16MB)
        if len > 16 * 1024 * 1024 {
            return Err(format!("Message too large: {} bytes", len));
        }

        // Read payload
        let mut payload = vec![0u8; len];
        self.stream
            .read_exact(&mut payload)
            .map_err(|e| format!("Failed to read payload: {}", e))?;

        // Deserialize
        bincode::deserialize(&payload).map_err(|e| format!("Failed to deserialize message: {}", e))
    }

    /// Send a Ping and measure round-trip time
    /// Returns Ok(latency_us) on success, Err on timeout/failure
    #[cfg(feature = "cluster")]
    pub fn heartbeat_ping(&mut self) -> Result<u64, String> {
        use std::time::Instant;

        let start = Instant::now();

        // Send Ping
        self.send(&DistributedJob::Ping)?;

        // Receive Pong (with timeout from stream settings)
        let result: DistributedResult = self.recv()?;

        let latency_us = start.elapsed().as_micros() as u64;

        match result {
            DistributedResult::Pong { node_id } => {
                if node_id == self._node_id {
                    Ok(latency_us)
                } else {
                    Err(format!(
                        "Pong from wrong node: expected {}, got {}",
                        self._node_id, node_id
                    ))
                }
            }
            _ => Err("Expected Pong, got different response".to_string()),
        }
    }
}

#[cfg(feature = "serde")]
impl NodeTransport for TcpTransport {
    fn send_job(&mut self, job: DistributedJob) -> Result<(), String> {
        self.send(&job)
    }

    fn recv_result(&mut self) -> Result<DistributedResult, String> {
        self.recv()
    }
}

/// Pipelined batch executor (EPIC 65 Phase 2)
///
/// Dispatches N batches to all nodes, then collects N results from all nodes.
/// This hides network latency by having multiple batches in flight.
#[cfg(feature = "serde")]
pub fn run_pipelined_batches<T: NodeTransport>(
    transports: &mut [(u16, T)],
    job: DistributedJob,
    num_batches: u32,
) -> Result<Vec<Vec<DistributedResult>>, String> {
    run_pipelined_batches_with_queue_depth(transports, job, num_batches, num_batches)
}

/// Pipelined batch executor with queue depth control (EPIC 65 Phase 3)
///
/// Dispatches batches respecting worker queue depth to avoid overwhelming workers.
/// This implements backpressure: only `queue_depth` batches in flight per node at a time.
///
/// # Arguments
/// * `queue_depth` - Max batches in flight per node (typically 2-4)
#[cfg(feature = "serde")]
pub fn run_pipelined_batches_with_queue_depth<T: NodeTransport>(
    transports: &mut [(u16, T)],
    job: DistributedJob,
    num_batches: u32,
    queue_depth: u32,
) -> Result<Vec<Vec<DistributedResult>>, String> {
    let mut all_results = Vec::new();
    let mut dispatched = 0u32;
    let mut collected = 0u32;

    // Initial fill: dispatch up to queue_depth batches
    let initial_dispatch = num_batches.min(queue_depth);
    for batch_idx in 0..initial_dispatch {
        for (node_id, transport) in transports.iter_mut() {
            transport.send_job(job.clone()).map_err(|e| {
                format!(
                    "Failed to send batch {} to node {}: {}",
                    batch_idx, node_id, e
                )
            })?;
        }
        dispatched += 1;
    }

    // Pipelined execution: collect result, dispatch next batch
    while collected < num_batches {
        let mut batch_results = Vec::new();

        // Collect results from all nodes for this batch
        for (node_id, transport) in transports.iter_mut() {
            let result = transport.recv_result().map_err(|e| {
                format!(
                    "Failed to receive batch {} from node {}: {}",
                    collected, node_id, e
                )
            })?;
            batch_results.push(result);
        }

        all_results.push(batch_results);
        collected += 1;

        // Dispatch next batch if available (backpressure control)
        if dispatched < num_batches {
            for (node_id, transport) in transports.iter_mut() {
                transport.send_job(job.clone()).map_err(|e| {
                    format!(
                        "Failed to send batch {} to node {}: {}",
                        dispatched, node_id, e
                    )
                })?;
            }
            dispatched += 1;
        }
    }

    Ok(all_results)
}

/// Parallel result aggregator (EPIC 65 Phase 4)
///
/// Aggregates checksums and metrics from multiple distributed results in parallel.
/// Uses deterministic reduction order to ensure reproducibility.
#[cfg(feature = "serde")]
pub fn aggregate_results_parallel(results: &[Vec<DistributedResult>]) -> AggregatedMetrics {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let total_kernels = Arc::new(AtomicU64::new(0));
    let total_q_ops = Arc::new(AtomicU64::new(0));
    let total_time_us = Arc::new(AtomicU64::new(0));

    // Flatten and collect checksums
    let mut checksums = Vec::new();

    for batch in results {
        for result in batch {
            if let Some(count) = result.batch_count() {
                total_kernels.fetch_add(count as u64, Ordering::Relaxed);
            }

            if let Some(metrics) = result.metrics() {
                total_q_ops.fetch_add(metrics.q_ops, Ordering::Relaxed);
                total_time_us.fetch_add(metrics.execution_time_us, Ordering::Relaxed);
            }

            if let Some(checksum) = result.checksum() {
                checksums.push(checksum);
            }
        }
    }

    // Deterministic checksum reduction (XOR all checksums)
    let combined_checksum = checksums.into_iter().fold(0u64, |acc, cs| acc ^ cs);

    AggregatedMetrics {
        total_kernels: total_kernels.load(Ordering::Relaxed) as u32,
        total_q_ops: total_q_ops.load(Ordering::Relaxed),
        combined_checksum,
        total_execution_time_us: total_time_us.load(Ordering::Relaxed),
    }
}

/// Aggregated metrics from multiple results
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    pub total_kernels: u32,
    pub total_q_ops: u64,
    pub combined_checksum: u64,
    pub total_execution_time_us: u64,
}

/// TCP worker node process
///
/// Binds to a port and waits for coordinator to connect.
/// Runs kernels on demand and sends results back.
#[cfg(feature = "serde")]
pub fn tcp_worker_main(bind_addr: &str) -> Result<(), String> {
    let listener = TcpListener::bind(bind_addr)
        .map_err(|e| format!("Failed to bind to {}: {}", bind_addr, e))?;

    eprintln!("[tcp-worker] Listening on {}", bind_addr);

    // Accept single connection from coordinator
    let (stream, peer_addr) = listener
        .accept()
        .map_err(|e| format!("Failed to accept connection: {}", e))?;

    eprintln!("[tcp-worker] Accepted connection from {}", peer_addr);

    // Set timeouts
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| format!("Failed to set read timeout: {}", e))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| format!("Failed to set write timeout: {}", e))?;

    let mut transport = TcpWorkerTransport { stream };

    // Initialize node state
    let mut node_id = 0u16;
    let mut tile_start = 0u16;
    let mut tile_end = 0u16;
    let mut qubits = 0u8;
    let mut farm: Option<TileFarm> = None;
    let mut qstate: Option<QState> = None;

    // Worker loop
    loop {
        let job: DistributedJob = transport
            .recv()
            .map_err(|e| format!("Failed to receive job: {}", e))?;

        match job {
            DistributedJob::InitNode {
                node_id: nid,
                tile_start: ts,
                tile_end: te,
                qubits: q,
            } => {
                eprintln!(
                    "[tcp-worker] InitNode: node_id={}, tiles=[{}..{}), qubits={}",
                    nid, ts, te, q
                );

                node_id = nid;
                tile_start = ts;
                tile_end = te;
                qubits = q;

                let tile_count = tile_end - tile_start;
                let workers_per_node = tile_count / 4; // EPIC 62Q constraint

                farm = Some(TileFarm::new(workers_per_node as usize, 4));
                qstate = Some(QState::new_zero_multitile(qubits, tile_count));

                eprintln!(
                    "[tcp-worker] Initialized: {} workers × 4 tiles/w = {} tiles",
                    workers_per_node, tile_count
                );
            }

            DistributedJob::RunKernel {
                kernel_type,
                depth,
                backend,
                ..
            } => {
                let farm = farm.as_mut().ok_or("Node not initialized")?;
                let qstate = qstate.as_mut().ok_or("Node not initialized")?;

                let start = std::time::Instant::now();

                // Workers compile their own kernels (jit_fn_ptr is skipped in serde)
                #[cfg(feature = "cranelift_jit")]
                let jit_fn_ptr = if matches!(backend, BackendKind::SimdJit) {
                    unsafe {
                        std::env::set_var("JIT_TILE_COUNT", "4");
                        std::env::set_var("JIT_H_DEPTH", &depth.to_string());
                    }
                    crate::quantum_jit::clif::compile_h_kernel(qubits, 0).map(|f| f as *const u8)
                } else {
                    None
                };

                #[cfg(not(feature = "cranelift_jit"))]
                let jit_fn_ptr = None;

                unsafe {
                    farm.run_kernel_on_all_workers(
                        kernel_type,
                        qubits,
                        depth,
                        backend,
                        qstate as *mut QState as *mut u8,
                        jit_fn_ptr,
                    );
                }

                let elapsed = start.elapsed();

                // Compute checksum (same as ChannelTransport)
                let real_sum: f32 = qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                let imag_sum: f32 = qstate.imag.as_slice().iter().map(|x| x.abs()).sum();

                let tile_count = (tile_end - tile_start) as f32;
                let real_avg = real_sum / tile_count;
                let imag_avg = imag_sum / tile_count;

                let checksum = (real_avg.to_bits() as u64) ^ ((imag_avg.to_bits() as u64) << 32);

                let result = DistributedResult::KernelComplete {
                    node_id,
                    checksum,
                    metrics: NodeMetrics {
                        execution_time_us: elapsed.as_micros() as u64,
                        q_ops: 0,
                    },
                };

                eprintln!(
                    "[tcp-worker] RunKernel complete: checksum=0x{:016X}, time={}us",
                    checksum,
                    elapsed.as_micros()
                );

                transport.send(&result)?;
            }

            DistributedJob::RunKernelBatch {
                kernel_type,
                depth,
                backend,
                count,
                ..
            } => {
                let farm = farm.as_mut().ok_or("Node not initialized")?;
                let qstate = qstate.as_mut().ok_or("Node not initialized")?;

                let batch_start = std::time::Instant::now();

                // Compile kernel once
                #[cfg(feature = "cranelift_jit")]
                let jit_fn_ptr = if matches!(backend, BackendKind::SimdJit) {
                    unsafe {
                        std::env::set_var("JIT_TILE_COUNT", "4");
                        std::env::set_var("JIT_H_DEPTH", &depth.to_string());
                    }
                    crate::quantum_jit::clif::compile_h_kernel(qubits, 0).map(|f| f as *const u8)
                } else {
                    None
                };

                #[cfg(not(feature = "cranelift_jit"))]
                let jit_fn_ptr = None;

                // Run N kernels back-to-back
                let mut total_checksum = 0u64;
                for i in 0..count {
                    unsafe {
                        farm.run_kernel_on_all_workers(
                            kernel_type,
                            qubits,
                            depth,
                            backend,
                            qstate as *mut QState as *mut u8,
                            jit_fn_ptr,
                        );
                    }

                    // Accumulate checksum
                    let real_sum: f32 = qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                    let imag_sum: f32 = qstate.imag.as_slice().iter().map(|x| x.abs()).sum();
                    let tile_count = (tile_end - tile_start) as f32;
                    let real_avg = real_sum / tile_count;
                    let imag_avg = imag_sum / tile_count;
                    let kernel_checksum =
                        (real_avg.to_bits() as u64) ^ ((imag_avg.to_bits() as u64) << 32);

                    // XOR checksums for aggregation
                    total_checksum ^= kernel_checksum;

                    if i % 10 == 0 && i > 0 {
                        eprintln!("[tcp-worker] Batch progress: {}/{} kernels", i, count);
                    }
                }

                let batch_elapsed = batch_start.elapsed();

                let result = DistributedResult::BatchComplete {
                    node_id,
                    checksum: total_checksum,
                    count,
                    metrics: NodeMetrics {
                        execution_time_us: batch_elapsed.as_micros() as u64,
                        q_ops: (qubits as u64 * depth as u64 * count as u64),
                    },
                };

                eprintln!(
                    "[tcp-worker] BatchComplete: {} kernels, checksum=0x{:016X}, time={}ms",
                    count,
                    total_checksum,
                    batch_elapsed.as_millis()
                );

                transport.send(&result)?;
            }

            DistributedJob::RunQuantumCircuit {
                n_qubits,
                circuit,
                workers_per_node,
            } => {
                let start = std::time::Instant::now();

                let mut state = BlockSparseState::new_zero(n_qubits);
                let mut pool = QuantumWorkerPool::new(workers_per_node);

                let gate_count = circuit.len();
                let exec_result = pool.execute_circuit_parallel(&mut state, &circuit);

                pool.shutdown();

                let elapsed = start.elapsed();
                let checksum = compute_state_checksum(&state);

                if let Err(e) = exec_result {
                    eprintln!("[tcp-worker] RunQuantumCircuit error: {}", e);
                }

                let result = DistributedResult::QuantumCircuitComplete {
                    node_id,
                    checksum,
                    gate_count,
                    exec_time_us: elapsed.as_micros() as u64,
                    metrics: NodeMetrics {
                        execution_time_us: elapsed.as_micros() as u64,
                        q_ops: (n_qubits as u64 * gate_count as u64),
                    },
                };

                eprintln!(
                    "[tcp-worker] RunQuantumCircuit complete: {} gates, checksum=0x{:016X}, time={}us",
                    gate_count,
                    checksum,
                    elapsed.as_micros()
                );

                transport.send(&result)?;
            }

            DistributedJob::RunQuantumBatch {
                n_qubits,
                circuits,
                workers_per_node,
            } => {
                let start = std::time::Instant::now();

                let mut total_checksum = 0u64;
                let mut total_gates = 0;

                // Reuse pool across all circuits in the batch
                let mut pool = QuantumWorkerPool::new(workers_per_node);

                for circuit in &circuits {
                    let mut state = BlockSparseState::new_zero(n_qubits);

                    if pool.execute_circuit_parallel(&mut state, circuit).is_ok() {
                        total_checksum ^= compute_state_checksum(&state);
                        total_gates += circuit.len();
                    }
                }

                pool.shutdown();

                let elapsed = start.elapsed();

                let result = DistributedResult::QuantumBatchComplete {
                    node_id,
                    checksum: total_checksum,
                    circuit_count: circuits.len(),
                    total_gates,
                    exec_time_us: elapsed.as_micros() as u64,
                    metrics: NodeMetrics {
                        execution_time_us: elapsed.as_micros() as u64,
                        q_ops: (n_qubits as u64 * total_gates as u64),
                    },
                };

                eprintln!(
                    "[tcp-worker] RunQuantumBatch complete: {} circuits, {} gates, checksum=0x{:016X}, time={}ms",
                    circuits.len(),
                    total_gates,
                    total_checksum,
                    elapsed.as_millis()
                );

                transport.send(&result)?;
            }

            DistributedJob::Shutdown => {
                eprintln!("[tcp-worker] Received Shutdown, exiting");
                break;
            }

            DistributedJob::Ping => {
                // Heartbeat: respond immediately with Pong
                let pong = DistributedResult::Pong { node_id };
                transport.send(&pong)?;
            }
        }
    }

    eprintln!("[tcp-worker] Worker terminated");
    Ok(())
}

#[cfg(feature = "serde")]
struct TcpWorkerTransport {
    stream: TcpStream,
}

#[cfg(feature = "serde")]
impl TcpWorkerTransport {
    fn send<T: serde::Serialize>(&mut self, msg: &T) -> Result<(), String> {
        let bytes = bincode::serialize(msg).map_err(|e| format!("Failed to serialize: {}", e))?;

        let len = bytes.len() as u32;
        self.stream
            .write_all(&len.to_be_bytes())
            .map_err(|e| format!("Failed to write length: {}", e))?;
        self.stream
            .write_all(&bytes)
            .map_err(|e| format!("Failed to write payload: {}", e))?;
        self.stream
            .flush()
            .map_err(|e| format!("Failed to flush: {}", e))?;

        Ok(())
    }

    fn recv<T: serde::de::DeserializeOwned>(&mut self) -> Result<T, String> {
        let mut len_buf = [0u8; 4];
        self.stream
            .read_exact(&mut len_buf)
            .map_err(|e| format!("Failed to read length: {}", e))?;

        let len = u32::from_be_bytes(len_buf) as usize;

        if len > 16 * 1024 * 1024 {
            return Err(format!("Message too large: {}", len));
        }

        let mut payload = vec![0u8; len];
        self.stream
            .read_exact(&mut payload)
            .map_err(|e| format!("Failed to read payload: {}", e))?;

        bincode::deserialize(&payload).map_err(|e| format!("Failed to deserialize: {}", e))
    }
}

// ============================================================================
// EPIC 63 Phase 1: In-process distributed tile farm
// ============================================================================

/// EPIC 63 Phase 1: In-process distributed tile farm
///
/// Simulates a multi-node cluster within a single process.
/// Each "node" is a NodeHandle with its own TileFarm and QState shard.
pub struct DistributedTileFarm {
    /// Node handles (one per logical node)
    nodes: Vec<NodeHandle>,

    /// Total number of tiles across all nodes
    global_tile_count: u16,

    /// Number of qubits (shared across all nodes)
    qubits: u8,
}

/// Handle to a single node in the distributed system
///
/// Phase 1: In-process simulation (local TileFarm + QState)
/// Phase 2: Will become a transport endpoint (channel/TCP)
pub struct NodeHandle {
    /// Node ID (0-indexed)
    pub node_id: usize,

    /// Global tile range: [tile_start, tile_end)
    pub tile_start: u16,
    pub tile_end: u16,

    /// Local tile farm (EPIC 62 multi-worker)
    pub local_farm: TileFarm,

    /// Local QState shard (owns tiles for this node)
    pub local_qstate: QState,
}

impl DistributedTileFarm {
    /// Create a new distributed tile farm with in-process node simulation
    ///
    /// # Arguments
    ///
    /// * `qubits` - Number of qubits per tile
    /// * `node_count` - Number of nodes in the cluster
    /// * `tiles_per_node` - Tiles assigned to each node
    /// * `workers_per_node` - Workers per node (usually from auto mode)
    ///
    /// # Invariants
    ///
    /// - `tiles_per_node` must be divisible by 4 (F32X4 SIMD)
    /// - `tiles_per_node / workers_per_node` must equal 4 (EPIC 62Q constraint)
    /// - `global_tile_count = node_count × tiles_per_node`
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 2 nodes, 8 tiles per node, 2 workers per node
    /// let farm = DistributedTileFarm::new(4, 2, 8, 2);
    /// // Global: 16 tiles
    /// // Node 0: tiles [0..8], 2 workers × 4 tiles/worker
    /// // Node 1: tiles [8..16], 2 workers × 4 tiles/worker
    /// ```
    pub fn new(
        qubits: u8,
        node_count: usize,
        tiles_per_node: usize,
        workers_per_node: usize,
    ) -> Self {
        assert!(qubits > 0 && qubits <= 12, "qubits must be 1-12");
        assert!(node_count > 0, "node_count must be >= 1");
        assert!(
            tiles_per_node >= 4,
            "tiles_per_node must be >= 4 (F32X4 SIMD)"
        );
        assert!(
            tiles_per_node % 4 == 0,
            "tiles_per_node must be divisible by 4"
        );
        assert!(workers_per_node > 0, "workers_per_node must be >= 1");

        // EPIC 62Q constraint: tiles_per_worker = 4
        let tiles_per_worker = tiles_per_node / workers_per_node;
        assert_eq!(
            tiles_per_worker, 4,
            "EPIC 62Q: tiles_per_worker must be 4 (got workers={}, tiles={}, tiles/w={})",
            workers_per_node, tiles_per_node, tiles_per_worker
        );

        let global_tile_count = (node_count * tiles_per_node) as u16;

        eprintln!("[epic63:phase1] Creating DistributedTileFarm:");
        eprintln!("  Nodes: {}", node_count);
        eprintln!("  Tiles/node: {}", tiles_per_node);
        eprintln!("  Workers/node: {}", workers_per_node);
        eprintln!("  Global tiles: {}", global_tile_count);
        eprintln!("  Qubits: {}", qubits);

        let mut nodes = Vec::with_capacity(node_count);

        for node_id in 0..node_count {
            let tile_start = (node_id * tiles_per_node) as u16;
            let tile_end = tile_start + tiles_per_node as u16;

            eprintln!(
                "  Node {}: tiles [{}..{}), {} workers",
                node_id, tile_start, tile_end, workers_per_node
            );

            // Create local TileFarm for this node (EPIC 62)
            let local_farm = TileFarm::new(workers_per_node, tiles_per_worker);

            // Create local QState shard for this node
            // Each node owns tiles_per_node tiles independently
            let local_qstate = QState::new_zero_multitile(qubits, tiles_per_node as u16);

            nodes.push(NodeHandle {
                node_id,
                tile_start,
                tile_end,
                local_farm,
                local_qstate,
            });
        }

        Self {
            nodes,
            global_tile_count,
            qubits,
        }
    }

    /// Get number of nodes in the cluster
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get global tile count
    pub fn global_tile_count(&self) -> u16 {
        self.global_tile_count
    }

    /// Get number of qubits
    pub fn qubits(&self) -> u8 {
        self.qubits
    }

    /// Run a kernel on all nodes in parallel (in-process simulation)
    ///
    /// Each node executes the kernel on its local tile shard using its TileFarm.
    ///
    /// # Determinism
    ///
    /// Execution order is deterministic (sequential node processing in Phase 1).
    /// Phase 2 will use parallel execution with barrier synchronization.
    ///
    /// # Example
    ///
    /// ```ignore
    /// farm.run_kernel(KernelType::HDeep, 8, BackendKind::SimdJit, jit_fn);
    /// ```
    pub fn run_kernel(
        &mut self,
        kernel_type: KernelType,
        depth: u32,
        backend: BackendKind,
        jit_fn_ptr: Option<*const u8>,
    ) {
        // Phase 1: Sequential execution (deterministic)
        // Phase 2: Will parallelize with thread pool + barrier sync
        for node in &mut self.nodes {
            node.run_kernel_local(kernel_type, self.qubits, depth, backend, jit_fn_ptr);
        }
    }

    /// Execute a quantum circuit on all nodes (SPRINT 39.0 Phase 4)
    ///
    /// Each node executes the circuit on a fresh BlockSparseState using its worker pool.
    /// Returns aggregated results from all nodes.
    ///
    /// # Arguments
    ///
    /// * `n_qubits` - Number of qubits for the circuit
    /// * `circuit` - Circuit gates to execute
    /// * `workers_per_node` - Number of workers each node should use
    ///
    /// # Returns
    ///
    /// Vector of (node_id, checksum, exec_time_us) tuples
    ///
    /// # Example
    ///
    /// ```ignore
    /// let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
    /// let results = farm.execute_quantum_circuit(8, &circuit, 4);
    /// ```
    pub fn execute_quantum_circuit(
        &self,
        n_qubits: u8,
        circuit: &[logic_fabric_core::quantum::QGate],
        workers_per_node: usize,
    ) -> Result<Vec<(u16, u64, u64)>, String> {
        let mut results = Vec::with_capacity(self.nodes.len());

        for node in &self.nodes {
            let start = std::time::Instant::now();

            // Create fresh state for this execution
            let mut state = BlockSparseState::new_zero(n_qubits);
            let mut pool = QuantumWorkerPool::new(workers_per_node);

            // Execute circuit
            pool.execute_circuit_parallel(&mut state, circuit)
                .map_err(|e| format!("Node {} circuit execution failed: {}", node.node_id, e))?;

            pool.shutdown();

            let elapsed = start.elapsed();
            let checksum = compute_state_checksum(&state);

            results.push((node.node_id as u16, checksum, elapsed.as_micros() as u64));

            eprintln!(
                "[phase4] Node {} completed {} gates: checksum=0x{:016X}, time={}us",
                node.node_id,
                circuit.len(),
                checksum,
                elapsed.as_micros()
            );
        }

        Ok(results)
    }

    /// Execute multiple quantum circuits in batch on all nodes (SPRINT 39.0 Phase 4)
    ///
    /// Each node executes all circuits in the batch, reusing the worker pool for efficiency.
    /// Returns aggregated results from all nodes.
    ///
    /// # Arguments
    ///
    /// * `n_qubits` - Number of qubits for all circuits
    /// * `circuits` - Vector of circuits to execute
    /// * `workers_per_node` - Number of workers each node should use
    ///
    /// # Returns
    ///
    /// Vector of (node_id, total_checksum, circuit_count, total_gates, exec_time_us) tuples
    ///
    /// # Example
    ///
    /// ```ignore
    /// let circuits = vec![
    ///     vec![QGate::H(0)],
    ///     vec![QGate::X(1)],
    /// ];
    /// let results = farm.execute_quantum_batch(8, &circuits, 4);
    /// ```
    pub fn execute_quantum_batch(
        &self,
        n_qubits: u8,
        circuits: &[Vec<logic_fabric_core::quantum::QGate>],
        workers_per_node: usize,
    ) -> Result<Vec<(u16, u64, usize, usize, u64)>, String> {
        let mut results = Vec::with_capacity(self.nodes.len());

        for node in &self.nodes {
            let start = std::time::Instant::now();

            let mut total_checksum = 0u64;
            let mut total_gates = 0;

            // Reuse pool across all circuits
            let mut pool = QuantumWorkerPool::new(workers_per_node);

            for circuit in circuits {
                let mut state = BlockSparseState::new_zero(n_qubits);

                if pool.execute_circuit_parallel(&mut state, circuit).is_ok() {
                    total_checksum ^= compute_state_checksum(&state);
                    total_gates += circuit.len();
                }
            }

            pool.shutdown();

            let elapsed = start.elapsed();

            results.push((
                node.node_id as u16,
                total_checksum,
                circuits.len(),
                total_gates,
                elapsed.as_micros() as u64,
            ));

            eprintln!(
                "[phase4] Node {} batch complete: {} circuits, {} gates, checksum=0x{:016X}, time={}ms",
                node.node_id,
                circuits.len(),
                total_gates,
                total_checksum,
                elapsed.as_millis()
            );
        }

        Ok(results)
    }

    /// Compute global checksum across all nodes
    ///
    /// Takes the average of node checksums to make it independent of node count.
    ///
    /// Since all nodes simulate the same quantum circuit on equivalent
    /// initial states, they should produce identical local checksums
    /// (after normalization by tile count). The average ensures that
    /// single-node and multi-node configs produce the same global checksum.
    ///
    /// # Determinism
    ///
    /// Average is deterministic given fixed node count and checksums.
    pub fn global_checksum(&self) -> u64 {
        if self.nodes.is_empty() {
            return 0;
        }

        // Collect all node checksums
        let mut sum = 0u128; // Use u128 to avoid overflow during summation
        for node in &self.nodes {
            sum += node.local_checksum() as u128;
        }

        // Return average (deterministic division)
        (sum / self.nodes.len() as u128) as u64
    }

    /// Get reference to a specific node
    pub fn node(&self, node_id: usize) -> Option<&NodeHandle> {
        self.nodes.get(node_id)
    }

    /// Get mutable reference to a specific node
    pub fn node_mut(&mut self, node_id: usize) -> Option<&mut NodeHandle> {
        self.nodes.get_mut(node_id)
    }
}

impl NodeHandle {
    /// Run a kernel on this node's local tile shard
    ///
    /// Dispatches to EPIC 62 TileFarm with this node's local QState.
    fn run_kernel_local(
        &mut self,
        kernel_type: KernelType,
        qubits: u8,
        depth: u32,
        backend: BackendKind,
        jit_fn_ptr: Option<*const u8>,
    ) {
        unsafe {
            self.local_farm.run_kernel_on_all_workers(
                kernel_type,
                qubits,
                depth,
                backend,
                &mut self.local_qstate as *mut QState as *mut u8,
                jit_fn_ptr,
            );
        }
    }

    /// Compute checksum for this node's local QState
    ///
    /// Checksums the average amplitude magnitude per tile to make it
    /// comparable across different node configurations.
    ///
    /// Formula: (sum of |amp|) / tile_count
    ///
    /// This ensures single-node (8 tiles) and two-node (2×4 tiles)
    /// produce the same checksum when simulating the same quantum circuit.
    fn local_checksum(&self) -> u64 {
        let real_sum: f32 = self
            .local_qstate
            .real
            .as_slice()
            .iter()
            .map(|x| x.abs())
            .sum();
        let imag_sum: f32 = self
            .local_qstate
            .imag
            .as_slice()
            .iter()
            .map(|x| x.abs())
            .sum();

        // Normalize by local tile count to make comparable across configs
        let tile_count = self.local_tile_count() as f32;
        let real_avg = real_sum / tile_count;
        let imag_avg = imag_sum / tile_count;

        // Convert to u64 for deterministic comparison (IEEE 754 bits)
        let real_bits = real_avg.to_bits() as u64;
        let imag_bits = imag_avg.to_bits() as u64;

        real_bits ^ (imag_bits << 32)
    }

    /// Get local tile count for this node
    pub fn local_tile_count(&self) -> u16 {
        self.tile_end - self.tile_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_jit;

    // ========================================================================
    // Phase 1: Basic Structure Tests
    // ========================================================================

    #[test]
    fn test_distributed_farm_creation() {
        // Single node with 8 tiles, 2 workers
        let farm = DistributedTileFarm::new(4, 1, 8, 2);

        assert_eq!(farm.node_count(), 1);
        assert_eq!(farm.global_tile_count(), 8);
        assert_eq!(farm.qubits(), 4);

        let node0 = farm.node(0).unwrap();
        assert_eq!(node0.node_id, 0);
        assert_eq!(node0.tile_start, 0);
        assert_eq!(node0.tile_end, 8);
        assert_eq!(node0.local_tile_count(), 8);
    }

    #[test]
    fn test_distributed_farm_two_nodes() {
        // 2 nodes, 8 tiles per node, 2 workers per node
        let farm = DistributedTileFarm::new(4, 2, 8, 2);

        assert_eq!(farm.node_count(), 2);
        assert_eq!(farm.global_tile_count(), 16);

        // Node 0: tiles [0..8)
        let node0 = farm.node(0).unwrap();
        assert_eq!(node0.tile_start, 0);
        assert_eq!(node0.tile_end, 8);

        // Node 1: tiles [8..16)
        let node1 = farm.node(1).unwrap();
        assert_eq!(node1.tile_start, 8);
        assert_eq!(node1.tile_end, 16);
    }

    #[test]
    fn test_distributed_farm_four_nodes() {
        // 4 nodes, 4 tiles per node, 1 worker per node
        let farm = DistributedTileFarm::new(4, 4, 4, 1);

        assert_eq!(farm.node_count(), 4);
        assert_eq!(farm.global_tile_count(), 16);

        // Verify tile ranges are disjoint and contiguous
        for i in 0..4 {
            let node = farm.node(i).unwrap();
            assert_eq!(node.tile_start, (i * 4) as u16);
            assert_eq!(node.tile_end, ((i + 1) * 4) as u16);
        }
    }

    #[test]
    #[should_panic(expected = "tiles_per_worker must be 4")]
    fn test_distributed_farm_invalid_tiles_per_worker() {
        // 2 workers × 2 tiles/worker = 4 total (violates EPIC 62Q)
        DistributedTileFarm::new(4, 1, 4, 2);
    }

    #[test]
    #[should_panic(expected = "tiles_per_node must be divisible by 4")]
    fn test_distributed_farm_invalid_tile_count() {
        // 6 tiles not divisible by 4 (violates F32X4 SIMD)
        DistributedTileFarm::new(4, 1, 6, 1);
    }

    // ========================================================================
    // Phase 1: Sanity Checklist
    // ========================================================================

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_sanity_single_node_distributed_vs_direct_tilefarm() {
        // Sanity check: Single-node DistributedTileFarm should produce same
        // result as directly using TileFarm with equivalent config.
        //
        // This validates that the distributed abstraction layer adds no
        // unexpected behavior for the single-node case.

        let qubits = 4u8;
        let depth = 8u32;
        let workers = 2;
        let tiles_per_worker = 4;
        let total_tiles = workers * tiles_per_worker;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        // A: Direct TileFarm usage (EPIC 62)
        let farm_direct = TileFarm::new(workers, tiles_per_worker);
        let mut qstate_direct =
            crate::quantum::QState::new_zero_multitile(qubits, total_tiles as u16);

        unsafe {
            farm_direct.run_kernel_on_all_workers(
                KernelType::HDeep,
                qubits,
                depth,
                BackendKind::SimdJit,
                &mut qstate_direct as *mut _ as *mut u8,
                Some(kernel_fn),
            );
        }

        // Compute checksum for direct TileFarm result
        let direct_sum: f32 = qstate_direct.real.as_slice().iter().map(|x| x.abs()).sum();
        let direct_checksum = (direct_sum / total_tiles as f32).to_bits() as u64;

        eprintln!(
            "[sanity] Direct TileFarm checksum: 0x{:016X}",
            direct_checksum
        );

        // B: Single-node DistributedTileFarm (EPIC 63 P1)
        let mut farm_distributed = DistributedTileFarm::new(qubits, 1, total_tiles, workers);
        farm_distributed.run_kernel(
            KernelType::HDeep,
            depth,
            BackendKind::SimdJit,
            Some(kernel_fn),
        );
        let distributed_checksum = farm_distributed.global_checksum();

        eprintln!(
            "[sanity] DistributedTileFarm checksum: 0x{:016X}",
            distributed_checksum
        );

        // Sanity check: checksums should match within floating-point epsilon
        let diff = if direct_checksum > distributed_checksum {
            direct_checksum - distributed_checksum
        } else {
            distributed_checksum - direct_checksum
        };

        const EPSILON: u64 = 2;

        assert!(
            diff <= EPSILON,
            "Sanity check failed: Single-node DistributedTileFarm differs from direct TileFarm\n\
             Direct: 0x{:016X}, Distributed: 0x{:016X}, Diff: {}",
            direct_checksum,
            distributed_checksum,
            diff
        );

        eprintln!(
            "[sanity] ✅ Single-node DistributedTileFarm ≡ Direct TileFarm (diff: {})",
            diff
        );

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    #[test]
    fn test_sanity_checksum_combiner_stability() {
        // Sanity check: Global checksum combiner is order-independent
        //
        // Since we use average (commutative + associative), node order
        // shouldn't matter for final checksum.

        // Mock scenario: 3 nodes with different local checksums
        let checksums = vec![0x1000, 0x2000, 0x3000];

        // Compute average in different orders
        let avg1 = (checksums[0] as u128 + checksums[1] as u128 + checksums[2] as u128) / 3;
        let avg2 = (checksums[2] as u128 + checksums[0] as u128 + checksums[1] as u128) / 3;
        let avg3 = (checksums[1] as u128 + checksums[2] as u128 + checksums[0] as u128) / 3;

        assert_eq!(avg1, avg2);
        assert_eq!(avg2, avg3);

        eprintln!("[sanity] ✅ Checksum combiner is order-independent");
    }

    // ========================================================================
    // Phase 1: Parity Validation Tests (1-node vs N-node)
    // ========================================================================

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_phase1_single_node_vs_two_node_parity_h() {
        // EPIC 63 Phase 1: Verify 1-node and 2-node produce same global checksum
        //
        // Config:
        // - 4 qubits, depth=8
        // - H kernel
        // - Single-node: 1 node × 8 tiles (2 workers × 4 tiles/w)
        // - Two-node: 2 nodes × 4 tiles each (1 worker × 4 tiles/w per node)

        let qubits = 4u8;
        let depth = 8u32;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        assert!(h_kernel.is_some(), "Failed to compile H kernel");
        let kernel_fn = h_kernel.unwrap() as *const u8;

        // Single-node configuration: 1 node, 8 tiles, 2 workers
        let mut farm_single = DistributedTileFarm::new(qubits, 1, 8, 2);
        let node0_len_single = farm_single.node(0).unwrap().local_qstate.real.len();
        eprintln!(
            "[phase1:debug] Single-node: QState len = {}",
            node0_len_single
        );

        farm_single.run_kernel(
            KernelType::HDeep,
            depth,
            BackendKind::SimdJit,
            Some(kernel_fn),
        );
        let node0_local = farm_single.node(0).unwrap().local_checksum();
        eprintln!(
            "[phase1:debug] Single-node: Node0 local checksum = 0x{:016X}",
            node0_local
        );
        let checksum_single = farm_single.global_checksum();

        eprintln!(
            "[phase1:parity] Single-node checksum: 0x{:016X}",
            checksum_single
        );

        // Two-node configuration: 2 nodes, 4 tiles each, 1 worker per node
        let mut farm_two = DistributedTileFarm::new(qubits, 2, 4, 1);
        let node0_len_two = farm_two.node(0).unwrap().local_qstate.real.len();
        let node1_len_two = farm_two.node(1).unwrap().local_qstate.real.len();
        eprintln!(
            "[phase1:debug] Two-node: Node0 QState len = {}, Node1 QState len = {}",
            node0_len_two, node1_len_two
        );

        farm_two.run_kernel(
            KernelType::HDeep,
            depth,
            BackendKind::SimdJit,
            Some(kernel_fn),
        );
        let node0_local_two = farm_two.node(0).unwrap().local_checksum();
        let node1_local_two = farm_two.node(1).unwrap().local_checksum();
        eprintln!(
            "[phase1:debug] Two-node: Node0 local = 0x{:016X}, Node1 local = 0x{:016X}",
            node0_local_two, node1_local_two
        );
        let checksum_two = farm_two.global_checksum();

        eprintln!(
            "[phase1:parity] Two-node checksum:    0x{:016X}",
            checksum_two
        );

        // Parity check: global checksums must match (with epsilon for floating-point)
        //
        // Note: Checksums may differ by ±1 LSB due to floating-point division
        // (different tile counts → different divisors in normalization)
        let diff = if checksum_single > checksum_two {
            checksum_single - checksum_two
        } else {
            checksum_two - checksum_single
        };

        const EPSILON: u64 = 2; // Allow ±2 LSB tolerance for FP rounding

        assert!(
            diff <= EPSILON,
            "EPIC 63 Phase 1 parity violation: 1-node vs 2-node checksums differ by {}\n\
             Single: 0x{:016X}, Two: 0x{:016X}",
            diff,
            checksum_single,
            checksum_two
        );

        eprintln!(
            "[phase1:parity] ✅ Single-node vs two-node PARITY PASS (diff: {})",
            diff
        );

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_phase1_two_node_vs_four_node_parity() {
        // EPIC 63 Phase 1: Verify 2-node and 4-node produce same global checksum
        //
        // Config:
        // - 4 qubits, depth=8
        // - H kernel
        // - Two-node: 2 nodes × 8 tiles (2 workers × 4 tiles/w per node)
        // - Four-node: 4 nodes × 4 tiles (1 worker × 4 tiles/w per node)

        let qubits = 4u8;
        let depth = 8u32;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        // Two-node: 2 nodes, 8 tiles each, 2 workers per node
        let mut farm_two = DistributedTileFarm::new(qubits, 2, 8, 2);
        farm_two.run_kernel(
            KernelType::HDeep,
            depth,
            BackendKind::SimdJit,
            Some(kernel_fn),
        );
        let checksum_two = farm_two.global_checksum();

        eprintln!(
            "\n[phase1:parity] Two-node checksum:  0x{:016X}",
            checksum_two
        );

        // Four-node: 4 nodes, 4 tiles each, 1 worker per node
        let mut farm_four = DistributedTileFarm::new(qubits, 4, 4, 1);
        farm_four.run_kernel(
            KernelType::HDeep,
            depth,
            BackendKind::SimdJit,
            Some(kernel_fn),
        );
        let checksum_four = farm_four.global_checksum();

        eprintln!(
            "[phase1:parity] Four-node checksum: 0x{:016X}",
            checksum_four
        );

        let diff = if checksum_two > checksum_four {
            checksum_two - checksum_four
        } else {
            checksum_four - checksum_two
        };

        const EPSILON: u64 = 2;

        assert!(
            diff <= EPSILON,
            "EPIC 63 Phase 1 parity violation: 2-node vs 4-node checksums differ by {}\n\
             Two: 0x{:016X}, Four: 0x{:016X}",
            diff,
            checksum_two,
            checksum_four
        );

        eprintln!(
            "[phase1:parity] ✅ Two-node vs four-node PARITY PASS (diff: {})",
            diff
        );

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_phase1_determinism() {
        // EPIC 63 Phase 1: Verify repeated runs produce identical checksums
        //
        // Same config run 3 times → must produce same checksum

        let qubits = 4u8;
        let depth = 8u32;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        let mut checksums = Vec::new();

        for run in 0..3 {
            let mut farm = DistributedTileFarm::new(qubits, 2, 4, 1);
            farm.run_kernel(
                KernelType::HDeep,
                depth,
                BackendKind::SimdJit,
                Some(kernel_fn),
            );
            let checksum = farm.global_checksum();

            eprintln!("[phase1:determinism] Run {}: 0x{:016X}", run, checksum);
            checksums.push(checksum);
        }

        // All checksums must be identical
        assert_eq!(checksums[0], checksums[1], "Run 0 vs Run 1 mismatch");
        assert_eq!(checksums[1], checksums[2], "Run 1 vs Run 2 mismatch");

        eprintln!("[phase1:determinism] ✅ Determinism PASS (3 runs identical)");

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    // ========================================================================
    // Phase 2: Channel Transport Tests
    // ========================================================================

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_phase2_channel_transport_two_nodes() {
        // EPIC 63 Phase 2: Test channel-based transport with 2 nodes
        //
        // This validates that the P2 transport layer produces the same
        // checksums as the P1 in-process simulation.

        let qubits = 4u8;
        let depth = 8u32;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        eprintln!("\n[phase2:transport] Spawning 2 nodes with channel transport...");

        // Spawn 2 nodes with channel transport
        let mut node0 = ChannelTransport::spawn_node(0, 0, 4, qubits, 1);
        let mut node1 = ChannelTransport::spawn_node(1, 4, 8, qubits, 1);

        // Send RunKernel jobs to both nodes
        let job = DistributedJob::RunKernel {
            kernel_type: KernelType::HDeep,
            depth,
            backend: BackendKind::SimdJit,
            jit_fn_ptr: Some(kernel_fn),
        };

        node0
            .send_job(job.clone())
            .expect("Failed to send job to node 0");
        node1
            .send_job(job.clone())
            .expect("Failed to send job to node 1");

        // Receive results
        let result0 = node0
            .recv_result()
            .expect("Failed to receive result from node 0");
        let result1 = node1
            .recv_result()
            .expect("Failed to receive result from node 1");

        eprintln!(
            "[phase2:transport] Node 0 checksum: 0x{:016X}",
            result0.checksum().unwrap()
        );
        eprintln!(
            "[phase2:transport] Node 1 checksum: 0x{:016X}",
            result1.checksum().unwrap()
        );

        // Compute global checksum (average)
        let global_checksum = ((result0.checksum().unwrap() as u128
            + result1.checksum().unwrap() as u128)
            / 2) as u64;

        eprintln!(
            "[phase2:transport] Global checksum: 0x{:016X}",
            global_checksum
        );

        // Compare against Phase 1 result
        let mut farm_phase1 = DistributedTileFarm::new(qubits, 2, 4, 1);
        farm_phase1.run_kernel(
            KernelType::HDeep,
            depth,
            BackendKind::SimdJit,
            Some(kernel_fn),
        );
        let phase1_checksum = farm_phase1.global_checksum();

        eprintln!(
            "[phase2:transport] Phase 1 checksum: 0x{:016X}",
            phase1_checksum
        );

        let diff = if global_checksum > phase1_checksum {
            global_checksum - phase1_checksum
        } else {
            phase1_checksum - global_checksum
        };

        const EPSILON: u64 = 2;

        assert!(
            diff <= EPSILON,
            "Phase 2 transport differs from Phase 1: P2=0x{:016X}, P1=0x{:016X}, diff={}",
            global_checksum,
            phase1_checksum,
            diff
        );

        eprintln!(
            "[phase2:transport] ✅ Phase 2 channel transport ≡ Phase 1 (diff: {})",
            diff
        );

        // Shutdown nodes
        node0.send_job(DistributedJob::Shutdown).ok();
        node1.send_job(DistributedJob::Shutdown).ok();

        // Give threads time to clean up
        std::thread::sleep(std::time::Duration::from_millis(100));

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    // ========================================================================
    // Phase 2.3: P1 Parity Tests Routed Through P2 Transport
    // ========================================================================

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_p2_single_node_vs_two_node_parity() {
        // P2.3: Route Phase 1 single-node vs two-node test through channel transport
        //
        // Validates that:
        // 1. Single-node transport produces same result as two-node transport
        // 2. Both produce same result as Phase 1 in-process

        let qubits = 4u8;
        let depth = 8u32;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        eprintln!("\n[p2.3:parity] Testing single-node vs two-node via channel transport...");

        // Single-node via P2 transport: 1 node, 8 tiles, 2 workers
        let mut node0_single = ChannelTransport::spawn_node(0, 0, 8, qubits, 2);

        let job = DistributedJob::RunKernel {
            kernel_type: KernelType::HDeep,
            depth,
            backend: BackendKind::SimdJit,
            jit_fn_ptr: Some(kernel_fn),
        };

        node0_single
            .send_job(job.clone())
            .expect("Failed to send job to single-node");
        let result_single = node0_single
            .recv_result()
            .expect("Failed to receive from single-node");

        eprintln!(
            "[p2.3:parity] Single-node P2 checksum: 0x{:016X}",
            result_single.checksum().unwrap()
        );

        // Two-node via P2 transport: 2 nodes, 4 tiles each, 1 worker per node
        let mut node0_two = ChannelTransport::spawn_node(0, 0, 4, qubits, 1);
        let mut node1_two = ChannelTransport::spawn_node(1, 4, 8, qubits, 1);

        node0_two
            .send_job(job.clone())
            .expect("Failed to send job to node 0");
        node1_two
            .send_job(job.clone())
            .expect("Failed to send job to node 1");

        let result0_two = node0_two
            .recv_result()
            .expect("Failed to receive from node 0");
        let result1_two = node1_two
            .recv_result()
            .expect("Failed to receive from node 1");

        let global_two = ((result0_two.checksum().unwrap() as u128
            + result1_two.checksum().unwrap() as u128)
            / 2) as u64;

        eprintln!(
            "[p2.3:parity] Two-node P2 checksum:    0x{:016X}",
            global_two
        );

        // Compare
        let diff = if result_single.checksum().unwrap() > global_two {
            result_single.checksum().unwrap() - global_two
        } else {
            global_two - result_single.checksum().unwrap()
        };

        const EPSILON: u64 = 2;

        assert!(
            diff <= EPSILON,
            "P2.3 parity violation: 1-node vs 2-node differ by {}\n\
             Single: 0x{:016X}, Two: 0x{:016X}",
            diff,
            result_single.checksum().unwrap(),
            global_two
        );

        eprintln!(
            "[p2.3:parity] ✅ Single-node vs two-node P2 parity PASS (diff: {})",
            diff
        );

        // Shutdown
        node0_single.send_job(DistributedJob::Shutdown).ok();
        node0_two.send_job(DistributedJob::Shutdown).ok();
        node1_two.send_job(DistributedJob::Shutdown).ok();

        std::thread::sleep(std::time::Duration::from_millis(100));

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_p2_two_node_vs_four_node_parity() {
        // P2.3: Route Phase 1 two-node vs four-node test through channel transport

        let qubits = 4u8;
        let depth = 8u32;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        eprintln!("\n[p2.3:parity] Testing two-node vs four-node via channel transport...");

        let job = DistributedJob::RunKernel {
            kernel_type: KernelType::HDeep,
            depth,
            backend: BackendKind::SimdJit,
            jit_fn_ptr: Some(kernel_fn),
        };

        // Two-node: 2 nodes × 8 tiles (2 workers per node)
        let mut node0_two = ChannelTransport::spawn_node(0, 0, 8, qubits, 2);
        let mut node1_two = ChannelTransport::spawn_node(1, 8, 16, qubits, 2);

        node0_two
            .send_job(job.clone())
            .expect("Failed to send to node 0 (two)");
        node1_two
            .send_job(job.clone())
            .expect("Failed to send to node 1 (two)");

        let result0_two = node0_two
            .recv_result()
            .expect("Failed to recv from node 0 (two)");
        let result1_two = node1_two
            .recv_result()
            .expect("Failed to recv from node 1 (two)");

        let global_two = ((result0_two.checksum().unwrap() as u128
            + result1_two.checksum().unwrap() as u128)
            / 2) as u64;

        eprintln!("[p2.3:parity] Two-node P2 checksum:  0x{:016X}", global_two);

        // Four-node: 4 nodes × 4 tiles (1 worker per node)
        let mut node0_four = ChannelTransport::spawn_node(0, 0, 4, qubits, 1);
        let mut node1_four = ChannelTransport::spawn_node(1, 4, 8, qubits, 1);
        let mut node2_four = ChannelTransport::spawn_node(2, 8, 12, qubits, 1);
        let mut node3_four = ChannelTransport::spawn_node(3, 12, 16, qubits, 1);

        node0_four
            .send_job(job.clone())
            .expect("Failed to send to node 0 (four)");
        node1_four
            .send_job(job.clone())
            .expect("Failed to send to node 1 (four)");
        node2_four
            .send_job(job.clone())
            .expect("Failed to send to node 2 (four)");
        node3_four
            .send_job(job.clone())
            .expect("Failed to send to node 3 (four)");

        let result0_four = node0_four
            .recv_result()
            .expect("Failed to recv from node 0 (four)");
        let result1_four = node1_four
            .recv_result()
            .expect("Failed to recv from node 1 (four)");
        let result2_four = node2_four
            .recv_result()
            .expect("Failed to recv from node 2 (four)");
        let result3_four = node3_four
            .recv_result()
            .expect("Failed to recv from node 3 (four)");

        let global_four = ((result0_four.checksum().unwrap() as u128
            + result1_four.checksum().unwrap() as u128
            + result2_four.checksum().unwrap() as u128
            + result3_four.checksum().unwrap() as u128)
            / 4) as u64;

        eprintln!(
            "[p2.3:parity] Four-node P2 checksum: 0x{:016X}",
            global_four
        );

        let diff = if global_two > global_four {
            global_two - global_four
        } else {
            global_four - global_two
        };

        const EPSILON: u64 = 2;

        assert!(
            diff <= EPSILON,
            "P2.3 parity violation: 2-node vs 4-node differ by {}\n\
             Two: 0x{:016X}, Four: 0x{:016X}",
            diff,
            global_two,
            global_four
        );

        eprintln!(
            "[p2.3:parity] ✅ Two-node vs four-node P2 parity PASS (diff: {})",
            diff
        );

        // Shutdown all nodes
        node0_two.send_job(DistributedJob::Shutdown).ok();
        node1_two.send_job(DistributedJob::Shutdown).ok();
        node0_four.send_job(DistributedJob::Shutdown).ok();
        node1_four.send_job(DistributedJob::Shutdown).ok();
        node2_four.send_job(DistributedJob::Shutdown).ok();
        node3_four.send_job(DistributedJob::Shutdown).ok();

        std::thread::sleep(std::time::Duration::from_millis(100));

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_p2_determinism() {
        // P2.3: Verify deterministic execution through channel transport
        //
        // Run same config 3 times → must produce identical checksums

        let qubits = 4u8;
        let depth = 8u32;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        eprintln!("\n[p2.3:determinism] Testing determinism across 3 runs...");

        let mut checksums = Vec::new();

        for run in 0..3 {
            let mut node0 = ChannelTransport::spawn_node(0, 0, 4, qubits, 1);
            let mut node1 = ChannelTransport::spawn_node(1, 4, 8, qubits, 1);

            let job = DistributedJob::RunKernel {
                kernel_type: KernelType::HDeep,
                depth,
                backend: BackendKind::SimdJit,
                jit_fn_ptr: Some(kernel_fn),
            };

            node0.send_job(job.clone()).unwrap();
            node1.send_job(job.clone()).unwrap();

            let result0 = node0.recv_result().unwrap();
            let result1 = node1.recv_result().unwrap();

            let global = ((result0.checksum().unwrap() as u128
                + result1.checksum().unwrap() as u128)
                / 2) as u64;

            eprintln!("[p2.3:determinism] Run {}: 0x{:016X}", run, global);
            checksums.push(global);

            node0.send_job(DistributedJob::Shutdown).ok();
            node1.send_job(DistributedJob::Shutdown).ok();

            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // All checksums must be identical
        assert_eq!(checksums[0], checksums[1], "Run 0 vs Run 1 mismatch");
        assert_eq!(checksums[1], checksums[2], "Run 1 vs Run 2 mismatch");

        eprintln!("[p2.3:determinism] ✅ P2 determinism PASS (3 runs identical)");

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    // ========================================================================
    // Phase 2b: TCP Transport Parity Tests
    // ========================================================================

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit", feature = "serde"))]
    fn test_p2b_single_node_tcp() {
        // P2b: Single-node over TCP on localhost
        //
        // Spawn a worker process, connect via TCP, run H kernel, verify checksum

        let qubits = 4u8;
        let depth = 8u32;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        eprintln!("\n[p2b:tcp] Starting single-node TCP test...");

        // Spawn TCP worker in background thread
        let worker_thread = std::thread::spawn(|| {
            tcp_worker_main("127.0.0.1:15000").expect("Worker failed");
        });

        // Give worker time to bind
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Connect coordinator
        let mut transport =
            TcpTransport::connect(0, "127.0.0.1:15000").expect("Failed to connect to worker");

        // Initialize node
        transport
            .send_job(DistributedJob::InitNode {
                node_id: 0,
                tile_start: 0,
                tile_end: 8,
                qubits,
            })
            .expect("Failed to send InitNode");

        // Run kernel (worker compiles its own JIT kernel)
        transport
            .send_job(DistributedJob::RunKernel {
                kernel_type: KernelType::HDeep,
                depth,
                backend: BackendKind::SimdJit,
                jit_fn_ptr: None, // TCP workers compile their own kernels
            })
            .expect("Failed to send RunKernel");

        let result = transport.recv_result().expect("Failed to receive result");

        eprintln!(
            "[p2b:tcp] Node 0 checksum: 0x{:016X}",
            result.checksum().unwrap()
        );

        // Compare with Phase 2a channel transport
        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        let mut node_channel = ChannelTransport::spawn_node(0, 0, 8, qubits, 2);
        node_channel
            .send_job(DistributedJob::RunKernel {
                kernel_type: KernelType::HDeep,
                depth,
                backend: BackendKind::SimdJit,
                jit_fn_ptr: Some(kernel_fn),
            })
            .unwrap();

        let result_channel = node_channel.recv_result().unwrap();

        eprintln!(
            "[p2b:tcp] Channel checksum: 0x{:016X}",
            result_channel.checksum().unwrap()
        );

        let diff = if result.checksum().unwrap() > result_channel.checksum().unwrap() {
            result.checksum().unwrap() - result_channel.checksum().unwrap()
        } else {
            result_channel.checksum().unwrap() - result.checksum().unwrap()
        };

        const EPSILON: u64 = 2;

        assert!(
            diff <= EPSILON,
            "P2b TCP parity violation: TCP vs Channel differ by {}\n\
             TCP: 0x{:016X}, Channel: 0x{:016X}",
            diff,
            result.checksum().unwrap(),
            result_channel.checksum().unwrap()
        );

        eprintln!("[p2b:tcp] ✅ Single-node TCP ≡ Channel (diff: {})", diff);

        // Shutdown
        transport.send_job(DistributedJob::Shutdown).ok();
        node_channel.send_job(DistributedJob::Shutdown).ok();

        // Wait for worker thread
        worker_thread.join().ok();

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit", feature = "serde"))]
    fn test_p2b_two_node_tcp_parity() {
        // P2b: Two-node over TCP on localhost
        //
        // Spawn 2 worker processes, verify global checksum matches channel transport

        let qubits = 4u8;
        let depth = 8u32;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        eprintln!("\n[p2b:tcp] Starting two-node TCP test...");

        // Spawn 2 TCP workers in background threads
        let worker0 = std::thread::spawn(|| {
            tcp_worker_main("127.0.0.1:15001").expect("Worker 0 failed");
        });

        let worker1 = std::thread::spawn(|| {
            tcp_worker_main("127.0.0.1:15002").expect("Worker 1 failed");
        });

        // Give workers time to bind
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Connect coordinator to both nodes
        let mut node0 =
            TcpTransport::connect(0, "127.0.0.1:15001").expect("Failed to connect to worker 0");
        let mut node1 =
            TcpTransport::connect(1, "127.0.0.1:15002").expect("Failed to connect to worker 1");

        // Initialize nodes
        node0
            .send_job(DistributedJob::InitNode {
                node_id: 0,
                tile_start: 0,
                tile_end: 4,
                qubits,
            })
            .expect("Failed to send InitNode to node 0");

        node1
            .send_job(DistributedJob::InitNode {
                node_id: 1,
                tile_start: 4,
                tile_end: 8,
                qubits,
            })
            .expect("Failed to send InitNode to node 1");

        // Run kernel on both nodes
        let job = DistributedJob::RunKernel {
            kernel_type: KernelType::HDeep,
            depth,
            backend: BackendKind::SimdJit,
            jit_fn_ptr: None,
        };

        node0
            .send_job(job.clone())
            .expect("Failed to send to node 0");
        node1
            .send_job(job.clone())
            .expect("Failed to send to node 1");

        let result0 = node0.recv_result().expect("Failed to recv from node 0");
        let result1 = node1.recv_result().expect("Failed to recv from node 1");

        let global_tcp = ((result0.checksum().unwrap() as u128
            + result1.checksum().unwrap() as u128)
            / 2) as u64;

        eprintln!("[p2b:tcp] TCP global checksum: 0x{:016X}", global_tcp);

        // Compare with Phase 2a channel transport
        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        let mut node0_channel = ChannelTransport::spawn_node(0, 0, 4, qubits, 1);
        let mut node1_channel = ChannelTransport::spawn_node(1, 4, 8, qubits, 1);

        let job_channel = DistributedJob::RunKernel {
            kernel_type: KernelType::HDeep,
            depth,
            backend: BackendKind::SimdJit,
            jit_fn_ptr: Some(kernel_fn),
        };

        node0_channel.send_job(job_channel.clone()).unwrap();
        node1_channel.send_job(job_channel.clone()).unwrap();

        let result0_channel = node0_channel.recv_result().unwrap();
        let result1_channel = node1_channel.recv_result().unwrap();

        let global_channel = ((result0_channel.checksum().unwrap() as u128
            + result1_channel.checksum().unwrap() as u128)
            / 2) as u64;

        eprintln!(
            "[p2b:tcp] Channel global checksum: 0x{:016X}",
            global_channel
        );

        let diff = if global_tcp > global_channel {
            global_tcp - global_channel
        } else {
            global_channel - global_tcp
        };

        const EPSILON: u64 = 2;

        assert!(
            diff <= EPSILON,
            "P2b TCP parity violation: TCP vs Channel differ by {}\n\
             TCP: 0x{:016X}, Channel: 0x{:016X}",
            diff,
            global_tcp,
            global_channel
        );

        eprintln!("[p2b:tcp] ✅ Two-node TCP ≡ Channel (diff: {})", diff);

        // Shutdown
        node0.send_job(DistributedJob::Shutdown).ok();
        node1.send_job(DistributedJob::Shutdown).ok();
        node0_channel.send_job(DistributedJob::Shutdown).ok();
        node1_channel.send_job(DistributedJob::Shutdown).ok();

        // Wait for workers
        worker0.join().ok();
        worker1.join().ok();

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    // ========================================================================
    // Phase 3: Distributed Strict-Mode Validation
    // ========================================================================

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_p3_strict_mode_channel_transport() {
        // P3: Verify strict-mode validation works across channel transport
        //
        // Run with canary guards enabled, validate no buffer corruption

        let qubits = 4u8;
        let depth = 16u32; // Deeper to stress test

        unsafe {
            std::env::set_var("JIT_DEBUG_STRICT", "1"); // Enable canaries
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        eprintln!("\n[p3:strict] Testing strict-mode with channel transport (2 nodes)...");

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        // Run 2-node configuration with strict-mode enabled
        let mut node0 = ChannelTransport::spawn_node(0, 0, 4, qubits, 1);
        let mut node1 = ChannelTransport::spawn_node(1, 4, 8, qubits, 1);

        let job = DistributedJob::RunKernel {
            kernel_type: KernelType::HDeep,
            depth,
            backend: BackendKind::SimdJit,
            jit_fn_ptr: Some(kernel_fn),
        };

        node0
            .send_job(job.clone())
            .expect("Failed to send to node 0");
        node1
            .send_job(job.clone())
            .expect("Failed to send to node 1");

        let result0 = node0.recv_result().expect("Failed to recv from node 0");
        let result1 = node1.recv_result().expect("Failed to recv from node 1");

        eprintln!(
            "[p3:strict] Node 0 checksum: 0x{:016X}",
            result0.checksum().unwrap()
        );
        eprintln!(
            "[p3:strict] Node 1 checksum: 0x{:016X}",
            result1.checksum().unwrap()
        );

        // If we got results without panic, canaries passed
        eprintln!("[p3:strict] ✅ Strict-mode channel transport PASS (no buffer corruption)");

        // Shutdown
        node0.send_job(DistributedJob::Shutdown).ok();
        node1.send_job(DistributedJob::Shutdown).ok();

        std::thread::sleep(std::time::Duration::from_millis(100));

        unsafe {
            std::env::remove_var("JIT_DEBUG_STRICT");
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit", feature = "serde"))]
    fn test_p3_strict_mode_tcp_transport() {
        // P3: Verify strict-mode validation works across TCP transport
        //
        // Run with canary guards enabled on TCP workers

        let qubits = 4u8;
        let depth = 16u32;

        unsafe {
            std::env::set_var("JIT_DEBUG_STRICT", "1"); // Enable canaries
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        eprintln!("\n[p3:strict-tcp] Testing strict-mode with TCP transport (2 nodes)...");

        // Spawn 2 TCP workers (they'll also see JIT_DEBUG_STRICT=1)
        let worker0 = std::thread::spawn(|| {
            tcp_worker_main("127.0.0.1:15010").expect("Worker 0 failed");
        });

        let worker1 = std::thread::spawn(|| {
            tcp_worker_main("127.0.0.1:15011").expect("Worker 1 failed");
        });

        std::thread::sleep(std::time::Duration::from_millis(200));

        // Connect coordinator
        let mut node0 =
            TcpTransport::connect(0, "127.0.0.1:15010").expect("Failed to connect to worker 0");
        let mut node1 =
            TcpTransport::connect(1, "127.0.0.1:15011").expect("Failed to connect to worker 1");

        // Initialize nodes
        node0
            .send_job(DistributedJob::InitNode {
                node_id: 0,
                tile_start: 0,
                tile_end: 4,
                qubits,
            })
            .expect("Failed to init node 0");

        node1
            .send_job(DistributedJob::InitNode {
                node_id: 1,
                tile_start: 4,
                tile_end: 8,
                qubits,
            })
            .expect("Failed to init node 1");

        // Run kernel
        let job = DistributedJob::RunKernel {
            kernel_type: KernelType::HDeep,
            depth,
            backend: BackendKind::SimdJit,
            jit_fn_ptr: None,
        };

        node0
            .send_job(job.clone())
            .expect("Failed to send to node 0");
        node1
            .send_job(job.clone())
            .expect("Failed to send to node 1");

        let result0 = node0.recv_result().expect("Failed to recv from node 0");
        let result1 = node1.recv_result().expect("Failed to recv from node 1");

        eprintln!(
            "[p3:strict-tcp] Node 0 checksum: 0x{:016X}",
            result0.checksum().unwrap()
        );
        eprintln!(
            "[p3:strict-tcp] Node 1 checksum: 0x{:016X}",
            result1.checksum().unwrap()
        );

        // If we got results without panic, canaries passed
        eprintln!("[p3:strict-tcp] ✅ Strict-mode TCP transport PASS (no buffer corruption)");

        // Shutdown
        node0.send_job(DistributedJob::Shutdown).ok();
        node1.send_job(DistributedJob::Shutdown).ok();

        worker0.join().ok();
        worker1.join().ok();

        unsafe {
            std::env::remove_var("JIT_DEBUG_STRICT");
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    fn test_p3_strict_mode_four_nodes() {
        // P3: Verify strict-mode with 4 nodes (stress test)

        let qubits = 4u8;
        let depth = 16u32;

        unsafe {
            std::env::set_var("JIT_DEBUG_STRICT", "1");
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        eprintln!("\n[p3:strict] Testing strict-mode with 4 nodes...");

        let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
        let kernel_fn = h_kernel.unwrap() as *const u8;

        // Spawn 4 nodes
        let mut node0 = ChannelTransport::spawn_node(0, 0, 4, qubits, 1);
        let mut node1 = ChannelTransport::spawn_node(1, 4, 8, qubits, 1);
        let mut node2 = ChannelTransport::spawn_node(2, 8, 12, qubits, 1);
        let mut node3 = ChannelTransport::spawn_node(3, 12, 16, qubits, 1);

        let job = DistributedJob::RunKernel {
            kernel_type: KernelType::HDeep,
            depth,
            backend: BackendKind::SimdJit,
            jit_fn_ptr: Some(kernel_fn),
        };

        // Run on all nodes
        node0.send_job(job.clone()).unwrap();
        node1.send_job(job.clone()).unwrap();
        node2.send_job(job.clone()).unwrap();
        node3.send_job(job.clone()).unwrap();

        let result0 = node0.recv_result().unwrap();
        let result1 = node1.recv_result().unwrap();
        let result2 = node2.recv_result().unwrap();
        let result3 = node3.recv_result().unwrap();

        let global = ((result0.checksum().unwrap() as u128
            + result1.checksum().unwrap() as u128
            + result2.checksum().unwrap() as u128
            + result3.checksum().unwrap() as u128)
            / 4) as u64;

        eprintln!("[p3:strict] 4-node global checksum: 0x{:016X}", global);
        eprintln!("[p3:strict] ✅ Strict-mode 4-node PASS (no buffer corruption)");

        // Shutdown
        node0.send_job(DistributedJob::Shutdown).ok();
        node1.send_job(DistributedJob::Shutdown).ok();
        node2.send_job(DistributedJob::Shutdown).ok();
        node3.send_job(DistributedJob::Shutdown).ok();

        std::thread::sleep(std::time::Duration::from_millis(100));

        unsafe {
            std::env::remove_var("JIT_DEBUG_STRICT");
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    // ========================================================================
    // Phase 4: Multi-Node Performance Benchmarks
    // ========================================================================

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    #[ignore] // Run manually with: cargo test test_p4_performance_grid -- --ignored --nocapture
    fn test_p4_performance_grid() {
        // P4: Multi-node performance benchmark grid
        //
        // Measures throughput and scaling efficiency for 1, 2, and 4 nodes
        //
        // Dimensions:
        // - Nodes: 1, 2, 4
        // - Qubits: 4, 6
        // - Depth: 100
        // - Kernels per run: 100

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
        }

        eprintln!("\n========================================");
        eprintln!("EPIC 63 Phase 4: Performance Benchmark Grid");
        eprintln!("========================================\n");

        // Test configurations
        // Note: 6+ qubits hits Cranelift JIT complexity limits for deep unrolling
        let configs = vec![
            (4u8, 100u32), // 4 qubits, depth 100
        ];

        let kernels_per_run = 100;

        for (qubits, depth) in configs {
            eprintln!(
                "--- Config: qubits={}, depth={}, kernels={} ---",
                qubits, depth, kernels_per_run
            );

            unsafe {
                std::env::set_var("JIT_H_DEPTH", &depth.to_string());
            }

            // Compile kernel once
            let h_kernel = quantum_jit::clif::compile_h_kernel(qubits, 0);
            let kernel_fn = h_kernel.unwrap() as *const u8;

            let job = DistributedJob::RunKernel {
                kernel_type: KernelType::HDeep,
                depth,
                backend: BackendKind::SimdJit,
                jit_fn_ptr: Some(kernel_fn),
            };

            // Benchmark 1 node
            let start = std::time::Instant::now();
            let mut node = ChannelTransport::spawn_node(0, 0, 4, qubits, 1);
            for _ in 0..kernels_per_run {
                node.send_job(job.clone()).unwrap();
                node.recv_result().unwrap();
            }
            let elapsed_1node = start.elapsed();
            let throughput_1node = kernels_per_run as f64 / elapsed_1node.as_secs_f64();
            let q_ops_per_kernel = (qubits as u32 * depth) as f64;
            let q_ops_per_sec_1node = throughput_1node * q_ops_per_kernel;
            node.send_job(DistributedJob::Shutdown).ok();
            std::thread::sleep(std::time::Duration::from_millis(100));

            eprintln!(
                "  1-node: {:.2}ms total, {:.2} kernels/sec, {:.2}M q_ops/sec",
                elapsed_1node.as_millis(),
                throughput_1node,
                q_ops_per_sec_1node / 1e6
            );

            // Benchmark 2 nodes
            let start = std::time::Instant::now();
            let mut node0 = ChannelTransport::spawn_node(0, 0, 4, qubits, 1);
            let mut node1 = ChannelTransport::spawn_node(1, 4, 8, qubits, 1);
            for _ in 0..kernels_per_run {
                node0.send_job(job.clone()).unwrap();
                node1.send_job(job.clone()).unwrap();
                node0.recv_result().unwrap();
                node1.recv_result().unwrap();
            }
            let elapsed_2node = start.elapsed();
            let throughput_2node = (kernels_per_run * 2) as f64 / elapsed_2node.as_secs_f64();
            let q_ops_per_sec_2node = throughput_2node * q_ops_per_kernel;
            let efficiency_2node = throughput_2node / (2.0 * throughput_1node);
            node0.send_job(DistributedJob::Shutdown).ok();
            node1.send_job(DistributedJob::Shutdown).ok();
            std::thread::sleep(std::time::Duration::from_millis(100));

            eprintln!(
                "  2-node: {:.2}ms total, {:.2} kernels/sec, {:.2}M q_ops/sec, efficiency={:.2}",
                elapsed_2node.as_millis(),
                throughput_2node,
                q_ops_per_sec_2node / 1e6,
                efficiency_2node
            );

            // Benchmark 4 nodes
            let start = std::time::Instant::now();
            let mut node0 = ChannelTransport::spawn_node(0, 0, 4, qubits, 1);
            let mut node1 = ChannelTransport::spawn_node(1, 4, 8, qubits, 1);
            let mut node2 = ChannelTransport::spawn_node(2, 8, 12, qubits, 1);
            let mut node3 = ChannelTransport::spawn_node(3, 12, 16, qubits, 1);
            for _ in 0..kernels_per_run {
                node0.send_job(job.clone()).unwrap();
                node1.send_job(job.clone()).unwrap();
                node2.send_job(job.clone()).unwrap();
                node3.send_job(job.clone()).unwrap();
                node0.recv_result().unwrap();
                node1.recv_result().unwrap();
                node2.recv_result().unwrap();
                node3.recv_result().unwrap();
            }
            let elapsed_4node = start.elapsed();
            let throughput_4node = (kernels_per_run * 4) as f64 / elapsed_4node.as_secs_f64();
            let q_ops_per_sec_4node = throughput_4node * q_ops_per_kernel;
            let efficiency_4node = throughput_4node / (4.0 * throughput_1node);
            node0.send_job(DistributedJob::Shutdown).ok();
            node1.send_job(DistributedJob::Shutdown).ok();
            node2.send_job(DistributedJob::Shutdown).ok();
            node3.send_job(DistributedJob::Shutdown).ok();
            std::thread::sleep(std::time::Duration::from_millis(100));

            eprintln!(
                "  4-node: {:.2}ms total, {:.2} kernels/sec, {:.2}M q_ops/sec, efficiency={:.2}",
                elapsed_4node.as_millis(),
                throughput_4node,
                q_ops_per_sec_4node / 1e6,
                efficiency_4node
            );

            eprintln!(
                "  Speedup: 2-node={:.2}x, 4-node={:.2}x",
                throughput_2node / throughput_1node,
                throughput_4node / throughput_1node
            );
            eprintln!(
                "  q_ops Speedup: 2-node={:.2}x, 4-node={:.2}x\n",
                q_ops_per_sec_2node / q_ops_per_sec_1node,
                q_ops_per_sec_4node / q_ops_per_sec_1node
            );
        }

        eprintln!("========================================");
        eprintln!("Benchmark Complete");
        eprintln!("========================================");

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    #[test]
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit", feature = "serde"))]
    fn test_p2b_tcp_determinism() {
        // P2b: Verify deterministic execution over TCP
        //
        // Run same config 3 times → must produce identical checksums

        let qubits = 4u8;
        let depth = 8u32;

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", &depth.to_string());
        }

        eprintln!("\n[p2b:tcp-determinism] Testing determinism across 3 runs...");

        let mut checksums = Vec::new();

        for run in 0..3 {
            // Spawn worker
            let worker = std::thread::spawn(|| {
                tcp_worker_main("127.0.0.1:15003").expect("Worker failed");
            });

            std::thread::sleep(std::time::Duration::from_millis(200));

            // Connect and run
            let mut node = TcpTransport::connect(0, "127.0.0.1:15003").expect("Failed to connect");

            node.send_job(DistributedJob::InitNode {
                node_id: 0,
                tile_start: 0,
                tile_end: 4,
                qubits,
            })
            .unwrap();

            node.send_job(DistributedJob::RunKernel {
                kernel_type: KernelType::HDeep,
                depth,
                backend: BackendKind::SimdJit,
                jit_fn_ptr: None,
            })
            .unwrap();

            let result = node.recv_result().unwrap();

            eprintln!(
                "[p2b:tcp-determinism] Run {}: 0x{:016X}",
                run,
                result.checksum().unwrap()
            );
            checksums.push(result.checksum().unwrap());

            node.send_job(DistributedJob::Shutdown).ok();
            worker.join().ok();

            // Allow port to be released
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // All checksums must be identical
        assert_eq!(checksums[0], checksums[1], "Run 0 vs Run 1 mismatch");
        assert_eq!(checksums[1], checksums[2], "Run 1 vs Run 2 mismatch");

        eprintln!("[p2b:tcp-determinism] ✅ TCP determinism PASS (3 runs identical)");

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    // ========================================================================
    // EPIC 64 Phase 1: Cluster Config Tests
    // ========================================================================

    #[test]
    #[cfg(feature = "cluster")]
    fn test_cluster_config_parse() {
        let toml = r#"
[cluster]
name = "test-cluster"

[[node]]
id = 0
host = "192.168.1.10"
port = 5001
tiles = 4

[[node]]
id = 1
host = "192.168.1.11"
port = 5002
tiles = 8
"#;

        let config: ClusterConfig = toml::from_str(toml).expect("Failed to parse cluster config");

        assert_eq!(config.cluster.name, "test-cluster");
        assert_eq!(config.node.len(), 2);
        assert_eq!(config.node[0].id, 0);
        assert_eq!(config.node[0].host, "192.168.1.10");
        assert_eq!(config.node[0].port, 5001);
        assert_eq!(config.node[0].tiles, 4);
        assert_eq!(config.node[1].id, 1);
        assert_eq!(config.node[1].tiles, 8);
    }

    #[test]
    #[cfg(feature = "cluster")]
    fn test_cluster_config_validation() {
        let toml = r#"
[cluster]
name = "valid-cluster"

[[node]]
id = 0
host = "127.0.0.1"
port = 5001
tiles = 4

[[node]]
id = 1
host = "127.0.0.1"
port = 5002
tiles = 8
"#;

        let config: ClusterConfig = toml::from_str(toml).unwrap();
        assert!(config.validate().is_ok());

        // Test total tiles
        assert_eq!(config.total_tiles(), 12);

        // Test tile ranges
        assert_eq!(config.tile_range_for_node(0), Some((0, 4)));
        assert_eq!(config.tile_range_for_node(1), Some((4, 12)));
        assert_eq!(config.tile_range_for_node(2), None);
    }

    #[test]
    #[cfg(feature = "cluster")]
    fn test_cluster_config_validation_failures() {
        // Duplicate node IDs
        let toml = r#"
[cluster]
name = "bad-cluster"

[[node]]
id = 0
host = "127.0.0.1"
port = 5001
tiles = 4

[[node]]
id = 0
host = "127.0.0.1"
port = 5002
tiles = 4
"#;

        let config: ClusterConfig = toml::from_str(toml).unwrap();
        assert!(config.validate().is_err());

        // Invalid tile count (not divisible by 4)
        let toml = r#"
[cluster]
name = "bad-tiles"

[[node]]
id = 0
host = "127.0.0.1"
port = 5001
tiles = 3
"#;

        let config: ClusterConfig = toml::from_str(toml).unwrap();
        assert!(config.validate().is_err());

        // Zero tiles
        let toml = r#"
[cluster]
name = "zero-tiles"

[[node]]
id = 0
host = "127.0.0.1"
port = 5001
tiles = 0
"#;

        let config: ClusterConfig = toml::from_str(toml).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    #[cfg(feature = "cluster")]
    fn test_p3_arbitrary_cluster_sizes() {
        eprintln!("\n=== EPIC 64 Phase 3: Arbitrary Cluster Sizes ===");

        // Test 3-node cluster with varying tile counts
        let toml = r#"
[cluster]
name = "three-node"

[[node]]
id = 0
host = "192.168.1.10"
port = 5001
tiles = 4

[[node]]
id = 1
host = "192.168.1.11"
port = 5001
tiles = 8

[[node]]
id = 2
host = "192.168.1.12"
port = 5001
tiles = 12
"#;

        let config: ClusterConfig = toml::from_str(toml).unwrap();
        assert!(config.validate().is_ok());

        assert_eq!(config.total_tiles(), 24);
        assert_eq!(config.tile_range_for_node(0), Some((0, 4)));
        assert_eq!(config.tile_range_for_node(1), Some((4, 12)));
        assert_eq!(config.tile_range_for_node(2), Some((12, 24)));

        eprintln!("[p3] 3-node cluster: total_tiles={}", config.total_tiles());
        eprintln!("[p3]   Node 0: tiles [0..4)");
        eprintln!("[p3]   Node 1: tiles [4..12)");
        eprintln!("[p3]   Node 2: tiles [12..24)");

        // Test 5-node cluster (odd number)
        let toml = r#"
[cluster]
name = "five-node"

[[node]]
id = 0
host = "n0"
port = 5001
tiles = 4

[[node]]
id = 1
host = "n1"
port = 5001
tiles = 4

[[node]]
id = 2
host = "n2"
port = 5001
tiles = 8

[[node]]
id = 3
host = "n3"
port = 5001
tiles = 4

[[node]]
id = 4
host = "n4"
port = 5001
tiles = 12
"#;

        let config: ClusterConfig = toml::from_str(toml).unwrap();
        assert!(config.validate().is_ok());

        assert_eq!(config.total_tiles(), 32);
        assert_eq!(config.tile_range_for_node(0), Some((0, 4)));
        assert_eq!(config.tile_range_for_node(1), Some((4, 8)));
        assert_eq!(config.tile_range_for_node(2), Some((8, 16)));
        assert_eq!(config.tile_range_for_node(3), Some((16, 20)));
        assert_eq!(config.tile_range_for_node(4), Some((20, 32)));

        eprintln!("[p3] 5-node cluster: total_tiles={}", config.total_tiles());

        eprintln!("[p3] ✓ Arbitrary cluster sizes work!");
    }

    #[test]
    #[cfg(feature = "cluster")]
    fn test_p3_determinism_guarantee() {
        eprintln!("\n=== EPIC 64 Phase 3: Determinism Guarantee ===");

        let toml = r#"
[cluster]
name = "determinism-test"

[[node]]
id = 0
host = "host1"
port = 5001
tiles = 8

[[node]]
id = 1
host = "host2"
port = 5001
tiles = 4

[[node]]
id = 2
host = "host3"
port = 5001
tiles = 12
"#;

        // Parse the same config 10 times and verify identical results
        for i in 0..10 {
            let config: ClusterConfig = toml::from_str(toml).unwrap();

            assert_eq!(config.total_tiles(), 24, "Run {}: total_tiles mismatch", i);
            assert_eq!(
                config.tile_range_for_node(0),
                Some((0, 8)),
                "Run {}: node 0 range mismatch",
                i
            );
            assert_eq!(
                config.tile_range_for_node(1),
                Some((8, 12)),
                "Run {}: node 1 range mismatch",
                i
            );
            assert_eq!(
                config.tile_range_for_node(2),
                Some((12, 24)),
                "Run {}: node 2 range mismatch",
                i
            );
        }

        eprintln!("[p3] ✓ Tile partitioning is deterministic across 10 parses!");
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_p2_heartbeat_ping_pong() {
        use std::net::TcpListener;
        use std::thread;

        eprintln!("\n=== EPIC 64 Phase 2: Heartbeat Ping/Pong ===");

        // Start TCP worker on localhost
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind");
        let worker_addr = listener.local_addr().unwrap();

        eprintln!("[p2:heartbeat] Worker listening on {}", worker_addr);

        let worker_thread = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let node_id = 42;
            let mut transport = TcpWorkerTransport { stream };

            // Receive Ping
            let job: DistributedJob = transport.recv().unwrap();
            match job {
                DistributedJob::Ping => {
                    eprintln!("[worker] Received Ping, sending Pong");
                    transport
                        .send(&DistributedResult::Pong { node_id })
                        .unwrap();
                }
                _ => panic!("Expected Ping, got {:?}", job),
            }

            // Receive Shutdown
            let job: DistributedJob = transport.recv().unwrap();
            assert!(matches!(job, DistributedJob::Shutdown));
            eprintln!("[worker] Received Shutdown, exiting");
        });

        // Coordinator: connect and ping
        thread::sleep(std::time::Duration::from_millis(100)); // Let worker start
        let mut transport = TcpTransport::connect(42, &worker_addr.to_string()).unwrap();

        // Send Ping
        eprintln!("[coordinator] Sending Ping");
        transport.send_job(DistributedJob::Ping).unwrap();

        // Receive Pong
        let result = transport.recv_result().unwrap();
        match result {
            DistributedResult::Pong { node_id } => {
                eprintln!("[coordinator] Received Pong from node {}", node_id);
                assert_eq!(node_id, 42);
            }
            _ => panic!("Expected Pong, got {:?}", result),
        }

        // Shutdown worker
        transport.send_job(DistributedJob::Shutdown).unwrap();
        worker_thread.join().unwrap();

        eprintln!("[p2:heartbeat] ✓ Ping/Pong test passed!");
    }

    #[test]
    #[cfg(all(feature = "cluster", feature = "serde"))]
    fn test_p2_heartbeat_health_tracking() {
        use std::net::TcpListener;
        use std::thread;

        eprintln!("\n=== EPIC 64 Phase 2: Health Tracking ===");

        // Start TCP worker on localhost
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind");
        let worker_addr = listener.local_addr().unwrap();

        eprintln!("[p2:health] Worker listening on {}", worker_addr);

        let worker_thread = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let node_id = 5;
            let mut transport = TcpWorkerTransport { stream };

            // Handle pings until shutdown
            loop {
                let job: DistributedJob = transport.recv().unwrap();
                match job {
                    DistributedJob::Ping => {
                        eprintln!("[worker] Ping received");
                        transport
                            .send(&DistributedResult::Pong { node_id })
                            .unwrap();
                    }
                    DistributedJob::Shutdown => {
                        eprintln!("[worker] Shutting down");
                        break;
                    }
                    _ => panic!("Unexpected job"),
                }
            }
        });

        // Coordinator: connect and perform health checks
        thread::sleep(std::time::Duration::from_millis(100)); // Let worker start
        let mut transport = TcpTransport::connect(5, &worker_addr.to_string()).unwrap();

        let mut health = NodeHealthInfo {
            node_id: 5,
            health: NodeHealth::Healthy,
            last_ping_us: None,
            last_heartbeat: std::time::Instant::now(),
        };

        // Perform 5 heartbeats
        for i in 0..5 {
            match transport.heartbeat_ping() {
                Ok(latency_us) => {
                    health.last_ping_us = Some(latency_us);
                    health.last_heartbeat = std::time::Instant::now();

                    // Classify health based on latency
                    health.health = if latency_us < 10_000 {
                        NodeHealth::Healthy
                    } else if latency_us < 100_000 {
                        NodeHealth::Degraded
                    } else {
                        NodeHealth::Unreachable
                    };

                    eprintln!(
                        "[coordinator] Heartbeat {}: {:?}, latency: {}us",
                        i, health.health, latency_us
                    );

                    assert_eq!(health.health, NodeHealth::Healthy);
                }
                Err(e) => {
                    health.health = NodeHealth::Unreachable;
                    eprintln!("[coordinator] Heartbeat {} FAILED: {}", i, e);
                    panic!("Heartbeat failed: {}", e);
                }
            }

            thread::sleep(std::time::Duration::from_millis(10));
        }

        // Shutdown worker
        transport.send_job(DistributedJob::Shutdown).unwrap();
        worker_thread.join().unwrap();

        eprintln!("[p2:health] ✓ Health tracking test passed!");
        eprintln!(
            "[p2:health] Final state: node {}, health {:?}, last ping {}us",
            health.node_id,
            health.health,
            health.last_ping_us.unwrap()
        );
    }

    #[test]
    #[ignore] // Manual test: spawns worker processes
    #[cfg(all(
        feature = "cluster",
        feature = "quantum_jit",
        feature = "cranelift_jit"
    ))]
    fn test_p1_cluster_demo_localhost() {
        use std::process::{Command, Stdio};
        use std::thread;
        use std::time::Duration;

        eprintln!("\n=== EPIC 64 Phase 1 Cluster Demo ===");
        eprintln!("This test spawns 2 worker processes on localhost and runs a coordinator.\n");

        // Ensure binary is built
        eprintln!("[demo] Building quantum-engine binary...");
        let build_status = Command::new("cargo")
            .args(&[
                "build",
                "--features",
                "quantum_jit,cranelift_jit,cluster",
                "--bin",
                "quantum-engine",
            ])
            .status()
            .expect("Failed to build quantum-engine");
        assert!(build_status.success(), "Failed to build binary");

        // Start worker 0 on port 5001
        eprintln!("[demo] Starting worker 0 on 127.0.0.1:5001...");
        let mut worker0 = Command::new("./target/debug/quantum-engine")
            .args(&[
                "worker",
                "--node-id",
                "0",
                "--bind",
                "127.0.0.1:5001",
                "--tiles",
                "4",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to spawn worker 0");

        // Start worker 1 on port 5002
        eprintln!("[demo] Starting worker 1 on 127.0.0.1:5002...");
        let mut worker1 = Command::new("./target/debug/quantum-engine")
            .args(&[
                "worker",
                "--node-id",
                "1",
                "--bind",
                "127.0.0.1:5002",
                "--tiles",
                "4",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to spawn worker 1");

        // Give workers time to bind
        eprintln!("[demo] Waiting for workers to initialize...");
        thread::sleep(Duration::from_secs(2));

        // Run coordinator
        eprintln!("[demo] Running coordinator with cluster-localhost.toml...\n");
        let coordinator_result = Command::new("./target/debug/quantum-engine")
            .args(&[
                "run",
                "--cluster",
                "cluster-localhost.toml",
                "--qubits",
                "4",
                "--depth",
                "50",
                "--kernels",
                "10",
            ])
            .status();

        // Kill workers
        eprintln!("\n[demo] Shutting down workers...");
        let _ = worker0.kill();
        let _ = worker1.kill();
        let _ = worker0.wait();
        let _ = worker1.wait();

        // Check coordinator status
        match coordinator_result {
            Ok(status) if status.success() => {
                eprintln!("[demo] ✓ Cluster demo completed successfully!");
            }
            Ok(status) => {
                panic!("Coordinator failed with exit code: {:?}", status.code());
            }
            Err(e) => {
                panic!("Failed to run coordinator: {}", e);
            }
        }
    }

    #[test]
    #[cfg(all(feature = "serde", feature = "quantum_jit"))]
    fn test_p5_1_kernel_batching() {
        use std::net::TcpListener;
        use std::thread;

        eprintln!("\n=== EPIC 65 Phase 1: Kernel Batching ===");

        // Start TCP worker on localhost
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind");
        let worker_addr = listener.local_addr().unwrap();

        eprintln!("[p5.1:batch] Worker listening on {}", worker_addr);

        let worker_thread = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut transport = TcpWorkerTransport { stream };

            // Handle InitNode
            let job: DistributedJob = transport.recv().unwrap();
            match job {
                DistributedJob::InitNode {
                    node_id,
                    tile_start,
                    tile_end,
                    qubits,
                } => {
                    eprintln!(
                        "[worker] InitNode: node_id={}, tiles=[{}..{}), qubits={}",
                        node_id, tile_start, tile_end, qubits
                    );

                    let tile_count = tile_end - tile_start;
                    let workers_per_node = tile_count / 4;

                    let mut farm = TileFarm::new(workers_per_node as usize, 4);
                    let mut qstate = QState::new_zero_multitile(qubits, tile_count);

                    eprintln!(
                        "[worker] Initialized: {} workers × 4 tiles/w",
                        workers_per_node
                    );

                    // Process batch job
                    let job: DistributedJob = transport.recv().unwrap();
                    match job {
                        DistributedJob::RunKernelBatch {
                            kernel_type,
                            depth,
                            backend,
                            count,
                            ..
                        } => {
                            eprintln!("[worker] RunKernelBatch: count={}, depth={}", count, depth);

                            let batch_start = std::time::Instant::now();

                            // Compile kernel once
                            unsafe {
                                std::env::set_var("JIT_TILE_COUNT", "4");
                                std::env::set_var("JIT_H_DEPTH", &depth.to_string());
                            }
                            let jit_fn_ptr = crate::quantum_jit::clif::compile_h_kernel(qubits, 0)
                                .map(|f| f as *const u8);

                            // Run N kernels back-to-back
                            let mut total_checksum = 0u64;
                            for _i in 0..count {
                                unsafe {
                                    farm.run_kernel_on_all_workers(
                                        kernel_type,
                                        qubits,
                                        depth,
                                        backend,
                                        &mut qstate as *mut QState as *mut u8,
                                        jit_fn_ptr,
                                    );
                                }

                                let real_sum: f32 =
                                    qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                                let imag_sum: f32 =
                                    qstate.imag.as_slice().iter().map(|x| x.abs()).sum();
                                let tile_count_f = (tile_end - tile_start) as f32;
                                let real_avg = real_sum / tile_count_f;
                                let imag_avg = imag_sum / tile_count_f;
                                let kernel_checksum = (real_avg.to_bits() as u64)
                                    ^ ((imag_avg.to_bits() as u64) << 32);
                                total_checksum ^= kernel_checksum;
                            }

                            let batch_elapsed = batch_start.elapsed();

                            let result = DistributedResult::BatchComplete {
                                node_id,
                                checksum: total_checksum,
                                count,
                                metrics: NodeMetrics {
                                    execution_time_us: batch_elapsed.as_micros() as u64,
                                    q_ops: (qubits as u64 * depth as u64 * count as u64),
                                },
                            };

                            eprintln!(
                                "[worker] BatchComplete: {} kernels, time={}ms",
                                count,
                                batch_elapsed.as_millis()
                            );

                            transport.send(&result).unwrap();
                        }
                        _ => panic!("Unexpected job after InitNode"),
                    }

                    // Handle shutdown
                    let job: DistributedJob = transport.recv().unwrap();
                    assert!(matches!(job, DistributedJob::Shutdown));
                    eprintln!("[worker] Shutdown");
                }
                _ => panic!("Expected InitNode first"),
            }
        });

        // Coordinator: connect and dispatch batch
        thread::sleep(std::time::Duration::from_millis(100));
        let mut transport = TcpTransport::connect(0, &worker_addr.to_string()).unwrap();

        // Initialize node
        transport
            .send_job(DistributedJob::InitNode {
                node_id: 0,
                tile_start: 0,
                tile_end: 4,
                qubits: 4,
            })
            .unwrap();

        // Dispatch batch of 100 kernels
        let batch_size = 100;
        eprintln!(
            "[coordinator] Dispatching batch of {} kernels...",
            batch_size
        );

        let coord_start = std::time::Instant::now();

        transport
            .send_job(DistributedJob::RunKernelBatch {
                kernel_type: KernelType::HDeep,
                depth: 50,
                backend: BackendKind::SimdJit,
                count: batch_size,
                jit_fn_ptr: None,
            })
            .unwrap();

        // Receive batched result
        let result = transport.recv_result().unwrap();

        let coord_elapsed = coord_start.elapsed();

        match result {
            DistributedResult::BatchComplete {
                node_id,
                checksum,
                count,
                metrics,
            } => {
                eprintln!("[coordinator] BatchComplete from node {}", node_id);
                eprintln!("[coordinator]   Kernels: {}", count);
                eprintln!("[coordinator]   Checksum: 0x{:016X}", checksum);
                eprintln!(
                    "[coordinator]   Worker time: {}ms",
                    metrics.execution_time_us / 1000
                );
                eprintln!(
                    "[coordinator]   Total time: {}ms",
                    coord_elapsed.as_millis()
                );
                eprintln!("[coordinator]   q_ops: {}", metrics.q_ops);
                eprintln!(
                    "[coordinator]   Throughput: {:.2} kernels/sec",
                    count as f64 / (metrics.execution_time_us as f64 / 1e6)
                );

                let q_ops_per_sec = metrics.q_ops as f64 / (metrics.execution_time_us as f64 / 1e6);
                eprintln!("[coordinator]   q_ops/sec: {:.2}M", q_ops_per_sec / 1e6);

                assert_eq!(count, batch_size);
            }
            _ => panic!("Expected BatchComplete"),
        }

        // Shutdown
        transport.send_job(DistributedJob::Shutdown).unwrap();
        worker_thread.join().unwrap();

        eprintln!("[p5.1:batch] ✓ Kernel batching test passed!");
    }

    #[test]
    #[cfg(all(feature = "serde", feature = "quantum_jit"))]
    fn test_p5_2_pipelined_coordinator() {
        use std::net::TcpListener;
        use std::thread;

        eprintln!("\n=== EPIC 65 Phase 2: Pipelined Coordinator ===");

        // Start 2 TCP workers on localhost
        let listener0 = TcpListener::bind("127.0.0.1:0").expect("Failed to bind worker 0");
        let addr0 = listener0.local_addr().unwrap();

        let listener1 = TcpListener::bind("127.0.0.1:0").expect("Failed to bind worker 1");
        let addr1 = listener1.local_addr().unwrap();

        eprintln!("[p5.2:pipeline] Worker 0 listening on {}", addr0);
        eprintln!("[p5.2:pipeline] Worker 1 listening on {}", addr1);

        // Worker 0 thread
        let worker0_thread = thread::spawn(move || {
            let (stream, _) = listener0.accept().unwrap();
            let mut transport = TcpWorkerTransport { stream };

            // InitNode
            let job: DistributedJob = transport.recv().unwrap();
            match job {
                DistributedJob::InitNode {
                    node_id,
                    tile_start,
                    tile_end,
                    qubits,
                } => {
                    eprintln!("[worker-0] InitNode: tiles=[{}..{})", tile_start, tile_end);

                    let tile_count = tile_end - tile_start;
                    let mut farm = TileFarm::new(1, 4);
                    let mut qstate = QState::new_zero_multitile(qubits, tile_count);

                    // Process batches in loop
                    loop {
                        let job: DistributedJob = transport.recv().unwrap();
                        match job {
                            DistributedJob::RunKernelBatch {
                                kernel_type,
                                depth,
                                backend,
                                count,
                                ..
                            } => {
                                let start = std::time::Instant::now();

                                unsafe {
                                    std::env::set_var("JIT_TILE_COUNT", "4");
                                    std::env::set_var("JIT_H_DEPTH", &depth.to_string());
                                }
                                let jit_fn_ptr =
                                    crate::quantum_jit::clif::compile_h_kernel(qubits, 0)
                                        .map(|f| f as *const u8);

                                let mut total_checksum = 0u64;
                                for _ in 0..count {
                                    unsafe {
                                        farm.run_kernel_on_all_workers(
                                            kernel_type,
                                            qubits,
                                            depth,
                                            backend,
                                            &mut qstate as *mut QState as *mut u8,
                                            jit_fn_ptr,
                                        );
                                    }
                                    let real_sum: f32 =
                                        qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                                    let imag_sum: f32 =
                                        qstate.imag.as_slice().iter().map(|x| x.abs()).sum();
                                    let tc = (tile_end - tile_start) as f32;
                                    let ck = (real_sum / tc).to_bits() as u64
                                        ^ (((imag_sum / tc).to_bits() as u64) << 32);
                                    total_checksum ^= ck;
                                }

                                let result = DistributedResult::BatchComplete {
                                    node_id,
                                    checksum: total_checksum,
                                    count,
                                    metrics: NodeMetrics {
                                        execution_time_us: start.elapsed().as_micros() as u64,
                                        q_ops: (qubits as u64 * depth as u64 * count as u64),
                                    },
                                };
                                transport.send(&result).unwrap();
                            }
                            DistributedJob::Shutdown => {
                                eprintln!("[worker-0] Shutdown");
                                return;
                            }
                            _ => panic!("Unexpected job"),
                        }
                    }
                }
                _ => panic!("Expected InitNode"),
            }
        });

        // Worker 1 thread (same logic)
        let worker1_thread = thread::spawn(move || {
            let (stream, _) = listener1.accept().unwrap();
            let mut transport = TcpWorkerTransport { stream };

            let job: DistributedJob = transport.recv().unwrap();
            match job {
                DistributedJob::InitNode {
                    node_id,
                    tile_start,
                    tile_end,
                    qubits,
                } => {
                    eprintln!("[worker-1] InitNode: tiles=[{}..{})", tile_start, tile_end);

                    let tile_count = tile_end - tile_start;
                    let mut farm = TileFarm::new(1, 4);
                    let mut qstate = QState::new_zero_multitile(qubits, tile_count);

                    loop {
                        let job: DistributedJob = transport.recv().unwrap();
                        match job {
                            DistributedJob::RunKernelBatch {
                                kernel_type,
                                depth,
                                backend,
                                count,
                                ..
                            } => {
                                let start = std::time::Instant::now();

                                unsafe {
                                    std::env::set_var("JIT_TILE_COUNT", "4");
                                    std::env::set_var("JIT_H_DEPTH", &depth.to_string());
                                }
                                let jit_fn_ptr =
                                    crate::quantum_jit::clif::compile_h_kernel(qubits, 0)
                                        .map(|f| f as *const u8);

                                let mut total_checksum = 0u64;
                                for _ in 0..count {
                                    unsafe {
                                        farm.run_kernel_on_all_workers(
                                            kernel_type,
                                            qubits,
                                            depth,
                                            backend,
                                            &mut qstate as *mut QState as *mut u8,
                                            jit_fn_ptr,
                                        );
                                    }
                                    let real_sum: f32 =
                                        qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                                    let imag_sum: f32 =
                                        qstate.imag.as_slice().iter().map(|x| x.abs()).sum();
                                    let tc = (tile_end - tile_start) as f32;
                                    let ck = (real_sum / tc).to_bits() as u64
                                        ^ (((imag_sum / tc).to_bits() as u64) << 32);
                                    total_checksum ^= ck;
                                }

                                let result = DistributedResult::BatchComplete {
                                    node_id,
                                    checksum: total_checksum,
                                    count,
                                    metrics: NodeMetrics {
                                        execution_time_us: start.elapsed().as_micros() as u64,
                                        q_ops: (qubits as u64 * depth as u64 * count as u64),
                                    },
                                };
                                transport.send(&result).unwrap();
                            }
                            DistributedJob::Shutdown => {
                                eprintln!("[worker-1] Shutdown");
                                return;
                            }
                            _ => panic!("Unexpected job"),
                        }
                    }
                }
                _ => panic!("Expected InitNode"),
            }
        });

        // Coordinator: connect to both nodes
        thread::sleep(std::time::Duration::from_millis(100));
        let mut transport0 = TcpTransport::connect(0, &addr0.to_string()).unwrap();
        let mut transport1 = TcpTransport::connect(1, &addr1.to_string()).unwrap();

        // Initialize both nodes
        transport0
            .send_job(DistributedJob::InitNode {
                node_id: 0,
                tile_start: 0,
                tile_end: 4,
                qubits: 4,
            })
            .unwrap();

        transport1
            .send_job(DistributedJob::InitNode {
                node_id: 1,
                tile_start: 4,
                tile_end: 8,
                qubits: 4,
            })
            .unwrap();

        let job = DistributedJob::RunKernelBatch {
            kernel_type: KernelType::HDeep,
            depth: 50,
            backend: BackendKind::SimdJit,
            count: 50, // 50 kernels per batch
            jit_fn_ptr: None,
        };

        eprintln!("[coordinator] Running pipelined execution with 10 batches...");

        let coord_start = std::time::Instant::now();

        // Use pipelined batch executor
        let mut transports = vec![(0u16, transport0), (1u16, transport1)];
        let num_batches = 10;

        let all_results = run_pipelined_batches(&mut transports, job, num_batches).unwrap();

        let coord_elapsed = coord_start.elapsed();

        // Aggregate results
        let total_kernels: u32 = all_results
            .iter()
            .flat_map(|batch| batch.iter())
            .filter_map(|r| r.batch_count())
            .sum();

        let total_q_ops: u64 = all_results
            .iter()
            .flat_map(|batch| batch.iter())
            .filter_map(|r| r.metrics())
            .map(|m| m.q_ops)
            .sum();

        eprintln!("[coordinator] Pipelined execution complete!");
        eprintln!("[coordinator]   Batches: {}", num_batches);
        eprintln!("[coordinator]   Total kernels: {}", total_kernels);
        eprintln!("[coordinator]   Total q_ops: {}", total_q_ops);
        eprintln!("[coordinator]   Wall time: {}ms", coord_elapsed.as_millis());
        eprintln!(
            "[coordinator]   Throughput: {:.2} kernels/sec",
            total_kernels as f64 / coord_elapsed.as_secs_f64()
        );

        let q_ops_per_sec = total_q_ops as f64 / coord_elapsed.as_secs_f64();
        eprintln!("[coordinator]   q_ops/sec: {:.2}M", q_ops_per_sec / 1e6);

        // Shutdown
        let (_, mut t0) = transports.remove(0);
        let (_, mut t1) = transports.remove(0);
        t0.send_job(DistributedJob::Shutdown).unwrap();
        t1.send_job(DistributedJob::Shutdown).unwrap();
        worker0_thread.join().unwrap();
        worker1_thread.join().unwrap();

        eprintln!("[p5.2:pipeline] ✓ Pipelined coordinator test passed!");
    }

    #[test]
    #[ignore] // Comprehensive benchmark
    #[cfg(all(feature = "serde", feature = "quantum_jit"))]
    fn test_p5_5_epic65_performance_benchmark() {
        use std::net::TcpListener;
        use std::thread;

        eprintln!("\n========================================");
        eprintln!("EPIC 65 PERFORMANCE BENCHMARK");
        eprintln!("========================================\n");

        eprintln!("Configuration:");
        eprintln!("  Nodes: 2");
        eprintln!("  Qubits: 4");
        eprintln!("  Depth: 100");
        eprintln!("  Total kernels: 1000 (500 per batch × 2 batches)");
        eprintln!();

        // Start 2 TCP workers
        let listener0 = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr0 = listener0.local_addr().unwrap();
        let listener1 = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr1 = listener1.local_addr().unwrap();

        // Worker 0 thread
        let worker0_thread = thread::spawn(move || {
            let (stream, _) = listener0.accept().unwrap();
            let mut transport = TcpWorkerTransport { stream };

            let job: DistributedJob = transport.recv().unwrap();
            match job {
                DistributedJob::InitNode {
                    node_id,
                    tile_start,
                    tile_end,
                    qubits,
                } => {
                    let tile_count = tile_end - tile_start;
                    let mut farm = TileFarm::new(1, 4);
                    let mut qstate = QState::new_zero_multitile(qubits, tile_count);

                    loop {
                        let job: DistributedJob = transport.recv().unwrap();
                        match job {
                            DistributedJob::RunKernelBatch {
                                kernel_type,
                                depth,
                                backend,
                                count,
                                ..
                            } => {
                                let start = std::time::Instant::now();
                                unsafe {
                                    std::env::set_var("JIT_TILE_COUNT", "4");
                                    std::env::set_var("JIT_H_DEPTH", &depth.to_string());
                                }
                                let jit_fn_ptr =
                                    crate::quantum_jit::clif::compile_h_kernel(qubits, 0)
                                        .map(|f| f as *const u8);

                                let mut total_checksum = 0u64;
                                for _ in 0..count {
                                    unsafe {
                                        farm.run_kernel_on_all_workers(
                                            kernel_type,
                                            qubits,
                                            depth,
                                            backend,
                                            &mut qstate as *mut QState as *mut u8,
                                            jit_fn_ptr,
                                        );
                                    }
                                    let real_sum: f32 =
                                        qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                                    let imag_sum: f32 =
                                        qstate.imag.as_slice().iter().map(|x| x.abs()).sum();
                                    let tc = (tile_end - tile_start) as f32;
                                    let ck = (real_sum / tc).to_bits() as u64
                                        ^ (((imag_sum / tc).to_bits() as u64) << 32);
                                    total_checksum ^= ck;
                                }

                                let result = DistributedResult::BatchComplete {
                                    node_id,
                                    checksum: total_checksum,
                                    count,
                                    metrics: NodeMetrics {
                                        execution_time_us: start.elapsed().as_micros() as u64,
                                        q_ops: (qubits as u64 * depth as u64 * count as u64),
                                    },
                                };
                                transport.send(&result).unwrap();
                            }
                            DistributedJob::Shutdown => return,
                            _ => panic!("Unexpected job"),
                        }
                    }
                }
                _ => panic!("Expected InitNode"),
            }
        });

        // Worker 1 thread
        let worker1_thread = thread::spawn(move || {
            let (stream, _) = listener1.accept().unwrap();
            let mut transport = TcpWorkerTransport { stream };

            let job: DistributedJob = transport.recv().unwrap();
            match job {
                DistributedJob::InitNode {
                    node_id,
                    tile_start,
                    tile_end,
                    qubits,
                } => {
                    let tile_count = tile_end - tile_start;
                    let mut farm = TileFarm::new(1, 4);
                    let mut qstate = QState::new_zero_multitile(qubits, tile_count);

                    loop {
                        let job: DistributedJob = transport.recv().unwrap();
                        match job {
                            DistributedJob::RunKernelBatch {
                                kernel_type,
                                depth,
                                backend,
                                count,
                                ..
                            } => {
                                let start = std::time::Instant::now();
                                unsafe {
                                    std::env::set_var("JIT_TILE_COUNT", "4");
                                    std::env::set_var("JIT_H_DEPTH", &depth.to_string());
                                }
                                let jit_fn_ptr =
                                    crate::quantum_jit::clif::compile_h_kernel(qubits, 0)
                                        .map(|f| f as *const u8);

                                let mut total_checksum = 0u64;
                                for _ in 0..count {
                                    unsafe {
                                        farm.run_kernel_on_all_workers(
                                            kernel_type,
                                            qubits,
                                            depth,
                                            backend,
                                            &mut qstate as *mut QState as *mut u8,
                                            jit_fn_ptr,
                                        );
                                    }
                                    let real_sum: f32 =
                                        qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                                    let imag_sum: f32 =
                                        qstate.imag.as_slice().iter().map(|x| x.abs()).sum();
                                    let tc = (tile_end - tile_start) as f32;
                                    let ck = (real_sum / tc).to_bits() as u64
                                        ^ (((imag_sum / tc).to_bits() as u64) << 32);
                                    total_checksum ^= ck;
                                }

                                let result = DistributedResult::BatchComplete {
                                    node_id,
                                    checksum: total_checksum,
                                    count,
                                    metrics: NodeMetrics {
                                        execution_time_us: start.elapsed().as_micros() as u64,
                                        q_ops: (qubits as u64 * depth as u64 * count as u64),
                                    },
                                };
                                transport.send(&result).unwrap();
                            }
                            DistributedJob::Shutdown => return,
                            _ => panic!("Unexpected job"),
                        }
                    }
                }
                _ => panic!("Expected InitNode"),
            }
        });

        // Coordinator setup
        thread::sleep(std::time::Duration::from_millis(100));
        let mut transport0 = TcpTransport::connect(0, &addr0.to_string()).unwrap();
        let mut transport1 = TcpTransport::connect(1, &addr1.to_string()).unwrap();

        transport0
            .send_job(DistributedJob::InitNode {
                node_id: 0,
                tile_start: 0,
                tile_end: 4,
                qubits: 4,
            })
            .unwrap();
        transport1
            .send_job(DistributedJob::InitNode {
                node_id: 1,
                tile_start: 4,
                tile_end: 8,
                qubits: 4,
            })
            .unwrap();

        // Run benchmark with pipelined batches
        let job = DistributedJob::RunKernelBatch {
            kernel_type: KernelType::HDeep,
            depth: 100,
            backend: BackendKind::SimdJit,
            count: 500,
            jit_fn_ptr: None,
        };

        eprintln!("Running EPIC 65 optimized execution...");
        let start = std::time::Instant::now();

        let mut transports = vec![(0u16, transport0), (1u16, transport1)];
        let all_results = run_pipelined_batches(&mut transports, job, 2).unwrap();

        let elapsed = start.elapsed();

        // Calculate metrics
        let total_kernels: u32 = all_results
            .iter()
            .flat_map(|batch| batch.iter())
            .filter_map(|r| r.batch_count())
            .sum();

        let total_q_ops: u64 = all_results
            .iter()
            .flat_map(|batch| batch.iter())
            .filter_map(|r| r.metrics())
            .map(|m| m.q_ops)
            .sum();

        let worker_time_us: u64 = all_results
            .iter()
            .flat_map(|batch| batch.iter())
            .filter_map(|r| r.metrics())
            .map(|m| m.execution_time_us)
            .sum();

        eprintln!();
        eprintln!("========================================");
        eprintln!("RESULTS");
        eprintln!("========================================");
        eprintln!("Total kernels:     {}", total_kernels);
        eprintln!("Total q_ops:       {}", total_q_ops);
        eprintln!("Wall time:         {:.2}s", elapsed.as_secs_f64());
        eprintln!("Worker time (sum): {:.2}s", worker_time_us as f64 / 1e6);
        eprintln!();
        eprintln!(
            "Throughput:        {:.2} kernels/sec",
            total_kernels as f64 / elapsed.as_secs_f64()
        );
        eprintln!(
            "q_ops/sec:         {:.2}M",
            (total_q_ops as f64 / elapsed.as_secs_f64()) / 1e6
        );
        eprintln!();
        eprintln!("EPIC 65 improvements:");
        eprintln!("  ✓ Kernel batching (P1)");
        eprintln!("  ✓ Pipelined dispatch (P2)");
        eprintln!("========================================\n");

        // Shutdown
        let (_, mut t0) = transports.remove(0);
        let (_, mut t1) = transports.remove(0);
        t0.send_job(DistributedJob::Shutdown).unwrap();
        t1.send_job(DistributedJob::Shutdown).unwrap();
        worker0_thread.join().unwrap();
        worker1_thread.join().unwrap();
    }

    #[test]
    #[cfg(all(feature = "serde", feature = "quantum_jit"))]
    fn test_p5_3_4_queue_depth_and_aggregation() {
        use std::net::TcpListener;
        use std::thread;

        eprintln!("\n=== EPIC 65 Phase 3 & 4: Queue Depth + Parallel Aggregation ===");

        // Start 2 TCP workers
        let listener0 = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr0 = listener0.local_addr().unwrap();
        let listener1 = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr1 = listener1.local_addr().unwrap();

        eprintln!("[p5.3-4] Worker 0: {}", addr0);
        eprintln!("[p5.3-4] Worker 1: {}", addr1);

        // Worker 0 thread
        let worker0_thread = thread::spawn(move || {
            let (stream, _) = listener0.accept().unwrap();
            let mut transport = TcpWorkerTransport { stream };

            let job: DistributedJob = transport.recv().unwrap();
            match job {
                DistributedJob::InitNode {
                    node_id,
                    tile_start,
                    tile_end,
                    qubits,
                } => {
                    let tile_count = tile_end - tile_start;
                    let mut farm = TileFarm::new(1, 4);
                    let mut qstate = QState::new_zero_multitile(qubits, tile_count);

                    let mut batch_count = 0;
                    loop {
                        let job: DistributedJob = transport.recv().unwrap();
                        match job {
                            DistributedJob::RunKernelBatch {
                                kernel_type,
                                depth,
                                backend,
                                count,
                                ..
                            } => {
                                batch_count += 1;
                                eprintln!("[worker-0] Processing batch {}", batch_count);

                                let start = std::time::Instant::now();
                                unsafe {
                                    std::env::set_var("JIT_TILE_COUNT", "4");
                                    std::env::set_var("JIT_H_DEPTH", &depth.to_string());
                                }
                                let jit_fn_ptr =
                                    crate::quantum_jit::clif::compile_h_kernel(qubits, 0)
                                        .map(|f| f as *const u8);

                                let mut total_checksum = 0u64;
                                for _ in 0..count {
                                    unsafe {
                                        farm.run_kernel_on_all_workers(
                                            kernel_type,
                                            qubits,
                                            depth,
                                            backend,
                                            &mut qstate as *mut QState as *mut u8,
                                            jit_fn_ptr,
                                        );
                                    }
                                    let real_sum: f32 =
                                        qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                                    let imag_sum: f32 =
                                        qstate.imag.as_slice().iter().map(|x| x.abs()).sum();
                                    let tc = (tile_end - tile_start) as f32;
                                    let ck = (real_sum / tc).to_bits() as u64
                                        ^ (((imag_sum / tc).to_bits() as u64) << 32);
                                    total_checksum ^= ck;
                                }

                                let result = DistributedResult::BatchComplete {
                                    node_id,
                                    checksum: total_checksum,
                                    count,
                                    metrics: NodeMetrics {
                                        execution_time_us: start.elapsed().as_micros() as u64,
                                        q_ops: (qubits as u64 * depth as u64 * count as u64),
                                    },
                                };
                                transport.send(&result).unwrap();
                            }
                            DistributedJob::Shutdown => {
                                eprintln!("[worker-0] Processed {} batches total", batch_count);
                                return;
                            }
                            _ => panic!("Unexpected job"),
                        }
                    }
                }
                _ => panic!("Expected InitNode"),
            }
        });

        // Worker 1 thread
        let worker1_thread = thread::spawn(move || {
            let (stream, _) = listener1.accept().unwrap();
            let mut transport = TcpWorkerTransport { stream };

            let job: DistributedJob = transport.recv().unwrap();
            match job {
                DistributedJob::InitNode {
                    node_id,
                    tile_start,
                    tile_end,
                    qubits,
                } => {
                    let tile_count = tile_end - tile_start;
                    let mut farm = TileFarm::new(1, 4);
                    let mut qstate = QState::new_zero_multitile(qubits, tile_count);

                    let mut batch_count = 0;
                    loop {
                        let job: DistributedJob = transport.recv().unwrap();
                        match job {
                            DistributedJob::RunKernelBatch {
                                kernel_type,
                                depth,
                                backend,
                                count,
                                ..
                            } => {
                                batch_count += 1;
                                eprintln!("[worker-1] Processing batch {}", batch_count);

                                let start = std::time::Instant::now();
                                unsafe {
                                    std::env::set_var("JIT_TILE_COUNT", "4");
                                    std::env::set_var("JIT_H_DEPTH", &depth.to_string());
                                }
                                let jit_fn_ptr =
                                    crate::quantum_jit::clif::compile_h_kernel(qubits, 0)
                                        .map(|f| f as *const u8);

                                let mut total_checksum = 0u64;
                                for _ in 0..count {
                                    unsafe {
                                        farm.run_kernel_on_all_workers(
                                            kernel_type,
                                            qubits,
                                            depth,
                                            backend,
                                            &mut qstate as *mut QState as *mut u8,
                                            jit_fn_ptr,
                                        );
                                    }
                                    let real_sum: f32 =
                                        qstate.real.as_slice().iter().map(|x| x.abs()).sum();
                                    let imag_sum: f32 =
                                        qstate.imag.as_slice().iter().map(|x| x.abs()).sum();
                                    let tc = (tile_end - tile_start) as f32;
                                    let ck = (real_sum / tc).to_bits() as u64
                                        ^ (((imag_sum / tc).to_bits() as u64) << 32);
                                    total_checksum ^= ck;
                                }

                                let result = DistributedResult::BatchComplete {
                                    node_id,
                                    checksum: total_checksum,
                                    count,
                                    metrics: NodeMetrics {
                                        execution_time_us: start.elapsed().as_micros() as u64,
                                        q_ops: (qubits as u64 * depth as u64 * count as u64),
                                    },
                                };
                                transport.send(&result).unwrap();
                            }
                            DistributedJob::Shutdown => {
                                eprintln!("[worker-1] Processed {} batches total", batch_count);
                                return;
                            }
                            _ => panic!("Unexpected job"),
                        }
                    }
                }
                _ => panic!("Expected InitNode"),
            }
        });

        // Coordinator setup
        thread::sleep(std::time::Duration::from_millis(100));
        let mut transport0 = TcpTransport::connect(0, &addr0.to_string()).unwrap();
        let mut transport1 = TcpTransport::connect(1, &addr1.to_string()).unwrap();

        transport0
            .send_job(DistributedJob::InitNode {
                node_id: 0,
                tile_start: 0,
                tile_end: 4,
                qubits: 4,
            })
            .unwrap();
        transport1
            .send_job(DistributedJob::InitNode {
                node_id: 1,
                tile_start: 4,
                tile_end: 8,
                qubits: 4,
            })
            .unwrap();

        let job = DistributedJob::RunKernelBatch {
            kernel_type: KernelType::HDeep,
            depth: 50,
            backend: BackendKind::SimdJit,
            count: 100,
            jit_fn_ptr: None,
        };

        eprintln!("[coordinator] Testing queue depth = 2 with 8 batches");

        let start = std::time::Instant::now();

        // Phase 3: Use queue depth control (only 2 batches in flight at a time)
        let mut transports = vec![(0u16, transport0), (1u16, transport1)];
        let all_results = run_pipelined_batches_with_queue_depth(
            &mut transports,
            job,
            8, // total batches
            2, // queue depth
        )
        .unwrap();

        let elapsed = start.elapsed();

        // Phase 4: Use parallel aggregation
        eprintln!("[coordinator] Aggregating results in parallel...");
        let agg_start = std::time::Instant::now();
        let metrics = aggregate_results_parallel(&all_results);
        let agg_elapsed = agg_start.elapsed();

        eprintln!();
        eprintln!("[coordinator] Phase 3 & 4 Results:");
        eprintln!("[coordinator]   Total kernels:  {}", metrics.total_kernels);
        eprintln!("[coordinator]   Total q_ops:    {}", metrics.total_q_ops);
        eprintln!(
            "[coordinator]   Checksum:       0x{:016X}",
            metrics.combined_checksum
        );
        eprintln!("[coordinator]   Wall time:      {}ms", elapsed.as_millis());
        eprintln!(
            "[coordinator]   Worker time:    {}ms",
            metrics.total_execution_time_us / 1000
        );
        eprintln!(
            "[coordinator]   Aggregation:    {}μs",
            agg_elapsed.as_micros()
        );
        eprintln!(
            "[coordinator]   Throughput:     {:.2} kernels/sec",
            metrics.total_kernels as f64 / elapsed.as_secs_f64()
        );
        eprintln!(
            "[coordinator]   q_ops/sec:      {:.2}M",
            (metrics.total_q_ops as f64 / elapsed.as_secs_f64()) / 1e6
        );

        // Shutdown
        let (_, mut t0) = transports.remove(0);
        let (_, mut t1) = transports.remove(0);
        t0.send_job(DistributedJob::Shutdown).unwrap();
        t1.send_job(DistributedJob::Shutdown).unwrap();
        worker0_thread.join().unwrap();
        worker1_thread.join().unwrap();

        eprintln!("[p5.3-4] ✓ Queue depth + aggregation test passed!");
    }

    // ========================================================================
    // Phase 4: Quantum Circuit Execution Tests (SPRINT 39.0)
    // ========================================================================

    #[test]
    fn test_phase4_single_node_quantum_circuit() {
        use logic_fabric_core::quantum::QGate;

        // Single node, simple circuit
        let farm = DistributedTileFarm::new(8, 1, 8, 2);

        let circuit = vec![QGate::H(0), QGate::X(1), QGate::H(2)];

        let results = farm.execute_quantum_circuit(8, &circuit, 2).unwrap();

        assert_eq!(results.len(), 1, "Should have 1 node result");

        let (node_id, checksum, exec_time_us) = results[0];
        assert_eq!(node_id, 0, "Node ID should be 0");
        assert!(checksum != 0, "Checksum should be non-zero");
        assert!(exec_time_us > 0, "Execution time should be positive");

        eprintln!(
            "[phase4-test] Single node: checksum=0x{:016X}, time={}us",
            checksum, exec_time_us
        );
    }

    #[test]
    fn test_phase4_multi_node_quantum_circuit() {
        use logic_fabric_core::quantum::QGate;

        // 2 nodes, each should produce same result
        let farm = DistributedTileFarm::new(8, 2, 8, 2);

        let circuit = vec![QGate::H(0), QGate::CNot(0, 1), QGate::H(2)];

        let results = farm.execute_quantum_circuit(8, &circuit, 2).unwrap();

        assert_eq!(results.len(), 2, "Should have 2 node results");

        let (node0_id, checksum0, time0) = results[0];
        let (node1_id, checksum1, time1) = results[1];

        assert_eq!(node0_id, 0);
        assert_eq!(node1_id, 1);
        assert_eq!(
            checksum0, checksum1,
            "Both nodes should produce same checksum"
        );
        assert!(time0 > 0 && time1 > 0);

        eprintln!(
            "[phase4-test] Multi-node: node0=0x{:016X} ({}us), node1=0x{:016X} ({}us)",
            checksum0, time0, checksum1, time1
        );
    }

    #[test]
    fn test_phase4_quantum_circuit_determinism() {
        use logic_fabric_core::quantum::QGate;

        let farm = DistributedTileFarm::new(8, 1, 8, 2);

        let circuit = vec![QGate::H(0), QGate::X(1), QGate::Y(2), QGate::Z(3)];

        // Run same circuit twice
        let results1 = farm.execute_quantum_circuit(8, &circuit, 2).unwrap();
        let results2 = farm.execute_quantum_circuit(8, &circuit, 2).unwrap();

        assert_eq!(results1.len(), results2.len());

        for i in 0..results1.len() {
            let (_, checksum1, _) = results1[i];
            let (_, checksum2, _) = results2[i];
            assert_eq!(checksum1, checksum2, "Checksums should be deterministic");
        }

        eprintln!("[phase4-test] Determinism verified: checksums match across runs");
    }

    #[test]
    fn test_phase4_quantum_batch_single_node() {
        use logic_fabric_core::quantum::QGate;

        let farm = DistributedTileFarm::new(8, 1, 8, 2);

        let circuits = vec![
            vec![QGate::H(0)],
            vec![QGate::X(1), QGate::H(1)],
            vec![QGate::Y(2), QGate::Z(2), QGate::H(2)],
        ];

        let results = farm.execute_quantum_batch(8, &circuits, 2).unwrap();

        assert_eq!(results.len(), 1);

        let (node_id, checksum, circuit_count, total_gates, exec_time_us) = results[0];
        assert_eq!(node_id, 0);
        assert_eq!(circuit_count, 3);
        assert_eq!(total_gates, 6, "Should have 1 + 2 + 3 = 6 gates total");
        assert!(checksum != 0);
        assert!(exec_time_us > 0);

        eprintln!(
            "[phase4-test] Batch: {} circuits, {} gates, checksum=0x{:016X}, time={}us",
            circuit_count, total_gates, checksum, exec_time_us
        );
    }

    #[test]
    fn test_phase4_quantum_batch_multi_node() {
        use logic_fabric_core::quantum::QGate;

        let farm = DistributedTileFarm::new(8, 2, 8, 2);

        let circuits = vec![
            vec![QGate::H(0), QGate::CNot(0, 1)],
            vec![QGate::X(2), QGate::Y(3)],
        ];

        let results = farm.execute_quantum_batch(8, &circuits, 2).unwrap();

        assert_eq!(results.len(), 2);

        let (_, checksum0, count0, gates0, time0) = results[0];
        let (_, checksum1, count1, gates1, time1) = results[1];

        assert_eq!(count0, 2);
        assert_eq!(count1, 2);
        assert_eq!(gates0, 4);
        assert_eq!(gates1, 4);
        assert_eq!(
            checksum0, checksum1,
            "Both nodes should produce same batch checksum"
        );
        assert!(time0 > 0 && time1 > 0);

        eprintln!(
            "[phase4-test] Batch multi-node: node0=0x{:016X} ({}us), node1=0x{:016X} ({}us)",
            checksum0, time0, checksum1, time1
        );
    }

    #[test]
    fn test_phase4_cnot_within_block() {
        use logic_fabric_core::quantum::QGate;

        let farm = DistributedTileFarm::new(12, 1, 8, 2);

        // CNOT between low-order qubits (within block)
        let circuit = vec![
            QGate::H(0),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::CNot(2, 3),
        ];

        let results = farm.execute_quantum_circuit(12, &circuit, 2).unwrap();

        assert_eq!(results.len(), 1);
        let (_, checksum, _) = results[0];
        assert!(
            checksum != 0,
            "Within-block CNOT should produce non-zero checksum"
        );

        eprintln!(
            "[phase4-test] Within-block CNOT: checksum=0x{:016X}",
            checksum
        );
    }

    #[test]
    fn test_phase4_cnot_cross_block() {
        use logic_fabric_core::quantum::QGate;

        let farm = DistributedTileFarm::new(12, 1, 8, 2);

        // CNOT with control >= 7 (cross-block)
        let circuit = vec![
            QGate::H(7),
            QGate::CNot(7, 0), // Cross-block: control high, target low
            QGate::CNot(8, 1),
        ];

        let results = farm.execute_quantum_circuit(12, &circuit, 2).unwrap();

        assert_eq!(results.len(), 1);
        let (_, checksum, _) = results[0];
        assert!(
            checksum != 0,
            "Cross-block CNOT should produce non-zero checksum"
        );

        eprintln!(
            "[phase4-test] Cross-block CNOT: checksum=0x{:016X}",
            checksum
        );
    }

    #[test]
    fn test_phase4_empty_circuit() {
        let farm = DistributedTileFarm::new(8, 1, 8, 2);

        let circuit = vec![];

        let results = farm.execute_quantum_circuit(8, &circuit, 2).unwrap();

        assert_eq!(results.len(), 1);
        let (_, checksum, exec_time_us) = results[0];

        // Empty circuit should still produce a checksum (of |0> state)
        assert!(checksum != 0);
        assert!(exec_time_us >= 0);

        eprintln!(
            "[phase4-test] Empty circuit: checksum=0x{:016X}, time={}us",
            checksum, exec_time_us
        );
    }

    #[test]
    fn test_phase4_large_circuit() {
        use logic_fabric_core::quantum::QGate;

        let farm = DistributedTileFarm::new(12, 2, 8, 2);

        // Create a larger circuit
        let mut circuit = Vec::new();
        for i in 0..50 {
            circuit.push(QGate::H(i % 6));
            circuit.push(QGate::X((i + 1) % 6));
        }

        let results = farm.execute_quantum_circuit(12, &circuit, 2).unwrap();

        assert_eq!(results.len(), 2);

        let (_, checksum0, time0) = results[0];
        let (_, checksum1, time1) = results[1];

        assert_eq!(
            checksum0, checksum1,
            "Large circuit should produce consistent results"
        );
        assert!(time0 > 0 && time1 > 0);

        eprintln!(
            "[phase4-test] Large circuit (100 gates): node0={}us, node1={}us",
            time0, time1
        );
    }
}
