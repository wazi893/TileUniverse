//! Sprint 295/296: V2FastCpuPool — persistent worker pool for parallel CPU execution.
//!
//! Runs N V2FastCpu instances on N persistent worker threads. Workers stay
//! alive across dispatches — no thread spawn/join overhead per call. Uses
//! mpsc channels for job dispatch and std::sync::Barrier for synchronization.
//!
//! **Phase safety**: Dispatch methods (`run_parallel`, `run_batch_parallel`)
//! take `&mut self`, preventing callers from holding a `CpuSnapshot` or
//! any other borrow across a dispatch boundary.

use crate::tile_cpu::v2_fast::V2FastCpu;
use crate::tile_cpu::v2_iss::V2IssState;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Barrier, Mutex};
use std::thread::JoinHandle;

// ---------------------------------------------------------------------------
// Internal message type
// ---------------------------------------------------------------------------

enum PoolMessage {
    /// Run tick loop for up to max_cycles.
    RunTick(u64),
    /// Run batch (ISS loop / JIT) for up to max_cycles.
    RunBatch(u64),
    /// Shut down worker thread.
    Shutdown,
}

// ---------------------------------------------------------------------------
// Worker handle
// ---------------------------------------------------------------------------

struct PoolWorker {
    thread: Option<JoinHandle<()>>,
    sender: Sender<PoolMessage>,
}

// ---------------------------------------------------------------------------
// Worker thread loop
// ---------------------------------------------------------------------------

fn pool_worker_loop(
    cpu: Arc<Mutex<V2FastCpu>>,
    receiver: Receiver<PoolMessage>,
    barrier: Arc<Barrier>,
) {
    loop {
        match receiver.recv() {
            Ok(PoolMessage::RunTick(max_cycles)) => {
                {
                    let mut cpu = cpu.lock().unwrap();
                    for _ in 0..max_cycles {
                        if cpu.is_halted() {
                            break;
                        }
                        cpu.tick();
                    }
                } // lock released
                barrier.wait();
            }
            Ok(PoolMessage::RunBatch(max_cycles)) => {
                {
                    let mut cpu = cpu.lock().unwrap();
                    cpu.run_batch(max_cycles);
                } // lock released
                barrier.wait();
            }
            Ok(PoolMessage::Shutdown) | Err(_) => break,
        }
    }
}

// ---------------------------------------------------------------------------
// CPU snapshot — owned copy, cannot deadlock dispatch
// ---------------------------------------------------------------------------

/// Owned snapshot of a V2FastCpu's observable state.
/// Returned by `cpu_snapshot()` — cannot be held across dispatch because
/// dispatch takes `&mut self`.
#[derive(Debug, Clone)]
pub struct CpuSnapshot {
    pub pc: u32,
    pub lr: u32,
    pub flag_z: bool,
    pub flag_c: bool,
    pub halted: bool,
    pub cycle_count: u64,
    pub retired_count: u64,
    pub regs: [u64; 16],
    pub ram: [u64; 128],
}

impl CpuSnapshot {
    fn from_cpu(cpu: &V2FastCpu) -> Self {
        let s = cpu.state();
        Self {
            pc: s.pc,
            lr: s.lr,
            flag_z: s.flag_z,
            flag_c: s.flag_c,
            halted: cpu.is_halted(),
            cycle_count: cpu.cycle_count(),
            retired_count: cpu.retired_count(),
            regs: s.regs,
            ram: s.ram,
        }
    }

    /// Compute the golden hash (same algorithm as hash_v2_iss_state).
    pub fn hash(&self) -> u64 {
        // Construct a temporary V2IssState for hashing.
        let iss_state = V2IssState {
            pc: self.pc,
            lr: self.lr,
            flag_z: self.flag_z,
            flag_c: self.flag_c,
            halted: self.halted,
            regs: self.regs,
            ram: self.ram,
            main_mem: Vec::new(),
            mem_addr_mask: 0x7F,
            iss_math_a: 0,
            iss_math_b: 0,
            iss_math_status: 0,
            iss_math_result: 0,
        };
        crate::tile_cpu::v2_fast::hash_v2_iss_state(&iss_state)
    }
}

// ---------------------------------------------------------------------------
// V2FastCpuPool
// ---------------------------------------------------------------------------

/// Persistent worker pool for N independent V2FastCpu instances.
///
/// **Phase contract**: Dispatch methods take `&mut self`. State read methods
/// also take `&mut self`. This prevents holding any state reference across
/// a dispatch, eliminating the deadlock class where a live MutexGuard blocks
/// a worker while the barrier blocks the main thread.
pub struct V2FastCpuPool {
    workers: Vec<PoolWorker>,
    barrier: Arc<Barrier>,
    cpus: Vec<Arc<Mutex<V2FastCpu>>>,
}

impl V2FastCpuPool {
    /// Build N independent CPUs from N separate programs.
    /// Spawns N persistent worker threads.
    pub fn new(programs: &[&[u32]]) -> Self {
        let n = programs.len();
        let barrier = Arc::new(Barrier::new(n + 1)); // +1 for main thread
        let mut workers = Vec::with_capacity(n);
        let mut cpus = Vec::with_capacity(n);

        for (i, program) in programs.iter().enumerate() {
            let cpu = Arc::new(Mutex::new(V2FastCpu::from_program(program)));
            let cpu_clone = cpu.clone();
            let barrier_clone = barrier.clone();
            let (sender, receiver) = std::sync::mpsc::channel();

            let thread = std::thread::Builder::new()
                .name(format!("v2-pool-{}", i))
                .spawn(move || {
                    pool_worker_loop(cpu_clone, receiver, barrier_clone);
                })
                .expect("failed to spawn pool worker");

            workers.push(PoolWorker {
                thread: Some(thread),
                sender,
            });
            cpus.push(cpu);
        }

        Self {
            workers,
            barrier,
            cpus,
        }
    }

    /// Build N independent CPUs all running the same program.
    pub fn uniform(program: &[u32], n: usize) -> Self {
        let programs: Vec<&[u32]> = vec![program; n];
        Self::new(&programs)
    }

    /// Dispatch: run all CPUs via tick loop for up to max_cycles.
    /// Blocks until all workers complete.
    pub fn run_parallel(&mut self, max_cycles: u64) {
        for w in &self.workers {
            w.sender.send(PoolMessage::RunTick(max_cycles)).unwrap();
        }
        self.barrier.wait();
    }

    /// Dispatch: run all CPUs via batch execution (ISS loop / JIT).
    /// Blocks until all workers complete.
    pub fn run_batch_parallel(&mut self, max_cycles: u64) {
        for w in &self.workers {
            w.sender.send(PoolMessage::RunBatch(max_cycles)).unwrap();
        }
        self.barrier.wait();
    }

    // -- State accessors (phase-safe: &mut self prevents cross-dispatch hold) --

    /// Snapshot CPU i's state. Returns an owned copy — no lock held after return.
    pub fn cpu_snapshot(&mut self, i: usize) -> CpuSnapshot {
        let cpu = self.cpus[i].lock().unwrap();
        CpuSnapshot::from_cpu(&cpu)
    }

    /// Snapshot all CPUs.
    pub fn all_snapshots(&mut self) -> Vec<CpuSnapshot> {
        (0..self.cpus.len())
            .map(|i| {
                let cpu = self.cpus[i].lock().unwrap();
                CpuSnapshot::from_cpu(&cpu)
            })
            .collect()
    }

    pub fn num_cpus(&self) -> usize {
        self.cpus.len()
    }

    pub fn all_halted(&mut self) -> bool {
        self.cpus.iter().all(|c| c.lock().unwrap().is_halted())
    }
}

impl Drop for V2FastCpuPool {
    fn drop(&mut self) {
        // Send shutdown to all workers.
        for w in &self.workers {
            let _ = w.sender.send(PoolMessage::Shutdown);
        }
        // Explicitly join each worker thread — JoinHandle::drop detaches,
        // so we must call join() to ensure threads have exited.
        for w in &mut self.workers {
            if let Some(handle) = w.thread.take() {
                let _ = handle.join();
            }
        }
    }
}

// ===========================================================================
// Sprint 297: V2FastFabric — mailbox-coupled sequential epoch execution
// ===========================================================================

/// Topology for V2FastFabric CPU arrangement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2FastFabricTopology {
    /// CPU[0]—CPU[1]—...—CPU[N-1]. N-1 links.
    Linear,
    /// CPU[0]—CPU[1]—...—CPU[N-1]—CPU[0]. N links (last wraps to first).
    Ring,
}

/// Mailbox-coupled V2FastCpu array with epoch-based synchronization.
///
/// **Epoch contract:** A value sent by CPU i in epoch N is visible to its
/// neighbor in epoch N+1, never within the same epoch.
///
/// **Mailbox semantics (matching V2LinkMailboxDevice):**
/// - addr 60 (MMIO_MAILBOX_IN): read = recv from left, write = send to left
/// - addr 61 (MMIO_MAILBOX_OUT): read = recv from right, write = send to right
///
/// Sequential execution only in this version. Parallel epochs deferred
/// until semantics are proven.
pub struct V2FastFabric {
    cpus: Vec<V2FastCpu>,
    topology: V2FastFabricTopology,
}

impl V2FastFabric {
    /// Create a fabric of N mailbox-coupled CPUs.
    pub fn new(programs: &[&[u32]], topology: V2FastFabricTopology) -> Self {
        let cpus = programs
            .iter()
            .map(|p| V2FastCpu::from_program_with_mmio(p))
            .collect();
        Self { cpus, topology }
    }

    /// Run epoch-based execution.
    ///
    /// Each epoch:
    /// 1. Inject recv: copy neighbor send values into this CPU's recv fields
    /// 2. Clear sends: reset all send fields to 0
    /// 3. Step: run each CPU for `epoch_size` ticks sequentially
    ///
    /// Repeats until `max_cycles` consumed or all CPUs halted.
    pub fn run_epochs(&mut self, max_cycles: u64, epoch_size: u64) {
        if epoch_size == 0 {
            return;
        }
        let mut cycles_used: u64 = 0;

        while cycles_used < max_cycles && !self.all_halted() {
            let ticks_this_epoch = epoch_size.min(max_cycles - cycles_used);

            // Phase 1: Inject recv from neighbors' sends
            self.sync_mailboxes();

            // No send clearing — sends persist as "latest value" until
            // overwritten by a new STB. This matches V2LinkMailboxDevice
            // semantics where Rc<Cell> retains the last written value.

            // Phase 3: Step each CPU sequentially (batch mode for tight ISS loop)
            for cpu in &mut self.cpus {
                if !cpu.is_halted() {
                    cpu.run_batch(ticks_this_epoch);
                }
            }

            cycles_used += ticks_this_epoch;
        }
    }

    /// Sync mailboxes: copy each CPU's neighbor's send values into recv fields.
    fn sync_mailboxes(&mut self) {
        let n = self.cpus.len();
        if n < 2 {
            return;
        }

        // Collect send values first (can't borrow mutably while reading neighbors).
        let sends: Vec<(u64, u64)> = self
            .cpus
            .iter()
            .map(|cpu| (cpu.mailbox_in_send(), cpu.mailbox_out_send()))
            .collect();

        for i in 0..n {
            // Left neighbor: cpu[i-1] sends right (mailbox_out_send) → cpu[i] recv left (mailbox_in_recv)
            let left_idx = if i > 0 {
                Some(i - 1)
            } else if self.topology == V2FastFabricTopology::Ring {
                Some(n - 1)
            } else {
                None
            };
            if let Some(li) = left_idx {
                self.cpus[i].set_mailbox_in_recv(sends[li].1); // left's out_send → my in_recv
            }

            // Right neighbor: cpu[i+1] sends left (mailbox_in_send) → cpu[i] recv right (mailbox_out_recv)
            let right_idx = if i + 1 < n {
                Some(i + 1)
            } else if self.topology == V2FastFabricTopology::Ring {
                Some(0)
            } else {
                None
            };
            if let Some(ri) = right_idx {
                self.cpus[i].set_mailbox_out_recv(sends[ri].0); // right's in_send → my out_recv
            }
        }
    }

    // -- State accessors --

    pub fn cpu(&self, i: usize) -> &V2FastCpu {
        &self.cpus[i]
    }

    pub fn cpu_mut(&mut self, i: usize) -> &mut V2FastCpu {
        &mut self.cpus[i]
    }

    pub fn num_cpus(&self) -> usize {
        self.cpus.len()
    }

    pub fn all_halted(&self) -> bool {
        self.cpus.iter().all(|c| c.is_halted())
    }
}

// ===========================================================================
// Sprint 300/302/303: V2ParallelFabric — parallel epoch execution
//
// Sprint 302: Worker-owned CPUs (no Arc<Mutex>), atomic mailbox buffers.
// Sprint 303: Generation-based dispatcher (replaces mpsc + Barrier),
// batch execution (run_batch), hybrid spin+park wait.
// ===========================================================================

/// Worker spin iterations before falling back to thread::park().
const FABRIC_SPIN_LIMIT: u32 = 128;

// ---------------------------------------------------------------------------
// Shared state — generation protocol + mailbox buffers
// ---------------------------------------------------------------------------

struct SharedFabricState {
    /// Per-CPU [in_send, out_send]. Worker writes after epoch, main reads during sync.
    sends: Vec<[AtomicU64; 2]>,
    /// Per-CPU [in_recv, out_recv]. Main writes during sync, worker reads before epoch.
    recvs: Vec<[AtomicU64; 2]>,
    /// Per-CPU halted flag. Worker writes after epoch, main reads in all_halted().
    halted: Vec<AtomicBool>,
    /// Per-CPU cycle count. Worker writes after epoch.
    cycle_counts: Vec<AtomicU64>,

    // -- Generation protocol (Sprint 303) --
    /// Epoch generation counter. Main increments to signal a new epoch.
    generation: AtomicU64,
    /// Epoch size for the current dispatch. Main sets before incrementing generation.
    epoch_size: AtomicU64,
    /// Monotonically increasing done counter. Workers increment after each epoch.
    /// Main waits for done_count >= generation * N.
    done_count: AtomicU64,
    /// Shutdown flag. Set by main on Drop.
    shutdown: AtomicBool,
}

impl SharedFabricState {
    fn new(n: usize) -> Self {
        let mut sends = Vec::with_capacity(n);
        let mut recvs = Vec::with_capacity(n);
        let mut halted = Vec::with_capacity(n);
        let mut cycle_counts = Vec::with_capacity(n);
        for _ in 0..n {
            sends.push([AtomicU64::new(0), AtomicU64::new(0)]);
            recvs.push([AtomicU64::new(0), AtomicU64::new(0)]);
            halted.push(AtomicBool::new(false));
            cycle_counts.push(AtomicU64::new(0));
        }
        Self {
            sends,
            recvs,
            halted,
            cycle_counts,
            generation: AtomicU64::new(0),
            epoch_size: AtomicU64::new(0),
            done_count: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        }
    }
}

// ---------------------------------------------------------------------------
// Control messages (cold path only — Snapshot, Reset)
// ---------------------------------------------------------------------------

enum FabricControl {
    /// Request a CpuSnapshot. Worker sends it back via the provided channel.
    Snapshot(Sender<CpuSnapshot>),
    /// Reset CPU with a new program (for reusable benchmarking).
    Reset(Vec<u32>),
}

struct FabricWorker {
    thread: Option<JoinHandle<()>>,
    control_tx: Sender<FabricControl>,
}

// ---------------------------------------------------------------------------
// Worker loop — generation-based wait, batch execution, done_count signal
// ---------------------------------------------------------------------------

fn fabric_worker_loop(
    mut cpu: V2FastCpu,
    my_id: usize,
    shared: Arc<SharedFabricState>,
    control_rx: Receiver<FabricControl>,
) {
    let mut last_gen: u64 = 0;

    loop {
        // --- Wait phase: spin briefly, then park ---
        let mut spins: u32 = 0;
        loop {
            // Check shutdown
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }

            // Check for new epoch
            let current_gen = shared.generation.load(Ordering::Acquire);
            if current_gen > last_gen {
                last_gen = current_gen;
                break;
            }

            // Check for control messages (non-blocking)
            match control_rx.try_recv() {
                Ok(FabricControl::Snapshot(resp)) => {
                    let snap = CpuSnapshot::from_cpu(&cpu);
                    let _ = resp.send(snap);
                }
                Ok(FabricControl::Reset(program)) => {
                    cpu = V2FastCpu::from_program_with_mmio(&program);
                    shared.sends[my_id][0].store(0, Ordering::Release);
                    shared.sends[my_id][1].store(0, Ordering::Release);
                    shared.recvs[my_id][0].store(0, Ordering::Release);
                    shared.recvs[my_id][1].store(0, Ordering::Release);
                    shared.halted[my_id].store(false, Ordering::Release);
                    shared.cycle_counts[my_id].store(0, Ordering::Release);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
            }

            // Hybrid wait: spin briefly, then park
            spins += 1;
            if spins < FABRIC_SPIN_LIMIT {
                std::hint::spin_loop();
            } else {
                std::thread::park();
                spins = 0;
            }
        }

        // --- Execute phase ---

        // Phase A: Load recv values from shared buffer
        let epoch_size = shared.epoch_size.load(Ordering::Acquire);
        cpu.set_mailbox_in_recv(shared.recvs[my_id][0].load(Ordering::Acquire));
        cpu.set_mailbox_out_recv(shared.recvs[my_id][1].load(Ordering::Acquire));

        // Phase B: Batch execution (replaces per-tick loop)
        cpu.run_batch(epoch_size);

        // Phase C: Export sends + status to shared buffer
        shared.sends[my_id][0].store(cpu.mailbox_in_send(), Ordering::Release);
        shared.sends[my_id][1].store(cpu.mailbox_out_send(), Ordering::Release);
        shared.halted[my_id].store(cpu.is_halted(), Ordering::Release);
        shared.cycle_counts[my_id].store(cpu.cycle_count(), Ordering::Release);

        // Signal completion (replaces barrier.wait())
        shared.done_count.fetch_add(1, Ordering::Release);
    }
}

/// Parallel mailbox-coupled V2FastCpu array with epoch-based synchronization.
///
/// Same epoch contract as V2FastFabric: a value sent by CPU i in epoch N
/// is visible to its neighbor in epoch N+1, never within the same epoch.
///
/// Sprint 303: Generation-based dispatcher replaces mpsc + Barrier.
/// Workers own V2FastCpu directly, use run_batch() for epoch execution,
/// and signal completion via atomic done_count. No blocking primitives
/// on the hot path.
pub struct V2ParallelFabric {
    workers: Vec<FabricWorker>,
    shared: Arc<SharedFabricState>,
    topology: V2FastFabricTopology,
    n: usize,
    /// Thread handles for unpark (cloned from JoinHandle at spawn time).
    thread_handles: Vec<std::thread::Thread>,
}

impl V2ParallelFabric {
    /// Create a parallel fabric of N mailbox-coupled CPUs.
    pub fn new(programs: &[&[u32]], topology: V2FastFabricTopology) -> Self {
        let n = programs.len();
        let shared = Arc::new(SharedFabricState::new(n));
        let mut workers = Vec::with_capacity(n);
        let mut thread_handles = Vec::with_capacity(n);

        for (i, program) in programs.iter().enumerate() {
            let cpu = V2FastCpu::from_program_with_mmio(program);
            let shared_clone = shared.clone();
            let (control_tx, control_rx) = std::sync::mpsc::channel();

            let thread = std::thread::Builder::new()
                .name(format!("v2-fabric-{}", i))
                .spawn(move || {
                    fabric_worker_loop(cpu, i, shared_clone, control_rx);
                })
                .expect("failed to spawn fabric worker");

            thread_handles.push(thread.thread().clone());
            workers.push(FabricWorker {
                thread: Some(thread),
                control_tx,
            });
        }

        Self {
            workers,
            shared,
            topology,
            n,
            thread_handles,
        }
    }

    /// Run epoch-based parallel execution.
    ///
    /// Each epoch:
    /// 1. Sync mailboxes: route send values to neighbor recv buffers (atomic ops)
    /// 2. Dispatch: increment generation, unpark workers
    /// 3. Wait: spin on done_count until all workers complete
    ///
    /// Repeats until `max_cycles` consumed or all CPUs halted.
    pub fn run_epochs(&mut self, max_cycles: u64, epoch_size: u64) {
        if epoch_size == 0 {
            return;
        }
        let mut cycles_used: u64 = 0;

        while cycles_used < max_cycles && !self.all_halted_atomic() {
            let ticks = epoch_size.min(max_cycles - cycles_used);

            // Phase 1: Sync mailboxes (pure atomic reads/writes)
            self.sync_mailboxes();

            // Phase 2: Dispatch epoch via generation protocol
            self.shared.epoch_size.store(ticks, Ordering::Release);
            let current_gen = self.shared.generation.fetch_add(1, Ordering::Release) + 1;
            for th in &self.thread_handles {
                th.unpark();
            }

            // Phase 3: Wait for all workers to complete (spin on done_count)
            let expected = current_gen * self.n as u64;
            while self.shared.done_count.load(Ordering::Acquire) < expected {
                std::hint::spin_loop();
            }

            cycles_used += ticks;
        }
    }

    /// Sync mailboxes: route neighbor send values to recv buffers via atomics.
    /// Called while workers are parked — no contention.
    fn sync_mailboxes(&self) {
        let n = self.n;
        if n < 2 {
            return;
        }

        for i in 0..n {
            // Left neighbor: cpu[i-1] sends right (out_send) → my in_recv
            let left_idx = if i > 0 {
                Some(i - 1)
            } else if self.topology == V2FastFabricTopology::Ring {
                Some(n - 1)
            } else {
                None
            };
            if let Some(li) = left_idx {
                let val = self.shared.sends[li][1].load(Ordering::Acquire);
                self.shared.recvs[i][0].store(val, Ordering::Release);
            }

            // Right neighbor: cpu[i+1] sends left (in_send) → my out_recv
            let right_idx = if i + 1 < n {
                Some(i + 1)
            } else if self.topology == V2FastFabricTopology::Ring {
                Some(0)
            } else {
                None
            };
            if let Some(ri) = right_idx {
                let val = self.shared.sends[ri][0].load(Ordering::Acquire);
                self.shared.recvs[i][1].store(val, Ordering::Release);
            }
        }
    }

    /// Hot-path halted check using atomics.
    fn all_halted_atomic(&self) -> bool {
        (0..self.n).all(|i| self.shared.halted[i].load(Ordering::Acquire))
    }

    // -- State accessors (phase-safe: &mut self prevents cross-dispatch hold) --

    /// Snapshot CPU i's state via control message to worker.
    /// Only valid when workers are parked (after run_epochs returns).
    pub fn cpu_snapshot(&mut self, i: usize) -> CpuSnapshot {
        let (resp_tx, resp_rx) = std::sync::mpsc::channel();
        self.workers[i]
            .control_tx
            .send(FabricControl::Snapshot(resp_tx))
            .unwrap();
        self.thread_handles[i].unpark();
        resp_rx.recv().unwrap()
    }

    /// Snapshot all CPUs via parallel control messages.
    pub fn all_snapshots(&mut self) -> Vec<CpuSnapshot> {
        let receivers: Vec<_> = (0..self.n)
            .map(|i| {
                let (resp_tx, resp_rx) = std::sync::mpsc::channel();
                self.workers[i]
                    .control_tx
                    .send(FabricControl::Snapshot(resp_tx))
                    .unwrap();
                self.thread_handles[i].unpark();
                resp_rx
            })
            .collect();
        receivers.into_iter().map(|rx| rx.recv().unwrap()).collect()
    }

    pub fn num_cpus(&self) -> usize {
        self.n
    }

    pub fn all_halted(&mut self) -> bool {
        self.all_halted_atomic()
    }

    /// Reset all CPUs with new programs without destroying/rebuilding threads.
    /// Useful for steady-state benchmarking.
    pub fn reset(&mut self, programs: &[&[u32]]) {
        assert_eq!(
            programs.len(),
            self.n,
            "reset must provide same number of programs"
        );
        // Send Reset to each worker and unpark
        for (i, program) in programs.iter().enumerate() {
            self.workers[i]
                .control_tx
                .send(FabricControl::Reset(program.to_vec()))
                .unwrap();
            self.thread_handles[i].unpark();
        }
        // Fence: send Snapshot to each worker and wait for response.
        // This guarantees Reset has been processed before we return.
        for i in 0..self.n {
            let (resp_tx, resp_rx) = std::sync::mpsc::channel();
            self.workers[i]
                .control_tx
                .send(FabricControl::Snapshot(resp_tx))
                .unwrap();
            self.thread_handles[i].unpark();
            let _ = resp_rx.recv().unwrap();
        }
    }
}

impl Drop for V2ParallelFabric {
    fn drop(&mut self) {
        // Signal shutdown via atomic flag + unpark all workers
        self.shared.shutdown.store(true, Ordering::Release);
        for th in &self.thread_handles {
            th.unpark();
        }
        // Join all worker threads
        for w in &mut self.workers {
            if let Some(handle) = w.thread.take() {
                let _ = handle.join();
            }
        }
    }
}

// ===========================================================================
// Sprint 342: V2TileSimPool — parallel tile-sim execution with independent
// Simulation per worker. Each worker owns (Simulation, TileCpuV2) built on
// the worker thread (no cross-thread Simulation movement).
// ===========================================================================

enum TileSimMessage {
    RunTick(u64),
    Reset {
        program: Vec<u32>,
        rom_size: usize,
        ram_size: usize,
    },
    Shutdown,
}

/// Result snapshot from a tile-sim worker.
#[derive(Debug, Clone)]
pub struct TileSimSnapshot {
    pub pc: u32,
    pub lr: u32,
    pub flag_z: bool,
    pub flag_c: bool,
    pub halted: bool,
    pub cycle_count: u64,
    pub retired_count: u64,
    pub hash: u64,
    pub hybrid: crate::tile_cpu::V2HybridAssistCounters,
    pub regs: [u64; 16],
    pub ram: [u64; 128],
}

impl TileSimSnapshot {
    /// Capture the same architectural surface used by the V2 golden hash,
    /// plus cycle/retire and hybrid-assist counters for pool diagnostics.
    pub fn capture(cpu: &crate::tile_cpu::TileCpuV2, sim: &crate::simulation::Simulation) -> Self {
        Self {
            pc: cpu.read_pc(sim),
            lr: cpu.read_lr(),
            flag_z: cpu.read_flag_z(sim),
            flag_c: cpu.read_flag_c(sim),
            halted: cpu.is_halted(),
            cycle_count: cpu.read_cycle_count(),
            retired_count: cpu.read_retired_count(),
            hash: crate::tile_cpu::v2_benchmarks::hash_v2_final_state(cpu, sim),
            hybrid: cpu.read_hybrid_assist_counters(),
            regs: std::array::from_fn(|i| cpu.read_reg(sim, i)),
            ram: std::array::from_fn(|i| cpu.read_ram(sim, i)),
        }
    }
}

/// Program spec for one tile-sim worker.
#[derive(Debug, Clone, Copy)]
pub struct TileSimProgramSpec<'a> {
    pub program: &'a [u32],
    pub rom_size: usize,
    pub ram_size: usize,
}

struct TileSimWorker {
    thread: Option<JoinHandle<()>>,
    sender: Sender<TileSimMessage>,
    result: Arc<Mutex<Option<TileSimSnapshot>>>,
}

/// Aggregate status for a V2TileSimPool after a dispatch or reset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileSimPoolSummary {
    pub workers: usize,
    pub ready_workers: usize,
    pub halted_workers: usize,
    pub total_cycles: u64,
    pub total_retired: u64,
    pub combined_hash: u64,
}

impl TileSimPoolSummary {
    pub fn all_ready(&self) -> bool {
        self.ready_workers == self.workers
    }

    pub fn all_halted(&self) -> bool {
        self.workers > 0 && self.halted_workers == self.workers
    }
}

/// One job in a reusable tile-sim batch.
#[derive(Debug, Clone, Copy)]
pub struct TileSimBatchJobSpec<'spec, 'program> {
    pub specs: &'spec [TileSimProgramSpec<'program>],
    pub max_cycles: u64,
}

/// Result for one batch job after reset + dispatch.
#[derive(Debug, Clone)]
pub struct TileSimBatchJobResult {
    pub job_index: usize,
    pub max_cycles: u64,
    pub reset_ns: u128,
    pub run_ns: u128,
    pub reset_summary: TileSimPoolSummary,
    pub summary: TileSimPoolSummary,
    pub snapshots: Vec<Option<TileSimSnapshot>>,
}

impl TileSimBatchJobResult {
    pub fn run_cycles_per_sec(&self) -> f64 {
        if self.run_ns == 0 {
            0.0
        } else {
            self.summary.total_cycles as f64 * 1_000_000_000.0 / self.run_ns as f64
        }
    }

    pub fn wall_cycles_per_sec(&self) -> f64 {
        let wall_ns = self.reset_ns.saturating_add(self.run_ns);
        if wall_ns == 0 {
            0.0
        } else {
            self.summary.total_cycles as f64 * 1_000_000_000.0 / wall_ns as f64
        }
    }
}

/// Aggregate result for a sequence of batch jobs run through one persistent pool.
#[derive(Debug, Clone, Default)]
pub struct TileSimBatchReport {
    pub workers: usize,
    pub jobs: Vec<TileSimBatchJobResult>,
    pub total_cycles: u64,
    pub total_retired: u64,
    pub total_reset_ns: u128,
    pub total_run_ns: u128,
    pub combined_hash: u64,
}

impl TileSimBatchReport {
    pub fn all_ready(&self) -> bool {
        !self.jobs.is_empty() && self.jobs.iter().all(|job| job.reset_summary.all_ready())
    }

    pub fn all_halted(&self) -> bool {
        !self.jobs.is_empty() && self.jobs.iter().all(|job| job.summary.all_halted())
    }

    pub fn run_cycles_per_sec(&self) -> f64 {
        if self.total_run_ns == 0 {
            0.0
        } else {
            self.total_cycles as f64 * 1_000_000_000.0 / self.total_run_ns as f64
        }
    }

    pub fn wall_cycles_per_sec(&self) -> f64 {
        let wall_ns = self.total_reset_ns.saturating_add(self.total_run_ns);
        if wall_ns == 0 {
            0.0
        } else {
            self.total_cycles as f64 * 1_000_000_000.0 / wall_ns as f64
        }
    }
}

fn build_tile_sim_worker_cpu(
    program: &[u32],
    rom_size: usize,
    ram_size: usize,
) -> (crate::simulation::Simulation, crate::tile_cpu::TileCpuV2) {
    use crate::tile_cpu::v2_builder::{V2Builder, V2SynthConfig};

    let mut sim = crate::simulation::Simulation::with_size_layered(128, 640, 16);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(program)
        .with_rom_size(rom_size)
        .with_ram_size(ram_size)
        .with_synth_blocks(V2SynthConfig::max_authority())
        .build(&mut sim);
    (sim, cpu)
}

/// Parallel pool of N independent (Simulation, TileCpuV2) pairs.
/// Each worker builds its own Simulation on the worker thread.
pub struct V2TileSimPool {
    workers: Vec<TileSimWorker>,
    barrier: Arc<Barrier>,
}

impl V2TileSimPool {
    /// Build workers from per-worker program specs.
    /// Each worker builds its Simulation on the worker thread.
    pub fn new(specs: &[TileSimProgramSpec<'_>]) -> Self {
        let n = specs.len();
        let barrier = Arc::new(Barrier::new(n + 1));
        let mut workers = Vec::with_capacity(n);

        for (i, spec) in specs.iter().enumerate() {
            let barrier_clone = barrier.clone();
            let (sender, receiver) = std::sync::mpsc::channel();
            let result: Arc<Mutex<Option<TileSimSnapshot>>> = Arc::new(Mutex::new(None));
            let result_clone = result.clone();
            let prog = spec.program.to_vec();
            let rom_size = spec.rom_size;
            let ram_size = spec.ram_size;

            let thread = std::thread::Builder::new()
                .name(format!("tile-sim-{}", i))
                .spawn(move || {
                    // Build Simulation + CPU on the worker thread.
                    let (mut sim, mut cpu) = build_tile_sim_worker_cpu(&prog, rom_size, ram_size);
                    *result_clone.lock().unwrap() = Some(TileSimSnapshot::capture(&cpu, &sim));
                    barrier_clone.wait();

                    loop {
                        match receiver.recv() {
                            Ok(TileSimMessage::RunTick(max_cycles)) => {
                                for _ in 0..max_cycles {
                                    if cpu.is_halted() {
                                        break;
                                    }
                                    cpu.tick(&mut sim);
                                }
                                *result_clone.lock().unwrap() =
                                    Some(TileSimSnapshot::capture(&cpu, &sim));
                                barrier_clone.wait();
                            }
                            Ok(TileSimMessage::Reset {
                                program,
                                rom_size,
                                ram_size,
                            }) => {
                                let rebuilt =
                                    build_tile_sim_worker_cpu(&program, rom_size, ram_size);
                                sim = rebuilt.0;
                                cpu = rebuilt.1;
                                *result_clone.lock().unwrap() =
                                    Some(TileSimSnapshot::capture(&cpu, &sim));
                                barrier_clone.wait();
                            }
                            Ok(TileSimMessage::Shutdown) | Err(_) => break,
                        }
                    }
                })
                .expect("failed to spawn tile-sim worker");

            workers.push(TileSimWorker {
                thread: Some(thread),
                sender,
                result,
            });
        }

        // Constructor returns only after every worker has built its local
        // Simulation/TileCpuV2 pair and published a zero-cycle snapshot.
        barrier.wait();

        Self { workers, barrier }
    }

    /// Build N workers, each running the same program with max_authority.
    /// Programs are built on worker threads to avoid moving Simulation across threads.
    pub fn uniform(program: &[u32], n: usize) -> Self {
        let specs: Vec<TileSimProgramSpec<'_>> = (0..n)
            .map(|_| TileSimProgramSpec {
                program,
                rom_size: 128,
                ram_size: 128,
            })
            .collect();
        Self::new(&specs)
    }

    /// Dispatch: run all workers for up to max_cycles. Blocks until all complete.
    pub fn run_parallel(&mut self, max_cycles: u64) {
        for w in &self.workers {
            w.sender.send(TileSimMessage::RunTick(max_cycles)).unwrap();
        }
        self.barrier.wait();
    }

    /// Reset all workers with same-sized replacement specs.
    ///
    /// Rebuild happens on each worker thread, preserving the pool object and
    /// thread ownership model while allowing callers to reuse the pool across
    /// mixed workloads. Blocks until every worker has rebuilt and published a
    /// fresh zero-cycle snapshot.
    pub fn reset(&mut self, specs: &[TileSimProgramSpec<'_>]) {
        assert_eq!(
            specs.len(),
            self.workers.len(),
            "reset must provide one spec per worker"
        );
        for (w, spec) in self.workers.iter().zip(specs.iter()) {
            w.sender
                .send(TileSimMessage::Reset {
                    program: spec.program.to_vec(),
                    rom_size: spec.rom_size,
                    ram_size: spec.ram_size,
                })
                .unwrap();
        }
        self.barrier.wait();
    }

    /// Reset all workers to the same program.
    pub fn reset_uniform(&mut self, program: &[u32]) {
        let specs: Vec<TileSimProgramSpec<'_>> = (0..self.workers.len())
            .map(|_| TileSimProgramSpec {
                program,
                rom_size: 128,
                ram_size: 128,
            })
            .collect();
        self.reset(&specs);
    }

    /// Run a sequence of mixed jobs through the persistent pool.
    ///
    /// Each job resets every worker to the job's specs, records reset readiness,
    /// dispatches `max_cycles`, then records final snapshots and aggregate timing.
    pub fn run_batch(&mut self, jobs: &[TileSimBatchJobSpec<'_, '_>]) -> TileSimBatchReport {
        let mut results = Vec::with_capacity(jobs.len());
        let mut total_cycles = 0u64;
        let mut total_retired = 0u64;
        let mut total_reset_ns = 0u128;
        let mut total_run_ns = 0u128;
        let mut combined_hash = 0xCBF2_9CE4_8422_2325u64;

        for (job_index, job) in jobs.iter().enumerate() {
            assert_eq!(
                job.specs.len(),
                self.workers.len(),
                "batch job {} must provide one spec per worker",
                job_index
            );

            let reset_start = std::time::Instant::now();
            self.reset(job.specs);
            let reset_ns = reset_start.elapsed().as_nanos();
            let reset_summary = self.summary();

            let run_start = std::time::Instant::now();
            self.run_parallel(job.max_cycles);
            let run_ns = run_start.elapsed().as_nanos();
            let summary = self.summary();
            let snapshots = self.snapshots();

            total_cycles = total_cycles.saturating_add(summary.total_cycles);
            total_retired = total_retired.saturating_add(summary.total_retired);
            total_reset_ns = total_reset_ns.saturating_add(reset_ns);
            total_run_ns = total_run_ns.saturating_add(run_ns);
            combined_hash ^= summary
                .combined_hash
                .rotate_left((job_index % 63) as u32 + 1);
            combined_hash = combined_hash.wrapping_mul(0x100_0000_01B3);

            results.push(TileSimBatchJobResult {
                job_index,
                max_cycles: job.max_cycles,
                reset_ns,
                run_ns,
                reset_summary,
                summary,
                snapshots,
            });
        }

        TileSimBatchReport {
            workers: self.workers.len(),
            jobs: results,
            total_cycles,
            total_retired,
            total_reset_ns,
            total_run_ns,
            combined_hash,
        }
    }

    /// Read worker i's result snapshot (after dispatch).
    pub fn snapshot(&self, i: usize) -> Option<TileSimSnapshot> {
        self.workers[i].result.lock().unwrap().clone()
    }

    /// Read all worker snapshots in worker-index order.
    pub fn snapshots(&self) -> Vec<Option<TileSimSnapshot>> {
        (0..self.workers.len()).map(|i| self.snapshot(i)).collect()
    }

    /// Aggregate published worker snapshots.
    pub fn summary(&self) -> TileSimPoolSummary {
        let snapshots = self.snapshots();
        let mut summary = TileSimPoolSummary {
            workers: self.workers.len(),
            ..TileSimPoolSummary::default()
        };
        let mut combined = 0xCBF2_9CE4_8422_2325u64;
        for (i, snap) in snapshots.iter().enumerate() {
            if let Some(s) = snap {
                summary.ready_workers += 1;
                summary.halted_workers += s.halted as usize;
                summary.total_cycles = summary.total_cycles.saturating_add(s.cycle_count);
                summary.total_retired = summary.total_retired.saturating_add(s.retired_count);
                combined ^= s.hash.rotate_left((i % 63) as u32 + 1);
                combined = combined.wrapping_mul(0x100_0000_01B3);
            }
        }
        summary.combined_hash = combined;
        summary
    }

    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    pub fn all_halted(&self) -> bool {
        self.workers.iter().all(|w| {
            w.result
                .lock()
                .unwrap()
                .as_ref()
                .map(|s| s.halted)
                .unwrap_or(false)
        })
    }
}

impl Drop for V2TileSimPool {
    fn drop(&mut self) {
        for w in &self.workers {
            let _ = w.sender.send(TileSimMessage::Shutdown);
        }
        for w in &mut self.workers {
            if let Some(handle) = w.thread.take() {
                let _ = handle.join();
            }
        }
    }
}
