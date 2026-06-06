use crate::block_sparse_state::BlockSparseState;
#[cfg(feature = "cuda")]
use crate::cuda::CudaRuntime;
use crate::quantum::QGate;

/// Hybrid Quantum State Manager (EPIC 29 Phase 2)
///
/// Implements an "L1 Cache" strategy for massive quantum states:
/// - **Storage**: `BlockSparseState` (CPU RAM) holds the full state (e.g., 60 qubits).
/// - **Compute**: `GpuBlockSparseState` (GPU VRAM) holds specific dense blocks during processing.
///
/// Strategies:
/// - **Lazy**: Operations are performed on CPU by default.
/// - **Eager**: If a block becomes dense (> threshold), it is marked for GPU offload.
pub struct HybridQState {
    /// The master copy of the state (CPU)
    pub cpu_state: BlockSparseState,

    /// Optional GPU runtime for acceleration
    #[cfg(feature = "cuda")]
    pub gpu_rt: Option<std::sync::Arc<CudaRuntime>>,

    /// Threshold for switching to GPU (blocks)
    pub gpu_threshold: usize,

    /// Compiled kernels (cached)
    #[cfg(feature = "cuda")]
    pub kernels: Option<crate::block_sparse_state::BlockSparseKernels>,
}

impl HybridQState {
    /// Create a new Hybrid State
    pub fn new(n_qubits: u8) -> Self {
        Self {
            cpu_state: BlockSparseState::new_zero(n_qubits),
            #[cfg(feature = "cuda")]
            gpu_rt: None,
            #[cfg(feature = "cuda")]
            kernels: None,
            gpu_threshold: 1024,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn attach_gpu(&mut self, rt: std::sync::Arc<CudaRuntime>) -> Result<(), String> {
        // Compile kernels on attach
        let kernels = crate::block_sparse_state::compile_block_sparse_kernels(rt.context())
            .map_err(|e| format!("Failed to compile kernels: {:?}", e))?;

        self.gpu_rt = Some(rt);
        self.kernels = Some(kernels);
        Ok(())
    }

    /// Apply a gate (routing logic)
    pub fn apply_gate(&mut self, gate: &QGate) {
        // ... (elided for brevity, kept same)
        let use_gpu = if cfg!(feature = "cuda") {
            self.cpu_state.block_count() >= self.gpu_threshold
        } else {
            false
        };

        match gate {
            QGate::H(_) | QGate::X(_) | QGate::Z(_) | QGate::Phase(_, _) if use_gpu => {
                if let Err(e) = self.apply_gate_gpu(gate) {
                    println!("GPU execution failed, falling back to CPU: {}", e);
                    self.apply_gate_cpu(gate);
                }
            }
            _ => self.apply_gate_cpu(gate),
        }
    }

    fn apply_gate_cpu(&mut self, gate: &QGate) {
        // Dispatch to BlockSparseState methods
        match gate {
            QGate::H(q) => self.cpu_state.apply_h(*q),
            QGate::X(q) => self.cpu_state.apply_x(*q),
            QGate::Z(q) => self.cpu_state.apply_z(*q),
            _ => {
                println!(
                    "Warning: Gate {:?} not yet supported in BlockSparse CPU",
                    gate
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn apply_gate_gpu(&mut self, gate: &QGate) -> Result<(), String> {
        use crate::block_sparse_state::{
            apply_gate_cross_block_gpu, launch_within_block_kernel, GpuBlockSparseState,
        };

        let rt = self.gpu_rt.as_ref().ok_or("GPU runtime not attached")?;
        let kernels = self.kernels.as_ref().ok_or("Kernels not compiled")?;

        // 0. Prepare CPU state (create partners if needed)
        let target_qubit = self.get_gate_qubit(gate);
        if target_qubit >= crate::block_sparse_state::BLOCK_SHIFT {
            self.cpu_state.ensure_partner_blocks_exist(target_qubit);
        }

        // 1. Hydrate (Upload to GPU)
        let mut gpu_state = GpuBlockSparseState::from_host(rt, &self.cpu_state)
            .map_err(|e| format!("Upload failed: {:?}", e))?;

        // 2. Execute
        let gate_matrix = self.get_gate_matrix_struct(gate);

        if target_qubit < crate::block_sparse_state::BLOCK_SHIFT {
            // Within block
            launch_within_block_kernel(
                kernels,
                &mut gpu_state,
                gate_matrix,
                target_qubit,
                rt.stream(),
            )
            .map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        } else {
            // Cross block
            apply_gate_cross_block_gpu(rt, &mut gpu_state, kernels, gate_matrix, target_qubit)
                .map_err(|e| format!("Cross-block launch failed: {:?}", e))?;
        }

        // 3. Dehydrate (Download from GPU)
        let new_cpu_state = gpu_state
            .to_host(rt)
            .map_err(|e| format!("Download failed: {:?}", e))?;

        self.cpu_state = new_cpu_state;

        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    fn apply_gate_gpu(&mut self, _gate: &QGate) -> Result<(), String> {
        Err("CUDA feature disabled".into())
    }

    #[cfg(feature = "cuda")]
    fn get_gate_matrix_struct(&self, gate: &QGate) -> crate::block_sparse_state::GateMatrix16 {
        use crate::block_sparse_state::GateMatrix16;
        use half::f16;

        match gate {
            QGate::H(_) => GateMatrix16::hadamard(),
            QGate::X(_) => GateMatrix16::x(),
            QGate::Z(_) => {
                // Z = [[1, 0], [0, -1]]
                GateMatrix16 {
                    re_00: f16::ONE,
                    im_00: f16::ZERO,
                    re_01: f16::ZERO,
                    im_01: f16::ZERO,
                    re_10: f16::ZERO,
                    im_10: f16::ZERO,
                    re_11: f16::NEG_ONE,
                    im_11: f16::ZERO,
                }
            }
            QGate::Phase(_, theta) => {
                // P(theta) = [[1, 0], [0, exp(i*theta)]]
                let (s, c) = theta.sin_cos();
                GateMatrix16 {
                    re_00: f16::ONE,
                    im_00: f16::ZERO,
                    re_01: f16::ZERO,
                    im_01: f16::ZERO,
                    re_10: f16::ZERO,
                    im_10: f16::ZERO,
                    re_11: f16::from_f32(c),
                    im_11: f16::from_f32(s),
                }
            }
            _ => GateMatrix16::x(), // Fallback/Panic? Default to X for now (obvious error if used wrong)
        }
    }

    fn get_gate_qubit(&self, gate: &QGate) -> u8 {
        match gate {
            QGate::H(q) | QGate::X(q) | QGate::Z(q) | QGate::Phase(q, _) => *q,
            _ => 0,
        }
    }
}
