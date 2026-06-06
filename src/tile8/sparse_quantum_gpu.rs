//! GPU-Accelerated Sparse Quantum Operations (Sprint 57.0)
//!
//! Uses wgpu compute shaders to process sparse quantum blocks in parallel.
//! Each block contains 128 complex amplitudes (f32 precision on GPU).
//!
//! # Performance Characteristics
//!
//! - Block-parallel: Each workgroup processes one block of 128 amplitudes
//! - Memory coalesced: SoA layout (separate real/imag buffers)
//! - GPU-native: No CPU-GPU sync between operations in a batch
//!
//! # Example
//!
//! ```ignore
//! use engine::tile8::sparse_quantum_gpu::SparseQuantumGpu;
//!
//! let mut gpu = pollster::block_on(SparseQuantumGpu::new(1_000_000));
//!
//! // Initialize to |0...0⟩
//! gpu.init_zero();
//!
//! // Apply Hadamard to qubit 0
//! gpu.hadamard(0);
//!
//! // Apply CNOT(0, 1)
//! gpu.cnot(0, 1);
//!
//! // Get results
//! let (real, imag) = pollster::block_on(gpu.download());
//! ```

use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// Block size: 128 complex amplitudes per block
pub const BLOCK_SIZE: usize = 128;

/// Quantum gate types for the shader
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum GateType {
    Hadamard = 0,
    PauliX = 1,
    PauliZ = 2,
    CNot = 3,
}

/// GPU parameters struct (must match WGSL)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantumParams {
    num_blocks: u32,
    gate_type: u32,
    target_qubit: u32,
    control_qubit: u32,
}

/// GPU-accelerated sparse quantum state
pub struct SparseQuantumGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Storage buffers for amplitudes (SoA layout)
    real_buffer: wgpu::Buffer,
    imag_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,

    // Compute pipelines
    hadamard_pipeline: wgpu::ComputePipeline,
    pauli_x_pipeline: wgpu::ComputePipeline,
    pauli_z_pipeline: wgpu::ComputePipeline,
    cnot_pipeline: wgpu::ComputePipeline,
    cnot_control_high_pipeline: wgpu::ComputePipeline,
    cnot_target_high_pipeline: wgpu::ComputePipeline,
    cnot_both_high_pipeline: wgpu::ComputePipeline,
    init_zero_pipeline: wgpu::ComputePipeline,
    create_w_pipeline: wgpu::ComputePipeline,

    // Bind group
    bind_group: wgpu::BindGroup,

    // State
    num_blocks: u32,
    n_qubits: usize,
}

impl SparseQuantumGpu {
    /// Create a new GPU sparse quantum state with capacity for n_qubits
    pub async fn new(n_qubits: usize) -> Self {
        assert!(n_qubits >= 7, "Minimum 7 qubits required");

        let num_blocks = ((n_qubits + BLOCK_SIZE - 1) / BLOCK_SIZE) as u32;
        let total_amplitudes = (num_blocks as usize) * BLOCK_SIZE;
        let buffer_size = (total_amplitudes * std::mem::size_of::<f32>()) as u64;

        // Initialize wgpu
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("Failed to find GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Sparse Quantum GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // Create storage buffers
        let real_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Real Amplitudes"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let imag_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Imaginary Amplitudes"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quantum Params"),
            contents: bytemuck::bytes_of(&QuantumParams {
                num_blocks,
                gate_type: 0,
                target_qubit: 0,
                control_qubit: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Staging buffer for readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size * 2, // For both real and imag
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sparse Quantum Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/sparse_quantum.wgsl"
            ))),
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sparse Quantum Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparse Quantum Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: real_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: imag_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sparse Quantum Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create pipelines for each kernel
        let hadamard_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hadamard Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "hadamard",
            compilation_options: Default::default(),
        });

        let pauli_x_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pauli X Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "pauli_x",
            compilation_options: Default::default(),
        });

        let pauli_z_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pauli Z Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "pauli_z",
            compilation_options: Default::default(),
        });

        let cnot_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CNOT Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cnot",
            compilation_options: Default::default(),
        });

        let cnot_control_high_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CNOT Control-High Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cnot_control_high",
                compilation_options: Default::default(),
            });

        let cnot_target_high_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CNOT Target-High Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cnot_target_high",
                compilation_options: Default::default(),
            });

        let cnot_both_high_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CNOT Both-High Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cnot_both_high",
                compilation_options: Default::default(),
            });

        let init_zero_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Init Zero Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "init_zero",
            compilation_options: Default::default(),
        });

        let create_w_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Create W State Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "create_w_state",
            compilation_options: Default::default(),
        });

        Self {
            device,
            queue,
            real_buffer,
            imag_buffer,
            params_buffer,
            staging_buffer,
            hadamard_pipeline,
            pauli_x_pipeline,
            pauli_z_pipeline,
            cnot_pipeline,
            cnot_control_high_pipeline,
            cnot_target_high_pipeline,
            cnot_both_high_pipeline,
            init_zero_pipeline,
            create_w_pipeline,
            bind_group,
            num_blocks,
            n_qubits,
        }
    }

    /// Get number of qubits
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get number of blocks
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// Update parameters and dispatch a compute pass
    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, target: u32, control: u32) {
        // Update params
        let params = QuantumParams {
            num_blocks: self.num_blocks,
            gate_type: 0, // Not used by most kernels
            target_qubit: target,
            control_qubit: control,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Quantum Op Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Quantum Op Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(self.num_blocks, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Initialize state to |0...0⟩
    pub fn init_zero(&self) {
        self.dispatch(&self.init_zero_pipeline, 0, 0);
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Apply Hadamard gate to target qubit (0-6 for local)
    pub fn hadamard(&self, target: u32) {
        assert!(target < 7, "Target qubit must be 0-6 for local operations");
        self.dispatch(&self.hadamard_pipeline, target, 0);
    }

    /// Apply Pauli X gate to target qubit
    pub fn pauli_x(&self, target: u32) {
        assert!(target < 7, "Target qubit must be 0-6 for local operations");
        self.dispatch(&self.pauli_x_pipeline, target, 0);
    }

    /// Apply Pauli Z gate to target qubit
    pub fn pauli_z(&self, target: u32) {
        assert!(target < 7, "Target qubit must be 0-6 for local operations");
        self.dispatch(&self.pauli_z_pipeline, target, 0);
    }

    /// Apply CNOT gate with control and target qubits (both 0-6)
    pub fn cnot(&self, control: u32, target: u32) {
        assert!(
            control < 7,
            "Control qubit must be 0-6 for local operations"
        );
        assert!(target < 7, "Target qubit must be 0-6 for local operations");
        assert!(control != target, "Control and target must be different");
        self.dispatch(&self.cnot_pipeline, target, control);
    }

    /// Apply CNOT gate with any qubit combination (supports cross-block)
    ///
    /// Automatically routes to the appropriate kernel:
    /// - Both < 7: within-block CNOT (parallel)
    /// - Control >= 7, Target < 7: control-high CNOT (parallel per-block)
    /// - Control < 7, Target >= 7: target-high CNOT (block pair swap)
    /// - Both >= 7: both-high CNOT (full block swap)
    pub fn cnot_any(&self, control: u32, target: u32) {
        assert!(control != target, "Control and target must be different");

        let control_high = control >= 7;
        let target_high = target >= 7;

        match (control_high, target_high) {
            (false, false) => {
                // Both within-block: use standard CNOT
                self.cnot(control, target);
            }
            (true, false) => {
                // Control >= 7, Target < 7: control-high
                self.cnot_control_high(control, target);
            }
            (false, true) => {
                // Control < 7, Target >= 7: target-high
                self.cnot_target_high(control, target);
            }
            (true, true) => {
                // Both >= 7: both-high
                self.cnot_both_high(control, target);
            }
        }
    }

    /// CNOT with control qubit >= 7, target qubit < 7
    ///
    /// Control bit is in block ID; target is local within block.
    /// Parallel execution: each block independently checks control bit.
    pub fn cnot_control_high(&self, control: u32, target: u32) {
        assert!(
            control >= 7,
            "Control qubit must be >= 7 for control-high CNOT"
        );
        assert!(target < 7, "Target qubit must be < 7 for control-high CNOT");

        // Pass control bit position (control - 7) and local target
        let params = QuantumParams {
            num_blocks: self.num_blocks,
            gate_type: 0,
            target_qubit: target,
            control_qubit: control - 7, // Block bit position
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CNOT Control-High Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CNOT Control-High Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cnot_control_high_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(self.num_blocks, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// CNOT with control qubit < 7, target qubit >= 7
    ///
    /// Control is local; target bit is in block ID.
    /// Block pair operation: swaps amplitudes conditionally based on control bit.
    pub fn cnot_target_high(&self, control: u32, target: u32) {
        assert!(
            control < 7,
            "Control qubit must be < 7 for target-high CNOT"
        );
        assert!(
            target >= 7,
            "Target qubit must be >= 7 for target-high CNOT"
        );

        // Pass local control and target bit position (target - 7)
        let params = QuantumParams {
            num_blocks: self.num_blocks,
            gate_type: 0,
            target_qubit: target - 7, // Block bit position
            control_qubit: control,   // Local bit position
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CNOT Target-High Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CNOT Target-High Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cnot_target_high_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            // Dispatch over all blocks; shader filters to process only primary blocks
            cpass.dispatch_workgroups(self.num_blocks, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// CNOT with both control and target qubits >= 7
    ///
    /// Both bits are in block ID.
    /// Block pair operation: swaps ALL amplitudes between paired blocks.
    pub fn cnot_both_high(&self, control: u32, target: u32) {
        assert!(
            control >= 7,
            "Control qubit must be >= 7 for both-high CNOT"
        );
        assert!(target >= 7, "Target qubit must be >= 7 for both-high CNOT");
        assert!(control != target, "Control and target must be different");

        // Pass block bit positions for both
        let params = QuantumParams {
            num_blocks: self.num_blocks,
            gate_type: 0,
            target_qubit: target - 7,   // Block bit position
            control_qubit: control - 7, // Block bit position
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CNOT Both-High Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CNOT Both-High Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cnot_both_high_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            // Dispatch over all blocks; shader filters to process valid pairs
            cpass.dispatch_workgroups(self.num_blocks, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Create W state on n qubits
    pub fn create_w_state(&self, n_qubits: u32) {
        // Use target_qubit field to pass n_qubits to shader
        let params = QuantumParams {
            num_blocks: self.num_blocks,
            gate_type: 0,
            target_qubit: n_qubits,
            control_qubit: 0,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Create W State Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create W State Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.create_w_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(self.num_blocks, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Wait for all GPU operations to complete
    pub fn sync(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Download amplitudes from GPU
    pub async fn download(&self) -> (Vec<f32>, Vec<f32>) {
        let total_amplitudes = (self.num_blocks as usize) * BLOCK_SIZE;
        let buffer_size = (total_amplitudes * std::mem::size_of::<f32>()) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Download Encoder"),
            });

        // Copy real buffer to staging
        encoder.copy_buffer_to_buffer(&self.real_buffer, 0, &self.staging_buffer, 0, buffer_size);
        // Copy imag buffer to staging (after real)
        encoder.copy_buffer_to_buffer(
            &self.imag_buffer,
            0,
            &self.staging_buffer,
            buffer_size,
            buffer_size,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let buffer_slice = self.staging_buffer.slice(..buffer_size * 2);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();

            let real: Vec<f32> = bytemuck::cast_slice(&data[..buffer_size as usize]).to_vec();
            let imag: Vec<f32> = bytemuck::cast_slice(&data[buffer_size as usize..]).to_vec();

            drop(data);
            self.staging_buffer.unmap();

            (real, imag)
        } else {
            panic!("Failed to map staging buffer");
        }
    }

    /// Upload amplitudes to GPU
    pub fn upload(&self, real: &[f32], imag: &[f32]) {
        self.queue
            .write_buffer(&self.real_buffer, 0, bytemuck::cast_slice(real));
        self.queue
            .write_buffer(&self.imag_buffer, 0, bytemuck::cast_slice(imag));
    }

    /// Create GHZ state: |0...0⟩ + |1...1⟩) / √2
    /// Only works for qubits 0-6 (local within first block)
    pub fn create_ghz_local(&self, n_qubits: u32) {
        assert!(n_qubits <= 7, "GHZ local only supports up to 7 qubits");

        // Initialize to |0...0⟩
        self.init_zero();
        self.sync();

        // H on qubit 0
        self.hadamard(0);

        // CNOT chain: CNOT(0,1), CNOT(1,2), ..., CNOT(n-2, n-1)
        for i in 0..(n_qubits - 1) {
            self.cnot(i, i + 1);
        }

        self.sync();
    }

    /// Verify GHZ state fidelity
    pub async fn verify_ghz(&self, n_qubits: u32) -> f64 {
        let (real, imag) = self.download().await;

        // GHZ state should have:
        // - amplitude 1/√2 at |0...0⟩ (index 0)
        // - amplitude 1/√2 at |1...1⟩ (index 2^n - 1)
        // - all others ~0

        let expected_amp = 1.0 / 2.0_f64.sqrt();
        let idx_all_ones = (1usize << n_qubits) - 1;

        let amp_00 = (real[0] as f64).powi(2) + (imag[0] as f64).powi(2);
        let amp_11 = (real[idx_all_ones] as f64).powi(2) + (imag[idx_all_ones] as f64).powi(2);

        // Fidelity = P(|0...0⟩) + P(|1...1⟩)
        let fidelity = amp_00.sqrt() + amp_11.sqrt();

        // Should be close to √2 * 1/√2 = 1
        fidelity / (2.0 * expected_amp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_init() {
        let gpu = pollster::block_on(SparseQuantumGpu::new(128));
        assert_eq!(gpu.num_blocks(), 1);
        assert_eq!(gpu.n_qubits(), 128);
    }

    #[test]
    fn test_gpu_init_zero() {
        let gpu = pollster::block_on(SparseQuantumGpu::new(128));
        gpu.init_zero();
        gpu.sync();

        let (real, imag) = pollster::block_on(gpu.download());

        // First amplitude should be 1.0
        assert!((real[0] - 1.0).abs() < 1e-6);
        assert!(imag[0].abs() < 1e-6);

        // Others should be 0
        for i in 1..128 {
            assert!(real[i].abs() < 1e-6, "real[{}] = {}", i, real[i]);
            assert!(imag[i].abs() < 1e-6, "imag[{}] = {}", i, imag[i]);
        }
    }

    #[test]
    fn test_gpu_hadamard() {
        let gpu = pollster::block_on(SparseQuantumGpu::new(128));
        gpu.init_zero();
        gpu.sync();

        // Apply H to qubit 0
        gpu.hadamard(0);
        gpu.sync();

        let (real, imag) = pollster::block_on(gpu.download());

        let expected = 1.0 / 2.0_f32.sqrt();

        // Should have equal superposition of |0⟩ and |1⟩
        assert!((real[0] - expected).abs() < 1e-5, "real[0] = {}", real[0]);
        assert!((real[1] - expected).abs() < 1e-5, "real[1] = {}", real[1]);
        assert!(imag[0].abs() < 1e-6);
        assert!(imag[1].abs() < 1e-6);
    }

    #[test]
    fn test_gpu_ghz_3() {
        let gpu = pollster::block_on(SparseQuantumGpu::new(128));
        gpu.create_ghz_local(3);

        let fidelity = pollster::block_on(gpu.verify_ghz(3));
        assert!(fidelity > 0.99, "GHZ fidelity: {}", fidelity);
    }

    #[test]
    fn test_gpu_cnot_control_high() {
        // Test CNOT where control qubit >= 7 (in block ID), target < 7 (local)
        // Need at least 2 blocks (256 amplitudes for 8 qubits)
        let gpu = pollster::block_on(SparseQuantumGpu::new(256));
        gpu.init_zero();
        gpu.sync();

        // Apply H to qubit 7 (block bit 0) - creates superposition across blocks
        // For this test, we manually set up a state
        // Block 0 and Block 1 should have amplitudes

        // First create |+⟩ on qubit 7 by setting blocks 0 and 1
        let mut real = vec![0.0f32; 256];
        let mut imag = vec![0.0f32; 256];
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();

        // |0⟩ on qubit 7 means block 0, |1⟩ means block 1
        // Set up |+⟩_7 ⊗ |0⟩_0 = (|00...0⟩ + |10...0⟩_7) / √2
        real[0] = inv_sqrt2; // Block 0, amplitude 0
        real[128] = inv_sqrt2; // Block 1, amplitude 0

        gpu.upload(&real, &imag);

        // Apply CNOT(7, 0) - control=7 (block bit), target=0 (local)
        // This should flip qubit 0 only in block 1 (where qubit 7 = 1)
        gpu.cnot_control_high(7, 0);
        gpu.sync();

        let (real_out, _imag_out) = pollster::block_on(gpu.download());

        // Expected: |00...0⟩/√2 + |10...1⟩/√2
        // Block 0, index 0: 1/√2 (unchanged)
        // Block 1, index 1: 1/√2 (qubit 0 flipped from 0 to 1)
        assert!(
            (real_out[0] - inv_sqrt2).abs() < 1e-5,
            "Block 0, idx 0: {}",
            real_out[0]
        );
        assert!(
            real_out[128].abs() < 1e-5,
            "Block 1, idx 0 should be 0: {}",
            real_out[128]
        );
        assert!(
            (real_out[129] - inv_sqrt2).abs() < 1e-5,
            "Block 1, idx 1: {}",
            real_out[129]
        );
    }

    #[test]
    fn test_gpu_cnot_target_high() {
        // Test CNOT where control < 7 (local), target >= 7 (block bit)
        let gpu = pollster::block_on(SparseQuantumGpu::new(256));

        let mut real = vec![0.0f32; 256];
        let mut imag = vec![0.0f32; 256];
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();

        // Set up |+⟩_0 ⊗ |0⟩_7 = (|0⟩ + |1⟩)/√2 in block 0
        real[0] = inv_sqrt2; // |0⟩
        real[1] = inv_sqrt2; // |1⟩

        gpu.upload(&real, &imag);

        // Apply CNOT(0, 7) - control=0 (local), target=7 (block bit)
        // This should swap amplitudes between blocks where control=1
        // Index 1 has control=1, so swap block0[1] <-> block1[1]
        gpu.cnot_target_high(0, 7);
        gpu.sync();

        let (real_out, _imag_out) = pollster::block_on(gpu.download());

        // Expected: (|0⟩_0|0⟩_7 + |1⟩_0|1⟩_7) / √2
        // Block 0, index 0: 1/√2 (unchanged - control=0)
        // Block 0, index 1: 0 (swapped to block 1)
        // Block 1, index 1: 1/√2 (swapped from block 0)
        assert!(
            (real_out[0] - inv_sqrt2).abs() < 1e-5,
            "Block 0, idx 0: {}",
            real_out[0]
        );
        assert!(
            real_out[1].abs() < 1e-5,
            "Block 0, idx 1 should be 0: {}",
            real_out[1]
        );
        assert!(
            (real_out[129] - inv_sqrt2).abs() < 1e-5,
            "Block 1, idx 1: {}",
            real_out[129]
        );
    }

    #[test]
    fn test_gpu_cnot_both_high() {
        // Test CNOT where both control and target >= 7 (both in block bits)
        // Need 4 blocks (512 amplitudes minimum)
        let gpu = pollster::block_on(SparseQuantumGpu::new(512));

        let mut real = vec![0.0f32; 512];
        let mut imag = vec![0.0f32; 512];
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();

        // Blocks: 0 (q7=0,q8=0), 1 (q7=1,q8=0), 2 (q7=0,q8=1), 3 (q7=1,q8=1)
        // Set up |+⟩_7 ⊗ |0⟩_8 = (|0⟩_7 + |1⟩_7)/√2 ⊗ |0⟩_8
        // = (|block0⟩ + |block1⟩) / √2
        real[0] = inv_sqrt2; // Block 0, index 0
        real[128] = inv_sqrt2; // Block 1, index 0

        gpu.upload(&real, &imag);

        // Apply CNOT(7, 8) - control=7 (block bit 0), target=8 (block bit 1)
        // Block 1 has control=1, target=0, so swap block1 <-> block3
        gpu.cnot_both_high(7, 8);
        gpu.sync();

        let (real_out, _imag_out) = pollster::block_on(gpu.download());

        // Expected: (|block0⟩ + |block3⟩) / √2
        // Block 0, index 0: 1/√2 (unchanged)
        // Block 1, index 0: 0 (swapped to block 3)
        // Block 3, index 0: 1/√2 (swapped from block 1)
        assert!(
            (real_out[0] - inv_sqrt2).abs() < 1e-5,
            "Block 0, idx 0: {}",
            real_out[0]
        );
        assert!(
            real_out[128].abs() < 1e-5,
            "Block 1, idx 0 should be 0: {}",
            real_out[128]
        );
        assert!(
            (real_out[384] - inv_sqrt2).abs() < 1e-5,
            "Block 3, idx 0: {}",
            real_out[384]
        );
    }

    #[test]
    fn test_gpu_cnot_any_routing() {
        // Test that cnot_any correctly routes to the right implementation
        let gpu = pollster::block_on(SparseQuantumGpu::new(512));
        gpu.init_zero();
        gpu.sync();

        // H on qubit 0
        gpu.hadamard(0);
        gpu.sync();

        // This should use within-block CNOT
        gpu.cnot_any(0, 1);
        gpu.sync();

        let (real, _imag) = pollster::block_on(gpu.download());

        // Should create Bell state (|00⟩ + |11⟩)/√2 in block 0
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        assert!(
            (real[0] - inv_sqrt2).abs() < 1e-5,
            "Expected |00⟩ amplitude"
        );
        assert!(
            (real[3] - inv_sqrt2).abs() < 1e-5,
            "Expected |11⟩ amplitude"
        );
    }
}
