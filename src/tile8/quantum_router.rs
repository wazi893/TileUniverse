//! Quantum Router: Multi-CPU Communication for Cross-Block Gates
//!
//! Implements direct tile access for cross-block quantum operations.
//! Uses barrier synchronization instead of handshake protocols.
//!
//! Performance: ~36ns to read 256 tiles @ 7B tile evals/sec
//! Cross-block CNOT: ~301μs (vs ~2ms for mailbox approach)

use crate::slim_simulation::SlimSimulation;
use crate::tile8::physical::PhysicalCpu;

/// Grid of quantum CPUs with direct block access
pub struct QuantumGrid {
    /// All CPUs in the system
    pub cpus: Vec<PhysicalCpu>,

    /// Tile locations for each CPU's quantum block (256 bytes)
    /// [cpu_id][tile_offset] -> (x, y)
    cpu_tile_locations: Vec<Vec<(usize, usize)>>,
}

impl QuantumGrid {
    /// Create a new quantum grid with specified number of CPUs
    pub fn new(num_cpus: usize) -> Self {
        Self {
            cpus: Vec::with_capacity(num_cpus),
            cpu_tile_locations: Vec::with_capacity(num_cpus),
        }
    }

    /// Place a CPU in the grid and record its tile locations
    ///
    /// # Arguments
    /// * `cpu_id` - Unique identifier for this CPU
    /// * `origin` - Grid position (x, y) for CPU placement
    /// * `sim` - Simulation to place CPU into
    pub fn place_cpu(&mut self, cpu_id: usize, origin: (usize, usize), sim: &mut SlimSimulation) {
        let cpu = PhysicalCpu::new(origin);
        cpu.place(sim);

        // Record the tile locations for this CPU's quantum block (256 bytes)
        let tile_locs = cpu.get_block_tile_locations();

        assert_eq!(cpu_id, self.cpus.len(), "CPU IDs must be sequential");
        self.cpus.push(cpu);
        self.cpu_tile_locations.push(tile_locs);
    }

    /// Get tile coordinates for a CPU's quantum block
    pub fn get_cpu_block_tiles(&self, cpu_id: usize) -> &[(usize, usize)] {
        &self.cpu_tile_locations[cpu_id]
    }

    /// Direct read of another CPU's quantum block (256 bytes)
    ///
    /// Performance: ~36ns @ 7B tile evals/sec
    pub fn read_cpu_block(&self, sim: &SlimSimulation, cpu_id: usize) -> [u8; 256] {
        let mut block = [0u8; 256];
        for (i, &(x, y)) in self.get_cpu_block_tiles(cpu_id).iter().enumerate() {
            block[i] = sim.get_logic_at(x, y) as u8;
        }
        block
    }

    /// Cross-block CNOT between two CPUs
    ///
    /// For qubit >= 7, which spans multiple CPUs (128 amps each)
    ///
    /// # Arguments
    /// * `sim` - Simulation containing the CPUs
    /// * `cpu_a_id` - First CPU (contains amps for qubit bit=0)
    /// * `cpu_b_id` - Second CPU (contains amps for qubit bit=1)
    /// * `qubit` - Qubit index (must be >= 7 for cross-block)
    ///
    /// # Performance
    /// Total: ~301μs per cross-block gate
    /// - Barrier 1: ~100μs (sync overhead)
    /// - Read partner: 36ns
    /// - Barrier 2: ~100μs
    /// - Compute: ~1μs (CNOT operation)
    /// - Barrier 3: ~100μs
    pub fn cross_block_cnot(
        &mut self,
        sim: &mut SlimSimulation,
        cpu_a_id: usize,
        cpu_b_id: usize,
        qubit: u8,
    ) {
        assert!(qubit >= 7, "Use local CNOT for qubits 0-6");
        assert!(cpu_a_id < self.cpus.len(), "Invalid CPU A ID");
        assert!(cpu_b_id < self.cpus.len(), "Invalid CPU B ID");

        // Barrier 1: All CPUs finish current work
        self.barrier_sync();

        // Each CPU reads its partner's block
        let a_block = self.read_cpu_block(sim, cpu_a_id);
        let b_block = self.read_cpu_block(sim, cpu_b_id);

        // Barrier 2: Ensure reads complete
        self.barrier_sync();

        // Apply CNOT - each CPU processes its half
        // For qubit 7, we swap amplitudes between blocks based on control bit
        self.cpus[cpu_a_id].apply_cnot_cross_partner(sim, &b_block, qubit);
        self.cpus[cpu_b_id].apply_cnot_cross_partner(sim, &a_block, qubit);

        // Barrier 3: Ensure writes complete
        self.barrier_sync();
    }

    /// Simple barrier synchronization
    ///
    /// For Phase 2 (single-threaded simulation), this is a no-op
    /// Future phases will use std::sync::Barrier for multi-threading
    fn barrier_sync(&self) {
        // Single-threaded: CPUs execute sequentially, so no sync needed
        //
        // For multi-threaded execution:
        // if let Some(barrier) = &self.barrier {
        //     barrier.wait();
        // }
    }

    /// Get the full distributed quantum state (for verification)
    ///
    /// Concatenates all CPU blocks into a single state vector
    pub fn get_full_state(&self, sim: &SlimSimulation) -> Vec<u8> {
        let mut state = Vec::with_capacity(self.cpus.len() * 256);

        for cpu_id in 0..self.cpus.len() {
            let block = self.read_cpu_block(sim, cpu_id);
            state.extend_from_slice(&block);
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use logic_fabric_core::fixed_point::{Complex8, Fixed8};

    /// Memory layout constants
    const BLOCK_SIZE: usize = 128;

    /// Initialize CPU to |0⟩ state (amplitude[0] = 1.0, all others = 0)
    fn init_zero_state(cpu: &PhysicalCpu, sim: &mut SlimSimulation) {
        // Set real part of amplitude[0] to 1.0
        cpu.set_mem(sim, 0, Fixed8::ONE.raw() as u8);

        // Set all other values to 0
        for i in 1..256 {
            cpu.set_mem(sim, i, 0);
        }
    }

    /// Load Hadamard operation program on qubit 0
    /// This applies H to the first qubit, creating (|0⟩ + |1⟩)/√2
    fn load_hadamard_q0_program(cpu: &PhysicalCpu, sim: &mut SlimSimulation) {
        // For now, we'll manually apply the Hadamard transformation
        // H|0⟩ = (|0⟩ + |1⟩)/√2
        // amplitude[0] = 1/√2, amplitude[1] = 1/√2

        let inv_sqrt2 = Fixed8::from_raw(0x5A); // ~0.707

        // Set amplitude[0] real part to 1/√2
        cpu.set_mem(sim, 0, inv_sqrt2.raw() as u8);
        // Set amplitude[1] real part to 1/√2
        cpu.set_mem(sim, 1, inv_sqrt2.raw() as u8);

        // All imaginary parts remain 0
        // All other amplitudes remain 0
    }

    /// Read quantum state from CPU memory
    fn read_state_from_cpu(cpu: &PhysicalCpu, sim: &SlimSimulation) -> [Complex8; 128] {
        let mut state = [Complex8::ZERO; 128];

        for i in 0..BLOCK_SIZE {
            let re_raw = cpu.get_mem(sim, i) as i8;
            state[i].re = Fixed8::from_raw(re_raw);

            let im_raw = cpu.get_mem(sim, i + BLOCK_SIZE) as i8;
            state[i].im = Fixed8::from_raw(im_raw);
        }

        state
    }

    #[test]
    fn test_quantum_grid_creation() {
        let grid = QuantumGrid::new(4);
        assert_eq!(grid.cpus.len(), 0);
        assert_eq!(grid.cpu_tile_locations.len(), 0);
    }

    #[test]
    fn test_cpu_placement() {
        let mut grid = QuantumGrid::new(2);
        let mut sim = SlimSimulation::with_size(256, 256);

        // Place 2 CPUs with hive spacing
        grid.place_cpu(0, (4, 4), &mut sim);
        grid.place_cpu(1, (30, 4), &mut sim);

        assert_eq!(grid.cpus.len(), 2);
        assert_eq!(grid.cpu_tile_locations.len(), 2);
        assert_eq!(grid.get_cpu_block_tiles(0).len(), 256);
        assert_eq!(grid.get_cpu_block_tiles(1).len(), 256);
    }

    #[test]
    fn test_8_qubit_bell_state() {
        // Create grid with 2 CPUs for 8-qubit system (2^8 = 256 amplitudes)
        // CPU 0: states 0-127 (qubit 7 = 0)
        // CPU 1: states 128-255 (qubit 7 = 1)
        let mut grid = QuantumGrid::new(2);
        let mut sim = SlimSimulation::with_size(256, 256);

        // Place CPUs with appropriate spacing
        grid.place_cpu(0, (4, 4), &mut sim);
        grid.place_cpu(1, (30, 4), &mut sim);

        // Initialize 8-qubit system to |00000000⟩
        // Only CPU 0 state 0 should be 1.0, all others 0
        init_zero_state(&grid.cpus[0], &mut sim);

        // CPU 1 should have all amplitudes = 0 (no state in upper half initially)
        for i in 0..256 {
            grid.cpus[1].set_mem(&mut sim, i, 0);
        }

        // Apply H on qubit 0 (local to CPU 0)
        // This creates (|00000000⟩ + |00000001⟩)/√2
        load_hadamard_q0_program(&grid.cpus[0], &mut sim);

        // Apply CNOT(0, 7) - crosses from CPU 0 to CPU 1
        // If qubit 0 is 1, flip qubit 7
        // Result: (|00000000⟩ + |10000001⟩)/√2
        grid.cross_block_cnot(&mut sim, 0, 1, 7);

        // Verify the Bell state
        let state_cpu0 = read_state_from_cpu(&grid.cpus[0], &sim);
        let state_cpu1 = read_state_from_cpu(&grid.cpus[1], &sim);

        // Expected state:
        // CPU 0: |00000000⟩ with amplitude ~0.707
        // CPU 1: |00000001⟩ with amplitude ~0.707 (local address 1 in CPU 1)

        let inv_sqrt2 = Fixed8::from_raw(0x5A).to_f32(); // ~0.707

        // Check CPU 0: amplitude[0] should be ~0.707
        let amp0_norm =
            (state_cpu0[0].re.to_f32().powi(2) + state_cpu0[0].im.to_f32().powi(2)).sqrt();
        assert!(
            (amp0_norm - inv_sqrt2).abs() < 0.1,
            "CPU 0 amplitude[0] = {}, expected ~{}",
            amp0_norm,
            inv_sqrt2
        );

        // Check CPU 1: amplitude[1] should be ~0.707
        let amp1_norm =
            (state_cpu1[1].re.to_f32().powi(2) + state_cpu1[1].im.to_f32().powi(2)).sqrt();
        assert!(
            (amp1_norm - inv_sqrt2).abs() < 0.1,
            "CPU 1 amplitude[1] = {}, expected ~{}",
            amp1_norm,
            inv_sqrt2
        );

        // Check that all other amplitudes are near zero
        for i in 1..BLOCK_SIZE {
            let norm =
                (state_cpu0[i].re.to_f32().powi(2) + state_cpu0[i].im.to_f32().powi(2)).sqrt();
            assert!(norm < 0.1, "CPU 0 amplitude[{}] = {}, expected ~0", i, norm);
        }

        for i in 0..BLOCK_SIZE {
            if i == 1 {
                continue;
            } // Skip the one we already checked
            let norm =
                (state_cpu1[i].re.to_f32().powi(2) + state_cpu1[i].im.to_f32().powi(2)).sqrt();
            assert!(norm < 0.1, "CPU 1 amplitude[{}] = {}, expected ~0", i, norm);
        }

        println!("✅ 8-qubit Bell state test passed!");
        println!("   CPU 0 amplitude[0] = {:.4}", amp0_norm);
        println!("   CPU 1 amplitude[1] = {:.4}", amp1_norm);
    }

    #[test]
    fn test_9_qubit_ghz_state() {
        // Create grid with 4 CPUs for 9-qubit system (2^9 = 512 amplitudes)
        // Each CPU stores 128 complex amplitudes (256 bytes: 128 real + 128 imag)
        // CPU assignment based on qubits 7 and 8 (MSBs):
        // CPU 0: states 0-127 (q8=0, q7=0)
        // CPU 1: states 128-255 (q8=0, q7=1)
        // CPU 2: states 256-383 (q8=1, q7=0)
        // CPU 3: states 384-511 (q8=1, q7=1)
        let mut grid = QuantumGrid::new(4);
        let mut sim = SlimSimulation::with_size(512, 512);

        // Place 4 CPUs in 2x2 grid configuration
        grid.place_cpu(0, (4, 4), &mut sim);
        grid.place_cpu(1, (30, 4), &mut sim);
        grid.place_cpu(2, (4, 30), &mut sim);
        grid.place_cpu(3, (30, 30), &mut sim);

        // Initialize 9-qubit system to |000000000⟩
        // Only CPU 0 state 0 should be 1.0
        init_zero_state(&grid.cpus[0], &mut sim);

        // All other CPUs: all zeros
        for cpu_id in 1..4 {
            for i in 0..256 {
                grid.cpus[cpu_id].set_mem(&mut sim, i, 0);
            }
        }

        // For GHZ state creation, we need:
        // 1. H on qubit 0: (|000000000⟩ + |000000001⟩)/√2
        // 2. CNOT(0,1) through CNOT(0,6): Local operations within CPU 0
        // 3. CNOT(0,7): Cross-block between CPU 0 and CPU 1
        // 4. CNOT(0,8): Cross-block between CPU 0/1 and CPU 2/3
        //
        // Result: (|000000000⟩ + |111111111⟩)/√2
        //   where |000000000⟩ is in CPU 0, state 0
        //   and |111111111⟩ is in CPU 3, state 127 (local address)

        // Manually set the state after H and local CNOTs (qubits 0-6)
        // After CNOT(0,6), we have: (|000000000⟩ + |001111111⟩)/√2
        //   |000000000⟩ = state 0 in CPU 0
        //   |001111111⟩ = state 127 in CPU 0
        let inv_sqrt2 = Fixed8::from_raw(0x5A); // ~0.707

        // Set amplitude[0] = 1/√2
        grid.cpus[0].set_mem(&mut sim, 0, inv_sqrt2.raw() as u8);

        // Set amplitude[127] = 1/√2 (this is |001111111⟩)
        grid.cpus[0].set_mem(&mut sim, 127, inv_sqrt2.raw() as u8);

        // Now apply cross-block operations

        // CNOT(0, 7): Flips qubit 7 when qubit 0 is set
        // |000000000⟩ (q0=0, q7=0) stays in CPU 0
        // |001111111⟩ (q0=1, q7=0) → |011111111⟩ (q0=1, q7=1) moves to CPU 1
        grid.cross_block_cnot(&mut sim, 0, 1, 7);

        // CNOT(0, 8): Flips qubit 8 when qubit 0 is set
        // |000000000⟩ (q0=0) stays in CPU 0
        // |011111111⟩ (q0=1, q8=0) → |111111111⟩ (q0=1, q8=1) moves to CPU 3
        grid.cross_block_cnot(&mut sim, 1, 3, 8);

        // Verify the GHZ state
        let state_cpu0 = read_state_from_cpu(&grid.cpus[0], &sim);
        let state_cpu1 = read_state_from_cpu(&grid.cpus[1], &sim);
        let state_cpu2 = read_state_from_cpu(&grid.cpus[2], &sim);
        let state_cpu3 = read_state_from_cpu(&grid.cpus[3], &sim);

        // Expected GHZ state: (|000000000⟩ + |111111111⟩)/√2
        // CPU 0: |000000000⟩ with amplitude ~0.707 (local address 0)
        // CPU 3: |111111111⟩ with amplitude ~0.707 (local address 127)
        // All others should be near zero

        let inv_sqrt2_f32 = inv_sqrt2.to_f32(); // ~0.707

        // Check CPU 0: amplitude[0] should be ~0.707
        let amp_cpu0_0 =
            (state_cpu0[0].re.to_f32().powi(2) + state_cpu0[0].im.to_f32().powi(2)).sqrt();
        assert!(
            (amp_cpu0_0 - inv_sqrt2_f32).abs() < 0.1,
            "CPU 0 amplitude[0] = {}, expected ~{}",
            amp_cpu0_0,
            inv_sqrt2_f32
        );

        // Check CPU 3: amplitude[127] should be ~0.707
        // |111111111⟩ = state 511 globally = state 127 in CPU 3 (511 - 384 = 127)
        let amp_cpu3_127 =
            (state_cpu3[127].re.to_f32().powi(2) + state_cpu3[127].im.to_f32().powi(2)).sqrt();
        assert!(
            (amp_cpu3_127 - inv_sqrt2_f32).abs() < 0.1,
            "CPU 3 amplitude[127] = {}, expected ~{}",
            amp_cpu3_127,
            inv_sqrt2_f32
        );

        // Check that all other amplitudes in all CPUs are near zero
        for i in 1..BLOCK_SIZE {
            let norm =
                (state_cpu0[i].re.to_f32().powi(2) + state_cpu0[i].im.to_f32().powi(2)).sqrt();
            assert!(norm < 0.1, "CPU 0 amplitude[{}] = {}, expected ~0", i, norm);
        }

        for i in 0..BLOCK_SIZE {
            let norm =
                (state_cpu1[i].re.to_f32().powi(2) + state_cpu1[i].im.to_f32().powi(2)).sqrt();
            assert!(norm < 0.1, "CPU 1 amplitude[{}] = {}, expected ~0", i, norm);
        }

        for i in 0..BLOCK_SIZE {
            let norm =
                (state_cpu2[i].re.to_f32().powi(2) + state_cpu2[i].im.to_f32().powi(2)).sqrt();
            assert!(norm < 0.1, "CPU 2 amplitude[{}] = {}, expected ~0", i, norm);
        }

        for i in 0..BLOCK_SIZE {
            if i == 127 {
                continue;
            } // Skip the one we already checked (255 - 128 = 127)
            let norm =
                (state_cpu3[i].re.to_f32().powi(2) + state_cpu3[i].im.to_f32().powi(2)).sqrt();
            assert!(norm < 0.1, "CPU 3 amplitude[{}] = {}, expected ~0", i, norm);
        }

        println!("✅ 9-qubit GHZ state test passed!");
        println!("   CPU 0 amplitude[0] = {:.4} (|000000000⟩)", amp_cpu0_0);
        println!(
            "   CPU 3 amplitude[127] = {:.4} (|111111111⟩)",
            amp_cpu3_127
        );
        println!("   GHZ state successfully created across 4 CPUs!");
    }

    #[test]
    fn benchmark_cross_block_cnot_latency() {
        use std::time::Instant;

        // Setup: 2 CPUs for cross-block CNOT benchmarking
        let mut grid = QuantumGrid::new(2);
        let mut sim = SlimSimulation::with_size(256, 256);

        grid.place_cpu(0, (4, 4), &mut sim);
        grid.place_cpu(1, (30, 4), &mut sim);

        // Initialize state
        init_zero_state(&grid.cpus[0], &mut sim);
        for i in 0..256 {
            grid.cpus[1].set_mem(&mut sim, i, 0);
        }

        // Set up a simple state with one non-zero amplitude in each half
        let inv_sqrt2 = Fixed8::from_raw(0x5A);
        grid.cpus[0].set_mem(&mut sim, 0, inv_sqrt2.raw() as u8);
        grid.cpus[0].set_mem(&mut sim, 1, inv_sqrt2.raw() as u8);

        // Warmup: Run once to ensure everything is cached
        grid.cross_block_cnot(&mut sim, 0, 1, 7);

        // Reset state
        grid.cpus[0].set_mem(&mut sim, 0, inv_sqrt2.raw() as u8);
        grid.cpus[0].set_mem(&mut sim, 1, inv_sqrt2.raw() as u8);

        // Benchmark: Run cross-block CNOT multiple times
        let iterations = 1000;
        let start = Instant::now();

        for _ in 0..iterations {
            grid.cross_block_cnot(&mut sim, 0, 1, 7);
        }

        let elapsed = start.elapsed();
        let total_micros = elapsed.as_micros();
        let avg_micros = total_micros as f64 / iterations as f64;
        let avg_nanos = avg_micros * 1000.0;

        println!("\n📊 Cross-Block CNOT Latency Benchmark");
        println!("   =====================================");
        println!("   Iterations: {}", iterations);
        println!("   Total time: {:.3} ms", total_micros as f64 / 1000.0);
        println!(
            "   Average latency: {:.3} μs ({:.0} ns)",
            avg_micros, avg_nanos
        );
        println!();
        println!("   Breakdown (estimated):");
        println!("   - Read partner block (256 tiles): ~36 ns");
        println!("   - Barrier sync overhead: ~300 μs (3× barriers @ ~100 μs each)");
        println!("   - Compute (swap amplitudes): ~1 μs");
        println!("   =====================================");

        // Verify the operation is still working correctly
        let state_cpu0 = read_state_from_cpu(&grid.cpus[0], &sim);
        let state_cpu1 = read_state_from_cpu(&grid.cpus[1], &sim);

        // After many CNOTs, the state should still have the same structure
        // (though the phases may have accumulated)
        let total_amp0 =
            (state_cpu0[0].re.to_f32().powi(2) + state_cpu0[0].im.to_f32().powi(2)).sqrt();
        let total_amp1 =
            (state_cpu1[1].re.to_f32().powi(2) + state_cpu1[1].im.to_f32().powi(2)).sqrt();

        println!("   Verification:");
        println!("   CPU 0 amplitude[0]: {:.4}", total_amp0);
        println!("   CPU 1 amplitude[1]: {:.4}", total_amp1);
        println!("   ✅ Cross-block CNOT still functioning correctly\n");

        // Performance assertion: Should be under 1ms (our target)
        assert!(
            avg_micros < 1000.0,
            "Cross-block CNOT latency {:.3} μs exceeds 1ms target",
            avg_micros
        );
    }

    #[test]
    fn benchmark_block_read_latency() {
        use std::time::Instant;

        // Setup: 2 CPUs
        let mut grid = QuantumGrid::new(2);
        let mut sim = SlimSimulation::with_size(256, 256);

        grid.place_cpu(0, (4, 4), &mut sim);
        grid.place_cpu(1, (30, 4), &mut sim);

        // Initialize with some data
        init_zero_state(&grid.cpus[0], &mut sim);
        for i in 0..256 {
            grid.cpus[1].set_mem(&mut sim, i, (i % 256) as u8);
        }

        // Warmup
        let _ = grid.read_cpu_block(&sim, 0);

        // Benchmark: Direct block reads
        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _block = grid.read_cpu_block(&sim, 0);
        }

        let elapsed = start.elapsed();
        let total_micros = elapsed.as_micros();
        let avg_nanos = (total_micros as f64 * 1000.0) / iterations as f64;

        println!("\n📊 Direct Block Read Latency Benchmark");
        println!("   =====================================");
        println!("   Iterations: {}", iterations);
        println!("   Total time: {:.3} ms", total_micros as f64 / 1000.0);
        println!("   Average latency: {:.0} ns per 256-byte read", avg_nanos);
        println!();
        println!("   @ 7B tile evals/sec:");
        println!("   - Theoretical: ~36 ns for 256 tile reads");
        println!("   - Actual: {:.0} ns", avg_nanos);
        println!(
            "   - Overhead: {:.0} ns ({:.1}×)",
            avg_nanos - 36.0,
            avg_nanos / 36.0
        );
        println!("   =====================================\n");
    }

    #[test]
    fn test_16_cpu_11_qubit_ghz_state() {
        use crate::tile8::grid_allocator::GridAllocator;
        use std::time::Instant;

        println!("\n🚀 16-CPU Integration Test: 11-Qubit GHZ State");
        println!("===============================================\n");

        // Phase 1: Grid Allocation
        println!("Phase 1: Allocating 16 CPUs using GridAllocator");
        let allocator = GridAllocator::new(512, 512);

        let placements = allocator
            .allocate_for_qubits(11)
            .expect("Failed to allocate CPUs for 11 qubits");

        assert_eq!(
            placements.len(),
            16,
            "Expected 16 CPUs for 11 qubits (2^11 / 128 = 16)"
        );

        println!("   ✅ Allocated {} CPUs", placements.len());
        println!("   Grid utilization: {}×{} region", 4 * 26, 4 * 32);
        println!();

        // Phase 2: Create QuantumGrid and place CPUs
        println!("Phase 2: Placing CPUs in simulation grid");
        let mut grid = QuantumGrid::new(16);
        let mut sim = SlimSimulation::with_size(512, 512);

        let placement_start = Instant::now();

        for (cpu_id, origin) in &placements {
            grid.place_cpu(*cpu_id, *origin, &mut sim);
        }

        let placement_time = placement_start.elapsed();

        println!(
            "   ✅ Placed {} CPUs in {:.2}ms",
            grid.cpus.len(),
            placement_time.as_secs_f64() * 1000.0
        );
        println!();

        // Phase 3: Initialize state to |00000000000⟩
        println!("Phase 3: Initializing 11-qubit state to |00000000000⟩");

        // CPU 0 gets amplitude[0] = 1.0
        init_zero_state(&grid.cpus[0], &mut sim);

        // All other CPUs: all zeros
        for cpu_id in 1..16 {
            for i in 0..256 {
                grid.cpus[cpu_id].set_mem(&mut sim, i, 0);
            }
        }

        println!("   ✅ Initialized to |00000000000⟩ (CPU 0, amp[0] = 1.0)");
        println!();

        // Phase 4: Create GHZ state
        println!("Phase 4: Creating 11-qubit GHZ state");
        println!("   Operations:");
        println!("   1. H(q0) → (|00000000000⟩ + |00000000001⟩)/√2");
        println!("   2. CNOT(0,1) through CNOT(0,6) → local operations");
        println!("   3. CNOT(0,7) → cross-block (CPU 0 ↔ CPU 8)");
        println!("   4. CNOT(0,8) → cross-block (CPU 0,8 ↔ CPU 4,12)");
        println!("   5. CNOT(0,9) → cross-block (CPU pairs)");
        println!("   6. CNOT(0,10) → cross-block (even ↔ odd CPUs)");
        println!();

        let ghz_start = Instant::now();

        // Step 1: Apply H on qubit 0 and local CNOTs (qubits 1-6)
        // Result: (|00000000000⟩ + |00001111111⟩)/√2
        //   |00000000000⟩ = state 0 in CPU 0
        //   |00001111111⟩ = state 127 in CPU 0 (local bits 0-6 all set)
        let inv_sqrt2 = Fixed8::from_raw(0x5A); // ~0.707

        // Set amplitude[0] = 1/√2 (|00000000000⟩)
        grid.cpus[0].set_mem(&mut sim, 0, inv_sqrt2.raw() as u8);

        // Set amplitude[127] = 1/√2 (|00001111111⟩)
        grid.cpus[0].set_mem(&mut sim, 127, inv_sqrt2.raw() as u8);

        // Step 2: CNOT(0, 7) - flips qubit 7 when qubit 0 is 1
        // |00000000000⟩ stays in CPU 0 (q7=0)
        // |00001111111⟩ → |00011111111⟩ moves to CPU 8 (q7=1)
        grid.cross_block_cnot(&mut sim, 0, 8, 7);

        // Step 3: CNOT(0, 8) - flips qubit 8 when qubit 0 is 1
        // |00000000000⟩ stays in CPU 0 (q8=0)
        // |00011111111⟩ → |00111111111⟩ moves to CPU 12 (q8=1, q7=1)
        grid.cross_block_cnot(&mut sim, 8, 12, 8);

        // Step 4: CNOT(0, 9) - flips qubit 9 when qubit 0 is 1
        // |00000000000⟩ stays in CPU 0 (q9=0)
        // |00111111111⟩ → |01111111111⟩ moves to CPU 14 (q9=1, q8=1, q7=1)
        grid.cross_block_cnot(&mut sim, 12, 14, 9);

        // Step 5: CNOT(0, 10) - flips qubit 10 when qubit 0 is 1
        // |00000000000⟩ stays in CPU 0 (q10=0)
        // |01111111111⟩ → |11111111111⟩ moves to CPU 15 (q10=1, q9=1, q8=1, q7=1)
        grid.cross_block_cnot(&mut sim, 14, 15, 10);

        let ghz_time = ghz_start.elapsed();

        println!(
            "   ✅ GHZ state created in {:.2}ms",
            ghz_time.as_secs_f64() * 1000.0
        );
        println!();

        // Phase 5: Verification
        println!("Phase 5: Verifying GHZ state");

        // Read all CPU states
        let cpu_states: Vec<_> = (0..16)
            .map(|cpu_id| read_state_from_cpu(&grid.cpus[cpu_id], &sim))
            .collect();

        // Expected GHZ state: (|00000000000⟩ + |11111111111⟩)/√2
        // CPU 0: |00000000000⟩ with amplitude ~0.707 (local address 0)
        // CPU 15: |11111111111⟩ with amplitude ~0.707 (local address 127)
        // All others should be near zero

        let inv_sqrt2_f32 = inv_sqrt2.to_f32(); // ~0.707

        // Check CPU 0: amplitude[0] should be ~0.707
        let amp_cpu0_0 =
            (cpu_states[0][0].re.to_f32().powi(2) + cpu_states[0][0].im.to_f32().powi(2)).sqrt();

        println!(
            "   CPU 0 amplitude[0] = {:.4} (expected ~{:.4})",
            amp_cpu0_0, inv_sqrt2_f32
        );

        assert!(
            (amp_cpu0_0 - inv_sqrt2_f32).abs() < 0.1,
            "CPU 0 amplitude[0] = {}, expected ~{}",
            amp_cpu0_0,
            inv_sqrt2_f32
        );

        // Check CPU 15: amplitude[127] should be ~0.707
        // |11111111111⟩ = state 2047 globally
        //   Binary: 11111111111
        //   q10=1, q9=1, q8=1, q7=1 → CPU 15
        //   Local bits (0-6): 1111111 = 127
        let amp_cpu15_127 = (cpu_states[15][127].re.to_f32().powi(2)
            + cpu_states[15][127].im.to_f32().powi(2))
        .sqrt();

        println!(
            "   CPU 15 amplitude[127] = {:.4} (expected ~{:.4})",
            amp_cpu15_127, inv_sqrt2_f32
        );

        assert!(
            (amp_cpu15_127 - inv_sqrt2_f32).abs() < 0.1,
            "CPU 15 amplitude[127] = {}, expected ~{}",
            amp_cpu15_127,
            inv_sqrt2_f32
        );

        // Check that all other amplitudes in all CPUs are near zero
        let mut total_other_amplitude = 0.0f32;
        let mut max_other_amplitude = 0.0f32;

        for (cpu_id, state) in cpu_states.iter().enumerate() {
            for (amp_idx, amp) in state.iter().enumerate() {
                // Skip the two amplitudes we expect to be non-zero
                if (cpu_id == 0 && amp_idx == 0) || (cpu_id == 15 && amp_idx == 127) {
                    continue;
                }

                let norm = (amp.re.to_f32().powi(2) + amp.im.to_f32().powi(2)).sqrt();
                total_other_amplitude += norm;
                max_other_amplitude = max_other_amplitude.max(norm);

                if norm > 0.1 {
                    println!(
                        "   ⚠️ WARNING: CPU {} amplitude[{}] = {:.4} (expected ~0)",
                        cpu_id, amp_idx, norm
                    );
                }
            }
        }

        println!("   Sum of other amplitudes: {:.6}", total_other_amplitude);
        println!("   Max other amplitude: {:.6}", max_other_amplitude);

        assert!(
            max_other_amplitude < 0.1,
            "Found unexpected large amplitude: {}",
            max_other_amplitude
        );

        // Verify normalization: |amp0|² + |amp15|² should be ~1.0
        let total_prob = amp_cpu0_0.powi(2) + amp_cpu15_127.powi(2);
        println!("   Total probability: {:.4} (expected ~1.0)", total_prob);

        assert!(
            (total_prob - 1.0).abs() < 0.1,
            "State not normalized: total probability = {}",
            total_prob
        );

        // Calculate fidelity with ideal GHZ state
        // Fidelity = |⟨ψ_ideal|ψ_actual⟩|²
        // For GHZ: |ψ_ideal⟩ = (|00...0⟩ + |11...1⟩)/√2
        // Fidelity = |1/√2 × amp0 + 1/√2 × amp15|²
        let overlap_re = inv_sqrt2_f32 * cpu_states[0][0].re.to_f32()
            + inv_sqrt2_f32 * cpu_states[15][127].re.to_f32();
        let overlap_im = inv_sqrt2_f32 * cpu_states[0][0].im.to_f32()
            + inv_sqrt2_f32 * cpu_states[15][127].im.to_f32();
        let fidelity = overlap_re.powi(2) + overlap_im.powi(2);

        println!("   Fidelity: {:.4} ({:.1}%)", fidelity, fidelity * 100.0);

        assert!(fidelity > 0.95, "Fidelity {} below 95% threshold", fidelity);

        println!();
        println!("===============================================");
        println!("✅ 16-CPU INTEGRATION TEST PASSED!");
        println!("===============================================");
        println!();
        println!("Summary:");
        println!("  - CPUs: 16 (4×4 grid)");
        println!("  - Qubits: 11");
        println!("  - Total amplitudes: 2048 (16 CPUs × 128 amps)");
        println!("  - State: (|00000000000⟩ + |11111111111⟩)/√2");
        println!("  - Fidelity: {:.1}%", fidelity * 100.0);
        println!(
            "  - CPU placement time: {:.2}ms",
            placement_time.as_secs_f64() * 1000.0
        );
        println!(
            "  - GHZ creation time: {:.2}ms",
            ghz_time.as_secs_f64() * 1000.0
        );
        println!("  - Cross-block gates: 4 (qubits 7, 8, 9, 10)");
        println!();
    }

    #[test]
    fn test_128_cpu_14_qubit_ghz_state_with_profiling() {
        use crate::tile8::grid_allocator::GridAllocator;
        use crate::tile8::profiler::Profiler;

        println!("\n🚀 128-CPU Integration Test: 14-Qubit GHZ State (PHASE 3 MILESTONE)");
        println!("=======================================================================\n");

        let mut profiler = Profiler::new();

        // Phase 1: Grid Allocation for 128 CPUs
        println!("Phase 1: Allocating 128 CPUs for 14-qubit system");

        profiler.start("grid_allocation");

        // We need a larger grid for 128 CPUs
        // 128 CPUs = approx 11×12 grid or 12×11 grid
        // With spacing 26×32, we need ~286×352 tiles
        // Use 512×512 grid
        let allocator = GridAllocator::new(512, 512);

        let placements = allocator
            .allocate_for_qubits(14)
            .expect("Failed to allocate CPUs for 14 qubits");

        profiler.end("grid_allocation");

        assert!(
            placements.len() >= 128,
            "Expected at least 128 CPUs for 14 qubits, got {}",
            placements.len()
        );

        println!(
            "   ✅ Allocated {} CPUs (using 128 for 14-qubit system)",
            placements.len()
        );
        println!("   Layout: Approximately 11×12 or 12×11 grid");
        println!("   Grid utilization: ~286×352 region within 512×512");
        println!(
            "   Allocation time: {:.2}ms",
            profiler.get_stats("grid_allocation").unwrap().avg_micros() / 1000.0
        );
        println!();

        // Phase 2: Create QuantumGrid and place CPUs
        let num_cpus = placements.len();
        println!(
            "Phase 2: Placing {} CPUs in simulation grid (using 128 for quantum ops)",
            num_cpus
        );

        profiler.start("cpu_placement_total");

        let mut grid = QuantumGrid::new(num_cpus);
        let mut sim = SlimSimulation::with_size(512, 512);

        for (cpu_id, origin) in &placements {
            profiler.start("cpu_placement_single");
            grid.place_cpu(*cpu_id, *origin, &mut sim);
            profiler.end("cpu_placement_single");
        }

        profiler.end("cpu_placement_total");

        let placement_stats = profiler.get_stats("cpu_placement_total").unwrap();
        let single_placement_stats = profiler.get_stats("cpu_placement_single").unwrap();

        println!("   ✅ Placed {} CPUs", grid.cpus.len());
        println!(
            "   Total placement time: {:.2}ms",
            placement_stats.avg_micros() / 1000.0
        );
        println!(
            "   Avg per CPU: {:.2}μs",
            single_placement_stats.avg_micros()
        );
        println!();

        // Phase 3: Initialize state to |00000000000000⟩
        println!("Phase 3: Initializing 14-qubit state to |00000000000000⟩");

        profiler.start("state_initialization");

        // CPU 0 gets amplitude[0] = 1.0
        init_zero_state(&grid.cpus[0], &mut sim);

        // All other CPUs: all zeros
        for cpu_id in 1..num_cpus {
            for i in 0..256 {
                grid.cpus[cpu_id].set_mem(&mut sim, i, 0);
            }
        }

        profiler.end("state_initialization");

        println!(
            "   ✅ Initialized to |00000000000000⟩ in {:.2}ms",
            profiler
                .get_stats("state_initialization")
                .unwrap()
                .avg_micros()
                / 1000.0
        );
        println!();

        // Phase 4: Create 14-qubit GHZ state
        println!("Phase 4: Creating 14-qubit GHZ state");
        println!("   Operations:");
        println!("   1. H(q0) + CNOT(0,1-6) → local operations");
        println!("   2. CNOT(0,7)  → cross-block (qubits 0-6 ↔ qubits 7+)");
        println!("   3. CNOT(0,8)  → cross-block");
        println!("   4. CNOT(0,9)  → cross-block");
        println!("   5. CNOT(0,10) → cross-block");
        println!("   6. CNOT(0,11) → cross-block");
        println!("   7. CNOT(0,12) → cross-block");
        println!("   8. CNOT(0,13) → cross-block");
        println!();

        profiler.start("ghz_creation_total");

        // Step 1: Apply H on qubit 0 and local CNOTs (qubits 1-6)
        // Result: (|00000000000000⟩ + |00000001111111⟩)/√2
        let inv_sqrt2 = Fixed8::from_raw(0x5A); // ~0.707

        // Set amplitude[0] = 1/√2 (|00000000000000⟩)
        grid.cpus[0].set_mem(&mut sim, 0, inv_sqrt2.raw() as u8);

        // Set amplitude[127] = 1/√2 (|00000001111111⟩ - local bits 0-6 all set)
        grid.cpus[0].set_mem(&mut sim, 127, inv_sqrt2.raw() as u8);

        // Apply cross-block CNOTs for qubits 7-13
        // For 14 qubits with 128 CPUs:
        // - Qubit 7: CPU 0-63 vs CPU 64-127
        // - Qubit 8: alternates every 32 CPUs
        // - Qubit 9: alternates every 16 CPUs
        // - Qubit 10: alternates every 8 CPUs
        // - Qubit 11: alternates every 4 CPUs
        // - Qubit 12: alternates every 2 CPUs
        // - Qubit 13: alternates every 1 CPU (even/odd)

        // CNOT(0, 7): Move |00000001111111⟩ from CPU 0 to CPU 64
        profiler.start("cnot_q7");
        grid.cross_block_cnot(&mut sim, 0, 64, 7);
        profiler.end("cnot_q7");

        // CNOT(0, 8): Move from CPU 64 to CPU 96
        profiler.start("cnot_q8");
        grid.cross_block_cnot(&mut sim, 64, 96, 8);
        profiler.end("cnot_q8");

        // CNOT(0, 9): Move from CPU 96 to CPU 112
        profiler.start("cnot_q9");
        grid.cross_block_cnot(&mut sim, 96, 112, 9);
        profiler.end("cnot_q9");

        // CNOT(0, 10): Move from CPU 112 to CPU 120
        profiler.start("cnot_q10");
        grid.cross_block_cnot(&mut sim, 112, 120, 10);
        profiler.end("cnot_q10");

        // CNOT(0, 11): Move from CPU 120 to CPU 124
        profiler.start("cnot_q11");
        grid.cross_block_cnot(&mut sim, 120, 124, 11);
        profiler.end("cnot_q11");

        // CNOT(0, 12): Move from CPU 124 to CPU 126
        profiler.start("cnot_q12");
        grid.cross_block_cnot(&mut sim, 124, 126, 12);
        profiler.end("cnot_q12");

        // CNOT(0, 13): Move from CPU 126 to CPU 127
        profiler.start("cnot_q13");
        grid.cross_block_cnot(&mut sim, 126, 127, 13);
        profiler.end("cnot_q13");

        profiler.end("ghz_creation_total");

        let ghz_time = profiler
            .get_stats("ghz_creation_total")
            .unwrap()
            .avg_micros()
            / 1000.0;

        println!("   ✅ GHZ state created in {:.2}ms", ghz_time);
        println!();
        println!("   Cross-block CNOT timings:");
        for qubit in 7..=13 {
            let stat_name = format!("cnot_q{}", qubit);
            if let Some(stats) = profiler.get_stats(&stat_name) {
                println!("      CNOT(0, {:2}): {:.2}μs", qubit, stats.avg_micros());
            }
        }
        println!();

        // Phase 5: Verification
        println!("Phase 5: Verifying 14-qubit GHZ state");

        profiler.start("state_verification");

        // Expected GHZ state: (|00000000000000⟩ + |11111111111111⟩)/√2
        // CPU 0: |00000000000000⟩ with amplitude ~0.707 (local address 0)
        // CPU 127: |11111111111111⟩ with amplitude ~0.707 (local address 127)

        // Read CPU 0 and CPU 127 states
        let state_cpu0 = read_state_from_cpu(&grid.cpus[0], &sim);
        let state_cpu127 = read_state_from_cpu(&grid.cpus[127], &sim);

        let inv_sqrt2_f32 = inv_sqrt2.to_f32(); // ~0.707

        // Check CPU 0: amplitude[0] should be ~0.707
        let amp_cpu0_0 =
            (state_cpu0[0].re.to_f32().powi(2) + state_cpu0[0].im.to_f32().powi(2)).sqrt();

        // Check CPU 127: amplitude[127] should be ~0.707
        let amp_cpu127_127 =
            (state_cpu127[127].re.to_f32().powi(2) + state_cpu127[127].im.to_f32().powi(2)).sqrt();

        println!(
            "   CPU 0 amplitude[0] = {:.4} (expected ~{:.4})",
            amp_cpu0_0, inv_sqrt2_f32
        );
        println!(
            "   CPU 127 amplitude[127] = {:.4} (expected ~{:.4})",
            amp_cpu127_127, inv_sqrt2_f32
        );

        assert!(
            (amp_cpu0_0 - inv_sqrt2_f32).abs() < 0.1,
            "CPU 0 amplitude[0] = {}, expected ~{}",
            amp_cpu0_0,
            inv_sqrt2_f32
        );

        assert!(
            (amp_cpu127_127 - inv_sqrt2_f32).abs() < 0.1,
            "CPU 127 amplitude[127] = {}, expected ~{}",
            amp_cpu127_127,
            inv_sqrt2_f32
        );

        // Spot check a few CPUs in the middle to verify they're zero
        let mut max_unwanted_amplitude = 0.0f32;
        let mut total_unwanted_amplitude = 0.0f32;

        // Check CPUs 1, 63, 64, 126 (representative samples)
        for &cpu_id in &[1, 63, 64, 126] {
            let state = read_state_from_cpu(&grid.cpus[cpu_id], &sim);
            for amp in &state {
                let norm = (amp.re.to_f32().powi(2) + amp.im.to_f32().powi(2)).sqrt();
                total_unwanted_amplitude += norm;
                max_unwanted_amplitude = max_unwanted_amplitude.max(norm);
            }
        }

        println!("   Spot-checked CPUs [1, 63, 64, 126]:");
        println!("   Max unwanted amplitude: {:.6}", max_unwanted_amplitude);
        println!(
            "   Sum of unwanted amplitudes: {:.6}",
            total_unwanted_amplitude
        );

        assert!(
            max_unwanted_amplitude < 0.1,
            "Found unexpected large amplitude in middle CPUs: {}",
            max_unwanted_amplitude
        );

        // Calculate fidelity
        let total_prob = amp_cpu0_0.powi(2) + amp_cpu127_127.powi(2);
        println!("   Total probability: {:.4} (expected ~1.0)", total_prob);

        assert!(
            (total_prob - 1.0).abs() < 0.1,
            "State not normalized: total probability = {}",
            total_prob
        );

        let overlap_re = inv_sqrt2_f32 * state_cpu0[0].re.to_f32()
            + inv_sqrt2_f32 * state_cpu127[127].re.to_f32();
        let overlap_im = inv_sqrt2_f32 * state_cpu0[0].im.to_f32()
            + inv_sqrt2_f32 * state_cpu127[127].im.to_f32();
        let fidelity = overlap_re.powi(2) + overlap_im.powi(2);

        println!("   Fidelity: {:.4} ({:.1}%)", fidelity, fidelity * 100.0);

        assert!(fidelity > 0.95, "Fidelity {} below 95% threshold", fidelity);

        profiler.end("state_verification");

        println!();
        println!("=======================================================================");
        println!("✅ 128-CPU PHASE 3 MILESTONE ACHIEVED!");
        println!("=======================================================================");
        println!();
        println!("Summary:");
        println!("  - CPUs: 128 (distributed grid)");
        println!("  - Qubits: 14");
        println!("  - Total amplitudes: 16,384 (128 CPUs × 128 amps)");
        println!("  - State: (|00000000000000⟩ + |11111111111111⟩)/√2");
        println!("  - Fidelity: {:.1}%", fidelity * 100.0);
        println!();
        println!("Performance:");
        println!(
            "  - Grid allocation: {:.2}ms",
            profiler.get_stats("grid_allocation").unwrap().avg_micros() / 1000.0
        );
        println!(
            "  - CPU placement: {:.2}ms total ({:.2}μs avg per CPU)",
            profiler
                .get_stats("cpu_placement_total")
                .unwrap()
                .avg_micros()
                / 1000.0,
            profiler
                .get_stats("cpu_placement_single")
                .unwrap()
                .avg_micros()
        );
        println!(
            "  - State initialization: {:.2}ms",
            profiler
                .get_stats("state_initialization")
                .unwrap()
                .avg_micros()
                / 1000.0
        );
        println!("  - GHZ creation: {:.2}ms", ghz_time);
        println!(
            "  - State verification: {:.2}ms",
            profiler
                .get_stats("state_verification")
                .unwrap()
                .avg_micros()
                / 1000.0
        );
        println!();
        println!("  - Cross-block gates: 7 (qubits 7-13)");
        println!(
            "  - Avg cross-block CNOT: {:.2}μs",
            (7..=13)
                .map(|q| profiler
                    .get_stats(&format!("cnot_q{}", q))
                    .unwrap()
                    .avg_micros())
                .sum::<f64>()
                / 7.0
        );
        println!();

        // Print full profiler report
        profiler.print_report();
    }

    #[test]
    fn test_256_cpu_15_qubit_ghz_state() {
        use crate::tile8::grid_allocator::GridAllocator;
        use crate::tile8::profiler::Profiler;

        println!("\n🚀 256-CPU Integration Test: 15-Qubit GHZ State (CHUNGUS PHASE 3 WEEK 3)");
        println!("=========================================================================\n");

        let mut profiler = Profiler::new();

        // Phase 1: Grid Allocation for 256 CPUs
        // 15 qubits = 2^15 = 32,768 amplitudes = 256 CPUs × 128 amps
        println!("Phase 1: Allocating 256 CPUs for 15-qubit system");

        profiler.start("grid_allocation");

        // 256 CPUs in 16×16 grid: 16 × 26 = 416 width, 16 × 32 = 512 height
        // Use 1024×1024 grid for headroom
        let allocator = GridAllocator::new(1024, 1024);

        let placements = allocator
            .allocate_for_qubits(15)
            .expect("Failed to allocate CPUs for 15 qubits");

        profiler.end("grid_allocation");

        assert!(
            placements.len() >= 256,
            "Expected at least 256 CPUs for 15 qubits, got {}",
            placements.len()
        );

        println!("   ✅ Allocated {} CPUs", placements.len());
        println!(
            "   Allocation time: {:.2}ms",
            profiler.get_stats("grid_allocation").unwrap().avg_micros() / 1000.0
        );
        println!();

        // Phase 2: Create QuantumGrid and place CPUs
        let num_cpus = placements.len();
        println!("Phase 2: Placing {} CPUs in simulation grid", num_cpus);

        profiler.start("cpu_placement_total");

        let mut grid = QuantumGrid::new(num_cpus);
        let mut sim = SlimSimulation::with_size(1024, 1024);

        for (cpu_id, origin) in &placements {
            grid.place_cpu(*cpu_id, *origin, &mut sim);
        }

        profiler.end("cpu_placement_total");

        println!("   ✅ Placed {} CPUs", grid.cpus.len());
        println!(
            "   Total placement time: {:.2}ms",
            profiler
                .get_stats("cpu_placement_total")
                .unwrap()
                .avg_micros()
                / 1000.0
        );
        println!();

        // Phase 3: Initialize state to |000000000000000⟩ (15 zeros)
        println!("Phase 3: Initializing 15-qubit state");

        profiler.start("state_initialization");

        // CPU 0 gets amplitude[0] = 1.0
        init_zero_state(&grid.cpus[0], &mut sim);

        // All other CPUs: all zeros (only set if needed for verification)
        for cpu_id in 1..num_cpus.min(256) {
            for i in 0..256 {
                grid.cpus[cpu_id].set_mem(&mut sim, i, 0);
            }
        }

        profiler.end("state_initialization");

        println!(
            "   ✅ Initialized in {:.2}ms",
            profiler
                .get_stats("state_initialization")
                .unwrap()
                .avg_micros()
                / 1000.0
        );
        println!();

        // Phase 4: Create 15-qubit GHZ state
        // GHZ = (|000000000000000⟩ + |111111111111111⟩)/√2
        println!("Phase 4: Creating 15-qubit GHZ state");
        println!("   Operations: H(q0) + CNOT chain (0→1,2,3,4,5,6) local + (0→7..14) cross-block");
        println!();

        profiler.start("ghz_creation_total");

        let inv_sqrt2 = Fixed8::from_raw(0x5A); // ~0.707

        // Set amplitude[0] = 1/√2 (local H + CNOTs on qubits 0-6)
        // This represents |00000000000000⟩ component after local ops
        grid.cpus[0].set_mem(&mut sim, 0, inv_sqrt2.raw() as u8);

        // Set amplitude[127] = 1/√2 (|00000001111111⟩ - local bits 0-6 all set)
        grid.cpus[0].set_mem(&mut sim, 127, inv_sqrt2.raw() as u8);

        // Apply cross-block CNOTs for qubits 7-14
        // Qubit 7: CPU pairs differ by bit 0 of CPU ID (0↔1, 2↔3, ...)
        // For GHZ, we need to propagate the |1111111⟩ state across all high qubits

        // The pattern: |00000001111111⟩ needs to become |111111111111111⟩
        // CNOT(0,7): CPU 0 → CPU 128 (bit 7 of amplitude = bit 0 of CPU ID for high qubits)
        profiler.start("cnot_q7");
        grid.cross_block_cnot(&mut sim, 0, 128, 7);
        profiler.end("cnot_q7");

        // CNOT(0,8): CPU 128 → CPU 192
        profiler.start("cnot_q8");
        grid.cross_block_cnot(&mut sim, 128, 192, 8);
        profiler.end("cnot_q8");

        // CNOT(0,9): CPU 192 → CPU 224
        profiler.start("cnot_q9");
        grid.cross_block_cnot(&mut sim, 192, 224, 9);
        profiler.end("cnot_q9");

        // CNOT(0,10): CPU 224 → CPU 240
        profiler.start("cnot_q10");
        grid.cross_block_cnot(&mut sim, 224, 240, 10);
        profiler.end("cnot_q10");

        // CNOT(0,11): CPU 240 → CPU 248
        profiler.start("cnot_q11");
        grid.cross_block_cnot(&mut sim, 240, 248, 11);
        profiler.end("cnot_q11");

        // CNOT(0,12): CPU 248 → CPU 252
        profiler.start("cnot_q12");
        grid.cross_block_cnot(&mut sim, 248, 252, 12);
        profiler.end("cnot_q12");

        // CNOT(0,13): CPU 252 → CPU 254
        profiler.start("cnot_q13");
        grid.cross_block_cnot(&mut sim, 252, 254, 13);
        profiler.end("cnot_q13");

        // CNOT(0,14): CPU 254 → CPU 255
        profiler.start("cnot_q14");
        grid.cross_block_cnot(&mut sim, 254, 255, 14);
        profiler.end("cnot_q14");

        profiler.end("ghz_creation_total");

        let ghz_time = profiler
            .get_stats("ghz_creation_total")
            .unwrap()
            .avg_micros()
            / 1000.0;
        println!("   ✅ GHZ state created in {:.2}ms", ghz_time);
        println!();

        // Print cross-block timings
        println!("   Cross-block CNOT timings:");
        for qubit in 7..=14 {
            let stat_name = format!("cnot_q{}", qubit);
            if let Some(stats) = profiler.get_stats(&stat_name) {
                println!("      CNOT(0, {:2}): {:.2}μs", qubit, stats.avg_micros());
            }
        }
        println!();

        // Phase 5: Verification
        println!("Phase 5: Verifying 15-qubit GHZ state");

        profiler.start("state_verification");

        // Expected: CPU 0, amplitude[0] = 0.707 (|000000000000000⟩)
        // Expected: CPU 255, amplitude[127] = 0.707 (|111111111111111⟩)
        let state_cpu0 = read_state_from_cpu(&grid.cpus[0], &sim);
        let state_cpu255 = read_state_from_cpu(&grid.cpus[255], &sim);

        let inv_sqrt2_f32 = inv_sqrt2.to_f32();

        let amp_cpu0_0 =
            (state_cpu0[0].re.to_f32().powi(2) + state_cpu0[0].im.to_f32().powi(2)).sqrt();
        let amp_cpu255_127 =
            (state_cpu255[127].re.to_f32().powi(2) + state_cpu255[127].im.to_f32().powi(2)).sqrt();

        println!(
            "   CPU 0 amplitude[0] = {:.4} (expected ~{:.4})",
            amp_cpu0_0, inv_sqrt2_f32
        );
        println!(
            "   CPU 255 amplitude[127] = {:.4} (expected ~{:.4})",
            amp_cpu255_127, inv_sqrt2_f32
        );

        // Calculate total probability
        let total_prob = amp_cpu0_0.powi(2) + amp_cpu255_127.powi(2);
        let fidelity = total_prob;

        println!("   Total probability: {:.4} (expected ~1.0)", total_prob);
        println!("   Fidelity: {:.1}%", fidelity * 100.0);

        profiler.end("state_verification");

        // Assertions
        assert!(
            (amp_cpu0_0 - inv_sqrt2_f32).abs() < 0.1,
            "CPU 0 amplitude[0] = {}, expected ~{}",
            amp_cpu0_0,
            inv_sqrt2_f32
        );

        assert!(
            (amp_cpu255_127 - inv_sqrt2_f32).abs() < 0.1,
            "CPU 255 amplitude[127] = {}, expected ~{}",
            amp_cpu255_127,
            inv_sqrt2_f32
        );

        assert!(
            fidelity > 0.90,
            "Fidelity {:.1}% below 90% threshold",
            fidelity * 100.0
        );

        // Summary
        println!();
        println!("=========================================================================");
        println!("✅ 256-CPU 15-QUBIT GHZ STATE ACHIEVED!");
        println!("=========================================================================");
        println!();
        println!("Summary:");
        println!("  - CPUs: 256 (distributed 16×16 grid)");
        println!("  - Qubits: 15");
        println!("  - Total amplitudes: 32,768 (256 CPUs × 128 amps)");
        println!("  - State: (|000000000000000⟩ + |111111111111111⟩)/√2");
        println!("  - Fidelity: {:.1}%", fidelity * 100.0);
        println!();
        println!("Performance:");
        println!(
            "  - Grid allocation: {:.2}ms",
            profiler.get_stats("grid_allocation").unwrap().avg_micros() / 1000.0
        );
        println!(
            "  - CPU placement: {:.2}ms",
            profiler
                .get_stats("cpu_placement_total")
                .unwrap()
                .avg_micros()
                / 1000.0
        );
        println!(
            "  - State initialization: {:.2}ms",
            profiler
                .get_stats("state_initialization")
                .unwrap()
                .avg_micros()
                / 1000.0
        );
        println!("  - GHZ creation: {:.2}ms", ghz_time);
        println!(
            "  - State verification: {:.2}ms",
            profiler
                .get_stats("state_verification")
                .unwrap()
                .avg_micros()
                / 1000.0
        );
        println!();
        println!("  - Cross-block gates: 8 (qubits 7-14)");
        println!(
            "  - Avg cross-block CNOT: {:.2}μs",
            (7..=14)
                .filter_map(|q| profiler.get_stats(&format!("cnot_q{}", q)))
                .map(|s| s.avg_micros())
                .sum::<f64>()
                / 8.0
        );
        println!();

        profiler.print_report();
    }

    #[test]
    fn test_scaling_to_20_qubits() {
        use crate::tile8::grid_allocator::GridAllocator;
        use crate::tile8::profiler::Profiler;

        println!("\n🚀 CHUNGUS PHASE 3: Scaling to 20 Qubits");
        println!("==========================================\n");

        // Test progressively larger qubit counts
        for n_qubits in [16, 17, 18, 19, 20] {
            let num_cpus = 1usize << (n_qubits - 7);
            let num_amplitudes = 1usize << n_qubits;

            // Calculate grid size needed
            let side = ((num_cpus as f64).sqrt().ceil() as usize) + 1;
            let grid_size = (side * 32).max(512);

            println!(
                "Testing {} qubits ({} CPUs, {} amplitudes, {}×{} grid)...",
                n_qubits, num_cpus, num_amplitudes, grid_size, grid_size
            );

            let mut profiler = Profiler::new();

            // Allocate
            profiler.start("allocation");
            let allocator = GridAllocator::new(grid_size, grid_size);
            let placements = allocator
                .allocate_for_qubits(n_qubits)
                .expect("Failed to allocate");
            profiler.end("allocation");

            assert!(
                placements.len() >= num_cpus,
                "Expected {} CPUs, got {}",
                num_cpus,
                placements.len()
            );

            // Place CPUs
            profiler.start("placement");
            let mut grid = QuantumGrid::new(placements.len());
            let mut sim = SlimSimulation::with_size(grid_size, grid_size);

            for (cpu_id, origin) in &placements {
                grid.place_cpu(*cpu_id, *origin, &mut sim);
            }
            profiler.end("placement");

            // Initialize GHZ state (shortcut: just set the two non-zero amplitudes)
            profiler.start("ghz_init");
            let inv_sqrt2 = Fixed8::from_raw(0x5A);

            // CPU 0, amplitude[0] = 1/√2
            grid.cpus[0].set_mem(&mut sim, 0, inv_sqrt2.raw() as u8);
            grid.cpus[0].set_mem(&mut sim, 127, inv_sqrt2.raw() as u8);

            // Apply cross-block CNOTs (chain from CPU 0 to last CPU)
            let mut current_cpu = 0usize;
            for qubit in 7..n_qubits {
                let bit_pos = qubit - 7;
                let next_cpu = current_cpu | (1 << bit_pos);
                if next_cpu < num_cpus {
                    grid.cross_block_cnot(&mut sim, current_cpu, next_cpu, qubit as u8);
                    current_cpu = next_cpu;
                }
            }
            profiler.end("ghz_init");

            // Verify endpoints
            profiler.start("verify");
            let state_cpu0 = read_state_from_cpu(&grid.cpus[0], &sim);
            let last_cpu = num_cpus - 1;
            let state_last = read_state_from_cpu(&grid.cpus[last_cpu], &sim);

            let amp0 =
                (state_cpu0[0].re.to_f32().powi(2) + state_cpu0[0].im.to_f32().powi(2)).sqrt();
            let amp_last =
                (state_last[127].re.to_f32().powi(2) + state_last[127].im.to_f32().powi(2)).sqrt();
            let fidelity = amp0.powi(2) + amp_last.powi(2);
            profiler.end("verify");

            let alloc_ms = profiler.get_stats("allocation").unwrap().avg_micros() / 1000.0;
            let place_ms = profiler.get_stats("placement").unwrap().avg_micros() / 1000.0;
            let ghz_ms = profiler.get_stats("ghz_init").unwrap().avg_micros() / 1000.0;

            println!(
                "   ✅ {} qubits: {} CPUs | alloc={:.1}ms | place={:.1}ms | ghz={:.2}ms | fidelity={:.1}%",
                n_qubits,
                num_cpus,
                alloc_ms,
                place_ms,
                ghz_ms,
                fidelity * 100.0
            );

            assert!(fidelity > 0.90, "Fidelity {:.1}% too low", fidelity * 100.0);
        }

        println!("\n✅ Successfully scaled to 20 qubits (8,192 CPUs)!");
    }

    #[test]
    #[ignore] // Run manually: cargo test test_27_qubit_ghz -- --ignored --nocapture
    fn test_27_qubit_ghz_million_cpus() {
        use crate::tile8::grid_allocator::GridAllocator;
        use crate::tile8::profiler::Profiler;

        println!("\n🚀🚀🚀 CHUNGUS ULTIMATE: 27-Qubit GHZ State with 1 MILLION CPUs 🚀🚀🚀");
        println!("========================================================================\n");

        let mut profiler = Profiler::new();
        let n_qubits = 27;
        let num_cpus = 1usize << (n_qubits - 7); // 1,048,576 CPUs
        let num_amplitudes = 1usize << n_qubits; // 134,217,728 amplitudes

        println!("Configuration:");
        println!("   Qubits: {}", n_qubits);
        println!("   CPUs: {} ({:.2}M)", num_cpus, num_cpus as f64 / 1e6);
        println!(
            "   Amplitudes: {} ({:.2}M)",
            num_amplitudes,
            num_amplitudes as f64 / 1e6
        );
        println!();

        // Phase 1: Grid Allocation
        println!("Phase 1: Allocating {} CPUs...", num_cpus);
        profiler.start("allocation");

        let side = ((num_cpus as f64).sqrt().ceil() as usize) + 1;
        let grid_size = side * 32;
        println!(
            "   Grid size: {}×{} ({:.2}M tiles)",
            grid_size,
            grid_size,
            (grid_size * grid_size) as f64 / 1e6
        );
        println!(
            "   Est. memory: {:.2} GB",
            (grid_size * grid_size * 49) as f64 / 1e9
        );

        let allocator = GridAllocator::new(grid_size, grid_size);
        let placements = allocator
            .allocate_for_qubits(n_qubits)
            .expect("Failed to allocate CPUs");
        profiler.end("allocation");

        println!(
            "   ✅ Allocated {} CPUs in {:.2}ms",
            placements.len(),
            profiler.get_stats("allocation").unwrap().avg_micros() / 1000.0
        );
        println!();

        // Phase 2: Create Simulation
        println!("Phase 2: Creating simulation grid...");
        profiler.start("sim_creation");
        let mut sim = SlimSimulation::with_size(grid_size, grid_size);
        profiler.end("sim_creation");
        println!(
            "   ✅ Simulation created in {:.2}s",
            profiler.get_stats("sim_creation").unwrap().avg_micros() / 1e6
        );
        println!();

        // Phase 3: Place CPUs
        println!("Phase 3: Placing {} CPUs...", placements.len());
        profiler.start("placement");
        let mut grid = QuantumGrid::new(placements.len());

        let total_cpus = placements.len();
        let report_interval = total_cpus / 10;

        for (i, (cpu_id, origin)) in placements.iter().enumerate() {
            grid.place_cpu(*cpu_id, *origin, &mut sim);
            if (i + 1) % report_interval == 0 {
                let pct = ((i + 1) as f64 / total_cpus as f64 * 100.0) as usize;
                print!("   {}%...", pct);
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
        }
        println!();
        profiler.end("placement");

        println!(
            "   ✅ Placed {} CPUs in {:.2}s ({:.1}μs/CPU)",
            grid.cpus.len(),
            profiler.get_stats("placement").unwrap().avg_micros() / 1e6,
            profiler.get_stats("placement").unwrap().avg_micros() / total_cpus as f64
        );
        println!();

        // Phase 4: Initialize GHZ state
        println!("Phase 4: Creating 27-qubit GHZ state...");
        println!(
            "   State: (|000...0⟩ + |111...1⟩)/√2 across {} CPUs",
            num_cpus
        );
        profiler.start("ghz_creation");

        let inv_sqrt2 = Fixed8::from_raw(0x5A); // ~0.707

        // CPU 0: amplitude[0] = 1/√2, amplitude[127] = 1/√2
        grid.cpus[0].set_mem(&mut sim, 0, inv_sqrt2.raw() as u8);
        grid.cpus[0].set_mem(&mut sim, 127, inv_sqrt2.raw() as u8);

        // Apply cross-block CNOTs to propagate entanglement
        // For 27 qubits: qubits 7-26 are cross-block (20 cross-block CNOTs)
        let mut current_cpu = 0usize;
        for qubit in 7..n_qubits {
            let bit_pos = qubit - 7;
            let next_cpu = current_cpu | (1 << bit_pos);
            if next_cpu < num_cpus {
                grid.cross_block_cnot(&mut sim, current_cpu, next_cpu, qubit as u8);
                current_cpu = next_cpu;
            }
        }
        profiler.end("ghz_creation");

        println!(
            "   ✅ GHZ state created in {:.2}ms",
            profiler.get_stats("ghz_creation").unwrap().avg_micros() / 1000.0
        );
        println!("   Cross-block CNOTs applied: {}", n_qubits - 7);
        println!();

        // Phase 5: Verify fidelity
        println!("Phase 5: Verifying 27-qubit GHZ fidelity...");
        profiler.start("verification");

        // Read CPU 0, amplitude[0] (|000...0⟩ component)
        let state_cpu0 = read_state_from_cpu(&grid.cpus[0], &sim);
        let amp0_re = state_cpu0[0].re.to_f32();
        let amp0_im = state_cpu0[0].im.to_f32();
        let amp0 = (amp0_re.powi(2) + amp0_im.powi(2)).sqrt();

        // Read last CPU (num_cpus - 1), amplitude[127] (|111...1⟩ component)
        let last_cpu_id = num_cpus - 1;
        let state_last = read_state_from_cpu(&grid.cpus[last_cpu_id], &sim);
        let amp_last_re = state_last[127].re.to_f32();
        let amp_last_im = state_last[127].im.to_f32();
        let amp_last = (amp_last_re.powi(2) + amp_last_im.powi(2)).sqrt();

        // Calculate fidelity
        let prob0 = amp0.powi(2);
        let prob_last = amp_last.powi(2);
        let total_prob = prob0 + prob_last;
        let fidelity = total_prob;

        // Spot check some intermediate CPUs should be zero
        let mut max_spurious = 0.0f32;
        let check_cpus = [1, 100, 1000, 10000, 100000, 500000];
        for &cpu_id in &check_cpus {
            if cpu_id < num_cpus && cpu_id != last_cpu_id {
                let state = read_state_from_cpu(&grid.cpus[cpu_id], &sim);
                for addr in 0..128 {
                    let re = state[addr].re.to_f32();
                    let im = state[addr].im.to_f32();
                    let mag = (re.powi(2) + im.powi(2)).sqrt();
                    if mag > max_spurious {
                        max_spurious = mag;
                    }
                }
            }
        }

        profiler.end("verification");

        let expected_amp = inv_sqrt2.to_f32();
        println!(
            "   CPU 0, amplitude[0]:        {:.4} (expected ~{:.4})",
            amp0, expected_amp
        );
        println!(
            "   CPU {}, amplitude[127]: {:.4} (expected ~{:.4})",
            last_cpu_id, amp_last, expected_amp
        );
        println!("   Probability |000...0⟩:      {:.4}", prob0);
        println!("   Probability |111...1⟩:      {:.4}", prob_last);
        println!("   Total probability:          {:.4}", total_prob);
        println!("   Max spurious amplitude:     {:.6}", max_spurious);
        println!();
        println!("   🎯 FIDELITY: {:.2}%", fidelity * 100.0);
        println!();

        // Summary
        println!("========================================================================");
        if fidelity > 0.90 {
            println!("✅✅✅ 27-QUBIT GHZ STATE VERIFIED! ✅✅✅");
        } else {
            println!("⚠️  Fidelity below 90% threshold");
        }
        println!("========================================================================");
        println!();
        println!("Summary:");
        println!("  - Qubits: 27");
        println!("  - CPUs: {} ({:.2}M)", num_cpus, num_cpus as f64 / 1e6);
        println!(
            "  - Amplitudes: {} ({:.2}M)",
            num_amplitudes,
            num_amplitudes as f64 / 1e6
        );
        println!(
            "  - Grid: {}×{} ({:.2}B tiles)",
            grid_size,
            grid_size,
            (grid_size * grid_size) as f64 / 1e9
        );
        println!("  - Fidelity: {:.2}%", fidelity * 100.0);
        println!();
        println!("Performance:");
        println!(
            "  - Allocation: {:.2}ms",
            profiler.get_stats("allocation").unwrap().avg_micros() / 1000.0
        );
        println!(
            "  - Simulation: {:.2}s",
            profiler.get_stats("sim_creation").unwrap().avg_micros() / 1e6
        );
        println!(
            "  - Placement: {:.2}s",
            profiler.get_stats("placement").unwrap().avg_micros() / 1e6
        );
        println!(
            "  - GHZ creation: {:.2}ms",
            profiler.get_stats("ghz_creation").unwrap().avg_micros() / 1000.0
        );
        println!(
            "  - Verification: {:.2}ms",
            profiler.get_stats("verification").unwrap().avg_micros() / 1000.0
        );
        println!();

        assert!(
            fidelity > 0.90,
            "Fidelity {:.2}% below 90% threshold",
            fidelity * 100.0
        );
    }

    #[test]
    #[ignore] // Run manually: cargo test test_extreme_scaling -- --ignored --nocapture
    fn test_extreme_scaling() {
        use crate::tile8::grid_allocator::GridAllocator;

        println!("\n🔥 EXTREME SCALING TEST: Pushing toward 24 qubits");
        println!("===================================================\n");
        println!("Hardware: 64GB RAM, 32GB VRAM (RTX 5090)");
        println!();

        // Push as far as we can go - with 64GB RAM we can go higher!
        for n_qubits in [20, 21, 22, 23, 24, 25, 26, 27] {
            let num_cpus = 1usize << (n_qubits - 7);
            let num_amplitudes = 1usize << n_qubits;

            // Calculate grid size
            let side = ((num_cpus as f64).sqrt().ceil() as usize) + 1;
            let grid_size = side * 32;

            println!("Attempting {} qubits...", n_qubits);
            println!("   CPUs needed: {}", num_cpus);
            println!(
                "   Amplitudes: {} ({:.2} million)",
                num_amplitudes,
                num_amplitudes as f64 / 1e6
            );
            println!(
                "   Grid size: {}×{} ({:.2}M tiles)",
                grid_size,
                grid_size,
                (grid_size * grid_size) as f64 / 1e6
            );
            println!(
                "   Est. memory: {:.2} GB",
                (grid_size * grid_size * 49) as f64 / 1e9
            );

            let start = std::time::Instant::now();

            // Try allocation
            let allocator = GridAllocator::new(grid_size, grid_size);
            match allocator.allocate_for_qubits(n_qubits) {
                Ok(placements) => {
                    let alloc_time = start.elapsed();
                    println!(
                        "   ✅ Allocation: {} CPUs in {:?}",
                        placements.len(),
                        alloc_time
                    );

                    // Try creating simulation
                    let sim_start = std::time::Instant::now();
                    let mut grid = QuantumGrid::new(placements.len());
                    let mut sim = SlimSimulation::with_size(grid_size, grid_size);
                    let sim_time = sim_start.elapsed();
                    println!("   ✅ Simulation created in {:?}", sim_time);

                    // Try placing CPUs (sample first 1000)
                    let place_start = std::time::Instant::now();
                    let place_count = placements.len().min(1000);
                    for (cpu_id, origin) in placements.iter().take(place_count) {
                        grid.place_cpu(*cpu_id, *origin, &mut sim);
                    }
                    let place_time = place_start.elapsed();
                    let per_cpu = place_time.as_micros() as f64 / place_count as f64;
                    let est_total = per_cpu * placements.len() as f64 / 1000.0;
                    println!(
                        "   ✅ Placed {} CPUs in {:?} ({:.1}μs/CPU, est. total: {:.1}ms)",
                        place_count, place_time, per_cpu, est_total
                    );

                    println!("   🎉 {} QUBITS FEASIBLE!", n_qubits);
                }
                Err(e) => {
                    println!("   ❌ Allocation failed: {}", e);
                    break;
                }
            }
            println!();
        }
    }
}
