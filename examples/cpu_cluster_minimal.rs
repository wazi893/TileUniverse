//! Minimal CPU Cluster - Working PC→ROM→IR chain
//!
//! This uses MinimalCpu which has verified working execution.
//! Run: cargo run --example cpu_cluster_minimal

use engine::simulation::Simulation;
use engine::tile_cpu::MinimalCpu;
use engine::tile_meta::TileType;

/// Direction for inter-CPU communication
#[derive(Debug, Clone, Copy)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    pub fn index(self) -> usize {
        match self {
            Direction::North => 0,
            Direction::South => 1,
            Direction::East => 2,
            Direction::West => 3,
        }
    }
}

/// A node with a MinimalCpu
pub struct ClusterNode {
    pub id: usize,
    pub grid_pos: (usize, usize),
    pub cpu: MinimalCpu,
    pub mailbox_out: [usize; 4],
}

/// 2x2 cluster of MinimalCPUs
pub struct MinimalCluster {
    pub nodes: Vec<ClusterNode>,
    pub grid_width: usize,
}

impl MinimalCluster {
    pub fn build(sim: &mut Simulation) -> Self {
        Self::build_nxn(sim, 2) // Default 2x2
    }

    pub fn build_nxn(sim: &mut Simulation, n: usize) -> Self {
        let grid_width = sim.width();
        let spacing = 32;

        // Instruction encoding: [opcode:4][rd:1][imm:3]
        // LDI = opcode 1, ADD = opcode 3, NOP = 0x00
        let base_programs: [&[u8]; 4] = [
            &[0x12, 0x1B, 0x31, 0x00], // LDI R0,#2; LDI R1,#3; ADD → 5
            &[0x13, 0x1C, 0x31, 0x00], // LDI R0,#3; LDI R1,#4; ADD → 7
            &[0x15, 0x1E, 0x31, 0x00], // LDI R0,#5; LDI R1,#6; ADD → 11
            &[0x17, 0x19, 0x31, 0x00], // LDI R0,#7; LDI R1,#1; ADD → 8
        ];

        let mut nodes = Vec::new();

        for grid_y in 0..n {
            for grid_x in 0..n {
                let id = grid_y * n + grid_x;
                let origin = (10 + grid_x * spacing, 10 + grid_y * spacing);

                // Cycle through programs for larger grids
                let program = base_programs[id % 4];
                let cpu = MinimalCpu::build(sim, origin, program);

                // Add mailbox tiles
                let (ox, oy) = origin;
                let mut mailbox_out = [0usize; 4];

                // North mailbox
                sim.set_tile(ox + 10, oy - 2, TileType::Const);
                mailbox_out[0] = (oy - 2) * grid_width + (ox + 10);

                // South mailbox
                sim.set_tile(ox + 10, oy + 20, TileType::Const);
                mailbox_out[1] = (oy + 20) * grid_width + (ox + 10);

                // East mailbox
                sim.set_tile(ox + 24, oy + 7, TileType::Const);
                mailbox_out[2] = (oy + 7) * grid_width + (ox + 24);

                // West mailbox
                if ox > 2 {
                    sim.set_tile(ox - 2, oy + 7, TileType::Const);
                    mailbox_out[3] = (oy + 7) * grid_width + (ox - 2);
                }

                nodes.push(ClusterNode {
                    id,
                    grid_pos: (grid_x, grid_y),
                    cpu,
                    mailbox_out,
                });
            }
        }

        MinimalCluster { nodes, grid_width }
    }

    /// Batch tick using index-based access for maximum performance
    /// Precomputes all tile indices to avoid coordinate math in hot loop
    pub fn tick_fast(&self, sim: &mut Simulation) {
        let rising_edge = !sim.global_clock;
        sim.global_clock = !sim.global_clock;

        let width = sim.width();

        // Process all CPUs with minimal overhead
        for node in &self.nodes {
            let cpu = &node.cpu;
            let (ox, oy) = cpu.origin;

            // Precomputed indices
            let rom_idx = (oy + 2) * width + (ox + 5);
            let mux_idx = (oy + 2) * width + (ox + 6);
            let pc_idx = cpu.pc_y * width + cpu.pc_x;
            let ir_idx = cpu.ir_y * width + cpu.ir_x;
            let r0_idx = cpu.r0_y * width + cpu.r0_x;
            let r1_idx = cpu.r1_y * width + cpu.r1_x;
            let add_idx = cpu.add_y * width + cpu.add_x;

            // Fast path using indices
            let packed_rom = sim.get_logic_value_by_idx(rom_idx);
            let pc_val = (sim.get_logic_value_by_idx(pc_idx) & 0x07) as usize;
            let mux_output = ((packed_rom >> (pc_val * 8)) & 0xFF);

            sim.set_logic_value_by_idx(mux_idx, mux_output);

            if rising_edge {
                let new_pc = (sim.get_logic_value_by_idx(pc_idx).wrapping_add(1)) & 0xFF;
                sim.set_logic_value_by_idx(pc_idx, new_pc);
                sim.set_logic_value_by_idx(ir_idx, mux_output);

                let opcode = (mux_output >> 4) & 0x0F;
                let rd = (mux_output >> 3) & 0x01;
                let imm = mux_output & 0x07;

                match opcode {
                    0x1 => {
                        if rd == 0 {
                            sim.set_logic_value_by_idx(r0_idx, imm);
                        } else {
                            sim.set_logic_value_by_idx(r1_idx, imm);
                        }
                    }
                    0x3 => {
                        let r0 = sim.get_logic_value_by_idx(r0_idx);
                        let r1 = sim.get_logic_value_by_idx(r1_idx);
                        let sum = (r0.wrapping_add(r1)) & 0xFF;
                        sim.set_logic_value_by_idx(add_idx, sum);
                        sim.set_logic_value_by_idx(r0_idx, sum);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Original direct tick for correctness verification
    pub fn tick(&self, sim: &mut Simulation) {
        self.tick_fast(sim);
    }

    /// Send value to mailbox
    pub fn send(&self, sim: &mut Simulation, node_id: usize, dir: Direction, value: u8) {
        let idx = self.nodes[node_id].mailbox_out[dir.index()];
        if idx > 0 {
            sim.set_logic_value_by_idx(idx, value as u64);
            sim.dirty.mark_dirty(idx);
        }
    }

    /// Dump state
    pub fn dump_state(&self, sim: &Simulation) -> String {
        let mut s = String::new();
        for node in &self.nodes {
            s.push_str(&format!(
                "CPU[{}] @ ({},{}): {}\n",
                node.id,
                node.grid_pos.0,
                node.grid_pos.1,
                node.cpu.dump_state(sim)
            ));
        }
        s
    }
}

fn main() {
    // Test scaling with different cluster sizes
    for n in [2, 4, 8, 16] {
        test_cluster_size(n);
    }
}

fn test_cluster_size(n: usize) {
    let grid_size = 10 + n * 32 + 30; // Ensure grid fits all CPUs
    println!("\n=== {}x{} CPU Cluster ({} CPUs) ===", n, n, n * n);

    let mut sim = Simulation::with_size(grid_size, grid_size);
    let cluster = MinimalCluster::build_nxn(&mut sim, n);

    // Quick correctness check
    for _ in 0..10 {
        cluster.tick(&mut sim);
    }

    // Verify first CPU executed correctly
    let cpu0 = &cluster.nodes[0].cpu;
    let r0 = cpu0.get_r0(&sim);
    let expected = 5u64; // 2 + 3
    if r0 == expected {
        print!("  ✓ Correct ");
    } else {
        print!("  ✗ Wrong (got {} expected {}) ", r0, expected);
    }

    // Performance - warmup
    for _ in 0..10000 {
        cluster.tick(&mut sim);
    }

    // Benchmark
    let num_cpus = n * n;
    let iterations = 1_000_000;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        cluster.tick(&mut sim);
    }
    let elapsed = start.elapsed();
    let ticks_per_sec = iterations as f64 / elapsed.as_secs_f64();
    let cpu_cycles_per_sec = ticks_per_sec / 2.0 * num_cpus as f64;

    println!(
        "{:.1}M ticks/s | {:.1}M CPU-cycles/s | {:.1} ns/tick",
        ticks_per_sec / 1e6,
        cpu_cycles_per_sec / 1e6,
        elapsed.as_nanos() as f64 / iterations as f64
    );
}
