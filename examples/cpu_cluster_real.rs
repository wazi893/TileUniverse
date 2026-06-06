//! Real TileCPU Cluster - 2x2 grid of actual tile-based processors
//!
//! This integrates the real TileCpu (with PC, ROM, ALU, registers) into
//! the cluster architecture. Each CPU executes programs via tile propagation.
//!
//! Run: cargo run --example cpu_cluster_real

use engine::simulation::Simulation;
use engine::tile_cpu::{TileCpu, TileCpuBuilder};
use engine::tile_meta::TileType;

/// Direction for inter-CPU communication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// A node in the CPU cluster containing a real TileCpu
pub struct ClusterNode {
    /// Node ID (0-3 for 2x2)
    pub id: usize,
    /// Grid position (0,0), (1,0), (0,1), (1,1)
    pub grid_pos: (usize, usize),
    /// The actual tile-based CPU
    pub cpu: TileCpu,
    /// Mailbox output tile indices [N, S, E, W]
    pub mailbox_out: [usize; 4],
    /// Mailbox input tile indices [N, S, E, W]
    pub mailbox_in: [usize; 4],
}

/// 2x2 cluster of real TileCPUs
pub struct RealCpuCluster {
    pub nodes: Vec<ClusterNode>,
    pub grid_width: usize,
    pub cpu_spacing: usize,
    pub total_ticks: u64,
}

impl RealCpuCluster {
    /// Build a 2x2 cluster with real TileCPUs
    ///
    /// Each CPU gets a simple test program that can be customized.
    pub fn build(sim: &mut Simulation, programs: &[&[u8]; 4]) -> Self {
        let grid_width = sim.width();
        // CPU dimensions: ~32 wide x ~52 tall, so use 64 spacing
        let cpu_spacing = 64;

        let mut nodes = Vec::with_capacity(4);

        for grid_y in 0..2 {
            for grid_x in 0..2 {
                let id = grid_y * 2 + grid_x;
                let origin = (20 + grid_x * cpu_spacing, 20 + grid_y * cpu_spacing);

                // Build the real TileCpu with full wiring (includes clock)
                let cpu = TileCpuBuilder::new()
                    .with_origin(origin.0, origin.1)
                    .with_program(programs[id])
                    .with_rom_size(16)
                    .with_ram_size(16)
                    .build(sim);

                // Add mailbox tiles around the CPU
                let mailboxes = Self::add_mailboxes(sim, origin, cpu_spacing, grid_width);

                nodes.push(ClusterNode {
                    id,
                    grid_pos: (grid_x, grid_y),
                    cpu,
                    mailbox_out: mailboxes.0,
                    mailbox_in: mailboxes.1,
                });
            }
        }

        // Wire connections between adjacent CPUs
        Self::wire_connections(sim, &nodes, cpu_spacing);

        RealCpuCluster {
            nodes,
            grid_width,
            cpu_spacing,
            total_ticks: 0,
        }
    }

    /// Add mailbox tiles around a CPU
    fn add_mailboxes(
        sim: &mut Simulation,
        origin: (usize, usize),
        _spacing: usize,
        grid_width: usize,
    ) -> ([usize; 4], [usize; 4]) {
        let (ox, oy) = origin;
        let cpu_width = 32;
        let cpu_height = 50;

        // Mailbox outputs (Const tiles) - positioned at CPU boundary
        let out_positions = [
            (ox + cpu_width / 2, oy.saturating_sub(5)),  // North out
            (ox + cpu_width / 2, oy + cpu_height + 5),   // South out
            (ox + cpu_width + 5, oy + cpu_height / 2),   // East out
            (ox.saturating_sub(5), oy + cpu_height / 2), // West out
        ];

        let mut mailbox_out = [0usize; 4];
        for (i, &(mx, my)) in out_positions.iter().enumerate() {
            if mx > 0 && my > 0 && mx < grid_width && my < sim.height() {
                sim.set_tile(mx, my, TileType::Const);
                sim.set_logic_value(mx, my, 0);
                mailbox_out[i] = my * grid_width + mx;
            }
        }

        // Mailbox inputs (Wire tiles) - positioned to receive from neighbors
        let in_positions = [
            (ox + cpu_width / 2, oy.saturating_sub(2)),  // North in
            (ox + cpu_width / 2, oy + cpu_height + 2),   // South in
            (ox + cpu_width + 2, oy + cpu_height / 2),   // East in
            (ox.saturating_sub(2), oy + cpu_height / 2), // West in
        ];

        let mut mailbox_in = [0usize; 4];
        for (i, &(mx, my)) in in_positions.iter().enumerate() {
            if mx > 0 && my > 0 && mx < grid_width && my < sim.height() {
                sim.set_tile(mx, my, TileType::Wire);
                sim.set_logic_value(mx, my, 0);
                mailbox_in[i] = my * grid_width + mx;
            }
        }

        (mailbox_out, mailbox_in)
    }

    /// Wire connections between adjacent CPUs
    fn wire_connections(sim: &mut Simulation, nodes: &[ClusterNode], _spacing: usize) {
        let grid_width = sim.width();

        // Connection pairs: (from_node, from_dir, to_node, to_dir)
        let pairs = [
            (0, Direction::East, 1, Direction::West),
            (1, Direction::West, 0, Direction::East),
            (0, Direction::South, 2, Direction::North),
            (2, Direction::North, 0, Direction::South),
            (1, Direction::South, 3, Direction::North),
            (3, Direction::North, 1, Direction::South),
            (2, Direction::East, 3, Direction::West),
            (3, Direction::West, 2, Direction::East),
        ];

        for (from_node, from_dir, to_node, to_dir) in pairs {
            let from_idx = nodes[from_node].mailbox_out[from_dir.index()];
            let to_idx = nodes[to_node].mailbox_in[to_dir.index()];

            if from_idx == 0 || to_idx == 0 {
                continue;
            }

            let (from_x, from_y) = (from_idx % grid_width, from_idx / grid_width);
            let (to_x, to_y) = (to_idx % grid_width, to_idx / grid_width);

            // Place wires between them
            if from_y == to_y {
                let (start_x, end_x) = if from_x < to_x {
                    (from_x, to_x)
                } else {
                    (to_x, from_x)
                };
                for x in (start_x + 1)..end_x {
                    sim.set_tile(x, from_y, TileType::WireH);
                }
            } else if from_x == to_x {
                let (start_y, end_y) = if from_y < to_y {
                    (from_y, to_y)
                } else {
                    (to_y, from_y)
                };
                for y in (start_y + 1)..end_y {
                    sim.set_tile(from_x, y, TileType::WireV);
                }
            }
        }
    }

    /// Execute one cycle on a specific CPU using its built-in tick
    pub fn tick_cpu(
        &mut self,
        sim: &mut Simulation,
        cpu_id: usize,
    ) -> engine::simulation::TimingStats {
        let node = &self.nodes[cpu_id];
        node.cpu.tick(sim)
    }

    /// Execute one cycle on ALL CPUs
    pub fn tick_all(&mut self, sim: &mut Simulation) {
        // For now, just tick each CPU sequentially
        // In a real implementation, we'd batch these
        for node in &self.nodes {
            let _ = node.cpu.tick(sim);
        }
        self.total_ticks += 1;
    }

    /// Run sparse propagation (evaluates only dirty tiles)
    pub fn propagate(&mut self, sim: &mut Simulation) {
        let mut batch: Vec<u32> = Vec::new();
        sim.dirty.fill_into(&mut batch);

        let width = sim.width();
        let height = sim.height();

        for &idx32 in batch.iter() {
            let idx = idx32 as usize;
            let x = idx % width;
            let y = idx / width;

            let tile = sim.tilemap.get_tile(x, y);
            let tt = tile.map(|t| t.meta.tile_type).unwrap_or(TileType::Wire);

            match tt {
                TileType::WireH | TileType::WireV | TileType::Wire => {
                    let mut val = 0u64;

                    if x > 0 {
                        let v = sim.get_logic_at(x - 1, y);
                        if v != 0 {
                            val = v;
                        }
                    }
                    if x + 1 < width {
                        let v = sim.get_logic_at(x + 1, y);
                        if v != 0 {
                            val = v;
                        }
                    }
                    if y > 0 {
                        let v = sim.get_logic_at(x, y - 1);
                        if v != 0 {
                            val = v;
                        }
                    }
                    if y + 1 < height {
                        let v = sim.get_logic_at(x, y + 1);
                        if v != 0 {
                            val = v;
                        }
                    }

                    if val != 0 {
                        let old_val = sim.get_logic_at(x, y);
                        if old_val != val {
                            sim.set_logic_value(x, y, val);
                            if x > 0 {
                                sim.dirty.mark_dirty(y * width + (x - 1));
                            }
                            if x + 1 < width {
                                sim.dirty.mark_dirty(y * width + (x + 1));
                            }
                            if y > 0 {
                                sim.dirty.mark_dirty((y - 1) * width + x);
                            }
                            if y + 1 < height {
                                sim.dirty.mark_dirty((y + 1) * width + x);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        self.total_ticks += 1;
    }

    /// Send a value to a neighbor via mailbox
    pub fn send(&self, sim: &mut Simulation, node_id: usize, direction: Direction, value: u8) {
        let mailbox_idx = self.nodes[node_id].mailbox_out[direction.index()];
        let width = sim.width();
        let height = sim.height();

        sim.set_logic_value_by_idx(mailbox_idx, value as u64);
        sim.dirty.mark_dirty(mailbox_idx);

        // Mark neighbors dirty
        let x = mailbox_idx % width;
        let y = mailbox_idx / width;
        if x > 0 {
            sim.dirty.mark_dirty(y * width + (x - 1));
        }
        if x + 1 < width {
            sim.dirty.mark_dirty(y * width + (x + 1));
        }
        if y > 0 {
            sim.dirty.mark_dirty((y - 1) * width + x);
        }
        if y + 1 < height {
            sim.dirty.mark_dirty((y + 1) * width + x);
        }
    }

    /// Receive a value from a neighbor
    pub fn recv(&self, sim: &Simulation, node_id: usize, from_direction: Direction) -> u8 {
        let mailbox_idx = self.nodes[node_id].mailbox_in[from_direction.index()];
        sim.get_logic_value_by_idx(mailbox_idx) as u8
    }

    /// Get dirty tile count for sparse eval stats
    pub fn dirty_count(&self, sim: &Simulation) -> usize {
        let mut count = 0;
        for word in sim.dirty.segments.iter() {
            let w = word.get();
            count += w.count_ones() as usize;
        }
        count
    }

    /// Dump cluster state
    pub fn dump_state(&self, sim: &Simulation) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "=== Real CPU Cluster State (tick {}) ===\n",
            self.total_ticks
        ));

        for node in &self.nodes {
            let pc = node.cpu.read_pc(sim);
            let r0 = node.cpu.read_reg(sim, 0);
            let r1 = node.cpu.read_reg(sim, 1);
            let r2 = node.cpu.read_reg(sim, 2);
            let r3 = node.cpu.read_reg(sim, 3);

            s.push_str(&format!(
                "CPU[{}] @ ({},{}) | PC={:02X} R0={} R1={} R2={} R3={} | tiles={}\n",
                node.id, node.grid_pos.0, node.grid_pos.1, pc, r0, r1, r2, r3, node.cpu.tile_count
            ));

            // Mailbox state
            let in_n = sim.get_logic_value_by_idx(node.mailbox_in[0]);
            let in_s = sim.get_logic_value_by_idx(node.mailbox_in[1]);
            let in_e = sim.get_logic_value_by_idx(node.mailbox_in[2]);
            let in_w = sim.get_logic_value_by_idx(node.mailbox_in[3]);
            s.push_str(&format!(
                "         mailbox_in[N={} S={} E={} W={}]\n",
                in_n, in_s, in_e, in_w
            ));
        }

        s
    }
}

fn main() {
    println!("=== Real TileCPU Cluster 2x2 ===\n");

    // Create large enough simulation grid (2x2 CPUs at 64 spacing = 128+40 = 168)
    let mut sim = Simulation::with_size(256, 256);

    // Programs for each CPU:
    // CPU 0: Load 10 into R0
    // CPU 1: Load 20 into R0
    // CPU 2: Load 30 into R0
    // CPU 3: Load 40 into R0
    //
    // Instruction format: 0xRV where R=register (0-3), V=value (0-15)
    // LDI R0, #10 = 0x1A (opcode 1, reg 0, imm 10... simplified)
    // Actually using the format from minimal.rs:
    // 0x12 = LDI R0, #2 (opcode=1, rd=0, imm=2)

    let programs: [&[u8]; 4] = [
        &[0x12, 0x00, 0x00, 0x00], // CPU 0: LDI R0, #2; NOP; NOP; NOP
        &[0x17, 0x00, 0x00, 0x00], // CPU 1: LDI R0, #7; NOP; NOP; NOP
        &[0x1A, 0x00, 0x00, 0x00], // CPU 2: LDI R0, #10; NOP; NOP; NOP
        &[0x1F, 0x00, 0x00, 0x00], // CPU 3: LDI R0, #15; NOP; NOP; NOP
    ];

    // Build the cluster
    let mut cluster = RealCpuCluster::build(&mut sim, &programs);

    // Initialize simulation for timing-aware execution
    sim.initialize();

    // Show initial state
    println!("{}", cluster.dump_state(&sim));

    // Calculate total tiles used
    let total_cpu_tiles: usize = cluster.nodes.iter().map(|n| n.cpu.tile_count).sum();
    println!(
        "Total CPU tiles: {} ({} per CPU)",
        total_cpu_tiles,
        total_cpu_tiles / 4
    );
    println!(
        "Grid size: {}x{} = {} tiles",
        sim.width(),
        sim.height(),
        sim.width() * sim.height()
    );
    println!(
        "CPU tile density: {:.2}%\n",
        100.0 * total_cpu_tiles as f64 / (sim.width() * sim.height()) as f64
    );

    // Test 1: CPU Execution via tile simulation
    println!("--- Test 1: CPU Execution ---");
    println!("Running 3 clock cycles on each CPU...\n");

    for cycle in 0..3 {
        println!("Cycle {}:", cycle);
        for node in &cluster.nodes {
            let pc_before = node.cpu.read_pc(&sim);
            let r0_before = node.cpu.read_reg(&sim, 0);

            let stats = node.cpu.tick(&mut sim);

            let pc_after = node.cpu.read_pc(&sim);
            let r0_after = node.cpu.read_reg(&sim, 0);

            println!(
                "  CPU[{}]: PC {:02X}->{:02X}, R0 {}->{}, deltas={}, conv={}",
                node.id,
                pc_before,
                pc_after,
                r0_before,
                r0_after,
                stats.total_deltas,
                stats.converged
            );
        }
        println!();
    }

    println!("After 3 cycles:");
    println!("{}", cluster.dump_state(&sim));

    // Test 2: Message passing between CPUs
    println!("--- Test 2: Message passing ---");

    cluster.send(&mut sim, 0, Direction::East, 42);
    println!("CPU[0] sent 42 East");

    // Propagate
    for _ in 0..20 {
        cluster.propagate(&mut sim);
    }

    let received = cluster.recv(&sim, 1, Direction::West);
    println!("CPU[1] received {} from West\n", received);

    // Test 3: Sparse eval performance
    println!("--- Test 3: Sparse evaluation performance ---");

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        cluster.propagate(&mut sim);
    }
    let elapsed = start.elapsed();
    println!(
        "1000 propagations: {:?} ({:.0} prop/sec)",
        elapsed,
        1000.0 / elapsed.as_secs_f64()
    );

    // Test with message sending
    let start = std::time::Instant::now();
    for i in 0..1000u64 {
        cluster.send(&mut sim, 0, Direction::East, (i & 0xFF) as u8);
        cluster.propagate(&mut sim);
    }
    let elapsed = start.elapsed();
    println!(
        "1000 send+propagate: {:?} ({:.0} cycles/sec)",
        elapsed,
        1000.0 / elapsed.as_secs_f64()
    );

    // Show dirty tile stats
    let dirty = cluster.dirty_count(&sim);
    println!(
        "Current dirty tiles: {} (sparse eval only touches these)\n",
        dirty
    );

    // Final state
    println!("{}", cluster.dump_state(&sim));

    println!("=== Integration Complete ===");
    println!("\nWhat this demonstrates:");
    println!("  - 4 real TileCPUs (PC, ROM, ALU, registers) on shared grid");
    println!(
        "  - Each CPU built from {} actual tiles",
        total_cpu_tiles / 4
    );
    println!("  - Mailbox tiles for inter-CPU communication");
    println!("  - Sparse evaluation: only dirty tiles processed");
    println!("\nThe datacenter vision:");
    println!("  - Scale to 100s of CPUs on larger grid");
    println!("  - Idle CPUs have no dirty tiles = zero cost");
    println!("  - Active CPUs propagate via sparse eval");
    println!("  - Kubernetes schedules 'programs' to CPUs");
}
