//! Quantum Scheduler - Real-World QAOA Application
//!
//! A practical scheduling tool that uses quantum-inspired optimization (QAOA)
//! to solve constraint-based scheduling problems.
//!
//! USE CASES:
//! - Meeting scheduling: "These people can't meet at the same time"
//! - Resource allocation: "These tasks need the same machine"
//! - Class timetabling: "These courses have the same students"
//! - Event planning: "These sessions overlap"
//!
//! HOW IT WORKS:
//! 1. Model conflicts as a graph (nodes = tasks, edges = conflicts)
//! 2. QAOA finds the optimal partition (minimum time slots)
//! 3. Visualize the solution as a color-coded schedule
//!
//! Run with: cargo run --example quantum_scheduler --features visualizer --release

use egui_macroquad::egui;
use macroquad::prelude::*;
use std::collections::HashSet;

// Import QAOA from engine
use engine::algorithms::qaoa::{Graph, QAOAConfig, brute_force_maxcut, run_qaoa};

// =============================================================================
// Constants
// =============================================================================

const NODE_RADIUS: f32 = 30.0;
const SLOT_COLORS: [Color; 8] = [
    Color::new(0.2, 0.6, 0.9, 1.0), // Blue
    Color::new(0.9, 0.4, 0.3, 1.0), // Red
    Color::new(0.3, 0.8, 0.4, 1.0), // Green
    Color::new(0.9, 0.7, 0.2, 1.0), // Yellow
    Color::new(0.7, 0.4, 0.9, 1.0), // Purple
    Color::new(0.3, 0.8, 0.8, 1.0), // Cyan
    Color::new(0.9, 0.5, 0.7, 1.0), // Pink
    Color::new(0.6, 0.6, 0.6, 1.0), // Gray
];

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Clone)]
struct Task {
    id: usize,
    name: String,
    pos: Vec2,
    time_slot: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Conflict(usize, usize);

#[derive(Clone)]
struct SolverStats {
    optimal_cut: f64,
    qaoa_cut: f64,
    ratio: f64,
    time_ms: f64,
    iterations: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum ExampleMode {
    Scheduling, // Meetings, Classes, Events
    Portfolio,  // Stock portfolio optimization
    Routing,    // Truck/delivery route optimization
    Warehouse,  // Warehouse/facility placement
    Cluster,    // Datacenter/cluster workload scheduling
    QEC,        // Quantum Error Correction verification
}

struct Scheduler {
    tasks: Vec<Task>,
    conflicts: HashSet<Conflict>,
    selected_task: Option<usize>,
    dragging: Option<usize>,
    connecting_from: Option<usize>,
    next_task_name: String,
    solved: bool,
    num_time_slots: usize,
    qaoa_stats: Option<SolverStats>,
    mode: ExampleMode,
}

impl Scheduler {
    fn new() -> Self {
        let mut scheduler = Self {
            tasks: Vec::new(),
            conflicts: HashSet::new(),
            selected_task: None,
            dragging: None,
            connecting_from: None,
            next_task_name: String::new(),
            solved: false,
            num_time_slots: 0,
            qaoa_stats: None,
            mode: ExampleMode::Scheduling,
        };
        scheduler.load_meeting_example();
        scheduler
    }

    fn add_task(&mut self, name: &str, pos: Vec2) -> usize {
        let id = self.tasks.len();
        self.tasks.push(Task {
            id,
            name: name.to_string(),
            pos,
            time_slot: None,
        });
        self.solved = false;
        id
    }

    fn add_conflict(&mut self, a: usize, b: usize) {
        if a != b && a < self.tasks.len() && b < self.tasks.len() {
            let conflict = if a < b {
                Conflict(a, b)
            } else {
                Conflict(b, a)
            };
            self.conflicts.insert(conflict);
            self.solved = false;
        }
    }

    fn remove_task(&mut self, id: usize) {
        if id >= self.tasks.len() {
            return;
        }
        self.tasks.remove(id);
        for (new_id, task) in self.tasks.iter_mut().enumerate() {
            task.id = new_id;
        }
        let new_conflicts: HashSet<Conflict> = self
            .conflicts
            .iter()
            .filter(|c| c.0 != id && c.1 != id)
            .map(|c| {
                let a = if c.0 > id { c.0 - 1 } else { c.0 };
                let b = if c.1 > id { c.1 - 1 } else { c.1 };
                if a < b {
                    Conflict(a, b)
                } else {
                    Conflict(b, a)
                }
            })
            .collect();
        self.conflicts = new_conflicts;
        self.solved = false;
        self.selected_task = None;
    }

    fn clear(&mut self) {
        self.tasks.clear();
        self.conflicts.clear();
        self.solved = false;
        self.num_time_slots = 0;
        self.qaoa_stats = None;
        self.selected_task = None;
        // Don't reset mode here - let each loader set it
    }

    fn solve_with_qaoa(&mut self) {
        if self.tasks.is_empty() {
            return;
        }

        let n = self.tasks.len();

        if n == 1 {
            self.tasks[0].time_slot = Some(0);
            self.num_time_slots = 1;
            self.solved = true;
            self.qaoa_stats = Some(SolverStats {
                optimal_cut: 0.0,
                qaoa_cut: 0.0,
                ratio: 1.0,
                time_ms: 0.0,
                iterations: 0,
            });
            return;
        }

        // Build graph from conflicts
        let mut graph = Graph::new(n);
        for conflict in &self.conflicts {
            graph.add_unweighted_edge(conflict.0, conflict.1);
        }

        if n > 12 {
            self.solve_greedy();
            return;
        }

        let start = std::time::Instant::now();
        let (optimal, _) = brute_force_maxcut(&graph);

        let result = run_qaoa(
            graph.clone(),
            QAOAConfig::with_depth(3).with_shots(500).with_seed(42),
        );

        let elapsed = start.elapsed();

        // Assign time slots based on partition
        for (i, task) in self.tasks.iter_mut().enumerate() {
            task.time_slot = Some(if result.best_bitstring[i] { 1 } else { 0 });
        }

        // Check if 2 colors suffice
        let mut valid = true;
        for conflict in &self.conflicts {
            if self.tasks[conflict.0].time_slot == self.tasks[conflict.1].time_slot {
                valid = false;
                break;
            }
        }

        if !valid {
            self.solve_greedy();
        } else {
            self.num_time_slots = 2;
            self.solved = true;
            self.qaoa_stats = Some(SolverStats {
                optimal_cut: optimal,
                qaoa_cut: result.best_cut,
                ratio: result.approximation_ratio.unwrap_or(0.0),
                time_ms: elapsed.as_secs_f64() * 1000.0,
                iterations: result.iterations,
            });
        }
    }

    fn solve_greedy(&mut self) {
        let n = self.tasks.len();
        let mut colors: Vec<Option<usize>> = vec![None; n];
        let mut max_color = 0;

        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for conflict in &self.conflicts {
            adj[conflict.0].insert(conflict.1);
            adj[conflict.1].insert(conflict.0);
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(adj[i].len()));

        for &node in &order {
            let used: HashSet<usize> = adj[node]
                .iter()
                .filter_map(|&neighbor| colors[neighbor])
                .collect();

            let mut color = 0;
            while used.contains(&color) {
                color += 1;
            }

            colors[node] = Some(color);
            max_color = max_color.max(color);
        }

        for (i, task) in self.tasks.iter_mut().enumerate() {
            task.time_slot = colors[i];
        }

        self.num_time_slots = max_color + 1;
        self.solved = true;
        self.qaoa_stats = Some(SolverStats {
            optimal_cut: 0.0,
            qaoa_cut: 0.0,
            ratio: 1.0,
            time_ms: 0.0,
            iterations: 0,
        });
    }

    fn load_meeting_example(&mut self) {
        self.clear();
        self.mode = ExampleMode::Scheduling;
        let m1 = self.add_task("Team Standup", vec2(200.0, 150.0));
        let m2 = self.add_task("1:1 Alice", vec2(400.0, 150.0));
        let m3 = self.add_task("1:1 Bob", vec2(600.0, 150.0));
        let m4 = self.add_task("Design Review", vec2(200.0, 350.0));
        let m5 = self.add_task("Sprint Planning", vec2(400.0, 350.0));
        let m6 = self.add_task("Client Call", vec2(600.0, 350.0));

        self.add_conflict(m1, m2);
        self.add_conflict(m1, m3);
        self.add_conflict(m2, m4);
        self.add_conflict(m3, m5);
        self.add_conflict(m4, m5);
        self.add_conflict(m5, m6);
    }

    fn load_classroom_example(&mut self) {
        self.clear();
        self.mode = ExampleMode::Scheduling;
        let c1 = self.add_task("Math 101", vec2(150.0, 100.0));
        let c2 = self.add_task("Physics", vec2(350.0, 100.0));
        let c3 = self.add_task("Chemistry", vec2(550.0, 100.0));
        let c4 = self.add_task("CS 101", vec2(750.0, 100.0));
        let c5 = self.add_task("English", vec2(150.0, 300.0));
        let c6 = self.add_task("History", vec2(350.0, 300.0));
        let c7 = self.add_task("Art", vec2(550.0, 300.0));
        let c8 = self.add_task("Music", vec2(750.0, 300.0));

        self.add_conflict(c1, c2);
        self.add_conflict(c2, c3);
        self.add_conflict(c1, c3);
        self.add_conflict(c1, c4);
        self.add_conflict(c5, c6);
        self.add_conflict(c4, c7);
        self.add_conflict(c3, c8);
    }

    fn load_event_example(&mut self) {
        self.clear();
        self.mode = ExampleMode::Scheduling;
        let _s1 = self.add_task("Keynote", vec2(400.0, 80.0));
        let s2 = self.add_task("AI Workshop", vec2(200.0, 180.0));
        let s3 = self.add_task("ML Panel", vec2(400.0, 180.0));
        let s4 = self.add_task("Cloud Talk", vec2(600.0, 180.0));
        let s5 = self.add_task("Security", vec2(150.0, 300.0));
        let s6 = self.add_task("DevOps", vec2(350.0, 300.0));
        let s7 = self.add_task("Frontend", vec2(550.0, 300.0));
        let s8 = self.add_task("Backend", vec2(750.0, 300.0));
        let s9 = self.add_task("Networking", vec2(300.0, 420.0));
        let s10 = self.add_task("Closing", vec2(500.0, 420.0));

        self.add_conflict(s2, s3);
        self.add_conflict(s4, s5);
        self.add_conflict(s6, s7);
        self.add_conflict(s3, s8);
        self.add_conflict(s5, s8);
        self.add_conflict(s9, s10);
        self.add_conflict(s2, s9);
    }

    /// Portfolio Optimization Example
    ///
    /// Nodes = Assets/Stocks
    /// Edges = High correlation (assets that move together)
    ///
    /// Goal: Partition into two groups to diversify
    /// - Group 1 (Blue): Assets to BUY
    /// - Group 2 (Red): Assets to HOLD/SELL
    ///
    /// QAOA finds partitions that maximize "cuts" across correlated pairs,
    /// meaning you don't hold too many correlated assets together.
    fn load_portfolio_example(&mut self) {
        self.clear();
        self.mode = ExampleMode::Portfolio;

        // Tech stocks (highly correlated with each other)
        let aapl = self.add_task("AAPL", vec2(150.0, 120.0));
        let msft = self.add_task("MSFT", vec2(300.0, 120.0));
        let googl = self.add_task("GOOGL", vec2(450.0, 120.0));
        let nvda = self.add_task("NVDA", vec2(600.0, 120.0));

        // Energy stocks (correlated with each other)
        let xom = self.add_task("XOM", vec2(150.0, 280.0));
        let cvx = self.add_task("CVX", vec2(300.0, 280.0));

        // Financial stocks
        let jpm = self.add_task("JPM", vec2(450.0, 280.0));
        let gs = self.add_task("GS", vec2(600.0, 280.0));

        // Consumer/Retail
        let amzn = self.add_task("AMZN", vec2(225.0, 420.0));
        let wmt = self.add_task("WMT", vec2(375.0, 420.0));

        // Bonds/Safe haven (anti-correlated with stocks)
        let tlt = self.add_task("TLT", vec2(525.0, 420.0));
        let gld = self.add_task("GLD", vec2(675.0, 420.0));

        // Tech stocks are highly correlated
        self.add_conflict(aapl, msft);
        self.add_conflict(aapl, googl);
        self.add_conflict(msft, googl);
        self.add_conflict(googl, nvda);
        self.add_conflict(msft, nvda);

        // Energy stocks correlated
        self.add_conflict(xom, cvx);

        // Financial stocks correlated
        self.add_conflict(jpm, gs);

        // AMZN correlates with both tech and retail
        self.add_conflict(amzn, aapl);
        self.add_conflict(amzn, msft);
        self.add_conflict(amzn, wmt);

        // Safe havens are anti-correlated with risk assets
        // (In our model, "conflict" means we want them in different groups,
        //  which is actually what we want for diversification!)
        self.add_conflict(tlt, jpm); // Bonds vs banks
        self.add_conflict(gld, nvda); // Gold vs growth tech
    }

    /// Truck Routing / Delivery Optimization Example
    ///
    /// Nodes = Delivery stops (customers, warehouses, stores)
    /// Edges = Stops in DIFFERENT zones (we want to separate them by truck)
    ///
    /// Goal: Partition stops into routes for 2 trucks
    /// - Truck 1 (Blue): North/West zone
    /// - Truck 2 (Red): South/East zone
    ///
    /// QAOA maximizes "cuts" - keeping distant stops on different trucks,
    /// so each truck serves a geographically coherent cluster.
    fn load_routing_example(&mut self) {
        self.clear();
        self.mode = ExampleMode::Routing;

        // Depot (starting point)
        let depot = self.add_task("🏭 Depot", vec2(400.0, 280.0));

        // North zone stops
        let n1 = self.add_task("N1: Mall", vec2(300.0, 80.0));
        let n2 = self.add_task("N2: Hospital", vec2(450.0, 100.0));
        let n3 = self.add_task("N3: School", vec2(550.0, 60.0));

        // South zone stops
        let s1 = self.add_task("S1: Airport", vec2(250.0, 480.0));
        let s2 = self.add_task("S2: Factory", vec2(400.0, 520.0));
        let s3 = self.add_task("S3: Port", vec2(550.0, 460.0));

        // West zone stops
        let w1 = self.add_task("W1: Store A", vec2(80.0, 200.0));
        let w2 = self.add_task("W2: Store B", vec2(100.0, 350.0));

        // East zone stops
        let e1 = self.add_task("E1: Office", vec2(700.0, 180.0));
        let e2 = self.add_task("E2: Warehouse", vec2(720.0, 380.0));

        // Cross-zone edges: we want to SEPARATE these (put on different trucks)
        // North vs South
        self.add_conflict(n1, s1);
        self.add_conflict(n1, s2);
        self.add_conflict(n2, s2);
        self.add_conflict(n2, s3);
        self.add_conflict(n3, s3);

        // West vs East
        self.add_conflict(w1, e1);
        self.add_conflict(w1, e2);
        self.add_conflict(w2, e1);
        self.add_conflict(w2, e2);

        // North-West cluster vs South-East cluster
        self.add_conflict(n1, e1);
        self.add_conflict(n1, s3);
        self.add_conflict(w1, s1);
        self.add_conflict(w2, s2);

        // Depot connects to both zones (neutral)
        // Not adding depot conflicts - it can go either way
        let _ = depot; // Depot is visually present but not constrained
    }

    /// Warehouse / Facility Placement Example
    ///
    /// Nodes = Customer regions that need to be served
    /// Edges = Regions that should be served by DIFFERENT warehouses
    ///         (to balance load or because they're in opposite directions)
    ///
    /// Goal: Assign regions to 2 warehouses
    /// - Warehouse A (Blue): Western regions
    /// - Warehouse B (Red): Eastern regions
    ///
    /// QAOA maximizes "cuts" - separating distant/opposite regions
    /// so each warehouse serves a coherent service area.
    fn load_warehouse_example(&mut self) {
        self.clear();
        self.mode = ExampleMode::Warehouse;

        // Two warehouse candidates (shown but not in optimization)
        let wh_west = self.add_task("📦 WH-West", vec2(150.0, 280.0));
        let wh_east = self.add_task("📦 WH-East", vec2(650.0, 280.0));

        // Western customer regions
        let cw1 = self.add_task("Downtown", vec2(100.0, 150.0));
        let cw2 = self.add_task("Westside", vec2(50.0, 280.0));
        let cw3 = self.add_task("Southwest", vec2(80.0, 420.0));
        let cw4 = self.add_task("Midtown", vec2(250.0, 200.0));

        // Central regions (could go either way)
        let cc1 = self.add_task("Central", vec2(400.0, 180.0));
        let cc2 = self.add_task("Uptown", vec2(400.0, 380.0));

        // Eastern customer regions
        let ce1 = self.add_task("Eastside", vec2(550.0, 150.0));
        let ce2 = self.add_task("Harbor", vec2(720.0, 200.0));
        let ce3 = self.add_task("Airport", vec2(700.0, 380.0));
        let ce4 = self.add_task("Southeast", vec2(600.0, 450.0));

        // West vs East separation (these should NOT be served by same warehouse)
        self.add_conflict(cw1, ce1);
        self.add_conflict(cw1, ce2);
        self.add_conflict(cw2, ce1);
        self.add_conflict(cw2, ce2);
        self.add_conflict(cw3, ce3);
        self.add_conflict(cw3, ce4);
        self.add_conflict(cw4, ce1);
        self.add_conflict(cw4, ce4);

        // Southwest vs Northeast diagonal
        self.add_conflict(cw3, ce2);
        self.add_conflict(cw2, ce4);

        // Central regions create tension (can swing either way)
        self.add_conflict(cc1, cw3); // Central shouldn't go with far Southwest
        self.add_conflict(cc1, ce4); // Central shouldn't go with far Southeast
        self.add_conflict(cc2, ce2); // Uptown shouldn't go with far Harbor

        // Warehouses are visual markers, not optimized
        let _ = (wh_west, wh_east);
    }

    /// Datacenter Cluster Scheduler Example
    ///
    /// Nodes = Workloads (pods, containers, VMs, jobs)
    /// Edges = Anti-affinity constraints (workloads that should NOT share a server)
    ///
    /// Real-world scenarios modeled:
    /// - Fault tolerance: DB replicas on different servers
    /// - Resource isolation: GPU jobs shouldn't compete
    /// - Blast radius: Critical services spread across failure domains
    ///
    /// QAOA partitions workloads into Cluster A vs Cluster B,
    /// maximizing separation of anti-affinity pairs.
    fn load_cluster_example(&mut self) {
        self.clear();
        self.mode = ExampleMode::Cluster;

        // Server nodes (visual markers)
        let _srv_a = self.add_task("🖥️ Rack-A", vec2(200.0, 280.0));
        let _srv_b = self.add_task("🖥️ Rack-B", vec2(600.0, 280.0));

        // Database replicas (MUST be on different servers for fault tolerance)
        let db1 = self.add_task("postgres-1", vec2(100.0, 100.0));
        let db2 = self.add_task("postgres-2", vec2(250.0, 100.0));
        let db3 = self.add_task("postgres-3", vec2(400.0, 100.0));

        // Redis replicas
        let redis1 = self.add_task("redis-master", vec2(550.0, 100.0));
        let redis2 = self.add_task("redis-replica", vec2(700.0, 100.0));

        // API service replicas
        let api1 = self.add_task("api-v1-a", vec2(100.0, 450.0));
        let api2 = self.add_task("api-v1-b", vec2(250.0, 450.0));
        let api3 = self.add_task("api-v1-c", vec2(400.0, 450.0));

        // ML/GPU workloads (compete for GPU resources)
        let ml1 = self.add_task("ml-train", vec2(550.0, 450.0));
        let ml2 = self.add_task("ml-inference", vec2(700.0, 450.0));

        // Background jobs
        let job1 = self.add_task("cron-backup", vec2(150.0, 280.0));
        let job2 = self.add_task("etl-pipeline", vec2(650.0, 280.0));

        // === Anti-affinity constraints ===

        // Database replicas: spread across servers (fault tolerance)
        self.add_conflict(db1, db2);
        self.add_conflict(db2, db3);
        self.add_conflict(db1, db3);

        // Redis master/replica: different servers
        self.add_conflict(redis1, redis2);

        // API replicas: spread for availability
        self.add_conflict(api1, api2);
        self.add_conflict(api2, api3);
        self.add_conflict(api1, api3);

        // GPU workloads: don't compete for same GPU
        self.add_conflict(ml1, ml2);

        // Critical path isolation: DB and API shouldn't all fail together
        self.add_conflict(db1, api1);
        self.add_conflict(db2, api2);

        // Background jobs shouldn't impact primary services
        self.add_conflict(job1, db1);
        self.add_conflict(job2, redis1);

        // ML training shouldn't starve API inference
        self.add_conflict(ml1, api3);
    }

    /// Quantum Error Correction (QEC) Verification Example
    ///
    /// Models a distance-3 surface code patch for syndrome extraction scheduling.
    ///
    /// In QEC:
    /// - Physical qubits are arranged in a 2D grid
    /// - "Stabilizers" are measured to detect errors without destroying data
    /// - X-stabilizers detect bit-flip errors (Z errors)
    /// - Z-stabilizers detect phase-flip errors (X errors)
    ///
    /// The problem: Stabilizers that share qubits can't be measured simultaneously.
    /// This is a graph coloring problem - exactly what QAOA solves!
    ///
    /// QAOA partitions stabilizers into Round 1 / Round 2 measurement groups.
    fn load_qec_example(&mut self) {
        self.clear();
        self.mode = ExampleMode::QEC;

        // === Distance-3 Surface Code Layout ===
        //
        //     D1 ---- Z1 ---- D2
        //     |       |       |
        //     X1 ---- D5 ---- X2
        //     |       |       |
        //     D3 ---- Z2 ---- D4
        //
        // D = Data qubits (hold the logical information)
        // X = X-stabilizers (detect Z errors)
        // Z = Z-stabilizers (detect X errors)

        // Data qubits (visual markers - the actual quantum data)
        let _d1 = self.add_task("D₁", vec2(150.0, 100.0));
        let _d2 = self.add_task("D₂", vec2(450.0, 100.0));
        let _d3 = self.add_task("D₃", vec2(150.0, 400.0));
        let _d4 = self.add_task("D₄", vec2(450.0, 400.0));
        let _d5 = self.add_task("D₅", vec2(300.0, 250.0)); // Center data qubit

        // X-stabilizers (measure in X basis, detect Z errors)
        let x1 = self.add_task("X₁-stab", vec2(100.0, 250.0));
        let x2 = self.add_task("X₂-stab", vec2(500.0, 250.0));

        // Z-stabilizers (measure in Z basis, detect X errors)
        let z1 = self.add_task("Z₁-stab", vec2(300.0, 100.0));
        let z2 = self.add_task("Z₂-stab", vec2(300.0, 400.0));

        // Additional stabilizers for a more complete code
        let x3 = self.add_task("X₃-stab", vec2(600.0, 150.0));
        let x4 = self.add_task("X₄-stab", vec2(600.0, 350.0));
        let z3 = self.add_task("Z₃-stab", vec2(700.0, 250.0));

        // Boundary stabilizers
        let b1 = self.add_task("B₁-stab", vec2(50.0, 150.0));
        let b2 = self.add_task("B₂-stab", vec2(50.0, 350.0));

        // === Measurement Conflicts ===
        // Stabilizers that share a data qubit can't be measured simultaneously
        // (measuring one would disturb the shared qubit before the other reads it)

        // X1 and Z1 share D1 and D5
        self.add_conflict(x1, z1);

        // X1 and Z2 share D3 and D5
        self.add_conflict(x1, z2);

        // X2 and Z1 share D2 and D5
        self.add_conflict(x2, z1);

        // X2 and Z2 share D4 and D5
        self.add_conflict(x2, z2);

        // Z1 and Z2 share D5 (center qubit)
        self.add_conflict(z1, z2);

        // X1 and X2 share D5 (center qubit)
        self.add_conflict(x1, x2);

        // Extended code conflicts
        self.add_conflict(x2, x3);
        self.add_conflict(x2, x4);
        self.add_conflict(x3, z3);
        self.add_conflict(x4, z3);
        self.add_conflict(x3, x4);

        // Boundary stabilizer conflicts
        self.add_conflict(b1, x1);
        self.add_conflict(b1, z1);
        self.add_conflict(b2, x1);
        self.add_conflict(b2, z2);
    }

    fn handle_input(&mut self) {
        let mouse_pos = vec2(mouse_position().0, mouse_position().1);

        let hovered = self
            .tasks
            .iter()
            .position(|t| (t.pos - mouse_pos).length() < NODE_RADIUS);

        if is_mouse_button_pressed(MouseButton::Left) {
            if let Some(id) = hovered {
                if is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift) {
                    self.connecting_from = Some(id);
                } else {
                    self.dragging = Some(id);
                    self.selected_task = Some(id);
                }
            } else {
                self.selected_task = None;
            }
        }

        if is_mouse_button_down(MouseButton::Left) {
            if let Some(id) = self.dragging {
                self.tasks[id].pos = mouse_pos;
            }
        }

        if is_mouse_button_released(MouseButton::Left) {
            if let Some(from) = self.connecting_from {
                if let Some(to) = hovered {
                    self.add_conflict(from, to);
                }
                self.connecting_from = None;
            }
            self.dragging = None;
        }

        if is_mouse_button_pressed(MouseButton::Right) {
            if let Some(id) = hovered {
                if is_key_down(KeyCode::LeftShift) {
                    self.remove_task(id);
                } else {
                    let to_remove: Vec<Conflict> = self
                        .conflicts
                        .iter()
                        .filter(|c| c.0 == id || c.1 == id)
                        .cloned()
                        .collect();
                    for c in to_remove {
                        self.conflicts.remove(&c);
                    }
                    self.solved = false;
                }
            }
        }

        if is_key_pressed(KeyCode::Delete) || is_key_pressed(KeyCode::Backspace) {
            if let Some(id) = self.selected_task {
                self.remove_task(id);
            }
        }
    }

    fn render(&self) {
        clear_background(Color::new(0.1, 0.1, 0.15, 1.0));

        let mouse_pos = vec2(mouse_position().0, mouse_position().1);

        // Draw conflicts (edges)
        for conflict in &self.conflicts {
            let p1 = self.tasks[conflict.0].pos;
            let p2 = self.tasks[conflict.1].pos;

            let color = if self.solved {
                if self.tasks[conflict.0].time_slot == self.tasks[conflict.1].time_slot {
                    RED
                } else {
                    Color::new(0.3, 0.3, 0.35, 1.0)
                }
            } else {
                Color::new(0.8, 0.3, 0.3, 0.8)
            };

            draw_line(p1.x, p1.y, p2.x, p2.y, 3.0, color);
        }

        // Draw connection line if connecting
        if let Some(from) = self.connecting_from {
            let p1 = self.tasks[from].pos;
            draw_line(p1.x, p1.y, mouse_pos.x, mouse_pos.y, 2.0, YELLOW);
        }

        // Draw tasks (nodes)
        for task in &self.tasks {
            let color = if self.solved {
                if let Some(slot) = task.time_slot {
                    SLOT_COLORS[slot % SLOT_COLORS.len()]
                } else {
                    GRAY
                }
            } else {
                Color::new(0.4, 0.4, 0.5, 1.0)
            };

            let is_selected = self.selected_task == Some(task.id);
            let is_hovered = (task.pos - mouse_pos).length() < NODE_RADIUS;
            let radius = if is_hovered {
                NODE_RADIUS + 5.0
            } else {
                NODE_RADIUS
            };

            draw_circle(task.pos.x, task.pos.y, radius, color);

            if is_selected {
                draw_circle_lines(task.pos.x, task.pos.y, radius + 3.0, 3.0, YELLOW);
            }

            let text_size = 14.0;
            let text_dims = measure_text(&task.name, None, text_size as u16, 1.0);
            draw_text(
                &task.name,
                task.pos.x - text_dims.width / 2.0,
                task.pos.y + 5.0,
                text_size,
                WHITE,
            );

            if self.solved {
                if let Some(slot) = task.time_slot {
                    let slot_text = match self.mode {
                        ExampleMode::Portfolio => {
                            if slot == 0 {
                                "BUY".to_string()
                            } else {
                                "HOLD".to_string()
                            }
                        }
                        ExampleMode::Scheduling => format!("T{}", slot + 1),
                        ExampleMode::Routing => {
                            if slot == 0 {
                                "🚛 1".to_string()
                            } else {
                                "🚚 2".to_string()
                            }
                        }
                        ExampleMode::Warehouse => {
                            if slot == 0 {
                                "← West".to_string()
                            } else {
                                "East →".to_string()
                            }
                        }
                        ExampleMode::Cluster => {
                            if slot == 0 {
                                "rack-a".to_string()
                            } else {
                                "rack-b".to_string()
                            }
                        }
                        ExampleMode::QEC => {
                            if slot == 0 {
                                "R1".to_string()
                            } else {
                                "R2".to_string()
                            }
                        }
                    };
                    let slot_dims = measure_text(&slot_text, None, 12, 1.0);
                    draw_text(
                        &slot_text,
                        task.pos.x - slot_dims.width / 2.0,
                        task.pos.y + NODE_RADIUS + 15.0,
                        12.0,
                        WHITE,
                    );
                }
            }
        }

        // Draw legend if solved
        if self.solved {
            let legend_x = screen_width() - 150.0;
            let legend_y = 100.0;

            let legend_title = match self.mode {
                ExampleMode::Portfolio => "Portfolio:",
                ExampleMode::Scheduling => "Time Slots:",
                ExampleMode::Routing => "Trucks:",
                ExampleMode::Warehouse => "Warehouses:",
                ExampleMode::Cluster => "Server Racks:",
                ExampleMode::QEC => "Measure Round:",
            };
            draw_text(legend_title, legend_x, legend_y, 16.0, WHITE);

            let labels: Vec<&str> = match self.mode {
                ExampleMode::Portfolio => vec!["BUY", "HOLD"],
                ExampleMode::Scheduling => {
                    vec!["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
                }
                ExampleMode::Routing => vec!["Truck 1", "Truck 2"],
                ExampleMode::Warehouse => vec!["WH-West", "WH-East"],
                ExampleMode::Cluster => vec!["Rack-A", "Rack-B"],
                ExampleMode::QEC => vec!["Round 1", "Round 2"],
            };

            for slot in 0..self.num_time_slots {
                let y = legend_y + 25.0 + slot as f32 * 25.0;
                draw_rectangle(
                    legend_x,
                    y - 10.0,
                    20.0,
                    20.0,
                    SLOT_COLORS[slot % SLOT_COLORS.len()],
                );
                let label = if slot < labels.len() {
                    labels[slot]
                } else {
                    "..."
                };
                draw_text(label, legend_x + 30.0, y + 5.0, 14.0, WHITE);
            }
        }
    }

    fn draw_ui(&mut self, ctx: &egui::Context) {
        egui::Window::new("Quantum Optimizer")
            .default_pos(egui::pos2(10.0, 10.0))
            .default_width(280.0)
            .show(ctx, |ui| {
                let (title, desc) = match self.mode {
                    ExampleMode::Portfolio => {
                        ("Portfolio Optimizer", "Diversify correlated assets")
                    }
                    ExampleMode::Scheduling => {
                        ("Schedule Optimizer", "Minimize time slot conflicts")
                    }
                    ExampleMode::Routing => ("Route Optimizer", "Split deliveries between trucks"),
                    ExampleMode::Warehouse => {
                        ("Facility Optimizer", "Assign regions to warehouses")
                    }
                    ExampleMode::Cluster => ("Cluster Scheduler", "Spread workloads across racks"),
                    ExampleMode::QEC => ("QEC Verifier", "Schedule stabilizer measurements"),
                };
                ui.heading(title);
                ui.label(desc);
                ui.label("Powered by QAOA quantum algorithm");
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Examples:");
                    if ui.button("Meetings").clicked() {
                        self.load_meeting_example();
                    }
                    if ui.button("Classes").clicked() {
                        self.load_classroom_example();
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Events").clicked() {
                        self.load_event_example();
                    }
                    if ui.button("Portfolio").clicked() {
                        self.load_portfolio_example();
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Routing").clicked() {
                        self.load_routing_example();
                    }
                    if ui.button("Warehouse").clicked() {
                        self.load_warehouse_example();
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Cluster").clicked() {
                        self.load_cluster_example();
                    }
                    if ui.button("QEC").clicked() {
                        self.load_qec_example();
                    }
                });

                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Add task:");
                    ui.text_edit_singleline(&mut self.next_task_name);
                    if ui.button("+").clicked() && !self.next_task_name.is_empty() {
                        let pos = vec2(
                            400.0 + (self.tasks.len() as f32 * 50.0) % 400.0,
                            200.0 + ((self.tasks.len() / 8) as f32 * 80.0),
                        );
                        self.add_task(&self.next_task_name.clone(), pos);
                        self.next_task_name.clear();
                    }
                });

                ui.separator();

                let (item_label, edge_label) = match self.mode {
                    ExampleMode::Portfolio => ("Assets", "Correlations"),
                    ExampleMode::Scheduling => ("Tasks", "Conflicts"),
                    ExampleMode::Routing => ("Stops", "Distance penalties"),
                    ExampleMode::Warehouse => ("Regions", "Separation constraints"),
                    ExampleMode::Cluster => ("Workloads", "Anti-affinity rules"),
                    ExampleMode::QEC => ("Qubits/Stabilizers", "Shared qubit conflicts"),
                };
                ui.label(format!("{}: {}", item_label, self.tasks.len()));
                ui.label(format!("{}: {}", edge_label, self.conflicts.len()));

                if self.solved {
                    let result_label = match self.mode {
                        ExampleMode::Portfolio => {
                            let buy_count =
                                self.tasks.iter().filter(|t| t.time_slot == Some(0)).count();
                            format!(
                                "BUY: {} | HOLD: {}",
                                buy_count,
                                self.tasks.len() - buy_count
                            )
                        }
                        ExampleMode::Scheduling => {
                            format!("Time slots needed: {}", self.num_time_slots)
                        }
                        ExampleMode::Routing => {
                            let truck1 =
                                self.tasks.iter().filter(|t| t.time_slot == Some(0)).count();
                            format!(
                                "Truck 1: {} | Truck 2: {}",
                                truck1,
                                self.tasks.len() - truck1
                            )
                        }
                        ExampleMode::Warehouse => {
                            let wh_a = self.tasks.iter().filter(|t| t.time_slot == Some(0)).count();
                            format!("WH-West: {} | WH-East: {}", wh_a, self.tasks.len() - wh_a)
                        }
                        ExampleMode::Cluster => {
                            let rack_a =
                                self.tasks.iter().filter(|t| t.time_slot == Some(0)).count();
                            format!("Rack-A: {} | Rack-B: {}", rack_a, self.tasks.len() - rack_a)
                        }
                        ExampleMode::QEC => {
                            let round1 =
                                self.tasks.iter().filter(|t| t.time_slot == Some(0)).count();
                            format!(
                                "Round 1: {} | Round 2: {}",
                                round1,
                                self.tasks.len() - round1
                            )
                        }
                    };
                    ui.label(result_label);

                    if let Some(stats) = &self.qaoa_stats {
                        if stats.optimal_cut > 0.0 {
                            ui.separator();
                            ui.label("QAOA Results:");
                            ui.label(format!("  Approx ratio: {:.2}", stats.ratio));
                            ui.label(format!("  Time: {:.1}ms", stats.time_ms));
                            ui.label(format!("  Iterations: {}", stats.iterations));
                        }
                    }
                }

                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("🔮 Optimize").clicked() {
                        self.solve_with_qaoa();
                    }
                    if ui.button("Clear").clicked() {
                        self.clear();
                    }
                });

                ui.separator();

                ui.collapsing("Instructions", |ui| {
                    ui.label("• Click + drag to move nodes");
                    ui.label("• Shift+click to add edges");
                    ui.label("• Right-click to remove edges");
                    ui.label("• Shift+right-click to delete node");
                    ui.label("");
                    match self.mode {
                        ExampleMode::Portfolio => {
                            ui.label("Blue = BUY, Red = HOLD");
                            ui.label("Lines = correlated assets");
                        }
                        ExampleMode::Scheduling => {
                            ui.label("Colors = time slots");
                            ui.label("Lines = conflicts");
                        }
                        ExampleMode::Routing => {
                            ui.label("Blue = Truck 1, Red = Truck 2");
                            ui.label("Lines = distant stops (separate)");
                        }
                        ExampleMode::Warehouse => {
                            ui.label("Blue = WH-West, Red = WH-East");
                            ui.label("Lines = regions to separate");
                        }
                        ExampleMode::Cluster => {
                            ui.label("Blue = Rack-A, Red = Rack-B");
                            ui.label("Lines = anti-affinity (separate)");
                        }
                        ExampleMode::QEC => {
                            ui.label("Blue = Round 1, Red = Round 2");
                            ui.label("Lines = shared qubits (can't measure together)");
                        }
                    }
                });

                if let Some(id) = self.selected_task {
                    if id < self.tasks.len() {
                        ui.separator();
                        ui.label(format!("Selected: {}", self.tasks[id].name));

                        let conflicts: Vec<_> = self
                            .conflicts
                            .iter()
                            .filter(|c| c.0 == id || c.1 == id)
                            .map(|c| {
                                let other = if c.0 == id { c.1 } else { c.0 };
                                self.tasks[other].name.clone()
                            })
                            .collect();

                        if !conflicts.is_empty() {
                            ui.label(format!("Conflicts: {}", conflicts.join(", ")));
                        }
                    }
                }
            });
    }
}

// =============================================================================
// Main
// =============================================================================

fn window_conf() -> Conf {
    Conf {
        window_title: "Quantum Scheduler - QAOA Optimization".to_string(),
        window_width: 1200,
        window_height: 700,
        window_resizable: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut scheduler = Scheduler::new();

    loop {
        scheduler.handle_input();
        scheduler.render();

        egui_macroquad::ui(|ctx| {
            scheduler.draw_ui(ctx);
        });
        egui_macroquad::draw();

        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        next_frame().await
    }
}
