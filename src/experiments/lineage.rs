// =============================================================================
// src/lineage.rs
// =============================================================================
//
// SPRINT 3.3: Lineage Tracking
//
// The missing holy metric. Population-level stats tell you WHO won.
// Lineage tracking tells you WHY.
//
// We track:
// - Birth/death events with timestamps
// - Parent-child relationships (family trees)
// - Genome snapshots at key moments
// - Fitness trajectories over time
// - Circuit evolution within lineages
//
// This answers: Do quantum organisms collapse from circuit drift?
// Do classical organisms converge on simple stable strategies?
//
// =============================================================================

use crate::constrained_circuits::{count_entangling_gates, entanglement_fraction};
use crate::quantum::QGate;

use std::collections::{HashMap, HashSet, VecDeque};

// =============================================================================
// Core Types
// =============================================================================

/// A snapshot of an organism at a moment in time.
#[derive(Clone, Debug)]
pub struct OrganismSnapshot {
    pub id: u64,
    pub tick: u64,
    pub energy: f32,
    pub age: u32,
    pub x: u32,
    pub y: u32,
    pub circuit_length: usize,
    pub entangling_gates: usize,
    pub quantum_ness: f32,
}

/// A node in the lineage tree.
#[derive(Clone, Debug)]
pub struct LineageNode {
    /// Unique organism ID
    pub id: u64,

    /// Parent ID (None for initial population)
    pub parent_id: Option<u64>,

    /// Generation number (0 for initial, +1 each reproduction)
    pub generation: u32,

    /// Tick when born
    pub birth_tick: u64,

    /// Tick when died (None if still alive)
    pub death_tick: Option<u64>,

    /// Lifespan in ticks (computed on death)
    pub lifespan: Option<u64>,

    /// Children IDs
    pub children: Vec<u64>,

    /// Number of descendants (computed lazily)
    pub descendant_count: Option<usize>,

    /// Genome at birth
    pub birth_genome: Vec<QGate>,

    /// Peak energy achieved
    pub peak_energy: f32,

    /// Total resources consumed over lifetime
    pub total_consumed: f32,

    /// Snapshots at regular intervals
    pub snapshots: Vec<OrganismSnapshot>,

    /// Circuit metrics at birth
    pub birth_circuit_length: usize,
    pub birth_entangling_gates: usize,
    pub birth_quantum_ness: f32,
}

impl LineageNode {
    /// Create a new lineage node for a newborn organism.
    pub fn new(
        id: u64,
        parent_id: Option<u64>,
        generation: u32,
        birth_tick: u64,
        genome: Vec<QGate>,
        initial_energy: f32,
    ) -> Self {
        let circuit_length = genome.len();
        let entangling = count_entangling_gates(&genome);
        let qness = entanglement_fraction(&genome);

        Self {
            id,
            parent_id,
            generation,
            birth_tick,
            death_tick: None,
            lifespan: None,
            children: Vec::new(),
            descendant_count: None,
            birth_genome: genome,
            peak_energy: initial_energy,
            total_consumed: 0.0,
            snapshots: Vec::new(),
            birth_circuit_length: circuit_length,
            birth_entangling_gates: entangling,
            birth_quantum_ness: qness,
        }
    }

    /// Record death.
    pub fn record_death(&mut self, tick: u64, final_consumed: f32) {
        self.death_tick = Some(tick);
        self.lifespan = Some(tick.saturating_sub(self.birth_tick));
        self.total_consumed = final_consumed;
    }

    /// Update peak energy if current is higher.
    pub fn update_peak_energy(&mut self, energy: f32) {
        if energy > self.peak_energy {
            self.peak_energy = energy;
        }
    }

    /// Add a periodic snapshot.
    pub fn add_snapshot(&mut self, snapshot: OrganismSnapshot) {
        self.snapshots.push(snapshot);
    }

    /// Check if this organism is a "founder" (has many descendants).
    pub fn is_founder(&self, threshold: usize) -> bool {
        self.descendant_count.unwrap_or(0) >= threshold
    }

    /// Get survival time.
    pub fn survival_time(&self, current_tick: u64) -> u64 {
        match self.death_tick {
            Some(death) => death.saturating_sub(self.birth_tick),
            None => current_tick.saturating_sub(self.birth_tick),
        }
    }
}

// =============================================================================
// Lineage Tracker
// =============================================================================

/// Tracks the full evolutionary history of a population.
#[derive(Clone, Debug)]
pub struct LineageTracker {
    /// All organisms ever born
    nodes: HashMap<u64, LineageNode>,

    /// Currently alive organisms
    alive: HashSet<u64>,

    /// Root organisms (no parent)
    roots: Vec<u64>,

    /// Current tick
    current_tick: u64,

    /// Snapshot interval (take snapshot every N ticks)
    snapshot_interval: u64,

    /// Track extinction events (generation -> tick)
    #[allow(dead_code)]
    extinction_events: Vec<(u32, u64)>,
}

impl LineageTracker {
    /// Create a new empty tracker.
    pub fn new(snapshot_interval: u64) -> Self {
        Self {
            nodes: HashMap::new(),
            alive: HashSet::new(),
            roots: Vec::new(),
            current_tick: 0,
            snapshot_interval,
            extinction_events: Vec::new(),
        }
    }

    /// Record a birth.
    pub fn record_birth(
        &mut self,
        id: u64,
        parent_id: Option<u64>,
        generation: u32,
        genome: Vec<QGate>,
        initial_energy: f32,
    ) {
        let node = LineageNode::new(
            id,
            parent_id,
            generation,
            self.current_tick,
            genome,
            initial_energy,
        );

        // Link to parent
        if let Some(pid) = parent_id {
            if let Some(parent) = self.nodes.get_mut(&pid) {
                parent.children.push(id);
            }
        } else {
            self.roots.push(id);
        }

        self.nodes.insert(id, node);
        self.alive.insert(id);
    }

    /// Record a death.
    pub fn record_death(&mut self, id: u64, total_consumed: f32) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.record_death(self.current_tick, total_consumed);
        }
        self.alive.remove(&id);
    }

    /// Record a periodic snapshot for a living organism.
    pub fn record_snapshot(
        &mut self,
        id: u64,
        energy: f32,
        age: u32,
        x: u32,
        y: u32,
        genome: &[QGate],
    ) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.update_peak_energy(energy);

            if self.current_tick % self.snapshot_interval == 0 {
                let snapshot = OrganismSnapshot {
                    id,
                    tick: self.current_tick,
                    energy,
                    age,
                    x,
                    y,
                    circuit_length: genome.len(),
                    entangling_gates: count_entangling_gates(genome),
                    quantum_ness: entanglement_fraction(genome),
                };
                node.add_snapshot(snapshot);
            }
        }
    }

    /// Advance the tick counter.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Set the current tick directly.
    pub fn set_tick(&mut self, tick: u64) {
        self.current_tick = tick;
    }

    // =========================================================================
    // Analysis Methods
    // =========================================================================

    /// Get total number of organisms ever born.
    pub fn total_births(&self) -> usize {
        self.nodes.len()
    }

    /// Get number currently alive.
    pub fn alive_count(&self) -> usize {
        self.alive.len()
    }

    /// Get maximum generation reached.
    pub fn max_generation(&self) -> u32 {
        self.nodes.values().map(|n| n.generation).max().unwrap_or(0)
    }

    /// Get average lifespan of dead organisms.
    pub fn average_lifespan(&self) -> f32 {
        let dead: Vec<_> = self
            .nodes
            .values()
            .filter(|n| n.death_tick.is_some())
            .collect();

        if dead.is_empty() {
            return 0.0;
        }

        let total: u64 = dead.iter().map(|n| n.lifespan.unwrap_or(0)).sum();

        total as f32 / dead.len() as f32
    }

    /// Get survival curve data (fraction alive at each generation).
    pub fn survival_curve(&self) -> Vec<(u32, f32)> {
        let max_gen = self.max_generation();
        let mut curve = Vec::with_capacity(max_gen as usize + 1);

        for generation in 0..=max_gen {
            let at_gen: Vec<_> = self
                .nodes
                .values()
                .filter(|n| n.generation == generation)
                .collect();

            if at_gen.is_empty() {
                curve.push((generation, 0.0));
                continue;
            }

            let survived = at_gen
                .iter()
                .filter(|n| n.death_tick.is_none() || n.lifespan.unwrap_or(0) > 50)
                .count();

            curve.push((generation, survived as f32 / at_gen.len() as f32));
        }

        curve
    }

    /// Calculate population diversity (entropy of circuit structures).
    pub fn diversity_entropy(&self) -> f32 {
        if self.alive.is_empty() {
            return 0.0;
        }

        // Bucket by circuit length
        let mut buckets: HashMap<usize, usize> = HashMap::new();

        for id in &self.alive {
            if let Some(node) = self.nodes.get(id) {
                *buckets.entry(node.birth_circuit_length).or_insert(0) += 1;
            }
        }

        let total = self.alive.len() as f32;
        let mut entropy = 0.0f32;

        for count in buckets.values() {
            let p = *count as f32 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Get quantum-ness distribution across generations.
    pub fn quantum_ness_by_generation(&self) -> Vec<(u32, f32, f32)> {
        let max_gen = self.max_generation();
        let mut result = Vec::new();

        for generation in 0..=max_gen {
            let qness_values: Vec<f32> = self
                .nodes
                .values()
                .filter(|n| n.generation == generation)
                .map(|n| n.birth_quantum_ness)
                .collect();

            if qness_values.is_empty() {
                continue;
            }

            let mean = qness_values.iter().sum::<f32>() / qness_values.len() as f32;
            let variance = qness_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                / qness_values.len() as f32;
            let std = variance.sqrt();

            result.push((generation, mean, std));
        }

        result
    }

    /// Find "founder" organisms (those with many descendants).
    pub fn find_founders(&mut self, min_descendants: usize) -> Vec<u64> {
        // First, compute descendant counts
        self.compute_descendant_counts();

        self.nodes
            .iter()
            .filter(|(_, n)| n.descendant_count.unwrap_or(0) >= min_descendants)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Compute descendant counts for all nodes (memoized BFS).
    fn compute_descendant_counts(&mut self) {
        // Process in reverse generation order (leaves first)
        let mut by_gen: Vec<(u32, u64)> = self
            .nodes
            .iter()
            .map(|(id, n)| (n.generation, *id))
            .collect();
        by_gen.sort_by(|a, b| b.0.cmp(&a.0)); // Descending

        for (_, id) in by_gen {
            let children_count: usize = {
                let node = self.nodes.get(&id).unwrap();
                node.children
                    .iter()
                    .map(|cid| {
                        self.nodes
                            .get(cid)
                            .map(|c| 1 + c.descendant_count.unwrap_or(0))
                            .unwrap_or(0)
                    })
                    .sum()
            };

            if let Some(node) = self.nodes.get_mut(&id) {
                node.descendant_count = Some(children_count);
            }
        }
    }

    /// Get the lineage (ancestors) of an organism.
    pub fn get_ancestors(&self, id: u64) -> Vec<u64> {
        let mut ancestors = Vec::new();
        let mut current = id;

        while let Some(node) = self.nodes.get(&current) {
            if let Some(parent_id) = node.parent_id {
                ancestors.push(parent_id);
                current = parent_id;
            } else {
                break;
            }
        }

        ancestors
    }

    /// Get all descendants of an organism.
    pub fn get_descendants(&self, id: u64) -> Vec<u64> {
        let mut descendants = Vec::new();
        let mut queue = VecDeque::new();

        if let Some(node) = self.nodes.get(&id) {
            queue.extend(node.children.iter());
        }

        while let Some(child_id) = queue.pop_front() {
            descendants.push(child_id);
            if let Some(child) = self.nodes.get(&child_id) {
                queue.extend(child.children.iter());
            }
        }

        descendants
    }

    /// Get the longest lineage (most generations in a single line).
    pub fn longest_lineage(&self) -> Vec<u64> {
        let mut best: Vec<u64> = Vec::new();

        // Find deepest organism
        if let Some((deepest_id, _)) = self.nodes.iter().max_by_key(|(_, n)| n.generation) {
            // Trace back to root
            let mut lineage = vec![*deepest_id];
            lineage.extend(self.get_ancestors(*deepest_id));
            lineage.reverse();
            best = lineage;
        }

        best
    }

    /// Calculate circuit similarity between parent and child.
    pub fn mutation_stability(&self) -> f32 {
        let mut similarities = Vec::new();

        for node in self.nodes.values() {
            if let Some(parent_id) = node.parent_id {
                if let Some(parent) = self.nodes.get(&parent_id) {
                    let sim = circuit_similarity(&parent.birth_genome, &node.birth_genome);
                    similarities.push(sim);
                }
            }
        }

        if similarities.is_empty() {
            return 1.0;
        }

        similarities.iter().sum::<f32>() / similarities.len() as f32
    }

    /// Get circuit length evolution over generations.
    pub fn circuit_length_by_generation(&self) -> Vec<(u32, f32, f32)> {
        let max_gen = self.max_generation();
        let mut result = Vec::new();

        for generation in 0..=max_gen {
            let lengths: Vec<f32> = self
                .nodes
                .values()
                .filter(|n| n.generation == generation)
                .map(|n| n.birth_circuit_length as f32)
                .collect();

            if lengths.is_empty() {
                continue;
            }

            let mean = lengths.iter().sum::<f32>() / lengths.len() as f32;
            let variance =
                lengths.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / lengths.len() as f32;

            result.push((generation, mean, variance.sqrt()));
        }

        result
    }

    // =========================================================================
    // Export Methods
    // =========================================================================

    /// Export to GraphViz DOT format.
    pub fn export_dot(&self) -> String {
        let mut dot = String::from("digraph Lineage {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=circle];\n\n");

        // Add nodes
        for (id, node) in &self.nodes {
            let color = if node.death_tick.is_none() {
                "green"
            } else if node.lifespan.unwrap_or(0) > 100 {
                "blue"
            } else {
                "gray"
            };

            let label = format!(
                "{}\\ng{}\\nq:{:.2}",
                id, node.generation, node.birth_quantum_ness
            );

            dot.push_str(&format!(
                "  {} [label=\"{}\" color=\"{}\"];\n",
                id, label, color
            ));
        }

        dot.push('\n');

        // Add edges
        for (id, node) in &self.nodes {
            for child_id in &node.children {
                dot.push_str(&format!("  {} -> {};\n", id, child_id));
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Export lineage statistics to CSV.
    pub fn export_csv(&self) -> String {
        let mut csv = String::from(
            "id,parent_id,generation,birth_tick,death_tick,lifespan,\
             children_count,circuit_length,entangling_gates,quantum_ness,\
             peak_energy,total_consumed\n",
        );

        for node in self.nodes.values() {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{:.4},{:.2},{:.2}\n",
                node.id,
                node.parent_id.map(|p| p.to_string()).unwrap_or_default(),
                node.generation,
                node.birth_tick,
                node.death_tick.map(|t| t.to_string()).unwrap_or_default(),
                node.lifespan.map(|l| l.to_string()).unwrap_or_default(),
                node.children.len(),
                node.birth_circuit_length,
                node.birth_entangling_gates,
                node.birth_quantum_ness,
                node.peak_energy,
                node.total_consumed,
            ));
        }

        csv
    }

    /// Get summary statistics.
    pub fn summary(&self) -> LineageSummary {
        let dead: Vec<_> = self
            .nodes
            .values()
            .filter(|n| n.death_tick.is_some())
            .collect();

        let avg_lifespan = if dead.is_empty() {
            0.0
        } else {
            dead.iter()
                .map(|n| n.lifespan.unwrap_or(0) as f32)
                .sum::<f32>()
                / dead.len() as f32
        };

        let avg_children = if self.nodes.is_empty() {
            0.0
        } else {
            self.nodes
                .values()
                .map(|n| n.children.len() as f32)
                .sum::<f32>()
                / self.nodes.len() as f32
        };

        let avg_qness = if self.nodes.is_empty() {
            0.0
        } else {
            self.nodes
                .values()
                .map(|n| n.birth_quantum_ness)
                .sum::<f32>()
                / self.nodes.len() as f32
        };

        LineageSummary {
            total_organisms: self.nodes.len(),
            currently_alive: self.alive.len(),
            max_generation: self.max_generation(),
            avg_lifespan,
            avg_children_per_organism: avg_children,
            avg_quantum_ness: avg_qness,
            diversity_entropy: self.diversity_entropy(),
            mutation_stability: self.mutation_stability(),
        }
    }
}

/// Summary statistics for a lineage.
#[derive(Clone, Debug, Default)]
pub struct LineageSummary {
    pub total_organisms: usize,
    pub currently_alive: usize,
    pub max_generation: u32,
    pub avg_lifespan: f32,
    pub avg_children_per_organism: f32,
    pub avg_quantum_ness: f32,
    pub diversity_entropy: f32,
    pub mutation_stability: f32,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Calculate similarity between two circuits (0.0 to 1.0).
fn circuit_similarity(a: &[QGate], b: &[QGate]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // Simple: compare lengths and gate type distributions
    let len_sim = 1.0 - (a.len() as f32 - b.len() as f32).abs() / (a.len().max(b.len()) as f32);

    // Compare gate type frequencies
    let a_types = gate_type_distribution(a);
    let b_types = gate_type_distribution(b);

    let mut type_sim = 0.0f32;
    let mut total_types = 0;

    for (gate_type, a_count) in &a_types {
        let b_count = b_types.get(gate_type).unwrap_or(&0);
        let max_count = (*a_count).max(*b_count) as f32;
        if max_count > 0.0 {
            type_sim += (*a_count).min(*b_count) as f32 / max_count;
            total_types += 1;
        }
    }

    for (gate_type, _) in &b_types {
        if !a_types.contains_key(gate_type) {
            total_types += 1;
        }
    }

    let type_sim = if total_types > 0 {
        type_sim / total_types as f32
    } else {
        1.0
    };

    (len_sim + type_sim) / 2.0
}

/// Get distribution of gate types in a circuit.
fn gate_type_distribution(circuit: &[QGate]) -> HashMap<&'static str, usize> {
    let mut dist = HashMap::new();

    for gate in circuit {
        let name = match gate {
            QGate::H(_) => "H",
            QGate::X(_) => "X",
            QGate::Z(_) => "Z",
            QGate::Phase(_, _) => "Phase",
            QGate::Rx(_, _) => "Rx",       // EPIC 83 Phase 3
            QGate::Ry(_, _) => "Ry",       // EPIC 83 Phase 3
            QGate::Rz(_, _) => "Rz",       // EPIC 83 Phase 3
            QGate::U3(_, _, _, _) => "U3", // SPRINT 32.0
            QGate::T(_) => "T",            // SPRINT 48.0
            QGate::Tdg(_) => "Tdg",        // SPRINT 48.0
            QGate::Y(_) => "Y",
            QGate::Swap(_, _) => "Swap",
            QGate::CNot(_, _) => "CNOT",
            QGate::CZ(_, _) => "CZ",
            QGate::CPhase(_, _, _) => "CPhase", // Controlled-Phase gate
            QGate::CRz(_, _, _) => "CRz",       // Controlled-Rz gate
            QGate::Toffoli(_, _, _) => "Toffoli",
            QGate::CCZ(_, _, _) => "CCZ",
            QGate::Measure(_) => "Measure",
        };
        *dist.entry(name).or_insert(0) += 1;
    }

    dist
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lineage_tracker_basic() {
        let mut tracker = LineageTracker::new(10);

        // Birth root organism
        tracker.record_birth(0, None, 0, vec![QGate::H(0)], 100.0);
        assert_eq!(tracker.total_births(), 1);
        assert_eq!(tracker.alive_count(), 1);

        // Birth child
        tracker.tick();
        tracker.record_birth(1, Some(0), 1, vec![QGate::H(0), QGate::X(1)], 50.0);
        assert_eq!(tracker.total_births(), 2);
        assert_eq!(tracker.alive_count(), 2);

        // Kill parent
        tracker.tick();
        tracker.record_death(0, 150.0);
        assert_eq!(tracker.alive_count(), 1);
    }

    #[test]
    fn test_ancestors() {
        let mut tracker = LineageTracker::new(10);

        tracker.record_birth(0, None, 0, vec![], 100.0);
        tracker.record_birth(1, Some(0), 1, vec![], 100.0);
        tracker.record_birth(2, Some(1), 2, vec![], 100.0);
        tracker.record_birth(3, Some(2), 3, vec![], 100.0);

        let ancestors = tracker.get_ancestors(3);
        assert_eq!(ancestors, vec![2, 1, 0]);
    }

    #[test]
    fn test_descendants() {
        let mut tracker = LineageTracker::new(10);

        tracker.record_birth(0, None, 0, vec![], 100.0);
        tracker.record_birth(1, Some(0), 1, vec![], 100.0);
        tracker.record_birth(2, Some(0), 1, vec![], 100.0);
        tracker.record_birth(3, Some(1), 2, vec![], 100.0);

        let descendants = tracker.get_descendants(0);
        assert_eq!(descendants.len(), 3);
        assert!(descendants.contains(&1));
        assert!(descendants.contains(&2));
        assert!(descendants.contains(&3));
    }

    #[test]
    fn test_quantum_ness_by_generation() {
        let mut tracker = LineageTracker::new(10);

        // Gen 0: low quantum-ness
        tracker.record_birth(0, None, 0, vec![QGate::H(0)], 100.0);

        // Gen 1: higher quantum-ness
        tracker.record_birth(1, Some(0), 1, vec![QGate::H(0), QGate::CNot(0, 1)], 100.0);

        let qness = tracker.quantum_ness_by_generation();
        assert_eq!(qness.len(), 2);
        assert!(qness[0].1 < qness[1].1); // Gen 1 has higher quantum-ness
    }

    #[test]
    fn test_circuit_similarity() {
        let a = vec![QGate::H(0), QGate::CNot(0, 1)];
        let b = vec![QGate::H(0), QGate::CNot(0, 1)];
        assert!((circuit_similarity(&a, &b) - 1.0).abs() < 0.01);

        let c = vec![QGate::X(0)];
        assert!(circuit_similarity(&a, &c) < 0.8);
    }

    #[test]
    fn test_export_dot() {
        let mut tracker = LineageTracker::new(10);
        tracker.record_birth(0, None, 0, vec![QGate::H(0)], 100.0);
        tracker.record_birth(1, Some(0), 1, vec![QGate::H(0)], 100.0);

        let dot = tracker.export_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("0 -> 1"));
    }

    #[test]
    fn test_export_csv() {
        let mut tracker = LineageTracker::new(10);
        tracker.record_birth(0, None, 0, vec![QGate::H(0)], 100.0);

        let csv = tracker.export_csv();
        assert!(csv.contains("id,parent_id"));
        assert!(csv.contains("0,,0")); // id=0, no parent, gen=0
    }
}
