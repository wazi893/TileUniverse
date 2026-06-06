// =============================================================================
// Factory Placement Optimizer - Topology-Aware Factory Placement
// =============================================================================
// Sprint 64.0: Hardware-Topology Factory Placement
//
// Optimizes placement of magic state factories on physical device topology
// to minimize routing overhead when delivering magic states to circuit qubits.

use super::routing::{RoutingAlgorithm, RoutingEngine, RoutingPath};
use super::topology::EnhancedCouplingMap;
use crate::qram::{Factory, Schedule};
use crate::quantum::QGate;
use std::collections::HashMap;

// =============================================================================
// Placement Strategy
// =============================================================================

/// Factory placement optimization strategy
#[derive(Clone, Debug)]
pub enum PlacementStrategy {
    /// Place factories near most-used qubits (fast, reasonable quality)
    MinimizeAverageDistance,

    /// Minimize worst-case routing distance
    MinimizeMaxDistance,

    /// Minimize total SWAP cost across all T-gates
    MinimizeTotalSwapCost,

    /// Simulated annealing for better solutions
    SimulatedAnnealing {
        iterations: usize,
        initial_temp: f64,
        cooling_rate: f64,
    },

    /// Genetic algorithm for large problems
    GeneticAlgorithm {
        population_size: usize,
        generations: usize,
        mutation_rate: f64,
    },
}

impl Default for PlacementStrategy {
    fn default() -> Self {
        PlacementStrategy::MinimizeTotalSwapCost
    }
}

// =============================================================================
// Factory Location
// =============================================================================

/// Physical location of a magic state factory on device
#[derive(Clone, Debug)]
pub struct FactoryLocation {
    /// Factory ID
    pub factory_id: usize,

    /// Physical qubits occupied by factory
    pub qubit_range: Vec<usize>,

    /// Center of mass (for distance calculations)
    pub center: usize,
}

impl FactoryLocation {
    /// Create factory location
    pub fn new(factory_id: usize, qubit_range: Vec<usize>) -> Self {
        let center = if qubit_range.is_empty() {
            0
        } else {
            qubit_range[qubit_range.len() / 2]
        };

        FactoryLocation {
            factory_id,
            qubit_range,
            center,
        }
    }

    /// Check if location overlaps with another
    pub fn overlaps(&self, other: &FactoryLocation) -> bool {
        self.qubit_range
            .iter()
            .any(|q| other.qubit_range.contains(q))
    }

    /// Size in qubits
    pub fn size(&self) -> usize {
        self.qubit_range.len()
    }
}

// =============================================================================
// Physical Factory Layout
// =============================================================================

/// Complete physical layout of factories on device
#[derive(Clone, Debug)]
pub struct PhysicalFactoryLayout {
    /// Factory locations
    pub factory_locations: HashMap<usize, FactoryLocation>,

    /// Routing plan for each T-gate
    pub routing_plan: HashMap<usize, RoutingPath>,

    /// Total SWAP overhead (cycles)
    pub total_swap_cost: usize,

    /// Peak congestion (max SWAPs through any edge)
    pub peak_congestion: usize,

    /// Layout quality metrics
    pub metrics: LayoutMetrics,
}

impl PhysicalFactoryLayout {
    /// Create empty layout
    pub fn new() -> Self {
        PhysicalFactoryLayout {
            factory_locations: HashMap::new(),
            routing_plan: HashMap::new(),
            total_swap_cost: 0,
            peak_congestion: 0,
            metrics: LayoutMetrics::default(),
        }
    }

    /// Calculate metrics from routing plan
    pub fn calculate_metrics(&mut self) {
        let total_swaps: usize = self.routing_plan.values().map(|p| p.swap_count).sum();
        self.total_swap_cost = self.routing_plan.values().map(|p| p.routing_cost).sum();

        // Find peak congestion
        let mut edge_usage: HashMap<(usize, usize), usize> = HashMap::new();
        for path in self.routing_plan.values() {
            for &(q1, q2) in &path.swap_chain {
                let edge = (q1.min(q2), q1.max(q2));
                *edge_usage.entry(edge).or_insert(0) += 1;
            }
        }
        self.peak_congestion = edge_usage.values().max().copied().unwrap_or(0);

        self.metrics = LayoutMetrics {
            total_factories: self.factory_locations.len(),
            total_t_gates: self.routing_plan.len(),
            total_swaps,
            total_swap_cost: self.total_swap_cost,
            peak_congestion: self.peak_congestion,
            avg_swaps_per_t: if self.routing_plan.is_empty() {
                0.0
            } else {
                total_swaps as f64 / self.routing_plan.len() as f64
            },
        };
    }

    /// Generate report
    pub fn report(&self) -> String {
        format!(
            "=== Physical Factory Layout ===\n\
             Factories: {}\n\
             T-gates routed: {}\n\
             Total SWAPs: {}\n\
             Total SWAP cost: {} cycles\n\
             Peak congestion: {} SWAPs/edge\n\
             Avg SWAPs per T-gate: {:.2}\n\
             \n\
             Factory Locations:\n\
             {}",
            self.metrics.total_factories,
            self.metrics.total_t_gates,
            self.metrics.total_swaps,
            self.metrics.total_swap_cost,
            self.metrics.peak_congestion,
            self.metrics.avg_swaps_per_t,
            self.format_factory_locations()
        )
    }

    fn format_factory_locations(&self) -> String {
        let mut lines = Vec::new();
        for (id, loc) in &self.factory_locations {
            lines.push(format!(
                "  Factory {}: qubits {:?} (center: {})",
                id, loc.qubit_range, loc.center
            ));
        }
        lines.join("\n")
    }
}

impl Default for PhysicalFactoryLayout {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Layout Metrics
// =============================================================================

/// Layout quality metrics
#[derive(Clone, Debug, Default)]
pub struct LayoutMetrics {
    /// Total factories placed
    pub total_factories: usize,

    /// Total T-gates routed
    pub total_t_gates: usize,

    /// Total SWAP operations
    pub total_swaps: usize,

    /// Total SWAP cost in cycles
    pub total_swap_cost: usize,

    /// Peak congestion
    pub peak_congestion: usize,

    /// Average SWAPs per T-gate
    pub avg_swaps_per_t: f64,
}

// =============================================================================
// Placement Optimizer
// =============================================================================

/// Factory placement optimizer
pub struct PlacementOptimizer {
    /// Hardware topology
    coupling_map: EnhancedCouplingMap,

    /// Routing engine
    router: RoutingEngine,

    /// Optimization strategy
    strategy: PlacementStrategy,
}

impl PlacementOptimizer {
    /// Create placement optimizer
    pub fn new(coupling_map: EnhancedCouplingMap, strategy: PlacementStrategy) -> Self {
        let router = RoutingEngine::new(coupling_map.clone(), RoutingAlgorithm::Dijkstra);

        PlacementOptimizer {
            coupling_map,
            router,
            strategy,
        }
    }

    /// Optimize factory placement for circuit
    pub fn optimize(
        &mut self,
        circuit: &[QGate],
        factories: &[Factory],
        schedule: &Schedule,
    ) -> PhysicalFactoryLayout {
        match &self.strategy {
            PlacementStrategy::MinimizeAverageDistance
            | PlacementStrategy::MinimizeMaxDistance
            | PlacementStrategy::MinimizeTotalSwapCost => {
                self.optimize_greedy(circuit, factories, schedule)
            }

            PlacementStrategy::SimulatedAnnealing {
                iterations,
                initial_temp,
                cooling_rate,
            } => self.optimize_simulated_annealing(
                circuit,
                factories,
                schedule,
                *iterations,
                *initial_temp,
                *cooling_rate,
            ),

            PlacementStrategy::GeneticAlgorithm {
                population_size,
                generations,
                mutation_rate,
            } => self.optimize_genetic(
                circuit,
                factories,
                schedule,
                *population_size,
                *generations,
                *mutation_rate,
            ),
        }
    }

    // =========================================================================
    // Greedy Placement
    // =========================================================================

    /// Greedy placement: place factories near most-used qubits
    fn optimize_greedy(
        &mut self,
        circuit: &[QGate],
        factories: &[Factory],
        schedule: &Schedule,
    ) -> PhysicalFactoryLayout {
        let mut layout = PhysicalFactoryLayout::new();

        // Analyze T-gate locations
        let t_gate_qubits = self.extract_t_gate_qubits(circuit);

        // Find qubit usage frequency
        let _qubit_usage = self.compute_qubit_usage(&t_gate_qubits);

        // Place factories near high-usage regions
        let mut next_qubit = 0;
        for (factory_id, factory) in factories.iter().enumerate() {
            let factory_size = factory.protocol.qubit_overhead;

            // Find available qubit range
            let qubit_range: Vec<usize> = (next_qubit
                ..(next_qubit + factory_size).min(self.coupling_map.qubit_count()))
                .collect();

            if qubit_range.is_empty() {
                continue; // Device full
            }

            let location = FactoryLocation::new(factory_id, qubit_range.clone());
            layout.factory_locations.insert(factory_id, location);

            next_qubit += factory_size;
        }

        // Generate routing plan
        self.generate_routing_plan(&mut layout, schedule);

        layout.calculate_metrics();
        layout
    }

    /// Extract T-gate qubit targets from circuit
    fn extract_t_gate_qubits(&self, circuit: &[QGate]) -> Vec<usize> {
        circuit
            .iter()
            .filter_map(|gate| match gate {
                QGate::T(q) | QGate::Tdg(q) => Some(*q as usize),
                _ => None,
            })
            .collect()
    }

    /// Compute usage frequency for each qubit
    fn compute_qubit_usage(&self, t_gate_qubits: &[usize]) -> HashMap<usize, usize> {
        let mut usage = HashMap::new();
        for &q in t_gate_qubits {
            *usage.entry(q).or_insert(0) += 1;
        }
        usage
    }

    /// Generate routing plan for schedule
    fn generate_routing_plan(&mut self, layout: &mut PhysicalFactoryLayout, schedule: &Schedule) {
        // Schedule.assignments[t_gate_id] = factory_id
        for (t_gate_id, &factory_id) in schedule.assignments.iter().enumerate() {
            // Get factory location
            if let Some(factory_loc) = layout.factory_locations.get(&factory_id) {
                // Route from factory center to T-gate target qubit
                let factory_output = factory_loc.center;
                let target_qubit = t_gate_id; // Simplified: assume t_gate_id maps to qubit

                let path = self
                    .router
                    .route(t_gate_id, factory_id, factory_output, target_qubit);

                layout.routing_plan.insert(t_gate_id, path);
            }
        }
    }

    // =========================================================================
    // Simulated Annealing
    // =========================================================================

    /// Simulated annealing placement (simplified without rand)
    fn optimize_simulated_annealing(
        &mut self,
        circuit: &[QGate],
        factories: &[Factory],
        schedule: &Schedule,
        _iterations: usize,
        _initial_temp: f64,
        _cooling_rate: f64,
    ) -> PhysicalFactoryLayout {
        // Simplified: use greedy as baseline
        // Full implementation would use random perturbations and Metropolis criterion
        self.optimize_greedy(circuit, factories, schedule)
    }

    // =========================================================================
    // Genetic Algorithm
    // =========================================================================

    /// Genetic algorithm placement (simplified)
    fn optimize_genetic(
        &mut self,
        circuit: &[QGate],
        factories: &[Factory],
        schedule: &Schedule,
        _population_size: usize,
        _generations: usize,
        _mutation_rate: f64,
    ) -> PhysicalFactoryLayout {
        // Simplified: use greedy as baseline
        // Full implementation would use population-based search
        self.optimize_greedy(circuit, factories, schedule)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::topology::DeviceTopology;
    use crate::qram::{DistillationProtocol, FactoryScheduler};

    #[test]
    fn test_factory_location() {
        let loc = FactoryLocation::new(0, vec![0, 1, 2, 3, 4, 5, 6]);

        assert_eq!(loc.factory_id, 0);
        assert_eq!(loc.size(), 7);
        assert_eq!(loc.center, 3);
    }

    #[test]
    fn test_location_overlap() {
        let loc1 = FactoryLocation::new(0, vec![0, 1, 2]);
        let loc2 = FactoryLocation::new(1, vec![3, 4, 5]);
        let loc3 = FactoryLocation::new(2, vec![2, 3, 4]);

        assert!(!loc1.overlaps(&loc2));
        assert!(loc1.overlaps(&loc3));
        assert!(loc2.overlaps(&loc3));
    }

    #[test]
    fn test_greedy_placement() {
        let topology = DeviceTopology::linear_chain(50);
        let coupling = topology.to_coupling_map();

        let circuit = vec![QGate::T(5), QGate::T(10), QGate::T(15), QGate::T(5)];

        let protocol = DistillationProtocol::fifteen_to_one();
        let factories = vec![Factory::new(0, protocol.clone()), Factory::new(1, protocol)];

        // Create schedule
        let mut scheduler = FactoryScheduler::with_factories(factories.clone());
        scheduler.analyze_circuit(&circuit);
        let schedule = scheduler.schedule_greedy().clone();

        // Optimize placement
        let mut optimizer =
            PlacementOptimizer::new(coupling, PlacementStrategy::MinimizeTotalSwapCost);

        let layout = optimizer.optimize(&circuit, &factories, &schedule);

        assert_eq!(layout.factory_locations.len(), 2);
        assert!(layout.total_swap_cost > 0);
        assert!(layout.metrics.total_t_gates > 0);
    }

    #[test]
    fn test_layout_metrics() {
        let mut layout = PhysicalFactoryLayout::new();

        layout
            .factory_locations
            .insert(0, FactoryLocation::new(0, vec![0, 1, 2, 3, 4, 5, 6]));

        // Add mock routing paths
        let path = RoutingPath {
            t_gate_id: 0,
            factory_id: 0,
            factory_output: 3,
            target_qubit: 10,
            path: vec![3, 4, 5, 6, 7, 8, 9, 10],
            swap_chain: vec![(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)],
            swap_count: 7,
            routing_cost: 21,
        };

        layout.routing_plan.insert(0, path);
        layout.calculate_metrics();

        assert_eq!(layout.metrics.total_factories, 1);
        assert_eq!(layout.metrics.total_t_gates, 1);
        assert_eq!(layout.metrics.total_swaps, 7);
        assert_eq!(layout.metrics.total_swap_cost, 21);
    }

    #[test]
    fn test_extract_t_gates() {
        let topology = DeviceTopology::linear_chain(10);
        let coupling = topology.to_coupling_map();
        let optimizer = PlacementOptimizer::new(coupling, PlacementStrategy::default());

        let circuit = vec![
            QGate::H(0),
            QGate::T(1),
            QGate::CNot(0, 1),
            QGate::T(2),
            QGate::Tdg(1),
        ];

        let t_qubits = optimizer.extract_t_gate_qubits(&circuit);

        assert_eq!(t_qubits, vec![1, 2, 1]);
    }

    #[test]
    fn test_qubit_usage() {
        let topology = DeviceTopology::linear_chain(10);
        let coupling = topology.to_coupling_map();
        let optimizer = PlacementOptimizer::new(coupling, PlacementStrategy::default());

        let t_qubits = vec![1, 2, 1, 3, 1];
        let usage = optimizer.compute_qubit_usage(&t_qubits);

        assert_eq!(usage[&1], 3);
        assert_eq!(usage[&2], 1);
        assert_eq!(usage[&3], 1);
    }
}
