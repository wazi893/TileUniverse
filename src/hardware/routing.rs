// =============================================================================
// Magic State Routing Engine - Advanced Routing for Factory Scheduling
// =============================================================================
// Sprint 64.0: Hardware-Topology Factory Placement
//
// Provides advanced routing algorithms for delivering magic states from
// factories to circuit qubits, considering congestion and batch routing.
//
// This extends swap_router.rs (Sprint 52) with factory-specific routing.

use super::topology::EnhancedCouplingMap;
use std::collections::HashMap;

// =============================================================================
// Routing Algorithms
// =============================================================================

/// Routing algorithm selection
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RoutingAlgorithm {
    /// Dijkstra shortest path (guaranteed optimal)
    #[default]
    Dijkstra,

    /// A* with heuristic (faster for grids)
    AStar,

    /// Greedy SWAP insertion
    GreedySwap,
}

// =============================================================================
// Routing Path
// =============================================================================

/// Route for delivering magic state from factory to target qubit
#[derive(Clone, Debug)]
pub struct RoutingPath {
    /// T-gate ID being served
    pub t_gate_id: usize,

    /// Factory ID providing magic state
    pub factory_id: usize,

    /// Factory output qubit (where magic state emerges)
    pub factory_output: usize,

    /// Target qubit (where circuit needs the magic state)
    pub target_qubit: usize,

    /// Sequence of qubits in path (factory_output → ... → target)
    pub path: Vec<usize>,

    /// SWAP chain: pairs of qubits to swap
    pub swap_chain: Vec<(usize, usize)>,

    /// Number of SWAPs needed
    pub swap_count: usize,

    /// Routing cost in cycles (swap_count * swap_cost)
    pub routing_cost: usize,
}

impl RoutingPath {
    /// Create routing path from BFS/Dijkstra result
    pub fn from_path(
        t_gate_id: usize,
        factory_id: usize,
        factory_output: usize,
        target_qubit: usize,
        path: Vec<usize>,
        swap_cost: usize,
    ) -> Self {
        // Convert path to SWAP chain
        let swap_chain = Self::path_to_swaps(&path);
        let swap_count = swap_chain.len();
        let routing_cost = swap_count * swap_cost;

        RoutingPath {
            t_gate_id,
            factory_id,
            factory_output,
            target_qubit,
            path,
            swap_chain,
            swap_count,
            routing_cost,
        }
    }

    /// Convert qubit path to SWAP chain
    /// Path [0, 1, 2, 3] → SWAPs [(0,1), (1,2), (2,3)]
    fn path_to_swaps(path: &[usize]) -> Vec<(usize, usize)> {
        if path.len() <= 1 {
            return vec![];
        }

        path.windows(2)
            .map(|window| (window[0], window[1]))
            .collect()
    }
}

// =============================================================================
// Routing Engine
// =============================================================================

/// Routing engine for magic state delivery
pub struct RoutingEngine {
    /// Hardware topology (connectivity graph)
    coupling_map: EnhancedCouplingMap,

    /// Routing algorithm
    algorithm: RoutingAlgorithm,

    /// SWAP cost in cycles (typically 3 CNOTs)
    swap_cost: usize,
}

impl RoutingEngine {
    /// Create routing engine
    pub fn new(coupling_map: EnhancedCouplingMap, algorithm: RoutingAlgorithm) -> Self {
        RoutingEngine {
            coupling_map,
            algorithm,
            swap_cost: 3, // Standard SWAP = 3 CNOTs
        }
    }

    /// Set SWAP cost
    pub fn with_swap_cost(mut self, swap_cost: usize) -> Self {
        self.swap_cost = swap_cost;
        self
    }

    /// Route single magic state from factory to target
    pub fn route(
        &self,
        t_gate_id: usize,
        factory_id: usize,
        factory_output: usize,
        target_qubit: usize,
    ) -> RoutingPath {
        let path = match self.algorithm {
            RoutingAlgorithm::Dijkstra | RoutingAlgorithm::GreedySwap => {
                // Use coupling map's built-in shortest path
                self.coupling_map
                    .shortest_path(factory_output, target_qubit)
            }
            RoutingAlgorithm::AStar => {
                // A* with Manhattan distance heuristic (if coordinates available)
                self.astar_route(factory_output, target_qubit)
            }
        };

        RoutingPath::from_path(
            t_gate_id,
            factory_id,
            factory_output,
            target_qubit,
            path,
            self.swap_cost,
        )
    }

    /// A* routing with heuristic
    fn astar_route(&self, start: usize, goal: usize) -> Vec<usize> {
        // For now, fallback to Dijkstra
        // Full A* requires coordinates for heuristic
        self.coupling_map.shortest_path(start, goal)
    }

    /// Route batch of magic states (considering congestion)
    pub fn route_batch(
        &self,
        requests: &[(usize, usize, usize, usize)], // (t_gate_id, factory_id, factory_output, target)
    ) -> Vec<RoutingPath> {
        // Simple batch routing: route each independently
        // More sophisticated would detect congestion and reroute

        requests
            .iter()
            .map(|&(t_gate_id, factory_id, factory_output, target)| {
                self.route(t_gate_id, factory_id, factory_output, target)
            })
            .collect()
    }

    /// Analyze congestion for batch routing
    pub fn analyze_congestion(&self, paths: &[RoutingPath]) -> CongestionAnalysis {
        let mut edge_usage: HashMap<(usize, usize), usize> = HashMap::new();

        // Count how many times each edge is used
        for path in paths {
            for &(q1, q2) in &path.swap_chain {
                let edge = (q1.min(q2), q1.max(q2));
                *edge_usage.entry(edge).or_insert(0) += 1;
            }
        }

        let peak_congestion = edge_usage.values().max().copied().unwrap_or(0);

        let total_swaps: usize = paths.iter().map(|p| p.swap_count).sum();

        // Identify hotspots (edges used ≥ 3 times)
        let hotspots: Vec<_> = edge_usage
            .iter()
            .filter(|(_, count)| **count >= 3)
            .map(|(edge, count)| (*edge, *count))
            .collect();

        CongestionAnalysis {
            total_paths: paths.len(),
            total_swaps,
            peak_congestion,
            hotspots,
            avg_swaps_per_path: if paths.is_empty() {
                0.0
            } else {
                total_swaps as f64 / paths.len() as f64
            },
        }
    }
}

// =============================================================================
// Congestion Analysis
// =============================================================================

/// Congestion analysis for batch routing
#[derive(Clone, Debug)]
pub struct CongestionAnalysis {
    /// Total paths routed
    pub total_paths: usize,

    /// Total SWAPs across all paths
    pub total_swaps: usize,

    /// Peak congestion (max SWAPs through single edge)
    pub peak_congestion: usize,

    /// Hotspot edges (used ≥ 3 times)
    pub hotspots: Vec<((usize, usize), usize)>,

    /// Average SWAPs per path
    pub avg_swaps_per_path: f64,
}

impl CongestionAnalysis {
    pub fn report(&self) -> String {
        format!(
            "Routing Congestion Analysis:\n\
             - Total paths: {}\n\
             - Total SWAPs: {}\n\
             - Peak congestion: {} (max SWAPs through single edge)\n\
             - Average SWAPs per path: {:.2}\n\
             - Hotspots ({} edges): {:?}",
            self.total_paths,
            self.total_swaps,
            self.peak_congestion,
            self.avg_swaps_per_path,
            self.hotspots.len(),
            self.hotspots
        )
    }
}

// =============================================================================
// Priority Queue Node for A*
// =============================================================================

#[allow(dead_code)]
#[derive(Clone, Eq, PartialEq)]
struct AStarNode {
    qubit: usize,
    cost: usize,      // g: actual cost from start
    heuristic: usize, // h: estimated cost to goal
}

impl AStarNode {
    #[allow(dead_code)]
    fn f_score(&self) -> usize {
        self.cost + self.heuristic
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: reverse f_score ordering
        other.f_score().cmp(&self.f_score())
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::topology::DeviceTopology;

    #[test]
    fn test_routing_linear_chain() {
        let topology = DeviceTopology::linear_chain(10);
        let coupling = topology.to_coupling_map();
        let router = RoutingEngine::new(coupling, RoutingAlgorithm::Dijkstra);

        let path = router.route(0, 0, 0, 9);

        assert_eq!(path.factory_output, 0);
        assert_eq!(path.target_qubit, 9);
        assert_eq!(path.swap_count, 9); // 9 hops = 9 SWAPs
        assert_eq!(path.path, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_routing_grid() {
        let topology = DeviceTopology::grid_2d(4, 4);
        let coupling = topology.to_coupling_map();
        let router = RoutingEngine::new(coupling, RoutingAlgorithm::Dijkstra);

        // Route from corner to corner
        let path = router.route(0, 0, 0, 15);

        assert_eq!(path.factory_output, 0);
        assert_eq!(path.target_qubit, 15);
        assert_eq!(path.swap_count, 6); // Manhattan distance in 4x4 grid
        assert!(path.routing_cost > 0);
    }

    #[test]
    fn test_batch_routing() {
        let topology = DeviceTopology::linear_chain(20);
        let coupling = topology.to_coupling_map();
        let router = RoutingEngine::new(coupling, RoutingAlgorithm::Dijkstra);

        let requests = vec![
            (0, 0, 0, 10), // T-gate 0: factory 0 → qubit 10
            (1, 0, 0, 15), // T-gate 1: factory 0 → qubit 15
            (2, 0, 0, 5),  // T-gate 2: factory 0 → qubit 5
        ];

        let paths = router.route_batch(&requests);

        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0].swap_count, 10);
        assert_eq!(paths[1].swap_count, 15);
        assert_eq!(paths[2].swap_count, 5);
    }

    #[test]
    fn test_congestion_analysis() {
        let topology = DeviceTopology::linear_chain(10);
        let coupling = topology.to_coupling_map();
        let router = RoutingEngine::new(coupling, RoutingAlgorithm::Dijkstra);

        // Multiple paths through same edges
        let requests = vec![(0, 0, 0, 5), (1, 0, 0, 7), (2, 0, 0, 9)];

        let paths = router.route_batch(&requests);
        let congestion = router.analyze_congestion(&paths);

        assert_eq!(congestion.total_paths, 3);
        assert!(congestion.peak_congestion >= 3); // Edges 0-1, 1-2, etc. used multiple times
        assert!(congestion.avg_swaps_per_path > 0.0);
    }

    #[test]
    fn test_path_to_swaps() {
        let path = vec![0, 1, 2, 3, 4];
        let swaps = RoutingPath::path_to_swaps(&path);

        assert_eq!(swaps, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn test_fully_connected_routing() {
        let topology = DeviceTopology::fully_connected(8);
        let coupling = topology.to_coupling_map();
        let router = RoutingEngine::new(coupling, RoutingAlgorithm::Dijkstra);

        // In fully connected graph, distance is always 1 (direct connection)
        let path = router.route(0, 0, 0, 7);

        assert_eq!(path.swap_count, 1); // Direct connection
    }
}
