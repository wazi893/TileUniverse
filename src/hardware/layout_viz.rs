// =============================================================================
// Factory Layout Visualization - ASCII Art and Reporting
// =============================================================================
// Sprint 64.0: Hardware-Topology Factory Placement
//
// Provides ASCII art visualization of factory layouts, routing paths,
// and congestion heatmaps for debugging and analysis.

use super::placement::PhysicalFactoryLayout;
use super::routing::RoutingPath;
use super::topology::{DeviceTopology, EnhancedCouplingMap};
use std::collections::HashMap;

// =============================================================================
// Layout Visualizer
// =============================================================================

/// Visualizer for factory layouts
pub struct LayoutVisualizer {
    /// Device topology
    topology: DeviceTopology,

    /// Coupling map
    #[allow(dead_code)]
    coupling: EnhancedCouplingMap,
}

impl LayoutVisualizer {
    /// Create visualizer
    pub fn new(topology: DeviceTopology) -> Self {
        let coupling = topology.to_coupling_map();
        LayoutVisualizer { topology, coupling }
    }

    /// Visualize layout as ASCII art
    pub fn visualize_layout(&self, layout: &PhysicalFactoryLayout) -> String {
        match &self.topology {
            DeviceTopology::LinearChain { length } => self.visualize_linear_layout(layout, *length),
            DeviceTopology::Grid2D { width, height } => {
                self.visualize_grid_layout(layout, *width, *height)
            }
            DeviceTopology::HeavyHex { rows, cols } => {
                self.visualize_heavy_hex_layout(layout, *rows, *cols)
            }
            _ => self.visualize_generic_layout(layout),
        }
    }

    /// Visualize linear chain topology
    fn visualize_linear_layout(&self, layout: &PhysicalFactoryLayout, length: usize) -> String {
        let mut lines = Vec::new();

        lines.push(format!("=== Linear Chain ({} qubits) ===\n", length));

        // Draw qubits
        let mut qubit_line = String::new();
        for q in 0..length {
            qubit_line.push_str(&format!("[{:2}]", q));
            if q + 1 < length {
                qubit_line.push('─');
            }
        }
        lines.push(qubit_line);
        lines.push(String::new());

        // Show factory locations
        lines.push("Factory Locations:".to_string());
        for (id, loc) in &layout.factory_locations {
            let range_str = if loc.qubit_range.len() <= 5 {
                format!("{:?}", loc.qubit_range)
            } else {
                format!(
                    "[{}..{}]",
                    loc.qubit_range.first().unwrap_or(&0),
                    loc.qubit_range.last().unwrap_or(&0)
                )
            };
            lines.push(format!(
                "  Factory {}: qubits {} (center: {})",
                id, range_str, loc.center
            ));
        }
        lines.push(String::new());

        // Show routing paths
        if !layout.routing_plan.is_empty() {
            lines.push("Routing Paths:".to_string());
            for (t_id, path) in &layout.routing_plan {
                lines.push(format!(
                    "  T{} (factory {} → qubit {}): {} SWAPs",
                    t_id, path.factory_id, path.target_qubit, path.swap_count
                ));
            }
            lines.push(String::new());
        }

        // Summary
        lines.push(format!("Total SWAPs: {}", layout.metrics.total_swaps));
        lines.push(format!(
            "Peak Congestion: {}",
            layout.metrics.peak_congestion
        ));

        lines.join("\n")
    }

    /// Visualize 2D grid topology
    fn visualize_grid_layout(
        &self,
        layout: &PhysicalFactoryLayout,
        width: usize,
        height: usize,
    ) -> String {
        let mut lines = Vec::new();

        lines.push(format!("=== 2D Grid ({}×{} qubits) ===\n", width, height));

        // Build factory map
        let mut factory_map: HashMap<usize, usize> = HashMap::new();
        for (id, loc) in &layout.factory_locations {
            for &q in &loc.qubit_range {
                factory_map.insert(q, *id);
            }
        }

        // Draw grid
        for row in 0..height {
            let mut row_line = String::new();
            for col in 0..width {
                let q = row * width + col;
                if let Some(&factory_id) = factory_map.get(&q) {
                    row_line.push_str(&format!("[F{}]", factory_id));
                } else {
                    row_line.push_str(&format!("[{:2}]", q));
                }
                if col + 1 < width {
                    row_line.push('─');
                }
            }
            lines.push(row_line);

            // Vertical connections
            if row + 1 < height {
                let mut vert_line = String::new();
                for col in 0..width {
                    vert_line.push_str(" │  ");
                    if col + 1 < width {
                        vert_line.push(' ');
                    }
                }
                lines.push(vert_line);
            }
        }
        lines.push(String::new());

        // Factory legend
        lines.push("Factory Locations:".to_string());
        for (id, loc) in &layout.factory_locations {
            lines.push(format!(
                "  F{}: {} qubits (center: {})",
                id,
                loc.size(),
                loc.center
            ));
        }
        lines.push(String::new());

        // Routing stats
        lines.push(format!("Total SWAPs: {}", layout.metrics.total_swaps));
        lines.push(format!(
            "Avg SWAPs per T-gate: {:.2}",
            layout.metrics.avg_swaps_per_t
        ));

        lines.join("\n")
    }

    /// Visualize heavy-hex topology (simplified)
    fn visualize_heavy_hex_layout(
        &self,
        layout: &PhysicalFactoryLayout,
        rows: usize,
        cols: usize,
    ) -> String {
        let mut lines = Vec::new();

        lines.push(format!("=== Heavy-Hex Topology ({}×{}) ===\n", rows, cols));

        // For heavy-hex, use simplified grid representation
        lines.push("(Simplified grid representation of heavy-hex)".to_string());
        lines.push(String::new());

        // Factory locations
        lines.push("Factory Locations:".to_string());
        for (id, loc) in &layout.factory_locations {
            lines.push(format!(
                "  Factory {}: qubits {:?} (center: {})",
                id, loc.qubit_range, loc.center
            ));
        }
        lines.push(String::new());

        // Routing stats
        lines.push(format!("Total SWAPs: {}", layout.metrics.total_swaps));
        lines.push(format!(
            "Peak Congestion: {}",
            layout.metrics.peak_congestion
        ));

        lines.join("\n")
    }

    /// Generic layout visualization
    fn visualize_generic_layout(&self, layout: &PhysicalFactoryLayout) -> String {
        let mut lines = Vec::new();

        lines.push(format!("=== {} Topology ===\n", self.topology.name()));

        // Factory locations
        lines.push("Factory Locations:".to_string());
        for (id, loc) in &layout.factory_locations {
            lines.push(format!(
                "  Factory {}: {} qubits at {:?} (center: {})",
                id,
                loc.size(),
                loc.qubit_range,
                loc.center
            ));
        }
        lines.push(String::new());

        // Metrics
        lines.push(layout.metrics_report());

        lines.join("\n")
    }

    /// Visualize congestion heatmap
    pub fn visualize_congestion(&self, layout: &PhysicalFactoryLayout) -> String {
        let mut lines = Vec::new();

        lines.push("=== Routing Congestion Heatmap ===\n".to_string());

        // Count edge usage
        let mut edge_usage: HashMap<(usize, usize), usize> = HashMap::new();
        for path in layout.routing_plan.values() {
            for &(q1, q2) in &path.swap_chain {
                let edge = (q1.min(q2), q1.max(q2));
                *edge_usage.entry(edge).or_insert(0) += 1;
            }
        }

        if edge_usage.is_empty() {
            lines.push("No routing paths (all factories co-located with targets)".to_string());
            return lines.join("\n");
        }

        // Sort by usage
        let mut sorted_edges: Vec<_> = edge_usage.iter().collect();
        sorted_edges.sort_by(|a, b| b.1.cmp(a.1));

        // Show top 10 most congested edges
        lines.push("Top Congested Edges:".to_string());
        for (i, (edge, count)) in sorted_edges.iter().take(10).enumerate() {
            let bar = self.make_bar(**count, layout.metrics.peak_congestion);
            lines.push(format!(
                "  {}. Edge {:?}: {} SWAPs {}",
                i + 1,
                edge,
                count,
                bar
            ));
        }
        lines.push(String::new());

        lines.push(format!(
            "Peak Congestion: {}",
            layout.metrics.peak_congestion
        ));
        lines.push(format!("Total Edges Used: {}", edge_usage.len()));

        lines.join("\n")
    }

    /// Make ASCII bar for visualization
    fn make_bar(&self, value: usize, max_value: usize) -> String {
        if max_value == 0 {
            return String::new();
        }
        let width = 20;
        let bar_len = (value * width) / max_value;
        "█".repeat(bar_len)
    }

    /// Visualize specific routing path
    pub fn visualize_path(&self, path: &RoutingPath) -> String {
        let mut lines = Vec::new();

        lines.push(format!(
            "=== Routing Path for T-gate {} ===\n",
            path.t_gate_id
        ));

        lines.push(format!(
            "Factory {} (output qubit {}) → Target qubit {}",
            path.factory_id, path.factory_output, path.target_qubit
        ));
        lines.push(format!("Path: {:?}", path.path));
        lines.push(format!(
            "SWAPs: {} ({} cycles)",
            path.swap_count, path.routing_cost
        ));

        if !path.swap_chain.is_empty() {
            lines.push("\nSWAP Chain:".to_string());
            for (i, &(q1, q2)) in path.swap_chain.iter().enumerate() {
                lines.push(format!("  {}. SWAP({}, {})", i + 1, q1, q2));
            }
        }

        lines.join("\n")
    }
}

// =============================================================================
// Extension Trait for PhysicalFactoryLayout
// =============================================================================

/// Extension methods for layout visualization
pub trait LayoutVisualization {
    /// Generate metrics report
    fn metrics_report(&self) -> String;

    /// Generate full report with visualization
    fn full_report(&self, visualizer: &LayoutVisualizer) -> String;
}

impl LayoutVisualization for PhysicalFactoryLayout {
    fn metrics_report(&self) -> String {
        format!(
            "Metrics:\n\
             - Factories: {}\n\
             - T-gates: {}\n\
             - Total SWAPs: {}\n\
             - SWAP cost: {} cycles\n\
             - Peak congestion: {}\n\
             - Avg SWAPs/T-gate: {:.2}",
            self.metrics.total_factories,
            self.metrics.total_t_gates,
            self.metrics.total_swaps,
            self.metrics.total_swap_cost,
            self.metrics.peak_congestion,
            self.metrics.avg_swaps_per_t
        )
    }

    fn full_report(&self, visualizer: &LayoutVisualizer) -> String {
        let mut parts = Vec::new();

        parts.push(visualizer.visualize_layout(self));
        parts.push("\n".to_string());
        parts.push(visualizer.visualize_congestion(self));

        parts.join("\n")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::placement::FactoryLocation;

    #[test]
    fn test_linear_visualization() {
        let topology = DeviceTopology::linear_chain(20);
        let viz = LayoutVisualizer::new(topology);

        let mut layout = PhysicalFactoryLayout::new();
        layout
            .factory_locations
            .insert(0, FactoryLocation::new(0, vec![0, 1, 2, 3, 4, 5, 6]));

        let output = viz.visualize_layout(&layout);

        assert!(output.contains("Linear Chain"));
        assert!(output.contains("Factory 0"));
    }

    #[test]
    fn test_grid_visualization() {
        let topology = DeviceTopology::grid_2d(4, 4);
        let viz = LayoutVisualizer::new(topology);

        let mut layout = PhysicalFactoryLayout::new();
        layout
            .factory_locations
            .insert(0, FactoryLocation::new(0, vec![0, 1, 4, 5]));

        let output = viz.visualize_layout(&layout);

        assert!(output.contains("2D Grid"));
        assert!(output.contains("Factory Locations"));
    }

    #[test]
    fn test_congestion_visualization() {
        let topology = DeviceTopology::linear_chain(20);
        let _coupling = topology.to_coupling_map();
        let viz = LayoutVisualizer::new(topology);

        let mut layout = PhysicalFactoryLayout::new();

        // Add mock routing paths
        let path = RoutingPath {
            t_gate_id: 0,
            factory_id: 0,
            factory_output: 0,
            target_qubit: 10,
            path: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            swap_chain: vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
            ],
            swap_count: 10,
            routing_cost: 30,
        };

        layout.routing_plan.insert(0, path);
        layout.calculate_metrics();

        let output = viz.visualize_congestion(&layout);

        assert!(output.contains("Congestion Heatmap"));
        assert!(output.contains("Peak Congestion"));
    }

    #[test]
    fn test_path_visualization() {
        let topology = DeviceTopology::linear_chain(10);
        let viz = LayoutVisualizer::new(topology);

        let path = RoutingPath {
            t_gate_id: 0,
            factory_id: 0,
            factory_output: 0,
            target_qubit: 5,
            path: vec![0, 1, 2, 3, 4, 5],
            swap_chain: vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            swap_count: 5,
            routing_cost: 15,
        };

        let output = viz.visualize_path(&path);

        assert!(output.contains("Routing Path"));
        assert!(output.contains("SWAP"));
        assert!(output.contains("Factory 0"));
    }

    #[test]
    fn test_metrics_report() {
        let mut layout = PhysicalFactoryLayout::new();
        layout.metrics.total_factories = 2;
        layout.metrics.total_t_gates = 10;
        layout.metrics.total_swaps = 50;
        layout.metrics.avg_swaps_per_t = 5.0;

        let report = layout.metrics_report();

        assert!(report.contains("Factories: 2"));
        assert!(report.contains("T-gates: 10"));
        assert!(report.contains("Total SWAPs: 50"));
    }
}
