//! CPU Datapath - Signal flow through the processor
//!
//! This module defines the datapath structure - how signals flow from
//! one component to another through tiles.

use crate::tile_cpu::{DATAPATH_WIDTH, NUM_REGISTERS};

/// Represents the CPU datapath structure
///
/// The datapath defines signal routing between components. In tile-based
/// implementation, each path is a sequence of Wire tiles.
#[derive(Debug, Clone)]
pub struct TileCpuDatapath {
    /// Width of data buses in bits
    pub data_width: usize,

    /// Number of registers
    pub num_registers: usize,

    /// Pipeline stages (1 = single-cycle, 2+ = pipelined)
    pub pipeline_stages: usize,

    /// Critical path segments and their delays
    pub critical_paths: Vec<CriticalPathSegment>,
}

/// A segment of the critical path
#[derive(Debug, Clone)]
pub struct CriticalPathSegment {
    /// Name of this segment (e.g., "Fetch", "Decode", "Execute")
    pub name: String,

    /// Tiles in this segment
    pub tiles: Vec<PathTile>,

    /// Total propagation delay in deltas
    pub total_delay: u32,
}

/// A tile in a path
#[derive(Debug, Clone, Copy)]
pub struct PathTile {
    /// Grid position
    pub x: usize,
    pub y: usize,

    /// Tile type (for delay calculation)
    pub delay: u8,
}

impl Default for TileCpuDatapath {
    fn default() -> Self {
        Self::single_cycle()
    }
}

impl TileCpuDatapath {
    /// Create a single-cycle datapath
    pub fn single_cycle() -> Self {
        Self {
            data_width: DATAPATH_WIDTH,
            num_registers: NUM_REGISTERS,
            pipeline_stages: 1,
            critical_paths: vec![
                // Fetch: PC → ROM → Instruction
                CriticalPathSegment {
                    name: "Fetch".into(),
                    tiles: vec![],
                    total_delay: 5, // PC(0) + Decode(2) + Mux(3)
                },
                // Decode: Instruction → Control Signals
                CriticalPathSegment {
                    name: "Decode".into(),
                    tiles: vec![],
                    total_delay: 4, // Decoder(2) + And/Or(2)
                },
                // RegRead: Reg Select → Reg Data
                CriticalPathSegment {
                    name: "RegRead".into(),
                    tiles: vec![],
                    total_delay: 2, // Mux(2)
                },
                // Execute: ALU operation
                CriticalPathSegment {
                    name: "Execute".into(),
                    tiles: vec![],
                    total_delay: 5, // Add/Sub(5)
                },
                // WriteBack: ALU → Register
                CriticalPathSegment {
                    name: "WriteBack".into(),
                    tiles: vec![],
                    total_delay: 0, // Synchronous capture
                },
            ],
        }
    }

    /// Create a 2-stage pipelined datapath (Fetch/Execute)
    pub fn two_stage_pipeline() -> Self {
        Self {
            data_width: DATAPATH_WIDTH,
            num_registers: NUM_REGISTERS,
            pipeline_stages: 2,
            critical_paths: vec![
                // Stage 1: Fetch + Decode
                CriticalPathSegment {
                    name: "Fetch/Decode".into(),
                    tiles: vec![],
                    total_delay: 9, // Fetch(5) + Decode(4)
                },
                // Stage 2: Execute + WriteBack
                CriticalPathSegment {
                    name: "Execute/WB".into(),
                    tiles: vec![],
                    total_delay: 7, // RegRead(2) + ALU(5)
                },
            ],
        }
    }

    /// Calculate total critical path delay
    pub fn total_critical_path(&self) -> u32 {
        // For single-cycle, sum all segments
        // For pipelined, take max of stages
        if self.pipeline_stages == 1 {
            self.critical_paths.iter().map(|s| s.total_delay).sum()
        } else {
            self.critical_paths.iter().map(|s| s.total_delay).max().unwrap_or(0)
        }
    }

    /// Estimate maximum frequency in MHz
    ///
    /// Assumes 1ns per delta (configurable in real implementation)
    pub fn estimated_max_mhz(&self, ns_per_delta: f64) -> f64 {
        let critical_ns = self.total_critical_path() as f64 * ns_per_delta;
        if critical_ns > 0.0 {
            1000.0 / critical_ns // MHz
        } else {
            f64::INFINITY
        }
    }

    /// Get routing requirements between stages
    pub fn routing_requirements(&self) -> RoutingRequirements {
        RoutingRequirements {
            // Horizontal buses for data flow
            horizontal_buses: vec![
                Bus { name: "AluA".into(), width: self.data_width, row: 8 },
                Bus { name: "AluB".into(), width: self.data_width, row: 9 },
                Bus { name: "AluOut".into(), width: self.data_width, row: 10 },
                Bus { name: "RegWrite".into(), width: self.data_width, row: 11 },
            ],
            // Vertical buses for register access
            vertical_buses: vec![
                Bus { name: "Reg0".into(), width: self.data_width, row: 0 },
                Bus { name: "Reg1".into(), width: self.data_width, row: 1 },
                Bus { name: "Reg2".into(), width: self.data_width, row: 2 },
                Bus { name: "Reg3".into(), width: self.data_width, row: 3 },
            ],
            // Control signal wires (1-bit each)
            control_wires: 12, // ~12 control signals
        }
    }
}

/// Bus routing requirements
#[derive(Debug, Clone)]
pub struct Bus {
    pub name: String,
    pub width: usize,
    pub row: usize,
}

/// Complete routing requirements for the CPU
#[derive(Debug, Clone)]
pub struct RoutingRequirements {
    pub horizontal_buses: Vec<Bus>,
    pub vertical_buses: Vec<Bus>,
    pub control_wires: usize,
}

impl RoutingRequirements {
    /// Estimate total wire tiles needed
    pub fn estimated_wire_tiles(&self) -> usize {
        let h_tiles: usize = self.horizontal_buses.iter()
            .map(|b| b.width * 16) // Assume 16 tiles horizontal span
            .sum();
        let v_tiles: usize = self.vertical_buses.iter()
            .map(|b| b.width * 8) // Assume 8 tiles vertical span
            .sum();
        let ctrl_tiles = self.control_wires * 10; // ~10 tiles per control wire

        h_tiles + v_tiles + ctrl_tiles
    }
}

// =============================================================================
// Datapath Analysis
// =============================================================================

impl TileCpuDatapath {
    /// Analyze datapath for optimization opportunities
    pub fn analyze(&self) -> DatapathAnalysis {
        let total_delay = self.total_critical_path();
        let routing = self.routing_requirements();

        DatapathAnalysis {
            total_critical_path: total_delay,
            estimated_max_mhz: self.estimated_max_mhz(1.0),
            bottleneck_stage: self.find_bottleneck(),
            pipeline_benefit: self.calculate_pipeline_benefit(),
            routing_overhead: routing.estimated_wire_tiles(),
        }
    }

    fn find_bottleneck(&self) -> String {
        self.critical_paths
            .iter()
            .max_by_key(|s| s.total_delay)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "Unknown".into())
    }

    fn calculate_pipeline_benefit(&self) -> f64 {
        let single = TileCpuDatapath::single_cycle().total_critical_path();
        let pipelined = TileCpuDatapath::two_stage_pipeline().total_critical_path();

        if pipelined > 0 {
            single as f64 / pipelined as f64
        } else {
            1.0
        }
    }
}

/// Results of datapath analysis
#[derive(Debug, Clone)]
pub struct DatapathAnalysis {
    /// Total critical path in deltas
    pub total_critical_path: u32,
    /// Estimated max frequency at 1ns/delta
    pub estimated_max_mhz: f64,
    /// Name of the bottleneck stage
    pub bottleneck_stage: String,
    /// Speedup from pipelining
    pub pipeline_benefit: f64,
    /// Estimated wire tiles for routing
    pub routing_overhead: usize,
}

impl std::fmt::Display for DatapathAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Datapath Analysis")?;
        writeln!(f, "=================")?;
        writeln!(f, "Critical path:    {} deltas", self.total_critical_path)?;
        writeln!(f, "Est. max freq:    {:.1} MHz", self.estimated_max_mhz)?;
        writeln!(f, "Bottleneck:       {}", self.bottleneck_stage)?;
        writeln!(f, "Pipeline speedup: {:.2}x", self.pipeline_benefit)?;
        writeln!(f, "Routing tiles:    ~{}", self.routing_overhead)?;
        Ok(())
    }
}
