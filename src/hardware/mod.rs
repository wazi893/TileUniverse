//! Hardware Mapping for Magic State Factories
//!
//! SPRINT 52.0: Hardware-Aware Resource Estimation
//!
//! This module provides hardware-specific resource estimation for
//! fault-tolerant quantum computing, mapping abstract magic state
//! budgets to real device constraints.
//!
//! # Overview
//!
//! The hardware mapping system consists of:
//! - **Topology**: Device connectivity (grids, chains, heavy-hex)
//! - **Profiles**: Hardware specifications (gate times, fidelities)
//! - **Factory Layout**: Spatial placement of distillation factories
//! - **SWAP Router**: Routing for restricted connectivity
//! - **Aware Budget**: Hardware-adjusted resource estimates
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::MagicStateBudget;
//! use engine::hardware::{HardwareAwareBudget, profiles};
//!
//! // Create abstract budget
//! let budget = MagicStateBudget::for_qram_routing(8, 100);
//!
//! // Map to IBM Heron
//! let ibm = profiles::ibm_heron();
//! let hw_budget = HardwareAwareBudget::from_budget(&budget, &ibm, 2);
//!
//! println!("Physical qubits: {}", hw_budget.physical_qubits);
//! println!("SWAP overhead: {}", hw_budget.swap_gates);
//! println!("Adjusted time: {:.3}s", hw_budget.adjusted_time_seconds);
//! ```
//!
//! # Hardware Comparison
//!
//! ```ignore
//! use engine::hardware::{compare_on_hardware, format_hardware_comparison, profiles};
//!
//! let budget = MagicStateBudget::for_qram_routing(8, 100);
//! let profiles = vec![
//!     profiles::ibm_heron(),
//!     profiles::ionq_aria(),
//!     profiles::google_sycamore(),
//! ];
//!
//! let budgets = compare_on_hardware(&budget, &profiles.iter().collect(), 2);
//! println!("{}", format_hardware_comparison(&budgets));
//! ```

pub mod aware_budget;
pub mod factory_layout;
pub mod profile;
pub mod swap_router;
pub mod topology;

// Sprint 64.0: Hardware-Topology Factory Placement
pub mod layout_viz;
pub mod placement;
pub mod routing;

// Re-exports for convenient access
pub use aware_budget::{
    BudgetComparison, ErrorSource, FidelityBreakdown, HardwareAwareBudget, compare_on_hardware,
    format_hardware_comparison,
};
pub use factory_layout::{
    FactoryFootprint, FactoryLayoutOptimizer, FactoryPlacement, LayoutResources,
};
pub use profile::{
    DeviceTechnology, HardwareProfile, ProfileComparison, ProfileMetrics, SwapOverhead,
    compare_profiles, profiles,
};
pub use swap_router::{
    GateCounts, LogicalGate, PhysicalGate, RoutedCircuit, SwapGate, SwapOverheadStats, SwapRouter,
};
pub use topology::{DeviceTopology, EnhancedCouplingMap, common as topologies};

// Sprint 64.0: Hardware-Topology Factory Placement re-exports
pub use layout_viz::{LayoutVisualization, LayoutVisualizer};
pub use placement::{
    FactoryLocation, LayoutMetrics, PhysicalFactoryLayout, PlacementOptimizer, PlacementStrategy,
};
pub use routing::{CongestionAnalysis, RoutingAlgorithm, RoutingEngine, RoutingPath};
