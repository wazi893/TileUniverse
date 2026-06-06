//! Network communication
//!
//! Distributed simulation networking.

pub mod net;
pub mod net_summary;

// Re-export main types
// Explicit re-exports to avoid ambiguity
pub use net::{NetId, NetKind, NetReport, NetStats, analyze_nets};
pub use net_summary::{GlobalSummary, RegionSummary, SummaryReport, summarize, summarize_nets};
