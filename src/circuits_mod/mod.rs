// Circuits module - circuit placement, patterns, and constraints

pub mod constrained;
pub mod patterns;
pub mod placement;

// Re-export from placement
pub use placement::{
    CircuitError, CircuitKind, CircuitPort, CircuitTemplate, LogicSpec, PlacementOptions,
    PlacementSummary, PortRole, TileSpec, list_templates, place_circuit, world_ports,
};

// Re-export from patterns
pub use patterns::{
    Blob, DetectionReport, EdgeSummary, FieldSelect, OscillatorCandidate, PatternParams,
    PatternSummary, detect_blobs_u32, detect_edges_u32, detect_oscillators, detect_patterns,
    summarize_field,
};

// Re-export from constrained
pub use constrained::{
    CircuitConstraints, EntanglementLevel, categorize_entanglement, count_entangling_gates,
    entanglement_fraction, is_entangling_gate, mutate_circuit_constrained,
    random_circuit_constrained,
};
