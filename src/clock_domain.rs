use std::cell::Cell;

/// Definition of a clock domain.
pub struct ClockDomainDef {
    pub name: String,
    /// Divider ratio from base clock. 1 = same as global, 2 = half freq, etc.
    pub divider: u32,
    /// Phase offset in base ticks (0 = aligned with global).
    pub phase_offset: u32,
}

/// Runtime state of a clock domain.
pub struct ClockDomainState {
    pub clock: bool,
    pub prev_clock: bool,
    /// Tick counter for divider logic.
    pub counter: u64,
}

/// State for a Synchronizer tile (2-FF CDC synchronizer).
pub struct SynchronizerState {
    /// Index into clock_domain_defs for the destination domain.
    pub domain_idx: usize,
    /// First flip-flop stage.
    pub stage1: Cell<u64>,
    /// Second flip-flop stage (this is the output).
    pub stage2: Cell<u64>,
}
