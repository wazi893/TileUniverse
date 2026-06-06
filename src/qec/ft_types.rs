//! Fault-Tolerant Quantum Computing Types
//!
//! Core types for fault-tolerant circuit execution with logical qubits.
//! These types follow the principle: "Surface code is spacetime, not just gates."
//!
//! # Design Principles
//!
//! 1. **LogicalQubit is minimal**: Just identity + basis + Pauli frame
//! 2. **Error budget has structure**: Separate QEC, injection, and idle errors
//! 3. **Costs are spacetime**: Everything measured in cycles, not gate counts
//! 4. **Two modes**: Abstract estimation vs sampled Pauli-frame simulation

use std::fmt;

// ============================================================================
// Pauli Frame Tracking
// ============================================================================

/// Pauli frame accumulated from measurements and corrections.
///
/// In fault-tolerant quantum computing, we often track Pauli corrections
/// in software (the "frame") rather than applying them physically.
/// This is because Pauli gates commute with Clifford operations in
/// predictable ways.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PauliFrame {
    /// X correction accumulated (bit flip)
    pub x: bool,
    /// Z correction accumulated (phase flip)
    pub z: bool,
}

impl PauliFrame {
    /// Identity frame (no corrections needed)
    pub const fn identity() -> Self {
        Self { x: false, z: false }
    }

    /// X frame only
    pub const fn x() -> Self {
        Self { x: true, z: false }
    }

    /// Z frame only
    pub const fn z() -> Self {
        Self { x: false, z: true }
    }

    /// Y frame (both X and Z)
    pub const fn y() -> Self {
        Self { x: true, z: true }
    }

    /// Compose two frames: self followed by other
    ///
    /// Frame composition is just XOR since Pauli gates square to identity.
    #[inline]
    pub fn compose(&mut self, other: &PauliFrame) {
        self.x ^= other.x;
        self.z ^= other.z;
    }

    /// Compose and return new frame
    #[inline]
    pub fn composed_with(&self, other: &PauliFrame) -> PauliFrame {
        PauliFrame {
            x: self.x ^ other.x,
            z: self.z ^ other.z,
        }
    }

    /// Apply Hadamard conjugation to frame: H P H† = P'
    ///
    /// H X H† = Z, H Z H† = X
    #[inline]
    pub fn conjugate_h(&mut self) {
        std::mem::swap(&mut self.x, &mut self.z);
    }

    /// Apply S gate conjugation to frame: S P S† = P'
    ///
    /// S X S† = Y = iXZ, S Z S† = Z
    /// In frame terms: X picks up Z
    #[inline]
    pub fn conjugate_s(&mut self) {
        if self.x {
            self.z ^= true;
        }
    }

    /// Apply S† gate conjugation to frame
    ///
    /// S† X S = -Y = -iXZ, S† Z S = Z
    #[inline]
    pub fn conjugate_sdg(&mut self) {
        // Same as S for frame purposes (global phase doesn't matter)
        self.conjugate_s();
    }

    /// Check if frame is identity
    #[inline]
    pub fn is_identity(&self) -> bool {
        !self.x && !self.z
    }

    /// Convert to Pauli index: I=0, X=1, Z=2, Y=3
    #[inline]
    pub fn to_pauli_index(&self) -> u8 {
        (self.x as u8) | ((self.z as u8) << 1)
    }

    /// Create from Pauli index
    #[inline]
    pub fn from_pauli_index(idx: u8) -> Self {
        Self {
            x: (idx & 1) != 0,
            z: (idx & 2) != 0,
        }
    }
}

impl fmt::Display for PauliFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.x, self.z) {
            (false, false) => write!(f, "I"),
            (true, false) => write!(f, "X"),
            (false, true) => write!(f, "Z"),
            (true, true) => write!(f, "Y"),
        }
    }
}

// ============================================================================
// Logical Basis States
// ============================================================================

/// Known logical basis state.
///
/// After certain operations (especially non-Clifford gates), the logical
/// state may become unknown or only known up to a Pauli frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalBasis {
    /// Z-basis eigenstate: |0⟩ (false) or |1⟩ (true)
    Z(bool),
    /// X-basis eigenstate: |+⟩ (false) or |-⟩ (true)
    X(bool),
}

impl LogicalBasis {
    /// |0⟩ state
    pub const ZERO: Self = Self::Z(false);
    /// |1⟩ state
    pub const ONE: Self = Self::Z(true);
    /// |+⟩ state
    pub const PLUS: Self = Self::X(false);
    /// |-⟩ state
    pub const MINUS: Self = Self::X(true);

    /// Apply X gate to basis state
    pub fn apply_x(self) -> Self {
        match self {
            Self::Z(b) => Self::Z(!b),
            Self::X(b) => Self::X(b), // X|+⟩ = |+⟩, X|-⟩ = |-⟩ (up to phase)
        }
    }

    /// Apply Z gate to basis state
    pub fn apply_z(self) -> Self {
        match self {
            Self::Z(b) => Self::Z(b), // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩ (phase)
            Self::X(b) => Self::X(!b),
        }
    }

    /// Apply Hadamard gate to basis state
    pub fn apply_h(self) -> Self {
        match self {
            Self::Z(b) => Self::X(b),
            Self::X(b) => Self::Z(b),
        }
    }
}

impl fmt::Display for LogicalBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Z(false) => write!(f, "|0⟩"),
            Self::Z(true) => write!(f, "|1⟩"),
            Self::X(false) => write!(f, "|+⟩"),
            Self::X(true) => write!(f, "|-⟩"),
        }
    }
}

// ============================================================================
// Known Basis (Phase 1.3) - Unified basis tracking with unknown reason
// ============================================================================

/// Complete basis state tracking including unknown states.
///
/// This unifies `Option<LogicalBasis>` + `UnknownReason` into a single enum,
/// enabling frame-aware measurement interpretation:
///
/// - **Z-measurement on Z(v)**: deterministic result `v ^ frame.x`
/// - **X-measurement on X(v)**: deterministic result `v ^ frame.z`
/// - **Measurement on Unknown**: must sample randomly
///
/// # Design Rationale
///
/// The key insight is that knowing the basis enables deterministic measurement
/// in that basis. CNOT on |+⟩|0⟩ creates entanglement (truly unknown), but
/// H on |0⟩ gives |+⟩ which is still trackable for X-measurement purposes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnownBasis {
    /// Z-basis eigenstate: |0⟩ (false) or |1⟩ (true)
    ///
    /// Z-measurement is deterministic: result = value ^ frame.x
    Z(bool),

    /// X-basis eigenstate: |+⟩ (false) or |-⟩ (true)
    ///
    /// X-measurement is deterministic: result = value ^ frame.z
    X(bool),

    /// State is unknown/entangled - measurement must sample
    Unknown(UnknownReason),
}

impl KnownBasis {
    /// |0⟩ state
    pub const ZERO: Self = Self::Z(false);
    /// |1⟩ state
    pub const ONE: Self = Self::Z(true);
    /// |+⟩ state
    pub const PLUS: Self = Self::X(false);
    /// |-⟩ state
    pub const MINUS: Self = Self::X(true);

    /// Create from LogicalBasis (always known)
    pub fn from_logical(basis: LogicalBasis) -> Self {
        match basis {
            LogicalBasis::Z(v) => Self::Z(v),
            LogicalBasis::X(v) => Self::X(v),
        }
    }

    /// Convert to LogicalBasis if known, None if unknown
    pub fn to_logical(&self) -> Option<LogicalBasis> {
        match self {
            Self::Z(v) => Some(LogicalBasis::Z(*v)),
            Self::X(v) => Some(LogicalBasis::X(*v)),
            Self::Unknown(_) => None,
        }
    }

    /// Check if the state is known (not Unknown)
    pub fn is_known(&self) -> bool {
        !matches!(self, Self::Unknown(_))
    }

    /// Check if the state is unknown
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }

    /// Check if in Z-basis (|0⟩ or |1⟩)
    pub fn is_z_basis(&self) -> bool {
        matches!(self, Self::Z(_))
    }

    /// Check if in X-basis (|+⟩ or |-⟩)
    pub fn is_x_basis(&self) -> bool {
        matches!(self, Self::X(_))
    }

    /// Get the reason if unknown, None if known
    pub fn unknown_reason(&self) -> Option<UnknownReason> {
        match self {
            Self::Unknown(reason) => Some(*reason),
            _ => None,
        }
    }

    /// Apply X gate to basis state
    ///
    /// - Z(v) → Z(!v)  (bit flip)
    /// - X(v) → X(v)   (eigenstate, up to phase)
    /// - Unknown → Unknown (reason preserved)
    pub fn apply_x(self) -> Self {
        match self {
            Self::Z(v) => Self::Z(!v),
            Self::X(v) => Self::X(v), // X|+⟩ = |+⟩, X|-⟩ = |-⟩
            Self::Unknown(r) => Self::Unknown(r),
        }
    }

    /// Apply Z gate to basis state
    ///
    /// - Z(v) → Z(v)   (eigenstate, up to phase)
    /// - X(v) → X(!v)  (phase flip)
    /// - Unknown → Unknown (reason preserved)
    pub fn apply_z(self) -> Self {
        match self {
            Self::Z(v) => Self::Z(v), // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
            Self::X(v) => Self::X(!v),
            Self::Unknown(r) => Self::Unknown(r),
        }
    }

    /// Apply Hadamard gate to basis state
    ///
    /// - Z(v) → X(v)   (|0⟩ → |+⟩, |1⟩ → |-⟩)
    /// - X(v) → Z(v)   (|+⟩ → |0⟩, |-⟩ → |1⟩)
    /// - Unknown → Unknown (reason preserved)
    pub fn apply_h(self) -> Self {
        match self {
            Self::Z(v) => Self::X(v),
            Self::X(v) => Self::Z(v),
            Self::Unknown(r) => Self::Unknown(r),
        }
    }

    /// Get Z-measurement result if deterministic (state is in Z-basis)
    ///
    /// Returns Some(value) if Z-measurement would be deterministic,
    /// None if measurement must sample.
    ///
    /// Note: Caller must apply frame correction: `result ^ frame.x`
    pub fn z_measurement_value(&self) -> Option<bool> {
        match self {
            Self::Z(v) => Some(*v),
            _ => None,
        }
    }

    /// Get X-measurement result if deterministic (state is in X-basis)
    ///
    /// Returns Some(value) if X-measurement would be deterministic,
    /// None if measurement must sample.
    ///
    /// Note: Caller must apply frame correction: `result ^ frame.z`
    pub fn x_measurement_value(&self) -> Option<bool> {
        match self {
            Self::X(v) => Some(*v),
            _ => None,
        }
    }
}

impl Default for KnownBasis {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for KnownBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Z(false) => write!(f, "|0⟩"),
            Self::Z(true) => write!(f, "|1⟩"),
            Self::X(false) => write!(f, "|+⟩"),
            Self::X(true) => write!(f, "|-⟩"),
            Self::Unknown(reason) => write!(f, "?[{}]", reason),
        }
    }
}

// ============================================================================
// Unknown Reason (Phase 3.2)
// ============================================================================

/// Reason why a qubit's basis state became unknown.
///
/// This is critical for user trust - when basis tracking says "unknown",
/// users need to know WHY. This distinguishes between:
/// - Entanglement (CNOT on superposition creates Bell state)
/// - Non-Clifford gates (T on X-basis creates non-stabilizer state)
/// - Operations with already-unknown qubits
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownReason {
    /// CNOT/CZ applied when control or target was in superposition.
    /// Example: CNOT on |+⟩|0⟩ creates entangled Bell state (|00⟩ + |11⟩)/√2
    EntanglingGateOnSuperposition,

    /// Two-qubit gate where partner qubit had unknown state.
    /// Entanglement with unknown state propagates uncertainty.
    EntanglingGateWithUnknown,

    /// T gate applied to X-basis state (superposition).
    /// T|+⟩ is not a stabilizer state - cannot be tracked by Pauli frame.
    NonCliffordOnSuperposition,

    /// T gate applied when basis was already unknown.
    NonCliffordOnUnknown,

    /// State was never initialized (shouldn't happen in normal use).
    NeverInitialized,

    /// Explicit reset to unknown (for testing or special cases).
    Explicit,
}

impl UnknownReason {
    /// Human-readable description of why the state is unknown.
    pub fn description(&self) -> &'static str {
        match self {
            Self::EntanglingGateOnSuperposition => {
                "Entangling gate (CNOT/CZ) on superposition creates entanglement"
            }
            Self::EntanglingGateWithUnknown => "Entangling gate with qubit in unknown state",
            Self::NonCliffordOnSuperposition => {
                "Non-Clifford gate (T) on superposition creates non-stabilizer state"
            }
            Self::NonCliffordOnUnknown => "Non-Clifford gate on qubit with unknown state",
            Self::NeverInitialized => "Qubit was never initialized",
            Self::Explicit => "Explicitly marked as unknown",
        }
    }

    /// Is this due to entanglement?
    pub fn is_entanglement(&self) -> bool {
        matches!(
            self,
            Self::EntanglingGateOnSuperposition | Self::EntanglingGateWithUnknown
        )
    }

    /// Is this due to non-Clifford operations?
    pub fn is_non_clifford(&self) -> bool {
        matches!(
            self,
            Self::NonCliffordOnSuperposition | Self::NonCliffordOnUnknown
        )
    }
}

impl fmt::Display for UnknownReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntanglingGateOnSuperposition => write!(f, "entangling_superposition"),
            Self::EntanglingGateWithUnknown => write!(f, "entangling_unknown"),
            Self::NonCliffordOnSuperposition => write!(f, "non_clifford_superposition"),
            Self::NonCliffordOnUnknown => write!(f, "non_clifford_unknown"),
            Self::NeverInitialized => write!(f, "never_initialized"),
            Self::Explicit => write!(f, "explicit"),
        }
    }
}

// ============================================================================
// Simulation Fidelity Configuration
// ============================================================================

/// Controls how accurately errors are simulated during execution.
///
/// Different fidelity levels trade off speed vs accuracy:
/// - **Abstract**: Fastest, uses analytical error formulas from QecModel
/// - **PhenomenologicalDecoder**: Applies noise per QEC round and runs actual decoder
///
/// # Example
///
/// ```ignore
/// use engine::qec::{FTConfig, SimulationFidelity, DecoderChoice};
///
/// // Fast estimation mode (default)
/// let config = FTConfig::new(7, 1e-3);
///
/// // Full decoder simulation
/// let config = FTConfig::new(7, 1e-3)
///     .with_fidelity(SimulationFidelity::PhenomenologicalDecoder {
///         decoder: DecoderChoice::UnionFind,
///         error_rate: 1e-3,
///     });
/// ```
#[derive(Clone, Debug)]
pub enum SimulationFidelity {
    /// Abstract error model using analytical QEC formulas.
    ///
    /// Fast and scalable - error probability is computed from:
    /// `p_logical = 0.1 × (p_phys / p_threshold)^((d+1)/2)` per cycle
    ///
    /// Use for resource estimation of large circuits.
    Abstract,

    /// Phenomenological noise model with decoder execution.
    ///
    /// Slower but more realistic:
    /// 1. Apply random errors per QEC round based on error_rate
    /// 2. Run decoder to attempt correction
    /// 3. Track decoder success/failure
    ///
    /// Use for verifying decoder behavior and realistic error statistics.
    PhenomenologicalDecoder {
        /// Which decoder to use
        decoder: DecoderChoice,
        /// Physical error rate per round (applied to syndrome extraction)
        error_rate: f64,
    },
}

impl SimulationFidelity {
    /// Create abstract (fast) simulation mode
    pub fn fast() -> Self {
        Self::Abstract
    }

    /// Create phenomenological decoder mode with Union-Find
    pub fn with_union_find(error_rate: f64) -> Self {
        Self::PhenomenologicalDecoder {
            decoder: DecoderChoice::UnionFind,
            error_rate,
        }
    }

    /// Create phenomenological decoder mode with MWPM
    pub fn with_mwpm(error_rate: f64) -> Self {
        Self::PhenomenologicalDecoder {
            decoder: DecoderChoice::MWPM,
            error_rate,
        }
    }

    /// Check if this is abstract mode
    pub fn is_abstract(&self) -> bool {
        matches!(self, Self::Abstract)
    }

    /// Check if decoder simulation is enabled
    pub fn uses_decoder(&self) -> bool {
        matches!(self, Self::PhenomenologicalDecoder { .. })
    }
}

impl Default for SimulationFidelity {
    fn default() -> Self {
        Self::Abstract
    }
}

impl fmt::Display for SimulationFidelity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Abstract => write!(f, "Abstract (analytical QEC model)"),
            Self::PhenomenologicalDecoder {
                decoder,
                error_rate,
            } => {
                write!(f, "Phenomenological ({}, p={:.2e})", decoder, error_rate)
            }
        }
    }
}

/// Choice of decoder for phenomenological simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderChoice {
    /// Union-Find decoder (fast, ~7% threshold)
    UnionFind,
    /// Minimum Weight Perfect Matching (slower, ~10% threshold)
    MWPM,
}

impl DecoderChoice {
    /// Get the approximate threshold for this decoder
    pub fn threshold(&self) -> f64 {
        match self {
            Self::UnionFind => 0.07,
            Self::MWPM => 0.103,
        }
    }
}

impl fmt::Display for DecoderChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnionFind => write!(f, "Union-Find"),
            Self::MWPM => write!(f, "MWPM"),
        }
    }
}

// ============================================================================
// Logical Qubit
// ============================================================================

/// A logical qubit: identity + basis state + Pauli frame.
///
/// This is intentionally minimal. The LogicalQubit does NOT contain:
/// - Stabilizer tableau (that's in the QEC code)
/// - Decoder (that's in QecModel)
/// - Error probability (that's in LogicalErrorBudget)
///
/// The LogicalQubit just tracks:
/// - Which logical qubit this is (id)
/// - What basis state it's in (Z, X, or Unknown with reason)
/// - What Pauli frame has accumulated from measurements
///
/// # Phase 1.3 Design
///
/// Uses `KnownBasis` which unifies the previous `Option<LogicalBasis>` and
/// `unknown_reason` into a single enum. This enables frame-aware measurement:
/// - Z-measurement on Z-basis: deterministic
/// - X-measurement on X-basis: deterministic
/// - Any measurement on Unknown: must sample
#[derive(Clone, Debug)]
pub struct LogicalQubit {
    /// Unique identifier
    pub id: usize,
    /// Basis state (Z, X, or Unknown with reason)
    pub basis: KnownBasis,
    /// Accumulated Pauli frame from measurements
    pub frame: PauliFrame,
}

impl LogicalQubit {
    /// Create a new logical qubit in |0⟩ state
    pub fn new(id: usize) -> Self {
        Self {
            id,
            basis: KnownBasis::ZERO,
            frame: PauliFrame::identity(),
        }
    }

    /// Create logical qubit with specified initial state (LogicalBasis)
    pub fn with_state(id: usize, basis: LogicalBasis) -> Self {
        Self {
            id,
            basis: KnownBasis::from_logical(basis),
            frame: PauliFrame::identity(),
        }
    }

    /// Create logical qubit with KnownBasis (includes Unknown)
    pub fn with_known_basis(id: usize, basis: KnownBasis) -> Self {
        Self {
            id,
            basis,
            frame: PauliFrame::identity(),
        }
    }

    /// Reset to |0⟩ state, clearing frame
    pub fn reset(&mut self) {
        self.basis = KnownBasis::ZERO;
        self.frame = PauliFrame::identity();
    }

    /// Apply X gate (updates basis, frame unchanged)
    pub fn apply_x(&mut self) {
        self.basis = self.basis.apply_x();
    }

    /// Apply Z gate (updates basis, frame unchanged)
    pub fn apply_z(&mut self) {
        self.basis = self.basis.apply_z();
    }

    /// Apply H gate (updates basis and conjugates frame)
    pub fn apply_h(&mut self) {
        self.basis = self.basis.apply_h();
        self.frame.conjugate_h();
    }

    /// Apply S gate (conjugates frame)
    pub fn apply_s(&mut self) {
        // S doesn't change computational basis states (up to phase)
        // but it does change how the frame propagates
        self.frame.conjugate_s();
    }

    /// Mark state as unknown with a reason.
    ///
    /// This is the preferred method - always provide a reason for user trust.
    pub fn mark_unknown_because(&mut self, reason: UnknownReason) {
        self.basis = KnownBasis::Unknown(reason);
    }

    /// Mark state as unknown (uses Explicit reason).
    ///
    /// Prefer `mark_unknown_because()` for better diagnostics.
    pub fn mark_unknown(&mut self) {
        self.mark_unknown_because(UnknownReason::Explicit);
    }

    /// Set basis to a known state (from LogicalBasis).
    pub fn set_basis(&mut self, basis: LogicalBasis) {
        self.basis = KnownBasis::from_logical(basis);
    }

    /// Set basis directly (including Unknown).
    pub fn set_known_basis(&mut self, basis: KnownBasis) {
        self.basis = basis;
    }

    /// Check if state is known (not Unknown)
    pub fn is_known(&self) -> bool {
        self.basis.is_known()
    }

    /// Check if state is in Z-basis (|0⟩ or |1⟩)
    pub fn is_z_basis(&self) -> bool {
        self.basis.is_z_basis()
    }

    /// Check if state is in X-basis (|+⟩ or |-⟩)
    pub fn is_x_basis(&self) -> bool {
        self.basis.is_x_basis()
    }

    /// Get the reason why state is unknown (if applicable).
    pub fn why_unknown(&self) -> Option<UnknownReason> {
        self.basis.unknown_reason()
    }

    /// Get Z-measurement value if deterministic.
    ///
    /// Returns Some(value) if the state is in Z-basis and measurement
    /// would be deterministic. The raw value needs frame correction:
    /// `actual_result = value ^ frame.x`
    pub fn z_measurement_value(&self) -> Option<bool> {
        self.basis.z_measurement_value()
    }

    /// Get X-measurement value if deterministic.
    ///
    /// Returns Some(value) if the state is in X-basis and measurement
    /// would be deterministic. The raw value needs frame correction:
    /// `actual_result = value ^ frame.z`
    pub fn x_measurement_value(&self) -> Option<bool> {
        self.basis.x_measurement_value()
    }

    /// Get frame-corrected Z-measurement value if deterministic.
    ///
    /// Applies frame correction automatically.
    pub fn corrected_z_value(&self) -> Option<bool> {
        self.z_measurement_value().map(|v| v ^ self.frame.x)
    }

    /// Get frame-corrected X-measurement value if deterministic.
    ///
    /// Applies frame correction automatically.
    pub fn corrected_x_value(&self) -> Option<bool> {
        self.x_measurement_value().map(|v| v ^ self.frame.z)
    }

    /// Get the effective state accounting for frame (legacy compatibility).
    ///
    /// Returns None if state is Unknown.
    pub fn effective_basis(&self) -> Option<LogicalBasis> {
        self.basis.to_logical().map(|b| {
            let mut result = b;
            if self.frame.x {
                result = result.apply_x();
            }
            if self.frame.z {
                result = result.apply_z();
            }
            result
        })
    }

    /// Get the raw LogicalBasis if known (legacy compatibility).
    ///
    /// Returns None if state is Unknown.
    #[deprecated(note = "Use `basis` field directly with KnownBasis methods")]
    pub fn logical_basis(&self) -> Option<LogicalBasis> {
        self.basis.to_logical()
    }
}

impl fmt::Display for LogicalQubit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q{}={}", self.id, self.basis)?;
        if !self.frame.is_identity() {
            write!(f, " [frame:{}]", self.frame)?;
        }
        Ok(())
    }
}

// ============================================================================
// Logical Error Budget
// ============================================================================

/// Structured error budget with explicit breakdown.
///
/// Error sources are tracked separately because:
/// 1. They have different scaling behaviors
/// 2. They may not be independent
/// 3. Different mitigation strategies apply to each
///
/// The total error uses a pessimistic union bound by default,
/// but more sophisticated combination methods can be used.
#[derive(Clone, Debug, Default)]
pub struct LogicalErrorBudget {
    /// Error from QEC failures (decoder misses logical errors)
    pub from_qec: f64,
    /// Error from T-gate injection process
    pub from_injection: f64,
    /// Error from idle/memory decoherence
    pub from_idle: f64,
    /// Total code cycles elapsed
    pub cycles_elapsed: usize,
    /// Number of T-gates processed
    pub t_gates_processed: usize,
}

impl LogicalErrorBudget {
    /// Create empty budget
    pub fn new() -> Self {
        Self::default()
    }

    /// Combined logical error probability (pessimistic union bound)
    ///
    /// This is an upper bound. The actual error may be lower if
    /// error sources are correlated in favorable ways.
    pub fn total(&self) -> f64 {
        // Union bound: P(A ∪ B ∪ C) ≤ P(A) + P(B) + P(C)
        (self.from_qec + self.from_injection + self.from_idle).min(1.0)
    }

    /// Combined error assuming independence
    ///
    /// P(no error) = (1 - p_qec)(1 - p_injection)(1 - p_idle)
    pub fn total_independent(&self) -> f64 {
        1.0 - (1.0 - self.from_qec) * (1.0 - self.from_injection) * (1.0 - self.from_idle)
    }

    /// Add QEC error from n additional cycles
    ///
    /// Uses the formula: P(error after n cycles) = 1 - (1 - p_per_cycle)^n
    pub fn add_qec_cycles(&mut self, p_per_cycle: f64, n: usize) {
        let p_new = 1.0 - (1.0 - p_per_cycle).powi(n as i32);
        // Combine with existing: P(A ∪ B) for independent events
        self.from_qec = 1.0 - (1.0 - self.from_qec) * (1.0 - p_new);
        self.cycles_elapsed += n;
    }

    /// Add T-injection error
    pub fn add_injection(&mut self, magic_state_error: f64) {
        self.from_injection = 1.0 - (1.0 - self.from_injection) * (1.0 - magic_state_error);
        self.t_gates_processed += 1;
    }

    /// Add idle/memory error
    pub fn add_idle(&mut self, p_idle: f64, cycles: usize) {
        let p_new = 1.0 - (1.0 - p_idle).powi(cycles as i32);
        self.from_idle = 1.0 - (1.0 - self.from_idle) * (1.0 - p_new);
    }

    /// Merge another budget into this one
    pub fn merge(&mut self, other: &LogicalErrorBudget) {
        self.from_qec = 1.0 - (1.0 - self.from_qec) * (1.0 - other.from_qec);
        self.from_injection = 1.0 - (1.0 - self.from_injection) * (1.0 - other.from_injection);
        self.from_idle = 1.0 - (1.0 - self.from_idle) * (1.0 - other.from_idle);
        self.cycles_elapsed += other.cycles_elapsed;
        self.t_gates_processed += other.t_gates_processed;
    }

    /// Reset all error accumulation
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for LogicalErrorBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Logical Error Budget:")?;
        writeln!(f, "  QEC failures:    {:.2e}", self.from_qec)?;
        writeln!(f, "  T-injection:     {:.2e}", self.from_injection)?;
        writeln!(f, "  Idle/memory:     {:.2e}", self.from_idle)?;
        writeln!(f, "  Total (union):   {:.2e}", self.total())?;
        writeln!(f, "  Cycles elapsed:  {}", self.cycles_elapsed)?;
        writeln!(f, "  T-gates:         {}", self.t_gates_processed)?;
        Ok(())
    }
}

// ============================================================================
// Logical Gates
// ============================================================================

/// Logical gate operations.
///
/// These are high-level logical operations that will be compiled into
/// timed operations with specific cycle costs.
#[derive(Clone, Debug, PartialEq)]
pub enum LogicalGate {
    // Single-qubit Clifford gates
    /// Pauli X gate
    X(usize),
    /// Pauli Y gate
    Y(usize),
    /// Pauli Z gate
    Z(usize),
    /// Hadamard gate
    H(usize),
    /// S gate (π/4 phase)
    S(usize),
    /// S† gate
    Sdg(usize),

    // Non-Clifford gates (require magic states)
    /// T gate (π/8 phase) - requires 1 magic state
    T(usize),
    /// T† gate - requires 1 magic state
    Tdg(usize),

    // Two-qubit gates
    /// CNOT (controlled-X)
    CNOT(usize, usize),
    /// CZ (controlled-Z)
    CZ(usize, usize),
    /// SWAP
    SWAP(usize, usize),

    // Multi-qubit gates
    /// Toffoli (CCX) - decomposes to ~7 T gates
    Toffoli(usize, usize, usize),
    /// CCZ - decomposes to ~7 T gates
    CCZ(usize, usize, usize),

    // Control operations
    /// Explicit QEC round on all qubits
    QECRound,
    /// Idle for specified cycles (memory/waiting)
    Idle(usize),

    // Measurement
    /// Measure in Z basis
    MeasureZ(usize),
    /// Measure in X basis
    MeasureX(usize),

    // Initialization
    /// Reset to |0⟩
    Reset(usize),
    /// Prepare |+⟩ state
    PreparePlus(usize),
}

impl LogicalGate {
    /// Check if gate requires magic states
    pub fn requires_magic_state(&self) -> bool {
        matches!(
            self,
            LogicalGate::T(_)
                | LogicalGate::Tdg(_)
                | LogicalGate::Toffoli(_, _, _)
                | LogicalGate::CCZ(_, _, _)
        )
    }

    /// Get T-count for this gate (magic states required)
    pub fn t_count(&self) -> usize {
        match self {
            LogicalGate::T(_) | LogicalGate::Tdg(_) => 1,
            LogicalGate::Toffoli(_, _, _) | LogicalGate::CCZ(_, _, _) => 7, // Standard decomposition
            _ => 0,
        }
    }

    /// Get qubits involved in this gate
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            LogicalGate::X(q)
            | LogicalGate::Y(q)
            | LogicalGate::Z(q)
            | LogicalGate::H(q)
            | LogicalGate::S(q)
            | LogicalGate::Sdg(q)
            | LogicalGate::T(q)
            | LogicalGate::Tdg(q)
            | LogicalGate::MeasureZ(q)
            | LogicalGate::MeasureX(q)
            | LogicalGate::Reset(q)
            | LogicalGate::PreparePlus(q) => vec![*q],

            LogicalGate::CNOT(a, b) | LogicalGate::CZ(a, b) | LogicalGate::SWAP(a, b) => {
                vec![*a, *b]
            }

            LogicalGate::Toffoli(a, b, c) | LogicalGate::CCZ(a, b, c) => vec![*a, *b, *c],

            LogicalGate::QECRound | LogicalGate::Idle(_) => vec![], // Affects all qubits
        }
    }

    /// Check if this is a Clifford gate
    pub fn is_clifford(&self) -> bool {
        matches!(
            self,
            LogicalGate::X(_)
                | LogicalGate::Y(_)
                | LogicalGate::Z(_)
                | LogicalGate::H(_)
                | LogicalGate::S(_)
                | LogicalGate::Sdg(_)
                | LogicalGate::CNOT(_, _)
                | LogicalGate::CZ(_, _)
                | LogicalGate::SWAP(_, _)
        )
    }
}

impl fmt::Display for LogicalGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalGate::X(q) => write!(f, "X({})", q),
            LogicalGate::Y(q) => write!(f, "Y({})", q),
            LogicalGate::Z(q) => write!(f, "Z({})", q),
            LogicalGate::H(q) => write!(f, "H({})", q),
            LogicalGate::S(q) => write!(f, "S({})", q),
            LogicalGate::Sdg(q) => write!(f, "S†({})", q),
            LogicalGate::T(q) => write!(f, "T({})", q),
            LogicalGate::Tdg(q) => write!(f, "T†({})", q),
            LogicalGate::CNOT(c, t) => write!(f, "CNOT({},{})", c, t),
            LogicalGate::CZ(a, b) => write!(f, "CZ({},{})", a, b),
            LogicalGate::SWAP(a, b) => write!(f, "SWAP({},{})", a, b),
            LogicalGate::Toffoli(a, b, c) => write!(f, "Toffoli({},{},{})", a, b, c),
            LogicalGate::CCZ(a, b, c) => write!(f, "CCZ({},{},{})", a, b, c),
            LogicalGate::QECRound => write!(f, "QEC"),
            LogicalGate::Idle(n) => write!(f, "Idle({})", n),
            LogicalGate::MeasureZ(q) => write!(f, "MZ({})", q),
            LogicalGate::MeasureX(q) => write!(f, "MX({})", q),
            LogicalGate::Reset(q) => write!(f, "Reset({})", q),
            LogicalGate::PreparePlus(q) => write!(f, "|+⟩({})", q),
        }
    }
}

// ============================================================================
// Timed Operations
// ============================================================================

/// A logical operation with timing information.
///
/// All operations are measured in surface code cycles, not abstract gate counts.
#[derive(Clone, Debug)]
pub struct TimedOp {
    /// The logical operation
    pub op: LogicalGate,
    /// Start cycle (0-indexed)
    pub start_cycle: usize,
    /// Duration in cycles
    pub duration_cycles: usize,
    /// Qubits involved (for dependency tracking)
    pub qubits_involved: Vec<usize>,
}

impl TimedOp {
    /// Create a new timed operation
    pub fn new(op: LogicalGate, start: usize, duration: usize) -> Self {
        let qubits = op.qubits();
        Self {
            op,
            start_cycle: start,
            duration_cycles: duration,
            qubits_involved: qubits,
        }
    }

    /// End cycle (exclusive)
    pub fn end_cycle(&self) -> usize {
        self.start_cycle + self.duration_cycles
    }

    /// Check if this operation overlaps with another in time
    pub fn overlaps(&self, other: &TimedOp) -> bool {
        self.start_cycle < other.end_cycle() && other.start_cycle < self.end_cycle()
    }

    /// Check if operations conflict (overlap in time AND share qubits)
    pub fn conflicts_with(&self, other: &TimedOp) -> bool {
        if !self.overlaps(other) {
            return false;
        }
        // Check qubit overlap
        for q in &self.qubits_involved {
            if other.qubits_involved.contains(q) {
                return true;
            }
        }
        false
    }
}

// ============================================================================
// Resource Tracking (Phase 2.2: Event-Log Pattern)
// ============================================================================

/// Resource delta from an operation (legacy, kept for compatibility).
#[derive(Clone, Debug, Default)]
pub struct ResourceDelta {
    /// Physical qubits used
    pub qubits: usize,
    /// Code cycles consumed
    pub cycles: usize,
    /// Magic states consumed
    pub magic_states: usize,
}

impl ResourceDelta {
    /// Create empty delta
    pub fn zero() -> Self {
        Self::default()
    }

    /// Add another delta
    pub fn add(&mut self, other: &ResourceDelta) {
        self.qubits = self.qubits.max(other.qubits); // Qubits are max, not sum
        self.cycles += other.cycles;
        self.magic_states += other.magic_states;
    }

    /// Spacetime volume (qubits × cycles)
    pub fn spacetime_volume(&self) -> usize {
        self.qubits * self.cycles
    }
}

// ============================================================================
// Resource Event Log (Phase 2.2)
// ============================================================================

/// A resource event from a single operation.
///
/// Every gate execution produces one of these. All summary metrics
/// are computed by reducing a sequence of events.
#[derive(Clone, Debug)]
pub struct ResourceEvent {
    /// Start cycle of this operation
    pub start_cycle: usize,
    /// Duration in cycles
    pub duration_cycles: usize,
    /// Data qubits active during this operation
    pub data_qubits: usize,
    /// Ancilla qubits used (temporary, for lattice surgery etc.)
    pub ancilla_qubits: usize,
    /// Factory qubits contributing during this operation
    pub factory_qubits: usize,
    /// Magic states consumed
    pub magic_states: usize,
}

impl ResourceEvent {
    /// Create an empty event (no resources consumed)
    pub fn zero(start_cycle: usize) -> Self {
        Self {
            start_cycle,
            duration_cycles: 0,
            data_qubits: 0,
            ancilla_qubits: 0,
            factory_qubits: 0,
            magic_states: 0,
        }
    }

    /// Create a Clifford gate event (cycles only, no magic states)
    pub fn clifford(start_cycle: usize, duration: usize, data_qubits: usize) -> Self {
        Self {
            start_cycle,
            duration_cycles: duration,
            data_qubits,
            ancilla_qubits: 0,
            factory_qubits: 0,
            magic_states: 0,
        }
    }

    /// Create a T-gate event (consumes magic state)
    pub fn t_gate(
        start_cycle: usize,
        duration: usize,
        data_qubits: usize,
        factory_qubits: usize,
    ) -> Self {
        Self {
            start_cycle,
            duration_cycles: duration,
            data_qubits,
            ancilla_qubits: 0,
            factory_qubits,
            magic_states: 1,
        }
    }

    /// Create a Toffoli event (7 T-gates)
    pub fn toffoli(
        start_cycle: usize,
        duration: usize,
        data_qubits: usize,
        factory_qubits: usize,
    ) -> Self {
        Self {
            start_cycle,
            duration_cycles: duration,
            data_qubits,
            ancilla_qubits: 0,
            factory_qubits,
            magic_states: 7,
        }
    }

    /// End cycle (exclusive)
    pub fn end_cycle(&self) -> usize {
        self.start_cycle + self.duration_cycles
    }

    /// Total physical qubits during this operation
    pub fn total_qubits(&self) -> usize {
        self.data_qubits + self.ancilla_qubits + self.factory_qubits
    }

    /// Spacetime contribution from this event
    pub fn spacetime_volume(&self) -> usize {
        self.total_qubits() * self.duration_cycles
    }
}

/// Log of resource events for computing summary metrics.
///
/// The event log is the "single source of truth" for resource tracking.
/// All summary metrics are derived by reducing the log.
#[derive(Clone, Debug, Default)]
pub struct ResourceLog {
    /// Ordered list of events
    events: Vec<ResourceEvent>,
    /// Configured data qubits (for peak calculation)
    data_qubits: usize,
    /// Configured factory qubits (for peak calculation)
    factory_qubits: usize,
}

impl ResourceLog {
    /// Create empty log with qubit configuration
    pub fn new(data_qubits: usize, factory_qubits: usize) -> Self {
        Self {
            events: Vec::new(),
            data_qubits,
            factory_qubits,
        }
    }

    /// Record an event
    pub fn record(&mut self, event: ResourceEvent) {
        self.events.push(event);
    }

    /// Get all recorded events
    pub fn events(&self) -> &[ResourceEvent] {
        &self.events
    }

    /// Compute summary from all events
    pub fn summarize(&self) -> ResourceSummary {
        // Makespan: max end cycle across all events
        let makespan_cycles = self.events.iter().map(|e| e.end_cycle()).max().unwrap_or(0);

        // Peak qubits: max total qubits at any point
        // For simplicity, use configured values (true peak would need timeline analysis)
        let peak_physical_qubits = self.data_qubits + self.factory_qubits;

        // Integrated spacetime: sum of all event contributions
        let spacetime_qubit_cycles: usize = self.events.iter().map(|e| e.spacetime_volume()).sum();

        // Total magic states
        let total_magic_states: usize = self.events.iter().map(|e| e.magic_states).sum();

        // Compute demand timeline (binned by 10-cycle intervals)
        let demand_timeline = self.compute_demand_timeline(makespan_cycles, 10);

        ResourceSummary {
            makespan_cycles,
            peak_physical_qubits,
            spacetime_qubit_cycles,
            total_magic_states,
            demand_timeline,
        }
    }

    /// Compute magic state demand over time (binned)
    fn compute_demand_timeline(&self, makespan: usize, bin_size: usize) -> Vec<DemandBin> {
        if makespan == 0 || bin_size == 0 {
            return Vec::new();
        }

        let n_bins = (makespan + bin_size - 1) / bin_size;
        let mut bins = vec![DemandBin::zero(); n_bins];

        for event in &self.events {
            if event.magic_states > 0 {
                let bin_idx = event.start_cycle / bin_size;
                if bin_idx < n_bins {
                    bins[bin_idx].magic_states += event.magic_states;
                    bins[bin_idx].t_gates += event.magic_states; // Assuming 1:1 for now
                }
            }
        }

        // Set cycle ranges
        for (i, bin) in bins.iter_mut().enumerate() {
            bin.start_cycle = i * bin_size;
            bin.end_cycle = ((i + 1) * bin_size).min(makespan);
        }

        bins
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

/// Summary metrics computed from resource log.
///
/// These are the "people actually use" metrics for FT resource estimation.
#[derive(Clone, Debug, Default)]
pub struct ResourceSummary {
    /// Total cycles from start to finish (critical path)
    pub makespan_cycles: usize,
    /// Peak physical qubits at any point (data + factory)
    pub peak_physical_qubits: usize,
    /// Integrated spacetime: ∑ qubits(t) × dt
    pub spacetime_qubit_cycles: usize,
    /// Total magic states consumed
    pub total_magic_states: usize,
    /// Magic state demand over time (binned)
    pub demand_timeline: Vec<DemandBin>,
}

impl ResourceSummary {
    /// Create zero summary
    pub fn zero() -> Self {
        Self::default()
    }

    /// Average qubits over time
    pub fn average_qubits(&self) -> f64 {
        if self.makespan_cycles == 0 {
            0.0
        } else {
            self.spacetime_qubit_cycles as f64 / self.makespan_cycles as f64
        }
    }

    /// Magic state rate (states per cycle)
    pub fn magic_rate(&self) -> f64 {
        if self.makespan_cycles == 0 {
            0.0
        } else {
            self.total_magic_states as f64 / self.makespan_cycles as f64
        }
    }

    /// Peak magic state demand in any bin
    pub fn peak_demand(&self) -> usize {
        self.demand_timeline
            .iter()
            .map(|b| b.magic_states)
            .max()
            .unwrap_or(0)
    }
}

impl fmt::Display for ResourceSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Resource Summary:")?;
        writeln!(f, "  Makespan:         {} cycles", self.makespan_cycles)?;
        writeln!(f, "  Peak qubits:      {}", self.peak_physical_qubits)?;
        writeln!(
            f,
            "  Spacetime vol:    {} qubit-cycles",
            self.spacetime_qubit_cycles
        )?;
        writeln!(f, "  Magic states:     {}", self.total_magic_states)?;
        writeln!(f, "  Avg qubits:       {:.1}", self.average_qubits())?;
        writeln!(f, "  Magic rate:       {:.3}/cycle", self.magic_rate())?;
        if !self.demand_timeline.is_empty() {
            writeln!(f, "  Peak demand:      {} states/bin", self.peak_demand())?;
        }
        Ok(())
    }
}

/// A time bin for demand analysis.
#[derive(Clone, Debug, Default)]
pub struct DemandBin {
    /// Start cycle of this bin
    pub start_cycle: usize,
    /// End cycle of this bin (exclusive)
    pub end_cycle: usize,
    /// Magic states demanded in this bin
    pub magic_states: usize,
    /// T-gates executed in this bin
    pub t_gates: usize,
}

impl DemandBin {
    /// Create an empty bin
    pub fn zero() -> Self {
        Self::default()
    }

    /// Duration of this bin in cycles
    pub fn duration(&self) -> usize {
        self.end_cycle.saturating_sub(self.start_cycle)
    }

    /// Demand rate (states per cycle in this bin)
    pub fn rate(&self) -> f64 {
        let dur = self.duration();
        if dur == 0 {
            0.0
        } else {
            self.magic_states as f64 / dur as f64
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_frame_composition() {
        let mut f = PauliFrame::identity();
        assert!(f.is_identity());

        f.compose(&PauliFrame::x());
        assert_eq!(f, PauliFrame::x());

        f.compose(&PauliFrame::z());
        assert_eq!(f, PauliFrame::y());

        f.compose(&PauliFrame::y());
        assert!(f.is_identity()); // Y * Y = I
    }

    #[test]
    fn test_pauli_frame_conjugation() {
        // H X H† = Z
        let mut f = PauliFrame::x();
        f.conjugate_h();
        assert_eq!(f, PauliFrame::z());

        // H Z H† = X
        let mut f = PauliFrame::z();
        f.conjugate_h();
        assert_eq!(f, PauliFrame::x());

        // S X S† = Y (X picks up Z)
        let mut f = PauliFrame::x();
        f.conjugate_s();
        assert_eq!(f, PauliFrame::y());
    }

    #[test]
    fn test_logical_basis_gates() {
        // H|0⟩ = |+⟩
        assert_eq!(LogicalBasis::ZERO.apply_h(), LogicalBasis::PLUS);

        // X|0⟩ = |1⟩
        assert_eq!(LogicalBasis::ZERO.apply_x(), LogicalBasis::ONE);

        // Z|+⟩ = |-⟩
        assert_eq!(LogicalBasis::PLUS.apply_z(), LogicalBasis::MINUS);
    }

    #[test]
    fn test_logical_qubit_gates() {
        let mut q = LogicalQubit::new(0);
        assert_eq!(q.basis, KnownBasis::ZERO);

        q.apply_h();
        assert_eq!(q.basis, KnownBasis::PLUS);
        assert_eq!(q.frame, PauliFrame::identity()); // Frame conjugated but still identity

        q.apply_x();
        assert_eq!(q.basis, KnownBasis::PLUS); // X|+⟩ = |+⟩
    }

    #[test]
    fn test_error_budget_accumulation() {
        let mut budget = LogicalErrorBudget::new();

        // Add some QEC error
        budget.add_qec_cycles(1e-6, 100);
        assert!(budget.from_qec > 0.0);
        assert!(budget.from_qec < 1e-3); // Should be small for low error rate

        // Add injection error
        budget.add_injection(1e-10);
        budget.add_injection(1e-10);
        assert_eq!(budget.t_gates_processed, 2);

        // Total should be bounded
        assert!(budget.total() <= 1.0);
    }

    #[test]
    fn test_logical_gate_t_count() {
        assert_eq!(LogicalGate::H(0).t_count(), 0);
        assert_eq!(LogicalGate::T(0).t_count(), 1);
        assert_eq!(LogicalGate::Toffoli(0, 1, 2).t_count(), 7);
    }

    #[test]
    fn test_timed_op_conflicts() {
        let op1 = TimedOp::new(LogicalGate::H(0), 0, 1);
        let op2 = TimedOp::new(LogicalGate::H(0), 1, 1);
        let op3 = TimedOp::new(LogicalGate::H(0), 0, 2);

        // Sequential ops don't conflict
        assert!(!op1.conflicts_with(&op2));

        // Overlapping ops on same qubit conflict
        assert!(op1.conflicts_with(&op3));
    }

    // =========================================================================
    // Resource Event Log Tests (Phase 2.2)
    // =========================================================================

    #[test]
    fn test_resource_event_constructors() {
        let clifford = ResourceEvent::clifford(0, 1, 100);
        assert_eq!(clifford.magic_states, 0);
        assert_eq!(clifford.duration_cycles, 1);
        assert_eq!(clifford.data_qubits, 100);

        let t_gate = ResourceEvent::t_gate(10, 5, 100, 200);
        assert_eq!(t_gate.magic_states, 1);
        assert_eq!(t_gate.start_cycle, 10);
        assert_eq!(t_gate.factory_qubits, 200);
        assert_eq!(t_gate.end_cycle(), 15);

        let toffoli = ResourceEvent::toffoli(0, 10, 100, 200);
        assert_eq!(toffoli.magic_states, 7);
    }

    #[test]
    fn test_resource_event_spacetime() {
        let event = ResourceEvent {
            start_cycle: 0,
            duration_cycles: 10,
            data_qubits: 100,
            ancilla_qubits: 20,
            factory_qubits: 50,
            magic_states: 1,
        };

        assert_eq!(event.total_qubits(), 170);
        assert_eq!(event.spacetime_volume(), 1700);
    }

    #[test]
    fn test_resource_log_empty() {
        let log = ResourceLog::new(100, 200);
        let summary = log.summarize();

        assert_eq!(summary.makespan_cycles, 0);
        assert_eq!(summary.total_magic_states, 0);
        assert_eq!(summary.peak_physical_qubits, 300);
    }

    #[test]
    fn test_resource_log_summary() {
        let mut log = ResourceLog::new(100, 200);

        // Record some events
        log.record(ResourceEvent::clifford(0, 5, 100));
        log.record(ResourceEvent::t_gate(5, 5, 100, 200));
        log.record(ResourceEvent::t_gate(10, 5, 100, 200));
        log.record(ResourceEvent::clifford(15, 5, 100));

        let summary = log.summarize();

        // Makespan should be 20 (0..5, 5..10, 10..15, 15..20)
        assert_eq!(summary.makespan_cycles, 20);
        // 2 T-gates = 2 magic states
        assert_eq!(summary.total_magic_states, 2);
        // Peak = 100 + 200
        assert_eq!(summary.peak_physical_qubits, 300);
    }

    #[test]
    fn test_resource_log_demand_timeline() {
        let mut log = ResourceLog::new(100, 200);

        // Cluster T-gates at different times
        log.record(ResourceEvent::t_gate(0, 5, 100, 200));
        log.record(ResourceEvent::t_gate(2, 5, 100, 200));
        log.record(ResourceEvent::t_gate(4, 5, 100, 200)); // All in bin 0 (0-10)
        log.record(ResourceEvent::t_gate(15, 5, 100, 200)); // In bin 1 (10-20)

        let summary = log.summarize();

        // Should have 2 bins (makespan = 20)
        assert_eq!(summary.demand_timeline.len(), 2);

        // First bin should have 3 T-gates, second should have 1
        assert_eq!(summary.demand_timeline[0].magic_states, 3);
        assert_eq!(summary.demand_timeline[1].magic_states, 1);

        // Peak demand should be 3
        assert_eq!(summary.peak_demand(), 3);
    }

    #[test]
    fn test_resource_summary_rates() {
        let summary = ResourceSummary {
            makespan_cycles: 100,
            peak_physical_qubits: 500,
            spacetime_qubit_cycles: 40000,
            total_magic_states: 20,
            demand_timeline: vec![],
        };

        assert_eq!(summary.average_qubits(), 400.0);
        assert_eq!(summary.magic_rate(), 0.2);
    }
}
