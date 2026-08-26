//! Physics-to-Logic Coupling
//!
//! This module implements bidirectional coupling between physics fields (heat, charge, power)
//! and tile logic computation. All coupling is deterministic (threshold-based, no RNG).
//!
//! ## Three Coupling Mechanisms
//!
//! 1. **Heat → Errors**: High heat causes deterministic bit degradation
//! 2. **Charge → Bias**: Charge shifts comparison thresholds, biases Mux selection
//! 3. **Power → Enable**: Tiles require minimum power to operate
//!
//! ## Usage
//!
//! ```ignore
//! use engine::physics::logic_coupling::*;
//!
//! let config = PhysicsCouplingConfig {
//!     enabled: true,
//!     heat_coupling: HeatErrorCoupling {
//!         enabled: true,
//!         error_threshold: 150,
//!         mode: HeatDegradationMode::BitFlip,
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```

use crate::tile_meta::TileType;

// ============================================================================
// Configuration Types
// ============================================================================

/// Master configuration for physics-to-logic coupling.
///
/// All coupling is deterministic - no RNG, pure threshold-based rules.
/// Disabled by default for backward compatibility.
#[derive(Debug, Clone, Default)]
pub struct PhysicsCouplingConfig {
    /// Enable/disable all coupling (master switch)
    pub enabled: bool,

    /// Heat → Error coupling configuration
    pub heat_coupling: HeatErrorCoupling,

    /// Charge → Bias coupling configuration
    pub charge_coupling: ChargeBiasCoupling,

    /// Power → Enable coupling configuration
    pub power_coupling: PowerEnableCoupling,
}

impl PhysicsCouplingConfig {
    /// Create a config with all coupling mechanisms enabled using defaults
    pub fn all_enabled() -> Self {
        Self {
            enabled: true,
            heat_coupling: HeatErrorCoupling {
                enabled: true,
                ..Default::default()
            },
            charge_coupling: ChargeBiasCoupling {
                enabled: true,
                ..Default::default()
            },
            power_coupling: PowerEnableCoupling {
                enabled: true,
                ..Default::default()
            },
        }
    }

    /// Create a config with only heat coupling enabled
    pub fn heat_only() -> Self {
        Self {
            enabled: true,
            heat_coupling: HeatErrorCoupling {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a config with only power coupling enabled
    pub fn power_only() -> Self {
        Self {
            enabled: true,
            power_coupling: PowerEnableCoupling {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

// ============================================================================
// Heat → Error Coupling
// ============================================================================

/// Heat causes deterministic bit errors/degradation.
///
/// When tile heat exceeds `error_threshold`, the output is modified according to `mode`.
/// At `critical_threshold`, maximum degradation occurs.
#[derive(Debug, Clone)]
pub struct HeatErrorCoupling {
    /// Enable heat-induced errors
    pub enabled: bool,

    /// Heat level at which degradation begins
    pub error_threshold: u32,

    /// Heat level at which maximum degradation occurs
    pub critical_threshold: u32,

    /// Bit mask of bits subject to degradation
    pub degradation_mask: u64,

    /// How degradation is applied
    pub mode: HeatDegradationMode,
}

impl Default for HeatErrorCoupling {
    fn default() -> Self {
        Self {
            enabled: false,
            error_threshold: 200,
            critical_threshold: 250,
            degradation_mask: 0xFF, // Lower 8 bits affected
            mode: HeatDegradationMode::BitFlip,
        }
    }
}

/// How heat affects logic output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatDegradationMode {
    /// XOR degradation_mask with output when heat >= error_threshold
    /// Flips specific bits deterministically
    BitFlip,

    /// AND with ~degradation_mask (clear bits) when heat >= error_threshold
    /// Clears specific bits, simulating signal loss
    BitClear,

    /// Zero output entirely when heat >= critical_threshold
    /// Complete thermal shutdown
    FullDisable,

    /// Hold previous value when heat >= critical_threshold
    /// Thermal latch - tile stops updating
    StickyLatch,
}

// ============================================================================
// Charge → Bias Coupling
// ============================================================================

/// Charge shifts comparison thresholds and biases Mux selection.
///
/// The bias amount is calculated as: `(charge - bias_threshold) / charge_scale`
/// clamped to `max_bias`. This bias is then applied to comparison operations.
#[derive(Debug, Clone)]
pub struct ChargeBiasCoupling {
    /// Enable charge biasing
    pub enabled: bool,

    /// Minimum charge to activate bias (below this, no effect)
    pub bias_threshold: u32,

    /// Divisor to convert charge to bias amount
    pub charge_scale: u32,

    /// Maximum bias value (prevents runaway)
    pub max_bias: i64,

    /// Which tile types are affected by charge bias
    pub affected_tiles: ChargeBiasAffectedTiles,
}

impl Default for ChargeBiasCoupling {
    fn default() -> Self {
        Self {
            enabled: false,
            bias_threshold: 10,
            charge_scale: 10,
            max_bias: 100,
            affected_tiles: ChargeBiasAffectedTiles::default(),
        }
    }
}

/// Which tile types respond to charge bias
#[derive(Debug, Clone, Copy)]
pub struct ChargeBiasAffectedTiles {
    /// Bias Lt/Gt/Lte/Gte/Eq/Neq comparisons
    pub comparisons: bool,

    /// Bias Mux selector threshold
    pub mux: bool,

    /// Bias Zero detector threshold
    pub zero_detector: bool,
}

impl Default for ChargeBiasAffectedTiles {
    fn default() -> Self {
        Self {
            comparisons: true,
            mux: true,
            zero_detector: false,
        }
    }
}

// ============================================================================
// Power → Enable Coupling
// ============================================================================

/// Power gating: tiles require minimum power to operate.
///
/// If a tile's power level is below `min_power`, it behaves according to
/// `unpowered_behavior` instead of computing normally.
#[derive(Debug, Clone)]
pub struct PowerEnableCoupling {
    /// Enable power gating
    pub enabled: bool,

    /// Minimum power level for tile operation
    pub min_power: u8,

    /// What happens when underpowered
    pub unpowered_behavior: UnpoweredBehavior,

    /// Bitmask of TileType discriminants exempt from power gating
    /// Bit N set = TileType with discriminant N is exempt
    pub exempt_tiles: u64,
}

impl Default for PowerEnableCoupling {
    fn default() -> Self {
        // Exempt ClockGlobal (7) and Const (33) from power gating
        // These tiles should always work regardless of power
        let exempt = (1u64 << 7) | (1u64 << 33);

        Self {
            enabled: false,
            min_power: 1,
            unpowered_behavior: UnpoweredBehavior::Hold,
            exempt_tiles: exempt,
        }
    }
}

/// Behavior when a tile doesn't have enough power
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnpoweredBehavior {
    /// Output zero when underpowered
    Zero,

    /// Hold previous value when underpowered (tile freezes)
    Hold,

    /// Output high-impedance pattern (detectable sentinel)
    HighZ,
}

// ============================================================================
// Physics Coupling Context (Snapshot)
// ============================================================================

/// Read-only snapshot of physics fields for tile evaluation.
///
/// Created at tick start, consumed during delta cycles.
/// This ensures physics values don't change mid-stabilization.
#[derive(Debug, Clone)]
pub struct PhysicsCouplingContext {
    /// Snapshotted heat values
    heat_snapshot: Vec<u32>,

    /// Snapshotted charge values
    charge_snapshot: Vec<u32>,

    /// Snapshotted power values
    power_snapshot: Vec<u8>,

    /// Grid width (for coordinate conversion if needed)
    pub width: usize,

    /// Grid height
    pub height: usize,
}

impl PhysicsCouplingContext {
    /// Create a new empty context (used when coupling disabled)
    pub fn empty() -> Self {
        Self {
            heat_snapshot: Vec::new(),
            charge_snapshot: Vec::new(),
            power_snapshot: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Create a new context with preallocated capacity
    pub fn with_capacity(tile_count: usize, width: usize, height: usize) -> Self {
        Self {
            heat_snapshot: vec![0; tile_count],
            charge_snapshot: vec![0; tile_count],
            power_snapshot: vec![255; tile_count], // Default to max power
            width,
            height,
        }
    }

    /// Set heat value at index
    #[inline(always)]
    pub fn set_heat(&mut self, idx: usize, value: u32) {
        if idx < self.heat_snapshot.len() {
            self.heat_snapshot[idx] = value;
        }
    }

    /// Set charge value at index
    #[inline(always)]
    pub fn set_charge(&mut self, idx: usize, value: u32) {
        if idx < self.charge_snapshot.len() {
            self.charge_snapshot[idx] = value;
        }
    }

    /// Set power value at index
    #[inline(always)]
    pub fn set_power(&mut self, idx: usize, value: u8) {
        if idx < self.power_snapshot.len() {
            self.power_snapshot[idx] = value;
        }
    }

    /// Get heat value at tile index
    #[inline(always)]
    pub fn get_heat(&self, idx: usize) -> u32 {
        self.heat_snapshot.get(idx).copied().unwrap_or(0)
    }

    /// Get charge value at tile index
    #[inline(always)]
    pub fn get_charge(&self, idx: usize) -> u32 {
        self.charge_snapshot.get(idx).copied().unwrap_or(0)
    }

    /// Get power value at tile index
    #[inline(always)]
    pub fn get_power(&self, idx: usize) -> u8 {
        self.power_snapshot.get(idx).copied().unwrap_or(255)
    }

    /// Get all physics values at once (heat, charge, power)
    #[inline(always)]
    pub fn get_all(&self, idx: usize) -> (u32, u32, u8) {
        (
            self.get_heat(idx),
            self.get_charge(idx),
            self.get_power(idx),
        )
    }

    /// Check if context is empty (coupling disabled)
    pub fn is_empty(&self) -> bool {
        self.heat_snapshot.is_empty()
    }
}

// ============================================================================
// Core Coupling Functions (Pure, Deterministic)
// ============================================================================

/// Apply all physics effects to computed tile output.
///
/// This is a pure function - deterministic, no side effects.
/// Called after normal tile computation to modify the result based on physics state.
///
/// # Arguments
/// * `raw_output` - The output computed by normal tile logic
/// * `current` - The tile's current (previous) value
/// * `tile_type` - The type of tile being evaluated
/// * `heat` - Heat level at this tile
/// * `charge` - Charge level at this tile (unused here, used in compute_with_charge_bias)
/// * `power` - Power level at this tile
/// * `config` - Coupling configuration
///
/// # Returns
/// Modified output value after applying physics effects
#[inline(always)]
pub fn apply_physics_coupling(
    raw_output: u64,
    current: u64,
    tile_type: TileType,
    heat: u32,
    _charge: u32, // Charge bias is applied during computation, not post-processing
    power: u8,
    config: &PhysicsCouplingConfig,
) -> u64 {
    if !config.enabled {
        return raw_output;
    }

    // Phase 1: Power gating (checked first - can short-circuit everything)
    if config.power_coupling.enabled {
        let tt_discriminant = tile_type as u8;
        let is_exempt = (config.power_coupling.exempt_tiles >> tt_discriminant) & 1 != 0;

        if !is_exempt && power < config.power_coupling.min_power {
            return match config.power_coupling.unpowered_behavior {
                UnpoweredBehavior::Zero => 0,
                UnpoweredBehavior::Hold => current,
                UnpoweredBehavior::HighZ => 0xDEAD_DEAD_DEAD_DEADu64,
            };
        }
    }

    // Phase 2: Heat degradation

    if config.heat_coupling.enabled && heat >= config.heat_coupling.error_threshold {
        apply_heat_degradation(raw_output, current, heat, &config.heat_coupling)
    } else {
        raw_output
    }
}

/// Apply heat-induced degradation to output.
///
/// Pure, deterministic function based on heat level and configured mode.
#[inline(always)]
pub fn apply_heat_degradation(
    output: u64,
    current: u64,
    heat: u32,
    config: &HeatErrorCoupling,
) -> u64 {
    match config.mode {
        HeatDegradationMode::BitFlip => {
            // XOR with mask - deterministically flip specific bits
            output ^ config.degradation_mask
        }
        HeatDegradationMode::BitClear => {
            // Clear bits in mask - signal loss
            output & !config.degradation_mask
        }
        HeatDegradationMode::FullDisable => {
            // Complete shutdown at critical threshold
            if heat >= config.critical_threshold {
                0
            } else {
                output
            }
        }
        HeatDegradationMode::StickyLatch => {
            // Hold value at critical threshold
            if heat >= config.critical_threshold {
                current
            } else {
                output
            }
        }
    }
}

/// Calculate charge bias amount.
///
/// Returns the bias to apply to comparisons based on charge level.
#[inline(always)]
pub fn calculate_charge_bias(charge: u32, config: &ChargeBiasCoupling) -> i64 {
    if !config.enabled || charge < config.bias_threshold {
        return 0;
    }

    let bias_raw = ((charge - config.bias_threshold) / config.charge_scale) as i64;
    bias_raw.min(config.max_bias)
}

/// Check if a tile type is affected by charge bias.
#[inline(always)]
pub fn is_charge_bias_affected(tile_type: TileType, affected: &ChargeBiasAffectedTiles) -> bool {
    match tile_type {
        TileType::Lt
        | TileType::Gt
        | TileType::Lte
        | TileType::Gte
        | TileType::Eq
        | TileType::Neq => affected.comparisons,
        TileType::Mux => affected.mux,
        TileType::Zero => affected.zero_detector,
        _ => false,
    }
}

/// Apply charge bias to a comparison operation.
///
/// For Lt/Lte: bias is added to right operand (makes comparison easier to satisfy)
/// For Gt/Gte: bias is subtracted from right operand (makes comparison easier to satisfy)
/// For Eq/Neq: bias creates a tolerance window
#[inline(always)]
pub fn apply_charge_bias_to_comparison(
    tile_type: TileType,
    left: u64,
    right: u64,
    bias: i64,
) -> u64 {
    match tile_type {
        TileType::Lt => {
            // left < (right + bias) -- charge makes it easier to be "less than"
            let biased_right = (right as i128).saturating_add(bias as i128);
            let biased_right = biased_right.clamp(0, u64::MAX as i128) as u64;
            if left < biased_right { u64::MAX } else { 0 }
        }
        TileType::Gt => {
            // left > (right - bias) -- charge makes it easier to be "greater than"
            let biased_right = (right as i128).saturating_sub(bias as i128);
            let biased_right = biased_right.max(0) as u64;
            if left > biased_right { u64::MAX } else { 0 }
        }
        TileType::Lte => {
            let biased_right = (right as i128).saturating_add(bias as i128);
            let biased_right = biased_right.clamp(0, u64::MAX as i128) as u64;
            if left <= biased_right { u64::MAX } else { 0 }
        }
        TileType::Gte => {
            let biased_right = (right as i128).saturating_sub(bias as i128);
            let biased_right = biased_right.max(0) as u64;
            if left >= biased_right { u64::MAX } else { 0 }
        }
        TileType::Eq => {
            // With bias, equality becomes "within tolerance"
            let diff = left.abs_diff(right);
            if diff <= bias as u64 { u64::MAX } else { 0 }
        }
        TileType::Neq => {
            // With bias, not-equal means "outside tolerance"
            let diff = left.abs_diff(right);
            if diff > bias as u64 { u64::MAX } else { 0 }
        }
        _ => {
            // Should not be called for non-comparison tiles
            0
        }
    }
}

/// Apply charge bias to Mux selection.
///
/// Normal: up != 0 selects left, else right
/// Biased: (up as i64 + bias) > 0 selects left
#[inline(always)]
pub fn apply_charge_bias_to_mux(up: u64, left: u64, right: u64, bias: i64) -> u64 {
    let biased_up = (up as i64).saturating_add(bias);
    if biased_up > 0 { left } else { right }
}

/// Apply charge bias to Zero detector.
///
/// Normal: left == 0 returns MAX, else 0
/// Biased: left < bias is considered "zero"
#[inline(always)]
pub fn apply_charge_bias_to_zero(left: u64, bias: i64) -> u64 {
    if (left as i64) < bias { u64::MAX } else { 0 }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupling_disabled_by_default() {
        let config = PhysicsCouplingConfig::default();
        assert!(!config.enabled);
        assert!(!config.heat_coupling.enabled);
        assert!(!config.charge_coupling.enabled);
        assert!(!config.power_coupling.enabled);
    }

    #[test]
    fn coupling_disabled_passthrough() {
        let config = PhysicsCouplingConfig::default();
        let result = apply_physics_coupling(0x1234, 0x5678, TileType::And, 255, 255, 0, &config);
        assert_eq!(result, 0x1234); // Unchanged
    }

    #[test]
    fn heat_bitflip_deterministic() {
        let config = PhysicsCouplingConfig {
            enabled: true,
            heat_coupling: HeatErrorCoupling {
                enabled: true,
                error_threshold: 100,
                critical_threshold: 250,
                degradation_mask: 0xFF,
                mode: HeatDegradationMode::BitFlip,
            },
            ..Default::default()
        };

        // Same inputs produce same outputs
        let out1 = apply_physics_coupling(0x1234, 0, TileType::And, 150, 0, 255, &config);
        let out2 = apply_physics_coupling(0x1234, 0, TileType::And, 150, 0, 255, &config);
        assert_eq!(out1, out2);
        assert_eq!(out1, 0x1234 ^ 0xFF);
    }

    #[test]
    fn heat_below_threshold_no_effect() {
        let config = PhysicsCouplingConfig {
            enabled: true,
            heat_coupling: HeatErrorCoupling {
                enabled: true,
                error_threshold: 100,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = apply_physics_coupling(0xABCD, 0, TileType::Wire, 50, 0, 255, &config);
        assert_eq!(result, 0xABCD); // Below threshold, no change
    }

    #[test]
    fn heat_bitclear_mode() {
        let config = HeatErrorCoupling {
            enabled: true,
            error_threshold: 100,
            degradation_mask: 0xF0,
            mode: HeatDegradationMode::BitClear,
            ..Default::default()
        };

        let result = apply_heat_degradation(0xFF, 0, 150, &config);
        assert_eq!(result, 0x0F); // Upper nibble cleared
    }

    #[test]
    fn heat_fulldisable_mode() {
        let config = HeatErrorCoupling {
            enabled: true,
            error_threshold: 100,
            critical_threshold: 200,
            degradation_mask: 0xFF,
            mode: HeatDegradationMode::FullDisable,
        };

        // Below critical: no change (BitFlip not applied in FullDisable)
        let result1 = apply_heat_degradation(0xABCD, 0, 150, &config);
        assert_eq!(result1, 0xABCD);

        // At/above critical: zero output
        let result2 = apply_heat_degradation(0xABCD, 0, 250, &config);
        assert_eq!(result2, 0);
    }

    #[test]
    fn heat_stickylatch_mode() {
        let config = HeatErrorCoupling {
            enabled: true,
            error_threshold: 100,
            critical_threshold: 200,
            degradation_mask: 0xFF,
            mode: HeatDegradationMode::StickyLatch,
        };

        // At/above critical: hold current value
        let result = apply_heat_degradation(0xABCD, 0x1234, 250, &config);
        assert_eq!(result, 0x1234); // Held previous value
    }

    #[test]
    fn power_gating_zero_behavior() {
        let config = PhysicsCouplingConfig {
            enabled: true,
            power_coupling: PowerEnableCoupling {
                enabled: true,
                min_power: 10,
                unpowered_behavior: UnpoweredBehavior::Zero,
                exempt_tiles: 0,
            },
            ..Default::default()
        };

        let result = apply_physics_coupling(0xABCD, 0x1234, TileType::And, 0, 0, 5, &config);
        assert_eq!(result, 0); // Underpowered -> zero
    }

    #[test]
    fn power_gating_hold_behavior() {
        let config = PhysicsCouplingConfig {
            enabled: true,
            power_coupling: PowerEnableCoupling {
                enabled: true,
                min_power: 10,
                unpowered_behavior: UnpoweredBehavior::Hold,
                exempt_tiles: 0,
            },
            ..Default::default()
        };

        let result = apply_physics_coupling(0xABCD, 0x1234, TileType::And, 0, 0, 5, &config);
        assert_eq!(result, 0x1234); // Held previous value
    }

    #[test]
    fn power_gating_exempt_tiles() {
        let config = PhysicsCouplingConfig {
            enabled: true,
            power_coupling: PowerEnableCoupling {
                enabled: true,
                min_power: 10,
                unpowered_behavior: UnpoweredBehavior::Zero,
                exempt_tiles: 1u64 << 7, // ClockGlobal exempt
            },
            ..Default::default()
        };

        // ClockGlobal is exempt, should pass through
        let result = apply_physics_coupling(0xFFFF, 0, TileType::ClockGlobal, 0, 0, 0, &config);
        assert_eq!(result, 0xFFFF);

        // And is not exempt, should be zeroed
        let result2 = apply_physics_coupling(0xFFFF, 0, TileType::And, 0, 0, 0, &config);
        assert_eq!(result2, 0);
    }

    #[test]
    fn charge_bias_calculation() {
        let config = ChargeBiasCoupling {
            enabled: true,
            bias_threshold: 10,
            charge_scale: 5,
            max_bias: 100,
            ..Default::default()
        };

        // charge=10: at threshold, bias = 0
        assert_eq!(calculate_charge_bias(10, &config), 0);

        // charge=20: bias = (20-10)/5 = 2
        assert_eq!(calculate_charge_bias(20, &config), 2);

        // charge=100: bias = (100-10)/5 = 18
        assert_eq!(calculate_charge_bias(100, &config), 18);

        // charge=1000: would be 198, but clamped to max_bias=100
        assert_eq!(calculate_charge_bias(1000, &config), 100);
    }

    #[test]
    fn charge_bias_lt_comparison() {
        // Without bias: 5 < 5 is false
        assert_eq!(apply_charge_bias_to_comparison(TileType::Lt, 5, 5, 0), 0);

        // With bias=2: 5 < (5+2) = 5 < 7 is true
        assert_eq!(
            apply_charge_bias_to_comparison(TileType::Lt, 5, 5, 2),
            u64::MAX
        );
    }

    #[test]
    fn charge_bias_gt_comparison() {
        // Without bias: 5 > 5 is false
        assert_eq!(apply_charge_bias_to_comparison(TileType::Gt, 5, 5, 0), 0);

        // With bias=2: 5 > (5-2) = 5 > 3 is true
        assert_eq!(
            apply_charge_bias_to_comparison(TileType::Gt, 5, 5, 2),
            u64::MAX
        );
    }

    #[test]
    fn charge_bias_mux() {
        // Without bias: up=0 selects right
        assert_eq!(apply_charge_bias_to_mux(0, 100, 200, 0), 200);

        // With bias=1: (0+1) > 0 selects left
        assert_eq!(apply_charge_bias_to_mux(0, 100, 200, 1), 100);
    }

    #[test]
    fn physics_context_snapshot() {
        let mut ctx = PhysicsCouplingContext::with_capacity(100, 10, 10);

        ctx.set_heat(5, 150);
        ctx.set_charge(5, 50);
        ctx.set_power(5, 100);

        assert_eq!(ctx.get_heat(5), 150);
        assert_eq!(ctx.get_charge(5), 50);
        assert_eq!(ctx.get_power(5), 100);

        let (h, c, p) = ctx.get_all(5);
        assert_eq!(h, 150);
        assert_eq!(c, 50);
        assert_eq!(p, 100);
    }

    #[test]
    fn combined_heat_and_power() {
        // Test that both heat and power coupling work together
        let config = PhysicsCouplingConfig {
            enabled: true,
            heat_coupling: HeatErrorCoupling {
                enabled: true,
                error_threshold: 100,
                degradation_mask: 0xF,
                mode: HeatDegradationMode::BitFlip,
                ..Default::default()
            },
            power_coupling: PowerEnableCoupling {
                enabled: true,
                min_power: 10,
                unpowered_behavior: UnpoweredBehavior::Zero,
                exempt_tiles: 0,
            },
            ..Default::default()
        };

        // Powered + hot: heat degradation applies
        let result1 = apply_physics_coupling(0xAB, 0, TileType::And, 150, 0, 255, &config);
        assert_eq!(result1, 0xAB ^ 0xF);

        // Underpowered: power gating takes precedence, returns 0
        let result2 = apply_physics_coupling(0xAB, 0, TileType::And, 150, 0, 5, &config);
        assert_eq!(result2, 0);
    }
}
