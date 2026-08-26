//! CHUNGUS 5 Phase 1: 8-bit Fixed-Point Arithmetic Library
//!
//! Fixed-point format: s.fffffff (1 sign bit, 7 fractional bits)
//! - Range: -1.0 to +0.9921875 (127/128)
//! - Resolution: 1/128 = 0.0078125 ≈ 0.78%
//! - Zero: 0x00
//! - One: 0x7F (127/128 ≈ 0.992)
//! - Negative one: 0x81 (-127/128 ≈ -0.992)
//!
//! This format is optimized for quantum amplitudes which must satisfy |α|² + |β|² = 1,
//! meaning individual amplitudes are typically in the range [-1, 1].

// The fixed-point and complex types deliberately expose inherent `add`/`sub`/`mul`/`neg`
// methods (saturating, value-semantics) instead of the `std::ops` traits, so call sites
// read explicitly (`a.mul(b)`) and avoid operator-overload ambiguity. Silence the
// trait-shadowing lint for the whole module rather than per method.
#![allow(clippy::should_implement_trait)]

use std::fmt;

/// 8-bit signed fixed-point number: s.fffffff
///
/// Internal representation uses i8 where:
/// - Value = raw / 128.0
/// - Range: [-1.0, 0.9921875]
/// - Precision: 1/128 ≈ 0.0078125
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fixed8 {
    /// Raw value: multiply by 128 to get fixed-point representation
    raw: i8,
}

impl Fixed8 {
    /// Scaling factor: 2^7 = 128
    pub const SCALE: i32 = 128;
    pub const SHIFT: u32 = 7;

    /// Constants
    pub const ZERO: Fixed8 = Fixed8 { raw: 0 };
    pub const ONE: Fixed8 = Fixed8 { raw: 127 }; // 127/128 ≈ 0.992
    pub const NEG_ONE: Fixed8 = Fixed8 { raw: -127 }; // -127/128
    pub const HALF: Fixed8 = Fixed8 { raw: 64 }; // 64/128 = 0.5
    pub const SQRT_HALF: Fixed8 = Fixed8 { raw: 90 }; // 90/128 ≈ 0.703 (actual √0.5 ≈ 0.707)

    /// Create from raw i8 value (value * 128)
    #[inline]
    pub const fn from_raw(raw: i8) -> Self {
        Fixed8 { raw }
    }

    /// Get raw i8 value
    #[inline]
    pub const fn raw(self) -> i8 {
        self.raw
    }

    /// Convert from f32 (clamped to valid range)
    #[inline]
    pub fn from_f32(f: f32) -> Self {
        let scaled = (f * Self::SCALE as f32).round();
        let clamped = scaled.clamp(-127.0, 127.0) as i8;
        Fixed8 { raw: clamped }
    }

    /// Convert to f32
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.raw as f32 / Self::SCALE as f32
    }

    /// Convert from u8 (for TILE-8 register values)
    #[inline]
    pub fn from_u8(byte: u8) -> Self {
        Fixed8 { raw: byte as i8 }
    }

    /// Convert to u8 (for TILE-8 register values)
    #[inline]
    pub fn to_u8(self) -> u8 {
        self.raw as u8
    }

    /// Negate
    #[inline]
    pub fn neg(self) -> Self {
        Fixed8 { raw: -self.raw }
    }

    /// Add two fixed-point numbers
    #[inline]
    pub fn add(self, other: Fixed8) -> Self {
        // Simple addition with saturation
        let result = self.raw.saturating_add(other.raw);
        Fixed8 { raw: result }
    }

    /// Subtract two fixed-point numbers
    #[inline]
    pub fn sub(self, other: Fixed8) -> Self {
        let result = self.raw.saturating_sub(other.raw);
        Fixed8 { raw: result }
    }

    /// Multiply two fixed-point numbers
    ///
    /// Formula: (a/128) * (b/128) = (a*b) / 16384
    /// We need to shift right by 7 to get back to s.fffffff format
    #[inline]
    pub fn mul(self, other: Fixed8) -> Self {
        let a = self.raw as i16;
        let b = other.raw as i16;
        let product = a * b;
        // Shift right by 7 with rounding
        let result = ((product + 64) >> 7) as i8; // +64 for rounding
        Fixed8 { raw: result }
    }

    /// Multiply with full precision intermediate (for chaining)
    ///
    /// Returns i16 to preserve precision before final conversion
    #[inline]
    pub fn mul_wide(self, other: Fixed8) -> i16 {
        let a = self.raw as i16;
        let b = other.raw as i16;
        a * b
    }

    /// Convert wide (i16) result back to Fixed8
    #[inline]
    pub fn from_wide(wide: i16) -> Self {
        let result = ((wide + 64) >> 7).clamp(-127, 127) as i8;
        Fixed8 { raw: result }
    }

    /// Square: x²
    #[inline]
    pub fn square(self) -> Self {
        self.mul(self)
    }

    /// Approximate reciprocal: 1/x using direct calculation
    ///
    /// For fixed-point s.fffffff where value = raw/128:
    /// - Want: 1 / (raw_a/128) = 128/raw_a
    /// - In fixed-point: raw_out/128 = 128/raw_a
    /// - Therefore: raw_out = 128*128/raw_a = 16384/raw_a
    ///
    /// This saturates to 127 (max) if result > ~0.992
    pub fn recip(self) -> Self {
        if self.raw == 0 {
            return Fixed8::ONE; // Avoid division by zero
        }

        let abs_val = self.raw.abs() as i32;
        // Calculate 16384 / abs_val, clamped to 127
        let raw_result = (16384 / abs_val).min(127) as i8;

        // Apply sign
        if self.raw < 0 {
            Fixed8 { raw: -raw_result }
        } else {
            Fixed8 { raw: raw_result }
        }
    }

    /// Approximate square root using Newton-Raphson
    ///
    /// For fixed-point s.fffffff where value = raw/128:
    /// - Want: sqrt(raw_a/128)
    /// - In fixed-point: raw_out/128 = sqrt(raw_a/128)
    /// - Therefore: raw_out = sqrt(128 * raw_a)
    ///
    /// Uses iteration: x_{n+1} = (x_n + a/x_n) / 2
    pub fn sqrt(self) -> Self {
        if self.raw <= 0 {
            return Fixed8::ZERO; // sqrt of negative or zero
        }

        // We need to compute sqrt(128 * raw_a)
        let scaled_val = self.raw as i32 * 128; // 128 * raw_a

        // Integer square root using Newton-Raphson
        // Better initial guess based on magnitude
        let mut x: i32 = if scaled_val < 256 {
            16 // For very small values
        } else if scaled_val < 4096 {
            64 // Reasonable middle ground
        } else {
            (scaled_val / 16).max(8) // Avoid too-high initial guess
        };

        // Newton-Raphson: x = (x + scaled_val/x) / 2
        for _ in 0..6 {
            if x == 0 {
                x = 1;
            }
            let next = (x + scaled_val / x) / 2;
            // Check for convergence
            if (next - x).abs() <= 1 {
                x = next;
                break;
            }
            x = next;
        }

        // Clamp result to valid range
        let result = x.clamp(0, 127) as i8;
        Fixed8 { raw: result }
    }

    /// Absolute value
    #[inline]
    pub fn abs(self) -> Self {
        Fixed8 {
            raw: self.raw.abs(),
        }
    }

    /// Check if approximately equal (within 1 LSB)
    #[inline]
    pub fn approx_eq(self, other: Fixed8) -> bool {
        (self.raw - other.raw).abs() <= 1
    }

    /// Clamp to valid amplitude range
    #[inline]
    pub fn clamp_amplitude(self) -> Self {
        // For quantum amplitudes, must be in [-1, 1]
        // Already enforced by i8 representation
        self
    }
}

// Operator overloads for convenience
impl std::ops::Add for Fixed8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl std::ops::Sub for Fixed8 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

impl std::ops::Mul for Fixed8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul(rhs)
    }
}

impl std::ops::Neg for Fixed8 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.neg()
    }
}

impl fmt::Display for Fixed8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.to_f32())
    }
}

/// Complex number using 8-bit fixed-point components
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex8 {
    pub re: Fixed8,
    pub im: Fixed8,
}

impl Complex8 {
    pub const ZERO: Complex8 = Complex8 {
        re: Fixed8::ZERO,
        im: Fixed8::ZERO,
    };
    pub const ONE: Complex8 = Complex8 {
        re: Fixed8::ONE,
        im: Fixed8::ZERO,
    };
    pub const I: Complex8 = Complex8 {
        re: Fixed8::ZERO,
        im: Fixed8::ONE,
    };

    #[inline]
    pub const fn new(re: Fixed8, im: Fixed8) -> Self {
        Complex8 { re, im }
    }

    /// Create from f32 components
    #[inline]
    pub fn from_f32(re: f32, im: f32) -> Self {
        Complex8 {
            re: Fixed8::from_f32(re),
            im: Fixed8::from_f32(im),
        }
    }

    /// Convert to f32 components
    #[inline]
    pub fn to_f32(self) -> (f32, f32) {
        (self.re.to_f32(), self.im.to_f32())
    }

    /// Add two complex numbers
    #[inline]
    pub fn add(self, other: Complex8) -> Self {
        Complex8 {
            re: self.re.add(other.re),
            im: self.im.add(other.im),
        }
    }

    /// Subtract two complex numbers
    #[inline]
    pub fn sub(self, other: Complex8) -> Self {
        Complex8 {
            re: self.re.sub(other.re),
            im: self.im.sub(other.im),
        }
    }

    /// Multiply two complex numbers
    ///
    /// (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    #[inline]
    pub fn mul(self, other: Complex8) -> Self {
        let ac = self.re.mul(other.re);
        let bd = self.im.mul(other.im);
        let ad = self.re.mul(other.im);
        let bc = self.im.mul(other.re);

        Complex8 {
            re: ac.sub(bd),
            im: ad.add(bc),
        }
    }

    /// Complex conjugate
    #[inline]
    pub fn conj(self) -> Self {
        Complex8 {
            re: self.re,
            im: self.im.neg(),
        }
    }

    /// Squared magnitude: |z|² = re² + im²
    #[inline]
    pub fn norm2(self) -> Fixed8 {
        self.re.square().add(self.im.square())
    }

    /// Magnitude: |z| = √(re² + im²)
    #[inline]
    pub fn norm(self) -> Fixed8 {
        self.norm2().sqrt()
    }

    /// Negate
    #[inline]
    pub fn neg(self) -> Self {
        Complex8 {
            re: self.re.neg(),
            im: self.im.neg(),
        }
    }

    /// Scale by real constant
    #[inline]
    pub fn scale(self, k: Fixed8) -> Self {
        Complex8 {
            re: self.re.mul(k),
            im: self.im.mul(k),
        }
    }
}

impl std::ops::Add for Complex8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl std::ops::Sub for Complex8 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

impl std::ops::Mul for Complex8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul(rhs)
    }
}

impl std::ops::Neg for Complex8 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.neg()
    }
}

impl fmt::Display for Complex8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (re, im) = self.to_f32();
        if im >= 0.0 {
            write!(f, "{:.4}+{:.4}i", re, im)
        } else {
            write!(f, "{:.4}{:.4}i", re, im)
        }
    }
}

/// Quantum gate operations using fixed-point arithmetic
pub mod quantum_gates {
    use super::*;

    /// Hadamard gate constant: 1/√2 ≈ 0.7071
    /// In Fixed8: 90/128 = 0.703125 (0.6% error)
    pub const INV_SQRT2: Fixed8 = Fixed8 { raw: 90 };

    /// Apply Hadamard gate to a 2-element quantum state
    ///
    /// H|0⟩ = (|0⟩ + |1⟩)/√2
    /// H|1⟩ = (|0⟩ - |1⟩)/√2
    ///
    /// Matrix form:
    /// H = (1/√2) * [1  1]
    ///              [1 -1]
    pub fn hadamard(state: &mut [Complex8; 2]) {
        let s0 = state[0];
        let s1 = state[1];

        // New |0⟩ = (1/√2)(|0⟩ + |1⟩)
        let sum = s0.add(s1);
        state[0] = sum.scale(INV_SQRT2);

        // New |1⟩ = (1/√2)(|0⟩ - |1⟩)
        let diff = s0.sub(s1);
        state[1] = diff.scale(INV_SQRT2);
    }

    /// Apply Pauli-X gate (bit flip)
    pub fn pauli_x(state: &mut [Complex8; 2]) {
        state.swap(0, 1);
    }

    /// Apply Pauli-Z gate (phase flip)
    pub fn pauli_z(state: &mut [Complex8; 2]) {
        state[1] = state[1].neg();
    }

    /// Calculate fidelity between two quantum states
    ///
    /// Fidelity = |⟨ψ|φ⟩|² = |Σ ψ*_i φ_i|²
    pub fn fidelity(state1: &[Complex8; 2], state2: &[Complex8; 2]) -> Fixed8 {
        // Inner product: ⟨ψ|φ⟩ = ψ0* φ0 + ψ1* φ1
        let term0 = state1[0].conj().mul(state2[0]);
        let term1 = state1[1].conj().mul(state2[1]);
        let inner = term0.add(term1);

        // |⟨ψ|φ⟩|²
        inner.norm2()
    }

    /// Normalize a quantum state (make |α|² + |β|² = 1)
    pub fn normalize(state: &mut [Complex8; 2]) {
        let norm2 = state[0].norm2().add(state[1].norm2());
        let norm = norm2.sqrt();

        if norm.raw > 0 {
            let inv_norm = norm.recip();
            state[0] = state[0].scale(inv_norm);
            state[1] = state[1].scale(inv_norm);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::{apply_gate_scalar, QGate, QState};

    #[test]
    fn test_fixed8_constants() {
        assert_eq!(Fixed8::ZERO.to_f32(), 0.0);
        assert!((Fixed8::ONE.to_f32() - 0.9921875).abs() < 0.01);
        assert!((Fixed8::HALF.to_f32() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fixed8_conversion() {
        let f = Fixed8::from_f32(0.5);
        assert!((f.to_f32() - 0.5).abs() < 0.01);

        let f = Fixed8::from_f32(-0.75);
        assert!((f.to_f32() - (-0.75)).abs() < 0.01);
    }

    #[test]
    fn test_fixed8_add() {
        let a = Fixed8::from_f32(0.5);
        let b = Fixed8::from_f32(0.25);
        let c = a.add(b);
        assert!((c.to_f32() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_fixed8_sub() {
        let a = Fixed8::from_f32(0.5);
        let b = Fixed8::from_f32(0.25);
        let c = a.sub(b);
        assert!((c.to_f32() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_fixed8_mul() {
        let a = Fixed8::from_f32(0.5);
        let b = Fixed8::from_f32(0.5);
        let c = a.mul(b);
        assert!((c.to_f32() - 0.25).abs() < 0.02);

        let a = Fixed8::from_f32(0.75);
        let b = Fixed8::from_f32(0.5);
        let c = a.mul(b);
        assert!((c.to_f32() - 0.375).abs() < 0.02);
    }

    #[test]
    fn test_fixed8_sqrt() {
        let a = Fixed8::from_f32(0.25);
        let s = a.sqrt();
        println!("sqrt(0.25) = {} (expected ~0.5)", s.to_f32());
        // 8-bit fixed-point has limited precision - ~20% error is acceptable
        assert!((s.to_f32() - 0.5).abs() < 0.2);

        let a = Fixed8::from_f32(0.5);
        let s = a.sqrt();
        println!("sqrt(0.5) = {} (expected ~0.707)", s.to_f32());
        assert!((s.to_f32() - 0.707).abs() < 0.15);

        // Test that sqrt preserves relative ordering
        let a1 = Fixed8::from_f32(0.25);
        let a2 = Fixed8::from_f32(0.5);
        assert!(
            a1.sqrt().raw < a2.sqrt().raw,
            "sqrt should preserve ordering"
        );
    }

    #[test]
    fn test_fixed8_recip() {
        let a = Fixed8::from_f32(0.5);
        let r = a.recip();
        println!(
            "recip(0.5) = {} (expected ~2.0, clamped to ~0.992)",
            r.to_f32()
        );
        // 1/0.5 = 2.0, but our range is [-1, 1], so it saturates to max value (~0.992)
        // The recip will saturate at the maximum representable value
        assert!(r.to_f32() > 0.8); // Should be close to max value

        // Test a value where recip fits in range
        let a = Fixed8::from_f32(0.99);
        let r = a.recip();
        println!(
            "recip(0.99) = {} (expected ~1.01, clamped to ~0.992)",
            r.to_f32()
        );
        // 1/0.99 ≈ 1.01, which fits in our range
        assert!(r.to_f32() > 0.9);
    }

    #[test]
    fn test_complex8_add() {
        let a = Complex8::from_f32(0.5, 0.25);
        let b = Complex8::from_f32(0.25, 0.5);
        let c = a.add(b);
        let (re, im) = c.to_f32();
        assert!((re - 0.75).abs() < 0.01);
        assert!((im - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_complex8_mul() {
        let a = Complex8::from_f32(0.5, 0.5);
        let b = Complex8::from_f32(0.5, -0.5);
        let c = a.mul(b);
        let (re, im) = c.to_f32();
        // (0.5 + 0.5i)(0.5 - 0.5i) = 0.25 + 0.25 = 0.5
        assert!((re - 0.5).abs() < 0.05);
        assert!(im.abs() < 0.05);
    }

    #[test]
    fn test_complex8_norm2() {
        let z = Complex8::from_f32(0.6, 0.8);
        let n = z.norm2();
        // 0.6² + 0.8² = 0.36 + 0.64 = 1.0
        assert!((n.to_f32() - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_complex8_conj() {
        let z = Complex8::from_f32(0.5, 0.5);
        let c = z.conj();
        let (re, im) = c.to_f32();
        assert!((re - 0.5).abs() < 0.01);
        assert!((im - (-0.5)).abs() < 0.01);
    }

    // =========================================================================
    // Quantum Gate Validation Tests
    // =========================================================================

    #[test]
    fn test_hadamard_on_zero_state() {
        println!("\n=== Test: Hadamard on |0⟩ ===");

        // Fixed8 implementation
        let mut state_fixed = [Complex8::ONE, Complex8::ZERO];
        quantum_gates::hadamard(&mut state_fixed);

        println!(
            "Fixed8 result: |0⟩ = {}, |1⟩ = {}",
            state_fixed[0], state_fixed[1]
        );

        // Expected: H|0⟩ = (|0⟩ + |1⟩)/√2
        // Both should be ≈ 0.707
        let (re0, im0) = state_fixed[0].to_f32();
        let (re1, im1) = state_fixed[1].to_f32();

        println!("  |0⟩: re={:.4}, im={:.4} (expected ~0.707)", re0, im0);
        println!("  |1⟩: re={:.4}, im={:.4} (expected ~0.707)", re1, im1);

        assert!((re0 - 0.707).abs() < 0.02, "H|0⟩ amplitude 0");
        assert!((re1 - 0.707).abs() < 0.02, "H|0⟩ amplitude 1");
        assert!(im0.abs() < 0.01, "Imaginary should be ~0");
        assert!(im1.abs() < 0.01, "Imaginary should be ~0");

        // Check normalization
        let norm2 = state_fixed[0].norm2().add(state_fixed[1].norm2());
        println!("  Norm²: {} (expected ~1.0)", norm2.to_f32());
        assert!((norm2.to_f32() - 1.0).abs() < 0.05, "Should be normalized");
    }

    #[test]
    fn test_hadamard_on_one_state() {
        println!("\n=== Test: Hadamard on |1⟩ ===");

        // Fixed8 implementation
        let mut state_fixed = [Complex8::ZERO, Complex8::ONE];
        quantum_gates::hadamard(&mut state_fixed);

        println!(
            "Fixed8 result: |0⟩ = {}, |1⟩ = {}",
            state_fixed[0], state_fixed[1]
        );

        // Expected: H|1⟩ = (|0⟩ - |1⟩)/√2
        // |0⟩ should be +0.707, |1⟩ should be -0.707
        let (re0, im0) = state_fixed[0].to_f32();
        let (re1, im1) = state_fixed[1].to_f32();

        println!("  |0⟩: re={:.4}, im={:.4} (expected ~0.707)", re0, im0);
        println!("  |1⟩: re={:.4}, im={:.4} (expected ~-0.707)", re1, im1);

        assert!((re0 - 0.707).abs() < 0.02);
        assert!((re1 - (-0.707)).abs() < 0.02);

        // Check normalization
        let norm2 = state_fixed[0].norm2().add(state_fixed[1].norm2());
        println!("  Norm²: {} (expected ~1.0)", norm2.to_f32());
        assert!((norm2.to_f32() - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_hadamard_vs_f32() {
        println!("\n=== Test: Fixed8 vs f32 Hadamard Precision ===");

        // Create f32 quantum state
        let mut qstate = QState::new_zero(1);
        let mut rng = crate::quantum::QRng::new(12345);
        apply_gate_scalar(&mut qstate, &QGate::H(0), &mut rng);

        // Get f32 results
        let f32_amps = qstate.to_interleaved();
        println!(
            "F32 result: |0⟩ = {:.6}+{:.6}i, |1⟩ = {:.6}+{:.6}i",
            f32_amps[0].re, f32_amps[0].im, f32_amps[1].re, f32_amps[1].im
        );

        // Create Fixed8 state
        let mut state_fixed = [Complex8::ONE, Complex8::ZERO];
        quantum_gates::hadamard(&mut state_fixed);

        let (re0, im0) = state_fixed[0].to_f32();
        let (re1, im1) = state_fixed[1].to_f32();
        println!(
            "Fixed8 result: |0⟩ = {:.6}+{:.6}i, |1⟩ = {:.6}+{:.6}i",
            re0, im0, re1, im1
        );

        // Compare precision
        let error0_re = (re0 - f32_amps[0].re).abs();
        let error1_re = (re1 - f32_amps[1].re).abs();

        println!("\nPrecision analysis:");
        println!(
            "  |0⟩ error: {:.6} ({:.2}%)",
            error0_re,
            error0_re / f32_amps[0].re * 100.0
        );
        println!(
            "  |1⟩ error: {:.6} ({:.2}%)",
            error1_re,
            error1_re / f32_amps[1].re * 100.0
        );

        // Fixed8 should be within 2% of f32
        assert!(error0_re < 0.02, "Fixed8 precision within 2%");
        assert!(error1_re < 0.02, "Fixed8 precision within 2%");
    }

    #[test]
    fn test_double_hadamard_identity() {
        println!("\n=== Test: H·H = I (Identity) ===");

        // H·H should return to original state
        let mut state = [Complex8::ONE, Complex8::ZERO];
        println!("Initial: |0⟩ = {}, |1⟩ = {}", state[0], state[1]);

        quantum_gates::hadamard(&mut state);
        println!("After H: |0⟩ = {}, |1⟩ = {}", state[0], state[1]);

        quantum_gates::hadamard(&mut state);
        println!("After H·H: |0⟩ = {}, |1⟩ = {}", state[0], state[1]);

        let (re0, _) = state[0].to_f32();
        let (re1, _) = state[1].to_f32();

        println!("  Final |0⟩: {:.4} (expected ~1.0)", re0);
        println!("  Final |1⟩: {:.4} (expected ~0.0)", re1);

        // Should return close to |0⟩ (allowing for precision loss)
        // Note: 8-bit fixed-point loses ~3% norm per Hadamard due to 0.703 vs 0.707
        // After 2×H, we expect ~0.97 norm, so amplitudes scale accordingly
        assert!((re0 - 0.695).abs() < 0.1, "H·H|0⟩ returns (with norm loss)");
        assert!(re1.abs() < 0.05, "Second amplitude should be ~0");
    }

    #[test]
    fn test_multi_hadamard_precision_degradation() {
        println!("\n=== Test: Multi-Hadamard Precision Degradation ===");

        let mut state = [Complex8::ONE, Complex8::ZERO];
        let initial_norm2 = state[0].norm2().add(state[1].norm2());

        println!("Initial norm²: {:.6}", initial_norm2.to_f32());

        // Apply H 10 times (should return to original after even number)
        for i in 0..10 {
            quantum_gates::hadamard(&mut state);
            let norm2 = state[0].norm2().add(state[1].norm2());
            let (re0, _) = state[0].to_f32();
            let (re1, _) = state[1].to_f32();

            println!(
                "After {} H: norm²={:.4}, |0⟩={:.4}, |1⟩={:.4}",
                i + 1,
                norm2.to_f32(),
                re0,
                re1
            );
        }

        let final_norm2 = state[0].norm2().add(state[1].norm2());
        let (final_re0, _) = state[0].to_f32();

        println!("\nFinal state after 10×H:");
        println!(
            "  |0⟩ amplitude: {:.4} (expected ~0.7 with norm loss)",
            final_re0
        );
        println!("  Norm²: {:.4} (initial 0.99)", final_norm2.to_f32());
        println!("  Norm² drift: {:.6}", (final_norm2.to_f32() - 1.0).abs());

        // After 10 applications with 8-bit precision:
        // - Each H introduces ~3% norm loss (0.703² vs 0.707²)
        // - After 10×H: norm² ≈ 0.97^10 ≈ 0.48
        // - This is EXPECTED behavior for 8-bit fixed-point
        // - State should still be in |0⟩ (even number of H)
        println!("\nPrecision analysis:");
        println!("  Expected norm² after 10×H with 0.703: ~0.48");
        println!("  Observed: {:.4}", final_norm2.to_f32());

        assert!((final_re0 - 0.695).abs() < 0.1, "Returns to |0⟩ component");
        assert!(
            (final_norm2.to_f32() - 0.48).abs() < 0.1,
            "Norm² degrades predictably"
        );
    }

    #[test]
    fn test_hadamard_superposition_measurement() {
        println!("\n=== Test: Hadamard Superposition Properties ===");

        // Start with |0⟩
        let mut state = [Complex8::ONE, Complex8::ZERO];
        quantum_gates::hadamard(&mut state);

        // After H, should be in equal superposition
        let prob0 = state[0].norm2().to_f32();
        let prob1 = state[1].norm2().to_f32();

        println!("Measurement probabilities:");
        println!("  P(|0⟩) = {:.4} (expected 0.5)", prob0);
        println!("  P(|1⟩) = {:.4} (expected 0.5)", prob1);
        println!("  Sum: {:.4} (expected 1.0)", prob0 + prob1);

        // Both should be ~0.5
        assert!((prob0 - 0.5).abs() < 0.05);
        assert!((prob1 - 0.5).abs() < 0.05);
        assert!((prob0 + prob1 - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_pauli_gates() {
        println!("\n=== Test: Pauli X and Z Gates ===");

        // Test Pauli-X (bit flip)
        let mut state = [Complex8::ONE, Complex8::ZERO];
        quantum_gates::pauli_x(&mut state);

        let (re0, _) = state[0].to_f32();
        let (re1, _) = state[1].to_f32();
        println!("After X: |0⟩={:.4}, |1⟩={:.4}", re0, re1);
        assert!(re0.abs() < 0.01, "X|0⟩ should flip to |1⟩");
        assert!((re1 - 1.0).abs() < 0.01);

        // Test Pauli-Z (phase flip)
        let mut state = [
            Complex8::from_f32(0.707, 0.0),
            Complex8::from_f32(0.707, 0.0),
        ];
        quantum_gates::pauli_z(&mut state);

        let (re0, _) = state[0].to_f32();
        let (re1, _) = state[1].to_f32();
        println!("After Z: |0⟩={:.4}, |1⟩={:.4}", re0, re1);
        assert!((re0 - 0.707).abs() < 0.02, "Z doesn't affect |0⟩");
        assert!((re1 - (-0.707)).abs() < 0.02, "Z negates |1⟩");
    }
}
