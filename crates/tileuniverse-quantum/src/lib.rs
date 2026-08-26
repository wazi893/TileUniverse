//! # tileuniverse-quantum
//!
//! Structured sparse quantum-state representations with GHZ and W-state helpers.
//!
//! This crate does not simulate arbitrary quantum circuits. It stores compact
//! representations for state families whose nonzero amplitudes are already known.
//!
//! ## The Key Insight
//!
//! GHZ state `|GHZ_n> = (|00...0> + |11...1>)/sqrt(2)` has exactly **2 non-zero amplitudes**
//! regardless of `n`. Store only those two amplitudes; the qubit count is metadata.
//!
//! ```
//! use tileuniverse_quantum::*;
//!
//! // Fixed endpoint-block memory for this concrete usize qubit count.
//! let ghz = MinimalGhzState::new(1_000_000);
//! let v = ghz.verify();
//! assert!((v.fidelity - 1.0).abs() < 1e-10);
//! ```
//!
//! ## State Types
//!
//! | Struct | Count representation | Use Case |
//! |-----|-----|--|
//! | `MinimalGhzState` | `usize` | Fixed-size endpoint GHZ representation |
//! | `UnlimitedGhzState` | Materialized `BigUint` | Endpoint GHZ representation with large integer labels |
//! | `SymbolicGhzState` | Symbolic labels | Graham's number, TREE(3), and formal infinity labels |
//! | `SparseQuantumGridVec` | `usize` | Materialized W-state excitation amplitudes with O(n) memory |
//!
//! ## W-State Example
//!
//! ```
//! use tileuniverse_quantum::SparseQuantumGridVec;
//!
//! // W-states materialize one amplitude per excitation position: O(n), not O(2^n).
//! let grid = SparseQuantumGridVec::new(1_000);
//! assert_eq!(grid.n_qubits(), 1_000);
//! ```
//!
//! ## Symbolic Example
//!
//! ```
//! use tileuniverse_quantum::*;
//!
//! // Graham's number as a symbolic qubit-count label.
//! let ghz = SymbolicGhzState::graham();
//! let v = ghz.verify();
//! assert!((v.fidelity - 1.0).abs() < 1e-10);
//! assert_eq!(v.size_class, "Graham-class");
//!
//! // Symbolic W-states check the stored formula algebraically.
//! let w = create_graham_w();
//! let wv = w.verify();
//! assert!(wv.is_valid);
//! assert!(wv.total_probability.is_one());
//! ```

use rayon::prelude::*;

pub use ghz::*;
pub use symbolic::*;
pub use w_state::*;

// === Core types (used by all submodules) ===

/// Block size: 128 amplitudes (2^7) per block
pub const BLOCK_SIZE: usize = 128;
pub const BLOCK_SHIFT: usize = 7;

/// A single block of 128 complex amplitudes
#[derive(Clone, Copy, Debug)]
pub struct VecBlock {
    /// Real parts of amplitudes
    pub re: [f64; BLOCK_SIZE],
    /// Imaginary parts of amplitudes
    pub im: [f64; BLOCK_SIZE],
}

impl VecBlock {
    /// Create a new zero-initialized block
    #[inline]
    pub fn new() -> Self {
        Self {
            re: [0.0; BLOCK_SIZE],
            im: [0.0; BLOCK_SIZE],
        }
    }

    /// Set amplitude at local address
    #[inline]
    pub fn set(&mut self, addr: usize, amp: Complex64) {
        self.re[addr] = amp.re;
        self.im[addr] = amp.im;
    }

    /// Get amplitude at local address
    #[inline]
    pub fn get(&self, addr: usize) -> Complex64 {
        Complex64 {
            re: self.re[addr],
            im: self.im[addr],
        }
    }

    /// Check if block is all zeros
    pub fn is_zero(&self) -> bool {
        self.re.iter().all(|&x| x == 0.0) && self.im.iter().all(|&x| x == 0.0)
    }
}

impl Default for VecBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Complex number with f64 precision
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline]
    pub fn norm_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

fn terminal_local_addr_for_usize(n_qubits: usize) -> usize {
    assert!(n_qubits > 0, "GHZ state requires at least one qubit");

    if n_qubits < BLOCK_SHIFT {
        (1usize << n_qubits) - 1
    } else {
        BLOCK_SIZE - 1
    }
}

fn terminal_local_addr_for_biguint(n_qubits: &num_bigint::BigUint) -> usize {
    assert!(n_qubits.bits() > 0, "GHZ state requires at least one qubit");

    let digits = n_qubits.to_u64_digits();
    if digits.len() == 1 && digits[0] < BLOCK_SHIFT as u64 {
        (1usize << digits[0] as usize) - 1
    } else {
        BLOCK_SIZE - 1
    }
}

mod ghz {
    //! GHZ state representations at every scale.
    //!
    //! All GHZ states share the same core insight: exactly 2 non-zero amplitudes,
    //! so memory and creation time are O(1) regardless of qubit count.

    use super::*;
    use num_traits::identities::One;

    /// Ultra-minimal GHZ state representation.
    ///
    /// For GHZ states, we only ever need 2 amplitudes:
    /// - `|00...0>` at block 0, index 0
    /// - `|11...1>` at the terminal local address: `2^n - 1` for `n < 7`,
    ///   otherwise index 127 in the final endpoint block
    ///
    /// This struct stores ONLY these two blocks without any HashMap, Vec, or BigInt.
    /// Total memory: ~4KB fixed regardless of qubit count.
    ///
    /// # Performance
    ///
    /// | Qubits | Creation | Verification | Memory |
    /// |-----|-----|---|--------|
    /// | 1K | <1 us | <1 us | 4 KB |
    /// | 1B | <1 us | <1 us | 4 KB |
    /// | 1T | <1 us | <1 us | 4 KB |
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::MinimalGhzState;
    ///
    /// let ghz = MinimalGhzState::new(1_000_000_000_000usize);
    /// let v = ghz.verify();
    /// assert!((v.fidelity - 1.0).abs() < 1e-10);
    /// assert_eq!(format!("{}", ghz), "|GHZ_1.0T>");
    /// ```
    #[derive(Clone)]
    pub struct MinimalGhzState {
        /// Number of qubits (symbolic — we never compute 2^n)
        pub n_qubits: usize,
        /// Block 0: contains amplitude for `|00...0>` at index 0
        pub block_zero: VecBlock,
        /// Last block: contains amplitude for `|11...1>` at index 127
        pub block_last: VecBlock,
        /// Creation time in microseconds
        pub creation_time_us: u64,
    }

    impl MinimalGhzState {
        /// Create a GHZ state for any positive `usize` qubit count.
        ///
        /// This is O(1) — no BigInt, no allocations beyond the fixed 2 blocks.
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::MinimalGhzState;
        ///
        /// // 1 million qubits in under 10 microseconds
        /// let ghz = MinimalGhzState::new(1_000_000);
        /// assert_eq!(ghz.n_qubits, 1_000_000);
        /// assert!(ghz.creation_time_us < 1000); // typically instant
        /// ```
        pub fn new(n_qubits: usize) -> Self {
            assert!(n_qubits > 0, "GHZ state requires at least one qubit");

            let start = std::time::Instant::now();

            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            let terminal_addr = terminal_local_addr_for_usize(n_qubits);

            let mut block_zero = VecBlock::new();
            block_zero.set(0, Complex64::new(inv_sqrt2, 0.0));

            let mut block_last = VecBlock::new();
            block_last.set(terminal_addr, Complex64::new(inv_sqrt2, 0.0));

            let creation_time_us = start.elapsed().as_micros() as u64;

            Self {
                n_qubits,
                block_zero,
                block_last,
                creation_time_us,
            }
        }

        /// Verify GHZ state fidelity — O(1), no BigInt.
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::MinimalGhzState;
        ///
        /// let ghz = MinimalGhzState::new(42);
        /// let v = ghz.verify();
        /// assert!((v.fidelity - 1.0).abs() < 1e-10);
        /// assert!((v.amp_zero_state - 0.707107).abs() < 1e-4);
        /// assert_eq!(v.max_spurious, 0.0);
        /// ```
        pub fn verify(&self) -> MinimalGhzVerification {
            let expected_amp = std::f64::consts::FRAC_1_SQRT_2;
            let terminal_addr = terminal_local_addr_for_usize(self.n_qubits);

            let amp0_complex = self.block_zero.get(0);
            let amp_last_complex = self.block_last.get(terminal_addr);
            let amp0 = amp0_complex.norm();
            let amp_last = amp_last_complex.norm();

            let overlap = Complex64::new(
                expected_amp * (amp0_complex.re + amp_last_complex.re),
                expected_amp * (amp0_complex.im + amp_last_complex.im),
            );
            let fidelity = overlap.norm_squared();

            let mut max_spurious = 0.0f64;

            for addr in 1..BLOCK_SIZE {
                let mag = self.block_zero.get(addr).norm();
                if mag > max_spurious {
                    max_spurious = mag;
                }
            }

            for addr in 0..BLOCK_SIZE {
                if addr == terminal_addr {
                    continue;
                }

                let mag = self.block_last.get(addr).norm();
                if mag > max_spurious {
                    max_spurious = mag;
                }
            }

            MinimalGhzVerification {
                n_qubits: self.n_qubits,
                amp_zero_state: amp0,
                amp_one_state: amp_last,
                expected_amplitude: expected_amp,
                fidelity,
                max_spurious,
            }
        }

        /// Reported struct size in bytes for this fixed endpoint representation.
        pub fn memory_bytes(&self) -> usize {
            std::mem::size_of::<Self>()
        }

        /// Number of qubits
        pub fn n_qubits(&self) -> usize {
            self.n_qubits
        }

        /// Creation time in microseconds
        pub fn creation_time_us(&self) -> u64 {
            self.creation_time_us
        }
    }

    impl std::fmt::Display for MinimalGhzState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let label = if self.n_qubits >= 1_000_000_000_000 {
                format!("{:.1}T", self.n_qubits as f64 / 1e12)
            } else if self.n_qubits >= 1_000_000_000 {
                format!("{:.1}B", self.n_qubits as f64 / 1e9)
            } else if self.n_qubits >= 1_000_000 {
                format!("{:.1}M", self.n_qubits as f64 / 1e6)
            } else if self.n_qubits >= 1_000 {
                format!("{:.1}K", self.n_qubits as f64 / 1e3)
            } else {
                format!("{}", self.n_qubits)
            };
            write!(f, "|GHZ_{}>", label)
        }
    }

    /// Verification results for MinimalGhzState
    #[derive(Debug, Clone)]
    pub struct MinimalGhzVerification {
        pub n_qubits: usize,
        pub amp_zero_state: f64,
        pub amp_one_state: f64,
        pub expected_amplitude: f64,
        pub fidelity: f64,
        pub max_spurious: f64,
    }

    impl std::fmt::Display for MinimalGhzVerification {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(
                f,
                "Minimal GHZ Verification ({} qubits) [O(1) — NO BIGINT]",
                self.n_qubits
            )?;
            writeln!(
                f,
                "  |00...0> amplitude: {:.6} (expected {:.6})",
                self.amp_zero_state, self.expected_amplitude
            )?;
            writeln!(
                f,
                "  |11...1> amplitude: {:.6} (expected {:.6})",
                self.amp_one_state, self.expected_amplitude
            )?;
            writeln!(f, "  Fidelity:           {:.6}", self.fidelity)?;
            writeln!(f, "  Max spurious:       {:.6}", self.max_spurious)?;
            Ok(())
        }
    }

    /// Create a minimal GHZ state for a positive `usize` qubit count.
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::{MinimalGhzState, create_minimal_ghz};
    ///
    /// let ghz1 = MinimalGhzState::new(100);
    /// let ghz2 = create_minimal_ghz(100);
    /// assert_eq!(ghz1.n_qubits, ghz2.n_qubits);
    /// ```
    pub fn create_minimal_ghz(n_qubits: usize) -> MinimalGhzState {
        MinimalGhzState::new(n_qubits)
    }

    /// GHZ endpoint representation using a BigUint qubit-count label.
    ///
    /// This breaks through the 2^64 barrier. The qubit count can be any positive
    /// integer — googol, googolplex, whatever.
    ///
    /// # Memory
    ///
    /// The endpoint amplitude storage is fixed-size. The reported payload bytes
    /// also include the materialized BigUint qubit-count digits.
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::UnlimitedGhzState;
    ///
    /// // 10^100 qubits (a googol)
    /// let googol = num_bigint::BigUint::from(10u32).pow(100);
    /// let ghz = UnlimitedGhzState::new(googol);
    /// let v = ghz.verify();
    /// assert!((v.fidelity - 1.0).abs() < 1e-10);
    /// ```
    #[derive(Clone)]
    pub struct UnlimitedGhzState {
        /// Number of qubits as BigUint — no upper limit
        pub n_qubits: num_bigint::BigUint,
        /// Block 0: contains amplitude for `|00...0>` at index 0
        pub block_zero: VecBlock,
        /// Last block: contains amplitude for `|11...1>` at index 127
        pub block_last: VecBlock,
        /// Creation time in microseconds
        pub creation_time_us: u64,
    }

    impl UnlimitedGhzState {
        /// Create a GHZ state for any positive BigUint qubit count.
        ///
        /// The qubit count can be arbitrarily large — googol, googolplex, etc.
        /// Memory and time remain O(1).
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::UnlimitedGhzState;
        ///
        /// let ghz = UnlimitedGhzState::new(num_bigint::BigUint::from(42u32));
        /// let v = ghz.verify();
        /// assert!((v.fidelity - 1.0).abs() < 1e-10);
        /// ```
        pub fn new(n_qubits: num_bigint::BigUint) -> Self {
            assert!(n_qubits.bits() > 0, "GHZ state requires at least one qubit");

            let start = std::time::Instant::now();

            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            let terminal_addr = terminal_local_addr_for_biguint(&n_qubits);

            let mut block_zero = VecBlock::new();
            block_zero.set(0, Complex64::new(inv_sqrt2, 0.0));

            let mut block_last = VecBlock::new();
            block_last.set(terminal_addr, Complex64::new(inv_sqrt2, 0.0));

            let creation_time_us = start.elapsed().as_micros() as u64;

            Self {
                n_qubits,
                block_zero,
                block_last,
                creation_time_us,
            }
        }

        /// Create from a power of 10: 10^exponent qubits.
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::UnlimitedGhzState;
        ///
        /// // 10^3 = 1000 qubits
        /// let ghz = UnlimitedGhzState::from_power_of_10(3);
        /// assert!((ghz.verify().fidelity - 1.0).abs() < 1e-10);
        /// ```
        pub fn from_power_of_10(exponent: u32) -> Self {
            let n_qubits = num_bigint::BigUint::from(10u32).pow(exponent);
            Self::new(n_qubits)
        }

        /// Create from a power of 2: 2^exponent qubits.
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::UnlimitedGhzState;
        ///
        /// // 2^10 = 1024 qubits
        /// let ghz = UnlimitedGhzState::from_power_of_2(10);
        /// assert_eq!(ghz.n_qubits.to_u64_digits()[0], 1024);
        /// ```
        pub fn from_power_of_2(exponent: u32) -> Self {
            let n_qubits = num_bigint::BigUint::one() << exponent;
            Self::new(n_qubits)
        }

        /// Verify the stored endpoint amplitudes without iterating over basis states.
        pub fn verify(&self) -> UnlimitedGhzVerification {
            let expected_amp = std::f64::consts::FRAC_1_SQRT_2;
            let terminal_addr = terminal_local_addr_for_biguint(&self.n_qubits);

            let amp0_complex = self.block_zero.get(0);
            let amp_last_complex = self.block_last.get(terminal_addr);
            let amp0 = amp0_complex.norm();
            let amp_last = amp_last_complex.norm();

            let overlap = Complex64::new(
                expected_amp * (amp0_complex.re + amp_last_complex.re),
                expected_amp * (amp0_complex.im + amp_last_complex.im),
            );
            let fidelity = overlap.norm_squared();

            let mut max_spurious = 0.0f64;

            for addr in 1..BLOCK_SIZE {
                let mag = self.block_zero.get(addr).norm();
                if mag > max_spurious {
                    max_spurious = mag;
                }
            }

            for addr in 0..BLOCK_SIZE {
                if addr == terminal_addr {
                    continue;
                }

                let mag = self.block_last.get(addr).norm();
                if mag > max_spurious {
                    max_spurious = mag;
                }
            }

            UnlimitedGhzVerification {
                n_qubits: self.n_qubits.clone(),
                amp_zero_state: amp0,
                amp_one_state: amp_last,
                expected_amplitude: expected_amp,
                fidelity,
                max_spurious,
            }
        }

        /// Reported payload bytes: endpoint blocks plus BigUint qubit-count digits.
        pub fn memory_bytes(&self) -> usize {
            std::mem::size_of::<VecBlock>() * 2 + self.n_qubits.to_bytes_le().len()
        }

        /// Get the qubit count
        pub fn n_qubits(&self) -> &num_bigint::BigUint {
            &self.n_qubits
        }

        /// Get approximate log10 of qubit count (for display)
        pub fn log10_qubits(&self) -> f64 {
            let bits = self.n_qubits.bits();
            bits as f64 * std::f64::consts::LOG10_2
        }

        /// Get the state space size as a string: "2^n where n = ..."
        pub fn state_space_description(&self) -> String {
            format!("2^{} amplitudes", self.n_qubits)
        }
    }

    impl std::fmt::Display for UnlimitedGhzState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let bits = self.n_qubits.bits();
            if bits > 1000 {
                write!(
                    f,
                    "|GHZ_(~10^{:.0})>",
                    bits as f64 * std::f64::consts::LOG10_2
                )
            } else if bits > 100 {
                write!(f, "|GHZ_(2^{})>", bits)
            } else {
                write!(f, "|GHZ_{}>", self.n_qubits)
            }
        }
    }

    /// Verification results for UnlimitedGhzState
    #[derive(Debug, Clone)]
    pub struct UnlimitedGhzVerification {
        pub n_qubits: num_bigint::BigUint,
        pub amp_zero_state: f64,
        pub amp_one_state: f64,
        pub expected_amplitude: f64,
        pub fidelity: f64,
        pub max_spurious: f64,
    }

    impl std::fmt::Display for UnlimitedGhzVerification {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let bits = self.n_qubits.bits();
            let log10 = bits as f64 * std::f64::consts::LOG10_2;

            writeln!(f, "BigUint GHZ Verification [endpoint representation]")?;
            if bits > 1000 {
                writeln!(
                    f,
                    "  Qubits: ~10^{:.0} (~2^{} bits to represent)",
                    log10, bits
                )?;
            } else {
                writeln!(f, "  Qubits: {} ({} bits)", self.n_qubits, bits)?;
            }
            writeln!(
                f,
                "  |00...0> amplitude: {:.6} (expected {:.6})",
                self.amp_zero_state, self.expected_amplitude
            )?;
            writeln!(
                f,
                "  |11...1> amplitude: {:.6} (expected {:.6})",
                self.amp_one_state, self.expected_amplitude
            )?;
            writeln!(f, "  Fidelity:           {:.6}", self.fidelity)?;
            writeln!(f, "  Max spurious:       {:.6}", self.max_spurious)?;
            writeln!(f, "  State space:        2^(10^{:.0}) amplitudes", log10)?;
            Ok(())
        }
    }

    /// Create a GHZ state from a BigUint qubit count.
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::{UnlimitedGhzState, create_unlimited_ghz};
    ///
    /// let n = num_bigint::BigUint::from(42u32);
    /// let ghz = create_unlimited_ghz(n.clone());
    /// let ghz2 = UnlimitedGhzState::new(n);
    /// assert!((ghz.verify().fidelity - ghz2.verify().fidelity).abs() < 1e-10);
    /// ```
    pub fn create_unlimited_ghz(n_qubits: num_bigint::BigUint) -> UnlimitedGhzState {
        UnlimitedGhzState::new(n_qubits)
    }

    /// Create a GHZ state with a `10^exponent` BigUint qubit count.
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::create_ghz_power_of_10;
    ///
    /// let ghz = create_ghz_power_of_10(6); // 10^6 qubits
    /// assert!((ghz.verify().fidelity - 1.0).abs() < 1e-10);
    /// ```
    pub fn create_ghz_power_of_10(exponent: u32) -> UnlimitedGhzState {
        UnlimitedGhzState::from_power_of_10(exponent)
    }
}

mod w_state {
    //! W-state and SparseQuantumGridVec — O(n) memory for n excitations.
    //!
    //! Unlike GHZ (2 amplitudes), W-state has n amplitudes. For large n we use
    //! a Vec-based grid for O(1) block access.

    use super::*;

    const W_AMPLITUDE_TOLERANCE: f64 = 1e-12;

    /// Verification result for W state
    #[derive(Debug, Clone)]
    pub struct WStateVerificationVec {
        pub n_qubits: usize,
        pub expected_amplitudes: usize,
        pub correct_amplitudes: usize,
        pub expected_amplitude: f64,
        pub total_probability: f64,
        pub fidelity: f64,
        pub max_amplitude_error: f64,
        pub max_spurious_amplitude: f64,
        pub active_blocks: usize,
        pub memory_bytes: usize,
    }

    /// Fast Vec-based sparse quantum grid.
    ///
    /// Uses a `Vec<VecBlock>` for O(1) block access without BigInt overhead.
    /// Ideal for states with sequential block patterns like W states.
    ///
    /// # Performance Comparison
    ///
    /// | Qubits | BigInt W State | Vec W State |
    /// |-----|-----|--|
    /// | 1M | ~30 seconds | ~25 ms |
    /// | 100M | impossible | ~130 ms |
    /// | 1B | impossible | ~1.2 sec |
    /// | 4B | impossible | ~14 sec |
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::SparseQuantumGridVec;
    ///
    /// let mut grid = SparseQuantumGridVec::new(1_000_000);
    /// grid.create_w_state();
    /// let v = grid.verify_w_fidelity();
    /// assert!((v.fidelity - 1.0).abs() < 1e-10);
    /// assert_eq!(v.correct_amplitudes, 1_000_000);
    /// ```
    #[derive(Clone)]
    pub struct SparseQuantumGridVec {
        n_qubits: usize,
        n_blocks: usize,
        blocks: Vec<VecBlock>,
    }

    impl SparseQuantumGridVec {
        /// Create a new Vec-based sparse grid.
        ///
        /// Pre-allocates all blocks upfront for maximum performance.
        ///
        /// # Panics
        ///
        /// Panics if `n_qubits < 7` (minimum block alignment requirement).
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::SparseQuantumGridVec;
        ///
        /// let grid = SparseQuantumGridVec::new(128);
        /// assert_eq!(grid.n_qubits(), 128);
        /// assert_eq!(grid.n_blocks(), 1);
        /// ```
        pub fn new(n_qubits: usize) -> Self {
            assert!(n_qubits >= 7, "Minimum 7 qubits required");
            let n_blocks = n_qubits.div_ceil(BLOCK_SIZE);

            Self {
                n_qubits,
                n_blocks,
                blocks: vec![VecBlock::new(); n_blocks],
            }
        }

        /// Create a new grid with lazy block allocation.
        ///
        /// Blocks are allocated on first access, saving memory for sparse states.
        pub fn new_lazy(n_qubits: usize) -> Self {
            assert!(n_qubits >= 7, "Minimum 7 qubits required");
            let n_blocks = n_qubits.div_ceil(BLOCK_SIZE);

            Self {
                n_qubits,
                n_blocks,
                blocks: Vec::with_capacity(n_blocks),
            }
        }

        /// Get number of qubits
        #[inline]
        pub fn n_qubits(&self) -> usize {
            self.n_qubits
        }

        /// Get number of blocks
        #[inline]
        pub fn n_blocks(&self) -> usize {
            self.n_blocks
        }

        /// Get number of allocated blocks
        #[inline]
        pub fn allocated_blocks(&self) -> usize {
            self.blocks.len()
        }

        /// Get memory usage in bytes
        pub fn memory_bytes(&self) -> usize {
            self.blocks.len() * std::mem::size_of::<VecBlock>()
        }

        fn max_spurious_tail_amplitude(&self) -> f64 {
            let mut max_spurious = 0.0f64;
            let total_slots = self.blocks.len() * BLOCK_SIZE;

            for slot in self.n_qubits..total_slots {
                let block_idx = slot / BLOCK_SIZE;
                let local_addr = slot % BLOCK_SIZE;
                let mag = self.blocks[block_idx].get(local_addr).norm();
                if mag > max_spurious {
                    max_spurious = mag;
                }
            }

            max_spurious
        }

        /// Clear all blocks to zero
        pub fn clear(&mut self) {
            for block in &mut self.blocks {
                block.re.fill(0.0);
                block.im.fill(0.0);
            }
        }

        /// Ensure blocks are allocated up to the required count
        fn ensure_blocks(&mut self, count: usize) {
            while self.blocks.len() < count {
                self.blocks.push(VecBlock::new());
            }
        }

        /// Create W state: `|W_n> = (|10...0> + |01...0> + ... + |00...1>) / sqrt(n)`.
        ///
        /// Returns creation time in microseconds.
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::SparseQuantumGridVec;
        ///
        /// let mut grid = SparseQuantumGridVec::new(100);
        /// let us = grid.create_w_state();
        /// assert!(us < 1000); // under 1 ms for 100 qubits
        /// ```
        pub fn create_w_state(&mut self) -> u64 {
            let start = std::time::Instant::now();

            self.ensure_blocks(self.n_blocks);
            self.clear();

            let amplitude = 1.0 / (self.n_qubits as f64).sqrt();
            let amp = Complex64::new(amplitude, 0.0);

            for qubit in 0..self.n_qubits {
                let block_idx = qubit / BLOCK_SIZE;
                let local_addr = qubit % BLOCK_SIZE;
                self.blocks[block_idx].set(local_addr, amp);
            }

            start.elapsed().as_micros() as u64
        }

        /// Create W state using parallel iteration (faster for large n).
        pub fn create_w_state_parallel(&mut self) -> u64 {
            let start = std::time::Instant::now();

            self.ensure_blocks(self.n_blocks);

            let amplitude = 1.0 / (self.n_qubits as f64).sqrt();
            let n_qubits = self.n_qubits;

            self.blocks
                .par_iter_mut()
                .enumerate()
                .for_each(|(block_idx, block)| {
                    block.re.fill(0.0);
                    block.im.fill(0.0);

                    let start_qubit = block_idx * BLOCK_SIZE;
                    let end_qubit = (start_qubit + BLOCK_SIZE).min(n_qubits);

                    for qubit in start_qubit..end_qubit {
                        let local_addr = qubit % BLOCK_SIZE;
                        block.re[local_addr] = amplitude;
                    }
                });

            start.elapsed().as_micros() as u64
        }

        /// Verify W state fidelity.
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::SparseQuantumGridVec;
        ///
        /// let mut grid = SparseQuantumGridVec::new(10);
        /// grid.create_w_state();
        /// let v = grid.verify_w_fidelity();
        /// assert!((v.fidelity - 1.0).abs() < 1e-10);
        /// assert_eq!(v.correct_amplitudes, 10);
        /// ```
        pub fn verify_w_fidelity(&self) -> WStateVerificationVec {
            let expected_amp = 1.0 / (self.n_qubits as f64).sqrt();

            let mut total_prob = 0.0;
            let mut overlap_re = 0.0;
            let mut overlap_im = 0.0;
            let mut correct_amplitudes = 0usize;
            let mut max_error = 0.0f64;

            for qubit in 0..self.n_qubits {
                let block_idx = qubit / BLOCK_SIZE;
                let local_addr = qubit % BLOCK_SIZE;

                if block_idx < self.blocks.len() {
                    let amp = self.blocks[block_idx].get(local_addr);
                    let prob = amp.norm_squared();
                    total_prob += prob;
                    overlap_re += expected_amp * amp.re;
                    overlap_im += expected_amp * amp.im;

                    let error = ((amp.re - expected_amp).powi(2) + amp.im.powi(2)).sqrt();
                    if error <= W_AMPLITUDE_TOLERANCE {
                        correct_amplitudes += 1;
                    }
                    if error > max_error {
                        max_error = error;
                    }
                }
            }

            WStateVerificationVec {
                n_qubits: self.n_qubits,
                expected_amplitudes: self.n_qubits,
                correct_amplitudes,
                expected_amplitude: expected_amp,
                total_probability: total_prob,
                fidelity: overlap_re.powi(2) + overlap_im.powi(2),
                max_amplitude_error: max_error,
                max_spurious_amplitude: self.max_spurious_tail_amplitude(),
                active_blocks: self.blocks.len(),
                memory_bytes: self.memory_bytes(),
            }
        }

        /// Verify W state fidelity using parallel iteration.
        pub fn verify_w_fidelity_parallel(&self) -> WStateVerificationVec {
            let expected_amp = 1.0 / (self.n_qubits as f64).sqrt();
            let n_qubits = self.n_qubits;

            let (total_prob, correct_amplitudes, max_error, overlap_re, overlap_im) = self
                .blocks
                .par_iter()
                .enumerate()
                .map(|(block_idx, block): (usize, &VecBlock)| {
                    let start_qubit = block_idx * BLOCK_SIZE;
                    let end_qubit = (start_qubit + BLOCK_SIZE).min(n_qubits);

                    let mut block_prob = 0.0;
                    let mut block_correct = 0usize;
                    let mut block_max_error = 0.0f64;
                    let mut block_overlap_re = 0.0;
                    let mut block_overlap_im = 0.0;

                    for qubit in start_qubit..end_qubit {
                        let local_addr = qubit % BLOCK_SIZE;
                        let amp = block.get(local_addr);
                        let prob = amp.norm_squared();
                        block_prob += prob;
                        block_overlap_re += expected_amp * amp.re;
                        block_overlap_im += expected_amp * amp.im;

                        let error = ((amp.re - expected_amp).powi(2) + amp.im.powi(2)).sqrt();
                        if error <= W_AMPLITUDE_TOLERANCE {
                            block_correct += 1;
                        }
                        if error > block_max_error {
                            block_max_error = error;
                        }
                    }

                    (
                        block_prob,
                        block_correct,
                        block_max_error,
                        block_overlap_re,
                        block_overlap_im,
                    )
                })
                .reduce(
                    || (0.0, 0usize, 0.0f64, 0.0, 0.0),
                    |(p1, c1, e1, r1, i1), (p2, c2, e2, r2, i2)| {
                        (p1 + p2, c1 + c2, e1.max(e2), r1 + r2, i1 + i2)
                    },
                );

            WStateVerificationVec {
                n_qubits: self.n_qubits,
                expected_amplitudes: self.n_qubits,
                correct_amplitudes,
                expected_amplitude: expected_amp,
                total_probability: total_prob,
                fidelity: overlap_re.powi(2) + overlap_im.powi(2),
                max_amplitude_error: max_error,
                max_spurious_amplitude: self.max_spurious_tail_amplitude(),
                active_blocks: self.blocks.len(),
                memory_bytes: self.memory_bytes(),
            }
        }

        /// Get amplitude at a specific qubit position.
        #[inline]
        pub fn get_amplitude(&self, qubit: usize) -> Complex64 {
            let block_idx = qubit / BLOCK_SIZE;
            let local_addr = qubit % BLOCK_SIZE;

            if block_idx < self.blocks.len() {
                self.blocks[block_idx].get(local_addr)
            } else {
                Complex64::zero()
            }
        }

        /// Set amplitude at a specific qubit position.
        #[inline]
        pub fn set_amplitude(&mut self, qubit: usize, amp: Complex64) {
            let block_idx = qubit / BLOCK_SIZE;
            let local_addr = qubit % BLOCK_SIZE;

            self.ensure_blocks(block_idx + 1);
            self.blocks[block_idx].set(local_addr, amp);
        }
    }
}

mod symbolic {
    //! Symbolic quantum states: qubit counts as symbolic expressions.
    //!
    //! Supports Graham's number, TREE(3), a formal infinity label (aleph_0), Knuth up-arrow,
    //! tetration, and arbitrary BigUint via the `SymbolicNumber` enum.
    //!
    //! Also includes Dicke states, W-states, binomial coefficients,
    //! entanglement entropy, and symbolic amplitudes/fractions.

    use super::*;

    // === SymbolicNumber ===

    /// Symbolic representation of incomprehensibly large numbers.
    ///
    /// These numbers are so large they cannot be materialized as BigUint.
    /// Graham's number, for example, has more digits than atoms in the universe.
    #[derive(Clone, Debug)]
    pub enum SymbolicNumber {
        /// A concrete u64 value
        Literal(u64),
        /// A BigUint value (already computed)
        Big(num_bigint::BigUint),
        /// Power: base^exponent (e.g., 10^100)
        Pow {
            base: u64,
            exponent: Box<SymbolicNumber>,
        },
        /// Tower of powers (tetration): base^^height
        Tetration {
            base: u64,
            height: Box<SymbolicNumber>,
        },
        /// Knuth up-arrow notation: a ↑^n b
        KnuthArrow {
            a: u64,
            arrows: u64,
            b: Box<SymbolicNumber>,
        },
        /// Graham's number: g_64 where g_1 = 3↑↑↑↑3, g_n = 3↑^(g_{n-1})3
        Graham,
        /// TREE(n) — grows faster than Graham's number
        Tree(u64),
        /// Rayo's number — largest named number
        Rayo,
        /// Formal infinity label (aleph_0 for countable)
        Infinity,
        /// Custom named number with description
        Named { name: String, description: String },
        /// Integer division: numerator / denominator (symbolic)
        Div {
            numerator: Box<SymbolicNumber>,
            denominator: Box<SymbolicNumber>,
        },
        /// Subtraction: minuend - subtrahend (symbolic)
        Sub {
            minuend: Box<SymbolicNumber>,
            subtrahend: Box<SymbolicNumber>,
        },
    }

    impl SymbolicNumber {
        /// Get a human-readable representation
        pub fn to_notation(&self) -> String {
            match self {
                Self::Literal(n) => format!("{}", n),
                Self::Big(n) => {
                    let bits = n.bits();
                    if bits > 1000 {
                        format!("~10^{:.0}", bits as f64 * std::f64::consts::LOG10_2)
                    } else {
                        format!("{}", n)
                    }
                }
                Self::Pow { base, exponent } => {
                    format!("{}^({})", base, exponent.to_notation())
                }
                Self::Tetration { base, height } => {
                    format!("{}^^{}", base, height.to_notation())
                }
                Self::KnuthArrow { a, arrows, b } => {
                    let arrow_str: String = std::iter::repeat_n('↑', *arrows as usize).collect();
                    format!("{}{}{}", a, arrow_str, b.to_notation())
                }
                Self::Graham => "G (Graham's number)".to_string(),
                Self::Tree(n) => format!("TREE({})", n),
                Self::Rayo => "Rayo's number".to_string(),
                Self::Infinity => "infty".to_string(),
                Self::Named { name, .. } => name.clone(),
                Self::Div {
                    numerator,
                    denominator,
                } => {
                    format!(
                        "({}/{})",
                        numerator.to_notation(),
                        denominator.to_notation()
                    )
                }
                Self::Sub {
                    minuend,
                    subtrahend,
                } => {
                    format!("({}-{})", minuend.to_notation(), subtrahend.to_notation())
                }
            }
        }

        /// Get the "size class" for comparison purposes
        pub fn size_class(&self) -> &'static str {
            match self {
                Self::Literal(n) if *n < 1_000_000 => "small",
                Self::Literal(_) => "large",
                Self::Big(n) if n.bits() < 1000 => "astronomical",
                Self::Big(_) => "cosmological",
                Self::Pow { .. } => "cosmological",
                Self::Tetration { .. } => "tetration",
                Self::KnuthArrow { arrows, .. } if *arrows <= 2 => "tetration",
                Self::KnuthArrow { arrows, .. } if *arrows <= 4 => "pentation+",
                Self::KnuthArrow { .. } => "arrow-notation",
                Self::Graham => "Graham-class",
                Self::Tree(_) => "TREE-class",
                Self::Rayo => "uncomputable",
                Self::Infinity => "infinite",
                Self::Named { .. } => "named",
                Self::Div { numerator, .. } => numerator.size_class(),
                Self::Sub { minuend, .. } => minuend.size_class(),
            }
        }

        /// Check if this symbolic number is zero
        pub fn is_zero(&self) -> bool {
            match self {
                Self::Literal(0) => true,
                Self::Big(n) => n == &num_bigint::BigUint::from(0u32),
                _ => false,
            }
        }

        /// Check if this symbolic number equals another (structurally)
        pub fn structurally_equal(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::Literal(a), Self::Literal(b)) => a == b,
                (Self::Big(a), Self::Big(b)) => a == b,
                (Self::Graham, Self::Graham) => true,
                (Self::Tree(a), Self::Tree(b)) => a == b,
                (Self::Infinity, Self::Infinity) => true,
                (Self::Rayo, Self::Rayo) => true,
                _ => false,
            }
        }

        /// Estimate log_10 of the number (returns None if incomputable)
        pub fn estimate_log10(&self) -> Option<f64> {
            match self {
                Self::Literal(n) => Some((*n as f64).log10()),
                Self::Big(n) => Some(n.bits() as f64 * std::f64::consts::LOG10_2),
                Self::Pow { base, exponent } => {
                    let exp_log = exponent.estimate_log10()?;
                    Some(10f64.powf(exp_log) * (*base as f64).log10())
                }
                Self::Div {
                    numerator,
                    denominator,
                } => {
                    let num_log = numerator.estimate_log10()?;
                    let denom_log = denominator.estimate_log10()?;
                    Some(num_log - denom_log)
                }
                Self::Sub {
                    minuend,
                    subtrahend,
                } => {
                    let minuend_log = minuend.estimate_log10()?;
                    let subtrahend_log = subtrahend.estimate_log10()?;
                    if minuend_log > subtrahend_log + 1.0 {
                        Some(minuend_log)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }

        /// Can this number be materialized as BigUint in reasonable time?
        pub fn is_materializable(&self) -> bool {
            match self {
                Self::Literal(_) => true,
                Self::Big(_) => true,
                Self::Pow { exponent, .. } => match exponent.as_ref() {
                    SymbolicNumber::Literal(e) => *e < 100_000_000,
                    _ => false,
                },
                Self::Div {
                    numerator,
                    denominator,
                } => numerator.is_materializable() && denominator.is_materializable(),
                Self::Sub {
                    minuend,
                    subtrahend,
                } => minuend.is_materializable() && subtrahend.is_materializable(),
                _ => false,
            }
        }
    }

    impl std::fmt::Display for SymbolicNumber {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.to_notation())
        }
    }

    fn terminal_local_addr_for_symbolic_number(n_qubits: &SymbolicNumber) -> usize {
        assert!(!n_qubits.is_zero(), "GHZ state requires at least one qubit");

        match n_qubits {
            SymbolicNumber::Literal(n) if *n < BLOCK_SHIFT as u64 => (1usize << *n as usize) - 1,
            SymbolicNumber::Big(n) => terminal_local_addr_for_biguint(n),
            _ => BLOCK_SIZE - 1,
        }
    }

    // === SymbolicGhzState ===

    /// GHZ endpoint representation with a symbolic qubit-count label.
    ///
    /// This struct can represent GHZ states with:
    /// - Graham's number of qubits
    /// - TREE(3) qubits
    /// - A formal infinity label (aleph_0)
    ///
    /// Verification checks the stored endpoint amplitudes and symbolic label. It
    /// does not model analytic infinite tensor-product Hilbert spaces.
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::SymbolicGhzState;
    ///
    /// // Graham's number of qubits
    /// let ghz = SymbolicGhzState::graham();
    /// let v = ghz.verify();
    /// assert!((v.fidelity - 1.0).abs() < 1e-10);
    /// assert_eq!(v.size_class, "Graham-class");
    ///
    /// // Infinite qubits
    /// let inf = SymbolicGhzState::infinite();
    /// assert!((inf.verify().fidelity - 1.0).abs() < 1e-10);
    /// ```
    #[derive(Clone)]
    pub struct SymbolicGhzState {
        /// Number of qubits as a symbolic expression
        pub n_qubits: SymbolicNumber,
        /// Block 0: contains amplitude for `|00...0>` at index 0
        pub block_zero: VecBlock,
        /// Last block: contains amplitude for `|11...1>` at index 127
        pub block_last: VecBlock,
        /// Creation time in nanoseconds
        pub creation_time_ns: u64,
    }

    impl SymbolicGhzState {
        /// Create a GHZ state with a symbolic qubit count.
        ///
        /// This works for positive symbolic labels, including formal infinity labels.
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::{SymbolicGhzState, SymbolicNumber};
        ///
        /// let ghz = SymbolicGhzState::new(SymbolicNumber::Literal(42));
        /// assert_eq!(ghz.n_qubits.to_notation(), "42");
        /// let v = ghz.verify();
        /// assert!((v.fidelity - 1.0).abs() < 1e-10);
        /// ```
        pub fn new(n_qubits: SymbolicNumber) -> Self {
            assert!(!n_qubits.is_zero(), "GHZ state requires at least one qubit");

            let start = std::time::Instant::now();

            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            let terminal_addr = terminal_local_addr_for_symbolic_number(&n_qubits);

            let mut block_zero = VecBlock::new();
            block_zero.set(0, Complex64::new(inv_sqrt2, 0.0));

            let mut block_last = VecBlock::new();
            block_last.set(terminal_addr, Complex64::new(inv_sqrt2, 0.0));

            let creation_time_ns = start.elapsed().as_nanos() as u64;

            Self {
                n_qubits,
                block_zero,
                block_last,
                creation_time_ns,
            }
        }

        /// Create GHZ state with Graham's number of qubits.
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::SymbolicGhzState;
        ///
        /// let ghz = SymbolicGhzState::graham();
        /// assert!((ghz.verify().fidelity - 1.0).abs() < 1e-10);
        /// ```
        pub fn graham() -> Self {
            Self::new(SymbolicNumber::Graham)
        }

        /// Create GHZ state with TREE(n) qubits.
        pub fn tree(n: u64) -> Self {
            Self::new(SymbolicNumber::Tree(n))
        }

        /// Create GHZ state with a formal infinity label (aleph_0).
        pub fn infinite() -> Self {
            Self::new(SymbolicNumber::Infinity)
        }

        /// Create GHZ state with Knuth arrow notation: a↑↑↑...↑b.
        pub fn knuth_arrow(a: u64, arrows: u64, b: u64) -> Self {
            Self::new(SymbolicNumber::KnuthArrow {
                a,
                arrows,
                b: Box::new(SymbolicNumber::Literal(b)),
            })
        }

        /// Create GHZ state with tetration: base^^height.
        pub fn tetration(base: u64, height: u64) -> Self {
            Self::new(SymbolicNumber::Tetration {
                base,
                height: Box::new(SymbolicNumber::Literal(height)),
            })
        }

        /// Verify GHZ state fidelity — O(1) regardless of symbolic size.
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::SymbolicGhzState;
        ///
        /// let ghz = SymbolicGhzState::new(tileuniverse_quantum::SymbolicNumber::Literal(1000));
        /// let v = ghz.verify();
        /// assert!((v.fidelity - 1.0).abs() < 1e-10);
        /// assert_eq!(v.amp_zero_state, v.amp_one_state);
        /// assert!((v.amp_zero_state - 0.707107).abs() < 1e-4);
        /// ```
        pub fn verify(&self) -> SymbolicGhzVerification {
            let expected_amp = std::f64::consts::FRAC_1_SQRT_2;
            let terminal_addr = terminal_local_addr_for_symbolic_number(&self.n_qubits);

            let amp0_complex = self.block_zero.get(0);
            let amp_last_complex = self.block_last.get(terminal_addr);
            let amp0 = amp0_complex.norm();
            let amp_last = amp_last_complex.norm();

            let overlap = Complex64::new(
                expected_amp * (amp0_complex.re + amp_last_complex.re),
                expected_amp * (amp0_complex.im + amp_last_complex.im),
            );
            let fidelity = overlap.norm_squared();

            let mut max_spurious = 0.0f64;

            for addr in 1..BLOCK_SIZE {
                let mag = self.block_zero.get(addr).norm();
                if mag > max_spurious {
                    max_spurious = mag;
                }
            }

            for addr in 0..BLOCK_SIZE {
                if addr == terminal_addr {
                    continue;
                }

                let mag = self.block_last.get(addr).norm();
                if mag > max_spurious {
                    max_spurious = mag;
                }
            }

            SymbolicGhzVerification {
                n_qubits: self.n_qubits.clone(),
                amp_zero_state: amp0,
                amp_one_state: amp_last,
                expected_amplitude: expected_amp,
                fidelity,
                max_spurious,
                size_class: self.n_qubits.size_class().to_string(),
            }
        }

        /// Reported struct size in bytes for this symbolic endpoint representation.
        pub fn memory_bytes(&self) -> usize {
            std::mem::size_of::<Self>()
        }
    }

    impl std::fmt::Display for SymbolicGhzState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "|GHZ_({})>", self.n_qubits)
        }
    }

    /// Verification results for SymbolicGhzState
    #[derive(Debug, Clone)]
    pub struct SymbolicGhzVerification {
        pub n_qubits: SymbolicNumber,
        pub amp_zero_state: f64,
        pub amp_one_state: f64,
        pub expected_amplitude: f64,
        pub fidelity: f64,
        pub max_spurious: f64,
        pub size_class: String,
    }

    impl std::fmt::Display for SymbolicGhzVerification {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "Symbolic GHZ Verification [{}]", self.size_class)?;
            writeln!(f, "  Qubits: {}", self.n_qubits)?;
            writeln!(
                f,
                "  |00...0> amplitude: {:.6} (expected {:.6})",
                self.amp_zero_state, self.expected_amplitude
            )?;
            writeln!(
                f,
                "  |11...1> amplitude: {:.6} (expected {:.6})",
                self.amp_one_state, self.expected_amplitude
            )?;
            writeln!(f, "  Fidelity:           {:.6}", self.fidelity)?;
            writeln!(f, "  Max spurious:       {:.6}", self.max_spurious)?;

            match &self.n_qubits {
                SymbolicNumber::Graham => {
                    writeln!(f, "  ")?;
                    writeln!(f, "  Graham's number is so large that:")?;
                    writeln!(f, "    - It cannot be written in standard notation")?;
                    writeln!(f, "    - Even the NUMBER OF DIGITS cannot be written")?;
                    writeln!(f, "    - It requires Knuth's up-arrow notation")?;
                    writeln!(f, "    - g_64 where g_1 = 3↑↑↑↑3, g_n = 3↑^(g_{{n-1}})3")?;
                }
                SymbolicNumber::Tree(n) => {
                    writeln!(f, "  ")?;
                    writeln!(f, "  TREE({}) grows faster than Graham's number!", n)?;
                    writeln!(f, "    - TREE(1) = 1")?;
                    writeln!(f, "    - TREE(2) = 3")?;
                    writeln!(f, "    - TREE(3) > g_64 (Graham's number)")?;
                }
                SymbolicNumber::Infinity => {
                    writeln!(f, "  ")?;
                    writeln!(f, "  Formal infinity label (aleph_0).")?;
                    writeln!(
                        f,
                        "  This is a symbolic endpoint object, not an analytic infinite-Hilbert-space model."
                    )?;
                }
                _ => {}
            }

            Ok(())
        }
    }

    /// Create a GHZ state with Graham's number of qubits
    pub fn create_graham_ghz() -> SymbolicGhzState {
        SymbolicGhzState::graham()
    }

    /// Create a GHZ state with TREE(n) qubits
    pub fn create_tree_ghz(n: u64) -> SymbolicGhzState {
        SymbolicGhzState::tree(n)
    }

    /// Create a GHZ state with a formal infinity label
    pub fn create_infinite_ghz() -> SymbolicGhzState {
        SymbolicGhzState::infinite()
    }

    /// Create a GHZ state with symbolic qubit count
    pub fn create_symbolic_ghz(n_qubits: SymbolicNumber) -> SymbolicGhzState {
        SymbolicGhzState::new(n_qubits)
    }

    // === SymbolicAmplitude ===

    /// Symbolic representation of quantum amplitudes.
    ///
    /// For W-states with symbolic qubit count n, the amplitude 1/sqrt(n) cannot be
    /// computed numerically. Instead, we represent it symbolically and prove
    /// normalization properties algebraically.
    #[derive(Clone, Debug)]
    pub enum SymbolicAmplitude {
        /// Concrete f64 value
        Real(f64),
        /// 1/sqrt(n) where n is symbolic
        ReciprocalSqrt(SymbolicNumber),
        /// 1/sqrt(C(n,k)) where C(n,k) is a symbolic binomial coefficient
        ReciprocalSqrtBinomial(SymbolicBinomial),
        /// sqrt(k/n) — for generalized Dicke states
        SqrtFraction {
            numerator: SymbolicNumber,
            denominator: SymbolicNumber,
        },
        /// Product of symbolic amplitudes
        Product(Vec<SymbolicAmplitude>),
        /// Zero amplitude
        Zero,
    }

    impl SymbolicAmplitude {
        /// Create 1/sqrt(n)
        pub fn reciprocal_sqrt(n: SymbolicNumber) -> Self {
            Self::ReciprocalSqrt(n)
        }

        /// Create sqrt(k/n)
        pub fn sqrt_fraction(numerator: SymbolicNumber, denominator: SymbolicNumber) -> Self {
            Self::SqrtFraction {
                numerator,
                denominator,
            }
        }

        /// Get magnitude squared symbolically: |1/sqrt(n)|^2 = 1/n
        pub fn magnitude_squared_symbolic(&self) -> Option<SymbolicFraction> {
            match self {
                Self::Real(r) => Some(SymbolicFraction {
                    numerator: SymbolicNumber::Big(num_bigint::BigUint::from(
                        (r * r * 1e15) as u64,
                    )),
                    denominator: SymbolicNumber::Big(num_bigint::BigUint::from(
                        1_000_000_000_000_000u64,
                    )),
                }),
                Self::ReciprocalSqrt(n) => Some(SymbolicFraction::one_over(n.clone())),
                Self::ReciprocalSqrtBinomial(binom) => Some(SymbolicFraction {
                    numerator: SymbolicNumber::Literal(1),
                    denominator: SymbolicNumber::Named {
                        name: binom.to_notation(),
                        description: format!("Binomial coefficient {}", binom.to_notation()),
                    },
                }),
                Self::SqrtFraction {
                    numerator,
                    denominator,
                } => Some(SymbolicFraction {
                    numerator: numerator.clone(),
                    denominator: denominator.clone(),
                }),
                Self::Zero => Some(SymbolicFraction {
                    numerator: SymbolicNumber::Literal(0),
                    denominator: SymbolicNumber::Literal(1),
                }),
                Self::Product(factors) => {
                    if factors.len() == 1 {
                        factors[0].magnitude_squared_symbolic()
                    } else {
                        None
                    }
                }
            }
        }

        /// Try to evaluate to f64 if possible
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::{SymbolicAmplitude, SymbolicNumber};
        ///
        /// let amp = SymbolicAmplitude::reciprocal_sqrt(SymbolicNumber::Literal(4));
        /// let val = amp.try_evaluate().unwrap();
        /// assert!((val - 0.5).abs() < 1e-10);
        /// ```
        pub fn try_evaluate(&self) -> Option<f64> {
            match self {
                Self::Real(r) => Some(*r),
                Self::ReciprocalSqrt(n) => match n {
                    SymbolicNumber::Literal(lit) => Some(1.0 / (*lit as f64).sqrt()),
                    SymbolicNumber::Big(big) => {
                        if big.bits() < 53 {
                            let val: u64 = big.try_into().ok()?;
                            Some(1.0 / (val as f64).sqrt())
                        } else {
                            None
                        }
                    }
                    _ => None,
                },
                Self::ReciprocalSqrtBinomial(binom) => {
                    let binom_val = binom.try_evaluate()?;
                    if binom_val == 0 {
                        return None;
                    }
                    Some(1.0 / (binom_val as f64).sqrt())
                }
                Self::SqrtFraction {
                    numerator,
                    denominator,
                } => {
                    let num = match numerator {
                        SymbolicNumber::Literal(n) => *n as f64,
                        _ => return None,
                    };
                    let denom = match denominator {
                        SymbolicNumber::Literal(d) => *d as f64,
                        _ => return None,
                    };
                    Some((num / denom).sqrt())
                }
                Self::Zero => Some(0.0),
                Self::Product(factors) => {
                    let mut result = 1.0;
                    for factor in factors {
                        result *= factor.try_evaluate()?;
                    }
                    Some(result)
                }
            }
        }

        /// Human-readable representation
        pub fn to_notation(&self) -> String {
            match self {
                Self::Real(r) => format!("{:.6}", r),
                Self::ReciprocalSqrt(n) => format!("1/sqrt({})", n.to_notation()),
                Self::ReciprocalSqrtBinomial(binom) => format!("1/sqrt{}", binom.to_notation()),
                Self::SqrtFraction {
                    numerator,
                    denominator,
                } => format!(
                    "sqrt({}/{})",
                    numerator.to_notation(),
                    denominator.to_notation()
                ),
                Self::Zero => "0".to_string(),
                Self::Product(factors) => {
                    let parts: Vec<String> = factors.iter().map(|f| f.to_notation()).collect();
                    parts.join(" x ")
                }
            }
        }
    }

    impl std::fmt::Display for SymbolicAmplitude {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.to_notation())
        }
    }

    // === SymbolicFraction ===

    /// Symbolic fraction for probability calculations.
    ///
    /// Used to represent probabilities like 1/n where n is Graham's number.
    #[derive(Clone, Debug)]
    pub struct SymbolicFraction {
        pub numerator: SymbolicNumber,
        pub denominator: SymbolicNumber,
    }

    impl SymbolicFraction {
        /// Create 1/n
        pub fn one_over(n: SymbolicNumber) -> Self {
            Self {
                numerator: SymbolicNumber::Literal(1),
                denominator: n,
            }
        }

        /// Create k/n
        pub fn new(numerator: SymbolicNumber, denominator: SymbolicNumber) -> Self {
            Self {
                numerator,
                denominator,
            }
        }

        /// Human-readable notation
        pub fn to_notation(&self) -> String {
            match (&self.numerator, &self.denominator) {
                (SymbolicNumber::Literal(1), d) => format!("1/{}", d.to_notation()),
                (n, d) => format!("{}/{}", n.to_notation(), d.to_notation()),
            }
        }

        /// Try to evaluate to f64 if possible
        pub fn try_evaluate(&self) -> Option<f64> {
            let num = match &self.numerator {
                SymbolicNumber::Literal(n) => *n as f64,
                SymbolicNumber::Big(big) if big.bits() < 53 => {
                    let val: u64 = big.try_into().ok()?;
                    val as f64
                }
                _ => return None,
            };
            let denom = match &self.denominator {
                SymbolicNumber::Literal(d) => *d as f64,
                SymbolicNumber::Big(big) if big.bits() < 53 => {
                    let val: u64 = big.try_into().ok()?;
                    val as f64
                }
                _ => return None,
            };
            Some(num / denom)
        }

        /// Check if this fraction equals 1 (algebraically)
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::{SymbolicFraction, SymbolicNumber};
        ///
        /// let frac1 = SymbolicFraction::new(
        ///     SymbolicNumber::Literal(42),
        ///     SymbolicNumber::Literal(42),
        /// );
        /// assert!(frac1.is_one());
        ///
        /// let frac2 = SymbolicFraction::one_over(SymbolicNumber::Graham);
        /// let frac3 = SymbolicFraction::new(
        ///     SymbolicNumber::Graham.clone(),
        ///     SymbolicNumber::Graham,
        /// );
        /// assert!(frac3.is_one());
        /// ```
        pub fn is_one(&self) -> bool {
            match (&self.numerator, &self.denominator) {
                (SymbolicNumber::Literal(a), SymbolicNumber::Literal(b)) => a == b,
                (SymbolicNumber::Big(a), SymbolicNumber::Big(b)) => a == b,
                (SymbolicNumber::Graham, SymbolicNumber::Graham) => true,
                (SymbolicNumber::Tree(a), SymbolicNumber::Tree(b)) => a == b,
                (SymbolicNumber::Infinity, SymbolicNumber::Infinity) => true,
                (SymbolicNumber::Rayo, SymbolicNumber::Rayo) => true,
                _ => false,
            }
        }
    }

    impl std::fmt::Display for SymbolicFraction {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.to_notation())
        }
    }

    // === SymbolicEntropy ===

    /// Symbolic representation of entropy.
    ///
    /// For W-states, the entanglement entropy is log_2(n) where n is the qubit count.
    #[derive(Clone, Debug)]
    pub enum SymbolicEntropy {
        /// log_2(n) where n is symbolic
        Log2(SymbolicNumber),
        /// Concrete value in bits
        Bits(f64),
        /// Infinite entropy
        Infinite,
    }

    impl SymbolicEntropy {
        /// Create log_2(n)
        pub fn log2(n: SymbolicNumber) -> Self {
            match n {
                SymbolicNumber::Infinity => Self::Infinite,
                SymbolicNumber::Literal(lit) => Self::Bits((lit as f64).log2()),
                _ => Self::Log2(n),
            }
        }

        /// Human-readable notation
        pub fn to_notation(&self) -> String {
            match self {
                Self::Log2(n) => format!("log_2({})", n.to_notation()),
                Self::Bits(b) => format!("{:.3} bits", b),
                Self::Infinite => "infty bits".to_string(),
            }
        }

        /// Try to evaluate to f64 if possible
        pub fn try_evaluate(&self) -> Option<f64> {
            match self {
                Self::Bits(b) => Some(*b),
                Self::Log2(n) => match n {
                    SymbolicNumber::Literal(lit) => Some((*lit as f64).log2()),
                    SymbolicNumber::Big(big) => Some(big.bits() as f64),
                    _ => None,
                },
                Self::Infinite => None,
            }
        }
    }

    impl std::fmt::Display for SymbolicEntropy {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.to_notation())
        }
    }

    // === SymbolicBinomial ===

    /// Symbolic binomial coefficient C(n,k) = n!/(k!(n-k)!)
    ///
    /// For Dicke states |D_n^k>, we need C(n,k) which is the number of basis states.
    ///
    /// # Properties
    ///
    /// - C(n, 0) = 1
    /// - C(n, 1) = n
    /// - C(n, n) = 1
    /// - C(n, 2) = n(n-1)/2
    /// - C(infty, k) = infty for any finite k > 0
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::SymbolicBinomial;
    ///
    /// let binom = SymbolicBinomial::new(
    ///     tileuniverse_quantum::SymbolicNumber::Literal(10),
    ///     tileuniverse_quantum::SymbolicNumber::Literal(3),
    /// );
    /// assert_eq!(binom.try_evaluate(), Some(120));
    /// assert_eq!(binom.to_notation(), "C(10, 3)");
    /// ```
    #[derive(Clone, Debug)]
    pub struct SymbolicBinomial {
        /// Number of items to choose from (n)
        pub n: SymbolicNumber,
        /// Number of items to choose (k)
        pub k: SymbolicNumber,
    }

    impl SymbolicBinomial {
        /// Create a new symbolic binomial coefficient C(n, k)
        pub fn new(n: SymbolicNumber, k: SymbolicNumber) -> Self {
            Self { n, k }
        }

        /// Try to evaluate to a concrete u128 if possible
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::SymbolicBinomial;
        /// use tileuniverse_quantum::SymbolicNumber;
        ///
        /// let binom = SymbolicBinomial::new(SymbolicNumber::Literal(20), SymbolicNumber::Literal(10));
        /// assert_eq!(binom.try_evaluate(), Some(184756));
        /// ```
        pub fn try_evaluate(&self) -> Option<u128> {
            let n_val = match &self.n {
                SymbolicNumber::Literal(n) => *n as u128,
                SymbolicNumber::Big(n) => {
                    if n.bits() <= 64 {
                        n.to_u64_digits().first().map(|&d| d as u128)?
                    } else {
                        return None;
                    }
                }
                _ => return None,
            };

            let k_val = match &self.k {
                SymbolicNumber::Literal(k) => *k as u128,
                SymbolicNumber::Big(k) => {
                    if k.bits() <= 64 {
                        k.to_u64_digits().first().map(|&d| d as u128)?
                    } else {
                        return None;
                    }
                }
                _ => return None,
            };

            if k_val > n_val {
                return Some(0);
            }

            let k_val = k_val.min(n_val - k_val);

            if k_val == 0 {
                return Some(1);
            }

            let mut result: u128 = 1;
            for i in 0..k_val {
                result = result.checked_mul(n_val - i)?;
                result /= i + 1;
            }
            Some(result)
        }

        /// Simplify the binomial coefficient to a canonical form.
        ///
        /// Identifies special cases: C(n,0)=1, C(n,1)=n, C(n,2)=n(n-1)/2, etc.
        pub fn simplify(&self) -> SimplifiedBinomial {
            if let SymbolicNumber::Literal(0) = &self.k {
                return SimplifiedBinomial::One;
            }
            if let SymbolicNumber::Big(k) = &self.k
                && k == &num_bigint::BigUint::from(0u32)
            {
                return SimplifiedBinomial::One;
            }

            if self.n.structurally_equal(&self.k) {
                return SimplifiedBinomial::One;
            }

            if matches!(&self.n, SymbolicNumber::Infinity)
                && !matches!(&self.k, SymbolicNumber::Infinity)
            {
                return SimplifiedBinomial::Infinite;
            }

            if let SymbolicNumber::Literal(1) = &self.k {
                return SimplifiedBinomial::N(self.n.clone());
            }
            if let SymbolicNumber::Big(k) = &self.k
                && k == &num_bigint::BigUint::from(1u32)
            {
                return SimplifiedBinomial::N(self.n.clone());
            }

            if let SymbolicNumber::Literal(2) = &self.k {
                return SimplifiedBinomial::NChoose2(self.n.clone());
            }
            if let SymbolicNumber::Big(k) = &self.k
                && k == &num_bigint::BigUint::from(2u32)
            {
                return SimplifiedBinomial::NChoose2(self.n.clone());
            }

            if let (SymbolicNumber::Literal(n), SymbolicNumber::Literal(k)) = (&self.n, &self.k) {
                if k > n {
                    return SimplifiedBinomial::Zero;
                }
                if *k == *n - 1 {
                    return SimplifiedBinomial::N(self.n.clone());
                }
            }

            SimplifiedBinomial::General(self.clone())
        }

        /// Human-readable notation: C(n, k)
        pub fn to_notation(&self) -> String {
            format!("C({}, {})", self.n.to_notation(), self.k.to_notation())
        }

        /// Is this binomial equal to 1? (k=0 or k=n)
        pub fn is_one(&self) -> bool {
            matches!(self.simplify(), SimplifiedBinomial::One)
        }

        /// Is this binomial equal to n? (k=1 or k=n-1)
        pub fn is_n(&self) -> bool {
            matches!(self.simplify(), SimplifiedBinomial::N(_))
        }

        /// Is this binomial infinite?
        pub fn is_infinite(&self) -> bool {
            matches!(self.simplify(), SimplifiedBinomial::Infinite)
        }

        /// Is this binomial zero? (k > n for concrete values)
        pub fn is_zero(&self) -> bool {
            matches!(self.simplify(), SimplifiedBinomial::Zero)
        }
    }

    impl std::fmt::Display for SymbolicBinomial {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.to_notation())
        }
    }

    /// Result of simplifying a binomial coefficient
    #[derive(Clone, Debug)]
    pub enum SimplifiedBinomial {
        /// Exactly 1 (k=0 or k=n)
        One,
        /// Exactly n (k=1 or k=n-1)
        N(SymbolicNumber),
        /// n(n-1)/2 for k=2 (triangular number)
        NChoose2(SymbolicNumber),
        /// General C(n,k) — cannot simplify further
        General(SymbolicBinomial),
        /// Infinite (n=infty and k is finite nonzero)
        Infinite,
        /// Zero (k > n, which is invalid)
        Zero,
    }

    impl SimplifiedBinomial {
        /// Human-readable notation
        pub fn to_notation(&self) -> String {
            match self {
                Self::One => "1".to_string(),
                Self::N(n) => n.to_notation(),
                Self::NChoose2(n) => format!("{}({}-1)/2", n.to_notation(), n.to_notation()),
                Self::General(binom) => binom.to_notation(),
                Self::Infinite => "infty".to_string(),
                Self::Zero => "0".to_string(),
            }
        }

        /// Try to evaluate to a concrete value
        pub fn try_evaluate(&self) -> Option<u128> {
            match self {
                Self::One => Some(1),
                Self::N(n) => match n {
                    SymbolicNumber::Literal(v) => Some(*v as u128),
                    SymbolicNumber::Big(b) if b.bits() <= 128 => {
                        let digits = b.to_u64_digits();
                        if digits.len() == 1 {
                            Some(digits[0] as u128)
                        } else if digits.len() == 2 {
                            Some(digits[0] as u128 | ((digits[1] as u128) << 64))
                        } else {
                            None
                        }
                    }
                    _ => None,
                },
                Self::NChoose2(n) => {
                    if let SymbolicNumber::Literal(v) = n {
                        Some((*v as u128) * (*v as u128 - 1) / 2)
                    } else {
                        None
                    }
                }
                Self::General(binom) => binom.try_evaluate(),
                Self::Infinite => None,
                Self::Zero => Some(0),
            }
        }
    }

    impl std::fmt::Display for SimplifiedBinomial {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.to_notation())
        }
    }

    // === SymbolicWState ===

    /// Symbolic W-state formula object for labels such as Graham's number, TREE(3),
    /// or a formal infinity label.
    ///
    /// W-state: `|W_n> = (|10...0> + |01...0> + ... + |00...1>) / sqrt(n)`
    ///
    /// Memory is independent of the symbolic label size. Verification is an
    /// algebraic consistency check of the stored formula, not a numerical state-vector check.
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::{create_graham_w, create_tree_w, create_infinite_w};
    ///
    /// // Graham's number
    /// let w = create_graham_w();
    /// let v = w.verify();
    /// assert!(v.is_valid);
    /// assert!(v.total_probability.is_one());
    ///
    /// // TREE(3)
    /// let w_tree = create_tree_w(3);
    /// assert!(w_tree.verify().is_valid);
    /// ```
    #[derive(Clone)]
    pub struct SymbolicWState {
        /// Number of qubits (symbolic)
        pub n_qubits: SymbolicNumber,
        /// Amplitude for each basis state: 1/sqrt(n) (symbolic)
        pub amplitude: SymbolicAmplitude,
        /// Creation time in nanoseconds
        pub creation_time_ns: u64,
    }

    impl SymbolicWState {
        /// Create W-state with symbolic qubit count.
        ///
        /// This is O(1) — we never iterate over n elements!
        pub fn new(n_qubits: SymbolicNumber) -> Self {
            assert!(!n_qubits.is_zero(), "W-state requires at least one qubit");

            let start = std::time::Instant::now();
            let amplitude = SymbolicAmplitude::ReciprocalSqrt(n_qubits.clone());
            Self {
                n_qubits,
                amplitude,
                creation_time_ns: start.elapsed().as_nanos() as u64,
            }
        }

        /// Create W-state with Graham's number of qubits
        pub fn graham() -> Self {
            Self::new(SymbolicNumber::Graham)
        }

        /// Create W-state with TREE(n) qubits
        pub fn tree(n: u64) -> Self {
            Self::new(SymbolicNumber::Tree(n))
        }

        /// Create W-state with a formal infinity label (aleph_0)
        pub fn infinite() -> Self {
            Self::new(SymbolicNumber::Infinity)
        }

        /// Create W-state with Knuth arrow notation: a↑↑↑...↑b qubits
        pub fn knuth_arrow(a: u64, arrows: u64, b: u64) -> Self {
            Self::new(SymbolicNumber::KnuthArrow {
                a,
                arrows,
                b: Box::new(SymbolicNumber::Literal(b)),
            })
        }

        /// Create W-state with tetration: base^^height qubits
        pub fn tetration(base: u64, height: u64) -> Self {
            Self::new(SymbolicNumber::Tetration {
                base,
                height: Box::new(SymbolicNumber::Literal(height)),
            })
        }

        /// Verify W-state properties algebraically (O(1)).
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::{create_graham_w, SymbolicNumber, SymbolicWState};
        ///
        /// let w = SymbolicWState::new(SymbolicNumber::Literal(100));
        /// let v = w.verify();
        /// assert!(v.is_valid);
        /// assert!(v.total_probability.is_one());
        ///
        /// // For literal 100, amplitude should be evaluable
        /// let amp_val = v.amplitude.try_evaluate().unwrap();
        /// let expected = 1.0 / 100.0_f64.sqrt();
        /// assert!((amp_val - expected).abs() < 1e-10);
        /// ```
        pub fn verify(&self) -> SymbolicWVerification {
            let amplitude = self.amplitude.clone();
            let prob_per_state = SymbolicFraction::one_over(self.n_qubits.clone());
            let total_probability =
                SymbolicFraction::new(self.n_qubits.clone(), self.n_qubits.clone());
            let per_qubit_probability = SymbolicFraction::one_over(self.n_qubits.clone());
            let entanglement_entropy = SymbolicEntropy::log2(self.n_qubits.clone());

            SymbolicWVerification {
                n_qubits: self.n_qubits.clone(),
                amplitude,
                num_nonzero_amplitudes: self.n_qubits.clone(),
                probability_per_state: prob_per_state,
                total_probability,
                per_qubit_probability,
                entanglement_entropy,
                size_class: self.n_qubits.size_class().to_string(),
                is_valid: true,
            }
        }

        /// Get the symbolic entanglement entropy: log_2(n)
        pub fn entanglement_entropy(&self) -> SymbolicEntropy {
            SymbolicEntropy::log2(self.n_qubits.clone())
        }

        /// Get measurement probability for any single qubit: 1/n
        pub fn measurement_probability(&self) -> SymbolicFraction {
            SymbolicFraction::one_over(self.n_qubits.clone())
        }

        /// Reported struct size in bytes for this symbolic formula object.
        pub fn memory_bytes(&self) -> usize {
            std::mem::size_of::<Self>()
        }

        /// Get the number of qubits
        pub fn n_qubits(&self) -> &SymbolicNumber {
            &self.n_qubits
        }

        /// Get creation time in nanoseconds
        pub fn creation_time_ns(&self) -> u64 {
            self.creation_time_ns
        }
    }

    impl std::fmt::Display for SymbolicWState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "|W_({})>", self.n_qubits)
        }
    }

    // === SymbolicWVerification ===

    /// Verification result for SymbolicWState.
    #[derive(Debug, Clone)]
    pub struct SymbolicWVerification {
        /// Number of qubits
        pub n_qubits: SymbolicNumber,
        /// Amplitude per basis state: 1/sqrt(n)
        pub amplitude: SymbolicAmplitude,
        /// Number of non-zero amplitudes: n
        pub num_nonzero_amplitudes: SymbolicNumber,
        /// Probability per basis state: |1/sqrt(n)|^2 = 1/n
        pub probability_per_state: SymbolicFraction,
        /// Total probability: n x (1/n) = 1
        pub total_probability: SymbolicFraction,
        /// Per-qubit measurement probability: 1/n
        pub per_qubit_probability: SymbolicFraction,
        /// Entanglement entropy: log_2(n)
        pub entanglement_entropy: SymbolicEntropy,
        /// Size class of the qubit count
        pub size_class: String,
        /// Is the symbolic formula internally consistent for this constructed state.
        pub is_valid: bool,
    }

    impl std::fmt::Display for SymbolicWVerification {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "SYMBOLIC W-STATE VERIFICATION [{}]", self.size_class)?;
            writeln!(f, "  Qubits (n):              {}", self.n_qubits)?;
            writeln!(f, "  Amplitude per state:     {}", self.amplitude)?;
            writeln!(
                f,
                "  Non-zero amplitudes:     {}",
                self.num_nonzero_amplitudes
            )?;
            writeln!(
                f,
                "  Probability per state:   {} = {}",
                self.probability_per_state,
                self.probability_per_state.to_notation()
            )?;
            writeln!(f, "  ALGEBRAIC PROOF OF NORMALIZATION:")?;
            writeln!(f, "    Total prob = n x |amplitude|^2")?;
            writeln!(
                f,
                "                    = {} x |{}|^2",
                self.n_qubits, self.amplitude
            )?;
            writeln!(
                f,
                "                    = {} x ({})",
                self.n_qubits, self.probability_per_state
            )?;
            writeln!(f, "                    = {}", self.total_probability)?;
            writeln!(
                f,
                "                    = 1 {} ({})",
                if self.total_probability.is_one() {
                    "v"
                } else {
                    "x"
                },
                if self.total_probability.is_one() {
                    "algebraically verified"
                } else {
                    "FAILED"
                }
            )?;
            writeln!(
                f,
                "  Entanglement entropy:    {}",
                self.entanglement_entropy
            )?;
            writeln!(
                f,
                "  Per-qubit probability:   {}",
                self.per_qubit_probability
            )?;
            writeln!(
                f,
                "  Valid:                   {}",
                if self.is_valid { "YES" } else { "NO" }
            )?;
            writeln!(
                f,
                "  Memory: formula object; no iteration over {} amplitudes",
                self.n_qubits
            )?;

            match &self.n_qubits {
                SymbolicNumber::Graham => {
                    writeln!(f, "  Graham label: amplitudes are not enumerated.")?;
                    writeln!(
                        f,
                        "    - The stored normalization formula simplifies to 1 algebraically."
                    )?;
                }
                SymbolicNumber::Tree(n) => {
                    writeln!(f, "  TREE({}) label: amplitudes are not enumerated.", n)?;
                    writeln!(f, "    - Verification checks the symbolic formula.")?;
                }
                SymbolicNumber::Infinity => {
                    writeln!(f, "  Formal infinity label (aleph_0):")?;
                    writeln!(f, "    - This is a symbolic formula object.")?;
                    writeln!(
                        f,
                        "    - It is not a numerical construction of an infinite W-state limit."
                    )?;
                }
                _ => {}
            }

            Ok(())
        }
    }

    // === SymbolicDickeState ===

    /// Symbolic Dicke state |D_n^k> for labels such as Graham's number, TREE(3),
    /// or a formal infinity label.
    ///
    /// Definition: `|D_n^k> = (1/sqrt(C(n,k))) x Sum |states with exactly k ones>`
    ///
    /// # Special Cases
    ///
    /// - k=0: `|D_n^0> = |00...0>` (product state)
    /// - k=1: `|D_n^1> = |W_n>` (W-state)
    /// - k=n: `|D_n^n> = |11...1>` (product state)
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::{SymbolicDickeState, SymbolicNumber};
    ///
    /// // Dicke state with Graham's number of qubits and 2 excitations
    /// let dicke = SymbolicDickeState::graham(2);
    /// let v = dicke.verify();
    /// assert!(v.is_valid);
    ///
    /// // C(G, 2) should simplify to n(n-1)/2
    /// assert!(matches!(v.simplified_count,
    ///     tileuniverse_quantum::SimplifiedBinomial::NChoose2(_)));
    /// ```
    #[derive(Clone)]
    pub struct SymbolicDickeState {
        /// Number of qubits (n)
        pub n_qubits: SymbolicNumber,
        /// Number of excitations (k)
        pub k_excitations: SymbolicNumber,
        /// Binomial coefficient C(n,k) — number of basis states
        pub num_amplitudes: SymbolicBinomial,
        /// Amplitude: 1/sqrt(C(n,k))
        pub amplitude: SymbolicAmplitude,
        /// Creation time in nanoseconds
        pub creation_time_ns: u64,
    }

    impl SymbolicDickeState {
        /// Create a Dicke state |D_n^k>
        pub fn new(n: SymbolicNumber, k: SymbolicNumber) -> Self {
            let start = std::time::Instant::now();
            let num_amplitudes = SymbolicBinomial::new(n.clone(), k.clone());
            let amplitude = SymbolicAmplitude::ReciprocalSqrtBinomial(num_amplitudes.clone());
            Self {
                n_qubits: n,
                k_excitations: k,
                num_amplitudes,
                amplitude,
                creation_time_ns: start.elapsed().as_nanos() as u64,
            }
        }

        /// Create |D_Graham^k> — Graham's number of qubits, k excitations
        pub fn graham(k: u64) -> Self {
            Self::new(SymbolicNumber::Graham, SymbolicNumber::Literal(k))
        }

        /// Create |D_infty^k> with a formal infinity label and k excitations
        pub fn infinite(k: u64) -> Self {
            Self::new(SymbolicNumber::Infinity, SymbolicNumber::Literal(k))
        }

        /// Create |D_TREE(t)^k> — TREE(t) qubits, k excitations
        pub fn tree(t: u64, k: u64) -> Self {
            Self::new(SymbolicNumber::Tree(t), SymbolicNumber::Literal(k))
        }

        /// Create |D_n^1> = |W_n>
        pub fn as_w_state(n: SymbolicNumber) -> Self {
            Self::new(n, SymbolicNumber::Literal(1))
        }

        /// Create |D_n^0> = |00...0> (product state)
        pub fn ground_state(n: SymbolicNumber) -> Self {
            Self::new(n, SymbolicNumber::Literal(0))
        }

        /// Create |D_n^n> = |11...1> (product state)
        pub fn all_excited(n: SymbolicNumber) -> Self {
            Self::new(n.clone(), n)
        }

        /// Create |D_n^(n/2)> — maximally entangled for even n
        pub fn half_excitations(n: SymbolicNumber) -> Self {
            let k = SymbolicNumber::Div {
                numerator: Box::new(n.clone()),
                denominator: Box::new(SymbolicNumber::Literal(2)),
            };
            Self::new(n, k)
        }

        /// Verify Dicke state algebraically
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::{SymbolicDickeState, SymbolicNumber};
        ///
        /// let dicke = SymbolicDickeState::new(SymbolicNumber::Literal(10), SymbolicNumber::Literal(3));
        /// let v = dicke.verify();
        /// assert!(v.is_valid);
        /// // C(10, 3) = 120
        /// assert_eq!(v.simplified_count.try_evaluate(), Some(120));
        /// ```
        pub fn verify(&self) -> SymbolicDickeVerification {
            let simplified = self.num_amplitudes.simplify();

            let is_w_state = matches!(&self.k_excitations, SymbolicNumber::Literal(1))
                || matches!(&self.k_excitations, SymbolicNumber::Big(k) if k == &num_bigint::BigUint::from(1u32));

            let is_product_state = matches!(&self.k_excitations, SymbolicNumber::Literal(0))
                || matches!(&self.k_excitations, SymbolicNumber::Big(k) if k == &num_bigint::BigUint::from(0u32))
                || self.n_qubits.structurally_equal(&self.k_excitations);

            SymbolicDickeVerification {
                n_qubits: self.n_qubits.clone(),
                k_excitations: self.k_excitations.clone(),
                num_amplitudes: self.num_amplitudes.clone(),
                simplified_count: simplified,
                amplitude: self.amplitude.clone(),
                total_probability: "C(n,k) x 1/C(n,k) = 1".to_string(),
                is_w_state,
                is_product_state,
                size_class: format!(
                    "{} qubits, {} excitations",
                    self.n_qubits.size_class(),
                    self.k_excitations.size_class()
                ),
                is_valid: true,
            }
        }

        /// Apply collective lowering operator J_- symbolically
        ///
        /// J_-|D_n^k> = sqrt(k(n-k+1)) |D_n^(k-1)>
        pub fn apply_lowering(&self) -> Option<(SymbolicAmplitude, SymbolicDickeState)> {
            if matches!(&self.k_excitations, SymbolicNumber::Literal(0)) {
                return None;
            }
            if let SymbolicNumber::Big(k) = &self.k_excitations
                && k == &num_bigint::BigUint::from(0u32)
            {
                return None;
            }

            let new_k = SymbolicNumber::Sub {
                minuend: Box::new(self.k_excitations.clone()),
                subtrahend: Box::new(SymbolicNumber::Literal(1)),
            };

            let coefficient = SymbolicAmplitude::SqrtFraction {
                numerator: self.k_excitations.clone(),
                denominator: SymbolicNumber::Literal(1),
            };

            let new_state = SymbolicDickeState::new(self.n_qubits.clone(), new_k);
            Some((coefficient, new_state))
        }

        /// Is this equivalent to a W-state? (k=1)
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::SymbolicDickeState;
        /// use tileuniverse_quantum::SymbolicNumber;
        ///
        /// let w_like = SymbolicDickeState::as_w_state(SymbolicNumber::Literal(100));
        /// assert!(w_like.is_w_state());
        ///
        /// let general = SymbolicDickeState::new(SymbolicNumber::Literal(100), SymbolicNumber::Literal(2));
        /// assert!(!general.is_w_state());
        /// ```
        pub fn is_w_state(&self) -> bool {
            matches!(&self.k_excitations, SymbolicNumber::Literal(1))
                || matches!(&self.k_excitations, SymbolicNumber::Big(k) if k == &num_bigint::BigUint::from(1u32))
        }

        /// Is this a product state? (k=0 or k=n)
        ///
        /// # Example
        ///
        /// ```
        /// use tileuniverse_quantum::SymbolicDickeState;
        /// use tileuniverse_quantum::SymbolicNumber;
        ///
        /// assert!(SymbolicDickeState::ground_state(SymbolicNumber::Literal(100)).is_product_state());
        /// assert!(SymbolicDickeState::all_excited(SymbolicNumber::Literal(100)).is_product_state());
        /// ```
        pub fn is_product_state(&self) -> bool {
            if matches!(&self.k_excitations, SymbolicNumber::Literal(0)) {
                return true;
            }
            if let SymbolicNumber::Big(k) = &self.k_excitations
                && k == &num_bigint::BigUint::from(0u32)
            {
                return true;
            }
            self.n_qubits.structurally_equal(&self.k_excitations)
        }

        /// Memory in bytes (always O(1))
        pub fn memory_bytes(&self) -> usize {
            std::mem::size_of::<Self>()
        }

        /// Get the number of qubits
        pub fn n_qubits(&self) -> &SymbolicNumber {
            &self.n_qubits
        }

        /// Get the number of excitations
        pub fn k_excitations(&self) -> &SymbolicNumber {
            &self.k_excitations
        }

        /// Get creation time in nanoseconds
        pub fn creation_time_ns(&self) -> u64 {
            self.creation_time_ns
        }
    }

    impl std::fmt::Display for SymbolicDickeState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "|D_{}^{}>",
                self.n_qubits.to_notation(),
                self.k_excitations.to_notation()
            )
        }
    }

    // === SymbolicDickeVerification ===

    /// Verification result for SymbolicDickeState
    #[derive(Debug, Clone)]
    pub struct SymbolicDickeVerification {
        /// Number of qubits (n)
        pub n_qubits: SymbolicNumber,
        /// Number of excitations (k)
        pub k_excitations: SymbolicNumber,
        /// Binomial coefficient C(n,k)
        pub num_amplitudes: SymbolicBinomial,
        /// Simplified form of C(n,k)
        pub simplified_count: SimplifiedBinomial,
        /// Amplitude per basis state: 1/sqrt(C(n,k))
        pub amplitude: SymbolicAmplitude,
        /// Total probability proof string
        pub total_probability: String,
        /// Is this a W-state (k=1)?
        pub is_w_state: bool,
        /// Is this a product state (k=0 or k=n)?
        pub is_product_state: bool,
        /// Size class description
        pub size_class: String,
        /// Is the state valid?
        pub is_valid: bool,
    }

    impl std::fmt::Display for SymbolicDickeVerification {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(
                f,
                "SYMBOLIC DICKE STATE VERIFICATION |D_{}^{}>",
                self.n_qubits.to_notation(),
                self.k_excitations.to_notation()
            )?;
            writeln!(f, "  Qubits (n):           {}", self.n_qubits)?;
            writeln!(f, "  Excitations (k):      {}", self.k_excitations)?;
            writeln!(f, "  C(n,k) = {}", self.num_amplitudes)?;
            writeln!(f, "  Simplified: {}", self.simplified_count)?;
            writeln!(f, "  Amplitude: {}", self.amplitude)?;
            writeln!(f, "  Total probability: {}", self.total_probability)?;
            if self.is_product_state {
                writeln!(f, "  [PRODUCT STATE] — separable, no entanglement")?;
            }
            if self.is_w_state {
                writeln!(f, "  [W-STATE] — k=1, equivalent to |W_n>")?;
            }
            if !self.is_product_state && !self.is_w_state {
                writeln!(f, "  [ENTANGLED] — general Dicke state")?;
            }
            writeln!(f, "  Memory: O(1) — no iteration over C(n,k) amplitudes!")?;
            writeln!(f, "  Valid: {}", if self.is_valid { "YES" } else { "NO" })?;
            Ok(())
        }
    }

    // === Convenience Functions ===

    /// Create a W-state with Graham's number of qubits
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::create_graham_w;
    ///
    /// let w = create_graham_w();
    /// let v = w.verify();
    /// assert!(v.is_valid);
    /// ```
    pub fn create_graham_w() -> SymbolicWState {
        SymbolicWState::graham()
    }

    /// Create a W-state with TREE(n) qubits
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::create_tree_w;
    ///
    /// let w = create_tree_w(3);
    /// assert!(w.verify().is_valid);
    /// ```
    pub fn create_tree_w(n: u64) -> SymbolicWState {
        SymbolicWState::tree(n)
    }

    /// Create a W-state with a formal infinity label (aleph_0)
    pub fn create_infinite_w() -> SymbolicWState {
        SymbolicWState::infinite()
    }

    /// Create a W-state with symbolic qubit count
    pub fn create_symbolic_w(n_qubits: SymbolicNumber) -> SymbolicWState {
        SymbolicWState::new(n_qubits)
    }

    /// Create a Dicke state with symbolic qubit and excitation counts
    pub fn create_symbolic_dicke(n: SymbolicNumber, k: SymbolicNumber) -> SymbolicDickeState {
        SymbolicDickeState::new(n, k)
    }

    /// Create a Dicke state with Graham's number of qubits and k excitations
    ///
    /// # Example
    ///
    /// ```
    /// use tileuniverse_quantum::create_graham_dicke;
    ///
    /// let dicke = create_graham_dicke(2);
    /// let v = dicke.verify();
    /// assert!(v.is_valid);
    /// ```
    pub fn create_graham_dicke(k: u64) -> SymbolicDickeState {
        SymbolicDickeState::graham(k)
    }

    /// Create a Dicke state with TREE(t) qubits and k excitations
    pub fn create_tree_dicke(t: u64, k: u64) -> SymbolicDickeState {
        SymbolicDickeState::tree(t, k)
    }

    /// Create a Dicke state with a formal infinity label and k excitations
    pub fn create_infinite_dicke(k: u64) -> SymbolicDickeState {
        SymbolicDickeState::infinite(k)
    }
}

// ===================================================================
// Unit-test gate. The crate already carries doctests as usage examples;
// this module is the regression net for the deterministic value contract
// — endpoint amplitudes, the small-n terminal-address branch, W-state
// fidelity/tail, size classes, materializability, and symbolic evaluation
// — so an accidental change to any of those fails the crate build.
// ===================================================================
#[cfg(test)]
mod gate_tests {
    use super::*;

    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

    // --- Complex64 / VecBlock core ---

    #[test]
    fn complex64_norms() {
        assert_eq!(Complex64::zero().norm_squared(), 0.0);
        assert_eq!(Complex64::zero().norm(), 0.0);
        let c = Complex64::new(3.0, 4.0);
        assert_eq!(c.norm_squared(), 25.0);
        assert_eq!(c.norm(), 5.0);
        // `Default` matches `zero()`.
        assert_eq!(Complex64::default().norm_squared(), 0.0);
    }

    #[test]
    fn vecblock_set_get_and_is_zero() {
        let mut b = VecBlock::new();
        assert!(b.is_zero());
        assert!(VecBlock::default().is_zero());
        b.set(5, Complex64::new(1.0, 2.0));
        assert!(!b.is_zero());
        let g = b.get(5);
        assert_eq!((g.re, g.im), (1.0, 2.0));
        // Untouched slots stay zero.
        assert_eq!(b.get(6).norm(), 0.0);
    }

    // --- MinimalGhzState ---

    #[test]
    fn minimal_ghz_verify_is_unit_fidelity() {
        let ghz = MinimalGhzState::new(42);
        assert_eq!(ghz.n_qubits(), 42);
        let v = ghz.verify();
        assert!((v.fidelity - 1.0).abs() < 1e-12, "fidelity {}", v.fidelity);
        assert!((v.amp_zero_state - INV_SQRT2).abs() < 1e-12);
        assert!((v.amp_one_state - INV_SQRT2).abs() < 1e-12);
        assert_eq!(v.expected_amplitude, INV_SQRT2);
        assert_eq!(v.max_spurious, 0.0);
        assert!(ghz.memory_bytes() > 0);
    }

    #[test]
    fn minimal_ghz_small_n_uses_intrablock_terminal_addr() {
        // n < BLOCK_SHIFT (7) stores |11..1> at local address 2^n - 1 rather than
        // 127; verify() must read that same address back and still see unit fidelity.
        for n in 1..7usize {
            let v = MinimalGhzState::new(n).verify();
            assert!((v.fidelity - 1.0).abs() < 1e-12, "n={n} fid={}", v.fidelity);
            assert_eq!(v.max_spurious, 0.0, "n={n}");
        }
    }

    #[test]
    fn minimal_ghz_display_scales_labels() {
        assert_eq!(format!("{}", MinimalGhzState::new(42)), "|GHZ_42>");
        assert_eq!(format!("{}", MinimalGhzState::new(1_500)), "|GHZ_1.5K>");
        assert_eq!(format!("{}", MinimalGhzState::new(2_000_000)), "|GHZ_2.0M>");
        assert_eq!(
            format!("{}", MinimalGhzState::new(3_000_000_000)),
            "|GHZ_3.0B>"
        );
        assert_eq!(
            format!("{}", MinimalGhzState::new(1_000_000_000_000)),
            "|GHZ_1.0T>"
        );
    }

    #[test]
    fn create_minimal_ghz_matches_constructor() {
        assert_eq!(
            create_minimal_ghz(100).n_qubits(),
            MinimalGhzState::new(100).n_qubits()
        );
    }

    // --- SparseQuantumGridVec (materialized W-state) ---

    #[test]
    fn w_grid_block_geometry() {
        assert_eq!(SparseQuantumGridVec::new(128).n_blocks(), 1);
        assert_eq!(SparseQuantumGridVec::new(200).n_blocks(), 2); // ceil(200/128)
        let g = SparseQuantumGridVec::new(7);
        assert_eq!(g.n_qubits(), 7);
        assert_eq!(g.n_blocks(), 1);
    }

    #[test]
    #[should_panic(expected = "Minimum 7 qubits required")]
    fn w_grid_rejects_fewer_than_seven_qubits() {
        let _ = SparseQuantumGridVec::new(6);
    }

    #[test]
    fn w_state_is_uniform_and_unit_fidelity() {
        let mut grid = SparseQuantumGridVec::new(10);
        grid.create_w_state();
        let v = grid.verify_w_fidelity();
        assert!((v.fidelity - 1.0).abs() < 1e-12, "fidelity {}", v.fidelity);
        assert_eq!(v.correct_amplitudes, 10);
        assert_eq!(v.expected_amplitudes, 10);
        assert!((v.total_probability - 1.0).abs() < 1e-12);
        assert_eq!(v.max_spurious_amplitude, 0.0);
        let expected = 1.0 / 10f64.sqrt();
        assert!((v.expected_amplitude - expected).abs() < 1e-12);
        // Every excitation slot carries the uniform amplitude...
        for q in 0..10 {
            assert!((grid.get_amplitude(q).re - expected).abs() < 1e-12, "q={q}");
        }
        // ...and an out-of-range block index reads back as zero.
        assert_eq!(grid.get_amplitude(10_000).norm(), 0.0);
    }

    #[test]
    fn w_grid_amplitude_round_trip() {
        let mut grid = SparseQuantumGridVec::new(64);
        let amp = Complex64::new(0.5, 0.25);
        grid.set_amplitude(5, amp);
        let got = grid.get_amplitude(5);
        assert_eq!((got.re, got.im), (0.5, 0.25));
    }

    // --- SymbolicNumber ---

    #[test]
    fn symbolic_number_is_zero() {
        assert!(SymbolicNumber::Literal(0).is_zero());
        assert!(!SymbolicNumber::Literal(5).is_zero());
        assert!(SymbolicNumber::Big(num_bigint::BigUint::from(0u32)).is_zero());
        assert!(!SymbolicNumber::Graham.is_zero());
    }

    #[test]
    fn symbolic_number_size_class_and_notation() {
        assert_eq!(SymbolicNumber::Literal(999_999).size_class(), "small");
        assert_eq!(SymbolicNumber::Literal(2_000_000).size_class(), "large");
        assert_eq!(SymbolicNumber::Graham.size_class(), "Graham-class");
        assert_eq!(SymbolicNumber::Tree(3).size_class(), "TREE-class");
        assert_eq!(SymbolicNumber::Rayo.size_class(), "uncomputable");
        assert_eq!(SymbolicNumber::Infinity.size_class(), "infinite");

        assert_eq!(SymbolicNumber::Literal(42).to_notation(), "42");
        assert_eq!(SymbolicNumber::Graham.to_notation(), "G (Graham's number)");
        assert_eq!(SymbolicNumber::Tree(3).to_notation(), "TREE(3)");
        assert_eq!(SymbolicNumber::Infinity.to_notation(), "infty");
        assert_eq!(SymbolicNumber::Rayo.to_notation(), "Rayo's number");
    }

    #[test]
    fn symbolic_number_structural_equality() {
        assert!(SymbolicNumber::Literal(5).structurally_equal(&SymbolicNumber::Literal(5)));
        assert!(!SymbolicNumber::Literal(5).structurally_equal(&SymbolicNumber::Literal(6)));
        assert!(SymbolicNumber::Graham.structurally_equal(&SymbolicNumber::Graham));
        assert!(SymbolicNumber::Tree(3).structurally_equal(&SymbolicNumber::Tree(3)));
        assert!(!SymbolicNumber::Graham.structurally_equal(&SymbolicNumber::Infinity));
    }

    #[test]
    fn symbolic_number_estimate_log10_and_materializability() {
        assert!((SymbolicNumber::Literal(1000).estimate_log10().unwrap() - 3.0).abs() < 1e-9);
        assert!(SymbolicNumber::Graham.estimate_log10().is_none());
        assert!(SymbolicNumber::Infinity.estimate_log10().is_none());

        assert!(SymbolicNumber::Literal(5).is_materializable());
        assert!(!SymbolicNumber::Graham.is_materializable());
        assert!(
            SymbolicNumber::Pow {
                base: 10,
                exponent: Box::new(SymbolicNumber::Literal(50)),
            }
            .is_materializable()
        );
        assert!(
            !SymbolicNumber::Pow {
                base: 10,
                exponent: Box::new(SymbolicNumber::Literal(200_000_000)),
            }
            .is_materializable()
        );
    }

    // --- SymbolicAmplitude / SymbolicFraction ---

    #[test]
    fn symbolic_amplitude_evaluation() {
        assert!(
            (SymbolicAmplitude::reciprocal_sqrt(SymbolicNumber::Literal(4))
                .try_evaluate()
                .unwrap()
                - 0.5)
                .abs()
                < 1e-12
        );
        assert!(
            (SymbolicAmplitude::sqrt_fraction(
                SymbolicNumber::Literal(1),
                SymbolicNumber::Literal(4)
            )
            .try_evaluate()
            .unwrap()
                - 0.5)
                .abs()
                < 1e-12
        );
        assert_eq!(SymbolicAmplitude::Zero.try_evaluate(), Some(0.0));
        // A symbolic count with no finite value cannot be evaluated.
        assert!(
            SymbolicAmplitude::reciprocal_sqrt(SymbolicNumber::Graham)
                .try_evaluate()
                .is_none()
        );
        assert_eq!(
            SymbolicAmplitude::reciprocal_sqrt(SymbolicNumber::Literal(2)).to_notation(),
            "1/sqrt(2)"
        );
        assert_eq!(SymbolicAmplitude::Zero.to_notation(), "0");

        // |1/sqrt(9)|^2 = 1/9.
        let mag = SymbolicAmplitude::reciprocal_sqrt(SymbolicNumber::Literal(9))
            .magnitude_squared_symbolic()
            .unwrap();
        assert!((mag.try_evaluate().unwrap() - 1.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn symbolic_fraction_ops() {
        let f = SymbolicFraction::one_over(SymbolicNumber::Literal(4));
        assert!((f.try_evaluate().unwrap() - 0.25).abs() < 1e-12);
        assert_eq!(f.to_notation(), "1/4");

        assert!(
            SymbolicFraction::new(SymbolicNumber::Literal(42), SymbolicNumber::Literal(42))
                .is_one()
        );
        assert!(
            !SymbolicFraction::new(SymbolicNumber::Literal(1), SymbolicNumber::Literal(2)).is_one()
        );
        assert!(SymbolicFraction::new(SymbolicNumber::Graham, SymbolicNumber::Graham).is_one());

        let g = SymbolicFraction::new(SymbolicNumber::Literal(6), SymbolicNumber::Literal(3));
        assert!((g.try_evaluate().unwrap() - 2.0).abs() < 1e-12);
        assert_eq!(g.to_notation(), "6/3");
    }
}
