// EPIC 49: Minimal scalar quantum core (deterministic)

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub const ZERO: Complex32 = Complex32 { re: 0.0, im: 0.0 };
    pub const ONE: Complex32 = Complex32 { re: 1.0, im: 0.0 };
    pub const I: Complex32 = Complex32 { re: 0.0, im: 1.0 };

    #[inline]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    #[inline]
    pub fn from_polar(r: f32, theta: f32) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }
    #[inline]
    pub fn add(self, other: Complex32) -> Complex32 {
        Complex32::new(self.re + other.re, self.im + other.im)
    }
    #[inline]
    pub fn sub(self, other: Complex32) -> Complex32 {
        Complex32::new(self.re - other.re, self.im - other.im)
    }
    #[inline]
    pub fn mul(self, other: Complex32) -> Complex32 {
        Complex32::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
    #[inline]
    pub fn div(self, other: Complex32) -> Complex32 {
        let denom = other.norm2();
        Complex32::new(
            (self.re * other.re + self.im * other.im) / denom,
            (self.im * other.re - self.re * other.im) / denom,
        )
    }
    #[inline]
    pub fn scale(self, k: f32) -> Complex32 {
        Complex32::new(self.re * k, self.im * k)
    }
    #[inline]
    pub fn norm2(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
    #[inline]
    pub fn norm(self) -> f32 {
        self.norm2().sqrt()
    }
    #[inline]
    pub fn conj(self) -> Complex32 {
        Complex32::new(self.re, -self.im)
    }
    #[inline]
    pub fn neg(self) -> Complex32 {
        Complex32::new(-self.re, -self.im)
    }
    #[inline]
    pub fn exp(self) -> Complex32 {
        let r = self.re.exp();
        Complex32::new(r * self.im.cos(), r * self.im.sin())
    }
    #[inline]
    pub fn arg(self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Complex square root using polar form: sqrt(r*e^{i*theta}) = sqrt(r)*e^{i*theta/2}
    #[inline]
    pub fn sqrt_c(self) -> Complex32 {
        let r = self.norm();
        let theta = self.arg();
        Complex32::from_polar(r.sqrt(), theta / 2.0)
    }

    /// Complex reciprocal: 1/z = conj(z) / |z|^2
    #[inline]
    pub fn recip(self) -> Complex32 {
        let n2 = self.norm2();
        Complex32::new(self.re / n2, -self.im / n2)
    }
}

impl std::ops::Add for Complex32 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        self.add(other)
    }
}

impl std::ops::Sub for Complex32 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        self.sub(other)
    }
}

impl std::ops::Mul for Complex32 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        self.mul(other)
    }
}

use std::alloc::{alloc, dealloc, Layout};
use std::fmt;
use std::mem;
use std::ptr::{self, NonNull};
use std::slice;

/// EPIC 62Q: Guard words for canary-based buffer overrun detection
const GUARD_WORDS: usize = 8;
const GUARD_PATTERN: u32 = 0xDEADBEEF;

pub struct AlignedVecF32 {
    ptr: NonNull<f32>, // Points to FIRST guard word in strict mode, user data otherwise
    len: usize,        // User-visible length (excludes guards)
    cap: usize,        // Total capacity including guards in strict mode
    strict_mode: bool, // Whether canary guards are active
}

impl AlignedVecF32 {
    pub fn with_len_zeroed(len: usize) -> Self {
        // EPIC 62Q: Enable strict mode canaries if JIT_DEBUG_STRICT=1
        let strict_mode = std::env::var("JIT_DEBUG_STRICT").ok().as_deref() == Some("1");

        // 64-byte alignment for SIMD/cache friendliness
        let elem = mem::size_of::<f32>();

        // EPIC 62Q: Allocate extra space for guard words in strict mode
        let total_words = if strict_mode {
            len + 2 * GUARD_WORDS // left guards + data + right guards
        } else {
            len
        };

        let bytes = total_words.checked_mul(elem).expect("len overflow");
        let layout = Layout::from_size_align(bytes.max(1), 64).expect("layout");
        let raw = unsafe { alloc(layout) };
        let ptr_u8 = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        // Zero memory
        unsafe {
            ptr::write_bytes(ptr_u8.as_ptr(), 0u8, bytes);
        }
        let ptr_f32 = NonNull::new(ptr_u8.as_ptr() as *mut f32).unwrap();

        // EPIC 62Q: Initialize guard words with sentinel pattern
        if strict_mode {
            unsafe {
                let ptr = ptr_f32.as_ptr();
                // Left guards: words [0..GUARD_WORDS)
                for i in 0..GUARD_WORDS {
                    ptr::write(ptr.add(i) as *mut u32, GUARD_PATTERN);
                }
                // Right guards: words [GUARD_WORDS + len .. GUARD_WORDS + len + GUARD_WORDS)
                for i in 0..GUARD_WORDS {
                    ptr::write(ptr.add(GUARD_WORDS + len + i) as *mut u32, GUARD_PATTERN);
                }
            }
        }

        Self {
            ptr: ptr_f32,
            len,
            cap: total_words,
            strict_mode,
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        unsafe {
            let data_ptr = if self.strict_mode {
                self.ptr.as_ptr().add(GUARD_WORDS) // Skip left guards
            } else {
                self.ptr.as_ptr()
            };
            slice::from_raw_parts(data_ptr, self.len)
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe {
            let data_ptr = if self.strict_mode {
                self.ptr.as_ptr().add(GUARD_WORDS) // Skip left guards
            } else {
                self.ptr.as_ptr()
            };
            slice::from_raw_parts_mut(data_ptr, self.len)
        }
    }

    /// EPIC 62Q: Check that guard words are intact (strict mode only)
    pub fn check_guards(&self, label: &str) {
        if !self.strict_mode {
            return;
        }

        unsafe {
            let ptr = self.ptr.as_ptr();

            // Check left guards
            for i in 0..GUARD_WORDS {
                let guard_value = ptr::read(ptr.add(i) as *const u32);
                if guard_value != GUARD_PATTERN {
                    panic!(
                        "EPIC 62Q: Left guard corruption detected at {}!\n\
                         Position: word {}/{}, Expected: 0x{:08X}, Found: 0x{:08X}",
                        label, i, GUARD_WORDS, GUARD_PATTERN, guard_value
                    );
                }
            }

            // Check right guards
            for i in 0..GUARD_WORDS {
                let guard_value = ptr::read(ptr.add(GUARD_WORDS + self.len + i) as *const u32);
                if guard_value != GUARD_PATTERN {
                    panic!(
                        "EPIC 62Q: Right guard corruption detected at {}!\n\
                         Position: word {}/{}, Expected: 0x{:08X}, Found: 0x{:08X}",
                        label, i, GUARD_WORDS, GUARD_PATTERN, guard_value
                    );
                }
            }
        }
    }
}

impl Default for AlignedVecF32 {
    fn default() -> Self {
        Self::with_len_zeroed(0)
    }
}

impl Clone for AlignedVecF32 {
    fn clone(&self) -> Self {
        // EPIC 62Q: Check guards before cloning
        self.check_guards("clone");

        let mut out = AlignedVecF32::with_len_zeroed(self.len);
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}

impl Drop for AlignedVecF32 {
    fn drop(&mut self) {
        // EPIC 62Q: Check guards before dropping
        self.check_guards("drop");

        let elem = mem::size_of::<f32>();
        let bytes = self.cap * elem;
        let layout = Layout::from_size_align(bytes.max(1), 64).expect("layout");
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

// SAFETY: AlignedVecF32 owns its heap allocation exclusively (allocated via
// alloc(), deallocated in Drop via dealloc()). No shared references (no Rc,
// no RefCell). Equivalent to Vec<f32> with manual allocation — safe to move
// between threads.
unsafe impl Send for AlignedVecF32 {}

impl fmt::Debug for AlignedVecF32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedVecF32")
            .field("len", &self.len)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct QState {
    pub n_qubits: u8,
    pub len: usize,
    pub real: AlignedVecF32,
    pub imag: AlignedVecF32,
    pub tile_count: u16, // EPIC 61/62: 1=scalar, 4=1 worker SIMD, up to 256=64 workers
}

impl QState {
    pub fn new_zero(n_qubits: u8) -> Self {
        // EPIC 114: Raised limit from 30 to 32 qubits for RTX 5090 (32GB VRAM)
        // 32 qubits = 2^32 = 4B amplitudes = 32GB (complex FP32)
        assert!(n_qubits > 0 && n_qubits <= 32, "n_qubits must be 1-32");

        // EPIC 114: Warn about high memory usage
        if n_qubits >= 30 {
            let mem_gb = (1u64 << n_qubits) * 8 / (1024 * 1024 * 1024);
            eprintln!(
                "Warning: {}-qubit dense state requires {}GB memory",
                n_qubits, mem_gb
            );
        }

        let len = 1usize << n_qubits;
        let mut real = AlignedVecF32::with_len_zeroed(len);
        let imag = AlignedVecF32::with_len_zeroed(len);
        real.as_mut_slice()[0] = 1.0;
        Self {
            n_qubits,
            len,
            real,
            imag,
            tile_count: 1,
        } // EPIC 61: default to single tile
    }

    // EPIC 61/62: Multi-tile constructor with interleaved SIMD layout
    //
    // Supports:
    // - EPIC 61: 4 tiles (single-core SIMD)
    // - EPIC 62: 8-256 tiles (multi-core: N workers × 4 tiles/worker)
    //
    // Examples:
    // - tile_count=4:   1 worker  × 4 tiles = 4 tiles
    // - tile_count=8:   2 workers × 4 tiles = 8 tiles
    // - tile_count=32:  8 workers × 4 tiles = 32 tiles
    // - tile_count=256: 64 workers × 4 tiles = 256 tiles
    pub fn new_zero_multitile(n_qubits: u8, tile_count: u16) -> Self {
        // EPIC 114: Raised limit from 30 to 32 qubits for RTX 5090 (32GB VRAM)
        assert!(n_qubits > 0 && n_qubits <= 32, "n_qubits must be 1-32");
        assert!((1..=256).contains(&tile_count), "tile_count must be 1-256");

        // EPIC 114: Warn about high memory usage
        if n_qubits >= 30 {
            let mem_gb = (1u64 << n_qubits) * 8 * (tile_count as u64) / (1024 * 1024 * 1024);
            eprintln!(
                "Warning: {}-qubit × {} tiles requires {}GB memory",
                n_qubits, tile_count, mem_gb
            );
        }

        // EPIC 62: tile_count should be 4×N for multi-worker (but allow flexibility for testing)
        // Common values: 1 (scalar), 4 (1 worker), 8 (2 workers), 16 (4 workers), 32 (8 workers), etc.

        if tile_count == 1 {
            return Self::new_zero(n_qubits);
        }

        // Multi-tile: interleaved layout for SIMD
        // Memory: [T0[0], T1[0], T2[0], T3[0], T0[1], T1[1], T2[1], T3[1], ...]
        // For multi-worker: [W0_T0[0], W0_T1[0], W0_T2[0], W0_T3[0], W1_T0[0], W1_T1[0], ...]
        let amps_per_tile = 1usize << n_qubits;
        let total_len = amps_per_tile * (tile_count as usize);

        let mut real = AlignedVecF32::with_len_zeroed(total_len);
        let imag = AlignedVecF32::with_len_zeroed(total_len);

        // Initialize all tiles to |0⟩ state: amplitude[0] = 1.0 + 0.0i
        // In interleaved layout, first `tile_count` floats are [T0[0], T1[0], ..., TN[0]]
        for tile_idx in 0..(tile_count as usize) {
            real.as_mut_slice()[tile_idx] = 1.0;
        }

        Self {
            n_qubits,
            len: amps_per_tile,
            real,
            imag,
            tile_count,
        }
    }

    pub fn from_interleaved(n_qubits: u8, amps: &[Complex32]) -> Self {
        let len = 1usize << n_qubits;
        assert_eq!(amps.len(), len);
        let mut real = AlignedVecF32::with_len_zeroed(len);
        let mut imag = AlignedVecF32::with_len_zeroed(len);
        for i in 0..len {
            real.as_mut_slice()[i] = amps[i].re;
            imag.as_mut_slice()[i] = amps[i].im;
        }
        Self {
            n_qubits,
            len,
            real,
            imag,
            tile_count: 1,
        } // EPIC 61: default to single tile
    }

    pub fn to_interleaved(&self) -> Vec<Complex32> {
        let mut v = vec![Complex32::default(); self.len];
        for i in 0..self.len {
            v[i] = Complex32::new(self.real.as_slice()[i], self.imag.as_slice()[i]);
        }
        v
    }

    pub fn iter_pairs_for_qubit(&self, target: u8) -> Vec<(usize, usize)> {
        let bit = 1usize << (target as usize);
        let mut pairs = Vec::with_capacity(self.len);
        let step = bit << 1;
        let n = self.len;
        let mut i = 0usize;
        while i < n {
            for j in 0..bit {
                pairs.push((i + j, i + j + bit));
            }
            i += step;
        }
        pairs
    }

    // EPIC 62: Multi-worker tile distribution helpers

    /// Number of tiles per worker (always 4 for F32X4 SIMD)
    pub fn tiles_per_worker() -> usize {
        4 // F32X4 SIMD width
    }

    /// Number of workers this state supports
    ///
    /// Examples:
    /// - tile_count=1 → 0 workers (scalar fallback, no multi-core)
    /// - tile_count=4 → 1 worker
    /// - tile_count=8 → 2 workers
    /// - tile_count=32 → 8 workers
    /// - tile_count=256 → 64 workers
    pub fn worker_count(&self) -> usize {
        if self.tile_count == 1 {
            0 // Scalar mode, no workers
        } else {
            (self.tile_count as usize) / Self::tiles_per_worker()
        }
    }

    /// Get the tile range for a specific worker
    ///
    /// Each worker processes `tiles_per_worker` (4) consecutive tiles.
    ///
    /// # Panics
    /// Panics if worker_idx >= worker_count()
    pub fn worker_tile_range(&self, worker_idx: usize) -> (usize, usize) {
        let num_workers = self.worker_count();
        assert!(
            worker_idx < num_workers,
            "worker_idx {} >= worker_count {}",
            worker_idx,
            num_workers
        );

        let tiles_per = Self::tiles_per_worker();
        let start_tile = worker_idx * tiles_per;
        let end_tile = start_tile + tiles_per;

        (start_tile, end_tile)
    }

    /// Validate that all worker tile ranges are disjoint and cover all tiles
    ///
    /// Returns true if ranges are valid, false otherwise.
    ///
    /// Checks:
    /// - No overlapping ranges
    /// - No gaps between ranges
    /// - All tiles from 0..tile_count are covered
    pub fn validate_worker_ranges(&self) -> bool {
        if self.tile_count == 1 {
            return true; // Scalar mode, no ranges to validate
        }

        let num_workers = self.worker_count();
        let mut covered = vec![false; self.tile_count as usize];

        for worker_idx in 0..num_workers {
            let (start, end) = self.worker_tile_range(worker_idx);

            // Check bounds
            if end > self.tile_count as usize {
                return false;
            }

            // Check for overlaps and mark coverage
            for tile in start..end {
                if covered[tile] {
                    return false; // Overlap detected
                }
                covered[tile] = true;
            }
        }

        // Check that all tiles are covered
        covered.iter().all(|&c| c)
    }

    /// Calculate memory offset for a specific worker's tile range
    ///
    /// Returns (real_offset, imag_offset) in bytes for the start of this worker's tiles.
    ///
    /// Used for cache-aligned access and validating disjoint memory regions.
    pub fn worker_memory_offsets(&self, worker_idx: usize) -> (usize, usize) {
        let (start_tile, _end_tile) = self.worker_tile_range(worker_idx);

        // In interleaved layout, tiles are laid out sequentially
        // Offset in floats = start_tile
        let offset_in_floats = start_tile;

        // Convert to bytes (for alignment checks)
        let offset_in_bytes = offset_in_floats * std::mem::size_of::<f32>();

        (offset_in_bytes, offset_in_bytes)
    }

    /// EPIC 62Q: Check QState invariants (Q3.2)
    ///
    /// Verifies:
    /// - real.len() == imag.len()
    /// - real.len() == tile_count * amps_per_tile
    /// - Buffer geometry matches expected layout
    pub fn check_invariants(&self, label: &str) {
        let amps_per_tile = self.len;
        let expected_total_len = (self.tile_count as usize) * amps_per_tile;

        assert_eq!(
            self.real.len(),
            self.imag.len(),
            "EPIC 62Q invariant violation at {}: real.len({}) != imag.len({})",
            label,
            self.real.len(),
            self.imag.len()
        );

        assert_eq!(
            self.real.len(),
            expected_total_len,
            "EPIC 62Q invariant violation at {}: real.len({}) != expected({})\n\
             tile_count={}, amps_per_tile={}",
            label,
            self.real.len(),
            expected_total_len,
            self.tile_count,
            amps_per_tile
        );

        assert!(
            self.tile_count >= 1 && self.tile_count <= 256,
            "EPIC 62Q invariant violation at {}: tile_count({}) out of range [1, 256]",
            label,
            self.tile_count
        );
    }

    /// EPIC 62Q: Check buffer guards (Q3.1)
    pub fn check_guards(&self, label: &str) {
        self.real.check_guards(&format!("{}.real", label));
        self.imag.check_guards(&format!("{}.imag", label));
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum QGate {
    H(u8),
    X(u8),
    Y(u8),
    Z(u8),
    Phase(u8, f32),
    CNot(u8, u8),
    Swap(u8, u8),
    Measure(u8),
    // SPRINT 2.1a: New gates for Grover's algorithm
    /// Controlled-Z: flip phase when both qubits are |1⟩
    CZ(u8, u8),
    /// Toffoli (CCX): flip target when both controls are |1⟩
    Toffoli(u8, u8, u8),
    /// Controlled-Controlled-Z: flip phase when all three qubits are |1⟩
    CCZ(u8, u8, u8),
    // Shor's algorithm: Controlled rotation gates
    /// Controlled-Phase: apply Phase(angle) to target if control is |1⟩
    CPhase(u8, u8, f32), // control, target, angle
    /// Controlled-Rz: apply Rz(angle) to target if control is |1⟩
    CRz(u8, u8, f32), // control, target, angle
    // EPIC 83 Phase 3: Rotation gates with spectral decomposition support
    /// Rotation around X axis: Rx(θ) = cos(θ/2)I - i·sin(θ/2)X
    Rx(u8, f32),
    /// Rotation around Y axis: Ry(θ) = cos(θ/2)I - i·sin(θ/2)Y
    Ry(u8, f32),
    /// Rotation around Z axis: Rz(θ) = cos(θ/2)I - i·sin(θ/2)Z
    /// Note: This is different from Phase(θ) by a global phase
    Rz(u8, f32),
    // SPRINT 32.0: General single-qubit unitary gate
    /// U3(θ,φ,λ) = Rz(φ) · Ry(θ) · Rz(λ) - the most general single-qubit gate
    /// Parameters: qubit, theta, phi, lambda
    U3(u8, f32, f32, f32),
    // SPRINT 48.0: T gates for Clifford+T synthesis
    /// T gate: [[1, 0], [0, e^(iπ/4)]] - π/8 rotation (key non-Clifford gate)
    /// T = diag(1, e^(iπ/4)) = S^(1/2) where S = Phase(π/2)
    T(u8),
    /// T† (T-dagger) gate: [[1, 0], [0, e^(-iπ/4)]] - adjoint of T
    Tdg(u8),
}

#[derive(Clone, Copy, Debug)]
pub struct QRng {
    pub state: u64,
}
impl QRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateOutcome {
    None,
    Measured { qubit: u8, bit: u8 },
}

#[allow(dead_code)]
fn hadamard_pair(a0: Complex32, a1: Complex32) -> (Complex32, Complex32) {
    // (|0> -> (|0>+|1>)/sqrt2, |1> -> (|0>-|1>)/sqrt2)
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let s0 = a0.add(a1).scale(INV_SQRT2);
    let s1 = a0.add(a1.scale(-1.0)).scale(INV_SQRT2);
    (s0, s1)
}

pub fn apply_gate_scalar(state: &mut QState, gate: &QGate, rng: &mut QRng) -> GateOutcome {
    match *gate {
        QGate::H(q) => {
            let pairs = state.iter_pairs_for_qubit(q);
            const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for (i0, i1) in pairs.into_iter() {
                let r0 = r[i0];
                let r1 = r[i1];
                let i0v = im[i0];
                let i1v = im[i1];
                let s0r = (r0 + r1) * INV_SQRT2;
                let s0i = (i0v + i1v) * INV_SQRT2;
                let s1r = (r0 - r1) * INV_SQRT2;
                let s1i = (i0v - i1v) * INV_SQRT2;
                r[i0] = s0r;
                im[i0] = s0i;
                r[i1] = s1r;
                im[i1] = s1i;
            }
            GateOutcome::None
        }
        QGate::Y(q) => {
            // Y = [0 -i; i 0]
            // |0> -> i|1>, |1> -> -i|0>
            // (a+bi)|0> + (c+di)|1> -> (a+bi)i|1> + (c+di)(-i)|0>
            // = (-d+ci)|0> + (-b+ai)|1>
            let pairs = state.iter_pairs_for_qubit(q);
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for (i0, i1) in pairs.into_iter() {
                let r0 = r[i0];
                let i0v = im[i0];
                let r1 = r[i1];
                let i1v = im[i1];

                // New |0> component comes from old |1> * -i
                // (r1 + i1v*i)(-i) = -r1*i + i1v = i1v - r1*i
                r[i0] = i1v;
                im[i0] = -r1;

                // New |1> component comes from old |0> * i
                // (r0 + i0v*i)(i) = r0*i - i0v = -i0v + r0*i
                r[i1] = -i0v;
                im[i1] = r0;
            }
            GateOutcome::None
        }
        QGate::Swap(a, b) => {
            if a == b {
                return GateOutcome::None;
            }

            let len = state.real.len();
            let mask_a = 1 << a;
            let mask_b = 1 << b;

            // Acquire mutable slices once
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();

            for i in 0..len {
                let bit_a = (i & mask_a) != 0;
                let bit_b = (i & mask_b) != 0;

                if bit_a && !bit_b {
                    let j = (i ^ mask_a) ^ mask_b;

                    if j > i {
                        r.swap(i, j);
                        im.swap(i, j);
                    }
                }
            }
            GateOutcome::None
        }
        QGate::X(q) => {
            let pairs = state.iter_pairs_for_qubit(q);
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for (i0, i1) in pairs.into_iter() {
                r.swap(i0, i1);
                im.swap(i0, i1);
            }
            GateOutcome::None
        }
        QGate::Z(q) => {
            let pairs = state.iter_pairs_for_qubit(q);
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for (_i0, i1) in pairs.into_iter() {
                r[i1] = -r[i1];
                im[i1] = -im[i1];
            }
            GateOutcome::None
        }
        QGate::Phase(q, theta) => {
            let pairs = state.iter_pairs_for_qubit(q);
            let (s, c) = theta.sin_cos();
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for (_i0, i1) in pairs.into_iter() {
                let re = r[i1];
                let imv = im[i1];
                r[i1] = re * c - imv * s;
                im[i1] = re * s + imv * c;
            }
            GateOutcome::None
        }
        QGate::CNot(ctrl, targ) => {
            let cbit = 1usize << (ctrl as usize);
            let tbit = 1usize << (targ as usize);
            // Swap amplitudes when control bit is 1 and target toggles
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for i in 0..state.len {
                if (i & cbit) != 0 {
                    let j = i ^ tbit;
                    if i < j {
                        r.swap(i, j);
                        im.swap(i, j);
                    }
                }
            }
            GateOutcome::None
        }
        // SPRINT 2.1a: CZ gate - flip phase when both qubits are |1⟩
        QGate::CZ(q0, q1) => {
            let bit0 = 1usize << (q0 as usize);
            let bit1 = 1usize << (q1 as usize);
            let mask = bit0 | bit1;
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for i in 0..state.len {
                if (i & mask) == mask {
                    r[i] = -r[i];
                    im[i] = -im[i];
                }
            }
            GateOutcome::None
        }
        // SPRINT 2.1a: Toffoli (CCX) gate - flip target when both controls are |1⟩
        QGate::Toffoli(ctrl0, ctrl1, targ) => {
            let cbit0 = 1usize << (ctrl0 as usize);
            let cbit1 = 1usize << (ctrl1 as usize);
            let tbit = 1usize << (targ as usize);
            let cmask = cbit0 | cbit1;
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for i in 0..state.len {
                if (i & cmask) == cmask {
                    let j = i ^ tbit;
                    if i < j {
                        r.swap(i, j);
                        im.swap(i, j);
                    }
                }
            }
            GateOutcome::None
        }
        // Shor's algorithm: Controlled-Phase gate
        QGate::CPhase(ctrl, targ, angle) => {
            let cbit = 1usize << (ctrl as usize);
            let tbit = 1usize << (targ as usize);
            let mask = cbit | tbit;
            let c = angle.cos();
            let s = angle.sin();
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            // Apply phase e^{i*angle} when both control and target are |1⟩
            for i in 0..state.len {
                if (i & mask) == mask {
                    let re = r[i];
                    let imv = im[i];
                    r[i] = re * c - imv * s;
                    im[i] = re * s + imv * c;
                }
            }
            GateOutcome::None
        }
        // Shor's algorithm: Controlled-Rz gate
        QGate::CRz(ctrl, targ, angle) => {
            let cbit = 1usize << (ctrl as usize);
            let tbit = 1usize << (targ as usize);
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            // Apply Rz(angle) to target qubit when control is |1⟩
            // Rz rotates |1⟩ component by e^{-i*angle/2} and |0⟩ by e^{i*angle/2}
            let c = (angle / 2.0).cos();
            let s = (angle / 2.0).sin();
            for i in 0..state.len {
                if (i & cbit) != 0 {
                    // Control is |1⟩, apply Rz to target
                    if (i & tbit) != 0 {
                        // Target is |1⟩: multiply by e^{-i*angle/2}
                        let re = r[i];
                        let imv = im[i];
                        r[i] = re * c + imv * s;
                        im[i] = -re * s + imv * c;
                    } else {
                        // Target is |0⟩: multiply by e^{i*angle/2}
                        let re = r[i];
                        let imv = im[i];
                        r[i] = re * c - imv * s;
                        im[i] = re * s + imv * c;
                    }
                }
            }
            GateOutcome::None
        }
        // SPRINT 2.1a: CCZ gate - flip phase when all three qubits are |1⟩
        QGate::CCZ(q0, q1, q2) => {
            let bit0 = 1usize << (q0 as usize);
            let bit1 = 1usize << (q1 as usize);
            let bit2 = 1usize << (q2 as usize);
            let mask = bit0 | bit1 | bit2;
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for i in 0..state.len {
                if (i & mask) == mask {
                    r[i] = -r[i];
                    im[i] = -im[i];
                }
            }
            GateOutcome::None
        }
        // EPIC 83 Phase 3: Rotation gates with spectral decomposition
        // Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
        QGate::Rx(q, theta) => {
            let pairs = state.iter_pairs_for_qubit(q);
            let half_theta = theta * 0.5;
            let (sin_half, cos_half) = half_theta.sin_cos();
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for (i0, i1) in pairs.into_iter() {
                // |0⟩ → cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩
                // |1⟩ → -i·sin(θ/2)|0⟩ + cos(θ/2)|1⟩
                let r0 = r[i0];
                let i0v = im[i0];
                let r1 = r[i1];
                let i1v = im[i1];
                // new|0⟩ = cos(θ/2)·|0⟩ - i·sin(θ/2)·|1⟩
                // (a + bi)·cos - i·(c + di)·sin = a·cos + sin·d + i(b·cos - c·sin)
                r[i0] = r0 * cos_half + i1v * sin_half;
                im[i0] = i0v * cos_half - r1 * sin_half;
                // new|1⟩ = -i·sin(θ/2)·|0⟩ + cos(θ/2)·|1⟩
                // -i·(a + bi)·sin + (c + di)·cos = c·cos + b·sin + i(d·cos - a·sin)
                r[i1] = r1 * cos_half + i0v * sin_half;
                im[i1] = i1v * cos_half - r0 * sin_half;
            }
            GateOutcome::None
        }
        // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
        QGate::Ry(q, theta) => {
            let pairs = state.iter_pairs_for_qubit(q);
            let half_theta = theta * 0.5;
            let (sin_half, cos_half) = half_theta.sin_cos();
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for (i0, i1) in pairs.into_iter() {
                // |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
                // |1⟩ → -sin(θ/2)|0⟩ + cos(θ/2)|1⟩
                let r0 = r[i0];
                let i0v = im[i0];
                let r1 = r[i1];
                let i1v = im[i1];
                r[i0] = r0 * cos_half - r1 * sin_half;
                im[i0] = i0v * cos_half - i1v * sin_half;
                r[i1] = r0 * sin_half + r1 * cos_half;
                im[i1] = i0v * sin_half + i1v * cos_half;
            }
            GateOutcome::None
        }
        // Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
        QGate::Rz(q, theta) => {
            let pairs = state.iter_pairs_for_qubit(q);
            let half_theta = theta * 0.5;
            let (sin_half, cos_half) = half_theta.sin_cos();
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for (i0, i1) in pairs.into_iter() {
                // |0⟩ → e^{-iθ/2}|0⟩ = (cos - i·sin)|0⟩
                let r0 = r[i0];
                let i0v = im[i0];
                r[i0] = r0 * cos_half + i0v * sin_half;
                im[i0] = i0v * cos_half - r0 * sin_half;
                // |1⟩ → e^{iθ/2}|1⟩ = (cos + i·sin)|1⟩
                let r1 = r[i1];
                let i1v = im[i1];
                r[i1] = r1 * cos_half - i1v * sin_half;
                im[i1] = i1v * cos_half + r1 * sin_half;
            }
            GateOutcome::None
        }
        // SPRINT 32.0: U3(θ,φ,λ) = [[cos(θ/2), -e^{iλ}·sin(θ/2)], [e^{iφ}·sin(θ/2), e^{i(φ+λ)}·cos(θ/2)]]
        QGate::U3(q, theta, phi, lambda) => {
            let pairs = state.iter_pairs_for_qubit(q);
            let half_theta = theta * 0.5;
            let (sin_h, cos_h) = half_theta.sin_cos();
            let (sin_l, cos_l) = lambda.sin_cos();
            let (sin_p, cos_p) = phi.sin_cos();
            let (sin_pl, cos_pl) = (phi + lambda).sin_cos();
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for (i0, i1) in pairs.into_iter() {
                let r0 = r[i0];
                let i0v = im[i0];
                let r1 = r[i1];
                let i1v = im[i1];
                // Matrix: [[cos_h, -e^{iλ}·sin_h], [e^{iφ}·sin_h, e^{i(φ+λ)}·cos_h]]
                // -e^{iλ}·sin_h = (-cos_l - i·sin_l)·sin_h
                // e^{iφ}·sin_h = (cos_p + i·sin_p)·sin_h
                // e^{i(φ+λ)}·cos_h = (cos_pl + i·sin_pl)·cos_h

                // new|0⟩ = cos_h·|0⟩ + (-e^{iλ}·sin_h)·|1⟩
                let a00_r = cos_h;
                let a01_r = -cos_l * sin_h;
                let a01_i = -sin_l * sin_h;
                r[i0] = a00_r * r0 + a01_r * r1 - a01_i * i1v;
                im[i0] = a00_r * i0v + a01_r * i1v + a01_i * r1;

                // new|1⟩ = (e^{iφ}·sin_h)·|0⟩ + (e^{i(φ+λ)}·cos_h)·|1⟩
                let a10_r = cos_p * sin_h;
                let a10_i = sin_p * sin_h;
                let a11_r = cos_pl * cos_h;
                let a11_i = sin_pl * cos_h;
                r[i1] = a10_r * r0 - a10_i * i0v + a11_r * r1 - a11_i * i1v;
                im[i1] = a10_r * i0v + a10_i * r0 + a11_r * i1v + a11_i * r1;
            }
            GateOutcome::None
        }
        // SPRINT 48.0: T gate = diag(1, e^(iπ/4)) = diag(1, (1+i)/√2)
        QGate::T(q) => {
            // T gate multiplies |1⟩ amplitudes by e^(iπ/4) = cos(π/4) + i·sin(π/4)
            const T_REAL: f32 = std::f32::consts::FRAC_1_SQRT_2; // cos(π/4) = 1/√2
            const T_IMAG: f32 = std::f32::consts::FRAC_1_SQRT_2; // sin(π/4) = 1/√2
            let bit = 1usize << (q as usize);
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for i in 0..state.len {
                if (i & bit) != 0 {
                    // |1⟩ basis state: multiply by e^(iπ/4)
                    let re = r[i];
                    let imv = im[i];
                    r[i] = T_REAL * re - T_IMAG * imv;
                    im[i] = T_REAL * imv + T_IMAG * re;
                }
            }
            GateOutcome::None
        }
        // SPRINT 48.0: T† gate = diag(1, e^(-iπ/4)) = diag(1, (1-i)/√2)
        QGate::Tdg(q) => {
            // T† gate multiplies |1⟩ amplitudes by e^(-iπ/4) = cos(π/4) - i·sin(π/4)
            const T_REAL: f32 = std::f32::consts::FRAC_1_SQRT_2; // cos(π/4) = 1/√2
            const T_IMAG: f32 = -std::f32::consts::FRAC_1_SQRT_2; // -sin(π/4) = -1/√2
            let bit = 1usize << (q as usize);
            let r = state.real.as_mut_slice();
            let im = state.imag.as_mut_slice();
            for i in 0..state.len {
                if (i & bit) != 0 {
                    // |1⟩ basis state: multiply by e^(-iπ/4)
                    let re = r[i];
                    let imv = im[i];
                    r[i] = T_REAL * re - T_IMAG * imv;
                    im[i] = T_REAL * imv + T_IMAG * re;
                }
            }
            GateOutcome::None
        }
        QGate::Measure(q) => {
            // Compute probability of 0 on qubit q
            let bit = 1usize << (q as usize);
            let mut p0: f32 = 0.0;
            for i in 0..state.len {
                if (i & bit) == 0 {
                    let re = state.real.as_slice()[i];
                    let imv = state.imag.as_slice()[i];
                    p0 += re * re + imv * imv;
                }
            }
            if p0 > 1.0 {
                p0 = 1.0;
            }
            let r = rng.next_f32();
            let outcome = if r < p0 { 0u8 } else { 1u8 };
            // Collapse
            let mut norm: f32 = 0.0;
            for i in 0..state.len {
                let keep0 = ((i & bit) == 0) as u8;
                let keep = if outcome == 0 { keep0 } else { 1 - keep0 };
                if keep == 1 {
                    let re = state.real.as_slice()[i];
                    let imv = state.imag.as_slice()[i];
                    norm += re * re + imv * imv;
                } else {
                    state.real.as_mut_slice()[i] = 0.0;
                    state.imag.as_mut_slice()[i] = 0.0;
                }
            }
            let inv = if norm > 0.0 { norm.sqrt().recip() } else { 0.0 };
            if inv != 0.0 {
                for i in 0..state.len {
                    state.real.as_mut_slice()[i] *= inv;
                    state.imag.as_mut_slice()[i] *= inv;
                }
            }
            GateOutcome::Measured {
                qubit: q,
                bit: outcome,
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct QTile {
    pub id: u16,
    pub tile_idx: usize,
    pub state: QState,
    pub program: Vec<QGate>,
    pub pc: usize,
    pub measured: Vec<Option<u8>>, // size = n_qubits
    pub rng: QRng,
}

// EPIC 51: Backend selection for quantum kernels
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QBackend {
    Scalar,
    Avx2,
    Avx512,
    Jit,
    /// EPIC 66: CUDA GPU backend
    #[cfg(feature = "cuda")]
    Cuda,
}

#[inline]
pub fn apply_gate_backend(
    state: &mut QState,
    gate: &QGate,
    rng: &mut QRng,
    backend: QBackend,
) -> GateOutcome {
    match backend {
        QBackend::Scalar => apply_gate_scalar(state, gate, rng),
        QBackend::Avx2 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                // Only accelerate selected gates; fall back for others or when AVX2 not available
                if is_avx2_available() {
                    match *gate {
                        QGate::H(q) => return unsafe { apply_h_avx2(state, q) },
                        QGate::X(q) => return unsafe { apply_x_avx2(state, q) },
                        QGate::Phase(q, theta) => {
                            return unsafe { apply_phase_avx2(state, q, theta) }
                        }
                        QGate::CNot(c, t) => return unsafe { apply_cnot_avx2(state, c, t) },
                        // EPIC 112.2: CZ, CCZ, Toffoli AVX2 dispatch
                        QGate::CZ(q0, q1) => return unsafe { apply_cz_avx2(state, q0, q1) },
                        QGate::CCZ(q0, q1, q2) => {
                            return unsafe { apply_ccz_avx2(state, q0, q1, q2) }
                        }
                        QGate::Toffoli(c0, c1, t) => {
                            return unsafe { apply_toffoli_avx2(state, c0, c1, t) }
                        }
                        _ => {}
                    }
                }
            }
            apply_gate_scalar(state, gate, rng)
        }
        QBackend::Avx512 => {
            // EPIC 112.1: Native AVX512 kernels for AMD 9950X3D / Zen 4
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_avx512f_available() {
                    // Use native AVX512 (16 lanes, 2x throughput)
                    match *gate {
                        QGate::H(q) => return unsafe { apply_h_avx512(state, q) },
                        QGate::X(q) => return unsafe { apply_x_avx512(state, q) },
                        QGate::Phase(q, theta) => {
                            return unsafe { apply_phase_avx512(state, q, theta) }
                        }
                        QGate::CNot(c, t) => return unsafe { apply_cnot_avx512(state, c, t) },
                        QGate::CZ(q0, q1) => return unsafe { apply_cz_avx512(state, q0, q1) },
                        QGate::CCZ(q0, q1, q2) => {
                            return unsafe { apply_ccz_avx512(state, q0, q1, q2) }
                        }
                        QGate::Toffoli(c0, c1, t) => {
                            return unsafe { apply_toffoli_avx2(state, c0, c1, t) }
                        }
                        _ => {}
                    }
                } else if is_avx2_available() {
                    // Fall back to AVX2 (8 lanes)
                    match *gate {
                        QGate::H(q) => return unsafe { apply_h_avx2(state, q) },
                        QGate::X(q) => return unsafe { apply_x_avx2(state, q) },
                        QGate::Phase(q, theta) => {
                            return unsafe { apply_phase_avx2(state, q, theta) }
                        }
                        QGate::CNot(c, t) => return unsafe { apply_cnot_avx2(state, c, t) },
                        QGate::CZ(q0, q1) => return unsafe { apply_cz_avx2(state, q0, q1) },
                        QGate::CCZ(q0, q1, q2) => {
                            return unsafe { apply_ccz_avx2(state, q0, q1, q2) }
                        }
                        QGate::Toffoli(c0, c1, t) => {
                            return unsafe { apply_toffoli_avx2(state, c0, c1, t) }
                        }
                        _ => {}
                    }
                }
            }
            apply_gate_scalar(state, gate, rng)
        }
        QBackend::Jit => {
            #[cfg(feature = "quantum_jit")]
            unsafe {
                crate::quantum_jit::run_jit_kernel(state, gate);
                return GateOutcome::None;
            }
            // Fallback without quantum_jit feature: route to AVX2 if available, else scalar
            #[cfg(not(feature = "quantum_jit"))]
            {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if is_avx2_available() {
                        match *gate {
                            QGate::H(q) => return unsafe { apply_h_avx2(state, q) },
                            QGate::X(q) => return unsafe { apply_x_avx2(state, q) },
                            QGate::Phase(q, theta) => {
                                return unsafe { apply_phase_avx2(state, q, theta) }
                            }
                            QGate::CNot(c, t) => return unsafe { apply_cnot_avx2(state, c, t) },
                            // EPIC 112.2: CZ, CCZ, Toffoli AVX2 dispatch
                            QGate::CZ(q0, q1) => return unsafe { apply_cz_avx2(state, q0, q1) },
                            QGate::CCZ(q0, q1, q2) => {
                                return unsafe { apply_ccz_avx2(state, q0, q1, q2) }
                            }
                            QGate::Toffoli(c0, c1, t) => {
                                return unsafe { apply_toffoli_avx2(state, c0, c1, t) }
                            }
                            _ => {}
                        }
                    }
                }
                apply_gate_scalar(state, gate, rng)
            }
        }
        #[cfg(feature = "cuda")]
        QBackend::Cuda => {
            // EPIC 66: CUDA GPU backend
            // Single-gate CUDA execution is not efficient due to kernel launch overhead.
            // For production use, batch gates via CudaContext::run_kernel_batch().
            // This fallback routes to scalar for correctness.
            eprintln!("Warning: CUDA backend used for single gate - falling back to scalar. Use batch API for performance.");
            apply_gate_scalar(state, gate, rng)
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub fn is_avx2_available() -> bool {
    #[cfg(target_arch = "x86")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub fn is_avx512f_available() -> bool {
    #[cfg(target_arch = "x86")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
}

/// Apply a Hadamard gate to `target` with an AVX2 kernel.
///
/// # Safety
///
/// The CPU must support AVX2 (`#[target_feature(enable = "avx2")]`): call only
/// when `is_x86_feature_detected!("avx2")` is true, otherwise the AVX2
/// instructions are undefined behavior. `target` must be a valid qubit index
/// for `state` (`1 << target < state.len`) so the strides stay in bounds of
/// `state.real`/`state.imag`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn apply_h_avx2(state: &mut QState, target: u8) -> GateOutcome {
    use core::arch::x86_64::*;
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let bit = 1usize << (target as usize);
    let step = bit << 1;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();
    let vscale = _mm256_set1_ps(INV_SQRT2);
    let mut i = 0usize;
    while i < n {
        let mut j = 0usize;
        while j < bit {
            // Process 8 lanes if available; else tail scalar for the remainder of this block
            if j + 8 <= bit {
                let i0 = i + j;
                let i1 = i + j + bit;
                let vr0 = _mm256_loadu_ps(r.as_ptr().add(i0));
                let vr1 = _mm256_loadu_ps(r.as_ptr().add(i1));
                let vi0 = _mm256_loadu_ps(im.as_ptr().add(i0));
                let vi1 = _mm256_loadu_ps(im.as_ptr().add(i1));
                // s0 = (a0 + a1) * invsqrt2; s1 = (a0 - a1) * invsqrt2
                let vr_add = _mm256_add_ps(vr0, vr1);
                let vr_sub = _mm256_sub_ps(vr0, vr1);
                let vi_add = _mm256_add_ps(vi0, vi1);
                let vi_sub = _mm256_sub_ps(vi0, vi1);
                let vr_s0 = _mm256_mul_ps(vr_add, vscale);
                let vr_s1 = _mm256_mul_ps(vr_sub, vscale);
                let vi_s0 = _mm256_mul_ps(vi_add, vscale);
                let vi_s1 = _mm256_mul_ps(vi_sub, vscale);
                _mm256_storeu_ps(r.as_mut_ptr().add(i0), vr_s0);
                _mm256_storeu_ps(r.as_mut_ptr().add(i1), vr_s1);
                _mm256_storeu_ps(im.as_mut_ptr().add(i0), vi_s0);
                _mm256_storeu_ps(im.as_mut_ptr().add(i1), vi_s1);
                j += 8;
            } else {
                // Tail: scalar per-lane within this block
                let i0 = i + j;
                let i1 = i + j + bit;
                let r0 = r[i0];
                let r1 = r[i1];
                let i0v = im[i0];
                let i1v = im[i1];
                r[i0] = (r0 + r1) * INV_SQRT2;
                im[i0] = (i0v + i1v) * INV_SQRT2;
                r[i1] = (r0 - r1) * INV_SQRT2;
                im[i1] = (i0v - i1v) * INV_SQRT2;
                j += 1;
            }
        }
        i += step;
    }
    GateOutcome::None
}

/// Apply a Pauli-X gate to `target` with an AVX2 kernel.
///
/// # Safety
///
/// The CPU must support AVX2 (`#[target_feature(enable = "avx2")]`): call only
/// when `is_x86_feature_detected!("avx2")` is true, otherwise the AVX2
/// instructions are undefined behavior. `target` must be a valid qubit index
/// for `state` (`1 << target < state.len`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn apply_x_avx2(state: &mut QState, target: u8) -> GateOutcome {
    use core::arch::x86_64::*;
    let bit = 1usize << (target as usize);
    let step = bit << 1;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();
    let mut i = 0usize;
    while i < n {
        let mut j = 0usize;
        while j < bit {
            if j + 8 <= bit {
                let i0 = i + j;
                let i1 = i + j + bit;
                let vr0 = _mm256_loadu_ps(r.as_ptr().add(i0));
                let vr1 = _mm256_loadu_ps(r.as_ptr().add(i1));
                let vi0 = _mm256_loadu_ps(im.as_ptr().add(i0));
                let vi1 = _mm256_loadu_ps(im.as_ptr().add(i1));
                _mm256_storeu_ps(r.as_mut_ptr().add(i0), vr1);
                _mm256_storeu_ps(r.as_mut_ptr().add(i1), vr0);
                _mm256_storeu_ps(im.as_mut_ptr().add(i0), vi1);
                _mm256_storeu_ps(im.as_mut_ptr().add(i1), vi0);
                j += 8;
            } else {
                let i0 = i + j;
                let i1 = i + j + bit;
                r.swap(i0, i1);
                im.swap(i0, i1);
                j += 1;
            }
        }
        i += step;
    }
    GateOutcome::None
}

/// Apply a phase rotation of `theta` to `target` with an AVX2 kernel.
///
/// # Safety
///
/// The CPU must support AVX2 (`#[target_feature(enable = "avx2")]`): call only
/// when `is_x86_feature_detected!("avx2")` is true, otherwise the AVX2
/// instructions are undefined behavior. `target` must be a valid qubit index
/// for `state` (`1 << target < state.len`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn apply_phase_avx2(state: &mut QState, target: u8, theta: f32) -> GateOutcome {
    use core::arch::x86_64::*;
    let bit = 1usize << (target as usize);
    let step = bit << 1;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();
    let (s, c) = theta.sin_cos();
    let vs = _mm256_set1_ps(s);
    let vc = _mm256_set1_ps(c);
    let mut i = 0usize;
    while i < n {
        let mut j = 0usize;
        while j < bit {
            if j + 8 <= bit {
                let i1 = i + j + bit;
                let vr1 = _mm256_loadu_ps(r.as_ptr().add(i1));
                let vi1 = _mm256_loadu_ps(im.as_ptr().add(i1));
                // re' = re*c - im*s; im' = re*s + im*c
                let re_c = _mm256_mul_ps(vr1, vc);
                let im_s = _mm256_mul_ps(vi1, vs);
                let re_p = _mm256_sub_ps(re_c, im_s);
                let re_s = _mm256_mul_ps(vr1, vs);
                let im_c = _mm256_mul_ps(vi1, vc);
                let im_p = _mm256_add_ps(re_s, im_c);
                _mm256_storeu_ps(r.as_mut_ptr().add(i1), re_p);
                _mm256_storeu_ps(im.as_mut_ptr().add(i1), im_p);
                j += 8;
            } else {
                let i1 = i + j + bit;
                let re = r[i1];
                let iv = im[i1];
                r[i1] = re * c - iv * s;
                im[i1] = re * s + iv * c;
                j += 1;
            }
        }
        i += step;
    }
    GateOutcome::None
}

/// Apply a CNOT gate (`control` -> `target`) with an AVX2 kernel.
///
/// # Safety
///
/// The CPU must support AVX2 (`#[target_feature(enable = "avx2")]`): call only
/// when `is_x86_feature_detected!("avx2")` is true, otherwise the AVX2
/// instructions are undefined behavior. `control` and `target` must be
/// distinct, valid qubit indices for `state` (`1 << control < state.len` and
/// `1 << target < state.len`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn apply_cnot_avx2(state: &mut QState, control: u8, target: u8) -> GateOutcome {
    use core::arch::x86_64::*;
    let cbit = 1usize << (control as usize);
    let bit = 1usize << (target as usize);
    let step = bit << 1;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();

    // Walk blocks of size 2*bit; lower half has target bit = 0, upper = 1.
    let mut base = 0usize;
    while base < n {
        let mut j = 0usize;
        while j < bit {
            if j + 8 <= bit {
                let low = base + j;
                let up = low + bit;
                // Load real/imag for lower and upper halves
                let vr_lo = _mm256_loadu_ps(r.as_ptr().add(low));
                let vr_up = _mm256_loadu_ps(r.as_ptr().add(up));
                let vi_lo = _mm256_loadu_ps(im.as_ptr().add(low));
                let vi_up = _mm256_loadu_ps(im.as_ptr().add(up));
                // Build mask: swap when control bit is set in the lower index
                // Index lanes = (low + [0..7])
                let idx = _mm256_setr_epi32(
                    low as i32,
                    (low + 1) as i32,
                    (low + 2) as i32,
                    (low + 3) as i32,
                    (low + 4) as i32,
                    (low + 5) as i32,
                    (low + 6) as i32,
                    (low + 7) as i32,
                );
                // Mask lanes where (idx & cbit) != 0
                let c_mask = _mm256_set1_epi32(cbit as i32);
                let idx_and = _mm256_and_si256(idx, c_mask);
                let zero = _mm256_setzero_si256();
                let neq = _mm256_cmpgt_epi32(idx_and, zero); // >0 => control bit set
                let m = _mm256_castsi256_ps(neq);
                // Blend swap based on mask
                let vr_lo2 = _mm256_blendv_ps(vr_lo, vr_up, m);
                let vr_up2 = _mm256_blendv_ps(vr_up, vr_lo, m);
                let vi_lo2 = _mm256_blendv_ps(vi_lo, vi_up, m);
                let vi_up2 = _mm256_blendv_ps(vi_up, vi_lo, m);
                _mm256_storeu_ps(r.as_mut_ptr().add(low), vr_lo2);
                _mm256_storeu_ps(r.as_mut_ptr().add(up), vr_up2);
                _mm256_storeu_ps(im.as_mut_ptr().add(low), vi_lo2);
                _mm256_storeu_ps(im.as_mut_ptr().add(up), vi_up2);
                j += 8;
            } else {
                let low = base + j;
                let up = low + bit;
                if (low & cbit) != 0 {
                    r.swap(low, up);
                    im.swap(low, up);
                }
                j += 1;
            }
        }
        base += step;
    }
    GateOutcome::None
}

// EPIC 112.2: CZ gate AVX2 implementation
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn apply_cz_avx2(state: &mut QState, q0: u8, q1: u8) -> GateOutcome {
    use core::arch::x86_64::*;

    let bit0 = 1usize << (q0 as usize);
    let bit1 = 1usize << (q1 as usize);
    let mask = bit0 | bit1;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();

    // Sign flip constant
    let neg_one = _mm256_set1_ps(-1.0);

    let mut i = 0usize;
    while i + 8 <= n {
        // Check which of the 8 lanes need sign flip
        let mut flip_mask: i32 = 0;
        for lane in 0..8 {
            if ((i + lane) & mask) == mask {
                flip_mask |= 1 << lane;
            }
        }

        if flip_mask != 0 {
            let vr = _mm256_loadu_ps(r.as_ptr().add(i));
            let vi = _mm256_loadu_ps(im.as_ptr().add(i));

            // Create blend mask from flip_mask
            let blend = _mm256_castsi256_ps(_mm256_set_epi32(
                if (flip_mask & 0x80) != 0 { -1 } else { 0 },
                if (flip_mask & 0x40) != 0 { -1 } else { 0 },
                if (flip_mask & 0x20) != 0 { -1 } else { 0 },
                if (flip_mask & 0x10) != 0 { -1 } else { 0 },
                if (flip_mask & 0x08) != 0 { -1 } else { 0 },
                if (flip_mask & 0x04) != 0 { -1 } else { 0 },
                if (flip_mask & 0x02) != 0 { -1 } else { 0 },
                if (flip_mask & 0x01) != 0 { -1 } else { 0 },
            ));

            let vr_neg = _mm256_mul_ps(vr, neg_one);
            let vi_neg = _mm256_mul_ps(vi, neg_one);

            let vr_out = _mm256_blendv_ps(vr, vr_neg, blend);
            let vi_out = _mm256_blendv_ps(vi, vi_neg, blend);

            _mm256_storeu_ps(r.as_mut_ptr().add(i), vr_out);
            _mm256_storeu_ps(im.as_mut_ptr().add(i), vi_out);
        }

        i += 8;
    }

    // Scalar tail
    while i < n {
        if (i & mask) == mask {
            r[i] = -r[i];
            im[i] = -im[i];
        }
        i += 1;
    }

    GateOutcome::None
}

// EPIC 112.2: CCZ gate AVX2 implementation
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn apply_ccz_avx2(state: &mut QState, q0: u8, q1: u8, q2: u8) -> GateOutcome {
    use core::arch::x86_64::*;

    let bit0 = 1usize << (q0 as usize);
    let bit1 = 1usize << (q1 as usize);
    let bit2 = 1usize << (q2 as usize);
    let mask = bit0 | bit1 | bit2;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();

    let neg_one = _mm256_set1_ps(-1.0);

    let mut i = 0usize;
    while i + 8 <= n {
        let mut flip_mask: i32 = 0;
        for lane in 0..8 {
            if ((i + lane) & mask) == mask {
                flip_mask |= 1 << lane;
            }
        }

        if flip_mask != 0 {
            let vr = _mm256_loadu_ps(r.as_ptr().add(i));
            let vi = _mm256_loadu_ps(im.as_ptr().add(i));

            let blend = _mm256_castsi256_ps(_mm256_set_epi32(
                if (flip_mask & 0x80) != 0 { -1 } else { 0 },
                if (flip_mask & 0x40) != 0 { -1 } else { 0 },
                if (flip_mask & 0x20) != 0 { -1 } else { 0 },
                if (flip_mask & 0x10) != 0 { -1 } else { 0 },
                if (flip_mask & 0x08) != 0 { -1 } else { 0 },
                if (flip_mask & 0x04) != 0 { -1 } else { 0 },
                if (flip_mask & 0x02) != 0 { -1 } else { 0 },
                if (flip_mask & 0x01) != 0 { -1 } else { 0 },
            ));

            let vr_neg = _mm256_mul_ps(vr, neg_one);
            let vi_neg = _mm256_mul_ps(vi, neg_one);

            let vr_out = _mm256_blendv_ps(vr, vr_neg, blend);
            let vi_out = _mm256_blendv_ps(vi, vi_neg, blend);

            _mm256_storeu_ps(r.as_mut_ptr().add(i), vr_out);
            _mm256_storeu_ps(im.as_mut_ptr().add(i), vi_out);
        }

        i += 8;
    }

    // Scalar tail
    while i < n {
        if (i & mask) == mask {
            r[i] = -r[i];
            im[i] = -im[i];
        }
        i += 1;
    }

    GateOutcome::None
}

// EPIC 112.2: Toffoli gate AVX2 implementation (scalar fallback due to swap pattern)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn apply_toffoli_avx2(
    state: &mut QState,
    ctrl0: u8,
    ctrl1: u8,
    targ: u8,
) -> GateOutcome {
    // Toffoli is harder to vectorize due to the swap pattern
    // Use scalar implementation (still correct, can optimize later if bottleneck)

    let cbit0 = 1usize << (ctrl0 as usize);
    let cbit1 = 1usize << (ctrl1 as usize);
    let tbit = 1usize << (targ as usize);
    let cmask = cbit0 | cbit1;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();

    for i in 0..state.len {
        if (i & cmask) == cmask {
            let j = i ^ tbit;
            if i < j {
                r.swap(i, j);
                im.swap(i, j);
            }
        }
    }

    GateOutcome::None
}

// =============================================================================
// EPIC 112.1: AVX512 Quantum Kernels (Native 512-bit for AMD 9950X3D / Zen 4)
// =============================================================================

// AVX512 Hadamard gate: 16 lanes per iteration (2x throughput vs AVX2)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn apply_h_avx512(state: &mut QState, target: u8) -> GateOutcome {
    use core::arch::x86_64::*;
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let bit = 1usize << (target as usize);
    let step = bit << 1;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();
    let vscale = _mm512_set1_ps(INV_SQRT2);

    let mut base = 0usize;
    while base < n {
        let mut j = 0usize;
        while j < bit {
            if j + 16 <= bit {
                let low = base + j;
                let up = low + bit;
                // Load 16 pairs at once
                let vr_lo = _mm512_loadu_ps(r.as_ptr().add(low));
                let vr_up = _mm512_loadu_ps(r.as_ptr().add(up));
                let vi_lo = _mm512_loadu_ps(im.as_ptr().add(low));
                let vi_up = _mm512_loadu_ps(im.as_ptr().add(up));
                // Hadamard: (a+b)/sqrt2, (a-b)/sqrt2
                let vr_sum = _mm512_add_ps(vr_lo, vr_up);
                let vr_dif = _mm512_sub_ps(vr_lo, vr_up);
                let vi_sum = _mm512_add_ps(vi_lo, vi_up);
                let vi_dif = _mm512_sub_ps(vi_lo, vi_up);
                let vr_lo2 = _mm512_mul_ps(vr_sum, vscale);
                let vr_up2 = _mm512_mul_ps(vr_dif, vscale);
                let vi_lo2 = _mm512_mul_ps(vi_sum, vscale);
                let vi_up2 = _mm512_mul_ps(vi_dif, vscale);
                _mm512_storeu_ps(r.as_mut_ptr().add(low), vr_lo2);
                _mm512_storeu_ps(r.as_mut_ptr().add(up), vr_up2);
                _mm512_storeu_ps(im.as_mut_ptr().add(low), vi_lo2);
                _mm512_storeu_ps(im.as_mut_ptr().add(up), vi_up2);
                j += 16;
            } else {
                // Scalar tail
                let low = base + j;
                let up = low + bit;
                let r0 = r[low];
                let r1 = r[up];
                let i0 = im[low];
                let i1 = im[up];
                r[low] = (r0 + r1) * INV_SQRT2;
                r[up] = (r0 - r1) * INV_SQRT2;
                im[low] = (i0 + i1) * INV_SQRT2;
                im[up] = (i0 - i1) * INV_SQRT2;
                j += 1;
            }
        }
        base += step;
    }
    GateOutcome::None
}

// AVX512 Pauli-X gate: swap pairs with 16-lane vector width
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn apply_x_avx512(state: &mut QState, target: u8) -> GateOutcome {
    use core::arch::x86_64::*;
    let bit = 1usize << (target as usize);
    let step = bit << 1;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();

    let mut base = 0usize;
    while base < n {
        let mut j = 0usize;
        while j < bit {
            if j + 16 <= bit {
                let low = base + j;
                let up = low + bit;
                // Load and swap
                let vr_lo = _mm512_loadu_ps(r.as_ptr().add(low));
                let vr_up = _mm512_loadu_ps(r.as_ptr().add(up));
                let vi_lo = _mm512_loadu_ps(im.as_ptr().add(low));
                let vi_up = _mm512_loadu_ps(im.as_ptr().add(up));
                _mm512_storeu_ps(r.as_mut_ptr().add(low), vr_up);
                _mm512_storeu_ps(r.as_mut_ptr().add(up), vr_lo);
                _mm512_storeu_ps(im.as_mut_ptr().add(low), vi_up);
                _mm512_storeu_ps(im.as_mut_ptr().add(up), vi_lo);
                j += 16;
            } else {
                let low = base + j;
                let up = low + bit;
                r.swap(low, up);
                im.swap(low, up);
                j += 1;
            }
        }
        base += step;
    }
    GateOutcome::None
}

// AVX512 Phase gate: rotate |1⟩ amplitudes by e^(i*theta)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn apply_phase_avx512(state: &mut QState, target: u8, theta: f32) -> GateOutcome {
    use core::arch::x86_64::*;
    let (s, c) = theta.sin_cos();
    let bit = 1usize << (target as usize);
    let step = bit << 1;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();
    let vc = _mm512_set1_ps(c);
    let vs = _mm512_set1_ps(s);

    let mut base = 0usize;
    while base < n {
        let mut j = 0usize;
        while j < bit {
            if j + 16 <= bit {
                let up = base + j + bit;
                // Only transform |1⟩ indices (the upper half of each pair)
                let vr = _mm512_loadu_ps(r.as_ptr().add(up));
                let vi = _mm512_loadu_ps(im.as_ptr().add(up));
                // Complex multiply: (r + i*im) * (c + i*s) = (r*c - im*s) + i*(r*s + im*c)
                let vr_new = _mm512_sub_ps(_mm512_mul_ps(vr, vc), _mm512_mul_ps(vi, vs));
                let vi_new = _mm512_add_ps(_mm512_mul_ps(vr, vs), _mm512_mul_ps(vi, vc));
                _mm512_storeu_ps(r.as_mut_ptr().add(up), vr_new);
                _mm512_storeu_ps(im.as_mut_ptr().add(up), vi_new);
                j += 16;
            } else {
                let up = base + j + bit;
                let re = r[up];
                let imv = im[up];
                r[up] = re * c - imv * s;
                im[up] = re * s + imv * c;
                j += 1;
            }
        }
        base += step;
    }
    GateOutcome::None
}

// AVX512 CNOT gate: swap when control bit is set
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn apply_cnot_avx512(state: &mut QState, control: u8, target: u8) -> GateOutcome {
    use core::arch::x86_64::*;
    let cbit = 1usize << (control as usize);
    let bit = 1usize << (target as usize);
    let step = bit << 1;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();

    let mut base = 0usize;
    while base < n {
        let mut j = 0usize;
        while j < bit {
            if j + 16 <= bit {
                let low = base + j;
                let up = low + bit;
                // Load
                let vr_lo = _mm512_loadu_ps(r.as_ptr().add(low));
                let vr_up = _mm512_loadu_ps(r.as_ptr().add(up));
                let vi_lo = _mm512_loadu_ps(im.as_ptr().add(low));
                let vi_up = _mm512_loadu_ps(im.as_ptr().add(up));
                // Build mask: lanes where (low+lane) & cbit != 0
                let mut mask_bits: u16 = 0;
                for lane in 0..16 {
                    if ((low + lane) & cbit) != 0 {
                        mask_bits |= 1 << lane;
                    }
                }
                let mask = _mm512_int2mask(mask_bits as i32);
                // Blend: swap where mask is set
                let vr_lo2 = _mm512_mask_blend_ps(mask, vr_lo, vr_up);
                let vr_up2 = _mm512_mask_blend_ps(mask, vr_up, vr_lo);
                let vi_lo2 = _mm512_mask_blend_ps(mask, vi_lo, vi_up);
                let vi_up2 = _mm512_mask_blend_ps(mask, vi_up, vi_lo);
                _mm512_storeu_ps(r.as_mut_ptr().add(low), vr_lo2);
                _mm512_storeu_ps(r.as_mut_ptr().add(up), vr_up2);
                _mm512_storeu_ps(im.as_mut_ptr().add(low), vi_lo2);
                _mm512_storeu_ps(im.as_mut_ptr().add(up), vi_up2);
                j += 16;
            } else {
                let low = base + j;
                let up = low + bit;
                if (low & cbit) != 0 {
                    r.swap(low, up);
                    im.swap(low, up);
                }
                j += 1;
            }
        }
        base += step;
    }
    GateOutcome::None
}

// AVX512 CZ gate: flip phase when both qubits are |1⟩
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn apply_cz_avx512(state: &mut QState, q0: u8, q1: u8) -> GateOutcome {
    use core::arch::x86_64::*;
    let bit0 = 1usize << (q0 as usize);
    let bit1 = 1usize << (q1 as usize);
    let mask = bit0 | bit1;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();
    let neg_one = _mm512_set1_ps(-1.0);
    let one = _mm512_set1_ps(1.0);

    let mut i = 0usize;
    while i + 16 <= n {
        // Build mask for lanes where both bits are set
        let mut flip_bits: u16 = 0;
        for lane in 0..16 {
            if ((i + lane) & mask) == mask {
                flip_bits |= 1 << lane;
            }
        }
        if flip_bits != 0 {
            let vr = _mm512_loadu_ps(r.as_ptr().add(i));
            let vi = _mm512_loadu_ps(im.as_ptr().add(i));
            let flip_mask = _mm512_int2mask(flip_bits as i32);
            // Multiply by -1 where mask is set, 1 otherwise
            let scale = _mm512_mask_blend_ps(flip_mask, one, neg_one);
            let vr_out = _mm512_mul_ps(vr, scale);
            let vi_out = _mm512_mul_ps(vi, scale);
            _mm512_storeu_ps(r.as_mut_ptr().add(i), vr_out);
            _mm512_storeu_ps(im.as_mut_ptr().add(i), vi_out);
        }
        i += 16;
    }
    // Scalar tail
    while i < n {
        if (i & mask) == mask {
            r[i] = -r[i];
            im[i] = -im[i];
        }
        i += 1;
    }
    GateOutcome::None
}

// AVX512 CCZ gate: flip phase when all three qubits are |1⟩
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn apply_ccz_avx512(state: &mut QState, q0: u8, q1: u8, q2: u8) -> GateOutcome {
    use core::arch::x86_64::*;
    let bit0 = 1usize << (q0 as usize);
    let bit1 = 1usize << (q1 as usize);
    let bit2 = 1usize << (q2 as usize);
    let mask = bit0 | bit1 | bit2;
    let n = state.len;
    let r = state.real.as_mut_slice();
    let im = state.imag.as_mut_slice();
    let neg_one = _mm512_set1_ps(-1.0);
    let one = _mm512_set1_ps(1.0);

    let mut i = 0usize;
    while i + 16 <= n {
        let mut flip_bits: u16 = 0;
        for lane in 0..16 {
            if ((i + lane) & mask) == mask {
                flip_bits |= 1 << lane;
            }
        }
        if flip_bits != 0 {
            let vr = _mm512_loadu_ps(r.as_ptr().add(i));
            let vi = _mm512_loadu_ps(im.as_ptr().add(i));
            let flip_mask = _mm512_int2mask(flip_bits as i32);
            let scale = _mm512_mask_blend_ps(flip_mask, one, neg_one);
            let vr_out = _mm512_mul_ps(vr, scale);
            let vi_out = _mm512_mul_ps(vi, scale);
            _mm512_storeu_ps(r.as_mut_ptr().add(i), vr_out);
            _mm512_storeu_ps(im.as_mut_ptr().add(i), vi_out);
        }
        i += 16;
    }
    while i < n {
        if (i & mask) == mask {
            r[i] = -r[i];
            im[i] = -im[i];
        }
        i += 1;
    }
    GateOutcome::None
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::*;
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn init_zero_state() {
        let s = QState::new_zero(2);
        assert_eq!(s.len, 4);
        assert!(approx(s.real.as_slice()[0], 1.0));
        assert!(approx(s.imag.as_slice()[0], 0.0));
        for i in 1..4 {
            assert!(approx(s.real.as_slice()[i], 0.0) && approx(s.imag.as_slice()[i], 0.0));
        }
    }

    #[test]
    fn h_on_single_qubit() {
        let mut s = QState::new_zero(1);
        let mut rng = QRng::new(123);
        let _ = apply_gate_scalar(&mut s, &QGate::H(0), &mut rng);
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!(approx(s.real.as_slice()[0], inv_sqrt2));
        assert!(approx(s.real.as_slice()[1], inv_sqrt2));
    }

    #[test]
    fn x_on_single_qubit() {
        let mut s = QState::new_zero(1);
        let mut rng = QRng::new(1);
        let _ = apply_gate_scalar(&mut s, &QGate::X(0), &mut rng);
        assert!(approx(s.real.as_slice()[0], 0.0));
        assert!(approx(s.real.as_slice()[1], 1.0));
    }

    #[test]
    fn bell_pair() {
        let mut s = QState::new_zero(2);
        let mut rng = QRng::new(42);
        let _ = apply_gate_scalar(&mut s, &QGate::H(0), &mut rng);
        let _ = apply_gate_scalar(&mut s, &QGate::CNot(0, 1), &mut rng);
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        // |00> and |11>
        assert!(approx(s.real.as_slice()[0], inv_sqrt2));
        assert!(approx(s.real.as_slice()[3], inv_sqrt2));
        for i in [1usize, 2usize] {
            assert!(approx(s.real.as_slice()[i], 0.0));
        }
    }

    #[test]
    fn measurement_determinism() {
        let mut s = QState::new_zero(1);
        let mut rng1 = QRng::new(55555);
        let mut rng2 = QRng::new(55555);
        let _ = apply_gate_scalar(&mut s, &QGate::H(0), &mut rng1);
        let out1 = apply_gate_scalar(&mut s, &QGate::Measure(0), &mut rng1);
        let mut s2 = QState::new_zero(1);
        let _ = apply_gate_scalar(&mut s2, &QGate::H(0), &mut rng2);
        let out2 = apply_gate_scalar(&mut s2, &QGate::Measure(0), &mut rng2);
        assert_eq!(out1, out2);
    }

    #[test]
    fn avx_backend_parity_small() {
        let q = 3u8;
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx = QState::new_zero(q);
        let prog = [QGate::H(0), QGate::Phase(1, 0.2), QGate::X(2)];
        let mut r1 = QRng::new(123);
        let mut r2 = QRng::new(123);
        for g in prog.iter() {
            let _ = apply_gate_scalar(&mut s_scalar, g, &mut r1);
            let _ = apply_gate_backend(&mut s_avx, g, &mut r2, QBackend::Avx2);
        }
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            let re0 = s_scalar.real.as_slice()[i];
            let im0 = s_scalar.imag.as_slice()[i];
            let re1 = s_avx.real.as_slice()[i];
            let im1 = s_avx.imag.as_slice()[i];
            assert!(
                (re0 - re1).abs() < eps,
                "re mismatch at {}: {} vs {}",
                i,
                re0,
                re1
            );
            assert!(
                (im0 - im1).abs() < eps,
                "im mismatch at {}: {} vs {}",
                i,
                im0,
                im1
            );
        }
    }

    #[test]
    fn avx_cnot_parity_small() {
        let q = 4u8;
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx = QState::new_zero(q);
        let prep = [QGate::H(0), QGate::Phase(2, 0.3), QGate::X(1)];
        let mut r1 = QRng::new(999);
        let mut r2 = QRng::new(999);
        for g in prep.iter() {
            let _ = apply_gate_scalar(&mut s_scalar, g, &mut r1);
            let _ = apply_gate_backend(&mut s_avx, g, &mut r2, QBackend::Avx2);
        }
        let _ = apply_gate_scalar(&mut s_scalar, &QGate::CNot(1, 3), &mut r1);
        let _ = apply_gate_backend(&mut s_avx, &QGate::CNot(1, 3), &mut r2, QBackend::Avx2);
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            let re0 = s_scalar.real.as_slice()[i];
            let im0 = s_scalar.imag.as_slice()[i];
            let re1 = s_avx.real.as_slice()[i];
            let im1 = s_avx.imag.as_slice()[i];
            assert!(
                (re0 - re1).abs() < eps,
                "re mismatch at {}: {} vs {}",
                i,
                re0,
                re1
            );
            assert!(
                (im0 - im1).abs() < eps,
                "im mismatch at {}: {} vs {}",
                i,
                im0,
                im1
            );
        }
    }

    // EPIC 112.2: CZ, CCZ, Toffoli AVX2 parity tests

    #[test]
    fn avx_cz_parity() {
        let q = 4u8;
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx = QState::new_zero(q);
        // Prepare superposition
        let prep = [QGate::H(0), QGate::H(1), QGate::H(2), QGate::H(3)];
        let mut r1 = QRng::new(111);
        let mut r2 = QRng::new(111);
        for g in prep.iter() {
            let _ = apply_gate_scalar(&mut s_scalar, g, &mut r1);
            let _ = apply_gate_backend(&mut s_avx, g, &mut r2, QBackend::Avx2);
        }
        // Apply CZ to multiple qubit pairs
        for (q0, q1) in [(0, 1), (2, 3), (0, 3)] {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::CZ(q0, q1), &mut r1);
            let _ = apply_gate_backend(&mut s_avx, &QGate::CZ(q0, q1), &mut r2, QBackend::Avx2);
        }
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            assert!(
                (s_scalar.real.as_slice()[i] - s_avx.real.as_slice()[i]).abs() < eps,
                "CZ: re mismatch at {}",
                i
            );
            assert!(
                (s_scalar.imag.as_slice()[i] - s_avx.imag.as_slice()[i]).abs() < eps,
                "CZ: im mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn avx_ccz_parity() {
        let q = 5u8;
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx = QState::new_zero(q);
        // Prepare superposition
        let mut r1 = QRng::new(222);
        let mut r2 = QRng::new(222);
        for i in 0..q {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::H(i), &mut r1);
            let _ = apply_gate_backend(&mut s_avx, &QGate::H(i), &mut r2, QBackend::Avx2);
        }
        // Apply CCZ
        for (q0, q1, q2) in [(0, 1, 2), (1, 2, 3), (2, 3, 4)] {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::CCZ(q0, q1, q2), &mut r1);
            let _ =
                apply_gate_backend(&mut s_avx, &QGate::CCZ(q0, q1, q2), &mut r2, QBackend::Avx2);
        }
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            assert!(
                (s_scalar.real.as_slice()[i] - s_avx.real.as_slice()[i]).abs() < eps,
                "CCZ: re mismatch at {}",
                i
            );
            assert!(
                (s_scalar.imag.as_slice()[i] - s_avx.imag.as_slice()[i]).abs() < eps,
                "CCZ: im mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn avx_toffoli_parity() {
        let q = 5u8;
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx = QState::new_zero(q);
        // Prepare mixed state
        let mut r1 = QRng::new(333);
        let mut r2 = QRng::new(333);
        for i in 0..q {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::H(i), &mut r1);
            let _ = apply_gate_backend(&mut s_avx, &QGate::H(i), &mut r2, QBackend::Avx2);
        }
        // Apply Toffoli gates
        for (c0, c1, t) in [(0, 1, 2), (1, 2, 3), (0, 2, 4)] {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::Toffoli(c0, c1, t), &mut r1);
            let _ = apply_gate_backend(
                &mut s_avx,
                &QGate::Toffoli(c0, c1, t),
                &mut r2,
                QBackend::Avx2,
            );
        }
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            assert!(
                (s_scalar.real.as_slice()[i] - s_avx.real.as_slice()[i]).abs() < eps,
                "Toffoli: re mismatch at {}",
                i
            );
            assert!(
                (s_scalar.imag.as_slice()[i] - s_avx.imag.as_slice()[i]).abs() < eps,
                "Toffoli: im mismatch at {}",
                i
            );
        }
    }

    // EPIC 112.1: AVX512 parity tests

    #[test]
    fn avx512_h_parity() {
        // Only run if AVX512 is available
        if !is_avx512f_available() {
            return;
        }
        let q = 6u8; // 64 amplitudes, tests both 16-lane vectorized and scalar paths
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx512 = QState::new_zero(q);
        let mut r1 = QRng::new(444);
        let mut r2 = QRng::new(444);
        // Apply H to all qubits
        for i in 0..q {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::H(i), &mut r1);
            let _ = apply_gate_backend(&mut s_avx512, &QGate::H(i), &mut r2, QBackend::Avx512);
        }
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            assert!(
                (s_scalar.real.as_slice()[i] - s_avx512.real.as_slice()[i]).abs() < eps,
                "AVX512 H: re mismatch at {}",
                i
            );
            assert!(
                (s_scalar.imag.as_slice()[i] - s_avx512.imag.as_slice()[i]).abs() < eps,
                "AVX512 H: im mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn avx512_x_parity() {
        if !is_avx512f_available() {
            return;
        }
        let q = 6u8;
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx512 = QState::new_zero(q);
        let mut r1 = QRng::new(555);
        let mut r2 = QRng::new(555);
        // Prepare and apply X
        for i in 0..q {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::H(i), &mut r1);
            let _ = apply_gate_backend(&mut s_avx512, &QGate::H(i), &mut r2, QBackend::Avx512);
        }
        for i in 0..q {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::X(i), &mut r1);
            let _ = apply_gate_backend(&mut s_avx512, &QGate::X(i), &mut r2, QBackend::Avx512);
        }
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            assert!(
                (s_scalar.real.as_slice()[i] - s_avx512.real.as_slice()[i]).abs() < eps,
                "AVX512 X: re mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn avx512_phase_parity() {
        if !is_avx512f_available() {
            return;
        }
        let q = 6u8;
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx512 = QState::new_zero(q);
        let mut r1 = QRng::new(666);
        let mut r2 = QRng::new(666);
        // Create superposition and apply Phase
        for i in 0..q {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::H(i), &mut r1);
            let _ = apply_gate_backend(&mut s_avx512, &QGate::H(i), &mut r2, QBackend::Avx512);
        }
        for (i, theta) in [(0, 0.5f32), (2, 1.0), (4, 2.5)] {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::Phase(i, theta), &mut r1);
            let _ = apply_gate_backend(
                &mut s_avx512,
                &QGate::Phase(i, theta),
                &mut r2,
                QBackend::Avx512,
            );
        }
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            assert!(
                (s_scalar.real.as_slice()[i] - s_avx512.real.as_slice()[i]).abs() < eps,
                "AVX512 Phase: re mismatch at {}",
                i
            );
            assert!(
                (s_scalar.imag.as_slice()[i] - s_avx512.imag.as_slice()[i]).abs() < eps,
                "AVX512 Phase: im mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn avx512_cnot_parity() {
        if !is_avx512f_available() {
            return;
        }
        let q = 6u8;
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx512 = QState::new_zero(q);
        let mut r1 = QRng::new(777);
        let mut r2 = QRng::new(777);
        // Create superposition
        for i in 0..q {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::H(i), &mut r1);
            let _ = apply_gate_backend(&mut s_avx512, &QGate::H(i), &mut r2, QBackend::Avx512);
        }
        // Apply CNOT gates
        for (c, t) in [(0, 1), (2, 3), (4, 5), (0, 5)] {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::CNot(c, t), &mut r1);
            let _ =
                apply_gate_backend(&mut s_avx512, &QGate::CNot(c, t), &mut r2, QBackend::Avx512);
        }
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            assert!(
                (s_scalar.real.as_slice()[i] - s_avx512.real.as_slice()[i]).abs() < eps,
                "AVX512 CNot: re mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn avx512_cz_parity() {
        if !is_avx512f_available() {
            return;
        }
        let q = 6u8;
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx512 = QState::new_zero(q);
        let mut r1 = QRng::new(888);
        let mut r2 = QRng::new(888);
        for i in 0..q {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::H(i), &mut r1);
            let _ = apply_gate_backend(&mut s_avx512, &QGate::H(i), &mut r2, QBackend::Avx512);
        }
        for (q0, q1) in [(0, 1), (2, 3), (4, 5)] {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::CZ(q0, q1), &mut r1);
            let _ =
                apply_gate_backend(&mut s_avx512, &QGate::CZ(q0, q1), &mut r2, QBackend::Avx512);
        }
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            assert!(
                (s_scalar.real.as_slice()[i] - s_avx512.real.as_slice()[i]).abs() < eps,
                "AVX512 CZ: re mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn avx512_ccz_parity() {
        if !is_avx512f_available() {
            return;
        }
        let q = 6u8;
        let mut s_scalar = QState::new_zero(q);
        let mut s_avx512 = QState::new_zero(q);
        let mut r1 = QRng::new(999);
        let mut r2 = QRng::new(999);
        for i in 0..q {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::H(i), &mut r1);
            let _ = apply_gate_backend(&mut s_avx512, &QGate::H(i), &mut r2, QBackend::Avx512);
        }
        for (q0, q1, q2) in [(0, 1, 2), (3, 4, 5)] {
            let _ = apply_gate_scalar(&mut s_scalar, &QGate::CCZ(q0, q1, q2), &mut r1);
            let _ = apply_gate_backend(
                &mut s_avx512,
                &QGate::CCZ(q0, q1, q2),
                &mut r2,
                QBackend::Avx512,
            );
        }
        let eps = 1e-5f32;
        for i in 0..s_scalar.len {
            assert!(
                (s_scalar.real.as_slice()[i] - s_avx512.real.as_slice()[i]).abs() < eps,
                "AVX512 CCZ: re mismatch at {}",
                i
            );
        }
    }

    // EPIC 62 Phase 2: Multi-worker tile distribution tests

    #[test]
    fn test_multitile_state_creation_8_tiles() {
        // 2 workers × 4 tiles = 8 tiles
        let state = QState::new_zero_multitile(4, 8);
        assert_eq!(state.tile_count, 8);
        assert_eq!(state.n_qubits, 4);
        assert_eq!(state.len, 16); // 2^4 amplitudes per tile
        assert_eq!(state.real.len(), 16 * 8); // total memory

        // Verify all tiles initialized to |0⟩: first 8 amplitudes should be 1.0
        for i in 0..8 {
            assert!(
                approx(state.real.as_slice()[i], 1.0),
                "tile {} real[0] should be 1.0",
                i
            );
        }
        // Rest should be 0
        for i in 8..(16 * 8) {
            assert!(
                approx(state.real.as_slice()[i], 0.0),
                "amplitude {} should be 0.0",
                i
            );
        }
    }

    #[test]
    fn test_multitile_state_creation_32_tiles() {
        // 8 workers × 4 tiles = 32 tiles
        let state = QState::new_zero_multitile(4, 32);
        assert_eq!(state.tile_count, 32);
        assert_eq!(state.worker_count(), 8);
        assert_eq!(state.len, 16); // 2^4 amplitudes per tile
        assert_eq!(state.real.len(), 16 * 32); // 512 floats total

        // Verify all 32 tiles initialized to |0⟩
        for i in 0..32 {
            assert!(
                approx(state.real.as_slice()[i], 1.0),
                "tile {} should be initialized to |0⟩",
                i
            );
        }
    }

    #[test]
    fn test_worker_count() {
        let s1 = QState::new_zero_multitile(4, 1);
        assert_eq!(s1.worker_count(), 0); // Scalar mode

        let s4 = QState::new_zero_multitile(4, 4);
        assert_eq!(s4.worker_count(), 1); // 1 worker

        let s8 = QState::new_zero_multitile(4, 8);
        assert_eq!(s8.worker_count(), 2); // 2 workers

        let s32 = QState::new_zero_multitile(4, 32);
        assert_eq!(s32.worker_count(), 8); // 8 workers

        let s256 = QState::new_zero_multitile(4, 256);
        assert_eq!(s256.worker_count(), 64); // 64 workers
    }

    #[test]
    fn test_worker_tile_range() {
        let state = QState::new_zero_multitile(4, 32); // 8 workers
        assert_eq!(state.worker_count(), 8);

        // Worker 0: tiles [0, 4)
        assert_eq!(state.worker_tile_range(0), (0, 4));

        // Worker 1: tiles [4, 8)
        assert_eq!(state.worker_tile_range(1), (4, 8));

        // Worker 7: tiles [28, 32)
        assert_eq!(state.worker_tile_range(7), (28, 32));
    }

    #[test]
    #[should_panic(expected = "worker_idx 8 >= worker_count 8")]
    fn test_worker_tile_range_invalid() {
        let state = QState::new_zero_multitile(4, 32); // 8 workers
        let _ = state.worker_tile_range(8); // Out of bounds
    }

    #[test]
    fn test_validate_worker_ranges() {
        // Test various configurations
        let s4 = QState::new_zero_multitile(4, 4); // 1 worker
        assert!(s4.validate_worker_ranges());

        let s8 = QState::new_zero_multitile(4, 8); // 2 workers
        assert!(s8.validate_worker_ranges());

        let s32 = QState::new_zero_multitile(4, 32); // 8 workers
        assert!(s32.validate_worker_ranges());

        let s256 = QState::new_zero_multitile(4, 256); // 64 workers
        assert!(s256.validate_worker_ranges());

        // Scalar mode
        let s1 = QState::new_zero_multitile(4, 1);
        assert!(s1.validate_worker_ranges());
    }

    #[test]
    fn test_worker_ranges_no_overlaps() {
        // Verify no overlapping tile ranges for 8 workers
        let state = QState::new_zero_multitile(4, 32);
        assert_eq!(state.worker_count(), 8);

        let mut all_tiles = Vec::new();
        for worker_idx in 0..8 {
            let (start, end) = state.worker_tile_range(worker_idx);
            for tile in start..end {
                assert!(
                    !all_tiles.contains(&tile),
                    "Tile {} assigned to multiple workers",
                    tile
                );
                all_tiles.push(tile);
            }
        }

        // Verify all tiles 0..32 are covered
        all_tiles.sort();
        assert_eq!(all_tiles, (0..32).collect::<Vec<_>>());
    }

    #[test]
    fn test_worker_memory_offsets() {
        let state = QState::new_zero_multitile(4, 32); // 8 workers

        // Worker 0 starts at offset 0
        let (real_off_0, imag_off_0) = state.worker_memory_offsets(0);
        assert_eq!(real_off_0, 0);
        assert_eq!(imag_off_0, 0);

        // Worker 1 starts at tile 4 = 4 floats = 16 bytes
        let (real_off_1, imag_off_1) = state.worker_memory_offsets(1);
        assert_eq!(real_off_1, 4 * 4); // 4 tiles × 4 bytes/float
        assert_eq!(imag_off_1, 4 * 4);

        // Worker 7 starts at tile 28 = 28 floats = 112 bytes
        let (real_off_7, imag_off_7) = state.worker_memory_offsets(7);
        assert_eq!(real_off_7, 28 * 4);
        assert_eq!(imag_off_7, 28 * 4);
    }

    #[test]
    fn test_tiles_per_worker() {
        // Tiles per worker is always 4 (F32X4 SIMD width)
        assert_eq!(QState::tiles_per_worker(), 4);
    }

    #[test]
    #[should_panic(expected = "tile_count must be 1-256")]
    fn test_multitile_tile_count_too_large() {
        let _ = QState::new_zero_multitile(4, 257);
    }

    #[test]
    #[should_panic(expected = "tile_count must be 1-256")]
    fn test_multitile_tile_count_zero() {
        let _ = QState::new_zero_multitile(4, 0);
    }

    // =============================================================================
    // SPRINT 2.1a: Tests for new gates (CZ, Toffoli, CCZ)
    // =============================================================================

    #[test]
    fn test_cz_flips_11_only() {
        // CZ should flip phase of |11⟩ only
        let mut state = QState::new_zero(2);
        let mut rng = QRng::new(0);

        // Prepare |11⟩: X(0), X(1)
        apply_gate_scalar(&mut state, &QGate::X(0), &mut rng);
        apply_gate_scalar(&mut state, &QGate::X(1), &mut rng);

        // State should be [0, 0, 0, 1] (index 3 = |11⟩)
        assert!(approx(state.real.as_slice()[3], 1.0));

        // Apply CZ
        apply_gate_scalar(&mut state, &QGate::CZ(0, 1), &mut rng);

        // |11⟩ should now have phase flipped to -1
        assert!(approx(state.real.as_slice()[3], -1.0));
    }

    #[test]
    fn test_cz_leaves_others_unchanged() {
        // CZ should not affect |00⟩, |01⟩, |10⟩
        let mut state = QState::new_zero(2);
        let mut rng = QRng::new(0);

        // Create uniform superposition
        apply_gate_scalar(&mut state, &QGate::H(0), &mut rng);
        apply_gate_scalar(&mut state, &QGate::H(1), &mut rng);

        // Store amplitudes before CZ
        let before: Vec<f32> = state.real.as_slice().to_vec();

        // Apply CZ
        apply_gate_scalar(&mut state, &QGate::CZ(0, 1), &mut rng);

        // Only index 3 (|11⟩) should change sign
        assert!(approx(state.real.as_slice()[0], before[0])); // |00⟩
        assert!(approx(state.real.as_slice()[1], before[1])); // |01⟩
        assert!(approx(state.real.as_slice()[2], before[2])); // |10⟩
        assert!(approx(state.real.as_slice()[3], -before[3])); // |11⟩ flipped
    }

    #[test]
    fn test_toffoli_110_to_111() {
        // Toffoli(0,1,2) on |110⟩ should give |111⟩
        let mut state = QState::new_zero(3);
        let mut rng = QRng::new(0);

        // Prepare |110⟩: X(1), X(2)
        // In our convention: q2=1, q1=1, q0=0 → index 6 = binary 110
        apply_gate_scalar(&mut state, &QGate::X(1), &mut rng);
        apply_gate_scalar(&mut state, &QGate::X(2), &mut rng);

        // State should be [...,0,1,0] (index 6)
        assert!(approx(state.real.as_slice()[6], 1.0));

        // Toffoli(1, 2, 0): controls are q1 and q2, target is q0
        apply_gate_scalar(&mut state, &QGate::Toffoli(1, 2, 0), &mut rng);

        // Should flip to |111⟩ = index 7
        assert!(approx(state.real.as_slice()[7], 1.0));
        assert!(approx(state.real.as_slice()[6], 0.0));
    }

    #[test]
    fn test_toffoli_111_to_110() {
        // Toffoli is self-inverse: |111⟩ → |110⟩
        let mut state = QState::new_zero(3);
        let mut rng = QRng::new(0);

        // Prepare |111⟩
        apply_gate_scalar(&mut state, &QGate::X(0), &mut rng);
        apply_gate_scalar(&mut state, &QGate::X(1), &mut rng);
        apply_gate_scalar(&mut state, &QGate::X(2), &mut rng);

        assert!(approx(state.real.as_slice()[7], 1.0));

        // Toffoli(1, 2, 0)
        apply_gate_scalar(&mut state, &QGate::Toffoli(1, 2, 0), &mut rng);

        // Should flip to |110⟩ = index 6
        assert!(approx(state.real.as_slice()[6], 1.0));
        assert!(approx(state.real.as_slice()[7], 0.0));
    }

    #[test]
    fn test_ccz_flips_111_only() {
        // CCZ should flip phase of |111⟩ only
        let mut state = QState::new_zero(3);
        let mut rng = QRng::new(0);

        // Prepare |111⟩
        apply_gate_scalar(&mut state, &QGate::X(0), &mut rng);
        apply_gate_scalar(&mut state, &QGate::X(1), &mut rng);
        apply_gate_scalar(&mut state, &QGate::X(2), &mut rng);

        assert!(approx(state.real.as_slice()[7], 1.0));

        // Apply CCZ
        apply_gate_scalar(&mut state, &QGate::CCZ(0, 1, 2), &mut rng);

        // |111⟩ should have phase flipped
        assert!(approx(state.real.as_slice()[7], -1.0));
    }

    #[test]
    fn test_ccz_leaves_others_unchanged() {
        // CCZ on superposition should only flip |111⟩
        let mut state = QState::new_zero(3);
        let mut rng = QRng::new(0);

        // Create uniform superposition
        apply_gate_scalar(&mut state, &QGate::H(0), &mut rng);
        apply_gate_scalar(&mut state, &QGate::H(1), &mut rng);
        apply_gate_scalar(&mut state, &QGate::H(2), &mut rng);

        let before: Vec<f32> = state.real.as_slice().to_vec();

        // Apply CCZ
        apply_gate_scalar(&mut state, &QGate::CCZ(0, 1, 2), &mut rng);

        // Only index 7 (|111⟩) should change sign
        for i in 0..7 {
            assert!(
                approx(state.real.as_slice()[i], before[i]),
                "Index {} changed unexpectedly",
                i
            );
        }
        assert!(
            approx(state.real.as_slice()[7], -before[7]),
            "|111⟩ should have flipped sign"
        );
    }

    #[test]
    fn test_cz_is_symmetric() {
        // CZ(a,b) should equal CZ(b,a)
        let mut state1 = QState::new_zero(2);
        let mut state2 = QState::new_zero(2);
        let mut rng1 = QRng::new(42);
        let mut rng2 = QRng::new(42);

        // Prepare same superposition
        apply_gate_scalar(&mut state1, &QGate::H(0), &mut rng1);
        apply_gate_scalar(&mut state1, &QGate::H(1), &mut rng1);
        apply_gate_scalar(&mut state2, &QGate::H(0), &mut rng2);
        apply_gate_scalar(&mut state2, &QGate::H(1), &mut rng2);

        // Apply CZ with different orderings
        apply_gate_scalar(&mut state1, &QGate::CZ(0, 1), &mut rng1);
        apply_gate_scalar(&mut state2, &QGate::CZ(1, 0), &mut rng2);

        // States should be identical
        for i in 0..state1.len {
            assert!(approx(state1.real.as_slice()[i], state2.real.as_slice()[i]));
        }
    }

    #[test]
    fn test_toffoli_self_inverse() {
        // Applying Toffoli twice should return to original state
        let mut state = QState::new_zero(3);
        let mut rng = QRng::new(0);

        // Create some superposition
        apply_gate_scalar(&mut state, &QGate::H(0), &mut rng);
        apply_gate_scalar(&mut state, &QGate::H(1), &mut rng);
        apply_gate_scalar(&mut state, &QGate::H(2), &mut rng);

        let before: Vec<f32> = state.real.as_slice().to_vec();

        // Apply Toffoli twice
        apply_gate_scalar(&mut state, &QGate::Toffoli(0, 1, 2), &mut rng);
        apply_gate_scalar(&mut state, &QGate::Toffoli(0, 1, 2), &mut rng);

        // Should return to original state
        for i in 0..state.len {
            assert!(approx(state.real.as_slice()[i], before[i]));
        }
    }
}
