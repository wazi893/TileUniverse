//! Solution candidate encoding for the search engine.
//!
//! Each candidate is a 64-bit solution with associated fitness.
//! Designed for efficient parallel evaluation and genetic operations.

use std::fmt;

/// A solution candidate in the search space.
///
/// Currently uses 64-bit encoding, expandable to 128-bit for larger problems.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Candidate {
    /// The solution bits (64-bit for now)
    pub bits: u64,
    /// Cached fitness score (0-255, higher is better)
    pub fitness: u8,
    /// Generation this candidate was created
    pub generation: u32,
    /// Flags: selected, elite, mutated, etc.
    pub flags: CandidateFlags,
}

/// Flags for tracking candidate state
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CandidateFlags {
    bits: u8,
}

impl CandidateFlags {
    pub const SELECTED: u8 = 0b0000_0001;
    pub const ELITE: u8 = 0b0000_0010;
    pub const MUTATED: u8 = 0b0000_0100;
    pub const CROSSED: u8 = 0b0000_1000;

    #[inline]
    pub fn new() -> Self {
        Self { bits: 0 }
    }

    #[inline]
    pub fn is_selected(&self) -> bool {
        self.bits & Self::SELECTED != 0
    }

    #[inline]
    pub fn is_elite(&self) -> bool {
        self.bits & Self::ELITE != 0
    }

    #[inline]
    pub fn is_mutated(&self) -> bool {
        self.bits & Self::MUTATED != 0
    }

    #[inline]
    pub fn is_crossed(&self) -> bool {
        self.bits & Self::CROSSED != 0
    }

    #[inline]
    pub fn set_selected(&mut self, v: bool) {
        if v {
            self.bits |= Self::SELECTED;
        } else {
            self.bits &= !Self::SELECTED;
        }
    }

    #[inline]
    pub fn set_elite(&mut self, v: bool) {
        if v {
            self.bits |= Self::ELITE;
        } else {
            self.bits &= !Self::ELITE;
        }
    }

    #[inline]
    pub fn set_mutated(&mut self, v: bool) {
        if v {
            self.bits |= Self::MUTATED;
        } else {
            self.bits &= !Self::MUTATED;
        }
    }

    #[inline]
    pub fn set_crossed(&mut self, v: bool) {
        if v {
            self.bits |= Self::CROSSED;
        } else {
            self.bits &= !Self::CROSSED;
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.bits = 0;
    }
}

impl Candidate {
    /// Create a new candidate with given bits
    #[inline]
    pub fn new(bits: u64) -> Self {
        Self {
            bits,
            fitness: 0,
            generation: 0,
            flags: CandidateFlags::new(),
        }
    }

    /// Create a candidate with all zeros
    #[inline]
    pub fn zeros() -> Self {
        Self::new(0)
    }

    /// Create a candidate with all ones
    #[inline]
    pub fn ones() -> Self {
        Self::new(u64::MAX)
    }

    /// Create a random candidate
    #[inline]
    pub fn random(rng: &mut impl Rng) -> Self {
        Self::new(rng.next_u64())
    }

    /// Create a random candidate with specified number of bits set
    pub fn random_with_density(rng: &mut impl Rng, num_bits: usize, density: f32) -> Self {
        let target_ones = ((num_bits as f32) * density) as usize;
        let mut bits = 0u64;
        let mut set_count = 0;

        while set_count < target_ones {
            let pos = (rng.next_u64() as usize) % num_bits;
            if bits & (1 << pos) == 0 {
                bits |= 1 << pos;
                set_count += 1;
            }
        }

        Self::new(bits)
    }

    /// Get a specific bit (0 or 1)
    #[inline]
    pub fn get_bit(&self, pos: usize) -> bool {
        debug_assert!(pos < 64, "Bit position out of range");
        (self.bits >> pos) & 1 == 1
    }

    /// Set a specific bit
    #[inline]
    pub fn set_bit(&mut self, pos: usize, value: bool) {
        debug_assert!(pos < 64, "Bit position out of range");
        if value {
            self.bits |= 1 << pos;
        } else {
            self.bits &= !(1 << pos);
        }
    }

    /// Flip a specific bit
    #[inline]
    pub fn flip_bit(&mut self, pos: usize) {
        debug_assert!(pos < 64, "Bit position out of range");
        self.bits ^= 1 << pos;
    }

    /// Count the number of 1-bits (population count)
    #[inline]
    pub fn count_ones(&self) -> u32 {
        self.bits.count_ones()
    }

    /// Count the number of 0-bits
    #[inline]
    pub fn count_zeros(&self) -> u32 {
        self.bits.count_zeros()
    }

    /// Hamming distance to another candidate
    #[inline]
    pub fn hamming_distance(&self, other: &Candidate) -> u32 {
        (self.bits ^ other.bits).count_ones()
    }

    /// Single-point crossover with another candidate
    #[inline]
    pub fn crossover_single_point(
        &self,
        other: &Candidate,
        point: usize,
    ) -> (Candidate, Candidate) {
        debug_assert!(point < 64, "Crossover point out of range");
        let mask = (1u64 << point) - 1; // Lower bits
        let child1 = Candidate::new((self.bits & mask) | (other.bits & !mask));
        let child2 = Candidate::new((other.bits & mask) | (self.bits & !mask));
        (child1, child2)
    }

    /// Two-point crossover
    pub fn crossover_two_point(
        &self,
        other: &Candidate,
        point1: usize,
        point2: usize,
    ) -> (Candidate, Candidate) {
        let (p1, p2) = if point1 < point2 {
            (point1, point2)
        } else {
            (point2, point1)
        };

        let mask = ((1u64 << p2) - 1) ^ ((1u64 << p1) - 1); // Middle bits
        let child1 = Candidate::new((self.bits & !mask) | (other.bits & mask));
        let child2 = Candidate::new((other.bits & !mask) | (self.bits & mask));
        (child1, child2)
    }

    /// Uniform crossover (each bit from random parent)
    #[inline]
    pub fn crossover_uniform(&self, other: &Candidate, rng: &mut impl Rng) -> Candidate {
        let mask = rng.next_u64();
        Candidate::new((self.bits & mask) | (other.bits & !mask))
    }

    /// Mutate with given probability per bit
    pub fn mutate(&mut self, probability: f32, rng: &mut impl Rng) {
        for i in 0..64 {
            if rng.next_f32() < probability {
                self.flip_bit(i);
            }
        }
        self.flags.set_mutated(true);
    }

    /// Mutate exactly n random bits
    pub fn mutate_n_bits(&mut self, n: usize, rng: &mut impl Rng) {
        let mut flipped = 0;
        while flipped < n {
            let pos = (rng.next_u64() as usize) % 64;
            self.flip_bit(pos);
            flipped += 1;
        }
        self.flags.set_mutated(true);
    }
}

impl fmt::Debug for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Candidate {{ bits: 0x{:016X}, fitness: {}, gen: {} }}",
            self.bits, self.fitness, self.generation
        )
    }
}

impl fmt::Display for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[f={:3}] {:064b}", self.fitness, self.bits)
    }
}

/// Simple RNG trait for candidate operations
/// (Avoids pulling in full rand crate for core types)
pub trait Rng {
    fn next_u64(&mut self) -> u64;

    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1 << 24) as f32
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Fast xorshift64 RNG
#[derive(Clone, Debug)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEADBEEF } else { seed },
        }
    }

    pub fn from_entropy() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xCAFEBABE);
        Self::new(seed)
    }
}

impl Rng for Xorshift64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

impl Default for Xorshift64 {
    fn default() -> Self {
        Self::new(0x12345678_9ABCDEF0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_creation() {
        let c = Candidate::new(0b1010_1010);
        assert_eq!(c.bits, 0b1010_1010);
        assert_eq!(c.fitness, 0);
    }

    #[test]
    fn test_bit_operations() {
        let mut c = Candidate::zeros();
        assert!(!c.get_bit(0));

        c.set_bit(5, true);
        assert!(c.get_bit(5));
        assert_eq!(c.bits, 0b100000);

        c.flip_bit(5);
        assert!(!c.get_bit(5));
        assert_eq!(c.bits, 0);
    }

    #[test]
    fn test_count_ones() {
        let c = Candidate::new(0b1111_0000_1111_0000);
        assert_eq!(c.count_ones(), 8);
    }

    #[test]
    fn test_hamming_distance() {
        let c1 = Candidate::new(0b1111_0000);
        let c2 = Candidate::new(0b1010_1010);
        assert_eq!(c1.hamming_distance(&c2), 4);
    }

    #[test]
    fn test_crossover_single_point() {
        let p1 = Candidate::new(0b11110000);
        let p2 = Candidate::new(0b00001111);
        let (c1, c2) = p1.crossover_single_point(&p2, 4);
        assert_eq!(c1.bits, 0b00000000);
        assert_eq!(c2.bits, 0b11111111);
    }

    #[test]
    fn test_crossover_uniform() {
        let p1 = Candidate::ones();
        let p2 = Candidate::zeros();
        let mut rng = Xorshift64::new(42);
        let child = p1.crossover_uniform(&p2, &mut rng);
        // Should have some bits from each parent
        assert!(child.count_ones() > 0);
        assert!(child.count_ones() < 64);
    }

    #[test]
    fn test_mutation() {
        let mut c = Candidate::zeros();
        let mut rng = Xorshift64::new(42);
        c.mutate(0.5, &mut rng);
        // With 50% mutation rate, expect roughly 32 bits flipped
        let ones = c.count_ones();
        assert!(ones > 10 && ones < 54, "Got {} ones", ones);
    }

    #[test]
    fn test_rng_distribution() {
        let mut rng = Xorshift64::new(12345);
        let mut sum = 0.0;
        for _ in 0..10000 {
            sum += rng.next_f64();
        }
        let mean = sum / 10000.0;
        assert!((mean - 0.5).abs() < 0.02, "Mean was {}", mean);
    }

    #[test]
    fn test_flags() {
        let mut c = Candidate::new(0);
        assert!(!c.flags.is_selected());

        c.flags.set_selected(true);
        assert!(c.flags.is_selected());

        c.flags.set_elite(true);
        assert!(c.flags.is_elite());
        assert!(c.flags.is_selected());

        c.flags.clear();
        assert!(!c.flags.is_selected());
        assert!(!c.flags.is_elite());
    }
}
