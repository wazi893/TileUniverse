/// Classical number theory utilities for Shor's algorithm
///
/// This module provides classical algorithms for:
/// - GCD and extended GCD
/// - Modular arithmetic
/// - Period finding (classical reference)
/// - Factor extraction from periods
/// - Primality testing

/// Greatest Common Divisor using Euclidean algorithm
///
/// # Examples
/// ```
/// use engine::number_theory::gcd;
/// assert_eq!(gcd(15, 21), 3);
/// assert_eq!(gcd(17, 19), 1);
/// ```
pub fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Extended Euclidean algorithm
/// Returns (gcd, x, y) such that ax + by = gcd(a, b)
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a, 1, 0)
    } else {
        let (g, x, y) = extended_gcd(b, a % b);
        (g, y, x - (a / b) * y)
    }
}

/// Modular exponentiation: (base^exp) mod modulus
/// Uses binary exponentiation for efficiency
///
/// # Examples
/// ```
/// use engine::number_theory::mod_exp;
/// assert_eq!(mod_exp(2, 4, 15), 1);  // 2^4 = 16 ≡ 1 (mod 15)
/// assert_eq!(mod_exp(7, 4, 15), 1);  // 7^4 ≡ 1 (mod 15)
/// ```
pub fn mod_exp(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }

    let mut result = 1;
    base %= modulus;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }

    result
}

/// Find the multiplicative order of a modulo n
/// Returns the smallest r > 0 such that a^r ≡ 1 (mod n)
///
/// This is a classical (brute force) implementation for testing purposes.
/// In practice, Shor's algorithm finds this quantum-mechanically.
///
/// # Examples
/// ```
/// use engine::number_theory::classical_order;
/// assert_eq!(classical_order(2, 15), Some(4));
/// assert_eq!(classical_order(7, 15), Some(4));
/// ```
pub fn classical_order(a: u64, n: u64) -> Option<u64> {
    if gcd(a, n) != 1 {
        return None; // a and n must be coprime
    }

    for r in 1..n {
        if mod_exp(a, r, n) == 1 {
            return Some(r);
        }
    }

    None
}

/// Extract factors from period using Shor's classical post-processing
///
/// Given N, a, and period r (where a^r ≡ 1 mod N), attempts to find factors.
///
/// Algorithm:
/// 1. Check if r is even (required for method)
/// 2. Compute x = a^(r/2) mod N
/// 3. Check x ≠ N-1 (otherwise trivial)
/// 4. Compute p = gcd(x-1, N) and q = gcd(x+1, N)
/// 5. Verify p, q are non-trivial factors
///
/// # Examples
/// ```
/// use engine::number_theory::factors_from_period;
/// // For N=15, a=7, period r=4
/// assert_eq!(factors_from_period(15, 7, 4), Some((5, 3)));
/// ```
pub fn factors_from_period(n: u64, a: u64, r: u64) -> Option<(u64, u64)> {
    // Period must be even
    if r % 2 == 1 {
        return None;
    }

    // Compute x = a^(r/2) mod N
    let x = mod_exp(a, r / 2, n);

    // Check for trivial case: x ≡ -1 (mod N)
    if x == n - 1 || x == 1 {
        return None;
    }

    // Try to extract factors
    let p = gcd(x + 1, n);
    let q = gcd(if x > 1 { x - 1 } else { n + x - 1 }, n);

    // Verify we found non-trivial factors
    if p > 1 && p < n && q > 1 && q < n {
        // Check if factors multiply to n
        if p * q == n {
            return Some((p, q));
        }
        // If not exact, return the non-trivial factor
        if p > 1 && p < n {
            return Some((p, n / p));
        }
        if q > 1 && q < n {
            return Some((q, n / q));
        }
    }

    None
}

/// Simple primality test using trial division
/// Sufficient for small numbers in our demo
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let limit = (n as f64).sqrt() as u64 + 1;
    for i in (3..=limit).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }

    true
}

/// Check if n is a prime power (p^k for some prime p and k > 1)
pub fn is_prime_power(n: u64) -> Option<(u64, u32)> {
    if n < 2 {
        return None;
    }

    // Check small primes
    for p in 2..=((n as f64).sqrt() as u64 + 1) {
        if n % p == 0 {
            let mut k = 0u32;
            let mut temp = n;
            while temp % p == 0 {
                temp /= p;
                k += 1;
            }
            if temp == 1 && k > 1 {
                return Some((p, k));
            }
        }
    }

    None
}

/// Continued fraction expansion to extract period from measurement
///
/// Given a measured value s and the size 2^n of the counting register,
/// finds the best rational approximation p/q where q might be the period.
///
/// Returns candidate periods (denominators of convergents)
pub fn continued_fraction_candidates(measured: u64, register_size: u64) -> Vec<u64> {
    if measured == 0 {
        return vec![];
    }

    let mut candidates = Vec::new();
    let mut a = measured;
    let mut b = register_size;

    // Convergents
    let mut h_prev2 = 0i64;
    let mut h_prev1 = 1i64;
    let mut k_prev2 = 1i64;
    let mut k_prev1 = 0i64;

    for _ in 0..20 {
        // Limit iterations
        if b == 0 {
            break;
        }

        let q = a / b;
        let r = a % b;

        // Compute next convergent
        let h = q as i64 * h_prev1 + h_prev2;
        let k = q as i64 * k_prev1 + k_prev2;

        if k > 0 && k < register_size as i64 {
            candidates.push(k as u64);
        }

        // Update for next iteration
        a = b;
        b = r;
        h_prev2 = h_prev1;
        h_prev1 = h;
        k_prev2 = k_prev1;
        k_prev1 = k;
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(15, 21), 3);
        assert_eq!(gcd(17, 19), 1);
        assert_eq!(gcd(100, 50), 50);
        assert_eq!(gcd(1, 1), 1);
        assert_eq!(gcd(0, 5), 5);
    }

    #[test]
    fn test_extended_gcd() {
        let (g, x, y) = extended_gcd(15, 21);
        assert_eq!(g, 3);
        assert_eq!(15 * x + 21 * y, 3);
    }

    #[test]
    fn test_mod_exp() {
        // 2^4 = 16 ≡ 1 (mod 15)
        assert_eq!(mod_exp(2, 4, 15), 1);

        // 7^4 ≡ 1 (mod 15) - period of 7 mod 15 is 4
        assert_eq!(mod_exp(7, 4, 15), 1);

        // 3^2 = 9 (mod 15)
        assert_eq!(mod_exp(3, 2, 15), 9);

        // Large exponent
        assert_eq!(mod_exp(2, 100, 15), 1); // Period is 4, so 2^100 = 2^(4*25) ≡ 1
    }

    #[test]
    fn test_classical_order() {
        // order(2, 15) = 4 because 2^4 ≡ 1 (mod 15)
        assert_eq!(classical_order(2, 15), Some(4));

        // order(7, 15) = 4 because 7^4 ≡ 1 (mod 15)
        assert_eq!(classical_order(7, 15), Some(4));

        // order(4, 15) = 2 because 4^2 ≡ 1 (mod 15)
        assert_eq!(classical_order(4, 15), Some(2));

        // order(2, 21) = 6 because 2^6 ≡ 1 (mod 21)
        assert_eq!(classical_order(2, 21), Some(6));
    }

    #[test]
    fn test_factors_from_period() {
        // N=15, a=7, r=4
        // 7^2 = 49 ≡ 4 (mod 15)
        // gcd(4-1, 15) = gcd(3, 15) = 3
        // gcd(4+1, 15) = gcd(5, 15) = 5
        let result = factors_from_period(15, 7, 4);
        assert!(result.is_some());
        let (p, q) = result.unwrap();
        assert!((p == 3 && q == 5) || (p == 5 && q == 3));

        // N=21, a=2, r=6
        let result = factors_from_period(21, 2, 6);
        assert!(result.is_some());
        let (p, q) = result.unwrap();
        assert!((p == 3 && q == 7) || (p == 7 && q == 3));
    }

    #[test]
    fn test_factors_from_period_odd() {
        // Odd period should fail
        assert_eq!(factors_from_period(15, 2, 3), None);
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(11));
        assert!(is_prime(13));
        assert!(is_prime(17));
        assert!(is_prime(19));

        assert!(!is_prime(1));
        assert!(!is_prime(4));
        assert!(!is_prime(15));
        assert!(!is_prime(21));
        assert!(!is_prime(100));
    }

    #[test]
    fn test_is_prime_power() {
        assert_eq!(is_prime_power(4), Some((2, 2))); // 2^2
        assert_eq!(is_prime_power(8), Some((2, 3))); // 2^3
        assert_eq!(is_prime_power(9), Some((3, 2))); // 3^2
        assert_eq!(is_prime_power(27), Some((3, 3))); // 3^3

        assert_eq!(is_prime_power(15), None); // 3 * 5, not a prime power
        assert_eq!(is_prime_power(6), None); // 2 * 3, not a prime power
    }

    #[test]
    fn test_continued_fraction_candidates() {
        // Test with register size 2^8 = 256
        let candidates = continued_fraction_candidates(85, 256);

        // Should include good approximations
        assert!(!candidates.is_empty());

        // For s/2^n = 85/256 ≈ 0.332, continued fraction should find period candidates
        // This is a probabilistic test - just verify we get some candidates
        assert!(candidates.len() > 0 && candidates.len() < 20);
    }
}
