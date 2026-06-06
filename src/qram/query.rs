//! QRAM Query Interface: High-level API for quantum memory queries
//!
//! Provides a unified interface for classical and quantum QRAM operations.
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::QRAMQuery;
//!
//! let mut qram = QRAMQuery::new(3);
//! qram.load(&[10, 20, 30, 40, 50, 60, 70, 80]);
//!
//! // Classical O(1) lookup
//! assert_eq!(qram.query_classical(5), Some(60));
//!
//! // Quantum superposition query
//! let results = qram.query_superposition(42);
//! for r in results {
//!     println!("Address {}: data={}, prob={:.3}", r.address, r.data, r.probability);
//! }
//! ```

use super::tree::QRAMTree;
use crate::quantum::{GateOutcome, QGate, QRng, QState, apply_gate_scalar};

/// Result of a QRAM query
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Measured address
    pub address: usize,
    /// Retrieved data value
    pub data: u64,
    /// Probability of this outcome
    pub probability: f64,
}

impl QueryResult {
    /// Create a new query result
    pub fn new(address: usize, data: u64, probability: f64) -> Self {
        Self {
            address,
            data,
            probability,
        }
    }
}

/// QRAM query executor
///
/// Wraps a QRAMTree and provides high-level query operations.
#[derive(Clone, Debug)]
pub struct QRAMQuery {
    tree: QRAMTree,
}

impl QRAMQuery {
    /// Create a new QRAM with 2^depth memory cells
    ///
    /// # Arguments
    /// * `depth` - Number of address bits (1-20)
    ///
    /// # Example
    /// ```ignore
    /// let qram = QRAMQuery::new(4); // 16 memory cells
    /// ```
    pub fn new(depth: usize) -> Self {
        Self {
            tree: QRAMTree::new(depth),
        }
    }

    /// Load data into QRAM memory cells
    ///
    /// Copies up to `size()` elements from the input slice.
    pub fn load(&mut self, data: &[u64]) {
        self.tree.load_data(data);
    }

    /// Set a single memory cell value
    pub fn set(&mut self, address: usize, value: u64) {
        self.tree.set(address, value);
    }

    /// Get a single memory cell value (direct access, not a query)
    pub fn get(&self, address: usize) -> Option<u64> {
        self.tree.get(address)
    }

    /// Number of memory cells (2^depth)
    pub fn size(&self) -> usize {
        self.tree.size()
    }

    /// Address bit width (tree depth)
    pub fn depth(&self) -> usize {
        self.tree.depth()
    }

    /// Total qubits needed for quantum simulation
    pub fn total_qubits(&self) -> usize {
        self.tree.total_qubits()
    }

    // =========================================================================
    // Classical Query
    // =========================================================================

    /// Classical query - O(1) lookup
    ///
    /// Routes through the tree classically and returns the data at the address.
    ///
    /// # Arguments
    /// * `address` - Memory address to query (0 to size()-1)
    ///
    /// # Returns
    /// The data value at the address, or None if out of bounds.
    pub fn query_classical(&mut self, address: usize) -> Option<u64> {
        if address >= self.tree.size() {
            return None;
        }

        self.tree.reset_routers();
        let _cell = self.tree.route_classical(address);
        self.tree.get(address)
    }

    // =========================================================================
    // Quantum Basis State Query
    // =========================================================================

    /// Quantum query with address in computational basis
    ///
    /// Prepares a specific address in the quantum register and queries.
    /// Since the address is a basis state, the result is deterministic.
    ///
    /// # Arguments
    /// * `address` - The address to query
    /// * `seed` - RNG seed for quantum simulation
    ///
    /// # Returns
    /// Query result with probability 1.0 (deterministic)
    pub fn query_basis(&mut self, address: usize, seed: u64) -> Option<QueryResult> {
        if address >= self.tree.size() {
            return None;
        }

        let n_qubits = self.tree.total_qubits();

        // Create quantum state
        let mut state = QState::new_zero(n_qubits as u8);
        let mut rng = QRng::new(seed);

        // Initialize address register to encode the specific address
        for gate in self.tree.init_address_circuit(address) {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }

        // Apply QRAM query circuit
        for gate in self.tree.query_circuit() {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }

        // For a basis state query, the result is deterministic
        let data = self.tree.get(address)?;

        Some(QueryResult {
            address,
            data,
            probability: 1.0,
        })
    }

    // =========================================================================
    // Quantum Superposition Query
    // =========================================================================

    /// Quantum query with address in uniform superposition
    ///
    /// Prepares all addresses in equal superposition and queries.
    /// Returns all possible outcomes with their probabilities.
    ///
    /// # Arguments
    /// * `seed` - RNG seed for quantum simulation
    ///
    /// # Returns
    /// Vector of query results, one per address with non-zero probability
    pub fn query_superposition(&mut self, seed: u64) -> Vec<QueryResult> {
        let depth = self.tree.depth();
        let n_qubits = self.tree.total_qubits();

        // Create quantum state
        let mut state = QState::new_zero(n_qubits as u8);
        let mut rng = QRng::new(seed);

        // Initialize address register in uniform superposition
        for gate in self.tree.init_superposition_circuit() {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }

        // Apply QRAM query circuit
        for gate in self.tree.query_circuit() {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }

        // Extract probabilities for each address
        self.extract_address_probabilities(&state, depth)
    }

    /// Query with custom initial state preparation
    ///
    /// Allows custom address register initialization before query.
    ///
    /// # Arguments
    /// * `init_gates` - Gates to prepare the address register
    /// * `seed` - RNG seed
    pub fn query_custom(&mut self, init_gates: &[QGate], seed: u64) -> Vec<QueryResult> {
        let depth = self.tree.depth();
        let n_qubits = self.tree.total_qubits();

        let mut state = QState::new_zero(n_qubits as u8);
        let mut rng = QRng::new(seed);

        // Apply custom initialization
        for gate in init_gates {
            apply_gate_scalar(&mut state, gate, &mut rng);
        }

        // Apply QRAM query circuit
        for gate in self.tree.query_circuit() {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }

        self.extract_address_probabilities(&state, depth)
    }

    /// Extract address probabilities from quantum state
    fn extract_address_probabilities(&self, state: &QState, depth: usize) -> Vec<QueryResult> {
        let mut results = Vec::with_capacity(self.tree.size());

        for addr in 0..self.tree.size() {
            // Calculate probability of measuring this address
            // Sum over all basis states consistent with this address pattern
            let prob = self.calculate_address_probability(state, addr, depth);

            if prob > 1e-10 {
                results.push(QueryResult {
                    address: addr,
                    data: self.tree.get(addr).unwrap_or(0),
                    probability: prob,
                });
            }
        }

        results
    }

    /// Calculate probability of measuring a specific address
    ///
    /// The address is encoded in the first `depth` qubits.
    /// We sum probabilities over all basis states where those qubits match.
    fn calculate_address_probability(&self, state: &QState, address: usize, depth: usize) -> f64 {
        let mut total_prob = 0.0;
        let n_qubits = self.tree.total_qubits();
        let non_address_qubits = n_qubits - depth;

        // Access the real and imaginary parts of the state
        let real = state.real.as_slice();
        let imag = state.imag.as_slice();

        // Iterate over all combinations of non-address qubits
        for extra in 0..(1usize << non_address_qubits) {
            // Build full basis state: address in low bits, extra in high bits
            let basis_state = address | (extra << depth);

            if basis_state < state.len {
                let re = real[basis_state];
                let im = imag[basis_state];
                total_prob += (re * re + im * im) as f64;
            }
        }

        total_prob
    }

    // =========================================================================
    // Measurement-Based Query
    // =========================================================================

    /// Query and measure the result
    ///
    /// Performs a superposition query and collapses to a single result
    /// via measurement.
    ///
    /// # Arguments
    /// * `seed` - RNG seed for simulation and measurement
    ///
    /// # Returns
    /// Single query result from measurement
    pub fn query_and_measure(&mut self, seed: u64) -> QueryResult {
        let depth = self.tree.depth();
        let n_qubits = self.tree.total_qubits();

        let mut state = QState::new_zero(n_qubits as u8);
        let mut rng = QRng::new(seed);

        // Superposition initialization
        for gate in self.tree.init_superposition_circuit() {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }

        // Apply QRAM circuit
        for gate in self.tree.query_circuit() {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }

        // Measure address qubits
        let mut measured_address = 0usize;
        for q in 0..depth {
            let outcome = apply_gate_scalar(&mut state, &QGate::Measure(q as u8), &mut rng);
            if let GateOutcome::Measured { qubit: _, bit } = outcome {
                if bit != 0 {
                    // Address is stored MSB first in our convention
                    measured_address |= 1 << (depth - 1 - q);
                }
            }
        }

        let data = self.tree.get(measured_address).unwrap_or(0);

        QueryResult {
            address: measured_address,
            data,
            probability: 1.0 / (self.tree.size() as f64), // Expected probability
        }
    }

    /// Run multiple queries and collect statistics
    ///
    /// Useful for verifying uniform distribution in superposition queries.
    ///
    /// # Arguments
    /// * `trials` - Number of measurement trials
    /// * `base_seed` - Starting seed (incremented for each trial)
    ///
    /// # Returns
    /// Vector of (address, count) pairs
    pub fn query_statistics(&mut self, trials: usize, base_seed: u64) -> Vec<(usize, usize)> {
        let mut counts = vec![0usize; self.tree.size()];

        for trial in 0..trials {
            let result = self.query_and_measure(base_seed.wrapping_add(trial as u64));
            if result.address < counts.len() {
                counts[result.address] += 1;
            }
        }

        counts
            .into_iter()
            .enumerate()
            .filter(|(_, count)| *count > 0)
            .collect()
    }

    // =========================================================================
    // Circuit Analysis
    // =========================================================================

    /// Get the QRAM query circuit
    pub fn circuit(&self) -> Vec<QGate> {
        self.tree.query_circuit()
    }

    /// Get circuit gate count
    pub fn gate_count(&self) -> usize {
        self.tree.gate_count()
    }

    /// Get circuit depth estimate
    pub fn circuit_depth(&self) -> usize {
        self.tree.circuit_depth()
    }
}

// =============================================================================
// Display Implementation
// =============================================================================

impl std::fmt::Display for QRAMQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QRAMQuery(depth={}, cells={}, qubits={}, gates={})",
            self.depth(),
            self.size(),
            self.total_qubits(),
            self.gate_count()
        )
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_creation() {
        let qram = QRAMQuery::new(3);
        assert_eq!(qram.size(), 8);
        assert_eq!(qram.depth(), 3);
    }

    #[test]
    fn test_query_load_and_get() {
        let mut qram = QRAMQuery::new(2);
        qram.load(&[10, 20, 30, 40]);

        assert_eq!(qram.get(0), Some(10));
        assert_eq!(qram.get(1), Some(20));
        assert_eq!(qram.get(2), Some(30));
        assert_eq!(qram.get(3), Some(40));
        assert_eq!(qram.get(4), None);
    }

    #[test]
    fn test_query_set() {
        let mut qram = QRAMQuery::new(2);
        qram.set(2, 999);
        assert_eq!(qram.get(2), Some(999));
    }

    #[test]
    fn test_classical_query() {
        let mut qram = QRAMQuery::new(3);
        qram.load(&[100, 200, 300, 400, 500, 600, 700, 800]);

        for addr in 0..8 {
            let result = qram.query_classical(addr);
            assert_eq!(result, Some((addr as u64 + 1) * 100));
        }

        // Out of bounds
        assert_eq!(qram.query_classical(8), None);
    }

    #[test]
    fn test_basis_query() {
        let mut qram = QRAMQuery::new(2);
        qram.load(&[42, 17, 99, 255]);

        let result = qram.query_basis(2, 12345).unwrap();
        assert_eq!(result.address, 2);
        assert_eq!(result.data, 99);
        assert!((result.probability - 1.0).abs() < 1e-6);

        // Out of bounds
        assert!(qram.query_basis(4, 12345).is_none());
    }

    #[test]
    fn test_superposition_query_depth_1() {
        let mut qram = QRAMQuery::new(1);
        qram.load(&[10, 20]);

        let results = qram.query_superposition(42);

        // Should have 2 results
        assert_eq!(results.len(), 2);

        // Each should have ~50% probability
        let total_prob: f64 = results.iter().map(|r| r.probability).sum();
        assert!(
            (total_prob - 1.0).abs() < 0.01,
            "Total prob: {}",
            total_prob
        );

        for result in &results {
            assert!(
                (result.probability - 0.5).abs() < 0.1,
                "Prob {} should be ~0.5",
                result.probability
            );
        }
    }

    #[test]
    fn test_superposition_query_depth_2() {
        let mut qram = QRAMQuery::new(2);
        qram.load(&[10, 20, 30, 40]);

        let results = qram.query_superposition(42);

        // Should have 4 results
        assert_eq!(results.len(), 4);

        // Total probability should be 1
        let total_prob: f64 = results.iter().map(|r| r.probability).sum();
        assert!(
            (total_prob - 1.0).abs() < 0.01,
            "Total prob: {}",
            total_prob
        );

        // Each should have ~25% probability
        for result in &results {
            assert!(
                (result.probability - 0.25).abs() < 0.1,
                "Address {} prob {} should be ~0.25",
                result.address,
                result.probability
            );
        }

        // Verify data values
        for result in &results {
            let expected = (result.address as u64 + 1) * 10;
            assert_eq!(result.data, expected);
        }
    }

    #[test]
    fn test_query_and_measure() {
        let mut qram = QRAMQuery::new(2);
        qram.load(&[10, 20, 30, 40]);

        let result = qram.query_and_measure(42);

        // Should return a valid address
        assert!(result.address < 4);

        // Data should match
        let expected = (result.address as u64 + 1) * 10;
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_query_statistics() {
        let mut qram = QRAMQuery::new(2);
        qram.load(&[10, 20, 30, 40]);

        let stats = qram.query_statistics(100, 0);

        // Should have results for some addresses
        // Some might be 0 by chance, so just check total
        let total: usize = stats.iter().map(|(_, count)| *count).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_custom_query() {
        let mut qram = QRAMQuery::new(2);
        qram.load(&[10, 20, 30, 40]);

        // Prepare address 3 (0b11) manually
        let init_gates = vec![QGate::X(0), QGate::X(1)];
        let results = qram.query_custom(&init_gates, 42);

        // Should have one result with high probability at address 3
        assert!(!results.is_empty());

        // Find address 3 result
        let addr3 = results.iter().find(|r| r.address == 3);
        assert!(addr3.is_some());
        assert!(addr3.unwrap().probability > 0.9);
    }

    #[test]
    fn test_circuit_metrics() {
        let qram = QRAMQuery::new(3);

        assert_eq!(qram.gate_count(), 21); // 7 routers * 3 gates
        assert_eq!(qram.circuit_depth(), 9); // 3 * depth
        assert_eq!(qram.total_qubits(), 18);
    }

    #[test]
    fn test_display() {
        let qram = QRAMQuery::new(3);
        let display = format!("{}", qram);

        assert!(display.contains("QRAMQuery"));
        assert!(display.contains("depth=3"));
        assert!(display.contains("cells=8"));
    }

    #[test]
    fn test_query_result_new() {
        let result = QueryResult::new(5, 42, 0.25);
        assert_eq!(result.address, 5);
        assert_eq!(result.data, 42);
        assert_eq!(result.probability, 0.25);
    }

    #[test]
    fn test_empty_data_query() {
        let mut qram = QRAMQuery::new(2);
        // Don't load any data - all zeros

        let result = qram.query_classical(0);
        assert_eq!(result, Some(0));

        let result = qram.query_basis(1, 42);
        assert!(result.is_some());
        assert_eq!(result.unwrap().data, 0);
    }

    #[test]
    fn test_partial_load() {
        let mut qram = QRAMQuery::new(3); // 8 cells
        qram.load(&[100, 200]); // Only load 2

        assert_eq!(qram.query_classical(0), Some(100));
        assert_eq!(qram.query_classical(1), Some(200));
        assert_eq!(qram.query_classical(2), Some(0)); // Default
        assert_eq!(qram.query_classical(7), Some(0)); // Default
    }
}
