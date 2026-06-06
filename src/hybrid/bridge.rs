//! Data bridge for inter-substrate communication.
//!
//! The DataBridge handles data transfer between different substrates
//! (classical, quantum, observer). It provides type-safe conversion
//! and caching of intermediate results.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::tile8::sparse_quantum_bigint::Complex64;

/// Type of data in the bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BridgeDataType {
    /// Floating point values
    Float,
    /// Integer values
    Int,
    /// Complex amplitudes
    Complex,
    /// Probability distribution
    Probability,
    /// Measurement counts
    Counts,
    /// Raw bytes
    Bytes,
    /// Parameter values for variational circuits
    Parameters,
}

/// Data stored in the bridge.
#[derive(Clone, Debug)]
pub enum BridgeData {
    /// Floating point values
    Float(Vec<f64>),

    /// Integer values
    Int(Vec<i64>),

    /// Complex amplitudes (state vector)
    Complex(Vec<Complex64>),

    /// Probability distribution (must sum to 1)
    Probability(Vec<f64>),

    /// Measurement counts (bitstring -> count)
    Counts(HashMap<String, usize>),

    /// Raw bytes
    Bytes(Vec<u8>),

    /// Parameter values (theta_0, theta_1, ...)
    Parameters(Vec<f64>),

    /// Empty/no data
    Empty,
}

impl BridgeData {
    /// Get the type of this data.
    pub fn data_type(&self) -> BridgeDataType {
        match self {
            BridgeData::Float(_) => BridgeDataType::Float,
            BridgeData::Int(_) => BridgeDataType::Int,
            BridgeData::Complex(_) => BridgeDataType::Complex,
            BridgeData::Probability(_) => BridgeDataType::Probability,
            BridgeData::Counts(_) => BridgeDataType::Counts,
            BridgeData::Bytes(_) => BridgeDataType::Bytes,
            BridgeData::Parameters(_) => BridgeDataType::Parameters,
            BridgeData::Empty => BridgeDataType::Float, // default
        }
    }

    /// Check if this is empty data.
    pub fn is_empty(&self) -> bool {
        matches!(self, BridgeData::Empty)
    }

    /// Get as float values if possible.
    pub fn as_floats(&self) -> Option<&Vec<f64>> {
        match self {
            BridgeData::Float(v) => Some(v),
            BridgeData::Probability(v) => Some(v),
            BridgeData::Parameters(v) => Some(v),
            _ => None,
        }
    }

    /// Get as integer values if possible.
    pub fn as_ints(&self) -> Option<&Vec<i64>> {
        match self {
            BridgeData::Int(v) => Some(v),
            _ => None,
        }
    }

    /// Get as complex amplitudes if possible.
    pub fn as_complex(&self) -> Option<&Vec<Complex64>> {
        match self {
            BridgeData::Complex(v) => Some(v),
            _ => None,
        }
    }

    /// Get as measurement counts if possible.
    pub fn as_counts(&self) -> Option<&HashMap<String, usize>> {
        match self {
            BridgeData::Counts(c) => Some(c),
            _ => None,
        }
    }

    /// Get as parameters if possible.
    pub fn as_parameters(&self) -> Option<&Vec<f64>> {
        match self {
            BridgeData::Parameters(p) => Some(p),
            _ => None,
        }
    }

    /// Convert counts to probability distribution.
    pub fn counts_to_probability(&self) -> Option<BridgeData> {
        match self {
            BridgeData::Counts(counts) => {
                let total: usize = counts.values().sum();
                if total == 0 {
                    return None;
                }

                // Convert to ordered probabilities (sorted by bitstring)
                let mut sorted: Vec<_> = counts.iter().collect();
                sorted.sort_by_key(|(k, _)| *k);

                let probs: Vec<f64> = sorted
                    .iter()
                    .map(|(_, count)| **count as f64 / total as f64)
                    .collect();

                Some(BridgeData::Probability(probs))
            }
            _ => None,
        }
    }

    /// Convert state vector to probability distribution.
    pub fn state_to_probability(&self) -> Option<BridgeData> {
        match self {
            BridgeData::Complex(amplitudes) => {
                let probs: Vec<f64> = amplitudes
                    .iter()
                    .map(|a| a.re * a.re + a.im * a.im) // |amplitude|^2
                    .collect();
                Some(BridgeData::Probability(probs))
            }
            _ => None,
        }
    }
}

/// Entry in the data bridge with metadata.
#[derive(Clone, Debug)]
pub struct BridgeEntry {
    /// The data
    pub data: BridgeData,

    /// Source substrate type
    pub source: String,

    /// Creation timestamp (ticks since start)
    pub created_at: u64,

    /// Whether this entry has been consumed
    pub consumed: bool,
}

impl BridgeEntry {
    /// Create a new bridge entry.
    pub fn new(data: BridgeData, source: &str, created_at: u64) -> Self {
        Self {
            data,
            source: source.to_string(),
            created_at,
            consumed: false,
        }
    }
}

/// Data bridge for inter-substrate communication.
///
/// The bridge acts as a shared memory space where substrates can
/// publish results and retrieve inputs from other substrates.
///
/// # Example
///
/// ```ignore
/// let bridge = DataBridge::new();
///
/// // Quantum substrate publishes measurement counts
/// bridge.put("quantum_result", BridgeData::Counts(counts));
///
/// // Classical substrate retrieves and converts to probabilities
/// let data = bridge.get("quantum_result").unwrap();
/// let probs = data.counts_to_probability().unwrap();
/// ```
#[derive(Debug)]
pub struct DataBridge {
    /// Stored data entries
    entries: RwLock<HashMap<String, BridgeEntry>>,

    /// Current tick (for timestamps)
    current_tick: RwLock<u64>,

    /// Maximum entries to keep (for memory management)
    max_entries: usize,
}

impl DataBridge {
    /// Create a new data bridge.
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            current_tick: RwLock::new(0),
            max_entries: 1000,
        }
    }

    /// Create a bridge with custom capacity.
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            current_tick: RwLock::new(0),
            max_entries,
        }
    }

    /// Advance the tick counter.
    pub fn tick(&self) {
        let mut tick = self.current_tick.write().unwrap();
        *tick += 1;
    }

    /// Get the current tick.
    pub fn current_tick(&self) -> u64 {
        *self.current_tick.read().unwrap()
    }

    /// Store data in the bridge.
    pub fn put(&self, key: &str, data: BridgeData, source: &str) {
        let tick = self.current_tick();
        let entry = BridgeEntry::new(data, source, tick);

        let mut entries = self.entries.write().unwrap();

        // If at capacity, remove oldest consumed entries
        if entries.len() >= self.max_entries {
            let to_remove: Vec<String> = entries
                .iter()
                .filter(|(_, e)| e.consumed)
                .map(|(k, _)| k.clone())
                .take(self.max_entries / 4)
                .collect();

            for key in to_remove {
                entries.remove(&key);
            }
        }

        entries.insert(key.to_string(), entry);
    }

    /// Retrieve data from the bridge.
    pub fn get(&self, key: &str) -> Option<BridgeData> {
        let entries = self.entries.read().unwrap();
        entries.get(key).map(|e| e.data.clone())
    }

    /// Retrieve and mark as consumed.
    pub fn consume(&self, key: &str) -> Option<BridgeData> {
        let mut entries = self.entries.write().unwrap();
        entries.get_mut(key).map(|e| {
            e.consumed = true;
            e.data.clone()
        })
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &str) -> bool {
        let entries = self.entries.read().unwrap();
        entries.contains_key(key)
    }

    /// Remove an entry.
    pub fn remove(&self, key: &str) -> Option<BridgeData> {
        let mut entries = self.entries.write().unwrap();
        entries.remove(key).map(|e| e.data)
    }

    /// List all keys.
    pub fn keys(&self) -> Vec<String> {
        let entries = self.entries.read().unwrap();
        entries.keys().cloned().collect()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        let mut entries = self.entries.write().unwrap();
        entries.clear();
    }

    /// Get entry metadata.
    pub fn get_entry(&self, key: &str) -> Option<BridgeEntry> {
        let entries = self.entries.read().unwrap();
        entries.get(key).cloned()
    }

    /// Get number of entries.
    pub fn len(&self) -> usize {
        let entries = self.entries.read().unwrap();
        entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        let entries = self.entries.read().unwrap();
        entries.is_empty()
    }
}

impl Default for DataBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for DataBridge {
    fn clone(&self) -> Self {
        let entries = self.entries.read().unwrap();
        let tick = *self.current_tick.read().unwrap();

        Self {
            entries: RwLock::new(entries.clone()),
            current_tick: RwLock::new(tick),
            max_entries: self.max_entries,
        }
    }
}

/// Convenience methods for common data transfers.
impl DataBridge {
    /// Store measurement counts.
    pub fn put_counts(&self, key: &str, counts: HashMap<String, usize>, source: &str) {
        self.put(key, BridgeData::Counts(counts), source);
    }

    /// Store parameters.
    pub fn put_parameters(&self, key: &str, params: Vec<f64>, source: &str) {
        self.put(key, BridgeData::Parameters(params), source);
    }

    /// Store state vector.
    pub fn put_state(&self, key: &str, state: Vec<Complex64>, source: &str) {
        self.put(key, BridgeData::Complex(state), source);
    }

    /// Store float values.
    pub fn put_floats(&self, key: &str, values: Vec<f64>, source: &str) {
        self.put(key, BridgeData::Float(values), source);
    }

    /// Get parameters.
    pub fn get_parameters(&self, key: &str) -> Option<Vec<f64>> {
        self.get(key).and_then(|d| d.as_parameters().cloned())
    }

    /// Get counts.
    pub fn get_counts(&self, key: &str) -> Option<HashMap<String, usize>> {
        self.get(key).and_then(|d| d.as_counts().cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_data_type() {
        let float_data = BridgeData::Float(vec![1.0, 2.0]);
        assert_eq!(float_data.data_type(), BridgeDataType::Float);

        let param_data = BridgeData::Parameters(vec![0.5, 1.5]);
        assert_eq!(param_data.data_type(), BridgeDataType::Parameters);
    }

    #[test]
    fn test_bridge_put_get() {
        let bridge = DataBridge::new();

        bridge.put("test", BridgeData::Float(vec![1.0, 2.0, 3.0]), "quantum");

        let retrieved = bridge.get("test").unwrap();
        let values = retrieved.as_floats().unwrap();
        assert_eq!(values, &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_bridge_consume() {
        let bridge = DataBridge::new();

        bridge.put("test", BridgeData::Int(vec![1, 2, 3]), "classical");

        let entry_before = bridge.get_entry("test").unwrap();
        assert!(!entry_before.consumed);

        let _ = bridge.consume("test");

        let entry_after = bridge.get_entry("test").unwrap();
        assert!(entry_after.consumed);
    }

    #[test]
    fn test_counts_to_probability() {
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 500);
        counts.insert("11".to_string(), 500);

        let data = BridgeData::Counts(counts);
        let probs = data.counts_to_probability().unwrap();

        if let BridgeData::Probability(p) = probs {
            assert_eq!(p.len(), 2);
            assert!((p[0] - 0.5).abs() < 1e-10);
            assert!((p[1] - 0.5).abs() < 1e-10);
        } else {
            panic!("Expected probability data");
        }
    }

    #[test]
    fn test_state_to_probability() {
        let state = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        let data = BridgeData::Complex(state);
        let probs = data.state_to_probability().unwrap();

        if let BridgeData::Probability(p) = probs {
            assert_eq!(p.len(), 4);
            assert!((p[0] - 0.5).abs() < 1e-10);
            assert!((p[1] - 0.0).abs() < 1e-10);
            assert!((p[2] - 0.0).abs() < 1e-10);
            assert!((p[3] - 0.5).abs() < 1e-10);
        } else {
            panic!("Expected probability data");
        }
    }

    #[test]
    fn test_bridge_tick() {
        let bridge = DataBridge::new();

        assert_eq!(bridge.current_tick(), 0);

        bridge.tick();
        bridge.tick();

        assert_eq!(bridge.current_tick(), 2);
    }

    #[test]
    fn test_bridge_keys() {
        let bridge = DataBridge::new();

        bridge.put("a", BridgeData::Empty, "test");
        bridge.put("b", BridgeData::Empty, "test");
        bridge.put("c", BridgeData::Empty, "test");

        let keys = bridge.keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
        assert!(keys.contains(&"c".to_string()));
    }

    #[test]
    fn test_bridge_convenience_methods() {
        let bridge = DataBridge::new();

        bridge.put_parameters("theta", vec![0.1, 0.2, 0.3], "optimizer");

        let params = bridge.get_parameters("theta").unwrap();
        assert_eq!(params, vec![0.1, 0.2, 0.3]);
    }
}
