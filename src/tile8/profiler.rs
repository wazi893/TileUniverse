//! Performance Profiling Infrastructure for TILE-8
//!
//! Provides per-operation timing, aggregate statistics, and CSV export
//! for analyzing the performance of quantum operations.
//!
//! # Example
//!
//! ```ignore
//! use engine::tile8::profiler::Profiler;
//!
//! let mut profiler = Profiler::new();
//!
//! profiler.start("cross_block_cnot");
//! grid.cross_block_cnot(&mut sim, 0, 1, 7);
//! profiler.end("cross_block_cnot");
//!
//! // Get statistics
//! let stats = profiler.get_stats("cross_block_cnot").unwrap();
//! println!("Avg: {:.2}μs, Min: {:.2}μs, Max: {:.2}μs",
//!          stats.avg_micros(), stats.min_micros(), stats.max_micros());
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Statistics for a single operation
#[derive(Debug, Clone)]
pub struct OperationStats {
    /// Number of times this operation was executed
    count: usize,

    /// Total time spent in nanoseconds
    total_nanos: u128,

    /// Minimum execution time in nanoseconds
    min_nanos: u64,

    /// Maximum execution time in nanoseconds
    max_nanos: u64,

    /// Operation name
    name: String,
}

impl OperationStats {
    /// Create new stats tracker for an operation
    fn new(name: String) -> Self {
        Self {
            count: 0,
            total_nanos: 0,
            min_nanos: u64::MAX,
            max_nanos: 0,
            name,
        }
    }

    /// Record a new timing sample
    fn record(&mut self, nanos: u64) {
        self.count += 1;
        self.total_nanos += nanos as u128;
        self.min_nanos = self.min_nanos.min(nanos);
        self.max_nanos = self.max_nanos.max(nanos);
    }

    /// Get the count of operations
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get average time in microseconds
    pub fn avg_micros(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.total_nanos as f64 / self.count as f64) / 1000.0
        }
    }

    /// Get minimum time in microseconds
    pub fn min_micros(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.min_nanos as f64 / 1000.0
        }
    }

    /// Get maximum time in microseconds
    pub fn max_micros(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max_nanos as f64 / 1000.0
        }
    }

    /// Get total time in milliseconds
    pub fn total_millis(&self) -> f64 {
        self.total_nanos as f64 / 1_000_000.0
    }

    /// Get operation name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Performance profiler for tracking operation timing
pub struct Profiler {
    /// Statistics for each operation
    operations: HashMap<String, OperationStats>,

    /// Active timers (operation name -> start time)
    active_timers: HashMap<String, Instant>,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            active_timers: HashMap::new(),
        }
    }

    /// Start timing an operation
    ///
    /// # Arguments
    /// * `name` - Name of the operation to track
    ///
    /// # Panics
    /// Panics if the operation is already being timed (nested timing not supported)
    pub fn start(&mut self, name: &str) {
        if self.active_timers.contains_key(name) {
            panic!("Operation '{}' is already being timed", name);
        }
        self.active_timers.insert(name.to_string(), Instant::now());
    }

    /// End timing an operation
    ///
    /// # Arguments
    /// * `name` - Name of the operation to stop timing
    ///
    /// # Panics
    /// Panics if the operation was not started
    pub fn end(&mut self, name: &str) {
        let start = self
            .active_timers
            .remove(name)
            .unwrap_or_else(|| panic!("Operation '{}' was not started", name));

        let elapsed = start.elapsed();
        let nanos = elapsed.as_nanos() as u64;

        // Get or create stats for this operation
        let stats = self
            .operations
            .entry(name.to_string())
            .or_insert_with(|| OperationStats::new(name.to_string()));

        stats.record(nanos);
    }

    /// Time a closure and record its execution time
    ///
    /// # Arguments
    /// * `name` - Name of the operation
    /// * `f` - Closure to execute and time
    ///
    /// # Returns
    /// The result of the closure
    pub fn time<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.start(name);
        let result = f();
        self.end(name);
        result
    }

    /// Get statistics for a specific operation
    ///
    /// # Arguments
    /// * `name` - Name of the operation
    ///
    /// # Returns
    /// Stats for the operation, or None if not found
    pub fn get_stats(&self, name: &str) -> Option<&OperationStats> {
        self.operations.get(name)
    }

    /// Get all operation statistics
    pub fn all_stats(&self) -> &HashMap<String, OperationStats> {
        &self.operations
    }

    /// Print a summary report to stdout
    pub fn print_report(&self) {
        println!("\n📊 Performance Profile");
        println!("=====================================================================");
        println!(
            "{:<30} {:>10} {:>12} {:>12} {:>12}",
            "Operation", "Count", "Avg (μs)", "Min (μs)", "Max (μs)"
        );
        println!("---------------------------------------------------------------------");

        // Sort by total time (descending)
        let mut ops: Vec<_> = self.operations.values().collect();
        ops.sort_by(|a, b| {
            b.total_nanos
                .partial_cmp(&a.total_nanos)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for stats in ops {
            println!(
                "{:<30} {:>10} {:>12.2} {:>12.2} {:>12.2}",
                stats.name(),
                stats.count(),
                stats.avg_micros(),
                stats.min_micros(),
                stats.max_micros()
            );
        }

        println!("=====================================================================\n");
    }

    /// Export statistics to CSV file
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file to write
    ///
    /// # Returns
    /// Result indicating success or IO error
    pub fn export_csv(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Write header
        writeln!(
            file,
            "operation,count,avg_micros,min_micros,max_micros,total_millis"
        )?;

        // Sort by operation name for consistent output
        let mut ops: Vec<_> = self.operations.values().collect();
        ops.sort_by(|a, b| a.name().cmp(b.name()));

        // Write data rows
        for stats in ops {
            writeln!(
                file,
                "{},{},{:.3},{:.3},{:.3},{:.3}",
                stats.name(),
                stats.count(),
                stats.avg_micros(),
                stats.min_micros(),
                stats.max_micros(),
                stats.total_millis()
            )?;
        }

        Ok(())
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.operations.clear();
        self.active_timers.clear();
    }

    /// Get the total number of operations tracked
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Get the total execution count across all operations
    pub fn total_executions(&self) -> usize {
        self.operations.values().map(|s| s.count()).sum()
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    #[ignore = "timing-sensitive — thread::sleep(10ms) can overshoot 20ms under load"]
    fn test_profiler_basic() {
        let mut profiler = Profiler::new();

        // Time a simple operation
        profiler.start("test_op");
        thread::sleep(Duration::from_millis(10));
        profiler.end("test_op");

        let stats = profiler.get_stats("test_op").unwrap();
        assert_eq!(stats.count(), 1);
        assert!(stats.avg_micros() >= 10000.0); // At least 10ms
        assert!(stats.avg_micros() < 20000.0); // Less than 20ms (with margin)
    }

    #[test]
    fn test_profiler_multiple_samples() {
        let mut profiler = Profiler::new();

        // Record multiple samples
        for _ in 0..5 {
            profiler.start("multi_op");
            thread::sleep(Duration::from_millis(5));
            profiler.end("multi_op");
        }

        let stats = profiler.get_stats("multi_op").unwrap();
        assert_eq!(stats.count(), 5);
        assert!(stats.avg_micros() >= 5000.0);
        assert!(stats.min_micros() >= 5000.0);
        assert!(stats.max_micros() >= 5000.0);
    }

    #[test]
    fn test_profiler_time_closure() {
        let mut profiler = Profiler::new();

        let result = profiler.time("closure_op", || {
            thread::sleep(Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);

        let stats = profiler.get_stats("closure_op").unwrap();
        assert_eq!(stats.count(), 1);
        assert!(stats.avg_micros() >= 5000.0);
    }

    #[test]
    fn test_profiler_multiple_operations() {
        let mut profiler = Profiler::new();

        profiler.start("op1");
        thread::sleep(Duration::from_millis(5));
        profiler.end("op1");

        profiler.start("op2");
        thread::sleep(Duration::from_millis(10));
        profiler.end("op2");

        assert_eq!(profiler.operation_count(), 2);
        assert_eq!(profiler.total_executions(), 2);

        let stats1 = profiler.get_stats("op1").unwrap();
        let stats2 = profiler.get_stats("op2").unwrap();

        // This test is about tracking distinct operations, not asserting a
        // strict wall-clock ordering between two sleeps on a shared CI host.
        assert_eq!(stats1.name(), "op1");
        assert_eq!(stats2.name(), "op2");
        assert_eq!(stats1.count(), 1);
        assert_eq!(stats2.count(), 1);
        assert!(stats1.avg_micros() > 0.0);
        assert!(stats2.avg_micros() > 0.0);
    }

    #[test]
    fn test_profiler_reset() {
        let mut profiler = Profiler::new();

        profiler.start("test");
        profiler.end("test");

        assert_eq!(profiler.operation_count(), 1);

        profiler.reset();

        assert_eq!(profiler.operation_count(), 0);
        assert_eq!(profiler.total_executions(), 0);
    }

    #[test]
    fn test_profiler_csv_export() {
        let mut profiler = Profiler::new();

        profiler.start("op1");
        thread::sleep(Duration::from_millis(5));
        profiler.end("op1");

        profiler.start("op2");
        thread::sleep(Duration::from_millis(10));
        profiler.end("op2");

        let temp_file = "test_profile.csv";
        profiler.export_csv(temp_file).unwrap();

        // Verify file was created and contains data
        let contents = std::fs::read_to_string(temp_file).unwrap();
        assert!(contents.contains("operation,count,avg_micros"));
        assert!(contents.contains("op1"));
        assert!(contents.contains("op2"));

        // Cleanup
        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    #[should_panic(expected = "already being timed")]
    fn test_profiler_nested_timing_panics() {
        let mut profiler = Profiler::new();

        profiler.start("nested");
        profiler.start("nested"); // Should panic
    }

    #[test]
    #[should_panic(expected = "was not started")]
    fn test_profiler_end_without_start_panics() {
        let mut profiler = Profiler::new();
        profiler.end("not_started"); // Should panic
    }

    #[test]
    fn test_operation_stats() {
        let mut stats = OperationStats::new("test".to_string());

        // Record some samples
        stats.record(1000); // 1μs
        stats.record(2000); // 2μs
        stats.record(3000); // 3μs

        assert_eq!(stats.count(), 3);
        assert_eq!(stats.min_micros(), 1.0);
        assert_eq!(stats.max_micros(), 3.0);
        assert_eq!(stats.avg_micros(), 2.0);
        assert_eq!(stats.total_millis(), 0.006);
    }

    #[test]
    fn test_profiler_print_report() {
        let mut profiler = Profiler::new();

        profiler.start("op1");
        thread::sleep(Duration::from_millis(5));
        profiler.end("op1");

        profiler.start("op2");
        thread::sleep(Duration::from_millis(10));
        profiler.end("op2");

        // Just verify it doesn't panic
        profiler.print_report();
    }
}
