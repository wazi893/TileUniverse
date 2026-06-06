// EPIC 78: Unified COPS (Computational Operations Per Second) Metric System
//
// This module provides standardized performance metrics across quantum and classical
// subsystems, enabling apples-to-apples comparisons and aggregate throughput reporting.

/// Computational Operations Per Second - Unified metric across quantum and classical
#[derive(Debug, Clone, Copy)]
pub struct CopsMetric {
    /// Base operations per second (gate ops or tile evals)
    pub raw_ops_per_sec: f64,

    /// Amplification factor (amplitudes, lanes, worlds, etc.)
    pub amplification: u64,

    /// Total COPS (raw_ops × amplification)
    pub cops: f64,
}

impl CopsMetric {
    /// Create COPS metric for quantum operations
    ///
    /// Each quantum gate touches all 2^n amplitudes, so:
    /// COPS = gate_ops_per_sec × (2^n_qubits)
    pub fn quantum(gate_ops_per_sec: f64, n_qubits: u8) -> Self {
        let amplitudes = 1u64 << n_qubits;
        let cops = gate_ops_per_sec * amplitudes as f64;
        Self {
            raw_ops_per_sec: gate_ops_per_sec,
            amplification: amplitudes,
            cops,
        }
    }

    /// Create COPS metric for classical tile operations
    ///
    /// Each tile eval processes `lanes` parallel bits across `worlds`:
    /// COPS = tile_evals_per_sec × lanes × worlds
    pub fn classical(tile_evals_per_sec: f64, lanes: u64, worlds: u64) -> Self {
        let amplification = lanes * worlds;
        let cops = tile_evals_per_sec * amplification as f64;
        Self {
            raw_ops_per_sec: tile_evals_per_sec,
            amplification,
            cops,
        }
    }

    /// Get TCOPS (trillions of ops/sec)
    pub fn tcops(&self) -> f64 {
        self.cops / 1e12
    }

    /// Get PCOPS (petaops/sec)
    pub fn pcops(&self) -> f64 {
        self.cops / 1e15
    }

    /// Human-readable display string
    pub fn display(&self) -> String {
        if self.cops >= 1e15 {
            format!("{:.2} PCOPS", self.cops / 1e15)
        } else if self.cops >= 1e12 {
            format!("{:.2} TCOPS", self.cops / 1e12)
        } else if self.cops >= 1e9 {
            format!("{:.2} GCOPS", self.cops / 1e9)
        } else if self.cops >= 1e6 {
            format!("{:.2} MCOPS", self.cops / 1e6)
        } else {
            format!("{:.2} KCOPS", self.cops / 1e3)
        }
    }

    /// Detailed breakdown string
    pub fn breakdown(&self) -> String {
        format!(
            "{:.2}M raw ops/sec × {}× amplification = {}",
            self.raw_ops_per_sec / 1e6,
            self.amplification,
            self.display()
        )
    }
}

/// Complete benchmark report with quantum, classical, and combined metrics
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub name: String,
    pub quantum: Option<CopsMetric>,
    pub classical: Option<CopsMetric>,
    pub combined: Option<CopsMetric>,
    pub elapsed_secs: f64,
    pub hardware: String,
}

impl BenchmarkReport {
    /// Print VC-ready summary (optimized for screenshot/demo)
    pub fn print_vc_summary(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════════════╗");
        println!("║  {} ", self.name.to_uppercase());
        println!("╚══════════════════════════════════════════════════════════════════════╝");
        println!();
        println!("  Hardware: {}", self.hardware);
        println!("  Elapsed:  {:.3}s", self.elapsed_secs);
        println!();

        if let Some(q) = &self.quantum {
            println!("  ┌─ QUANTUM SUBSTRATE ────────────────────────────────────────────┐");
            println!("  │  Gate ops/sec:      {:.2}M", q.raw_ops_per_sec / 1e6);
            println!("  │  Amplitudes:        {}×", q.amplification);
            println!("  │  → THROUGHPUT:      {} ◄", q.display());
            println!("  └────────────────────────────────────────────────────────────────┘");
            println!();
        }

        if let Some(c) = &self.classical {
            println!("  ┌─ CLASSICAL TILE ENGINE ────────────────────────────────────────┐");
            println!("  │  Tile evals/sec:    {:.2}B", c.raw_ops_per_sec / 1e9);
            println!(
                "  │  Amplification:     {}× (lanes × worlds)",
                c.amplification
            );
            println!("  │  → THROUGHPUT:      {} ◄", c.display());
            println!("  └────────────────────────────────────────────────────────────────┘");
            println!();
        }

        if let Some(combined) = &self.combined {
            println!("  ╔════════════════════════════════════════════════════════════════╗");
            println!(
                "  ║  🚀 TOTAL THROUGHPUT: {:>20}                    ║",
                combined.display()
            );
            println!("  ╚════════════════════════════════════════════════════════════════╝");
            println!();
        }
    }

    /// Compact single-line summary
    pub fn one_liner(&self) -> String {
        if let Some(combined) = &self.combined {
            format!(
                "{}: {} ({:.2}s)",
                self.name,
                combined.display(),
                self.elapsed_secs
            )
        } else {
            format!("{}: (no combined metric)", self.name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cops_quantum_12q() {
        // 12 qubits = 4096 amplitudes
        let cops = CopsMetric::quantum(1_000_000.0, 12); // 1M gate ops/sec
        assert_eq!(cops.amplification, 4096);
        assert!((cops.cops - 4.096e9).abs() < 1e6); // ~4.096 billion ops
        assert!((cops.tcops() - 0.004096).abs() < 0.001);
    }

    #[test]
    fn cops_classical_64lane_1world() {
        let cops = CopsMetric::classical(40e9, 64, 1); // 40B evals/sec, 64 lanes, 1 world
        assert_eq!(cops.amplification, 64);
        assert!((cops.cops - 2.56e12).abs() < 1e9); // ~2.56 trillion
        assert!((cops.tcops() - 2.56).abs() < 0.01);
    }

    #[test]
    fn cops_classical_64lane_100worlds() {
        let cops = CopsMetric::classical(40e9, 64, 100); // 100 worlds
        assert_eq!(cops.amplification, 6400);
        assert!((cops.cops - 2.56e14).abs() < 1e11); // ~256 trillion
        assert!((cops.tcops() - 256.0).abs() < 1.0);
    }

    #[test]
    fn display_format() {
        let tcops = CopsMetric::classical(40e9, 64, 100);
        assert!(tcops.display().contains("TCOPS"));

        let pcops = CopsMetric::classical(40e9, 64, 10000); // Petaops scale
        assert!(pcops.display().contains("PCOPS"));
    }
}
