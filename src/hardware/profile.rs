//! Hardware Device Profiles
//!
//! SPRINT 52.0 Phase 2: Hardware Profiles
//!
//! Defines hardware capability specifications for various quantum devices:
//! - Gate times and fidelities
//! - Coherence times (T1, T2)
//! - SWAP costs
//! - Device-specific constraints
//!
//! # Example
//!
//! ```ignore
//! use engine::hardware::{HardwareProfile, profiles};
//!
//! let ibm = profiles::ibm_falcon();
//! println!("2Q fidelity: {:.4}", ibm.t2q_fidelity);
//! println!("SWAP cost: {} 2Q gates", ibm.swap_cost);
//! ```

use std::fmt;

use super::topology::{DeviceTopology, EnhancedCouplingMap};

/// Hardware device profile with physical specifications
#[derive(Clone, Debug)]
pub struct HardwareProfile {
    /// Device name
    pub name: String,
    /// Device vendor
    pub vendor: String,
    /// Physical topology
    pub topology: DeviceTopology,
    /// Qubit count
    pub qubit_count: usize,

    // === Gate Times (nanoseconds) ===
    /// Single-qubit gate time (ns)
    pub t1q_time_ns: f64,
    /// Two-qubit gate time (ns)
    pub t2q_time_ns: f64,
    /// Measurement time (ns)
    pub measure_time_ns: f64,
    /// Reset time (ns)
    pub reset_time_ns: f64,

    // === Gate Fidelities ===
    /// Single-qubit gate fidelity (0-1)
    pub t1q_fidelity: f64,
    /// Two-qubit gate fidelity (0-1)
    pub t2q_fidelity: f64,
    /// Measurement fidelity (0-1)
    pub measure_fidelity: f64,
    /// State preparation fidelity (0-1)
    pub prep_fidelity: f64,

    // === Coherence Times (microseconds) ===
    /// T1 relaxation time (µs)
    pub t1_us: f64,
    /// T2 dephasing time (µs)
    pub t2_us: f64,

    // === SWAP and Routing ===
    /// SWAP gate cost in native 2Q gates (typically 3 CNOTs)
    pub swap_cost: usize,
    /// Whether device supports native SWAP
    pub native_swap: bool,

    // === Device Class ===
    /// Technology type
    pub technology: DeviceTechnology,
}

/// Quantum computing technology type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceTechnology {
    /// Superconducting transmon qubits
    Superconducting,
    /// Trapped ion qubits
    TrappedIon,
    /// Neutral atom arrays
    NeutralAtom,
    /// Photonic qubits
    Photonic,
    /// Spin qubits (silicon, diamond NV)
    Spin,
    /// Topological qubits
    Topological,
    /// Simulation/ideal device
    Simulation,
}

impl HardwareProfile {
    /// Create a new hardware profile
    pub fn new(name: &str, vendor: &str, topology: DeviceTopology) -> Self {
        let qubit_count = topology.qubit_count();
        Self {
            name: name.to_string(),
            vendor: vendor.to_string(),
            topology,
            qubit_count,
            // Default values (override with builder methods)
            t1q_time_ns: 30.0,
            t2q_time_ns: 300.0,
            measure_time_ns: 500.0,
            reset_time_ns: 500.0,
            t1q_fidelity: 0.999,
            t2q_fidelity: 0.99,
            measure_fidelity: 0.99,
            prep_fidelity: 0.99,
            t1_us: 100.0,
            t2_us: 50.0,
            swap_cost: 3,
            native_swap: false,
            technology: DeviceTechnology::Superconducting,
        }
    }

    // === Builder Methods ===

    /// Set gate times
    pub fn with_gate_times(mut self, t1q_ns: f64, t2q_ns: f64, measure_ns: f64) -> Self {
        self.t1q_time_ns = t1q_ns;
        self.t2q_time_ns = t2q_ns;
        self.measure_time_ns = measure_ns;
        self
    }

    /// Set gate fidelities
    pub fn with_fidelities(mut self, t1q: f64, t2q: f64, measure: f64) -> Self {
        self.t1q_fidelity = t1q;
        self.t2q_fidelity = t2q;
        self.measure_fidelity = measure;
        self
    }

    /// Set coherence times
    pub fn with_coherence(mut self, t1_us: f64, t2_us: f64) -> Self {
        self.t1_us = t1_us;
        self.t2_us = t2_us;
        self
    }

    /// Set SWAP configuration
    pub fn with_swap(mut self, cost: usize, native: bool) -> Self {
        self.swap_cost = cost;
        self.native_swap = native;
        self
    }

    /// Set technology type
    pub fn with_technology(mut self, tech: DeviceTechnology) -> Self {
        self.technology = tech;
        self
    }

    // === Derived Metrics ===

    /// Get coupling map for this device
    pub fn coupling_map(&self) -> EnhancedCouplingMap {
        self.topology.to_coupling_map()
    }

    /// Calculate error per SWAP gate
    pub fn swap_error(&self) -> f64 {
        1.0 - self.t2q_fidelity.powi(self.swap_cost as i32)
    }

    /// Calculate circuit cycle time (typical depth-1 layer)
    pub fn cycle_time_ns(&self) -> f64 {
        // A cycle typically includes 2Q gate + measurement feedback
        self.t2q_time_ns
    }

    /// Estimate circuit fidelity
    ///
    /// Simple model: F = F_1q^n1q * F_2q^n2q * F_m^nm
    pub fn estimate_fidelity(&self, n_1q: usize, n_2q: usize, n_measure: usize) -> f64 {
        self.t1q_fidelity.powi(n_1q as i32)
            * self.t2q_fidelity.powi(n_2q as i32)
            * self.measure_fidelity.powi(n_measure as i32)
    }

    /// Estimate circuit time in nanoseconds
    pub fn estimate_time_ns(&self, depth_1q: usize, depth_2q: usize, n_measure: usize) -> f64 {
        (depth_1q as f64 * self.t1q_time_ns)
            + (depth_2q as f64 * self.t2q_time_ns)
            + (n_measure as f64 * self.measure_time_ns)
    }

    /// Check if circuit fits within coherence time
    pub fn fits_in_coherence(&self, time_ns: f64) -> bool {
        time_ns < self.t2_us * 1000.0 // Convert T2 to ns
    }

    /// Calculate decoherence error for given circuit time
    pub fn decoherence_error(&self, time_ns: f64) -> f64 {
        let t2_ns = self.t2_us * 1000.0;
        1.0 - (-time_ns / t2_ns).exp()
    }

    /// Calculate total SWAP overhead for a logical gate on non-adjacent qubits
    pub fn swap_overhead(&self, q1: usize, q2: usize) -> SwapOverhead {
        let coupling = self.coupling_map();
        let swap_count = coupling.swap_count(q1, q2);
        let gates = swap_count * self.swap_cost;
        let time_ns = gates as f64 * self.t2q_time_ns;
        let error = 1.0 - self.t2q_fidelity.powi(gates as i32);

        SwapOverhead {
            swap_count,
            gate_count: gates,
            time_ns,
            error,
        }
    }

    /// Check if device has full (all-to-all) connectivity
    pub fn has_full_connectivity(&self) -> bool {
        self.topology.has_full_connectivity()
    }

    /// Get maximum distance between any two qubits
    pub fn max_routing_distance(&self) -> usize {
        self.coupling_map().diameter()
    }

    /// Calculate average SWAP overhead per 2Q gate (for random qubit pairs)
    pub fn average_swap_overhead(&self) -> f64 {
        let coupling = self.coupling_map();
        let n = self.qubit_count;
        if n < 2 {
            return 0.0;
        }

        let mut total_swaps = 0usize;
        let mut pairs = 0usize;

        for i in 0..n {
            for j in (i + 1)..n {
                total_swaps += coupling.swap_count(i, j);
                pairs += 1;
            }
        }

        if pairs > 0 {
            (total_swaps as f64 / pairs as f64) * self.swap_cost as f64
        } else {
            0.0
        }
    }
}

impl fmt::Display for HardwareProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({}, {} qubits, {:.2}% 2Q fidelity)",
            self.name,
            self.vendor,
            self.qubit_count,
            self.t2q_fidelity * 100.0
        )
    }
}

/// SWAP overhead calculation result
#[derive(Clone, Debug)]
pub struct SwapOverhead {
    /// Number of SWAP gates needed
    pub swap_count: usize,
    /// Total 2Q gates (SWAP * swap_cost)
    pub gate_count: usize,
    /// Time in nanoseconds
    pub time_ns: f64,
    /// Error probability
    pub error: f64,
}

impl fmt::Display for SwapOverhead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SwapOverhead({} SWAPs, {} gates, {:.1}ns, {:.2}% error)",
            self.swap_count,
            self.gate_count,
            self.time_ns,
            self.error * 100.0
        )
    }
}

/// Predefined hardware profiles
pub mod profiles {
    use super::*;
    use crate::hardware::topology::common;

    /// IBM Falcon processor (27 qubits)
    ///
    /// Based on IBM Quantum Falcon r5.11 specifications
    pub fn ibm_falcon() -> HardwareProfile {
        HardwareProfile::new("Falcon", "IBM", common::ibm_falcon())
            .with_gate_times(35.0, 300.0, 500.0)
            .with_fidelities(0.9996, 0.995, 0.97)
            .with_coherence(100.0, 80.0)
            .with_swap(3, false)
            .with_technology(DeviceTechnology::Superconducting)
    }

    /// IBM Heron processor (133 qubits)
    ///
    /// Based on IBM Quantum Heron specifications (2024)
    pub fn ibm_heron() -> HardwareProfile {
        HardwareProfile::new("Heron", "IBM", common::ibm_heron())
            .with_gate_times(30.0, 68.0, 400.0) // CZ gate is faster
            .with_fidelities(0.9998, 0.997, 0.98)
            .with_coherence(200.0, 150.0)
            .with_swap(3, false)
            .with_technology(DeviceTechnology::Superconducting)
    }

    /// Google Sycamore processor (53 qubits)
    ///
    /// Based on Google Quantum AI specifications
    pub fn google_sycamore() -> HardwareProfile {
        HardwareProfile::new("Sycamore", "Google", common::google_sycamore())
            .with_gate_times(25.0, 32.0, 400.0) // Fast gates
            .with_fidelities(0.9996, 0.995, 0.96)
            .with_coherence(15.0, 12.0) // Shorter coherence
            .with_swap(3, false)
            .with_technology(DeviceTechnology::Superconducting)
    }

    /// IonQ Aria processor (25 qubits)
    ///
    /// Based on IonQ Aria specifications
    pub fn ionq_aria() -> HardwareProfile {
        HardwareProfile::new("Aria", "IonQ", common::ionq_aria())
            .with_gate_times(15000.0, 200000.0, 100000.0) // Slow gates (µs scale)
            .with_fidelities(0.9998, 0.996, 0.995)
            .with_coherence(100000.0, 1000.0) // Long T1, shorter T2
            .with_swap(3, true) // Native SWAP via shuttling
            .with_technology(DeviceTechnology::TrappedIon)
    }

    /// Quantinuum H2 processor (56 qubits)
    ///
    /// Based on Quantinuum H2 specifications
    pub fn quantinuum_h2() -> HardwareProfile {
        HardwareProfile::new("H2", "Quantinuum", common::quantinuum_h2())
            .with_gate_times(10000.0, 300000.0, 50000.0)
            .with_fidelities(0.99995, 0.998, 0.997)
            .with_coherence(1000000.0, 5000.0) // Very long T1
            .with_swap(3, true)
            .with_technology(DeviceTechnology::TrappedIon)
    }

    /// QuEra Aquila processor (256 atoms)
    ///
    /// Based on QuEra Aquila specifications
    pub fn quera_aquila() -> HardwareProfile {
        HardwareProfile::new("Aquila", "QuEra", common::quera_aquila())
            .with_gate_times(100.0, 500.0, 1000.0) // Rydberg gates
            .with_fidelities(0.995, 0.98, 0.95)
            .with_coherence(10.0, 5.0) // Short coherence
            .with_swap(3, false)
            .with_technology(DeviceTechnology::NeutralAtom)
    }

    /// Ideal/simulation device (configurable qubits)
    pub fn ideal(qubit_count: usize) -> HardwareProfile {
        let topology = DeviceTopology::fully_connected(qubit_count);
        HardwareProfile::new("Ideal", "Simulation", topology)
            .with_gate_times(1.0, 1.0, 1.0)
            .with_fidelities(1.0, 1.0, 1.0)
            .with_coherence(f64::INFINITY, f64::INFINITY)
            .with_swap(0, true)
            .with_technology(DeviceTechnology::Simulation)
    }

    /// Create custom superconducting device
    pub fn custom_superconducting(
        name: &str,
        width: usize,
        height: usize,
        t2q_fidelity: f64,
    ) -> HardwareProfile {
        let topology = DeviceTopology::grid_2d(width, height);
        HardwareProfile::new(name, "Custom", topology)
            .with_gate_times(30.0, 300.0, 500.0)
            .with_fidelities(0.999, t2q_fidelity, 0.97)
            .with_coherence(100.0, 80.0)
            .with_swap(3, false)
            .with_technology(DeviceTechnology::Superconducting)
    }

    /// Create custom ion trap device
    pub fn custom_ion_trap(name: &str, qubit_count: usize, t2q_fidelity: f64) -> HardwareProfile {
        let topology = DeviceTopology::fully_connected(qubit_count);
        HardwareProfile::new(name, "Custom", topology)
            .with_gate_times(15000.0, 200000.0, 100000.0)
            .with_fidelities(0.9998, t2q_fidelity, 0.995)
            .with_coherence(100000.0, 1000.0)
            .with_swap(3, true)
            .with_technology(DeviceTechnology::TrappedIon)
    }
}

/// Compare multiple hardware profiles
pub fn compare_profiles(profiles: &[HardwareProfile]) -> ProfileComparison {
    let comparisons: Vec<_> = profiles
        .iter()
        .map(|p| ProfileMetrics {
            name: p.name.clone(),
            qubit_count: p.qubit_count,
            t2q_fidelity: p.t2q_fidelity,
            cycle_time_ns: p.cycle_time_ns(),
            swap_overhead: p.average_swap_overhead(),
            max_distance: p.max_routing_distance(),
        })
        .collect();

    ProfileComparison {
        profiles: comparisons,
    }
}

/// Metrics for profile comparison
#[derive(Clone, Debug)]
pub struct ProfileMetrics {
    pub name: String,
    pub qubit_count: usize,
    pub t2q_fidelity: f64,
    pub cycle_time_ns: f64,
    pub swap_overhead: f64,
    pub max_distance: usize,
}

/// Profile comparison result
#[derive(Clone, Debug)]
pub struct ProfileComparison {
    pub profiles: Vec<ProfileMetrics>,
}

impl fmt::Display for ProfileComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:>15} {:>8} {:>10} {:>12} {:>10} {:>8}",
            "Device", "Qubits", "2Q Fid", "Cycle(ns)", "SWAP Ovhd", "MaxDist"
        )?;
        writeln!(f, "{}", "-".repeat(70))?;
        for p in &self.profiles {
            writeln!(
                f,
                "{:>15} {:>8} {:>10.4} {:>12.1} {:>10.2} {:>8}",
                p.name,
                p.qubit_count,
                p.t2q_fidelity,
                p.cycle_time_ns,
                p.swap_overhead,
                p.max_distance
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibm_falcon() {
        let profile = profiles::ibm_falcon();
        assert_eq!(profile.name, "Falcon");
        assert!(profile.qubit_count >= 25);
        assert!(profile.t2q_fidelity > 0.99);
        assert_eq!(profile.swap_cost, 3);
    }

    #[test]
    fn test_ionq_aria() {
        let profile = profiles::ionq_aria();
        assert_eq!(profile.qubit_count, 25);
        assert!(profile.has_full_connectivity());
        assert_eq!(profile.max_routing_distance(), 1); // All-to-all
    }

    #[test]
    fn test_swap_overhead() {
        let profile = profiles::ibm_falcon();
        let coupling = profile.coupling_map();

        // Adjacent qubits have no SWAP overhead
        if coupling.qubit_count() >= 2 {
            let edges = coupling.edges();
            if !edges.is_empty() {
                let (q1, q2) = edges[0];
                let overhead = profile.swap_overhead(q1, q2);
                assert_eq!(overhead.swap_count, 0);
            }
        }
    }

    #[test]
    fn test_fidelity_estimation() {
        let profile = profiles::ibm_falcon();
        let fid = profile.estimate_fidelity(10, 5, 1);
        assert!(fid > 0.9);
        assert!(fid < 1.0);
    }

    #[test]
    fn test_ideal_device() {
        let ideal = profiles::ideal(10);
        assert_eq!(ideal.t1q_fidelity, 1.0);
        assert_eq!(ideal.t2q_fidelity, 1.0);
        assert!(ideal.has_full_connectivity());
        assert_eq!(ideal.swap_cost, 0);
    }

    #[test]
    fn test_compare_profiles() {
        let profiles = vec![
            profiles::ibm_falcon(),
            profiles::ionq_aria(),
            profiles::google_sycamore(),
        ];
        let comparison = compare_profiles(&profiles);
        assert_eq!(comparison.profiles.len(), 3);
    }

    #[test]
    fn test_technology_types() {
        assert_eq!(
            profiles::ibm_falcon().technology,
            DeviceTechnology::Superconducting
        );
        assert_eq!(
            profiles::ionq_aria().technology,
            DeviceTechnology::TrappedIon
        );
        assert_eq!(
            profiles::quera_aquila().technology,
            DeviceTechnology::NeutralAtom
        );
    }

    #[test]
    fn test_display() {
        let profile = profiles::ibm_falcon();
        let s = format!("{}", profile);
        assert!(s.contains("Falcon"));
        assert!(s.contains("IBM"));
    }

    #[test]
    fn test_average_swap_overhead() {
        let grid = profiles::custom_superconducting("Grid4x4", 4, 4, 0.99);
        let ion = profiles::ionq_aria();

        // Grid should have non-zero SWAP overhead
        assert!(grid.average_swap_overhead() > 0.0);

        // Ion trap (all-to-all) should have zero SWAP overhead
        assert_eq!(ion.average_swap_overhead(), 0.0);
    }
}
