// =============================================================================
// src/phase_diagram.rs
// =============================================================================
//
// SPRINT 3.3: Phase Diagram Generation
//
// Systematic parameter sweeps to map the quantum advantage landscape.
// Generates heatmaps showing: Quantum Advantage = f(param1, param2)
//
// This turns individual discoveries into a complete picture:
// - Where does quantum help?
// - Where does classical win?
// - Where does everyone die?
//
// =============================================================================

use crate::quantum_vs_classical::{ComparisonConfig, run_quantum_vs_classical};

// =============================================================================
// Phase Diagram Types
// =============================================================================

/// A parameter that can be varied in a sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SweepParameter {
    /// Number of simulation ticks
    Ticks,
    /// Resource scarcity (heat patches per spawn)
    ResourcePatches,
    /// Resource value (heat per patch)
    ResourceValue,
    /// Number of qubits per organism
    Qubits,
    /// Initial circuit length
    CircuitLength,
    /// Maximum circuit length
    MaxCircuitLength,
    /// Mutation rate
    MutationRate,
    /// Initial population
    InitialPopulation,
    /// Reproduction threshold
    ReproductionThreshold,
    /// Movement cost
    MovementCost,
    /// Metabolism cost
    MetabolismCost,
}

/// Range specification for a parameter sweep.
#[derive(Clone, Debug)]
pub struct ParameterRange {
    pub param: SweepParameter,
    pub min: f32,
    pub max: f32,
    pub steps: usize,
}

impl ParameterRange {
    pub fn new(param: SweepParameter, min: f32, max: f32, steps: usize) -> Self {
        Self {
            param,
            min,
            max,
            steps,
        }
    }

    /// Generate values for this range.
    pub fn values(&self) -> Vec<f32> {
        if self.steps <= 1 {
            return vec![self.min];
        }

        let step_size = (self.max - self.min) / (self.steps - 1) as f32;
        (0..self.steps)
            .map(|i| self.min + i as f32 * step_size)
            .collect()
    }
}

/// Result of a single point in the phase diagram.
#[derive(Clone, Debug)]
pub struct PhasePoint {
    pub x_value: f32,
    pub y_value: f32,
    pub quantum_advantage: f32,
    pub quantum_population: f32,
    pub classical_population: f32,
    pub quantum_extinction_rate: f32,
    pub classical_extinction_rate: f32,
    pub quantum_generation: f32,
    pub classical_generation: f32,
}

/// A complete phase diagram (2D parameter sweep).
#[derive(Clone, Debug)]
pub struct PhaseDiagram {
    pub x_param: SweepParameter,
    pub y_param: SweepParameter,
    pub x_values: Vec<f32>,
    pub y_values: Vec<f32>,
    pub points: Vec<Vec<PhasePoint>>, // [y][x] indexing
    pub base_config: ComparisonConfig,
    pub replications: usize,
}

impl PhaseDiagram {
    /// Get the quantum advantage at a specific grid position.
    pub fn advantage_at(&self, x_idx: usize, y_idx: usize) -> Option<f32> {
        self.points
            .get(y_idx)?
            .get(x_idx)
            .map(|p| p.quantum_advantage)
    }

    /// Find the maximum quantum advantage point.
    pub fn max_advantage_point(&self) -> Option<&PhasePoint> {
        self.points
            .iter()
            .flat_map(|row| row.iter())
            .max_by(|a, b| {
                a.quantum_advantage
                    .partial_cmp(&b.quantum_advantage)
                    .unwrap()
            })
    }

    /// Find the minimum quantum advantage point (max classical advantage).
    pub fn min_advantage_point(&self) -> Option<&PhasePoint> {
        self.points
            .iter()
            .flat_map(|row| row.iter())
            .min_by(|a, b| {
                a.quantum_advantage
                    .partial_cmp(&b.quantum_advantage)
                    .unwrap()
            })
    }

    /// Export to CSV format.
    pub fn export_csv(&self) -> String {
        let mut csv = format!(
            "{},{},quantum_adv,q_pop,c_pop,q_extinct,c_extinct,q_gen,c_gen\n",
            param_name(self.x_param),
            param_name(self.y_param),
        );

        for row in &self.points {
            for point in row {
                csv.push_str(&format!(
                    "{:.3},{:.3},{:.2},{:.1},{:.1},{:.2},{:.2},{:.1},{:.1}\n",
                    point.x_value,
                    point.y_value,
                    point.quantum_advantage,
                    point.quantum_population,
                    point.classical_population,
                    point.quantum_extinction_rate,
                    point.classical_extinction_rate,
                    point.quantum_generation,
                    point.classical_generation,
                ));
            }
        }

        csv
    }

    /// Generate ASCII heatmap visualization.
    pub fn ascii_heatmap(&self) -> String {
        let chars = [' ', '░', '▒', '▓', '█'];
        let q_chars = ['─', '○', '●', '◉', '★']; // Quantum wins
        let c_chars = ['─', '□', '■', '◆', '✦']; // Classical wins

        let mut output = String::new();

        // Find range
        let advantages: Vec<f32> = self
            .points
            .iter()
            .flat_map(|row| row.iter().map(|p| p.quantum_advantage))
            .collect();

        let max_adv = advantages.iter().cloned().fold(f32::MIN, f32::max);
        let min_adv = advantages.iter().cloned().fold(f32::MAX, f32::min);
        let range = (max_adv - min_adv).max(1.0);

        output.push_str(&format!(
            "\n{} vs {} Phase Diagram\n",
            param_name(self.x_param),
            param_name(self.y_param)
        ));
        output.push_str(&format!(
            "Quantum advantage range: {:.1} to {:.1}\n\n",
            min_adv, max_adv
        ));

        // Header row
        output.push_str(&format!("{:>8} |", param_name(self.y_param)));
        for x in &self.x_values {
            output.push_str(&format!(" {:>5.1}", x));
        }
        output.push('\n');
        output.push_str(&format!(
            "{:>8} +{}\n",
            "",
            "-".repeat(self.x_values.len() * 6)
        ));

        // Data rows (reverse y for natural orientation)
        for (y_idx, y_val) in self.y_values.iter().enumerate().rev() {
            output.push_str(&format!("{:>8.1} |", y_val));

            for point in &self.points[y_idx] {
                let normalized = (point.quantum_advantage - min_adv) / range;
                let char_idx = (normalized * 4.0).round() as usize;
                let char_idx = char_idx.min(4);

                let symbol = if point.quantum_advantage > 5.0 {
                    q_chars[char_idx]
                } else if point.quantum_advantage < -5.0 {
                    c_chars[char_idx]
                } else {
                    chars[char_idx]
                };

                output.push_str(&format!("    {} ", symbol));
            }
            output.push('\n');
        }

        output.push_str(&format!("\n{:>8}   ", ""));
        for _ in 0..self.x_values.len() {
            output.push_str("------");
        }
        output.push_str(&format!("\n{:>8}   ", param_name(self.x_param)));
        output.push('\n');

        output.push_str("\nLegend: ★/● = Strong Quantum, ✦/■ = Strong Classical, ░▒▓ = Neutral\n");

        output
    }
}

// =============================================================================
// Phase Diagram Generation
// =============================================================================

/// Configuration for generating a phase diagram.
#[derive(Clone, Debug)]
pub struct PhaseDiagramConfig {
    pub x_range: ParameterRange,
    pub y_range: ParameterRange,
    pub base_config: ComparisonConfig,
    pub replications_per_point: usize,
}

impl Default for PhaseDiagramConfig {
    fn default() -> Self {
        Self {
            x_range: ParameterRange::new(SweepParameter::Ticks, 250.0, 2000.0, 8),
            y_range: ParameterRange::new(SweepParameter::ResourcePatches, 10.0, 40.0, 7),
            base_config: ComparisonConfig::default(),
            replications_per_point: 3,
        }
    }
}

/// Generate a complete phase diagram.
pub fn generate_phase_diagram(config: &PhaseDiagramConfig) -> PhaseDiagram {
    let x_values = config.x_range.values();
    let y_values = config.y_range.values();

    let mut points = Vec::with_capacity(y_values.len());

    let total_points = x_values.len() * y_values.len();
    let mut completed = 0;

    for y_val in &y_values {
        let mut row = Vec::with_capacity(x_values.len());

        for x_val in &x_values {
            // Create config with these parameter values
            let mut point_config = config.base_config.clone();
            point_config.replications = config.replications_per_point;

            apply_parameter(&mut point_config, config.x_range.param, *x_val);
            apply_parameter(&mut point_config, config.y_range.param, *y_val);

            // Run the comparison
            let result = run_quantum_vs_classical(&point_config);

            // Extract metrics
            let point = PhasePoint {
                x_value: *x_val,
                y_value: *y_val,
                quantum_advantage: result.summary.population_advantage,
                quantum_population: result.summary.quantum_mean_population,
                classical_population: result.summary.classical_mean_population,
                quantum_extinction_rate: result.summary.quantum_extinction_rate,
                classical_extinction_rate: result.summary.classical_extinction_rate,
                quantum_generation: result.summary.quantum_mean_generation,
                classical_generation: result.summary.classical_mean_generation,
            };

            row.push(point);

            completed += 1;
            if completed % 10 == 0 {
                eprintln!(
                    "Phase diagram progress: {}/{} points",
                    completed, total_points
                );
            }
        }

        points.push(row);
    }

    PhaseDiagram {
        x_param: config.x_range.param,
        y_param: config.y_range.param,
        x_values,
        y_values,
        points,
        base_config: config.base_config.clone(),
        replications: config.replications_per_point,
    }
}

/// Apply a parameter value to a comparison config.
fn apply_parameter(config: &mut ComparisonConfig, param: SweepParameter, value: f32) {
    match param {
        SweepParameter::Ticks => {
            config.max_ticks = value as u64;
        }
        SweepParameter::ResourcePatches => {
            config.heat_patches = value as usize;
        }
        SweepParameter::ResourceValue => {
            config.heat_value = value as u32;
        }
        SweepParameter::Qubits => {
            config.forager_config.n_qubits = value as u8;
        }
        SweepParameter::CircuitLength => {
            config.forager_config.initial_circuit_length = value as usize;
        }
        SweepParameter::MaxCircuitLength => {
            config.forager_config.max_circuit_length = value as usize;
        }
        SweepParameter::MutationRate => {
            config.forager_config.mutation_rate = value;
        }
        SweepParameter::InitialPopulation => {
            config.forager_config.initial_population = value as usize;
        }
        SweepParameter::ReproductionThreshold => {
            config.forager_config.reproduction_threshold = value;
        }
        SweepParameter::MovementCost => {
            config.forager_config.movement_cost = value;
        }
        SweepParameter::MetabolismCost => {
            config.forager_config.metabolism_cost = value;
        }
    }
}

/// Get human-readable parameter name.
fn param_name(param: SweepParameter) -> &'static str {
    match param {
        SweepParameter::Ticks => "Ticks",
        SweepParameter::ResourcePatches => "Patches",
        SweepParameter::ResourceValue => "Value",
        SweepParameter::Qubits => "Qubits",
        SweepParameter::CircuitLength => "CircLen",
        SweepParameter::MaxCircuitLength => "MaxLen",
        SweepParameter::MutationRate => "MutRate",
        SweepParameter::InitialPopulation => "InitPop",
        SweepParameter::ReproductionThreshold => "ReproThr",
        SweepParameter::MovementCost => "MoveCost",
        SweepParameter::MetabolismCost => "MetaCost",
    }
}

// =============================================================================
// Pre-built Phase Diagram Configs
// =============================================================================

/// Time vs Resource Scarcity — the fundamental landscape.
pub fn time_vs_scarcity_config() -> PhaseDiagramConfig {
    PhaseDiagramConfig {
        x_range: ParameterRange::new(SweepParameter::Ticks, 250.0, 3000.0, 12),
        y_range: ParameterRange::new(SweepParameter::ResourcePatches, 10.0, 50.0, 9),
        replications_per_point: 3,
        ..Default::default()
    }
}

/// Qubits vs Circuit Length — complexity landscape.
pub fn qubits_vs_length_config() -> PhaseDiagramConfig {
    PhaseDiagramConfig {
        x_range: ParameterRange::new(SweepParameter::Qubits, 2.0, 8.0, 7),
        y_range: ParameterRange::new(SweepParameter::CircuitLength, 4.0, 20.0, 9),
        replications_per_point: 3,
        ..Default::default()
    }
}

/// Mutation Rate vs Time — evolutionary dynamics.
pub fn mutation_vs_time_config() -> PhaseDiagramConfig {
    PhaseDiagramConfig {
        x_range: ParameterRange::new(SweepParameter::Ticks, 250.0, 2000.0, 8),
        y_range: ParameterRange::new(SweepParameter::MutationRate, 0.1, 0.95, 9),
        replications_per_point: 3,
        ..Default::default()
    }
}

/// Resource Patches vs Value — resource quality vs quantity.
pub fn patches_vs_value_config() -> PhaseDiagramConfig {
    PhaseDiagramConfig {
        x_range: ParameterRange::new(SweepParameter::ResourcePatches, 10.0, 50.0, 9),
        y_range: ParameterRange::new(SweepParameter::ResourceValue, 20.0, 80.0, 7),
        replications_per_point: 3,
        ..Default::default()
    }
}

/// Movement Cost vs Metabolism — energy economics.
pub fn movement_vs_metabolism_config() -> PhaseDiagramConfig {
    PhaseDiagramConfig {
        x_range: ParameterRange::new(SweepParameter::MovementCost, 0.2, 1.5, 7),
        y_range: ParameterRange::new(SweepParameter::MetabolismCost, 0.1, 1.0, 7),
        replications_per_point: 3,
        ..Default::default()
    }
}

// =============================================================================
// 1D Parameter Sweep (Simpler Analysis)
// =============================================================================

/// Result of a 1D parameter sweep.
#[derive(Clone, Debug)]
pub struct ParameterSweep1D {
    pub param: SweepParameter,
    pub values: Vec<f32>,
    pub advantages: Vec<f32>,
    pub quantum_populations: Vec<f32>,
    pub classical_populations: Vec<f32>,
    pub quantum_extinctions: Vec<f32>,
    pub classical_extinctions: Vec<f32>,
}

impl ParameterSweep1D {
    /// Find the value that maximizes quantum advantage.
    pub fn optimal_value(&self) -> Option<f32> {
        self.advantages
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| self.values[i])
    }

    /// Find the crossover point (where advantage changes sign).
    pub fn crossover_point(&self) -> Option<f32> {
        for i in 1..self.advantages.len() {
            let prev = self.advantages[i - 1];
            let curr = self.advantages[i];

            if (prev >= 0.0 && curr < 0.0) || (prev <= 0.0 && curr > 0.0) {
                // Linear interpolation
                let frac = prev.abs() / (prev.abs() + curr.abs());
                return Some(self.values[i - 1] + frac * (self.values[i] - self.values[i - 1]));
            }
        }
        None
    }

    /// Export to CSV.
    pub fn export_csv(&self) -> String {
        let mut csv = format!(
            "{},quantum_adv,q_pop,c_pop,q_extinct,c_extinct\n",
            param_name(self.param)
        );

        for i in 0..self.values.len() {
            csv.push_str(&format!(
                "{:.3},{:.2},{:.1},{:.1},{:.2},{:.2}\n",
                self.values[i],
                self.advantages[i],
                self.quantum_populations[i],
                self.classical_populations[i],
                self.quantum_extinctions[i],
                self.classical_extinctions[i],
            ));
        }

        csv
    }

    /// Generate ASCII plot.
    pub fn ascii_plot(&self) -> String {
        let mut output = String::new();

        let max_adv = self.advantages.iter().cloned().fold(f32::MIN, f32::max);
        let min_adv = self.advantages.iter().cloned().fold(f32::MAX, f32::min);
        let range = (max_adv - min_adv).max(1.0);

        output.push_str(&format!(
            "\n{} Sweep: Quantum Advantage\n",
            param_name(self.param)
        ));
        output.push_str(&format!("Range: {:.1} to {:.1}\n\n", min_adv, max_adv));

        let height = 10;

        for row in (0..height).rev() {
            let threshold = min_adv + (row as f32 / height as f32) * range;
            output.push_str(&format!("{:>8.1} |", threshold));

            for adv in &self.advantages {
                if *adv >= threshold {
                    if *adv > 0.0 {
                        output.push_str(" ● ");
                    } else {
                        output.push_str(" ○ ");
                    }
                } else {
                    output.push_str("   ");
                }
            }
            output.push('\n');
        }

        // Zero line
        if min_adv < 0.0 && max_adv > 0.0 {
            output.push_str(&format!("{:>8} +", "0"));
            output.push_str(&"---".repeat(self.values.len()));
            output.push('\n');
        }

        output.push_str(&format!("{:>8} +", ""));
        output.push_str(&"---".repeat(self.values.len()));
        output.push('\n');

        output.push_str(&format!("{:>8}  ", ""));
        for v in &self.values {
            output.push_str(&format!("{:>3.0}", v));
        }
        output.push('\n');
        output.push_str(&format!("{:>8}  {}\n", "", param_name(self.param)));

        if let Some(opt) = self.optimal_value() {
            output.push_str(&format!(
                "\nOptimal {} for quantum: {:.1}\n",
                param_name(self.param),
                opt
            ));
        }
        if let Some(cross) = self.crossover_point() {
            output.push_str(&format!("Crossover point: {:.1}\n", cross));
        }

        output
    }
}

/// Run a 1D parameter sweep.
pub fn run_parameter_sweep(
    param: SweepParameter,
    range: &ParameterRange,
    base_config: &ComparisonConfig,
    replications: usize,
) -> ParameterSweep1D {
    let values = range.values();
    let mut advantages = Vec::with_capacity(values.len());
    let mut q_pops = Vec::with_capacity(values.len());
    let mut c_pops = Vec::with_capacity(values.len());
    let mut q_ext = Vec::with_capacity(values.len());
    let mut c_ext = Vec::with_capacity(values.len());

    for (i, value) in values.iter().enumerate() {
        let mut config = base_config.clone();
        config.replications = replications;
        apply_parameter(&mut config, param, *value);

        let result = run_quantum_vs_classical(&config);

        advantages.push(result.summary.population_advantage);
        q_pops.push(result.summary.quantum_mean_population);
        c_pops.push(result.summary.classical_mean_population);
        q_ext.push(result.summary.quantum_extinction_rate);
        c_ext.push(result.summary.classical_extinction_rate);

        eprintln!(
            "Sweep progress: {}/{} ({} = {:.1})",
            i + 1,
            values.len(),
            param_name(param),
            value
        );
    }

    ParameterSweep1D {
        param,
        values,
        advantages,
        quantum_populations: q_pops,
        classical_populations: c_pops,
        quantum_extinctions: q_ext,
        classical_extinctions: c_ext,
    }
}

// =============================================================================
// Summary Statistics
// =============================================================================

/// Summary of a phase diagram analysis.
#[derive(Clone, Debug)]
pub struct PhaseDiagramSummary {
    pub total_points: usize,
    pub quantum_wins: usize,
    pub classical_wins: usize,
    pub ties: usize,
    pub both_extinct: usize,
    pub max_quantum_advantage: f32,
    pub max_classical_advantage: f32,
    pub optimal_x: f32,
    pub optimal_y: f32,
    pub quantum_region_fraction: f32,
}

impl PhaseDiagram {
    pub fn summary(&self) -> PhaseDiagramSummary {
        let mut quantum_wins = 0;
        let mut classical_wins = 0;
        let mut ties = 0;
        let mut both_extinct = 0;
        let mut max_q_adv = f32::MIN;
        let mut max_c_adv = f32::MIN;
        let mut optimal_x = 0.0f32;
        let mut optimal_y = 0.0f32;

        for row in &self.points {
            for point in row {
                if point.quantum_extinction_rate > 0.9 && point.classical_extinction_rate > 0.9 {
                    both_extinct += 1;
                } else if point.quantum_advantage > 10.0 {
                    quantum_wins += 1;
                } else if point.quantum_advantage < -10.0 {
                    classical_wins += 1;
                } else {
                    ties += 1;
                }

                if point.quantum_advantage > max_q_adv {
                    max_q_adv = point.quantum_advantage;
                    optimal_x = point.x_value;
                    optimal_y = point.y_value;
                }

                if -point.quantum_advantage > max_c_adv {
                    max_c_adv = -point.quantum_advantage;
                }
            }
        }

        let total = self.points.iter().map(|r| r.len()).sum::<usize>();
        let quantum_region = quantum_wins as f32 / total.max(1) as f32;

        PhaseDiagramSummary {
            total_points: total,
            quantum_wins,
            classical_wins,
            ties,
            both_extinct,
            max_quantum_advantage: max_q_adv,
            max_classical_advantage: max_c_adv,
            optimal_x,
            optimal_y,
            quantum_region_fraction: quantum_region,
        }
    }

    /// Print formatted summary.
    pub fn print_summary(&self) {
        let s = self.summary();

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                  PHASE DIAGRAM SUMMARY                        ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        println!(
            "Parameters: {} vs {}",
            param_name(self.x_param),
            param_name(self.y_param)
        );
        println!(
            "Grid size: {} x {} = {} points",
            self.x_values.len(),
            self.y_values.len(),
            s.total_points
        );
        println!("Replications per point: {}\n", self.replications);

        println!("Results:");
        println!(
            "  Quantum wins:     {:>4} ({:.1}%)",
            s.quantum_wins,
            s.quantum_wins as f32 / s.total_points as f32 * 100.0
        );
        println!(
            "  Classical wins:   {:>4} ({:.1}%)",
            s.classical_wins,
            s.classical_wins as f32 / s.total_points as f32 * 100.0
        );
        println!(
            "  Ties:             {:>4} ({:.1}%)",
            s.ties,
            s.ties as f32 / s.total_points as f32 * 100.0
        );
        println!(
            "  Both extinct:     {:>4} ({:.1}%)",
            s.both_extinct,
            s.both_extinct as f32 / s.total_points as f32 * 100.0
        );
        println!();
        println!("Max quantum advantage:   {:>+.1}", s.max_quantum_advantage);
        println!(
            "Max classical advantage: {:>+.1}",
            s.max_classical_advantage
        );
        println!(
            "Optimal point for quantum: {} = {:.1}, {} = {:.1}",
            param_name(self.x_param),
            s.optimal_x,
            param_name(self.y_param),
            s.optimal_y
        );
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_range() {
        let range = ParameterRange::new(SweepParameter::Ticks, 100.0, 500.0, 5);
        let values = range.values();

        assert_eq!(values.len(), 5);
        assert!((values[0] - 100.0).abs() < 0.01);
        assert!((values[4] - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_param_name() {
        assert_eq!(param_name(SweepParameter::Ticks), "Ticks");
        assert_eq!(param_name(SweepParameter::Qubits), "Qubits");
    }

    #[test]
    fn test_apply_parameter() {
        let mut config = ComparisonConfig::default();

        apply_parameter(&mut config, SweepParameter::Ticks, 1000.0);
        assert_eq!(config.max_ticks, 1000);

        apply_parameter(&mut config, SweepParameter::Qubits, 6.0);
        assert_eq!(config.forager_config.n_qubits, 6);
    }

    #[test]
    fn test_crossover_detection() {
        let sweep = ParameterSweep1D {
            param: SweepParameter::Ticks,
            values: vec![100.0, 200.0, 300.0, 400.0, 500.0],
            advantages: vec![20.0, 10.0, 0.0, -10.0, -20.0],
            quantum_populations: vec![50.0; 5],
            classical_populations: vec![50.0; 5],
            quantum_extinctions: vec![0.0; 5],
            classical_extinctions: vec![0.0; 5],
        };

        let crossover = sweep.crossover_point();
        assert!(crossover.is_some());
        let cross = crossover.unwrap();
        assert!(cross > 200.0 && cross < 400.0);
    }

    #[test]
    fn test_optimal_value() {
        let sweep = ParameterSweep1D {
            param: SweepParameter::Ticks,
            values: vec![100.0, 200.0, 300.0],
            advantages: vec![10.0, 50.0, 20.0],
            quantum_populations: vec![50.0; 3],
            classical_populations: vec![50.0; 3],
            quantum_extinctions: vec![0.0; 3],
            classical_extinctions: vec![0.0; 3],
        };

        let optimal = sweep.optimal_value();
        assert_eq!(optimal, Some(200.0));
    }
}
