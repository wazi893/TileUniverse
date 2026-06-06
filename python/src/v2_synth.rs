use engine::synth::mapping::evaluate_aig;
use engine::synth::{
    self, Aig, CellLibrary, PlaceConfig, RouteConfig, SynthConfig, evaluate_exported,
    export_to_simulation, materialize_inversions, place_mapped_netlist, read_aiger,
    route_placed_netlist, write_aiger, write_aiger_ascii,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::io::BufReader;

struct ValidationSummary {
    mismatches: Vec<synth::OutputMismatch>,
    converged: bool,
    convergence_rounds: u32,
}

// ---------------------------------------------------------------------------
// PySynthResult — result of synthesize()
// ---------------------------------------------------------------------------

/// Result from synthesizing a boolean function to tile fabric.
#[pyclass(name = "SynthResult")]
pub struct PySynthResult {
    #[pyo3(get)]
    pub num_inputs: usize,
    #[pyo3(get)]
    pub num_outputs: usize,
    #[pyo3(get)]
    pub num_and_nodes: usize,
    #[pyo3(get)]
    pub aig_depth: u32,
    #[pyo3(get)]
    pub mapped_gates: usize,
    #[pyo3(get)]
    pub grid_width: usize,
    #[pyo3(get)]
    pub grid_height: usize,
    #[pyo3(get)]
    pub num_layers: usize,
    #[pyo3(get)]
    pub route_count: usize,
    #[pyo3(get)]
    pub combos_tested: u32,
    #[pyo3(get)]
    pub validated: bool,
    #[pyo3(get)]
    pub num_mismatches: usize,
    #[pyo3(get)]
    pub first_mismatch_output: Option<usize>,
    #[pyo3(get)]
    pub first_mismatch_input_combo: Option<u32>,
    #[pyo3(get)]
    pub first_mismatch_expected: Option<bool>,
    #[pyo3(get)]
    pub first_mismatch_actual: Option<bool>,
    #[pyo3(get)]
    pub per_output_pass: Vec<bool>,
    #[pyo3(get)]
    pub per_output_mismatches: Vec<usize>,
    /// Whether propagation converged within the iteration limit.
    #[pyo3(get)]
    pub converged: bool,
    /// Number of outer propagation rounds used (each ≤100 inner delta cycles).
    #[pyo3(get)]
    pub convergence_rounds: u32,
}

#[pymethods]
impl PySynthResult {
    fn summary(&self) -> String {
        format!(
            "{} inputs, {} outputs, {} AND nodes, {} mapped gates, depth {}, grid {}x{}x{}, {} route tiles, {} combos tested, {}",
            self.num_inputs,
            self.num_outputs,
            self.num_and_nodes,
            self.mapped_gates,
            self.aig_depth,
            self.grid_width,
            self.grid_height,
            self.num_layers,
            self.route_count,
            self.combos_tested,
            if self.validated { "PASS" } else { "FAIL" },
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "SynthResult(inputs={}, and_nodes={}, mapped_gates={}, depth={}, grid={}x{}x{}, routes={}, validated={})",
            self.num_inputs,
            self.num_and_nodes,
            self.mapped_gates,
            self.aig_depth,
            self.grid_width,
            self.grid_height,
            self.num_layers,
            self.route_count,
            self.validated,
        )
    }
}

// ---------------------------------------------------------------------------
// synthesize() — Python entry point
// ---------------------------------------------------------------------------

/// Synthesize a boolean function from a truth table to tile fabric.
///
/// Args:
///     truth_table: Integer truth table (e.g., 0xE8 for 3-input majority).
///     num_inputs: Number of input variables (1-6).
///
/// Returns:
///     SynthResult with AIG stats, mapped gate count, exported grid dimensions,
///     route tile count, and validation status.
///
/// Example:
///     result = synthesize(truth_table=0xE8, num_inputs=3)
///     assert result.validated
#[pyfunction]
#[pyo3(signature = (truth_table, num_inputs))]
pub fn synthesize(truth_table: u64, num_inputs: u32) -> PyResult<PySynthResult> {
    validate_truth_tables(&[truth_table], num_inputs, 6)?;

    let aig = Aig::from_truth_table(truth_table, num_inputs);
    let lib = CellLibrary::tile_native();
    let synth_result = synth::synthesize(&aig, &lib, &SynthConfig::default());
    let stats = synth_result.stats.clone();
    let place_config = PlaceConfig::standalone();
    let route_config = RouteConfig::standalone();
    let materialized = materialize_inversions(&synth_result.netlist, &lib);
    let placed = place_mapped_netlist(&materialized, &lib, &place_config)
        .map_err(|e| PyValueError::new_err(format!("Synthesis failed: placement: {}", e)))?;
    let routed = route_placed_netlist(&materialized, &placed, &route_config)
        .map_err(|e| PyValueError::new_err(format!("Synthesis failed: routing: {}", e)))?;
    let mut export = export_to_simulation(&routed, &materialized);
    let summary = validate_export_against_aig(&aig, &mut export);
    let first_mismatch = summary.mismatches.first();
    let (per_pass, per_mm) = per_output_stats(&summary.mismatches, stats.aig_outputs);

    Ok(PySynthResult {
        num_inputs: num_inputs as usize,
        num_outputs: stats.aig_outputs,
        num_and_nodes: stats.aig_and_nodes,
        aig_depth: stats.aig_depth,
        mapped_gates: stats.mapped_gates,
        grid_width: export.sim.width(),
        grid_height: export.sim.height(),
        num_layers: export.sim.num_layers(),
        route_count: routed.routes.len(),
        combos_tested: total_combos(num_inputs as usize),
        validated: summary.mismatches.is_empty(),
        num_mismatches: summary.mismatches.len(),
        first_mismatch_output: first_mismatch.map(|m| m.output_idx),
        first_mismatch_input_combo: first_mismatch.map(|m| m.input_combo),
        first_mismatch_expected: first_mismatch.map(|m| m.expected),
        first_mismatch_actual: first_mismatch.map(|m| m.actual),
        per_output_pass: per_pass,
        per_output_mismatches: per_mm,
        converged: summary.converged,
        convergence_rounds: summary.convergence_rounds,
    })
}

// ---------------------------------------------------------------------------
// synthesize_multi() — multi-output synthesis
// ---------------------------------------------------------------------------

/// Synthesize a multi-output boolean function from truth tables to tile fabric.
///
/// Args:
///     truth_tables: List of integer truth tables (one per output).
///     num_inputs: Number of input variables (1-6).
///
/// Returns:
///     SynthResult with multi-output validation and per-output reporting.
#[pyfunction]
#[pyo3(signature = (truth_tables, num_inputs))]
pub fn synthesize_multi(truth_tables: Vec<u64>, num_inputs: u32) -> PyResult<PySynthResult> {
    if truth_tables.is_empty() {
        return Err(PyValueError::new_err("truth_tables must not be empty"));
    }
    validate_truth_tables(&truth_tables, num_inputs, 6)?;

    let tt_pairs: Vec<(u64, u32)> = truth_tables.iter().map(|&tt| (tt, num_inputs)).collect();
    let aig = Aig::from_truth_tables(&tt_pairs);
    synth_aig_to_result(&aig)
}

// ---------------------------------------------------------------------------
// AIGER import/export
// ---------------------------------------------------------------------------

/// Synthesize a circuit from an AIGER file on disk.
///
/// Combinational only (latches not supported — returns error for L > 0).
#[pyfunction]
#[pyo3(signature = (path,))]
pub fn synthesize_from_aiger(path: String) -> PyResult<PySynthResult> {
    let file = std::fs::File::open(&path)
        .map_err(|e| PyValueError::new_err(format!("cannot open '{}': {}", path, e)))?;
    let mut reader = BufReader::new(file);
    let aig = read_aiger(&mut reader)
        .map_err(|e| PyValueError::new_err(format!("AIGER parse error: {}", e)))?;
    synth_aig_to_result(&aig)
}

/// Synthesize a circuit from AIGER bytes in memory.
///
/// Accepts both binary and ASCII AIGER formats.
#[pyfunction]
#[pyo3(signature = (data,))]
pub fn synthesize_from_aiger_bytes(data: Vec<u8>) -> PyResult<PySynthResult> {
    let mut reader = BufReader::new(&data[..]);
    let aig = read_aiger(&mut reader)
        .map_err(|e| PyValueError::new_err(format!("AIGER parse error: {}", e)))?;
    synth_aig_to_result(&aig)
}

/// Export truth tables to binary AIGER format.
///
/// Args:
///     truth_tables: List of integer truth tables (one per output).
///     num_inputs: Number of input variables (1-6).
///
/// Returns:
///     bytes: Binary AIGER data.
#[pyfunction]
#[pyo3(signature = (truth_tables, num_inputs))]
pub fn export_to_aiger(truth_tables: Vec<u64>, num_inputs: u32) -> PyResult<Vec<u8>> {
    if truth_tables.is_empty() {
        return Err(PyValueError::new_err("truth_tables must not be empty"));
    }
    validate_truth_tables(&truth_tables, num_inputs, 6)?;

    let tt_pairs: Vec<(u64, u32)> = truth_tables.iter().map(|&tt| (tt, num_inputs)).collect();
    let aig = Aig::from_truth_tables(&tt_pairs);
    let mut buf = Vec::new();
    write_aiger(&aig, &mut buf)
        .map_err(|e| PyValueError::new_err(format!("AIGER write error: {}", e)))?;
    Ok(buf)
}

/// Export truth tables to ASCII AIGER format (human-readable).
///
/// Args:
///     truth_tables: List of integer truth tables (one per output).
///     num_inputs: Number of input variables (1-6).
///
/// Returns:
///     str: ASCII AIGER text.
#[pyfunction]
#[pyo3(signature = (truth_tables, num_inputs))]
pub fn export_to_aiger_ascii(truth_tables: Vec<u64>, num_inputs: u32) -> PyResult<String> {
    if truth_tables.is_empty() {
        return Err(PyValueError::new_err("truth_tables must not be empty"));
    }
    validate_truth_tables(&truth_tables, num_inputs, 6)?;

    let tt_pairs: Vec<(u64, u32)> = truth_tables.iter().map(|&tt| (tt, num_inputs)).collect();
    let aig = Aig::from_truth_tables(&tt_pairs);
    let mut buf = Vec::new();
    write_aiger_ascii(&aig, &mut buf)
        .map_err(|e| PyValueError::new_err(format!("AIGER write error: {}", e)))?;
    String::from_utf8(buf)
        .map_err(|e| PyValueError::new_err(format!("AIGER ASCII encoding error: {}", e)))
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Shared truth-table validation: num_inputs in 1..=max_n, each tt < 2^(2^num_inputs).
fn validate_truth_tables(truth_tables: &[u64], num_inputs: u32, max_n: u32) -> PyResult<()> {
    if num_inputs == 0 || num_inputs > max_n {
        return Err(PyValueError::new_err(format!(
            "num_inputs must be 1-{}",
            max_n,
        )));
    }
    // For num_inputs < 6, max_tt = 2^(2^num_inputs). For 6, truth table is 64 bits — all u64 valid.
    let max_tt = if num_inputs < 6 {
        1u64 << (1u64 << num_inputs)
    } else {
        0 // no overflow check needed for 6 inputs (full u64 range)
    };
    for (i, &tt) in truth_tables.iter().enumerate() {
        if max_tt != 0 && tt >= max_tt {
            return Err(PyValueError::new_err(format!(
                "truth_table{} too large for {} inputs (max {})",
                if truth_tables.len() > 1 {
                    format!("[{}]", i)
                } else {
                    String::new()
                },
                num_inputs,
                max_tt - 1,
            )));
        }
    }
    Ok(())
}

fn total_combos(num_inputs: usize) -> u32 {
    if num_inputs <= 20 {
        1u32 << num_inputs
    } else {
        1024
    }
}

fn per_output_stats(
    mismatches: &[synth::OutputMismatch],
    num_outputs: usize,
) -> (Vec<bool>, Vec<usize>) {
    let mut per_mm = vec![0usize; num_outputs];
    for m in mismatches {
        if m.output_idx < num_outputs {
            per_mm[m.output_idx] += 1;
        }
    }
    let per_pass: Vec<bool> = per_mm.iter().map(|&c| c == 0).collect();
    (per_pass, per_mm)
}

/// Shared pipeline: AIG → synth → place → route → export → validate → PySynthResult.
fn synth_aig_to_result(aig: &Aig) -> PyResult<PySynthResult> {
    let lib = CellLibrary::tile_native();
    let synth_result = synth::synthesize(aig, &lib, &SynthConfig::default());
    let stats = synth_result.stats.clone();
    let place_config = PlaceConfig::standalone();
    let route_config = RouteConfig::standalone();
    let materialized = materialize_inversions(&synth_result.netlist, &lib);
    let placed = place_mapped_netlist(&materialized, &lib, &place_config)
        .map_err(|e| PyValueError::new_err(format!("Synthesis failed: placement: {}", e)))?;
    let routed = route_placed_netlist(&materialized, &placed, &route_config)
        .map_err(|e| PyValueError::new_err(format!("Synthesis failed: routing: {}", e)))?;
    let mut export = export_to_simulation(&routed, &materialized);
    let summary = validate_export_against_aig(aig, &mut export);
    let first_mismatch = summary.mismatches.first();
    let (per_pass, per_mm) = per_output_stats(&summary.mismatches, stats.aig_outputs);

    Ok(PySynthResult {
        num_inputs: aig.num_inputs() as usize,
        num_outputs: stats.aig_outputs,
        num_and_nodes: stats.aig_and_nodes,
        aig_depth: stats.aig_depth,
        mapped_gates: stats.mapped_gates,
        grid_width: export.sim.width(),
        grid_height: export.sim.height(),
        num_layers: export.sim.num_layers(),
        route_count: routed.routes.len(),
        combos_tested: total_combos(aig.num_inputs() as usize),
        validated: summary.mismatches.is_empty(),
        num_mismatches: summary.mismatches.len(),
        first_mismatch_output: first_mismatch.map(|m| m.output_idx),
        first_mismatch_input_combo: first_mismatch.map(|m| m.input_combo),
        first_mismatch_expected: first_mismatch.map(|m| m.expected),
        first_mismatch_actual: first_mismatch.map(|m| m.actual),
        per_output_pass: per_pass,
        per_output_mismatches: per_mm,
        converged: summary.converged,
        convergence_rounds: summary.convergence_rounds,
    })
}

fn validate_export_against_aig(
    aig: &Aig,
    export: &mut engine::synth::SynthExport,
) -> ValidationSummary {
    let num_inputs = aig.num_inputs() as usize;
    let total_combos = total_combos(num_inputs);
    let mut mismatches = Vec::new();
    let mut converged = true;
    let mut convergence_rounds = 0u32;

    for combo in 0..total_combos {
        let inputs: Vec<bool> = (0..num_inputs).map(|i| (combo >> i) & 1 != 0).collect();
        let expected = evaluate_aig(aig, &inputs);
        let actual = evaluate_exported(export, &inputs);
        converged &= export.last_converged;
        convergence_rounds = convergence_rounds.max(export.last_convergence_rounds);

        for (out_idx, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
            if exp != act {
                mismatches.push(synth::OutputMismatch {
                    output_idx: out_idx,
                    input_combo: combo,
                    expected: *exp,
                    actual: *act,
                });
            }
        }
    }

    ValidationSummary {
        mismatches,
        converged,
        convergence_rounds,
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySynthResult>()?;
    m.add_function(wrap_pyfunction!(synthesize, m)?)?;
    m.add_function(wrap_pyfunction!(synthesize_multi, m)?)?;
    m.add_function(wrap_pyfunction!(synthesize_from_aiger, m)?)?;
    m.add_function(wrap_pyfunction!(synthesize_from_aiger_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(export_to_aiger, m)?)?;
    m.add_function(wrap_pyfunction!(export_to_aiger_ascii, m)?)?;
    Ok(())
}
