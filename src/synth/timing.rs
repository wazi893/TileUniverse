//! Static Timing Analysis (STA) for mapped netlists.
//!
//! This is the yardstick for the rest of the synthesis flow: it answers
//! "how fast is this circuit, and where is the bottleneck?" the same way a
//! production EDA tool (PrimeTime / Tempus) does — by computing **arrival**,
//! **required**, and **slack** times on every net, then reporting the
//! critical path.
//!
//! # Model
//!
//! Block-oriented STA on the [`MappedNetlist`]. Each cell contributes its
//! [`Cell::delay`](crate::synth::cell_lib::Cell::delay) (sourced from
//! `TileType::propagation_delay()`) as a lumped gate delay. Wire delay is not
//! modelled here — the netlist has no placement yet — so this is a pre-layout,
//! gate-level timing view. (Post-route wire delay is a future extension that
//! plugs into the same forward/backward passes.)
//!
//! - **Arrival** `a[net]`: latest time the signal on `net` is valid.
//!   PIs start at their input-arrival constraint (default 0); a gate's output
//!   net is `max(a[fanin]) + cell.delay`.
//! - **Required** `r[net]`: latest time the signal may arrive while still
//!   meeting the target. POs are constrained to the target time; required
//!   propagates backward as `r[fanin] = min(r[out] - cell.delay)`.
//! - **Slack** `r - a`. Negative slack = timing violation.
//!
//! Because [`MappedNetlist::gates`] is in topological order, arrival is a
//! single forward pass and required is a single reverse pass.

use crate::synth::cell_lib::CellLibrary;
use crate::synth::mapping::MappedNetlist;

/// Timing constraints driving the analysis.
#[derive(Clone, Debug, Default)]
pub struct TimingConstraint {
    /// Per-PI arrival time. `None` → all inputs arrive at 0.
    /// Length must equal the number of primary inputs when `Some`.
    pub input_arrivals: Option<Vec<f64>>,
    /// Target time at every primary output (a clock period, minus setup).
    /// `None` → use the achieved circuit delay, so the critical endpoint has
    /// exactly zero slack (a pure "how fast is it" measurement).
    pub clock_period: Option<f64>,
}

impl TimingConstraint {
    /// Measure intrinsic speed: required = achieved delay (critical slack = 0).
    pub fn intrinsic() -> Self {
        Self::default()
    }

    /// Constrain to a fixed clock period (negative slack reveals violations).
    pub fn with_period(period: f64) -> Self {
        TimingConstraint {
            input_arrivals: None,
            clock_period: Some(period),
        }
    }
}

/// Full timing picture: per-net arrival/required/slack plus circuit-level
/// summary numbers.
#[derive(Clone, Debug)]
pub struct TimingReport {
    /// Arrival time per net (indexed by net id).
    pub arrival: Vec<f64>,
    /// Required time per net (indexed by net id). `INFINITY` = unconstrained.
    pub required: Vec<f64>,
    /// Longest PO arrival — the combinational delay of the circuit.
    pub circuit_delay: f64,
    /// The target time required times were computed against.
    pub target: f64,
    /// Index (into `netlist.primary_outputs`) of the worst-slack endpoint.
    pub critical_endpoint: usize,
    /// Worst (minimum) slack across all endpoints.
    pub worst_slack: f64,
    /// Cell count — a simple area proxy (each tile-native cell ≈ one tile).
    pub area_cells: usize,
}

impl TimingReport {
    /// Slack at a net: `required - arrival`.
    pub fn slack(&self, net: u32) -> f64 {
        let i = net as usize;
        self.required[i] - self.arrival[i]
    }

    /// Whether all endpoints meet timing (no negative slack).
    pub fn meets_timing(&self) -> bool {
        self.worst_slack >= -f64::EPSILON
    }
}

/// One hop along a reported path.
#[derive(Clone, Debug)]
pub struct PathNode {
    /// Net id at this point in the path.
    pub net_id: u32,
    /// Cell name driving this net (`"PI"` for a primary input startpoint).
    pub cell_name: String,
    /// Delay contributed at this hop (0 at the startpoint).
    pub incr_delay: f64,
    /// Cumulative arrival at this net.
    pub arrival: f64,
}

/// A start-to-end critical (or any) path with its slack.
#[derive(Clone, Debug)]
pub struct CriticalPath {
    /// Startpoint name (a primary input).
    pub startpoint: String,
    /// Endpoint name (a primary output).
    pub endpoint: String,
    /// Path hops from startpoint to endpoint.
    pub nodes: Vec<PathNode>,
    /// Data arrival at the endpoint.
    pub arrival: f64,
    /// Required time at the endpoint.
    pub required: f64,
    /// `required - arrival`.
    pub slack: f64,
}

/// Run static timing analysis on a mapped netlist.
pub fn analyze_timing(
    netlist: &MappedNetlist,
    lib: &CellLibrary,
    constraint: &TimingConstraint,
) -> TimingReport {
    let num_nets = netlist.num_nets as usize;
    let num_pi = netlist.primary_inputs.len();

    // --- Forward pass: arrival ---
    let mut arrival = vec![0.0f64; num_nets];
    if let Some(ia) = &constraint.input_arrivals {
        assert_eq!(
            ia.len(),
            num_pi,
            "input_arrivals length ({}) != primary input count ({})",
            ia.len(),
            num_pi
        );
        for (i, &a) in ia.iter().enumerate() {
            arrival[i] = a;
        }
    }

    for (gate_idx, gate) in netlist.gates.iter().enumerate() {
        let net_id = num_pi + gate_idx;
        if net_id >= num_nets {
            continue;
        }
        let d = lib.cell(gate.cell_idx).delay as f64;
        let mut max_in = 0.0f64;
        for &f in &gate.fanins {
            let fi = f as usize;
            if fi < num_nets && arrival[fi] > max_in {
                max_in = arrival[fi];
            }
        }
        arrival[net_id] = max_in + d;
    }

    // --- Circuit delay = worst PO arrival ---
    let mut circuit_delay = 0.0f64;
    for (_, net_id, _) in &netlist.primary_outputs {
        if *net_id != u32::MAX {
            let a = arrival[*net_id as usize];
            if a > circuit_delay {
                circuit_delay = a;
            }
        }
    }

    let target = constraint.clock_period.unwrap_or(circuit_delay);

    // --- Backward pass: required ---
    let mut required = vec![f64::INFINITY; num_nets];
    for (_, net_id, _) in &netlist.primary_outputs {
        if *net_id != u32::MAX {
            let i = *net_id as usize;
            if target < required[i] {
                required[i] = target;
            }
        }
    }
    for (gate_idx, gate) in netlist.gates.iter().enumerate().rev() {
        let net_id = num_pi + gate_idx;
        if net_id >= num_nets {
            continue;
        }
        let req_out = required[net_id];
        if !req_out.is_finite() {
            continue;
        }
        let d = lib.cell(gate.cell_idx).delay as f64;
        let req_in = req_out - d;
        for &f in &gate.fanins {
            let fi = f as usize;
            if fi < num_nets && req_in < required[fi] {
                required[fi] = req_in;
            }
        }
    }

    // --- Worst-slack endpoint ---
    let mut critical_endpoint = 0usize;
    let mut worst_slack = f64::INFINITY;
    for (i, (_, net_id, _)) in netlist.primary_outputs.iter().enumerate() {
        let a = if *net_id == u32::MAX {
            0.0
        } else {
            arrival[*net_id as usize]
        };
        let slack = target - a;
        if slack < worst_slack {
            worst_slack = slack;
            critical_endpoint = i;
        }
    }
    if netlist.primary_outputs.is_empty() {
        worst_slack = 0.0;
    }

    TimingReport {
        arrival,
        required,
        circuit_delay,
        target,
        critical_endpoint,
        worst_slack,
        area_cells: netlist.num_gates(),
    }
}

/// Trace the critical path for a given endpoint (PO index).
///
/// Walks backward from the endpoint, at each gate following the fanin with the
/// largest arrival (the timing-limiting input), until a primary input is hit.
pub fn trace_path(
    netlist: &MappedNetlist,
    lib: &CellLibrary,
    report: &TimingReport,
    po_index: usize,
) -> CriticalPath {
    let num_pi = netlist.primary_inputs.len();
    let (po_name, po_net, _po_inv) = &netlist.primary_outputs[po_index];

    let mut nodes: Vec<PathNode> = Vec::new();

    // Constant output: degenerate path.
    if *po_net == u32::MAX {
        return CriticalPath {
            startpoint: "<const>".to_string(),
            endpoint: po_name.clone(),
            nodes,
            arrival: 0.0,
            required: report.target,
            slack: report.target,
        };
    }

    let mut cur = *po_net;
    loop {
        let ci = cur as usize;
        if cur < num_pi as u32 {
            // Primary input startpoint.
            nodes.push(PathNode {
                net_id: cur,
                cell_name: format!("PI:{}", netlist.primary_inputs[ci]),
                incr_delay: report.arrival[ci],
                arrival: report.arrival[ci],
            });
            break;
        }
        let gate_idx = ci - num_pi;
        let gate = &netlist.gates[gate_idx];
        let cell = lib.cell(gate.cell_idx);
        nodes.push(PathNode {
            net_id: cur,
            cell_name: cell.name.to_string(),
            incr_delay: cell.delay as f64,
            arrival: report.arrival[ci],
        });
        // Follow the critical fanin (largest arrival). Const-0 / no-fanin
        // gates terminate the walk.
        let mut best: Option<u32> = None;
        let mut best_arr = f64::NEG_INFINITY;
        for &f in &gate.fanins {
            let fa = report.arrival[f as usize];
            if fa > best_arr {
                best_arr = fa;
                best = Some(f);
            }
        }
        match best {
            Some(f) => cur = f,
            None => break,
        }
    }

    nodes.reverse();

    let arrival = if *po_net == u32::MAX {
        0.0
    } else {
        report.arrival[*po_net as usize]
    };
    CriticalPath {
        startpoint: nodes
            .first()
            .map(|n| n.cell_name.clone())
            .unwrap_or_else(|| "<none>".to_string()),
        endpoint: po_name.clone(),
        nodes,
        arrival,
        required: report.target,
        slack: report.target - arrival,
    }
}

/// Trace the worst-slack (critical) path of the whole circuit.
pub fn critical_path(
    netlist: &MappedNetlist,
    lib: &CellLibrary,
    report: &TimingReport,
) -> CriticalPath {
    trace_path(netlist, lib, report, report.critical_endpoint)
}

/// Produce a `report_timing`-style text report for the critical path.
pub fn report_timing(
    netlist: &MappedNetlist,
    lib: &CellLibrary,
    constraint: &TimingConstraint,
) -> String {
    let report = analyze_timing(netlist, lib, constraint);
    let path = critical_path(netlist, lib, &report);
    format_path(&path, &report)
}

/// Format a single path into a PrimeTime-like report block.
pub fn format_path(path: &CriticalPath, report: &TimingReport) -> String {
    let mut s = String::new();
    s.push_str("============================================================\n");
    s.push_str(" Static Timing Report (gate-level, pre-layout)\n");
    s.push_str("============================================================\n");
    s.push_str(&format!("  Startpoint: {}\n", path.startpoint));
    s.push_str(&format!("  Endpoint:   {}\n", path.endpoint));
    s.push_str(&format!(
        "  Cells: {}   Circuit delay: {:.2}\n",
        report.area_cells, report.circuit_delay
    ));
    s.push_str("------------------------------------------------------------\n");
    s.push_str(&format!(
        "  {:<22} {:>8} {:>10}\n",
        "Point", "Incr", "Arrival"
    ));
    s.push_str("------------------------------------------------------------\n");
    for node in &path.nodes {
        let label = format!("{} (net {})", node.cell_name, node.net_id);
        s.push_str(&format!(
            "  {:<22} {:>8.2} {:>10.2}\n",
            label, node.incr_delay, node.arrival
        ));
    }
    s.push_str("------------------------------------------------------------\n");
    s.push_str(&format!(
        "  {:<22} {:>8} {:>10.2}\n",
        "data arrival time", "", path.arrival
    ));
    s.push_str(&format!(
        "  {:<22} {:>8} {:>10.2}\n",
        "data required time", "", path.required
    ));
    s.push_str("------------------------------------------------------------\n");
    let verdict = if path.slack >= -f64::EPSILON {
        "MET"
    } else {
        "VIOLATED"
    };
    s.push_str(&format!(
        "  {:<22} {:>8} {:>10.2}   ({})\n",
        "slack", "", path.slack, verdict
    ));
    s.push_str("============================================================\n");
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::aig::Aig;
    use crate::synth::cell_lib::CellLibrary;
    use crate::synth::mapping::map_to_library;
    use crate::synth::mapping::MapConfig;

    /// Build a chain of `depth` ANDs: y = ((((a&b)&c)&d)...) so the critical
    /// path is unambiguous and its length is known.
    fn and_chain(depth: usize) -> (Aig, CellLibrary) {
        let mut aig = Aig::new();
        let inputs: Vec<_> = (0..=depth)
            .map(|i| aig.add_input(&format!("i{}", i)))
            .collect();
        let mut acc = inputs[0];
        for &x in &inputs[1..] {
            acc = aig.and(acc, x);
        }
        aig.add_output("y", acc);
        (aig, CellLibrary::tile_native())
    }

    #[test]
    fn single_and_delay_equals_cell_delay() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);
        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());

        let report = analyze_timing(&netlist, &lib, &TimingConstraint::intrinsic());
        // Circuit delay must be positive and equal to the AND cell's delay
        // (single gate, both inputs arrive at 0).
        let and_delay = crate::tiles::tile_meta::TileType::And.propagation_delay() as f64;
        assert!(report.circuit_delay > 0.0);
        assert_eq!(report.circuit_delay, and_delay);
        // Intrinsic constraint → critical endpoint has zero slack.
        assert!(report.worst_slack.abs() < 1e-9);
    }

    #[test]
    fn deeper_chain_has_larger_delay() {
        let (a2, lib) = and_chain(2);
        let (a8, _) = and_chain(8);
        let n2 = map_to_library(&a2, &lib, &MapConfig::default());
        let n8 = map_to_library(&a8, &lib, &MapConfig::default());
        let r2 = analyze_timing(&n2, &lib, &TimingConstraint::intrinsic());
        let r8 = analyze_timing(&n8, &lib, &TimingConstraint::intrinsic());
        assert!(
            r8.circuit_delay > r2.circuit_delay,
            "depth-8 delay {} should exceed depth-2 delay {}",
            r8.circuit_delay,
            r2.circuit_delay
        );
    }

    #[test]
    fn critical_path_starts_at_pi_ends_at_po() {
        let (aig, lib) = and_chain(4);
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let report = analyze_timing(&netlist, &lib, &TimingConstraint::intrinsic());
        let path = critical_path(&netlist, &lib, &report);

        assert!(path.nodes.len() >= 2, "path too short: {:?}", path.nodes);
        assert!(
            path.nodes.first().unwrap().cell_name.starts_with("PI:"),
            "path must start at a PI, got {}",
            path.nodes.first().unwrap().cell_name
        );
        assert_eq!(path.endpoint, "y");
        // Arrival must be monotonically non-decreasing along the path.
        for w in path.nodes.windows(2) {
            assert!(w[1].arrival >= w[0].arrival - 1e-9);
        }
        // Endpoint arrival equals the circuit delay for this single-output ckt.
        assert!((path.arrival - report.circuit_delay).abs() < 1e-9);
    }

    #[test]
    fn tight_period_produces_negative_slack() {
        let (aig, lib) = and_chain(6);
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let intrinsic = analyze_timing(&netlist, &lib, &TimingConstraint::intrinsic());
        // Constrain to half the achieved delay → must violate.
        let tight = TimingConstraint::with_period(intrinsic.circuit_delay / 2.0);
        let report = analyze_timing(&netlist, &lib, &tight);
        assert!(
            !report.meets_timing(),
            "halved period should violate; worst_slack={}",
            report.worst_slack
        );
        assert!(report.worst_slack < 0.0);
    }

    #[test]
    fn slack_zero_at_intrinsic_period() {
        let (aig, lib) = and_chain(5);
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let report = analyze_timing(&netlist, &lib, &TimingConstraint::intrinsic());
        assert!(report.meets_timing());
        assert!(report.worst_slack.abs() < 1e-9);
    }

    #[test]
    fn report_string_is_well_formed() {
        let (aig, lib) = and_chain(3);
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let text = report_timing(&netlist, &lib, &TimingConstraint::intrinsic());
        assert!(text.contains("Startpoint:"));
        assert!(text.contains("Endpoint:"));
        assert!(text.contains("slack"));
        assert!(text.contains("data arrival time"));
    }
}
