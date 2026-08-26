//! SA route-frontier benchmark for the AlphaFabric `madd` citation circuit.
//!
//! Measures how far simulated-annealing placement extends the routable frontier
//! for the HLS datapath `madd(a, b, c) = a * b + c` used in the AlphaFabric
//! write-up. "Verified" means the placed+routed netlist exports to the tile
//! fabric and matches the AIG via [`PlacementEnv::verify_physical`].

use std::time::{Duration, Instant};

use tileuniverse_v2_lang::{ArithOp, Expr, Func, Stmt};

use super::anneal::{AnnealConfig, AnnealResult, anneal};
use super::circuit::Circuit;
use super::env::{EnvError, PlacementEnv};
use super::reward::RewardWeights;
use crate::synth::hls::synthesize_func;
use crate::synth::{HlsConfig, PlaceConfig, RouteConfig};

/// Default datapath widths in the citation sweep.
pub const DEFAULT_WIDTHS: &[u32] = &[6, 8];

/// Default placement/routing halos in the citation sweep.
pub const DEFAULT_HALOS: &[usize] = &[3, 4];

/// Default SA iterations per width/halo case.
pub const DEFAULT_ITERS: usize = 30_000;

/// Configuration for a route-frontier benchmark run.
#[derive(Clone, Debug)]
pub struct RouteFrontierConfig {
    pub widths: Vec<u32>,
    pub halos: Vec<usize>,
    pub cases: Option<Vec<RouteFrontierCase>>,
    pub iterations: usize,
    pub seed: u64,
    pub row_major_all: bool,
    pub verify_row_major: bool,
}

/// One width/halo pair to measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteFrontierCase {
    pub width: u32,
    pub halo: usize,
}

/// Row-major routing outcome for one case.
#[derive(Clone, Copy, Debug)]
pub struct RouteOutcome {
    pub routable: bool,
    pub physical: Option<bool>,
    pub elapsed: Duration,
    pub wire_tiles: usize,
    pub vias: usize,
}

/// Simulated-annealing outcome for one case.
#[derive(Clone, Copy, Debug)]
pub struct SaOutcome {
    pub routable: bool,
    pub physical: bool,
    pub elapsed: Duration,
    pub hpwl_improvement: f64,
    pub best_hpwl: usize,
    pub wire_tiles: usize,
    pub vias: usize,
    pub accepted: usize,
}

/// One benchmark row: row-major baseline (optional) plus SA result.
#[derive(Clone, Debug)]
pub struct RouteFrontierRow {
    pub width: u32,
    pub gates: usize,
    pub inputs: usize,
    pub halo: usize,
    pub row_major: Option<RouteOutcome>,
    pub sa: SaOutcome,
}

impl Default for RouteFrontierConfig {
    fn default() -> Self {
        Self {
            widths: DEFAULT_WIDTHS.to_vec(),
            halos: DEFAULT_HALOS.to_vec(),
            cases: Some(default_claim_cases()),
            iterations: DEFAULT_ITERS,
            seed: AnnealConfig::default().seed,
            row_major_all: false,
            verify_row_major: false,
        }
    }
}

impl RouteFrontierConfig {
    /// Resolve explicit cases or the width × halo Cartesian product.
    pub fn resolved_cases(&self) -> Vec<RouteFrontierCase> {
        if let Some(cases) = &self.cases {
            return cases.clone();
        }

        self.widths
            .iter()
            .flat_map(|&width| {
                self.halos
                    .iter()
                    .map(move |&halo| RouteFrontierCase { width, halo })
            })
            .collect()
    }
}

/// Default cases that reproduce the AlphaFabric blog citation claims.
pub fn default_claim_cases() -> Vec<RouteFrontierCase> {
    vec![
        RouteFrontierCase { width: 6, halo: 3 },
        RouteFrontierCase { width: 8, halo: 3 },
        RouteFrontierCase { width: 8, halo: 4 },
    ]
}

/// The citation HLS function: `madd(a, b, c) = a * b + c`.
pub fn madd_func() -> Func {
    Func {
        name: "madd".to_string(),
        params: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        body: vec![Stmt::Return(Expr::Bin(
            ArithOp::Add,
            Box::new(Expr::Bin(
                ArithOp::Mul,
                Box::new(Expr::Var("a".to_string())),
                Box::new(Expr::Var("b".to_string())),
            )),
            Box::new(Expr::Var("c".to_string())),
        ))],
    }
}

/// Synthesize and map the citation `madd` circuit at the given datapath width.
pub fn madd_circuit(width: u32) -> Result<Circuit, String> {
    let aig = synthesize_func(&madd_func(), &HlsConfig { width })
        .map_err(|err| format!("madd@{width} synthesis failed: {err:?}"))?;
    Ok(Circuit::from_aig(format!("madd{width}"), aig))
}

/// Run the full route-frontier sweep.
pub fn run_route_frontier(config: &RouteFrontierConfig) -> Result<Vec<RouteFrontierRow>, String> {
    let cases = config.resolved_cases();
    let mut rows = Vec::new();

    for width in unique_widths(&cases) {
        let circuit = madd_circuit(width)?;
        if !circuit.mapping_is_correct() {
            return Err(format!(
                "mapped madd@{width} netlist is not equivalent to the AIG"
            ));
        }

        for halo in halos_for_width(&cases, width) {
            rows.push(run_route_frontier_row(config, &circuit, width, halo)?);
        }
    }

    Ok(rows)
}

/// Run one width/halo case.
pub fn run_route_frontier_row(
    config: &RouteFrontierConfig,
    circuit: &Circuit,
    width: u32,
    halo: usize,
) -> Result<RouteFrontierRow, String> {
    let mut env = PlacementEnv::with_config(
        circuit,
        place_config(halo),
        route_config(),
        RewardWeights::default(),
    )
    .map_err(|err: EnvError| format!("madd@{width} halo {halo}: env: {err:?}"))?;

    let row_major = if should_measure_row_major(config, width, halo) {
        let row_start = Instant::now();
        let row_score = env.score();
        let row_physical = if config.verify_row_major && row_score.metrics.routable {
            Some(env.verify_physical())
        } else {
            None
        };
        Some(RouteOutcome {
            routable: row_score.metrics.routable,
            physical: row_physical,
            elapsed: row_start.elapsed(),
            wire_tiles: row_score.metrics.wire_tiles,
            vias: row_score.metrics.vias,
        })
    } else {
        None
    };

    let anneal_config = AnnealConfig {
        iterations: config.iterations,
        seed: config.seed,
        ..AnnealConfig::default()
    };
    let sa_start = Instant::now();
    let anneal_result: AnnealResult = anneal(&mut env, &anneal_config);
    let sa_physical = anneal_result.best.metrics.routable && env.verify_physical();

    Ok(RouteFrontierRow {
        width,
        gates: circuit.num_gates(),
        inputs: circuit.num_inputs(),
        halo,
        row_major,
        sa: SaOutcome {
            routable: anneal_result.best.metrics.routable,
            physical: sa_physical,
            elapsed: sa_start.elapsed(),
            hpwl_improvement: anneal_result.hpwl_improvement(),
            best_hpwl: anneal_result.best_hpwl,
            wire_tiles: anneal_result.best.metrics.wire_tiles,
            vias: anneal_result.best.metrics.vias,
            accepted: anneal_result.accepted,
        },
    })
}

/// Validate the default AlphaFabric citation claims against measured rows.
pub fn check_default_claims(rows: &[RouteFrontierRow]) -> Result<(), String> {
    let width6_halo3 = find_row(rows, 6, 3)?;
    let width6_row = width6_halo3
        .row_major
        .ok_or_else(|| "claim check failed: width 6 row-major baseline was skipped".to_string())?;
    if width6_row.routable {
        return Err(
            "claim check failed: width 6 row-major unexpectedly routed at halo 3".to_string(),
        );
    }
    if !(width6_halo3.sa.routable && width6_halo3.sa.physical) {
        return Err("claim check failed: width 6 SA did not route+verify at halo 3".to_string());
    }

    let width8_halo3 = find_row(rows, 8, 3)?;
    if width8_halo3.sa.routable && width8_halo3.sa.physical {
        return Err("claim check failed: width 8 SA already route+verified at halo 3".to_string());
    }

    let width8_halo4 = find_row(rows, 8, 4)?;
    if !(width8_halo4.sa.routable && width8_halo4.sa.physical) {
        return Err("claim check failed: width 8 SA did not route+verify at halo 4".to_string());
    }

    Ok(())
}

/// Lowest halo in `rows` where `pred` holds.
pub fn frontier<F>(rows: &[RouteFrontierRow], pred: F) -> Option<&RouteFrontierRow>
where
    F: Fn(&RouteFrontierRow) -> bool,
{
    rows.iter()
        .filter(|row| pred(row))
        .min_by_key(|row| row.halo)
}

pub fn route_config() -> RouteConfig {
    RouteConfig {
        prefer_horizontal_first: true,
        no_crossings: true,
        max_z: 3,
    }
}

pub fn place_config(halo: usize) -> PlaceConfig {
    PlaceConfig {
        min_x: 3,
        min_y: 3,
        z: 0,
        max_columns: None,
        halo,
        max_width: None,
    }
}

fn find_row(
    rows: &[RouteFrontierRow],
    width: u32,
    halo: usize,
) -> Result<&RouteFrontierRow, String> {
    rows.iter()
        .find(|row| row.width == width && row.halo == halo)
        .ok_or_else(|| format!("missing benchmark row width={width} halo={halo}"))
}

fn unique_widths(cases: &[RouteFrontierCase]) -> Vec<u32> {
    let mut widths = Vec::new();
    for case in cases {
        if !widths.contains(&case.width) {
            widths.push(case.width);
        }
    }
    widths
}

fn halos_for_width(cases: &[RouteFrontierCase], width: u32) -> Vec<usize> {
    cases
        .iter()
        .filter(|case| case.width == width)
        .map(|case| case.halo)
        .collect()
}

fn should_measure_row_major(config: &RouteFrontierConfig, width: u32, halo: usize) -> bool {
    config.row_major_all
        || config.cases.is_none()
        || (width == 6 && halo == 3)
        || (width == 8 && halo == 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_row(
        width: u32,
        halo: usize,
        row_routable: Option<bool>,
        sa_routable: bool,
        sa_physical: bool,
    ) -> RouteFrontierRow {
        RouteFrontierRow {
            width,
            gates: 0,
            inputs: 0,
            halo,
            row_major: row_routable.map(|routable| RouteOutcome {
                routable,
                physical: None,
                elapsed: Duration::ZERO,
                wire_tiles: 0,
                vias: 0,
            }),
            sa: SaOutcome {
                routable: sa_routable,
                physical: sa_physical,
                elapsed: Duration::ZERO,
                hpwl_improvement: 0.0,
                best_hpwl: 0,
                wire_tiles: 0,
                vias: 0,
                accepted: 0,
            },
        }
    }

    #[test]
    fn madd_circuit_builds_and_maps() {
        for width in [6, 8] {
            let circuit = madd_circuit(width).expect("madd synthesizes");
            assert!(circuit.num_gates() > 0, "madd@{width} has gates");
            assert!(
                circuit.mapping_is_correct(),
                "madd@{width} mapped netlist must match AIG"
            );
        }
    }

    #[test]
    fn claim_checks_pass_on_citation_rows() {
        let rows = vec![
            synthetic_row(6, 3, Some(false), true, true),
            synthetic_row(8, 3, Some(false), false, false),
            synthetic_row(8, 4, None, true, true),
        ];
        check_default_claims(&rows).expect("citation claims should pass");
    }

    #[test]
    fn claim_checks_fail_when_width6_row_major_routes() {
        let rows = vec![
            synthetic_row(6, 3, Some(true), true, true),
            synthetic_row(8, 3, Some(false), false, false),
            synthetic_row(8, 4, None, true, true),
        ];
        let err = check_default_claims(&rows).expect_err("row-major route should fail claim");
        assert!(err.contains("width 6 row-major unexpectedly routed"));
    }

    #[test]
    fn frontier_picks_lowest_verified_halo() {
        let rows = vec![
            synthetic_row(8, 3, None, false, false),
            synthetic_row(8, 4, None, true, true),
            synthetic_row(8, 5, None, true, true),
        ];
        let best = frontier(&rows, |row| row.sa.routable && row.sa.physical)
            .expect("verified frontier exists");
        assert_eq!(best.halo, 4);
    }
}
