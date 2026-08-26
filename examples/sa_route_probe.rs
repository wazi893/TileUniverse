//! AlphaFabric SA route-frontier benchmark.
//!
//! Measures how far simulated-annealing placement extends the routable frontier
//! for the HLS `madd(a, b, c) = a * b + c` datapath used in the AlphaFabric
//! post. The default run reproduces the citation claims:
//!
//! - width 6: row-major is unroutable at halo 3; SA routes and verifies at halo 3.
//! - width 8: SA first routes and verifies at halo 4 in the default sweep.
//!
//! Run:
//!   cargo run --release --example sa_route_probe
//!
//! Useful knobs:
//!   cargo run --release --example sa_route_probe -- --cases 6:3,8:3,8:4
//!   cargo run --release --example sa_route_probe -- --widths 6,8 --halos 3,4 --iters 30000

use std::env;
use std::error::Error;
use std::fmt;
use std::io::Write;
use std::time::Duration;

use engine::synth::alphafabric::{
    RouteFrontierCase, RouteFrontierConfig, RouteFrontierRow, RouteOutcome, check_default_claims,
    frontier, madd_circuit, run_route_frontier,
};

const DEFAULT_WIDTHS: &[u32] = &[6, 8];

fn main() -> Result<(), Box<dyn Error>> {
    let Some(config) = parse_args()? else {
        print_help();
        return Ok(());
    };

    println!("AlphaFabric SA route-frontier benchmark");
    println!("circuit: madd(a, b, c) = a * b + c");
    println!(
        "config: {} iters={} seed=0x{:016X} route=no_crossings,max_z=3 row_major={} row_major_verify={}",
        config.case_label(),
        config.iterations,
        config.seed,
        if config.row_major_all {
            "all"
        } else {
            "claim-baseline"
        },
        config.verify_row_major
    );
    println!("host: {}-{}\n", env::consts::OS, env::consts::ARCH);

    let cases = config.resolved_cases();
    for width in unique_widths(&cases) {
        let circuit = madd_circuit(width)?;
        println!(
            "madd@{width}: {} gates, {} inputs, {} outputs",
            circuit.num_gates(),
            circuit.num_inputs(),
            circuit.num_outputs()
        );
        let _ = std::io::stdout().flush();
    }
    println!();

    let rows = run_route_frontier(&config)?;
    for row in &rows {
        print_detail_row(row);
        let _ = std::io::stdout().flush();
    }

    print_detail_table(&rows);
    print_markdown_summary(&rows);

    if config.check_claims {
        check_default_claims(&rows)?;
        println!("\nclaim checks: passed");
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct BenchConfig {
    inner: RouteFrontierConfig,
    check_claims: bool,
}

impl BenchConfig {
    fn resolved_cases(&self) -> Vec<RouteFrontierCase> {
        self.inner.resolved_cases()
    }

    fn case_label(&self) -> String {
        if let Some(cases) = &self.inner.cases {
            let labels = cases
                .iter()
                .map(|case| format!("{}:{}", case.width, case.halo))
                .collect::<Vec<_>>()
                .join(",");
            format!("cases={labels}")
        } else {
            format!(
                "widths={} halos={}",
                join_display(&self.inner.widths),
                join_display(&self.inner.halos)
            )
        }
    }
}

impl std::ops::Deref for BenchConfig {
    type Target = RouteFrontierConfig;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

fn print_detail_row(row: &RouteFrontierRow) {
    println!(
        "  madd@{:<2} halo {:>2}: row route={} phys={} {:>8} ms | SA route={} phys={} hpwl={:>5.1}% {:>8.0} ms",
        row.width,
        row.halo,
        route_outcome_cell(row.row_major),
        route_physical_cell(row.row_major),
        row_ms_cell(row.row_major),
        yes_no(row.sa.routable),
        yes_no(row.sa.physical),
        row.sa.hpwl_improvement * 100.0,
        ms(row.sa.elapsed),
    );
}

fn print_detail_table(rows: &[RouteFrontierRow]) {
    println!("\nDetailed results");
    println!(
        "{:<5} {:>5} {:>6} {:>5} {:>9} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9} {:>8}",
        "width",
        "gates",
        "inputs",
        "halo",
        "row_rt",
        "row_phys",
        "row_ms",
        "row_wire",
        "sa_rt",
        "sa_phys",
        "sa_ms",
        "sa_hpwl",
        "sa_best",
        "sa_wire",
        "accepted",
    );
    println!("{}", "-".repeat(136));
    for row in rows {
        println!(
            "{:<5} {:>5} {:>6} {:>5} {:>9} {:>8} {:>8} {:>9} {:>8} {:>8} {:>9.0} {:>8.1}% {:>9} {:>9} {:>8}",
            row.width,
            row.gates,
            row.inputs,
            row.halo,
            route_outcome_cell(row.row_major),
            route_physical_cell(row.row_major),
            row_ms_cell(row.row_major),
            row_wire_cell(row.row_major),
            yes_no(row.sa.routable),
            yes_no(row.sa.physical),
            ms(row.sa.elapsed),
            row.sa.hpwl_improvement * 100.0,
            row.sa.best_hpwl,
            row.sa.wire_tiles,
            row.sa.accepted,
        );
    }
}

fn print_markdown_summary(rows: &[RouteFrontierRow]) {
    println!("\nMarkdown citation table");
    println!(
        "| madd width | gates | measured halos | row-major routed frontier | SA verified frontier | claim-ready observation |"
    );
    println!("|---:|---:|---|---|---|---|");

    for width in unique_widths_from_rows(rows) {
        let width_rows: Vec<&RouteFrontierRow> =
            rows.iter().filter(|row| row.width == width).collect();
        if width_rows.is_empty() {
            continue;
        }
        let gates = width_rows[0].gates;
        let row_frontier = frontier(rows, |row| {
            row.width == width
                && row
                    .row_major
                    .map(|outcome| outcome.routable && outcome.physical.unwrap_or(true))
                    .unwrap_or(false)
        });
        let sa_frontier = frontier(rows, |row| {
            row.width == width && row.sa.routable && row.sa.physical
        });
        let observation = match (width, sa_frontier) {
            (6, Some(row))
                if row.halo == 3
                    && row
                        .row_major
                        .map(|outcome| !outcome.routable)
                        .unwrap_or(false) =>
            {
                "row-major fails at halo 3; SA routes and physically verifies at halo 3".to_string()
            }
            (8, Some(row)) if row.halo == 4 => {
                let h3 = width_rows.iter().find(|candidate| candidate.halo == 3);
                match h3 {
                    Some(h3) if !(h3.sa.routable && h3.sa.physical) => {
                        "SA fails at halo 3 and routes+verifies at halo 4".to_string()
                    }
                    _ => "SA routes and physically verifies at halo 4".to_string(),
                }
            }
            (_, Some(row)) => format!("SA routes and physically verifies at halo {}", row.halo),
            (_, None) => "no SA-routed+verified layout in this sweep".to_string(),
        };

        println!(
            "| {} | {} | {} | {} | {} | {} |",
            width,
            gates,
            join_display(&halos_from_rows(&width_rows)),
            row_frontier_cell(row_frontier),
            sa_frontier_cell(sa_frontier),
            observation,
        );
    }
}

fn parse_args() -> Result<Option<BenchConfig>, Box<dyn Error>> {
    let mut inner = RouteFrontierConfig::default();
    let mut check_claims = true;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--cases" => {
                let value = args
                    .next()
                    .ok_or("--cases needs comma-separated width:halo pairs")?;
                inner.cases = Some(parse_cases(&value)?);
            }
            "--widths" => {
                let value = args
                    .next()
                    .ok_or("--widths needs a comma-separated value")?;
                inner.widths = parse_csv(&value)?;
                inner.cases = None;
            }
            "--halos" => {
                let value = args.next().ok_or("--halos needs a comma-separated value")?;
                inner.halos = parse_csv(&value)?;
                inner.cases = None;
            }
            "--iters" => {
                let value = args.next().ok_or("--iters needs a value")?;
                inner.iterations = value.parse()?;
            }
            "--seed" => {
                let value = args.next().ok_or("--seed needs a value")?;
                inner.seed = parse_seed(&value)?;
            }
            "--row-major-all" => inner.row_major_all = true,
            "--no-row-verify" => inner.verify_row_major = false,
            "--no-claim-check" => check_claims = false,
            "--full" => {
                inner.widths = DEFAULT_WIDTHS.to_vec();
                inner.halos = vec![3, 4, 5, 6];
                inner.cases = None;
            }
            "--quick" => {
                inner.widths = vec![6];
                inner.halos = vec![3, 4];
                inner.cases = Some(vec![RouteFrontierCase { width: 6, halo: 3 }]);
                inner.iterations = 6_000;
                check_claims = false;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    if inner.widths.is_empty() {
        return Err("--widths must include at least one width".into());
    }
    if inner.halos.is_empty() {
        return Err("--halos must include at least one halo".into());
    }
    if matches!(&inner.cases, Some(cases) if cases.is_empty()) {
        return Err("--cases must include at least one width:halo pair".into());
    }
    if inner.iterations == 0 {
        return Err("--iters must be greater than zero".into());
    }

    Ok(Some(BenchConfig {
        inner,
        check_claims,
    }))
}

fn parse_csv<T>(value: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<T>().map_err(|err| err.into()))
        .collect()
}

fn parse_seed(value: &str) -> Result<u64, Box<dyn Error>> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        Ok(u64::from_str_radix(hex, 16)?)
    } else {
        Ok(value.parse()?)
    }
}

fn parse_cases(value: &str) -> Result<Vec<RouteFrontierCase>, Box<dyn Error>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            let (width, halo) = part
                .split_once(':')
                .ok_or_else(|| format!("case '{part}' must be formatted as width:halo"))?;
            Ok(RouteFrontierCase {
                width: width.trim().parse()?,
                halo: halo.trim().parse()?,
            })
        })
        .collect()
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

fn unique_widths_from_rows(rows: &[RouteFrontierRow]) -> Vec<u32> {
    let mut widths = Vec::new();
    for row in rows {
        if !widths.contains(&row.width) {
            widths.push(row.width);
        }
    }
    widths
}

fn halos_from_rows(rows: &[&RouteFrontierRow]) -> Vec<usize> {
    rows.iter().map(|row| row.halo).collect()
}

fn print_help() {
    println!("AlphaFabric SA route-frontier benchmark");
    println!();
    println!("USAGE:");
    println!("  cargo run --release --example sa_route_probe -- [options]");
    println!();
    println!("OPTIONS:");
    println!("  --cases <csv>         Width:halo pairs [default: 6:3,8:3,8:4]");
    println!("  --widths <csv>        Datapath widths to synthesize [default: 6,8]");
    println!("  --halos <csv>         Placement/routing halos to sweep [default: 3,4]");
    println!("  --iters <n>           SA iterations per width/halo [default: 30000]");
    println!("  --seed <n|0xhex>      SA seed [default: AnnealConfig::default()]");
    println!("  --full                Sweep widths 6,8 over halos 3,4,5,6");
    println!("  --row-major-all       Route row-major for every measured case");
    println!("  --no-row-verify       Skip physical verification for row-major routed layouts");
    println!("  --no-claim-check      Do not fail if the default AlphaFabric claims move");
    println!("  --quick               Case 6:3, 6000 iters, no claim check");
}

fn join_display<T: fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn route_outcome_cell(value: Option<RouteOutcome>) -> &'static str {
    match value {
        Some(outcome) => yes_no(outcome.routable),
        None => "skip",
    }
}

fn route_physical_cell(value: Option<RouteOutcome>) -> &'static str {
    match value {
        Some(outcome) => option_yes_no(outcome.physical),
        None => "n/a",
    }
}

fn row_ms(value: Option<RouteOutcome>) -> Option<f64> {
    value.map(|outcome| ms(outcome.elapsed))
}

fn row_ms_cell(value: Option<RouteOutcome>) -> String {
    match row_ms(value) {
        Some(value) => format!("{value:.0}"),
        None => "skip".to_string(),
    }
}

fn row_wire_cell(value: Option<RouteOutcome>) -> String {
    match value {
        Some(outcome) => outcome.wire_tiles.to_string(),
        None => "skip".to_string(),
    }
}

fn option_yes_no(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "no",
        None => "n/a",
    }
}

fn row_frontier_cell(row: Option<&RouteFrontierRow>) -> String {
    match row {
        Some(row) => match row.row_major {
            Some(outcome) => format!(
                "halo {} (phys={}, wire={}, vias={})",
                row.halo,
                option_yes_no(outcome.physical),
                outcome.wire_tiles,
                outcome.vias,
            ),
            None => "skipped".to_string(),
        },
        None => "none in sweep".to_string(),
    }
}

fn sa_frontier_cell(row: Option<&RouteFrontierRow>) -> String {
    match row {
        Some(row) => format!(
            "halo {} (phys={}, wire={}, vias={})",
            row.halo,
            yes_no(row.sa.physical),
            row.sa.wire_tiles,
            row.sa.vias,
        ),
        None => "none in sweep".to_string(),
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
