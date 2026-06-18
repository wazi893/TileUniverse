use std::io::{self, Write};

use engine::blueprint::Blueprint;
use engine::simulation::Simulation;
use engine::tile_meta::TileType;
use engine::tilemap::{HEIGHT, WIDTH};

fn parse_u64_any(s: &str) -> Option<u64> {
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).ok()
    } else {
        s.parse::<u64>().ok()
    }
}

fn parse_usize(s: &str) -> Option<usize> {
    s.parse::<usize>().ok()
}
fn clamp_steps(n: u32) -> u32 {
    n.min(1_000_000)
}

fn parse_tile_type(s: &str) -> Option<TileType> {
    match s.to_ascii_lowercase().as_str() {
        "wire" => Some(TileType::Wire),
        "and" => Some(TileType::And),
        "or" => Some(TileType::Or),
        "xor" => Some(TileType::Xor),
        "not" => Some(TileType::Not),
        "latch" => Some(TileType::Latch),
        "register8" | "reg8" => Some(TileType::Register8),
        "clockglobal" | "clock" | "clk" => Some(TileType::ClockGlobal),
        "vmspawner" | "vm_spawn" | "vmspn" => Some(TileType::VmSpawner),
        "vmstatus" | "vm_stat" | "vmst" => Some(TileType::VmStatus),
        _ => None,
    }
}

struct CliConfig {
    verbose: bool,
    delta_headers: bool,
}

fn print_region(sim: &Simulation, x0: usize, y0: usize, w: usize, h: usize) {
    sim.print_region(x0, y0, w, h);
}

fn print_tile_summary(sim: &Simulation, x: usize, y: usize) {
    if x >= WIDTH || y >= HEIGHT {
        println!("Tile ({},{}) is out of bounds", x, y);
        return;
    }
    let idx = y * WIDTH + x;
    let t = match sim.tilemap.get_tile(x, y) {
        Some(t) => t,
        None => {
            println!("Tile ({},{}) not accessible", x, y);
            return;
        }
    };
    let ty = match t.meta.tile_type {
        TileType::Wire => "Wire",
        TileType::And => "And",
        TileType::Or => "Or",
        TileType::Xor => "Xor",
        TileType::Not => "Not",
        TileType::Latch => "Latch",
        TileType::Register8 => "Register8",
        TileType::ClockGlobal => "ClockGlobal",
        TileType::VmSpawner => "VmSpawner",
        TileType::VmStatus => "VmStatus",
        TileType::QDemo => "QDemo",
        // EPIC 103: Arithmetic Tiles
        TileType::Add => "Add",
        TileType::Sub => "Sub",
        TileType::Mul => "Mul",
        TileType::Div => "Div",
        TileType::Mod => "Mod",
        TileType::Shl => "Shl",
        TileType::Shr => "Shr",
        // EPIC 103: Comparison Tiles
        TileType::Lt => "Lt",
        TileType::Gt => "Gt",
        TileType::Eq => "Eq",
        TileType::Neq => "Neq",
        TileType::Lte => "Lte",
        TileType::Gte => "Gte",
        // EPIC 103: Routing & Special
        TileType::Mux => "Mux",
        TileType::Zero => "Zero",
        TileType::Neg => "Neg",
        TileType::Abs => "Abs",
        // EPIC 104: Memory Tiles
        TileType::Ram => "Ram",
        TileType::Counter => "Counter",
        TileType::Const => "Const",
        // Wire Crossing Tiles
        TileType::Cross => "Cross",
        TileType::WireH => "WireH",
        TileType::WireV => "WireV",
        // Unidirectional Wires
        TileType::WireDown => "WireDown",
        TileType::WireRight => "WireRight",
        TileType::WireUp => "WireUp",
        TileType::WireLeft => "WireLeft",
        TileType::ComponentOutput => "ComponentOutput",
        // Bus Architecture
        TileType::BusInterface => "BusInterface",
        // Memory Controller
        TileType::MemoryPort => "MemoryPort",
        // EPIC 116: CPU Tiles
        TileType::CpuHead => "CpuHead",
        TileType::Register => "Register",
        TileType::Console => "Console",
        // CPU Building Blocks
        TileType::Decoder3to8 => "Decoder3to8",
        TileType::Mux8to1 => "Mux8to1",
        TileType::Mux4to1 => "Mux4to1",
        TileType::Demux1to8 => "Demux1to8",
        TileType::RegEnable => "RegEnable",
        TileType::ProgramCounter => "ProgramCounter",
        // SPRINT 66: Evolutionary Selection
        TileType::Selector => "Selector",
        // Ising Mode Tiles
        TileType::IsingNode => "IsingNode",
        TileType::IsingBias => "IsingBias",
        // Multi-Clock Domains
        TileType::ClockDivider => "ClockDivider",
        TileType::Synchronizer => "Synchronizer",
        // Phase 1: Fully Tile-Based CPU
        TileType::AddCarry => "AddCarry",
        TileType::BitSelect => "BitSelect",
        // Wire Crossing Tiles (unidirectional)
        TileType::WireCross => "WireCross",
        TileType::WireCrossVert => "WireCrossVert",
        TileType::VBusIn => "VBusIn",
        TileType::VBusOut => "VBusOut",
        TileType::ViaUp => "ViaUp",
        TileType::ViaDown => "ViaDown",
        TileType::SubBorrow => "SubBorrow",
        TileType::Mux16to1 => "Mux16to1",
        // Sprint 127: 64-bit Scaling Primitives
        TileType::Register64 => "Register64",
        TileType::CarryDetect => "CarryDetect",
        TileType::Decoder6to64 => "Decoder6to64",
        // Sprint 160: Weighted Via Tiles
        TileType::WeightedViaUp => "WeightedViaUp",
        TileType::WeightedViaDown => "WeightedViaDown",
        TileType::ThresholdViaUp => "ThresholdViaUp",
        TileType::ThresholdViaDown => "ThresholdViaDown",
    };
    let val = sim.tilemap.value(idx);
    println!("Tile ({},{})", x, y);
    println!("  Type: {}", ty);
    println!("  Value: 0x{:016x}", val);
    if let Some(info) = sim.explain_tile(x, y) {
        // Convert neighbor indices to coordinates
        let mut parts: Vec<String> = Vec::new();
        for opt in info.neighbors {
            if let Some(i) = opt {
                let ny = i / WIDTH;
                let nx = i % WIDTH;
                parts.push(format!("({},{})", nx, ny));
            } else {
                parts.push("(OOB)".to_string());
            }
        }
        println!(
            "  Last change: delta {}, old=0x{:016x} new=0x{:016x}, neighbors=[{}]",
            info.delta,
            info.old,
            info.new,
            parts.join(", ")
        );
    } else {
        println!("  Last change: none");
    }
    let _ = idx; // silence if not used in future tweaks
}

fn print_structural_report(sim: &Simulation, max_fanout: u32) {
    let mut issues = Vec::new();
    issues.extend(sim.check_fanout_bounds(max_fanout).issues);
    issues.extend(sim.check_unclocked_registers().issues);
    issues.extend(sim.check_orphan_logic().issues);

    if issues.is_empty() {
        println!("Structural issues: none.");
        return;
    }

    for issue in issues {
        match issue.kind {
            engine::simulation::StructuralIssueKind::FanoutExceeded { fanout, max } => {
                println!(
                    "FANOUT_EXCEEDED x={} y={} fanout={} max={}",
                    issue.x, issue.y, fanout, max
                );
            }
            engine::simulation::StructuralIssueKind::UnclockedRegister => {
                println!("UNCLOCKED_REGISTER x={} y={}", issue.x, issue.y);
            }
            engine::simulation::StructuralIssueKind::OrphanLogic => {
                println!("ORPHAN_LOGIC x={} y={}", issue.x, issue.y);
            }
        }
    }
}

fn print_nets_summary(sim: &Simulation) {
    let report = sim.analyze_nets();
    if report.nets.is_empty() {
        println!("Nets: none.");
        return;
    }

    let total = report.nets.len() as u32;
    let clock = report
        .nets
        .iter()
        .filter(|n| n.kind == engine::net::NetKind::Clock || n.has_clock)
        .count() as u32;
    let floating = report.nets.iter().filter(|n| n.is_floating).count() as u32;
    let mixed = report.nets.iter().filter(|n| n.is_mixed_region).count() as u32;

    println!(
        "Nets: total={} clock_nets={} floating_nets={} mixed_region_nets={}",
        total, clock, floating, mixed
    );
    for net in &report.nets {
        let kind = match net.kind {
            engine::net::NetKind::Data => "Data",
            engine::net::NetKind::Clock => "Clock",
        };
        let mut flags = Vec::new();
        if net.kind == engine::net::NetKind::Clock || net.has_clock {
            flags.push("clock");
        }
        if net.is_floating {
            flags.push("floating");
        }
        if net.is_mixed_region {
            flags.push("mixed_region");
        }
        let flags_str = if flags.is_empty() {
            "[]".to_string()
        } else {
            format!("[{}]", flags.join(","))
        };
        println!(
            "id={} kind={} nodes={} drivers={} sinks={} fanout_max={} bbox=({},{})-({},{}) flags={}",
            net.id,
            kind,
            net.node_count,
            net.driver_count,
            net.sink_count,
            net.fanout_max,
            net.x0,
            net.y0,
            net.x1,
            net.y1,
            flags_str
        );
    }
}

fn print_lint_result(res: &engine::lint::LintResult) {
    if res.summary.is_ok {
        println!("Lint: OK");
        return;
    }
    println!(
        "Lint: {} issues ({} error, {} warnings)",
        res.summary.issue_count, res.summary.error_count, res.summary.warning_count
    );
    for (idx, issue) in res.issues.iter().enumerate() {
        let sev = match issue.severity {
            engine::lint::LintSeverity::Info => "INFO",
            engine::lint::LintSeverity::Warning => "WARNING",
            engine::lint::LintSeverity::Error => "ERROR",
        };
        let kind_str = match &issue.kind {
            engine::lint::LintKind::StructuralIssue(k) => format!("Structural {:?}", k),
            engine::lint::LintKind::HighFanout { net_id, .. } => {
                format!("HighFanout net {}", net_id)
            }
            engine::lint::LintKind::FloatingNet { net_id } => format!("FloatingNet net {}", net_id),
            engine::lint::LintKind::MixedRegionNet { net_id } => {
                format!("MixedRegionNet net {}", net_id)
            }
        };
        println!("{}: {} {}", idx + 1, sev, kind_str);
    }
}

fn parse_region(args: &[&str]) -> Option<engine::physics_scenarios::Region> {
    if args.len() < 4 {
        return None;
    }
    let x0 = parse_usize(args[0])?;
    let y0 = parse_usize(args[1])?;
    let w = parse_usize(args[2])?;
    let h = parse_usize(args[3])?;
    Some(engine::physics_scenarios::Region {
        x0: x0 as u32,
        y0: y0 as u32,
        w: w as u32,
        h: h as u32,
    })
}

fn print_probe_trace(trace: &engine::probe::ProbeTrace) {
    let kind = match trace.kind {
        engine::probe::ProbeKind::Logic => "Logic",
        engine::probe::ProbeKind::FieldPower => "FieldPower",
        engine::probe::ProbeKind::FieldLogic => "FieldLogic",
        engine::probe::ProbeKind::FieldClock => "FieldClock",
    };
    println!(
        "Probe {} (x={}, y={}), steps={}",
        kind, trace.x, trace.y, trace.steps
    );
    println!("t  value");
    for (i, v) in trace.values.iter().enumerate() {
        println!("{}  0x{:016x}", i, v);
    }
}

fn main() {
    println!("Logic CLI ready. Type 'exit' to quit.");
    let mut sim = Simulation::new();
    let mut cfg = CliConfig {
        verbose: false,
        delta_headers: false,
    };
    let mut tick_counter: u64 = 0;

    loop {
        print!("> ");
        // Ensure prompt is visible before reading
        if io::stdout().flush().is_err() {
            println!("Output error. Exiting.");
            return;
        }

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            println!("Input error. Exiting.");
            return;
        }

        let line = input.trim();

        if line.eq_ignore_ascii_case("exit") {
            println!("Goodbye.");
            return;
        }

        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
        match cmd.as_str() {
            "help" => {
                println!("Commands:");
                println!("  clear");
                println!("  describe x y");
                println!("  eval x y");
                println!("  fill x0 y0 w h TYPE");
                println!("  help");
                println!("  lint [relaxed|strict]");
                println!("  load <path>");
                println!("  field_snapshot <power|logic|clock> x0 y0 w h");
                println!("  logic x y");
                println!("  nets_summary");
                println!("  print x0 y0 w h");
                println!("  reset");
                println!("  reset_logic x y");
                println!("  check_struct [max_fanout]");
                println!("  save <path>");
                println!("  step_fields n [power_decay d] [power_min m] [clock_period p]");
                println!("  probe_logic x y steps");
                println!(
                    "  probe_field <power|logic|clock> x y steps [power_decay d] [power_min m] [clock_period p]"
                );
                println!("  step_coupled n [inject d] [decay d] [max m] [threshold t]");
                println!("  eval_coupled x y");
                println!(
                    "  detect_patterns [fields f1,f2,...] [blob_threshold bt] [edge_threshold et] [min_blob_area a] [max_results n]"
                );
                println!("  render_region x0 y0 w h path [field_mode] [scale]");
                println!("  list_circuits");
                println!("  place_circuit NAME x y [allow_overwrite on|off]");
                println!("  list_agents");
                println!("  run_agent NAME [dry_run on|off]");
                println!("  list_organisms");
                println!("  place_organism NAME x y [allow_overwrite on|off]");
                println!("  run_organism NAME steps N [origin x y] [render on|off]");
                println!("  record_organism NAME steps N [origin x y] [dir PATH]");
                println!("  replay_run DIR");
                println!("  diff_runs DIR1 DIR2");
                println!("  gui_frames DIR");
                println!("  gui_demo_organism NAME steps N [origin x y]");
                println!("  gui_export DIR OUT_DIR");
                println!("  gui_info DIR");
                println!("  gui_frame DIR INDEX");
                println!("  gui_dump DIR OUT_DIR");
                println!("  run_ecosystem [steps N] [region x0 y0 w h] [render on|off]");
                println!("  feedback_snapshot x0 y0 w h");
                println!(
                    "  feedback [steps n] [charge_threshold ct] [heat_threshold ht] [logic_mask lv] [clamp m]"
                );
                println!("  quantum_demo [ticks N]");
                println!("  step_heat n [inject_scale s] [decay d] [max_heat m]");
                println!("  heat_snapshot x0 y0 w h");
                println!("  step_charge n [inject_scale s] [decay d] [max_charge m]");
                println!("  charge_snapshot x0 y0 w h");
                println!(
                    "  step_reaction n [heat_consume h] [charge_consume c] [heat_yield yh] [charge_yield yc] [min_heat mh] [min_charge mc] [max_heat H] [max_charge C]"
                );
                println!(
                    "  step_interact [steps n] [heat_affects_charge hac] [charge_affects_heat cah] [max_delta md] [clamp m]"
                );
                println!("  run_scenario NAME [steps n] [tick_stride k] [region x0 y0 w h]");
                println!(
                    "  run_experiment NAME PARAM start end steps [tick_stride k] [region x0 y0 w h]"
                );
                println!("  run_script NAME");
                println!("  demo PRESET [ticks N] [record DIR]");
                println!("  set_clock x y");
                println!("  set_logic x y VALUE");
                println!("  set_tile x y TYPE");
                println!("  snapshot x0 y0 w h");
                println!("  tick");
                println!("  wire x1 y1 x2 y2");
                println!("  dump <path>");
                println!("  exit");
            }
            "demo" => {
                let first = parts.next();
                let mut preset: u32 = 0;
                if let Some(p) = first {
                    preset = p.parse::<u32>().unwrap_or(0);
                }
                let mut ticks: u32 = 200;
                let mut record_dir: Option<String> = None;
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "ticks" => {
                            if args.len() >= 2 {
                                ticks = args[1].parse::<u32>().unwrap_or(200);
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "record" => {
                            if args.len() >= 2 {
                                record_dir = Some(args[1].to_string());
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                // Build preset
                let info = match engine::demo::build_preset(&mut sim, preset) {
                    Ok(i) => i,
                    Err(e) => {
                        println!("DemoInvalid(reason=\"{}\")", e.reason);
                        continue;
                    }
                };
                // Prepare state
                let rec_flag = if record_dir.is_some() { "yes" } else { "no" };
                println!(
                    "Demo starting: preset={} region={}x{} ticks={} record={}",
                    preset, info.w, info.h, ticks, rec_flag
                );

                let mut state = engine::demo::DemoState::new(&info);

                // If recording, do two identical runs in fresh sims and diff
                if let Some(dir) = record_dir.clone() {
                    use std::path::PathBuf;
                    let base = PathBuf::from(dir.clone());
                    let dir_a = base.clone();
                    // Run A
                    let mut sim_a = Simulation::new();
                    if let Err(e) = engine::demo::build_preset(&mut sim_a, preset) {
                        println!("DemoInvalid(reason=\"{}\")", e.reason);
                        continue;
                    }
                    let mut st_a = engine::demo::DemoState::new(&info);
                    let _man_a = engine::recorder::record_region_with(
                        &mut sim_a,
                        info.x0 as u32,
                        info.y0 as u32,
                        info.w as u32,
                        info.h as u32,
                        ticks,
                        &dir_a,
                        |s, _i| {
                            let _ = engine::demo::step_demo(s, &mut st_a);
                        },
                        &info.name,
                    );
                    // Run B (replay by regeneration)
                    let dir_b = base.with_extension("replay");
                    let mut sim_b = Simulation::new();
                    if let Err(e) = engine::demo::build_preset(&mut sim_b, preset) {
                        println!("DemoInvalid(reason=\"{}\")", e.reason);
                        continue;
                    }
                    let mut st_b = engine::demo::DemoState::new(&info);
                    let _man_b = engine::recorder::record_region_with(
                        &mut sim_b,
                        info.x0 as u32,
                        info.y0 as u32,
                        info.w as u32,
                        info.h as u32,
                        ticks,
                        &dir_b,
                        |s, _i| {
                            let _ = engine::demo::step_demo(s, &mut st_b);
                        },
                        &info.name,
                    );
                    // Diff
                    match engine::recorder::diff_runs(&dir_a, &dir_b) {
                        Ok((mismatches, len_diff)) => {
                            // checksum of last frame in A
                            let man = engine::recorder::read_manifest(&dir_a).unwrap();
                            let last = man
                                .artifacts
                                .iter()
                                .rev()
                                .find(|a| a.kind == "frame")
                                .map(|a| a.checksum_hex.clone())
                                .unwrap_or_else(|| "00000000".to_string());
                            let drift = mismatches + len_diff;
                            let replay = if drift == 0 { "ok" } else { "fail" };
                            println!(
                                "Demo OK: ticks={} replay={} drift={} checksum=0x{}",
                                ticks, replay, drift, last
                            );
                        }
                        Err(e) => {
                            println!("DemoInvalid(reason=\"diff error: {}\")", e);
                        }
                    }
                    continue;
                }

                // Non-recorded run: just step ticks with demo updates
                for _ in 0..ticks {
                    sim.tick();
                    let _ = engine::demo::step_demo(&mut sim, &mut state);
                }
                println!(
                    "Demo OK: ticks={} replay=na drift=0 checksum=0x00000000",
                    ticks
                );
            }
            "quantum_demo" => {
                // Build a fresh sim and register a 2-qubit Bell program on a QDemo tile
                let mut ticks: u32 = 6;
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    if flag.eq_ignore_ascii_case("ticks") {
                        let _ = args.remove(0);
                        if let Some(n) = args.first().cloned() {
                            let _ = args.remove(0);
                            ticks = n.parse::<u32>().unwrap_or(6);
                        }
                        continue;
                    }
                    break;
                }
                let mut sim = Simulation::new();
                let state = engine::quantum::QState::new_zero(2);
                let program = vec![
                    engine::quantum::QGate::H(0),
                    engine::quantum::QGate::CNot(0, 1),
                    engine::quantum::QGate::Measure(0),
                    engine::quantum::QGate::Measure(1),
                ];
                sim.register_qdemo_tile(10, 10, state, program, 0x1234_5678);
                let steps = ticks.min(32);
                for _ in 0..steps {
                    sim.tick();
                }
                // Print summary
                let summary = sim.get_quantum_tiles_debug();
                let mut all_bits: Vec<String> = Vec::new();
                for (id, n, amps, measured) in summary {
                    println!("QTile id={} qubits={}", id, n);
                    for (i, (re, im)) in amps.iter().enumerate() {
                        println!("  amp[{:02}]: [{:.6}, {:.6}]", i, re, im);
                    }
                    let mut mparts: Vec<String> = Vec::new();
                    for (i, mb) in measured.iter().enumerate() {
                        if let Some(b) = mb {
                            mparts.push(format!("q{}={}", i, b));
                        }
                    }
                    if mparts.is_empty() {
                        println!("  measured: none");
                    } else {
                        println!("  measured: {}", mparts.join(", "));
                    }
                    all_bits.extend(mparts);
                }
                let measured_summary = if all_bits.is_empty() {
                    "none".to_string()
                } else {
                    all_bits.join(", ")
                };
                println!("Quantum OK: ticks={} measured={}", steps, measured_summary);
            }
            "clear" => {
                if parts.next().is_some() {
                    println!("Usage: clear");
                } else {
                    for _ in 0..40 {
                        println!();
                    }
                }
            }
            "set_tile" => {
                let (sx, sy, st) = (parts.next(), parts.next(), parts.next());
                match (
                    sx.and_then(parse_usize),
                    sy.and_then(parse_usize),
                    st.and_then(parse_tile_type),
                ) {
                    (Some(x), Some(y), Some(tt)) if x < WIDTH && y < HEIGHT => {
                        sim.set_tile(x, y, tt);
                        if cfg.verbose {
                            println!("OK (Type={:?} at ({},{}))", tt, x, y);
                        } else {
                            println!("OK");
                        }
                    }
                    _ => println!("Invalid set_tile usage. Example: set_tile 10 10 Wire"),
                }
            }
            "set_logic" => {
                let (sx, sy, sv) = (parts.next(), parts.next(), parts.next());
                match (
                    sx.and_then(parse_usize),
                    sy.and_then(parse_usize),
                    sv.and_then(parse_u64_any),
                ) {
                    (Some(x), Some(y), Some(v)) if x < WIDTH && y < HEIGHT => {
                        let ok = sim.set_logic_value(x, y, v);
                        if ok {
                            if cfg.verbose {
                                println!("OK (0x{:016x} at ({},{}))", v, x, y);
                            } else {
                                println!("OK");
                            }
                        } else {
                            println!("OOB");
                        }
                    }
                    _ => println!("Invalid set_logic usage. Example: set_logic 10 10 0xFFFF"),
                }
            }
            "reset_logic" => {
                let (sx, sy) = (parts.next(), parts.next());
                match (sx.and_then(parse_usize), sy.and_then(parse_usize)) {
                    (Some(x), Some(y)) if x < WIDTH && y < HEIGHT => {
                        let _ = sim.set_logic_value(x, y, 0);
                        println!("OK");
                    }
                    _ => println!("Invalid reset_logic usage. Example: reset_logic 10 10"),
                }
            }
            "tick" => {
                sim.tick();
                if cfg.delta_headers {
                    println!("---- delta cycle {} ----", tick_counter);
                }
                tick_counter = tick_counter.saturating_add(1);
                println!("OK");
            }
            "eval" => {
                let (sx, sy) = (parts.next(), parts.next());
                match (sx.and_then(parse_usize), sy.and_then(parse_usize)) {
                    (Some(x), Some(y)) if x < WIDTH && y < HEIGHT => {
                        let changed = sim.eval_at(x, y);
                        if cfg.verbose {
                            let val = sim.tilemap.value_at(x, y).unwrap_or(0);
                            println!(
                                "{} (value=0x{:016x})",
                                if changed { "changed" } else { "nochange" },
                                val
                            );
                        } else {
                            println!("{}", if changed { "changed" } else { "nochange" });
                        }
                    }
                    _ => println!("Invalid eval usage. Example: eval 10 10"),
                }
            }
            "logic" => {
                let (sx, sy) = (parts.next(), parts.next());
                match (sx.and_then(parse_usize), sy.and_then(parse_usize)) {
                    (Some(x), Some(y)) if x < WIDTH && y < HEIGHT => {
                        let val = sim.tilemap.value_at(x, y).unwrap_or(0);
                        println!("Logic at ({},{}): 0x{:016x}", x, y, val);
                    }
                    _ => println!("OOB"),
                }
            }
            "snapshot" => {
                let (sx0, sy0, sw, sh) = (parts.next(), parts.next(), parts.next(), parts.next());
                match (
                    sx0.and_then(parse_usize),
                    sy0.and_then(parse_usize),
                    sw.and_then(parse_usize),
                    sh.and_then(parse_usize),
                ) {
                    (Some(x0), Some(y0), Some(w), Some(h)) => {
                        println!("{}x{} snapshot from ({},{})", w, h, x0, y0);
                        print_region(&sim, x0, y0, w, h);
                    }
                    _ => println!("Invalid snapshot usage. Example: snapshot 0 0 3 3"),
                }
            }
            "print" => {
                let (sx0, sy0, sw, sh) = (parts.next(), parts.next(), parts.next(), parts.next());
                match (
                    sx0.and_then(parse_usize),
                    sy0.and_then(parse_usize),
                    sw.and_then(parse_usize),
                    sh.and_then(parse_usize),
                ) {
                    (Some(x0), Some(y0), Some(w), Some(h)) => {
                        print_region(&sim, x0, y0, w, h);
                    }
                    _ => println!("Invalid print usage. Example: print 0 0 5 5"),
                }
            }
            "fill" => {
                let (sx0, sy0, sw, sh, st) = (
                    parts.next(),
                    parts.next(),
                    parts.next(),
                    parts.next(),
                    parts.next(),
                );
                match (
                    sx0.and_then(parse_usize),
                    sy0.and_then(parse_usize),
                    sw.and_then(parse_usize),
                    sh.and_then(parse_usize),
                    st.and_then(parse_tile_type),
                ) {
                    (Some(x0), Some(y0), Some(w), Some(h), Some(tt)) => {
                        let x1 = x0.saturating_add(w);
                        let y1 = y0.saturating_add(h);
                        if x0 >= WIDTH || y0 >= HEIGHT || x1 > WIDTH || y1 > HEIGHT {
                            println!("OOB");
                        } else {
                            for y in y0..y1 {
                                for x in x0..x1 {
                                    sim.set_tile(x, y, tt);
                                }
                            }
                            println!("Filled ({},{},{},{}) with {:?}", x0, y0, w, h, tt);
                        }
                    }
                    _ => println!("Invalid fill usage. Example: fill 0 0 5 5 Wire"),
                }
            }
            "save" => {
                if let Some(path) = parts.next() {
                    let bp = Blueprint::from_simulation(&sim);
                    match bp.save_to_path(path) {
                        Ok(()) => println!("Saved to {}", path),
                        Err(e) => println!("Save failed: {:?}", e),
                    }
                } else {
                    println!("Usage: save <path>");
                }
            }
            "load" => {
                if let Some(path) = parts.next() {
                    match Blueprint::load_from_path(path) {
                        Ok(bp) => match bp.apply_to_simulation(&mut sim) {
                            Ok(()) => println!("Loaded from {}", path),
                            Err(e) => println!("Apply failed: {:?}", e),
                        },
                        Err(e) => println!("Load failed: {:?}", e),
                    }
                } else {
                    println!("Usage: load <path>");
                }
            }
            "dump" => {
                if let Some(path) = parts.next() {
                    let bp = Blueprint::from_simulation(&sim);
                    match bp.save_to_path(path) {
                        Ok(()) => println!("Dumped to {}", path),
                        Err(e) => println!("Dump failed: {:?}", e),
                    }
                } else {
                    println!("Usage: dump <path>");
                }
            }
            "describe" => {
                let (sx, sy) = (parts.next(), parts.next());
                match (sx.and_then(parse_usize), sy.and_then(parse_usize)) {
                    (Some(x), Some(y)) => print_tile_summary(&sim, x, y),
                    _ => println!("Invalid describe usage. Example: describe 10 10"),
                }
            }
            "wire" => {
                let (sx1, sy1, sx2, sy2) = (parts.next(), parts.next(), parts.next(), parts.next());
                match (
                    sx1.and_then(parse_usize),
                    sy1.and_then(parse_usize),
                    sx2.and_then(parse_usize),
                    sy2.and_then(parse_usize),
                ) {
                    (Some(x1), Some(y1), Some(x2), Some(y2))
                        if x1 < WIDTH && y1 < HEIGHT && x2 < WIDTH && y2 < HEIGHT =>
                    {
                        sim.wire_line(x1, y1, x2, y2);
                        println!("Drew wire from ({},{}) to ({},{}).", x1, y1, x2, y2);
                    }
                    _ => println!("Invalid wire usage. Example: wire 1 1 3 1"),
                }
            }
            "set_clock" => {
                let (sx, sy) = (parts.next(), parts.next());
                match (sx.and_then(parse_usize), sy.and_then(parse_usize)) {
                    (Some(x), Some(y)) if x < WIDTH && y < HEIGHT => {
                        sim.set_tile(x, y, TileType::ClockGlobal);
                        println!("OK");
                    }
                    _ => println!("Invalid set_clock usage. Example: set_clock 10 10"),
                }
            }
            "check_struct" => {
                const DEFAULT_MAX_FANOUT: u32 = 16;
                match (parts.next(), parts.next()) {
                    (None, None) => {
                        print_structural_report(&sim, DEFAULT_MAX_FANOUT);
                    }
                    (Some(arg), None) => match arg.parse::<u32>() {
                        Ok(v) => print_structural_report(&sim, v),
                        Err(_) => println!("Usage: check_struct [max_fanout]"),
                    },
                    _ => println!("Usage: check_struct [max_fanout]"),
                }
            }
            "nets_summary" => match parts.next() {
                None => print_nets_summary(&sim),
                _ => println!("Usage: nets_summary"),
            },
            "step_fields" => {
                let steps = parts.next().and_then(|s| s.parse::<u32>().ok());
                if steps.is_none() {
                    println!("Usage: step_fields n [power_decay d] [power_min m] [clock_period p]");
                    continue;
                }
                let mut params = engine::fieldstep::FieldStepParams::default();
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "power_decay" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u8>() {
                                    params.power_decay = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "power_min" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u8>() {
                                    params.power_min = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "clock_period" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.clock_period = v.max(1);
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let steps = clamp_steps(steps.unwrap());
                sim.step_fields_n(&params, steps);
                println!(
                    "Fields stepped {} times (power_decay={}, power_min={}, clock_period={})",
                    steps, params.power_decay, params.power_min, params.clock_period
                );
            }
            "field_snapshot" => {
                let kind = parts.next();
                let coords: Vec<_> = parts.take(4).collect();
                if let (Some(k), Some(x0), Some(y0), Some(w), Some(h)) = (
                    kind,
                    coords.first(),
                    coords.get(1),
                    coords.get(2),
                    coords.get(3),
                ) {
                    if let (Some(x0), Some(y0), Some(w), Some(h)) = (
                        parse_usize(x0),
                        parse_usize(y0),
                        parse_usize(w),
                        parse_usize(h),
                    ) {
                        let fk = match k.to_ascii_lowercase().as_str() {
                            "power" => Some(engine::simulation::FieldKind::Power),
                            "logic" => Some(engine::simulation::FieldKind::Logic),
                            "clock" => Some(engine::simulation::FieldKind::Clock),
                            _ => None,
                        };
                        if let Some(fk) = fk {
                            let snap = sim.snapshot_field_region(fk, x0, y0, w, h);
                            println!("{} field [{}..{}, {}..{}]:", k, x0, x0 + w, y0, y0 + h);
                            for row in snap {
                                for val in row {
                                    print!("{:016x} ", val);
                                }
                                println!();
                            }
                        } else {
                            println!("Unknown field kind. Use power|logic|clock.");
                        }
                    } else {
                        println!("Usage: field_snapshot <power|logic|clock> x0 y0 w h");
                    }
                } else {
                    println!("Usage: field_snapshot <power|logic|clock> x0 y0 w h");
                }
            }
            "probe_logic" => {
                let (sx, sy, ssteps) = (parts.next(), parts.next(), parts.next());
                match (
                    sx.and_then(parse_usize),
                    sy.and_then(parse_usize),
                    ssteps.and_then(parse_usize),
                ) {
                    (Some(x), Some(y), Some(steps)) if x < WIDTH && y < HEIGHT => {
                        let steps = clamp_steps(steps as u32);
                        match sim.probe_logic(x as u32, y as u32, steps) {
                            Ok(trace) => print_probe_trace(&trace),
                            Err(engine::probe::ProbeError::OutOfBounds) => {
                                println!("Probe coordinates OOB")
                            }
                        }
                    }
                    _ => println!("Usage: probe_logic x y steps"),
                }
            }
            "probe_field" => {
                let kind = parts.next();
                let (sx, sy, ssteps) = (parts.next(), parts.next(), parts.next());
                let mut args: Vec<&str> = parts.collect();
                match (
                    kind,
                    sx.and_then(parse_usize),
                    sy.and_then(parse_usize),
                    ssteps.and_then(parse_usize),
                ) {
                    (Some(k), Some(x), Some(y), Some(steps)) if x < WIDTH && y < HEIGHT => {
                        let fk = match k.to_ascii_lowercase().as_str() {
                            "power" => Some(engine::simulation::FieldKind::Power),
                            "logic" => Some(engine::simulation::FieldKind::Logic),
                            "clock" => Some(engine::simulation::FieldKind::Clock),
                            _ => None,
                        };
                        if fk.is_none() {
                            println!("Unknown field kind. Use power|logic|clock.");
                            continue;
                        }
                        let mut params = engine::fieldstep::FieldStepParams::default();
                        while let Some(flag) = args.first().cloned() {
                            match flag {
                                "power_decay" => {
                                    if args.len() >= 2 {
                                        if let Ok(v) = args[1].parse::<u8>() {
                                            params.power_decay = v;
                                        }
                                        args.drain(0..2);
                                        continue;
                                    }
                                }
                                "power_min" => {
                                    if args.len() >= 2 {
                                        if let Ok(v) = args[1].parse::<u8>() {
                                            params.power_min = v;
                                        }
                                        args.drain(0..2);
                                        continue;
                                    }
                                }
                                "clock_period" => {
                                    if args.len() >= 2 {
                                        if let Ok(v) = args[1].parse::<u32>() {
                                            params.clock_period = v.max(1);
                                        }
                                        args.drain(0..2);
                                        continue;
                                    }
                                }
                                _ => {}
                            }
                            args.drain(0..1);
                        }
                        let steps = clamp_steps(steps as u32);
                        match sim.probe_field(fk.unwrap(), x as u32, y as u32, steps, &params) {
                            Ok(trace) => print_probe_trace(&trace),
                            Err(engine::probe::ProbeError::OutOfBounds) => {
                                println!("Probe coordinates OOB")
                            }
                        }
                    }
                    _ => println!(
                        "Usage: probe_field <power|logic|clock> x y steps [power_decay d] [power_min m] [clock_period p]"
                    ),
                }
            }
            "step_coupled" => {
                let steps = parts.next().and_then(|s| s.parse::<u32>().ok());
                if steps.is_none() {
                    println!("Usage: step_coupled n [inject d] [decay d] [max m] [threshold t]");
                    continue;
                }
                let mut params = engine::coupling::CoupledParams::default();
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "inject" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.logic_inject_power = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "decay" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.decay_per_step = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "max" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.max_power = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "threshold" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.field_to_logic_threshold = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let steps = clamp_steps(steps.unwrap());
                sim.coupled_step_n(steps, &params);
                println!(
                    "Coupled step: n={} inject={} decay={} max={} threshold={}",
                    steps,
                    params.logic_inject_power,
                    params.decay_per_step,
                    params.max_power,
                    params.field_to_logic_threshold
                );
            }
            "eval_coupled" => {
                let (sx, sy) = (parts.next(), parts.next());
                match (sx.and_then(parse_usize), sy.and_then(parse_usize)) {
                    (Some(x), Some(y)) if x < WIDTH && y < HEIGHT => {
                        let params = engine::coupling::CoupledParams::default();
                        let val = sim.eval_with_coupling(x, y, &params);
                        println!("Coupled eval at ({},{}): 0x{:016x}", x, y, val);
                    }
                    _ => println!("Usage: eval_coupled x y"),
                }
            }
            "detect_patterns" => {
                // Defaults
                let mut fields: Vec<engine::patterns::FieldSelect> = vec![
                    engine::patterns::FieldSelect::Heat,
                    engine::patterns::FieldSelect::Charge,
                ];
                let mut params = engine::patterns::PatternParams::default();

                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "fields" => {
                            if args.len() >= 2 {
                                let list = args[1];
                                let mut parsed = Vec::new();
                                for f in list.split(',') {
                                    match f.to_ascii_lowercase().as_str() {
                                        "power" => {
                                            parsed.push(engine::patterns::FieldSelect::Power)
                                        }
                                        "logic_field" => {
                                            parsed.push(engine::patterns::FieldSelect::LogicField)
                                        }
                                        "clock" => {
                                            parsed.push(engine::patterns::FieldSelect::Clock)
                                        }
                                        "heat" => parsed.push(engine::patterns::FieldSelect::Heat),
                                        "charge" => {
                                            parsed.push(engine::patterns::FieldSelect::Charge)
                                        }
                                        _ => {}
                                    }
                                }
                                if !parsed.is_empty() {
                                    fields = parsed;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "blob_threshold" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.blob_threshold = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "edge_threshold" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.edge_threshold = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "min_blob_area" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.min_blob_area = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "max_results" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.max_results = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }

                let report = engine::patterns::detect_patterns(&sim, &fields, &params);
                for s in report.field_summaries {
                    println!(
                        "Field={} blobs={} edges={} bbox=({},{})-({},{})",
                        s.field,
                        s.blobs.len(),
                        s.edges.count,
                        s.edges.x0,
                        s.edges.y0,
                        s.edges.x1,
                        s.edges.y1
                    );
                    for b in s.blobs {
                        println!(
                            "blob id={} area={} bbox=({},{})-({},{}) peak={} mean={}",
                            b.id, b.area, b.x0, b.y0, b.x1, b.y1, b.peak, b.mean
                        );
                    }
                }
                println!("oscillators={}", report.oscillators.len());
                for o in report.oscillators {
                    println!("osc x={} y={} reason={}", o.x, o.y, o.reason);
                }
            }
            "render_region" => {
                // render_region x0 y0 w h path [field_mode] [scale]
                let x0 = parts.next().and_then(parse_usize);
                let y0 = parts.next().and_then(parse_usize);
                let w = parts.next().and_then(parse_usize);
                let h = parts.next().and_then(parse_usize);
                let path = parts.next().map(|s| s.to_string());
                let field_mode = parts.next();
                let scale = parts.next().and_then(parse_usize);

                let (x0, y0, w, h, path) = match (x0, y0, w, h, path) {
                    (Some(x0), Some(y0), Some(w), Some(h), Some(p)) => (x0, y0, w, h, p),
                    _ => {
                        println!("Usage: render_region x0 y0 w h path [field_mode] [scale]");
                        continue;
                    }
                };

                let mode = match field_mode.map(|s| s.to_ascii_lowercase()) {
                    Some(m) if m == "logic" => engine::render::FieldLayerMode::Logic,
                    Some(m) if m == "heat" => engine::render::FieldLayerMode::Heat,
                    Some(m) if m == "charge" => engine::render::FieldLayerMode::Charge,
                    Some(m) if m == "heat_charge" => engine::render::FieldLayerMode::HeatAndCharge,
                    Some(m) if m == "logic_heat" => engine::render::FieldLayerMode::LogicAndHeat,
                    _ => engine::render::FieldLayerMode::Heat,
                };
                let scl = scale.unwrap_or(4).max(1) as u32;
                let params = engine::render::RenderParams {
                    fields: mode,
                    scale: scl,
                    format: engine::render::ImageFormat::Ppm,
                };
                match engine::render::render_region(
                    &sim, x0 as u32, y0 as u32, w as u32, h as u32, &params,
                ) {
                    Ok(img) => {
                        match std::fs::File::create(&path) {
                            Ok(mut f) => {
                                if let Err(e) = engine::render::write_ppm(&img, &mut f) {
                                    println!("Render error writing file: {}", e);
                                    continue;
                                }
                            }
                            Err(e) => {
                                println!("Render error creating file: {}", e);
                                continue;
                            }
                        }
                        let x1 = x0 + w;
                        let y1 = y0 + h;
                        let mode_str = match mode {
                            engine::render::FieldLayerMode::Logic => "logic",
                            engine::render::FieldLayerMode::Heat => "heat",
                            engine::render::FieldLayerMode::Charge => "charge",
                            engine::render::FieldLayerMode::HeatAndCharge => "heat_charge",
                            engine::render::FieldLayerMode::LogicAndHeat => "logic_heat",
                        };
                        println!(
                            "Render: region=({},{})-({},{}) mode={} scale={} -> {}",
                            x0, y0, x1, y1, mode_str, scl, path
                        );
                    }
                    Err(engine::render::RenderError::InvalidRegion) => {
                        println!("Render error: invalid region")
                    }
                    Err(engine::render::RenderError::OutOfBounds) => {
                        println!("Render error: out of bounds")
                    }
                    Err(engine::render::RenderError::InvalidScale) => {
                        println!("Render error: invalid scale")
                    }
                }
            }
            "list_circuits" => {
                let templates = engine::circuits::list_templates();
                println!("Circuits:");
                for t in templates {
                    let mut ports: Vec<String> = Vec::new();
                    for p in t.ports {
                        let role = match p.role {
                            engine::circuits::PortRole::Input => "in",
                            engine::circuits::PortRole::Output => "out",
                            engine::circuits::PortRole::Bidirectional => "io",
                            engine::circuits::PortRole::Clock => "clk",
                        };
                        ports.push(format!("{}:{}@{},{}", p.name, role, p.local_x, p.local_y));
                    }
                    let name = match t.kind {
                        engine::circuits::CircuitKind::WireBus => "wire_bus",
                        engine::circuits::CircuitKind::LatchStrip => "latch_strip",
                        engine::circuits::CircuitKind::RingOscillator => "ring_oscillator",
                        engine::circuits::CircuitKind::RegisterFile8 => "register_file_8",
                    };
                    println!(
                        "name={} size={}x{} ports=[{}]",
                        name,
                        t.width,
                        t.height,
                        ports.join("; ")
                    );
                }
            }
            "place_circuit" => {
                let name = parts.next();
                let x = parts.next().and_then(parse_usize);
                let y = parts.next().and_then(parse_usize);
                let mut allow_overwrite = false;
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "allow_overwrite" => {
                            if args.len() >= 2 {
                                allow_overwrite = matches!(
                                    args[1].to_ascii_lowercase().as_str(),
                                    "on" | "true" | "1"
                                );
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let (name, x, y) = match (name, x, y) {
                    (Some(n), Some(x), Some(y)) => (n.to_ascii_lowercase(), x, y),
                    _ => {
                        println!("Usage: place_circuit NAME x y [allow_overwrite on|off]");
                        continue;
                    }
                };
                let kind = match name.as_str() {
                    "wire_bus" => Some(engine::circuits::CircuitKind::WireBus),
                    "latch_strip" => Some(engine::circuits::CircuitKind::LatchStrip),
                    "ring_oscillator" => Some(engine::circuits::CircuitKind::RingOscillator),
                    _ => None,
                };
                if let Some(k) = kind {
                    let opts = engine::circuits::PlacementOptions { allow_overwrite };
                    match engine::circuits::place_circuit(&mut sim, k, x as u32, y as u32, &opts) {
                        Ok(sum) => {
                            println!(
                                "Circuit placed: name={} origin=({}, {}) size={}x{}",
                                name, sum.origin_x, sum.origin_y, sum.width, sum.height
                            );
                        }
                        Err(engine::circuits::CircuitError::UnknownCircuit) => {
                            println!("Unknown circuit")
                        }
                        Err(engine::circuits::CircuitError::OutOfBounds { .. }) => {
                            println!("Circuit placement OOB")
                        }
                        Err(engine::circuits::CircuitError::Collision { .. }) => {
                            println!("Circuit placement collision")
                        }
                        Err(engine::circuits::CircuitError::InvalidParams) => {
                            println!("Invalid circuit params")
                        }
                    }
                } else {
                    println!("Unknown circuit name");
                }
            }
            "list_agents" => {
                let plans = engine::agent::list_plans();
                println!("Agents:");
                for p in plans {
                    let cat = match p.category {
                        engine::agent::AgentCategory::CircuitPlacement => "CircuitPlacement",
                    };
                    println!("name={} category={} summary={}", p.name, cat, p.summary);
                }
            }
            "run_agent" => {
                let name = parts.next();
                let mut dry_run = false;
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "dry_run" => {
                            if args.len() >= 2 {
                                dry_run = matches!(
                                    args[1].to_ascii_lowercase().as_str(),
                                    "on" | "true" | "1"
                                );
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                if let Some(n) = name {
                    match engine::agent::run_plan_named(&mut sim, n, dry_run) {
                        Ok(res) => {
                            println!(
                                "Agent {} result: {}",
                                n,
                                if res.error.is_none() { "OK" } else { "ERROR" }
                            );
                            for s in res.log {
                                println!(
                                    "step={} label={} ok={} notes={}",
                                    s.index, s.label, s.ok, s.notes
                                );
                            }
                            if let Some(e) = res.error {
                                println!("error={}", e);
                            }
                        }
                        Err(e) => println!("Agent error: {}", e),
                    }
                } else {
                    println!("Usage: run_agent NAME [dry_run on|off]");
                }
            }
            "list_organisms" => {
                let orgs = engine::organisms::list_organisms();
                println!("Organisms:");
                for o in orgs {
                    println!(
                        "name={} size={}x{} zones={} circuits={}",
                        o.name,
                        o.width,
                        o.height,
                        o.zones.len(),
                        o.circuits.len()
                    );
                }
            }
            "place_organism" => {
                let name = parts.next();
                let x = parts.next().and_then(parse_usize);
                let y = parts.next().and_then(parse_usize);
                let mut allow_overwrite = false;
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "allow_overwrite" => {
                            if args.len() >= 2 {
                                allow_overwrite = matches!(
                                    args[1].to_ascii_lowercase().as_str(),
                                    "on" | "true" | "1"
                                );
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let (name, x, y) = match (name, x, y) {
                    (Some(n), Some(x), Some(y)) => (n.to_ascii_lowercase(), x, y),
                    _ => {
                        println!("Usage: place_organism NAME x y [allow_overwrite on|off]");
                        continue;
                    }
                };
                let kind = match name.as_str() {
                    "heat_emitter" => Some(engine::organisms::OrganismKind::HeatEmitter),
                    "ring_cluster" => Some(engine::organisms::OrganismKind::RingCluster),
                    "charge_crawler" => Some(engine::organisms::OrganismKind::ChargeCrawler),
                    "wire_bundle" => Some(engine::organisms::OrganismKind::WireBundle),
                    "oscillator_nest" => Some(engine::organisms::OrganismKind::OscillatorNest),
                    _ => None,
                };
                if let Some(k) = kind {
                    match engine::organisms::place_organism(
                        &mut sim,
                        k,
                        x as u32,
                        y as u32,
                        allow_overwrite,
                    ) {
                        Ok(()) => println!("Organism placed: name={} origin=({}, {})", name, x, y),
                        Err(engine::organisms::OrganismError::UnknownOrganism) => {
                            println!("Unknown organism")
                        }
                        Err(engine::organisms::OrganismError::OutOfBounds) => {
                            println!("Organism placement OOB")
                        }
                        Err(engine::organisms::OrganismError::Collision) => {
                            println!("Organism placement collision")
                        }
                        Err(engine::organisms::OrganismError::InvalidParams) => {
                            println!("Invalid organism params")
                        }
                    }
                } else {
                    println!("Unknown organism name");
                }
            }
            "run_organism" => {
                let name = parts.next();
                let mut steps_opt: Option<u32> = None;
                let mut origin: Option<(usize, usize)> = None;
                let mut render = false;
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "steps" => {
                            if args.len() >= 2 {
                                steps_opt = args[1].parse::<u32>().ok();
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "origin" => {
                            if args.len() >= 3 {
                                if let (Some(x), Some(y)) =
                                    (parse_usize(args[1]), parse_usize(args[2]))
                                {
                                    origin = Some((x, y));
                                }
                                args.drain(0..3);
                                continue;
                            }
                        }
                        "render" => {
                            if args.len() >= 2 {
                                render = matches!(
                                    args[1].to_ascii_lowercase().as_str(),
                                    "on" | "true" | "1"
                                );
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let (name, steps) = match (name, steps_opt) {
                    (Some(n), Some(s)) => (n.to_ascii_lowercase(), s),
                    _ => {
                        println!("Usage: run_organism NAME steps N [origin x y] [render on|off]");
                        continue;
                    }
                };
                let kind = match name.as_str() {
                    "heat_emitter" => Some(engine::organisms::OrganismKind::HeatEmitter),
                    "ring_cluster" => Some(engine::organisms::OrganismKind::RingCluster),
                    "charge_crawler" => Some(engine::organisms::OrganismKind::ChargeCrawler),
                    "wire_bundle" => Some(engine::organisms::OrganismKind::WireBundle),
                    "oscillator_nest" => Some(engine::organisms::OrganismKind::OscillatorNest),
                    _ => None,
                };
                let (ox, oy) = origin.unwrap_or((0, 0));
                if let Some(k) = kind {
                    let params = engine::organisms::OrganismParams {
                        steps,
                        render_each: render,
                    };
                    let res =
                        engine::organisms::run_organism(&mut sim, k, ox as u32, oy as u32, &params);
                    println!("Organism {} run: steps_run={}", name, res.steps_run);
                    for a in res.actions {
                        println!("action {}", a);
                    }
                    if let Some(e) = res.error {
                        println!("error {}", e);
                    }
                } else {
                    println!("Unknown organism name");
                }
            }
            "record_organism" => {
                let name = parts.next();
                let mut steps_opt: Option<u32> = None;
                let mut origin: Option<(usize, usize)> = None;
                let mut out_dir: Option<String> = None;
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "steps" => {
                            if args.len() >= 2 {
                                steps_opt = args[1].parse::<u32>().ok();
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "origin" => {
                            if args.len() >= 3 {
                                if let (Some(x), Some(y)) =
                                    (parse_usize(args[1]), parse_usize(args[2]))
                                {
                                    origin = Some((x, y));
                                }
                                args.drain(0..3);
                                continue;
                            }
                        }
                        "dir" => {
                            if args.len() >= 2 {
                                out_dir = Some(args[1].to_string());
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let (name, steps) = match (name, steps_opt) {
                    (Some(n), Some(s)) => (n.to_ascii_lowercase(), s),
                    _ => {
                        println!("Usage: record_organism NAME steps N [origin x y] [dir PATH]");
                        continue;
                    }
                };
                let kind = match name.as_str() {
                    "heat_emitter" => Some(engine::organisms::OrganismKind::HeatEmitter),
                    "ring_cluster" => Some(engine::organisms::OrganismKind::RingCluster),
                    "charge_crawler" => Some(engine::organisms::OrganismKind::ChargeCrawler),
                    "wire_bundle" => Some(engine::organisms::OrganismKind::WireBundle),
                    "oscillator_nest" => Some(engine::organisms::OrganismKind::OscillatorNest),
                    _ => None,
                };
                let (ox, oy) = origin.unwrap_or((0, 0));
                let dir = out_dir.unwrap_or_else(|| "record_out".to_string());
                if let Some(k) = kind {
                    // Ensure placed once before recording
                    match engine::organisms::place_organism(
                        &mut sim, k, ox as u32, oy as u32, false,
                    ) {
                        Ok(()) => {}
                        Err(engine::organisms::OrganismError::Collision) => {
                            println!("Placement collision; aborting record.");
                            continue;
                        }
                        Err(engine::organisms::OrganismError::OutOfBounds) => {
                            println!("Placement OOB; aborting record.");
                            continue;
                        }
                        Err(_) => {}
                    }
                    match engine::recorder::record_organism(
                        &mut sim,
                        k,
                        ox as u32,
                        oy as u32,
                        steps,
                        std::path::Path::new(&dir),
                    ) {
                        Ok(man) => {
                            println!(
                                "Record: kind={} name={} steps={} dir={}",
                                man.run_kind, man.name, man.steps, dir
                            );
                            println!(
                                "Manifest: {}",
                                std::path::Path::new(&dir).join("manifest.txt").display()
                            );
                        }
                        Err(e) => println!("Record error: {}", e),
                    }
                } else {
                    println!("Unknown organism name");
                }
            }
            "replay_run" => {
                let dir = parts.next();
                if let Some(d) = dir {
                    match engine::recorder::replay_run(std::path::Path::new(d)) {
                        Ok(true) => println!("Replay OK"),
                        Ok(false) => println!("Replay mismatch"),
                        Err(e) => println!("Replay error: {}", e),
                    }
                } else {
                    println!("Usage: replay_run DIR");
                }
            }
            "diff_runs" => {
                let d1 = parts.next();
                let d2 = parts.next();
                match (d1, d2) {
                    (Some(a), Some(b)) => match engine::recorder::diff_runs(
                        std::path::Path::new(a),
                        std::path::Path::new(b),
                    ) {
                        Ok((m, len_diff)) => {
                            println!("Diff: mismatches={} length_diff={}", m, len_diff)
                        }
                        Err(e) => println!("Diff error: {}", e),
                    },
                    _ => println!("Usage: diff_runs DIR1 DIR2"),
                }
            }
            "gui_frames" => {
                let dir = parts.next();
                if let Some(d) = dir {
                    match engine::gui_backend::open_recorded_frames(std::path::Path::new(d)) {
                        Ok(stream) => {
                            for i in 0..(stream.len() as u32) {
                                if let (Some(m), Some(_b)) = (stream.meta(i), stream.frame_bytes(i))
                                {
                                    let field = match m.field {
                                        engine::gui_backend::FrameFieldKind::Heat => "heat",
                                    };
                                    println!(
                                        "frame index={} t={} w={} h={} field={} checksum=0x{:08x}",
                                        m.index, m.t, m.width, m.height, field, m.checksum
                                    );
                                }
                            }
                        }
                        Err(_) => {
                            println!("GUI frames error");
                        }
                    }
                } else {
                    println!("Usage: gui_frames DIR");
                }
            }
            "gui_demo_organism" => {
                let name = parts.next();
                let mut steps_opt: Option<u32> = None;
                let mut origin: Option<(usize, usize)> = None;
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "steps" => {
                            if args.len() >= 2 {
                                steps_opt = args[1].parse::<u32>().ok();
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "origin" => {
                            if args.len() >= 3 {
                                if let (Some(x), Some(y)) =
                                    (parse_usize(args[1]), parse_usize(args[2]))
                                {
                                    origin = Some((x, y));
                                }
                                args.drain(0..3);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let (name, steps) = match (name, steps_opt) {
                    (Some(n), Some(s)) => (n.to_ascii_lowercase(), s),
                    _ => {
                        println!("Usage: gui_demo_organism NAME steps N [origin x y]");
                        continue;
                    }
                };
                let (ox, oy) = origin.unwrap_or((0, 0));
                match engine::gui_backend::run_organism_frames(
                    &mut sim, &name, ox as u32, oy as u32, steps,
                ) {
                    Ok(stream) => {
                        for i in 0..(stream.len() as u32) {
                            if let Some(m) = stream.meta(i) {
                                let field = match m.field {
                                    engine::gui_backend::FrameFieldKind::Heat => "heat",
                                };
                                println!(
                                    "frame index={} t={} w={} h={} field={} checksum=0x{:08x}",
                                    m.index, m.t, m.width, m.height, field, m.checksum
                                );
                            }
                        }
                    }
                    Err(_) => println!("GUI demo error"),
                }
            }
            "gui_export" => {
                let dir = parts.next();
                let out = parts.next();
                match (dir, out) {
                    (Some(d), Some(o)) => {
                        match engine::gui_backend::open_recorded_frames(std::path::Path::new(d)) {
                            Ok(stream) => {
                                let outp = std::path::Path::new(o);
                                let _ = std::fs::create_dir_all(outp);
                                for i in 0..(stream.len() as u32) {
                                    if let Some(bytes) = stream.frame_bytes(i) {
                                        let fname = format!("frame_{:04}.ppm", i);
                                        let mut f = match std::fs::File::create(outp.join(&fname)) {
                                            Ok(f) => f,
                                            Err(e) => {
                                                println!("export error: {}", e);
                                                break;
                                            }
                                        };
                                        let _ = f.write_all(bytes);
                                    }
                                }
                                println!("GUI export OK");
                            }
                            Err(_) => println!("GUI frames error"),
                        }
                    }
                    _ => println!("Usage: gui_export DIR OUT_DIR"),
                }
            }
            "gui_info" => {
                let dir = parts.next();
                if let Some(d) = dir {
                    match engine::gui_api::load_run(std::path::Path::new(d)) {
                        Ok(info) => {
                            println!(
                                "GUI Run: frames={} size={}x{} fields={}",
                                info.frame_count,
                                info.width,
                                info.height,
                                info.fields.len()
                            );
                        }
                        Err(e) => println!("GUI info error: {}", e),
                    }
                } else {
                    println!("Usage: gui_info DIR");
                }
            }
            "gui_frame" => {
                let dir = parts.next();
                let idx = parts.next().and_then(parse_usize);
                match (dir, idx) {
                    (Some(d), Some(i)) => {
                        match engine::gui_api::load_frame(std::path::Path::new(d), i) {
                            Ok(f) => {
                                let m = f.meta;
                                let field = match m.field {
                                    engine::gui_backend::FrameFieldKind::Heat => "heat",
                                };
                                println!(
                                    "frame index={} t={} w={} h={} field={} checksum=0x{:08x}",
                                    m.index, m.t, m.width, m.height, field, m.checksum
                                );
                            }
                            Err(e) => println!("GUI frame error: {}", e),
                        }
                    }
                    _ => println!("Usage: gui_frame DIR INDEX"),
                }
            }
            "gui_dump" => {
                let dir = parts.next();
                let out = parts.next();
                match (dir, out) {
                    (Some(d), Some(o)) => {
                        match engine::gui_api::stream_from_recorded(std::path::Path::new(d)) {
                            Ok(s) => {
                                let p = std::path::Path::new(o);
                                let _ = std::fs::create_dir_all(p);
                                for i in 0..(s.frames.len() as u32) {
                                    if let Some(bytes) = s.frames.frame_bytes(i) {
                                        let fname = format!("frame_{:04}.ppm", i);
                                        if let Ok(mut f) = std::fs::File::create(p.join(&fname)) {
                                            let _ = f.write_all(bytes);
                                        }
                                    }
                                }
                                println!("GUI dump OK");
                            }
                            Err(e) => println!("GUI dump error: {}", e),
                        }
                    }
                    _ => println!("Usage: gui_dump DIR OUT_DIR"),
                }
            }
            "run_ecosystem" => {
                // Minimal two-species demo config
                let mut steps: u32 = 3;
                let mut region: Option<(usize, usize, usize, usize)> = Some((0, 0, 32, 16));
                let mut render = true;
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "steps" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    steps = v.min(1000);
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "region" => {
                            if args.len() >= 5 {
                                if let (Some(x0), Some(y0), Some(w), Some(h)) = (
                                    parse_usize(args[1]),
                                    parse_usize(args[2]),
                                    parse_usize(args[3]),
                                    parse_usize(args[4]),
                                ) {
                                    region = Some((x0, y0, w, h));
                                }
                                args.drain(0..5);
                                continue;
                            }
                        }
                        "render" => {
                            if args.len() >= 2 {
                                render = matches!(
                                    args[1].to_ascii_lowercase().as_str(),
                                    "on" | "true" | "1"
                                );
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let (x0, y0, w, h) = region.unwrap_or((0, 0, 32, 16));
                let species = vec![
                    engine::ecosystem::SpeciesConfig {
                        name: "predator",
                        kind: engine::organisms::OrganismKind::RingCluster,
                        spawn_points: vec![(x0 as u32 + 2, y0 as u32 + 2)],
                        policy: engine::ecosystem::SpeciesPolicy {
                            move_axis: Some(engine::ecosystem::Axis::X),
                            allow_overwrite: false,
                            max_instances: 1,
                        },
                    },
                    engine::ecosystem::SpeciesConfig {
                        name: "crawler",
                        kind: engine::organisms::OrganismKind::ChargeCrawler,
                        spawn_points: vec![(x0 as u32 + 1, y0 as u32 + (h as u32 / 2))],
                        policy: engine::ecosystem::SpeciesPolicy {
                            move_axis: Some(engine::ecosystem::Axis::X),
                            allow_overwrite: false,
                            max_instances: 1,
                        },
                    },
                ];
                let cfg = engine::ecosystem::EcosystemConfig {
                    region: engine::ecosystem::Region {
                        x0: x0 as u32,
                        y0: y0 as u32,
                        w: w as u32,
                        h: h as u32,
                    },
                    steps,
                    species,
                    budgets: engine::ecosystem::ResourceBudgets {
                        max_heat: 1000,
                        max_charge: 1000,
                        per_step_cap_heat: 10,
                        per_step_cap_charge: 10,
                    },
                    claim_mode: engine::ecosystem::ClaimMode::Exclusive,
                    interaction: engine::ecosystem::InteractionRules {
                        predator_prey: true,
                        cooperation: false,
                        thresholds: engine::ecosystem::Thresholds {
                            predator_heat: 1,
                            cooperate_charge: 255,
                        },
                    },
                    render_each: render,
                };
                let out = engine::ecosystem::run_ecosystem(&mut sim, &cfg);
                println!("Ecosystem: steps_run={}", out.steps_run);
                for line in out.log {
                    println!("{}", line);
                }
            }
            "feedback_snapshot" => {
                let coords: Vec<_> = parts.take(4).collect();
                if let (Some(x0), Some(y0), Some(w), Some(h)) =
                    (coords.first(), coords.get(1), coords.get(2), coords.get(3))
                {
                    if let (Some(x0), Some(y0), Some(w), Some(h)) = (
                        parse_usize(x0),
                        parse_usize(y0),
                        parse_usize(w),
                        parse_usize(h),
                    ) {
                        let snap = sim.snapshot_feedback_logic_region(x0, y0, w, h);
                        println!(
                            "feedback logic field [{}..{}, {}..{}]:",
                            x0,
                            x0 + w,
                            y0,
                            y0 + h
                        );
                        for row in snap {
                            for val in row {
                                print!("{:016x} ", val);
                            }
                            println!();
                        }
                    } else {
                        println!("Usage: feedback_snapshot x0 y0 w h");
                    }
                } else {
                    println!("Usage: feedback_snapshot x0 y0 w h");
                }
            }
            "feedback" => {
                let mut steps: Option<u32> = None;
                let mut params = engine::physics_feedback::FeedbackParams::default();
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "steps" => {
                            if args.len() >= 2 {
                                steps = args[1].parse::<u32>().ok();
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "charge_threshold" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.charge_to_logic_threshold = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "heat_threshold" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.heat_to_logic_threshold = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "logic_mask" => {
                            if args.len() >= 2 {
                                if let Some(v) = parse_u64_any(args[1]) {
                                    params.logic_mask = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "clamp" => {
                            if args.len() >= 2 {
                                if let Some(v) = parse_u64_any(args[1]) {
                                    params.clamp_value = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let steps = clamp_steps(steps.unwrap_or(1));
                sim.feedback_n(steps, &params);
                println!(
                    "Feedback: steps={} charge_threshold={} heat_threshold={} logic_mask=0x{:016x} clamp=0x{:016x}",
                    steps,
                    params.charge_to_logic_threshold,
                    params.heat_to_logic_threshold,
                    params.logic_mask,
                    params.clamp_value
                );
            }
            "step_heat" => {
                let steps = parts.next().and_then(|s| s.parse::<u32>().ok());
                if steps.is_none() {
                    println!("Usage: step_heat n [inject_scale s] [decay d] [max_heat m]");
                    continue;
                }
                let mut params = engine::heat::HeatParams::default();
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "inject_scale" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.inject_scale = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "decay" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.decay_per_step = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "max_heat" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.max_heat = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let steps = clamp_steps(steps.unwrap());
                sim.step_heat_n(steps, &params);
                println!(
                    "Heat step: n={} inject_scale={} decay={} max_heat={}",
                    steps, params.inject_scale, params.decay_per_step, params.max_heat
                );
            }
            "heat_snapshot" => {
                let coords: Vec<_> = parts.take(4).collect();
                if let (Some(x0), Some(y0), Some(w), Some(h)) =
                    (coords.first(), coords.get(1), coords.get(2), coords.get(3))
                {
                    if let (Some(x0), Some(y0), Some(w), Some(h)) = (
                        parse_usize(x0),
                        parse_usize(y0),
                        parse_usize(w),
                        parse_usize(h),
                    ) {
                        let snap =
                            sim.snapshot_heat_region(x0 as u32, y0 as u32, w as u32, h as u32);
                        println!("heat field [{}..{}, {}..{}]:", x0, x0 + w, y0, y0 + h);
                        for row in snap {
                            for val in row {
                                print!("{:08x} ", val);
                            }
                            println!();
                        }
                    } else {
                        println!("Usage: heat_snapshot x0 y0 w h");
                    }
                } else {
                    println!("Usage: heat_snapshot x0 y0 w h");
                }
            }
            "step_charge" => {
                let steps = parts.next().and_then(|s| s.parse::<u32>().ok());
                if steps.is_none() {
                    println!("Usage: step_charge n [inject_scale s] [decay d] [max_charge m]");
                    continue;
                }
                let mut params = engine::charge::ChargeParams::default();
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "inject_scale" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.inject_scale = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "decay" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.decay_per_step = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "max_charge" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.max_charge = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let steps = clamp_steps(steps.unwrap());
                sim.step_charge_n(steps, &params);
                println!(
                    "Charge step: n={} inject_scale={} decay={} max_charge={}",
                    steps, params.inject_scale, params.decay_per_step, params.max_charge
                );
            }
            "charge_snapshot" => {
                let coords: Vec<_> = parts.take(4).collect();
                if let (Some(x0), Some(y0), Some(w), Some(h)) =
                    (coords.first(), coords.get(1), coords.get(2), coords.get(3))
                {
                    if let (Some(x0), Some(y0), Some(w), Some(h)) = (
                        parse_usize(x0),
                        parse_usize(y0),
                        parse_usize(w),
                        parse_usize(h),
                    ) {
                        let snap =
                            sim.snapshot_charge_region(x0 as u32, y0 as u32, w as u32, h as u32);
                        println!("charge field [{}..{}, {}..{}]:", x0, x0 + w, y0, y0 + h);
                        for row in snap {
                            for val in row {
                                print!("{:08x} ", val);
                            }
                            println!();
                        }
                    } else {
                        println!("Usage: charge_snapshot x0 y0 w h");
                    }
                } else {
                    println!("Usage: charge_snapshot x0 y0 w h");
                }
            }
            "diffuse_heat" => {
                let steps = parts.next().and_then(|s| s.parse::<u32>().ok());
                if steps.is_none() {
                    println!("Usage: diffuse_heat n [spread s/d] [decay d] [max m]");
                    continue;
                }
                let mut params = engine::diffuse::DiffuseParams::default();
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "spread" => {
                            if args.len() >= 2 {
                                let frac = args[1];
                                if let Some((num, den)) = frac.split_once('/') {
                                    if let (Ok(n), Ok(d)) = (num.parse::<u32>(), den.parse::<u32>())
                                    {
                                        params.spread_numerator = n.min(d);
                                        params.spread_denominator = d.max(1);
                                    }
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "decay" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.decay_per_step = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "max" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.max_value = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let steps = clamp_steps(steps.unwrap());
                sim.step_heat_diffuse_n(steps, &params);
                println!(
                    "Heat diffusion: n={} spread={}/{} decay={} max={}",
                    steps,
                    params.spread_numerator,
                    params.spread_denominator,
                    params.decay_per_step,
                    params.max_value
                );
            }
            "diffuse_charge" => {
                let steps = parts.next().and_then(|s| s.parse::<u32>().ok());
                if steps.is_none() {
                    println!("Usage: diffuse_charge n [spread s/d] [decay d] [max m]");
                    continue;
                }
                let mut params = engine::diffuse::DiffuseParams::default();
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "spread" => {
                            if args.len() >= 2 {
                                let frac = args[1];
                                if let Some((num, den)) = frac.split_once('/') {
                                    if let (Ok(n), Ok(d)) = (num.parse::<u32>(), den.parse::<u32>())
                                    {
                                        params.spread_numerator = n.min(d);
                                        params.spread_denominator = d.max(1);
                                    }
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "decay" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.decay_per_step = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "max" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.max_value = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let steps = clamp_steps(steps.unwrap());
                sim.step_charge_diffuse_n(steps, &params);
                println!(
                    "Charge diffusion: n={} spread={}/{} decay={} max={}",
                    steps,
                    params.spread_numerator,
                    params.spread_denominator,
                    params.decay_per_step,
                    params.max_value
                );
            }
            "step_reaction" => {
                let steps = parts.next().and_then(|s| s.parse::<u32>().ok());
                if steps.is_none() {
                    println!(
                        "Usage: step_reaction n [heat_consume h] [charge_consume c] [heat_yield yh] [charge_yield yc] [min_heat mh] [min_charge mc] [max_heat H] [max_charge C]"
                    );
                    continue;
                }
                let mut params = engine::reaction::ReactionParams::default();
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "heat_consume" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.heat_consume = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "charge_consume" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.charge_consume = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "heat_yield" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.heat_yield = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "charge_yield" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.charge_yield = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "min_heat" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.min_heat = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "min_charge" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.min_charge = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "max_heat" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.max_heat = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "max_charge" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.max_charge = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let steps = clamp_steps(steps.unwrap());
                sim.step_reaction_n(steps, &params);
                println!(
                    "Reaction: steps={} params=({},{},{},{},{},{},{},{})",
                    steps,
                    params.heat_consume,
                    params.charge_consume,
                    params.heat_yield,
                    params.charge_yield,
                    params.min_heat,
                    params.min_charge,
                    params.max_heat,
                    params.max_charge
                );
            }
            "step_interact" => {
                let steps = parts.next().and_then(|s| s.parse::<u32>().ok());
                let mut params = engine::physics_interact::InteractionParams::default();
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "heat_affects_charge" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.heat_affects_charge = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "charge_affects_heat" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.charge_affects_heat = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "max_delta" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.max_delta = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "clamp" => {
                            if args.len() >= 2 {
                                if let Ok(v) = args[1].parse::<u32>() {
                                    params.clamp = v;
                                }
                                args.drain(0..2);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let steps = clamp_steps(steps.unwrap_or(1));
                sim.step_interact_n(steps, &params);
                println!(
                    "Interact OK: steps={} params=({},{},{},{})",
                    steps,
                    params.heat_affects_charge,
                    params.charge_affects_heat,
                    params.max_delta,
                    params.clamp
                );
            }
            "run_scenario" => {
                let name = parts.next();
                let mut steps = None;
                let mut tick_stride = Some(1u32);
                let mut region_args: Option<Vec<&str>> = None;
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "steps" => {
                            if args.len() >= 2 {
                                steps = args[1].parse::<u32>().ok();
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "tick_stride" => {
                            if args.len() >= 2 {
                                tick_stride = args[1].parse::<u32>().ok();
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "region" => {
                            if args.len() >= 5 {
                                region_args = Some(args[1..5].to_vec());
                                args.drain(0..5);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let kind = match name.map(|s| s.to_ascii_lowercase()) {
                    Some(k) if k == "heat_pulse_1d" => {
                        Some(engine::physics_scenarios::ScenarioKind::HeatPulse1D)
                    }
                    Some(k) if k == "heat_box" => {
                        Some(engine::physics_scenarios::ScenarioKind::HeatBox)
                    }
                    Some(k) if k == "charge_wave" => {
                        Some(engine::physics_scenarios::ScenarioKind::ChargeWave)
                    }
                    Some(k) if k == "interact_osc" => {
                        Some(engine::physics_scenarios::ScenarioKind::InteractOscillator)
                    }
                    _ => None,
                };
                if kind.is_none() {
                    println!(
                        "Usage: run_scenario <heat_pulse_1d|heat_box|charge_wave|interact_osc> [steps n] [tick_stride k] [region x0 y0 w h]"
                    );
                    continue;
                }
                let scenario = engine::physics_scenarios::PhysicsScenario {
                    kind: kind.unwrap(),
                    steps: steps.unwrap_or(10).min(1000),
                    tick_stride: tick_stride.unwrap_or(1),
                    region: parse_region(region_args.as_deref().unwrap_or(&[])),
                    heat_params: None,
                    charge_params: None,
                    diffuse_params: None,
                    reaction_params: None,
                    interact_params: None,
                    coupling_params: None,
                };
                println!("Scenario {:?}", name.unwrap_or("unknown"));
                let summary = sim.run_scenario(&scenario);
                println!(
                    "steps={} max_heat={} max_charge={} avg_heat={:.2} avg_charge={:.2}",
                    summary.steps,
                    summary.max_heat,
                    summary.max_charge,
                    summary.final_avg_heat,
                    summary.final_avg_charge
                );
            }
            "run_experiment" => {
                let name = parts.next();
                let param = parts.next();
                let start = parts.next().and_then(|s| s.parse::<u32>().ok());
                let end = parts.next().and_then(|s| s.parse::<u32>().ok());
                let nsteps = parts.next().and_then(|s| s.parse::<u32>().ok());
                let mut tick_stride = Some(1u32);
                let mut _region_args: Vec<&str> = Vec::new();
                let mut args: Vec<&str> = parts.collect();
                while let Some(flag) = args.first().cloned() {
                    match flag {
                        "tick_stride" => {
                            if args.len() >= 2 {
                                tick_stride = args[1].parse::<u32>().ok();
                                args.drain(0..2);
                                continue;
                            }
                        }
                        "region" => {
                            if args.len() >= 5 {
                                _region_args = args[1..5].to_vec();
                                args.drain(0..5);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    args.drain(0..1);
                }
                let sk = match name.map(|s| s.to_ascii_lowercase()) {
                    Some(k) if k == "heat_pulse_1d" => {
                        Some(engine::physics_scenarios::ScenarioKind::HeatPulse1D)
                    }
                    Some(k) if k == "heat_box" => {
                        Some(engine::physics_scenarios::ScenarioKind::HeatBox)
                    }
                    Some(k) if k == "charge_wave" => {
                        Some(engine::physics_scenarios::ScenarioKind::ChargeWave)
                    }
                    Some(k) if k == "interact_osc" => {
                        Some(engine::physics_scenarios::ScenarioKind::InteractOscillator)
                    }
                    _ => None,
                };
                let ep = match param.map(|s| s.to_ascii_lowercase()) {
                    Some(p) if p == "max_heat" => {
                        Some(engine::physics_experiments::ExperimentParam::MaxHeat)
                    }
                    Some(p) if p == "max_charge" => {
                        Some(engine::physics_experiments::ExperimentParam::MaxCharge)
                    }
                    Some(p) if p == "spread" => {
                        Some(engine::physics_experiments::ExperimentParam::SpreadNumerator)
                    }
                    Some(p) if p == "decay" => {
                        Some(engine::physics_experiments::ExperimentParam::DecayPerStep)
                    }
                    Some(p) if p == "coupling" => {
                        Some(engine::physics_experiments::ExperimentParam::CouplingStrength)
                    }
                    Some(p) if p == "interaction" => {
                        Some(engine::physics_experiments::ExperimentParam::InteractionStrength)
                    }
                    _ => None,
                };
                let (start, end, nsteps) = match (start, end, nsteps) {
                    (Some(a), Some(b), Some(n)) => (a, b, n),
                    _ => {
                        println!(
                            "Usage: run_experiment NAME PARAM start end steps [tick_stride k] [region x0 y0 w h]"
                        );
                        continue;
                    }
                };
                if sk.is_none() || ep.is_none() {
                    println!(
                        "Usage: run_experiment NAME PARAM start end steps [tick_stride k] [region x0 y0 w h]"
                    );
                    continue;
                }
                let spec = engine::physics_experiments::ExperimentKind::Sweep1D {
                    scenario: sk.unwrap(),
                    param: ep.unwrap(),
                    range: engine::physics_experiments::Range1D {
                        start,
                        end,
                        steps: nsteps.max(1),
                    },
                    tick_stride: tick_stride.unwrap_or(1),
                };
                println!("Experiment");
                let res = sim.run_experiment(&spec);
                for run in &res.runs {
                    println!(
                        "value={} max_heat={} max_charge={} avg_heat={:.2} avg_charge={:.2}",
                        run.param_value,
                        run.summary.max_heat,
                        run.summary.max_charge,
                        run.summary.final_avg_heat,
                        run.summary.final_avg_charge
                    );
                }
                println!(
                    "summary: max_heat_overall={} max_charge_overall={} avg_final_heat={:.2} avg_final_charge={:.2}",
                    res.max_heat_overall,
                    res.max_charge_overall,
                    res.avg_final_heat,
                    res.avg_final_charge
                );
            }
            "run_script" => {
                let name = parts.next();
                let preset = name.and_then(|n| {
                    engine::scripts::preset_scripts()
                        .into_iter()
                        .find(|p| p.name.eq_ignore_ascii_case(&n))
                });
                if let Some(preset) = preset {
                    println!("Script: {}", preset.name);
                    match sim.run_script(&preset.script) {
                        Ok(res) => {
                            for snap in res.snapshots {
                                println!(
                                    "snapshot label={} region=({},{},{},{})",
                                    snap.label,
                                    snap.region.x0,
                                    snap.region.y0,
                                    snap.region.w,
                                    snap.region.h
                                );
                                for row in snap.data {
                                    for val in row {
                                        print!("{:016x} ", val);
                                    }
                                    println!();
                                }
                            }
                            for meas in res.measurements {
                                println!(
                                    "measure label={} region=({},{},{},{}) max_heat={} avg_heat={:.2} max_charge={} avg_charge={:.2}",
                                    meas.label,
                                    meas.region.x0,
                                    meas.region.y0,
                                    meas.region.w,
                                    meas.region.h,
                                    meas.max_heat,
                                    meas.avg_heat,
                                    meas.max_charge,
                                    meas.avg_charge
                                );
                            }
                        }
                        Err(err) => {
                            println!("Script error: {:?}", err);
                        }
                    }
                } else {
                    let available: Vec<String> = engine::scripts::preset_scripts()
                        .into_iter()
                        .map(|p| p.name.to_string())
                        .collect();
                    println!("Unknown script. Available: {}", available.join(", "));
                }
            }
            "lint" => {
                let mode = parts.next().map(|s| s.to_ascii_lowercase());
                if let Some(extra) = parts.next() {
                    let _ = extra;
                    println!("Usage: lint [relaxed|strict]");
                    continue;
                }
                match mode.as_deref() {
                    Some("strict") => {
                        let res = sim.lint_strict();
                        print_lint_result(&res);
                    }
                    Some("relaxed") | None => {
                        let res = sim.lint_default();
                        print_lint_result(&res);
                    }
                    _ => {
                        println!("Usage: lint [relaxed|strict]");
                    }
                }
            }
            "reset" => {
                sim = Simulation::new();
                tick_counter = 0;
                println!("Simulation reset.");
            }
            "verbose" => match (parts.next(), parts.next()) {
                (Some(onoff), None) => match onoff.to_ascii_lowercase().as_str() {
                    "on" => {
                        cfg.verbose = true;
                        println!("Verbose mode: ON");
                    }
                    "off" => {
                        cfg.verbose = false;
                        println!("Verbose mode: OFF");
                    }
                    _ => println!("Usage: verbose on|off"),
                },
                _ => println!("Usage: verbose on|off"),
            },
            "delta_headers" => match (parts.next(), parts.next()) {
                (Some(onoff), None) => match onoff.to_ascii_lowercase().as_str() {
                    "on" => {
                        cfg.delta_headers = true;
                        println!("Delta headers: ON");
                    }
                    "off" => {
                        cfg.delta_headers = false;
                        println!("Delta headers: OFF");
                    }
                    _ => println!("Usage: delta_headers on|off"),
                },
                _ => println!("Usage: delta_headers on|off"),
            },
            _ => println!("Unrecognized command: {}", line),
        }
    }
}
