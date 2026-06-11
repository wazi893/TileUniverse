use std::env;
use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use engine::simulation::Simulation;
use engine::tile_cpu::{
    V2_TRACE_VIZ_SUMMARY_FILE, V2Builder, V2TraceVizDeltaKind, V2TraceVizStepSummary,
    V2VizLayerLayout, V2VizRenderParams, V2VizSession, assemble_v2, capture_v2_trace_viz_timeline,
    export_v2_trace_viz_bundle, load_v2_trace_viz_bundle_summaries, v2_viz_find_next_by_asm,
    v2_viz_find_next_by_delta, v2_viz_find_next_by_pc, v2_viz_find_prev_by_delta,
    write_v2_trace_viz_summary_tsv,
};

#[derive(Debug, Clone)]
struct CliArgs {
    step: Option<usize>,
    out_dir: PathBuf,
    bundle_out: Option<PathBuf>,
    bundle_in: Option<PathBuf>,
    summary_out: Option<PathBuf>,
    interactive: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            step: None,
            out_dir: PathBuf::from("target/test_output/v2_trace_viz"),
            bundle_out: None,
            bundle_in: None,
            summary_out: None,
            interactive: false,
        }
    }
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs::default();
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--step" => {
                args.step = iter.next().and_then(|v| v.parse::<usize>().ok());
            }
            "--out" => {
                if let Some(v) = iter.next() {
                    args.out_dir = PathBuf::from(v);
                }
            }
            "--bundle-out" => {
                if let Some(v) = iter.next() {
                    args.bundle_out = Some(PathBuf::from(v));
                }
            }
            "--bundle-in" => {
                if let Some(v) = iter.next() {
                    args.bundle_in = Some(PathBuf::from(v));
                }
            }
            "--summary-out" => {
                if let Some(v) = iter.next() {
                    args.summary_out = Some(PathBuf::from(v));
                }
            }
            "--interactive" => {
                args.interactive = true;
            }
            _ => {}
        }
    }
    args
}

fn delta_flags(summary: &V2TraceVizStepSummary) -> String {
    let mut flags = Vec::new();
    if summary.reg_deltas > 0 {
        flags.push("regs");
    }
    if summary.ram_deltas > 0 {
        flags.push("ram");
    }
    if summary.mmio_reads > 0 || summary.mmio_writes > 0 {
        flags.push("mmio");
    }
    if flags.is_empty() {
        "none".to_string()
    } else {
        flags.join(",")
    }
}

fn print_selected(summary: &V2TraceVizStepSummary, bundle_dir: &Path, total_steps: usize) {
    println!("trace_viz_artifacts={}", bundle_dir.display());
    println!("retired_steps={total_steps}");
    println!(
        "selected_step={} cycle={} pc={} asm=\"{}\"",
        summary.retired, summary.cycle, summary.pc, summary.asm
    );
    println!(
        "selected_frame_checksum=0x{:08X} active_tiles={} flow_edges={}",
        summary.frame_checksum, summary.active_tiles, summary.flow_edges
    );
    println!(
        "selected_deltas regs={} ram={} mmio_r={} mmio_w={} flags={}",
        summary.reg_deltas,
        summary.ram_deltas,
        summary.mmio_reads,
        summary.mmio_writes,
        delta_flags(summary)
    );
    println!(
        "selected_frame_file={}",
        bundle_dir.join(&summary.frame_file).display()
    );
}

fn print_help() {
    println!("commands:");
    println!("  n | p                   next / previous");
    println!("  g <idx> | <idx>         jump to step index");
    println!("  pc <value>              jump to next step with PC (decimal or 0xNN)");
    println!("  asm <substring>         jump to next step containing mnemonic text");
    println!("  next <regs|ram|mmio>    jump to next state-delta match");
    println!("  prev <regs|ram|mmio>    jump to previous state-delta match");
    println!("  ctx                     show current step context");
    println!("  h | help                show this help");
    println!("  q                       quit");
}

fn parse_pc(raw: &str) -> Option<u32> {
    let trimmed = raw.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u32::from_str_radix(hex, 16).ok()
    } else {
        trimmed.parse::<u32>().ok()
    }
}

fn parse_delta_kind(raw: &str) -> Option<V2TraceVizDeltaKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "regs" | "reg" => Some(V2TraceVizDeltaKind::Regs),
        "ram" => Some(V2TraceVizDeltaKind::Ram),
        "mmio" => Some(V2TraceVizDeltaKind::Mmio),
        _ => None,
    }
}

fn interactive_view(
    summaries: &[V2TraceVizStepSummary],
    bundle_dir: &Path,
    start_idx: usize,
) -> Result<(), Box<dyn Error>> {
    if summaries.is_empty() {
        return Err("timeline has no retired steps".into());
    }

    let mut idx = start_idx.min(summaries.len().saturating_sub(1));
    print_help();
    print_selected(&summaries[idx], bundle_dir, summaries.len());

    let stdin = io::stdin();
    loop {
        print!("viz> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let cmd = line.trim();
        if cmd.is_empty() {
            continue;
        }
        if cmd == "q" {
            break;
        }
        if cmd == "h" || cmd == "help" {
            print_help();
            continue;
        }
        if cmd == "ctx" {
            print_selected(&summaries[idx], bundle_dir, summaries.len());
            continue;
        }
        if cmd == "n" {
            idx = (idx + 1).min(summaries.len().saturating_sub(1));
            print_selected(&summaries[idx], bundle_dir, summaries.len());
            continue;
        }
        if cmd == "p" {
            idx = idx.saturating_sub(1);
            print_selected(&summaries[idx], bundle_dir, summaries.len());
            continue;
        }

        if let Some(rest) = cmd.strip_prefix("pc ") {
            let Some(pc) = parse_pc(rest) else {
                println!("invalid PC value: {rest}");
                continue;
            };
            if let Some(next_idx) = v2_viz_find_next_by_pc(summaries, idx, pc) {
                idx = next_idx;
                print_selected(&summaries[idx], bundle_dir, summaries.len());
            } else {
                println!("no later step found with pc={pc}");
            }
            continue;
        }

        if let Some(rest) = cmd.strip_prefix("asm ") {
            if let Some(next_idx) = v2_viz_find_next_by_asm(summaries, idx, rest) {
                idx = next_idx;
                print_selected(&summaries[idx], bundle_dir, summaries.len());
            } else {
                println!("no later step contains asm substring '{rest}'");
            }
            continue;
        }

        if let Some(rest) = cmd.strip_prefix("next ") {
            let Some(kind) = parse_delta_kind(rest) else {
                println!("unknown delta kind: {rest}");
                continue;
            };
            if let Some(next_idx) = v2_viz_find_next_by_delta(summaries, idx, kind) {
                idx = next_idx;
                print_selected(&summaries[idx], bundle_dir, summaries.len());
            } else {
                println!("no later step found for delta kind '{rest}'");
            }
            continue;
        }

        if let Some(rest) = cmd.strip_prefix("prev ") {
            let Some(kind) = parse_delta_kind(rest) else {
                println!("unknown delta kind: {rest}");
                continue;
            };
            if let Some(next_idx) = v2_viz_find_prev_by_delta(summaries, idx, kind) {
                idx = next_idx;
                print_selected(&summaries[idx], bundle_dir, summaries.len());
            } else {
                println!("no previous step found for delta kind '{rest}'");
            }
            continue;
        }

        let target = if let Some(rest) = cmd.strip_prefix("g ") {
            rest.trim().parse::<usize>().ok()
        } else {
            cmd.parse::<usize>().ok()
        };

        if let Some(target_idx) = target {
            idx = target_idx.min(summaries.len().saturating_sub(1));
            print_selected(&summaries[idx], bundle_dir, summaries.len());
        } else {
            println!("unrecognized command: {cmd}");
        }
    }

    Ok(())
}

fn build_demo_timeline(
    max_cycles: u64,
) -> Result<
    (
        engine::tile_cpu::V2TraceVizTimeline,
        Vec<V2TraceVizStepSummary>,
    ),
    Box<dyn Error>,
> {
    let source = r#"
        LDI R8, 0
        LDI R9, 3
    loop:
        ADD R8, R9
        DEC R9
        JNZ loop
        STB [40], R8
        LDB R10, [40]
        HALT
    "#;
    let program = assemble_v2(source)?;

    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(64)
        .with_ram_size(64)
        .build(&mut sim);

    let params = V2VizRenderParams {
        scale: 1,
        layout: V2VizLayerLayout::Grid2x2,
        highlight_active: true,
        show_flow: true,
        max_flow_edges: 2048,
    };

    let mut viz = V2VizSession::new(&sim);
    let timeline = capture_v2_trace_viz_timeline(&cpu, &mut sim, &mut viz, &params, max_cycles);
    if timeline.is_empty() {
        return Err("timeline is empty".into());
    }
    if timeline.trace.len() != timeline.frames.len() {
        return Err(format!(
            "trace/frame mismatch: trace={} frames={}",
            timeline.trace.len(),
            timeline.frames.len()
        )
        .into());
    }

    let summaries = timeline.step_summaries();
    if summaries.len() != timeline.len() {
        return Err("timeline summary length mismatch".into());
    }

    Ok((timeline, summaries))
}

fn write_summary_artifact(
    summaries: &[V2TraceVizStepSummary],
    summary_out: &Path,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = summary_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_v2_trace_viz_summary_tsv(summaries, summary_out)?;
    println!("summary_tsv={}", summary_out.display());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args();

    if let Some(bundle_in_dir) = &args.bundle_in {
        let summaries = load_v2_trace_viz_bundle_summaries(bundle_in_dir)?;
        if summaries.is_empty() {
            return Err(format!("no trace-viz steps in {}", bundle_in_dir.display()).into());
        }
        let summary_out = args
            .summary_out
            .clone()
            .unwrap_or_else(|| bundle_in_dir.join(V2_TRACE_VIZ_SUMMARY_FILE));
        write_summary_artifact(&summaries, &summary_out)?;

        let selected = args
            .step
            .unwrap_or_else(|| summaries.len().saturating_sub(1))
            .min(summaries.len().saturating_sub(1));
        if args.interactive {
            interactive_view(&summaries, bundle_in_dir, selected)?;
        } else {
            print_selected(&summaries[selected], bundle_in_dir, summaries.len());
        }
        return Ok(());
    }

    let (timeline, summaries) = build_demo_timeline(256)?;
    let bundle_out = args
        .bundle_out
        .clone()
        .unwrap_or_else(|| args.out_dir.clone());
    let exported = export_v2_trace_viz_bundle(&timeline, &bundle_out)?;
    if exported != summaries {
        return Err("exported summaries mismatch live summaries".into());
    }

    let summary_out = args
        .summary_out
        .clone()
        .unwrap_or_else(|| bundle_out.join(V2_TRACE_VIZ_SUMMARY_FILE));
    write_summary_artifact(&summaries, &summary_out)?;

    let selected = args
        .step
        .unwrap_or_else(|| summaries.len().saturating_sub(1))
        .min(summaries.len().saturating_sub(1));

    if args.interactive {
        interactive_view(&summaries, &bundle_out, selected)?;
    } else {
        print_selected(&summaries[selected], &bundle_out, summaries.len());
    }

    Ok(())
}
