use std::env;
use std::error::Error;
use std::fs::{create_dir_all, write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use engine::tile_cpu::{
    V2BenchmarkOutcome, benchmark_cases, run_v2_benchmark_case, validate_benchmark_outcome,
    validate_benchmark_performance,
};

fn format_hash(v: u64) -> String {
    format!("0x{v:016X}")
}

fn format_rate(v: f64) -> String {
    format!("{v:.2}")
}

fn format_elapsed_ms(outcome: &V2BenchmarkOutcome) -> String {
    format!("{:.3}", outcome.elapsed.as_secs_f64() * 1000.0)
}

fn machine_metadata() -> String {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!(
        "- unix_time: {unix}\n- os: {}\n- arch: {}\n- cpu_threads: {cpus}\n- profile: release=false\n",
        env::consts::OS,
        env::consts::ARCH
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let print_goldens = env::args().any(|arg| arg == "--print-goldens");
    let out_dir = Path::new("User notes/SPRINTS/SPRINT 140.0");
    create_dir_all(out_dir)?;

    let mut md = String::new();
    md.push_str("# Sprint 140 Benchmark Report\n\n");
    md.push_str("## Machine Metadata\n");
    md.push_str(&machine_metadata());
    md.push('\n');
    md.push_str("## Run Config\n");
    md.push_str("- grid: 128x128x4\n");
    md.push_str("- trace modes: off + on\n");
    md.push_str("- instruction width: 32-bit\n");
    md.push_str("- register model: 16 regs (hybrid active bank)\n");
    md.push_str("- ram model: 64 cells (hybrid bank)\n\n");
    md.push_str("## Results\n\n");
    md.push_str("| case | trace | cycles | retired | ipc | final_hash | trace_entries | trace_hash | elapsed_ms | deltas/s | evals/s | switched/s | f_switch | f_mixed | x_mixed | rom_ovr | ram_hi_rd | z_shift_carry |\n");
    md.push_str("|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");

    let mut lines = Vec::new();
    let mut computed_goldens = Vec::new();

    for case in benchmark_cases() {
        let no_trace = run_v2_benchmark_case(case, false)?;
        let with_trace = run_v2_benchmark_case(case, true)?;

        if no_trace.final_state_hash != with_trace.final_state_hash {
            return Err(format!(
                "final hash mismatch between trace modes for '{}': {} vs {}",
                case.name,
                format_hash(no_trace.final_state_hash),
                format_hash(with_trace.final_state_hash)
            )
            .into());
        }

        if !print_goldens {
            validate_benchmark_outcome(&no_trace)?;
            validate_benchmark_outcome(&with_trace)?;
            validate_benchmark_performance(&no_trace)?;
            validate_benchmark_performance(&with_trace)?;
        }

        computed_goldens.push((case.name, no_trace.final_state_hash));
        lines.push(format!(
            "{} no_trace hash={} cycles={} retired={} ipc={:.4}",
            case.name,
            format_hash(no_trace.final_state_hash),
            no_trace.metrics.cycles,
            no_trace.metrics.instructions_executed,
            no_trace.metrics.ipc
        ));

        for outcome in [&no_trace, &with_trace] {
            let trace_label = if outcome.trace_enabled { "on" } else { "off" };
            let trace_hash = outcome
                .trace_hash
                .map(format_hash)
                .unwrap_or_else(|| "-".to_string());
            md.push_str(&format!(
                "| {} | {} | {} | {} | {:.4} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                outcome.case_name,
                trace_label,
                outcome.metrics.cycles,
                outcome.metrics.instructions_executed,
                outcome.metrics.ipc,
                format_hash(outcome.final_state_hash),
                outcome.trace_entries,
                trace_hash,
                format_elapsed_ms(outcome),
                format_rate(outcome.deltas_per_sec()),
                format_rate(outcome.evals_per_sec()),
                format_rate(outcome.switched_per_sec()),
                outcome.hybrid.stage_f_bank_switches,
                outcome.hybrid.stage_f_mixed_dual_capture,
                outcome.hybrid.stage_x_mixed_software,
                outcome.hybrid.ram_high_bank_read_swaps,
                outcome.hybrid.rom_upper_bank_group_select,
            ));
        }
    }

    let summary_path = out_dir.join("Sprint 140 Benchmark Report.md");
    write(&summary_path, md)?;

    let raw_path = out_dir.join("Sprint 140 Benchmark Raw.txt");
    write(&raw_path, lines.join("\n"))?;

    println!("wrote report: {}", summary_path.display());
    println!("wrote raw: {}", raw_path.display());
    if print_goldens {
        println!("\ncomputed goldens:");
        for (name, hash) in computed_goldens {
            println!("(\"{name}\", {}),", format_hash(hash));
        }
        println!("\n--print-goldens mode skips golden validation.");
    } else {
        println!("golden validation passed for all benchmark cases.");
    }
    Ok(())
}
