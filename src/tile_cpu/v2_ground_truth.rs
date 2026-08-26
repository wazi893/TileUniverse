//! V2 ground-truth trace export: one-call JSONL artifact for downstream
//! interpretability tooling.
//!
//! [`V2TraceEntry::to_json`](crate::tile_cpu::V2TraceEntry::to_json) and
//! [`V2TraceLog::to_jsonl`](crate::tile_cpu::V2TraceLog::to_jsonl) produce
//! machine-readable step records, but nothing wires them to a runnable program
//! or a file. This module closes that gap: run a program on a freshly built V2
//! CPU and emit a single JSONL document — a header record (schema version,
//! program words, initial register file, final cycle/retired counts) followed
//! by one record per retired instruction, in cycle order.
//!
//! Step records carry register/RAM *deltas* (the values written that step);
//! folding them over the header's `initial_regs` yields the absolute register
//! file at every step. MMIO reads/writes per step are the OUT channel.

use std::error::Error;
use std::fs::write;
use std::path::Path;

use crate::simulation::Simulation;
use crate::tile_cpu::{V2Builder, V2TraceLog};

pub const V2_GROUND_TRUTH_SCHEMA_VERSION: u32 = 1;

/// Run `program_words` to completion on a freshly built V2 CPU and return the
/// ground-truth JSONL document (header line + one line per retired
/// instruction). Deterministic: identical program + `max_cycles` yields a
/// byte-identical document.
///
/// Errors if the program does not halt within `max_cycles`.
pub fn capture_v2_ground_truth_jsonl(
    program_words: &[u32],
    max_cycles: u64,
) -> Result<String, Box<dyn Error>> {
    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(program_words)
        .with_rom_size(64)
        .with_ram_size(64)
        .build(&mut sim);

    let initial_regs: Vec<u64> = (0..16).map(|r| cpu.read_reg(&sim, r)).collect();

    let mut trace = V2TraceLog::new();
    for _ in 0..max_cycles {
        if cpu.is_halted() {
            break;
        }
        let _ = cpu.step_with_trace(&mut sim, &mut trace);
    }
    if !cpu.is_halted() {
        return Err(format!("program did not halt within {max_cycles} cycles").into());
    }

    let words = program_words
        .iter()
        .map(|w| w.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let regs = initial_regs
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let header = format!(
        "{{\"type\":\"program\",\"schema_version\":{},\"program_words\":[{}],\"initial_regs\":[{}],\"cycle_count\":{},\"retired_count\":{}}}",
        V2_GROUND_TRUTH_SCHEMA_VERSION,
        words,
        regs,
        cpu.read_cycle_count(),
        cpu.read_retired_count(),
    );

    let steps = trace.to_jsonl();
    if steps.is_empty() {
        Ok(header)
    } else {
        Ok(format!("{header}\n{steps}"))
    }
}

/// Capture and write the ground-truth JSONL document to `path`.
pub fn write_v2_ground_truth_jsonl(
    program_words: &[u32],
    max_cycles: u64,
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let doc = capture_v2_ground_truth_jsonl(program_words, max_cycles)?;
    write(path, doc)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::tile_cpu::V2FastCpu;
    use crate::tile_cpu::v2_components::{v2_dec, v2_halt, v2_inc, v2_jnz, v2_ldi};

    fn unique_temp_file(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "engine_v2_ground_truth_{label}_{}_{}.jsonl",
            std::process::id(),
            nanos
        ));
        p
    }

    #[test]
    fn test_v2_ground_truth_header_and_step_records() {
        let program = [v2_ldi(0, 41), v2_inc(0), v2_halt()];
        let doc = capture_v2_ground_truth_jsonl(&program, 32).expect("capture should succeed");
        let lines: Vec<&str> = doc.lines().collect();
        assert!(
            lines.len() >= 3,
            "expected header + >=2 steps, got {lines:?}"
        );

        let header = lines[0];
        assert!(header.starts_with("{\"type\":\"program\""));
        assert!(header.contains(&format!(
            "\"schema_version\":{V2_GROUND_TRUTH_SCHEMA_VERSION}"
        )));
        assert!(header.contains(&format!(
            "\"program_words\":[{},{},{}]",
            program[0], program[1], program[2]
        )));
        let regs_field = header
            .split("\"initial_regs\":[")
            .nth(1)
            .and_then(|rest| rest.split(']').next())
            .expect("header must carry initial_regs");
        assert_eq!(regs_field.split(',').count(), 16);

        // Delta semantics: LDI writes R0=41, INC folds it to 42.
        assert!(doc.contains("\"reg_writes\":[{\"reg\":0,\"value\":41}]"));
        assert!(doc.contains("\"reg_writes\":[{\"reg\":0,\"value\":42}]"));
        for step in &lines[1..] {
            assert!(step.contains("\"asm\":"), "step missing asm: {step}");
        }
    }

    #[test]
    fn test_v2_ground_truth_write_deterministic_roundtrip() {
        let program = [v2_ldi(0, 7), v2_inc(0), v2_halt()];
        let path = unique_temp_file("io");
        write_v2_ground_truth_jsonl(&program, 32, &path).expect("write should succeed");
        let on_disk = std::fs::read_to_string(&path).expect("read back should succeed");
        let recaptured =
            capture_v2_ground_truth_jsonl(&program, 32).expect("recapture should succeed");
        assert_eq!(on_disk, recaptured);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_v2_ground_truth_non_halting_program_errors() {
        let program = [v2_ldi(0, 1), v2_inc(0)];
        assert!(capture_v2_ground_truth_jsonl(&program, 16).is_err());
    }

    fn parse_initial_regs(header: &str) -> [u64; 16] {
        let body = header
            .split("\"initial_regs\":[")
            .nth(1)
            .and_then(|rest| rest.split(']').next())
            .expect("header must carry initial_regs");
        let vals: Vec<u64> = body
            .split(',')
            .map(|s| s.parse().expect("initial reg value"))
            .collect();
        assert_eq!(vals.len(), 16);
        std::array::from_fn(|i| vals[i])
    }

    fn fold_step_reg_writes(step: &str, regs: &mut [u64; 16]) {
        let Some(rest) = step.split("\"reg_writes\":[").nth(1) else {
            return;
        };
        let body = rest.split(']').next().unwrap_or("");
        for obj in body.split("},").filter(|s| !s.trim().is_empty()) {
            let reg: usize = obj
                .split("\"reg\":")
                .nth(1)
                .expect("reg field")
                .split(',')
                .next()
                .unwrap()
                .parse()
                .expect("reg index");
            let value: u64 = obj
                .split("\"value\":")
                .nth(1)
                .expect("value field")
                .trim_end_matches('}')
                .parse()
                .expect("reg write value");
            regs[reg] = value;
        }
    }

    /// The exported deltas are only trustworthy ground truth if folding them
    /// reproduces the state an independent implementation computes: fold the
    /// physical capture's reg_writes over initial_regs and require exact
    /// agreement with the ISS final register file and retired count.
    #[test]
    fn test_v2_ground_truth_reg_fold_matches_iss_differential() {
        let straight: &[u32] = &[v2_ldi(0, 41), v2_inc(0), v2_halt()];
        let countdown: &[u32] = &[
            v2_ldi(0, 3),
            v2_ldi(1, 0),
            v2_inc(1),
            v2_dec(0),
            v2_jnz(2),
            v2_halt(),
        ];
        for (label, program) in [("straight", straight), ("countdown", countdown)] {
            let doc = capture_v2_ground_truth_jsonl(program, 128).expect("capture should succeed");
            let mut lines = doc.lines();
            let header = lines.next().expect("header line");
            let mut folded = parse_initial_regs(header);
            for step in lines {
                fold_step_reg_writes(step, &mut folded);
            }

            let mut iss = V2FastCpu::from_program(program);
            iss.run_batch(256);
            assert!(iss.is_halted(), "{label}: ISS must halt");
            for (r, &folded_val) in folded.iter().enumerate() {
                assert_eq!(
                    folded_val,
                    iss.read_reg(r),
                    "{label}: R{r} diverges (physical fold vs ISS)"
                );
            }

            let retired: u64 = header
                .split("\"retired_count\":")
                .nth(1)
                .expect("header must carry retired_count")
                .trim_end_matches('}')
                .parse()
                .expect("retired count");
            assert_eq!(
                retired,
                iss.retired_count(),
                "{label}: retired count diverges"
            );
        }
    }
}
