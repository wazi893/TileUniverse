//! Deterministic V2 replay bundle capture, IO, and verification.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{create_dir_all, read_to_string, write};
use std::path::Path;

use crate::simulation::Simulation;
use crate::tile_cpu::v2_mmio::V2MmioHandle;
use crate::tile_cpu::{V2Builder, V2HybridAssistCounters, V2TraceLog, hash_v2_final_state};

pub const V2_REPLAY_SCHEMA_VERSION_V1: u32 = 1;
pub const V2_REPLAY_SCHEMA_VERSION_V2: u32 = 2;
pub const V2_REPLAY_SCHEMA_VERSION: u32 = V2_REPLAY_SCHEMA_VERSION_V2;

pub const V2_REPLAY_MANIFEST_FILE: &str = "manifest.kv";
pub const V2_REPLAY_PROGRAM_FILE: &str = "program.hex";
pub const V2_REPLAY_TRACE_FILE: &str = "trace.log";
pub const V2_REPLAY_MMIO_INITIAL_FILE: &str = "mmio_initial.hex";
pub const V2_REPLAY_MMIO_FINAL_FILE: &str = "mmio_final.hex";

pub const V2_REPLAY_MMIO_MODE_NONE: &str = "none";
pub const V2_REPLAY_MMIO_MODE_SNAPSHOT: &str = "snapshot";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2ReplayBundle {
    pub schema_version: u32,
    pub program_words: Vec<u32>,
    pub max_cycles: u64,
    pub cycle_count: u64,
    pub retired_count: u64,
    pub trace_entries: usize,
    pub trace_hash: u64,
    pub final_state_hash: u64,
    pub hybrid: V2HybridAssistCounters,
    pub mmio_mode: String,
    pub mmio_snapshot_kind: Option<String>,
    pub mmio_initial_state: Option<Vec<u8>>,
    pub mmio_final_state: Option<Vec<u8>>,
    pub trace_lines: String,
}

pub fn hash_trace_lines(trace_lines: &str) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &b in trace_lines.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01B3);
    }
    h
}

pub fn capture_v2_replay_bundle(
    program_words: &[u32],
    max_cycles: u64,
) -> Result<V2ReplayBundle, Box<dyn Error>> {
    capture_v2_replay_bundle_inner(program_words, max_cycles, None, V2_REPLAY_MMIO_MODE_NONE)
}

pub fn capture_v2_replay_bundle_with_mmio_snapshot(
    program_words: &[u32],
    max_cycles: u64,
    mmio: V2MmioHandle,
) -> Result<V2ReplayBundle, Box<dyn Error>> {
    capture_v2_replay_bundle_inner(
        program_words,
        max_cycles,
        Some(mmio),
        V2_REPLAY_MMIO_MODE_SNAPSHOT,
    )
}

pub fn verify_v2_replay_bundle(bundle: &V2ReplayBundle) -> Result<(), String> {
    verify_v2_replay_bundle_with_mmio(bundle, None)
}

pub fn verify_v2_replay_bundle_with_mmio(
    bundle: &V2ReplayBundle,
    mmio: Option<V2MmioHandle>,
) -> Result<(), String> {
    if !is_supported_schema(bundle.schema_version) {
        return Err(format!(
            "unsupported replay schema: got {}, supported [{}|{}]",
            bundle.schema_version, V2_REPLAY_SCHEMA_VERSION_V1, V2_REPLAY_SCHEMA_VERSION_V2
        ));
    }

    let rerun = match bundle.mmio_mode.as_str() {
        V2_REPLAY_MMIO_MODE_NONE => capture_v2_replay_bundle_inner(
            &bundle.program_words,
            bundle.max_cycles,
            None,
            V2_REPLAY_MMIO_MODE_NONE,
        )
        .map_err(|e| format!("failed to replay program: {e}"))?,
        V2_REPLAY_MMIO_MODE_SNAPSHOT => {
            if bundle.schema_version < V2_REPLAY_SCHEMA_VERSION_V2 {
                return Err("snapshot mode requires replay schema v2".to_string());
            }
            let Some(mmio) = mmio else {
                return Err("snapshot replay verification requires MMIO handle".to_string());
            };
            let Some(expected_initial) = bundle.mmio_initial_state.as_ref() else {
                return Err("bundle missing mmio_initial_state".to_string());
            };

            if let Some(expected_kind) = bundle.mmio_snapshot_kind.as_ref() {
                let got = mmio.snapshot_kind().ok_or_else(|| {
                    "MMIO handle does not expose snapshot_kind for snapshot replay".to_string()
                })?;
                if got != expected_kind {
                    return Err(format!(
                        "MMIO snapshot kind mismatch: expected '{expected_kind}', got '{got}'"
                    ));
                }
            }

            mmio.restore_state(expected_initial)
                .map_err(|e| format!("failed to restore MMIO initial snapshot: {e}"))?;
            capture_v2_replay_bundle_inner(
                &bundle.program_words,
                bundle.max_cycles,
                Some(mmio),
                V2_REPLAY_MMIO_MODE_SNAPSHOT,
            )
            .map_err(|e| format!("failed to replay program: {e}"))?
        }
        other => return Err(format!("unsupported mmio_mode '{other}'")),
    };

    compare_replay_bundle(bundle, &rerun)
}

pub fn write_v2_replay_bundle_dir(
    bundle: &V2ReplayBundle,
    dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    if !is_supported_schema(bundle.schema_version) {
        return Err(format!(
            "unsupported replay schema for write: {}",
            bundle.schema_version
        )
        .into());
    }

    let dir = dir.as_ref();
    create_dir_all(dir)?;

    let mut manifest = String::new();
    manifest.push_str(&format!("schema_version={}\n", bundle.schema_version));
    manifest.push_str(&format!("max_cycles={}\n", bundle.max_cycles));
    manifest.push_str(&format!("cycle_count={}\n", bundle.cycle_count));
    manifest.push_str(&format!("retired_count={}\n", bundle.retired_count));
    manifest.push_str(&format!("trace_entries={}\n", bundle.trace_entries));
    manifest.push_str(&format!("trace_hash=0x{:016X}\n", bundle.trace_hash));
    manifest.push_str(&format!(
        "final_state_hash=0x{:016X}\n",
        bundle.final_state_hash
    ));
    manifest.push_str(&format!(
        "hybrid_stage_f_bank_switches={}\n",
        bundle.hybrid.stage_f_bank_switches
    ));
    manifest.push_str(&format!(
        "hybrid_stage_f_mixed_dual_capture={}\n",
        bundle.hybrid.stage_f_mixed_dual_capture
    ));
    manifest.push_str(&format!(
        "hybrid_stage_x_mixed_software={}\n",
        bundle.hybrid.stage_x_mixed_software
    ));
    manifest.push_str(&format!(
        "hybrid_ram_high_bank_read_swaps={}\n",
        bundle.hybrid.ram_high_bank_read_swaps
    ));
    manifest.push_str(&format!(
        "hybrid_rom_upper_bank_group_select={}\n",
        bundle.hybrid.rom_upper_bank_group_select
    ));
    manifest.push_str(&format!("mmio_mode={}\n", bundle.mmio_mode));

    if bundle.schema_version >= V2_REPLAY_SCHEMA_VERSION_V2 {
        if let Some(kind) = bundle.mmio_snapshot_kind.as_ref() {
            manifest.push_str(&format!("mmio_snapshot_kind={kind}\n"));
        }
        if bundle.mmio_mode == V2_REPLAY_MMIO_MODE_SNAPSHOT {
            let Some(initial) = bundle.mmio_initial_state.as_ref() else {
                return Err("snapshot mode bundle missing mmio_initial_state".into());
            };
            let Some(final_state) = bundle.mmio_final_state.as_ref() else {
                return Err("snapshot mode bundle missing mmio_final_state".into());
            };
            manifest.push_str(&format!(
                "mmio_initial_file={V2_REPLAY_MMIO_INITIAL_FILE}\n"
            ));
            manifest.push_str(&format!("mmio_final_file={V2_REPLAY_MMIO_FINAL_FILE}\n"));
            write(
                dir.join(V2_REPLAY_MMIO_INITIAL_FILE),
                encode_hex(initial.as_slice()),
            )?;
            write(
                dir.join(V2_REPLAY_MMIO_FINAL_FILE),
                encode_hex(final_state.as_slice()),
            )?;
        }
    }

    write(dir.join(V2_REPLAY_MANIFEST_FILE), manifest)?;

    let mut program_hex = String::new();
    for &word in &bundle.program_words {
        program_hex.push_str(&format!("{word:08X}\n"));
    }
    write(dir.join(V2_REPLAY_PROGRAM_FILE), program_hex)?;
    write(dir.join(V2_REPLAY_TRACE_FILE), &bundle.trace_lines)?;
    Ok(())
}

pub fn read_v2_replay_bundle_dir(dir: impl AsRef<Path>) -> Result<V2ReplayBundle, Box<dyn Error>> {
    let dir = dir.as_ref();
    let manifest_text = read_to_string(dir.join(V2_REPLAY_MANIFEST_FILE))?;
    let manifest = parse_manifest(&manifest_text)?;

    let schema_version = parse_u32(
        manifest
            .get("schema_version")
            .ok_or("missing schema_version in manifest")?,
    )?;
    if !is_supported_schema(schema_version) {
        return Err(format!("unsupported replay schema: {schema_version}").into());
    }

    let max_cycles = parse_u64(
        manifest
            .get("max_cycles")
            .ok_or("missing max_cycles in manifest")?,
    )?;
    let cycle_count = parse_u64(
        manifest
            .get("cycle_count")
            .ok_or("missing cycle_count in manifest")?,
    )?;
    let retired_count = parse_u64(
        manifest
            .get("retired_count")
            .ok_or("missing retired_count in manifest")?,
    )?;
    let trace_entries = parse_u64(
        manifest
            .get("trace_entries")
            .ok_or("missing trace_entries in manifest")?,
    )? as usize;
    let trace_hash = parse_u64(
        manifest
            .get("trace_hash")
            .ok_or("missing trace_hash in manifest")?,
    )?;
    let final_state_hash = parse_u64(
        manifest
            .get("final_state_hash")
            .ok_or("missing final_state_hash in manifest")?,
    )?;
    let hybrid = V2HybridAssistCounters {
        stage_f_bank_switches: parse_optional_u64(
            manifest
                .get("hybrid_stage_f_bank_switches")
                .map(String::as_str),
        )?,
        stage_f_mixed_dual_capture: parse_optional_u64(
            manifest
                .get("hybrid_stage_f_mixed_dual_capture")
                .map(String::as_str),
        )?,
        stage_x_mixed_software: parse_optional_u64(
            manifest
                .get("hybrid_stage_x_mixed_software")
                .map(String::as_str),
        )?,
        // Sprint 153: rom_bank_overwrite_hits removed. Old bundles may contain the
        // key "hybrid_rom_bank_overwrite_hits" — it is silently ignored on parse.
        ram_high_bank_read_swaps: parse_optional_u64(
            manifest
                .get("hybrid_ram_high_bank_read_swaps")
                .map(String::as_str),
        )?,
        // Sprint 186: zero_shift_carry_overrides removed. Old bundles may still
        // contain the key — silently ignore it (same pattern as rom_bank_overwrite above).
        rom_upper_bank_group_select: parse_optional_u64(
            manifest
                .get("hybrid_rom_upper_bank_group_select")
                .map(String::as_str),
        )?,
    };
    let mmio_mode = manifest
        .get("mmio_mode")
        .cloned()
        .unwrap_or_else(|| V2_REPLAY_MMIO_MODE_NONE.to_string());

    let (mmio_snapshot_kind, mmio_initial_state, mmio_final_state) =
        if schema_version >= V2_REPLAY_SCHEMA_VERSION_V2 {
            let kind = manifest.get("mmio_snapshot_kind").and_then(|raw| {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            });

            if mmio_mode == V2_REPLAY_MMIO_MODE_SNAPSHOT {
                let initial_file = manifest
                    .get("mmio_initial_file")
                    .map(String::as_str)
                    .unwrap_or(V2_REPLAY_MMIO_INITIAL_FILE);
                let final_file = manifest
                    .get("mmio_final_file")
                    .map(String::as_str)
                    .unwrap_or(V2_REPLAY_MMIO_FINAL_FILE);
                let initial_text = read_to_string(dir.join(initial_file))?;
                let final_text = read_to_string(dir.join(final_file))?;
                (
                    kind,
                    Some(decode_hex(initial_text.as_str())?),
                    Some(decode_hex(final_text.as_str())?),
                )
            } else {
                (kind, None, None)
            }
        } else {
            (None, None, None)
        };

    let program_text = read_to_string(dir.join(V2_REPLAY_PROGRAM_FILE))?;
    let mut program_words = Vec::new();
    for (lineno, raw) in program_text.lines().enumerate() {
        let s = raw.trim();
        if s.is_empty() {
            continue;
        }
        let word = u32::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|e| {
            format!(
                "invalid program word at {}:{} '{}': {e}",
                V2_REPLAY_PROGRAM_FILE,
                lineno + 1,
                raw
            )
        })?;
        program_words.push(word);
    }
    let trace_lines = read_to_string(dir.join(V2_REPLAY_TRACE_FILE))?;

    Ok(V2ReplayBundle {
        schema_version,
        program_words,
        max_cycles,
        cycle_count,
        retired_count,
        trace_entries,
        trace_hash,
        final_state_hash,
        hybrid,
        mmio_mode,
        mmio_snapshot_kind,
        mmio_initial_state,
        mmio_final_state,
        trace_lines,
    })
}

fn capture_v2_replay_bundle_inner(
    program_words: &[u32],
    max_cycles: u64,
    mmio: Option<V2MmioHandle>,
    mmio_mode: &str,
) -> Result<V2ReplayBundle, Box<dyn Error>> {
    let (mmio_snapshot_kind, mmio_initial_state) = match mmio_mode {
        V2_REPLAY_MMIO_MODE_NONE => (None, None),
        V2_REPLAY_MMIO_MODE_SNAPSHOT => {
            let mmio = mmio
                .as_ref()
                .ok_or("snapshot mode requires MMIO handle for capture")?;
            let kind = mmio
                .snapshot_kind()
                .ok_or("MMIO device does not expose snapshot_kind")?;
            let snapshot = mmio
                .snapshot_state()
                .ok_or("MMIO device does not expose snapshot_state")?;
            (Some(kind.to_string()), Some(snapshot))
        }
        other => return Err(format!("unsupported mmio_mode '{other}'").into()),
    };

    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let mut builder = V2Builder::new()
        .with_origin(0, 0)
        .with_program(program_words)
        .with_rom_size(64)
        .with_ram_size(64);
    if let Some(mmio) = mmio.as_ref() {
        builder = builder.with_mmio(mmio.clone());
    }
    let cpu = builder.build(&mut sim);

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

    let mmio_final_state = match mmio_mode {
        V2_REPLAY_MMIO_MODE_NONE => None,
        V2_REPLAY_MMIO_MODE_SNAPSHOT => {
            let mmio = mmio
                .as_ref()
                .ok_or("snapshot mode requires MMIO handle for final snapshot")?;
            Some(
                mmio.snapshot_state()
                    .ok_or("MMIO device does not expose final snapshot_state")?,
            )
        }
        _ => unreachable!(),
    };

    let trace_lines = trace.to_lines();
    Ok(V2ReplayBundle {
        schema_version: V2_REPLAY_SCHEMA_VERSION,
        program_words: program_words.to_vec(),
        max_cycles,
        cycle_count: cpu.read_cycle_count(),
        retired_count: cpu.read_retired_count(),
        trace_entries: trace.len(),
        trace_hash: hash_trace_lines(trace_lines.as_str()),
        final_state_hash: hash_v2_final_state(&cpu, &sim),
        hybrid: cpu.read_hybrid_assist_counters(),
        mmio_mode: mmio_mode.to_string(),
        mmio_snapshot_kind,
        mmio_initial_state,
        mmio_final_state,
        trace_lines,
    })
}

fn compare_replay_bundle(expected: &V2ReplayBundle, rerun: &V2ReplayBundle) -> Result<(), String> {
    if rerun.cycle_count != expected.cycle_count {
        return Err(format!(
            "cycle_count mismatch: expected {}, got {}",
            expected.cycle_count, rerun.cycle_count
        ));
    }
    if rerun.retired_count != expected.retired_count {
        return Err(format!(
            "retired_count mismatch: expected {}, got {}",
            expected.retired_count, rerun.retired_count
        ));
    }
    if rerun.trace_entries != expected.trace_entries {
        return Err(format!(
            "trace_entries mismatch: expected {}, got {}",
            expected.trace_entries, rerun.trace_entries
        ));
    }
    if rerun.trace_hash != expected.trace_hash {
        return Err(format!(
            "trace_hash mismatch: expected 0x{:016X}, got 0x{:016X}",
            expected.trace_hash, rerun.trace_hash
        ));
    }
    if rerun.final_state_hash != expected.final_state_hash {
        return Err(format!(
            "final_state_hash mismatch: expected 0x{:016X}, got 0x{:016X}",
            expected.final_state_hash, rerun.final_state_hash
        ));
    }
    if rerun.hybrid != expected.hybrid {
        return Err(format!(
            "hybrid counter mismatch: expected {:?}, got {:?}",
            expected.hybrid, rerun.hybrid
        ));
    }
    if rerun.trace_lines != expected.trace_lines {
        return Err("trace_lines mismatch".to_string());
    }
    if rerun.mmio_mode != expected.mmio_mode {
        return Err(format!(
            "mmio_mode mismatch: expected '{}', got '{}'",
            expected.mmio_mode, rerun.mmio_mode
        ));
    }

    if expected.mmio_mode == V2_REPLAY_MMIO_MODE_SNAPSHOT {
        if rerun.mmio_snapshot_kind != expected.mmio_snapshot_kind {
            return Err(format!(
                "mmio_snapshot_kind mismatch: expected {:?}, got {:?}",
                expected.mmio_snapshot_kind, rerun.mmio_snapshot_kind
            ));
        }
        if rerun.mmio_initial_state != expected.mmio_initial_state {
            return Err("mmio_initial_state mismatch".to_string());
        }
        if rerun.mmio_final_state != expected.mmio_final_state {
            return Err("mmio_final_state mismatch".to_string());
        }
    }

    Ok(())
}

fn is_supported_schema(schema_version: u32) -> bool {
    matches!(
        schema_version,
        V2_REPLAY_SCHEMA_VERSION_V1 | V2_REPLAY_SCHEMA_VERSION_V2
    )
}

fn parse_manifest(text: &str) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut out = BTreeMap::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            return Err(format!("invalid manifest line {}: '{}'", lineno + 1, raw).into());
        };
        out.insert(k.trim().to_string(), v.trim().to_string());
    }
    Ok(out)
}

fn parse_u64(raw: &str) -> Result<u64, Box<dyn Error>> {
    let trimmed = raw.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        Ok(u64::from_str_radix(hex, 16)?)
    } else {
        Ok(trimmed.parse::<u64>()?)
    }
}

fn parse_u32(raw: &str) -> Result<u32, Box<dyn Error>> {
    let v = parse_u64(raw)?;
    u32::try_from(v).map_err(|_| format!("value out of range for u32: {v}").into())
}

fn parse_optional_u64(raw: Option<&str>) -> Result<u64, Box<dyn Error>> {
    match raw {
        Some(value) => parse_u64(value),
        None => Ok(0),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2 + 1);
    for &b in bytes {
        out.push(hex_digit((b >> 4) & 0x0F));
        out.push(hex_digit(b & 0x0F));
    }
    out.push('\n');
    out
}

fn hex_digit(v: u8) -> char {
    match v {
        0..=9 => (b'0' + v) as char,
        10..=15 => (b'A' + (v - 10)) as char,
        _ => '?',
    }
}

fn decode_hex(raw: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut cleaned = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_whitespace() {
            continue;
        }
        if !ch.is_ascii_hexdigit() {
            return Err(format!("invalid hex character '{ch}' in snapshot payload").into());
        }
        cleaned.push(ch);
    }

    if cleaned.len() % 2 != 0 {
        return Err("invalid hex payload length (must be even)".into());
    }

    let mut out = Vec::with_capacity(cleaned.len() / 2);
    for i in (0..cleaned.len()).step_by(2) {
        let byte = u8::from_str_radix(&cleaned[i..i + 2], 16)?;
        out.push(byte);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::tile_cpu::v2_components::{v2_halt, v2_inc, v2_ldb, v2_ldi, v2_stb};
    use crate::tile_cpu::v2_mmio::V2_MMIO_BASE;
    use crate::tile_cpu::v2_mmio_devices::{
        MMIO_CONSOLE_COUNT, MMIO_CONSOLE_DATA, MMIO_RNG_DATA, V2_MMIO_REF_SNAPSHOT_KIND,
        V2MmioRefDevicePack,
    };

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "engine_v2_replay_{label}_{}_{}",
            std::process::id(),
            nanos
        ));
        p
    }

    #[test]
    fn test_v2_replay_capture_and_verify_roundtrip() {
        let program = [v2_ldi(0, 41), v2_inc(0), v2_halt()];
        let bundle = capture_v2_replay_bundle(&program, 32).expect("capture should succeed");
        assert_eq!(bundle.schema_version, V2_REPLAY_SCHEMA_VERSION);
        assert_eq!(bundle.mmio_mode, V2_REPLAY_MMIO_MODE_NONE);
        assert!(bundle.trace_entries >= 2);
        verify_v2_replay_bundle(&bundle).expect("verification should pass");
    }

    #[test]
    fn test_v2_replay_bundle_io_roundtrip() {
        let program = [v2_ldi(0, 7), v2_inc(0), v2_halt()];
        let bundle = capture_v2_replay_bundle(&program, 32).expect("capture should succeed");
        let out_dir = unique_temp_dir("io");
        write_v2_replay_bundle_dir(&bundle, &out_dir).expect("write should succeed");
        let loaded = read_v2_replay_bundle_dir(&out_dir).expect("read should succeed");

        assert_eq!(bundle.schema_version, loaded.schema_version);
        assert_eq!(bundle.program_words, loaded.program_words);
        assert_eq!(bundle.max_cycles, loaded.max_cycles);
        assert_eq!(bundle.cycle_count, loaded.cycle_count);
        assert_eq!(bundle.retired_count, loaded.retired_count);
        assert_eq!(bundle.trace_entries, loaded.trace_entries);
        assert_eq!(bundle.trace_hash, loaded.trace_hash);
        assert_eq!(bundle.final_state_hash, loaded.final_state_hash);
        assert_eq!(bundle.hybrid, loaded.hybrid);
        assert_eq!(bundle.mmio_mode, loaded.mmio_mode);
        assert_eq!(bundle.mmio_snapshot_kind, loaded.mmio_snapshot_kind);
        assert_eq!(bundle.mmio_initial_state, loaded.mmio_initial_state);
        assert_eq!(bundle.mmio_final_state, loaded.mmio_final_state);
        assert_eq!(bundle.trace_lines, loaded.trace_lines);
        verify_v2_replay_bundle(&loaded).expect("loaded bundle should verify");

        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn test_v2_replay_schema_v1_loader_compatibility() {
        let program = [v2_ldi(0, 1), v2_halt()];
        let mut bundle = capture_v2_replay_bundle(&program, 16).expect("capture should succeed");
        bundle.schema_version = V2_REPLAY_SCHEMA_VERSION_V1;
        bundle.mmio_mode = V2_REPLAY_MMIO_MODE_NONE.to_string();
        bundle.mmio_snapshot_kind = None;
        bundle.mmio_initial_state = None;
        bundle.mmio_final_state = None;

        let out_dir = unique_temp_dir("v1_compat");
        write_v2_replay_bundle_dir(&bundle, &out_dir).expect("write should succeed");
        let loaded = read_v2_replay_bundle_dir(&out_dir).expect("read should succeed");

        assert_eq!(loaded.schema_version, V2_REPLAY_SCHEMA_VERSION_V1);
        assert_eq!(loaded.mmio_mode, V2_REPLAY_MMIO_MODE_NONE);
        assert!(loaded.mmio_snapshot_kind.is_none());
        assert!(loaded.mmio_initial_state.is_none());
        assert!(loaded.mmio_final_state.is_none());
        verify_v2_replay_bundle(&loaded).expect("v1-compatible bundle should verify");

        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn test_v2_replay_loader_missing_hybrid_keys_defaults_zero() {
        let program = [v2_ldi(0, 1), v2_halt()];
        let bundle = capture_v2_replay_bundle(&program, 16).expect("capture should succeed");
        let out_dir = unique_temp_dir("missing_hybrid");
        write_v2_replay_bundle_dir(&bundle, &out_dir).expect("write should succeed");

        let manifest_path = out_dir.join(V2_REPLAY_MANIFEST_FILE);
        let manifest = read_to_string(&manifest_path).expect("manifest read should succeed");
        let filtered = manifest
            .lines()
            .filter(|line| !line.starts_with("hybrid_"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        write(&manifest_path, filtered).expect("manifest rewrite should succeed");

        let loaded = read_v2_replay_bundle_dir(&out_dir).expect("read should succeed");
        assert_eq!(loaded.hybrid, V2HybridAssistCounters::default());
        verify_v2_replay_bundle(&loaded).expect("bundle with missing hybrid keys should verify");

        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn test_v2_replay_mmio_snapshot_capture_and_verify() {
        let program = [
            v2_ldi(0, 0x41),
            v2_stb(0, MMIO_CONSOLE_DATA),
            v2_ldb(1, MMIO_CONSOLE_COUNT),
            v2_ldb(2, MMIO_RNG_DATA),
            v2_ldb(3, V2_MMIO_BASE),
            v2_halt(),
        ];
        let capture_mmio = V2MmioHandle::from_rc(Rc::new(V2MmioRefDevicePack::new(1234)));
        let bundle = capture_v2_replay_bundle_with_mmio_snapshot(&program, 64, capture_mmio)
            .expect("snapshot capture should succeed");
        assert_eq!(bundle.schema_version, V2_REPLAY_SCHEMA_VERSION_V2);
        assert_eq!(bundle.mmio_mode, V2_REPLAY_MMIO_MODE_SNAPSHOT);
        assert_eq!(
            bundle.mmio_snapshot_kind.as_deref(),
            Some(V2_MMIO_REF_SNAPSHOT_KIND)
        );
        assert!(bundle.mmio_initial_state.is_some());
        assert!(bundle.mmio_final_state.is_some());

        let out_dir = unique_temp_dir("mmio_snapshot");
        write_v2_replay_bundle_dir(&bundle, &out_dir).expect("write should succeed");
        let loaded = read_v2_replay_bundle_dir(&out_dir).expect("read should succeed");
        let verify_mmio = V2MmioHandle::from_rc(Rc::new(V2MmioRefDevicePack::new(0)));
        verify_v2_replay_bundle_with_mmio(&loaded, Some(verify_mmio))
            .expect("snapshot replay verification should pass");

        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn test_v2_replay_mmio_snapshot_verify_requires_handle() {
        let program = [v2_ldi(0, 7), v2_stb(0, MMIO_CONSOLE_DATA), v2_halt()];
        let capture_mmio = V2MmioHandle::from_rc(Rc::new(V2MmioRefDevicePack::new(99)));
        let bundle = capture_v2_replay_bundle_with_mmio_snapshot(&program, 64, capture_mmio)
            .expect("snapshot capture should succeed");
        let err =
            verify_v2_replay_bundle(&bundle).expect_err("snapshot verify without MMIO must fail");
        assert!(err.contains("requires MMIO handle"));
    }

    #[test]
    fn test_v2_replay_verify_detects_hash_mismatch() {
        let program = [v2_ldi(0, 1), v2_halt()];
        let mut bundle = capture_v2_replay_bundle(&program, 16).expect("capture should succeed");
        bundle.final_state_hash ^= 0x1234;
        let err = verify_v2_replay_bundle(&bundle).expect_err("must reject mismatched hash");
        assert!(err.contains("final_state_hash mismatch"));
    }
}
