//! Protective test gate for the self-contained HTML "player" family.
//!
//! Each player ships as an `export_default_*` function that captures real engine
//! data and writes one interactive, self-contained HTML file. Those export paths
//! had no automated coverage (the player modules unit-test `build_*_html`, but not
//! the file-write path or the on-disk result), so a broken template, a missing
//! `include_str!`, or a capture panic could silently rot a portfolio artifact.
//!
//! This gate exercises every CPU-only export end-to-end (capture -> serialize ->
//! write -> validate the bytes on disk). The CUDA-gated cellular player is excluded
//! on purpose so the gate stays GPU-independent and fast.

use std::fs;
use std::path::{Path, PathBuf};

/// A unique temp path under `target/` (gitignored) for one export, removed after.
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    p.push(format!("player_export_{tag}.html"));
    p
}

/// Assert the written file is a real, populated, self-contained HTML document with no
/// leftover injection marker (every template's marker ends in `__*/null`; a successful
/// export replaces exactly that token, so the substring must be gone).
fn assert_valid_player(path: &Path, min_kb: f64) {
    let bytes = fs::read(path).expect("export should have written a file");
    let kb = bytes.len() as f64 / 1024.0;
    assert!(
        kb >= min_kb,
        "{} is only {:.1} KB (< {:.1} KB) — capture or template likely empty",
        path.display(),
        kb,
        min_kb
    );
    let html = String::from_utf8(bytes).expect("player HTML must be valid UTF-8");
    assert!(
        html.starts_with("<!DOCTYPE html"),
        "{} should be an HTML document",
        path.display()
    );
    assert!(
        html.contains("</html>"),
        "{} should be a complete HTML document",
        path.display()
    );
    assert!(
        !html.contains("BUNDLE__*/null"),
        "{} still contains an unfilled data marker — injection failed",
        path.display()
    );
    let _ = fs::remove_file(path);
}

/// Run one export to a temp file, validate the result, and clean up.
fn check(tag: &str, min_kb: f64, export: impl Fn(&Path) -> Result<(), String>) {
    let path = tmp_path(tag);
    export(&path).unwrap_or_else(|e| panic!("export {tag} failed: {e}"));
    assert_valid_player(&path, min_kb);
}

#[test]
fn aicontrol_family_exports_are_valid() {
    use engine::aicontrol_hub::export_default_hub;
    use engine::control_eval_player::export_default_ctrl_player;
    use engine::sched_eval_player::export_default_sched_player;

    check("sched", 12.0, |p| export_default_sched_player(p));
    check("control", 40.0, |p| export_default_ctrl_player(p));
    // The hub embeds both players, so it is necessarily the largest.
    check("hub", 100.0, |p| export_default_hub(p));
}

#[test]
fn alphafabric_player_export_is_valid() {
    check("alphafabric", 8.0, |p| {
        engine::alphafabric_player::export_default_af_player(p)
    });
}

#[test]
fn qec_player_export_is_valid() {
    check("qec", 6.0, |p| {
        engine::qec_player::export_default_qec_player(p)
    });
}

#[test]
fn snn_players_export_is_valid() {
    check("snn", 6.0, |p| {
        engine::snn_player::export_default_snn_player(p)
    });
    check("snn_tile_lif", 6.0, |p| {
        engine::snn::tile_lif_player::export_default_player(p)
    });
}

#[test]
fn tile_cpu_v2_player_export_is_valid() {
    check("tile_cpu_v2", 6.0, |p| {
        engine::tile_cpu::v2_player::export_default_player(p)
    });
}
