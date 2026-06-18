//! Step 1 artifact: the full-phase CPU prover report for the V2-on-CUDA charter set.
//!
//! Proves, on the CPU, the device-resident cycle spec the CUDA megakernel will run:
//!   * per-phase compact-op fidelity (op-array eval == propagate_compact) for all 5 phases
//!   * the unified slot universe size (the megakernel's vals[slot*K + lane] buffer)
//!   * LD/ST counts, MUL usage, inhibit/MMIO status
//!   * settle-substituted full-cycle hash parity vs TileCpuV2::step (the oracle)
//!   * pass/fail per charter program  -> clean handoff into the megakernel
//!
//! Run: `cargo run --release --example v2_full_phase_prover`

use engine::tile_cpu::{CharterProgramReport, PHASE_NAMES, charter_cases, prove_charter};

fn print_report(r: &CharterProgramReport) {
    println!("\n### {}  ({} cycles, R0={}{})", r.name, r.cycles, r.r0, {
        if r.r0 == r.expected_r0 {
            String::new()
        } else {
            format!(" EXPECTED {}", r.expected_r0)
        }
    });

    // Per-phase preflight + op-array fidelity.
    println!("  phases (tick order: Stage X branch/commit/clock -> Stage F cone/settle):");
    println!(
        "    {:<8}{:>8}{:>9}{:>9}",
        "phase", "ops", "builds", "equiv"
    );
    for (i, p) in r.phases.iter().enumerate() {
        let name = PHASE_NAMES.get(i).copied().unwrap_or(p.name);
        println!(
            "    {:<8}{:>8}{:>9}{:>9}{}",
            name,
            p.ops,
            if p.kernel_builds { "yes" } else { "NO" },
            if p.equiv_ok { "yes" } else { "NO" },
            p.build_err
                .as_ref()
                .map(|e| format!("   ({e})"))
                .unwrap_or_default(),
        );
    }

    let subst = match r.hash_substituted {
        Some(h) if h == r.hash_authority => {
            format!("yes (parity, {} settles)", r.substituted_dense_settles)
        }
        Some(_) => "NO (HASH MISMATCH)".to_string(),
        None => "skipped".to_string(),
    };
    println!(
        "  slot-union (megakernel buffer): {} tiles | RAM SoA: {} words | static LD/ST/MUL: {}/{}/{}",
        r.slot_union, r.ram_words, r.stat.ld, r.stat.st, r.stat.mul
    );
    println!(
        "  inhibit: {} | mmio: {} | authority hash: {:#018x} | settle-substituted: {}",
        r.inhibit,
        if r.mmio { "yes" } else { "no" },
        r.hash_authority,
        subst
    );
    println!(
        "  final state: pc={} z={} c={} halted={} | R0..R3 = {:?}",
        r.final_state.pc,
        r.final_state.flag_z as u8,
        r.final_state.flag_c as u8,
        r.final_state.halted,
        &r.final_state.regs[0..4],
    );
    println!("  ==> {}", if r.pass { "PASS" } else { "FAIL" });
}

fn main() {
    println!("V2 -> CUDA migration, Step 1: full-phase CPU prover (charter set)");
    println!("oracle: bit-exact hash_v2_final_state parity vs TileCpuV2::step\n");

    let mut all_pass = true;
    let mut reports = Vec::new();
    for case in charter_cases() {
        match prove_charter(case) {
            Ok(r) => {
                all_pass &= r.pass;
                print_report(&r);
                reports.push(r);
            }
            Err(e) => {
                all_pass = false;
                println!("\n### {} -> PROVE ERROR: {e}", case.name);
            }
        }
    }

    println!("\n{}", "=".repeat(72));
    let union_max = reports.iter().map(|r| r.slot_union).max().unwrap_or(0);
    let phase_ops: Vec<String> = PHASE_NAMES
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let ops = reports.first().map(|r| r.phases[i].ops).unwrap_or(0);
            format!("{n}={ops}")
        })
        .collect();
    println!("MEGAKERNEL HANDOFF:");
    println!(
        "  unified slot-universe (max): {union_max} tiles  -> vals[{union_max} * K] u64 buffer"
    );
    println!("  per-cycle phase op arrays: {}", phase_ops.join("  "));
    println!(
        "  charter set: {}",
        charter_cases()
            .iter()
            .map(|c| c.name)
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "\nNEXT INCREMENT (not in this step): lock the intra-Stage-X sub-order from\n  \
         run_stage_x, then port the scalar control shell (PC/latch/RAM-bank-swap/scalar-MUL)\n  \
         so all 5 phases run substituted device-side, not just settle."
    );
    println!("\nOVERALL: {}", if all_pass { "PASS" } else { "FAIL" });
    assert!(all_pass, "full-phase prover charter set did not pass");
}
