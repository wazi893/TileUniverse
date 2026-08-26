//! Datacenter control-evaluation dashboard.
//!
//! Runs the thermal covert-channel red-team twice: once with the adversary active (the attack)
//! and once with it silent (the false-positive baseline), then prints the control-eval report.
//!
//! Run: `cargo run --release --example control_eval`

use engine::control_eval::{ControlEvalConfig, run};

fn main() {
    println!("Deterministic datacenter control evaluation");
    println!("===========================================\n");

    // 1. The attack: an untrusted co-tenant exfiltrates a secret over a thermal side channel.
    let attack = run(ControlEvalConfig {
        secret: 0xB4,
        adversary_enabled: true,
        ..Default::default()
    });
    print!("{}", attack.summary());

    // 2. The control: the same rack with no adversary — does the monitor stay quiet?
    let baseline = run(ControlEvalConfig {
        adversary_enabled: false,
        ..Default::default()
    });
    println!();
    print!("{}", baseline.summary());

    println!("\nVerdict");
    println!("-------");
    let detected = attack.detection_tick.is_some();
    let no_false_pos = baseline.detection_tick.is_none();
    println!(
        "  detection of real exfiltration : {}",
        if detected { "PASS" } else { "FAIL" }
    );
    println!(
        "  no false positive on benign    : {}",
        if no_false_pos { "PASS" } else { "FAIL" }
    );
    println!(
        "  channel contained              : pre-BER {}% -> post-BER {}%",
        attack.pre_containment_ber_pct, attack.post_containment_ber_pct
    );
    println!(
        "  bits leaked before catch       : {}/{}",
        attack.bits_leaked_before_detection, attack.bits_transmitted_before_detection
    );
    println!(
        "\nEvery number above is deterministic: re-run and it is bit-identical.\n\
         The CLI `control_eval ... record DIR` proves it — two independent runs, frame diff = black."
    );
}
