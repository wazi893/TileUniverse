use std::process::Command;

fn bin_path() -> String {
    std::env::var("CARGO_BIN_EXE_logic_cli")
        .unwrap_or_else(|_| "target/debug/logic-cli".to_string())
}

#[test]
#[ignore] // Requires logic-cli binary
fn nets_summary_reports_clock_and_floating() {
    let mut cmd = Command::new(bin_path());
    cmd.arg("");
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn cli");
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("stdin");
        writeln!(stdin, "set_tile 1 1 ClockGlobal").unwrap();
        writeln!(stdin, "set_tile 2 1 Wire").unwrap();
        writeln!(stdin, "set_logic 2 1 0xAAAA").unwrap();
        writeln!(stdin, "nets_summary").unwrap();
        writeln!(stdin, "exit").unwrap();
    }
    let output = child.wait_with_output().expect("wait");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Nets: total="));
    assert!(stdout.contains("clock_nets="));
    assert!(stdout.contains("floating_nets="));
    assert!(stdout.contains("kind=Clock"));
    assert!(stdout.contains("floating"));
}
