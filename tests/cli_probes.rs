use std::process::Command;

fn bin_path() -> String {
    std::env::var("CARGO_BIN_EXE_logic_cli")
        .unwrap_or_else(|_| "target/debug/logic-cli".to_string())
}

#[test]
#[ignore] // Requires logic-cli binary
fn probe_logic_smoke() {
    let mut cmd = Command::new(bin_path());
    cmd.arg("");
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn cli");
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("stdin");
        writeln!(stdin, "probe_logic 0 0 3").unwrap();
        writeln!(stdin, "exit").unwrap();
    }
    let output = child.wait_with_output().expect("wait");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Probe Logic"));
    assert!(stdout.contains("t  value"));
}

#[test]
#[ignore] // Requires logic-cli binary
fn probe_field_smoke() {
    let mut cmd = Command::new(bin_path());
    cmd.arg("");
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn cli");
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("stdin");
        writeln!(stdin, "set_logic 1 1 0x1").unwrap();
        writeln!(stdin, "probe_field power 1 1 2").unwrap();
        writeln!(stdin, "exit").unwrap();
    }
    let output = child.wait_with_output().expect("wait");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Probe FieldPower"));
    assert!(stdout.contains("t  value"));
}
