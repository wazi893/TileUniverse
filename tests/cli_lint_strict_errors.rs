use std::process::Command;

fn bin_path() -> String {
    std::env::var("CARGO_BIN_EXE_logic_cli")
        .unwrap_or_else(|_| "target/debug/logic-cli".to_string())
}

#[test]
#[ignore] // Requires logic-cli binary
fn lint_strict_reports_issues() {
    let mut cmd = Command::new(bin_path());
    cmd.arg("");
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn cli");
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("stdin");
        // Create a floating net and an unclocked register to trigger issues
        writeln!(stdin, "set_tile 5 5 Register8").unwrap();
        writeln!(stdin, "set_tile 1 1 Wire").unwrap();
        writeln!(stdin, "set_logic 1 1 0x1").unwrap();
        writeln!(stdin, "lint strict").unwrap();
        writeln!(stdin, "exit").unwrap();
    }
    let output = child.wait_with_output().expect("wait");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Lint:"));
    assert!(
        stdout.to_ascii_lowercase().contains("error")
            || stdout.to_ascii_lowercase().contains("warning")
    );
}
