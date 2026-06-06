use std::io::Write;
use std::process::{Command, Stdio};

#[test]
#[ignore] // Requires logic-cli binary to be built first
fn cli_boots_and_exits() {
    // Spawn the already-built CLI binary
    let exe = std::env::var("CARGO_BIN_EXE_logic_cli").unwrap_or_else(|_| {
        if cfg!(windows) {
            String::from("target\\debug\\logic-cli.exe")
        } else {
            String::from("target/debug/logic-cli")
        }
    });
    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start CLI");

    {
        let stdin = child.stdin.as_mut().expect("Failed to open stdin");
        stdin
            .write_all(b"exit\n")
            .expect("Failed to write exit command");
    }

    let status = child.wait().expect("CLI did not exit correctly");
    assert!(status.success());
}
