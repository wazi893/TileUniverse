use std::io::Write;
use std::process::{Command, Stdio};

#[test]
#[ignore] // Requires logic-cli binary
fn cli_verbose_stress() {
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
        let script = b"verbose on\nverbose off\nverbose on\nverbose off\nverbose on\nexit\n";
        stdin.write_all(script).expect("Failed to write script");
    }

    let status = child.wait().expect("CLI did not exit correctly");
    assert!(status.success());
}
