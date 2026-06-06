use std::process::Command;

fn bin_path() -> String {
    std::env::var("CARGO_BIN_EXE_logic_cli")
        .unwrap_or_else(|_| "target/debug/logic-cli".to_string())
}

#[test]
#[ignore] // Requires logic-cli binary
fn gui_frames_smoke() {
    let outdir = "target/test_output/gui_rec_cli";
    let _ = std::fs::create_dir_all("target/test_output");
    let _ = std::fs::remove_dir_all(outdir);
    let mut cmd = Command::new(bin_path());
    cmd.arg("");
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn cli");
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("stdin");
        writeln!(
            stdin,
            "record_organism ring_cluster steps 2 origin 3 3 dir {}",
            outdir
        )
        .unwrap();
        writeln!(stdin, "gui_frames {}", outdir).unwrap();
        writeln!(stdin, "exit").unwrap();
    }
    let output = child.wait_with_output().expect("wait");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("frame index=0"));
    assert!(stdout.contains("checksum=0x"));
}
