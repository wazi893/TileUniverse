use std::process::Command;

fn bin_path() -> String {
    std::env::var("CARGO_BIN_EXE_logic_cli")
        .unwrap_or_else(|_| "target/debug/logic-cli".to_string())
}

#[test]
#[ignore] // Requires logic-cli binary
fn gui_info_and_frame_smoke() {
    let outdir = "target/test_output/gui_api_rec_cli";
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
            "record_organism heat_emitter steps 1 origin 2 2 dir {}",
            outdir
        )
        .unwrap();
        writeln!(stdin, "gui_info {}", outdir).unwrap();
        writeln!(stdin, "gui_frame {} 0", outdir).unwrap();
        writeln!(stdin, "gui_dump {} {}_dump", outdir, outdir).unwrap();
        writeln!(stdin, "exit").unwrap();
    }
    let output = child.wait_with_output().expect("wait");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GUI Run:"));
    assert!(stdout.contains("frame index=0"));
    assert!(stdout.contains("GUI dump OK"));
}
