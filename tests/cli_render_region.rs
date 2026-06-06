use std::process::Command;

fn bin_path() -> String {
    std::env::var("CARGO_BIN_EXE_logic_cli")
        .unwrap_or_else(|_| "target/debug/logic-cli".to_string())
}

#[test]
#[ignore] // Requires logic-cli binary
fn render_region_ppm_smoke() {
    let out = "out_epic21.ppm";
    let _ = std::fs::remove_file(out);
    let mut cmd = Command::new(bin_path());
    cmd.arg("");
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    let mut child = cmd.spawn().expect("spawn cli");
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("stdin");
        // Render a tiny region
        writeln!(stdin, "render_region 0 0 4 3 {} heat 2", out).unwrap();
        writeln!(stdin, "exit").unwrap();
    }
    let output = child.wait_with_output().expect("wait");
    assert!(output.status.success());
    // File exists and header is correct
    let bytes = std::fs::read(out).expect("read ppm");
    assert!(bytes.starts_with(b"P6\n8 6\n255\n"));
    // Width=4, height=3, scale=2 => 8x6; data len matches
    let header_len = b"P6\n8 6\n255\n".len();
    assert_eq!(bytes.len(), header_len + 8 * 6 * 3);
    let _ = std::fs::remove_file(out);
}
