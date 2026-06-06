use engine::assembler;

fn main() {
    let source = r#"
        ; Test Program
        MOV A, 10
        MOV B, 20
        ADD A, B
        OUT A
        JMP start
        start:
        MOV C, 0xFF
    "#;

    println!("Assembling...");
    match assembler::assemble(source) {
        Ok(bytes) => {
            println!("Success! {} bytes generated.", bytes.len());
            for (i, chunk) in bytes.chunks(4).enumerate() {
                println!(
                    "{:04X}: {:02X} {:02X} {:02X} {:02X}",
                    i * 4,
                    chunk[0],
                    chunk[1],
                    chunk[2],
                    chunk[3]
                );
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
