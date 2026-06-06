use engine::chungus2_assembler::assemble;

fn main() {
    let source = r#"
        // CHUNGUS 2 Test Program
        // Calculate Fibonacci
        
        LDI R0, 0      ; Fib A
        LDI R1, 1      ; Fib B
        LDI R2, 10     ; Counter
        LDI R3, 1      ; Decrement
        
        start:
        ADD R4, R0, R1 ; Temp = A + B
        MOV R0, R1     ; A = B
        MOV R1, R4     ; B = Temp
        
        SUB R2, R2, R3 ; Dec Counter
        JNZ R2, start  ; Loop if not zero
        
        JMP end
        
        end:
        NOP
    "#;

    println!("Assembling CHUNGUS 2 Code...");
    match assemble(source) {
        Ok(binary) => {
            println!("Success! Size: {} bytes", binary.len());
            print!("Hex: ");
            for b in &binary {
                print!("{:02X} ", b);
            }
            println!("\n");

            // Decode print
            println!("Disassembly (Approx):");
            let mut pc = 0;
            while pc < binary.len() {
                if pc + 1 >= binary.len() {
                    break;
                }
                let b_low = binary[pc];
                let b_high = binary[pc + 1];
                let word = (b_high as u16) << 8 | (b_low as u16);

                let op = (word >> 11) & 0x1F;
                let rd = (word >> 8) & 0x7;
                let rs1 = (word >> 5) & 0x7;
                let rs2 = (word >> 2) & 0x7;

                println!(
                    "PC {:02X}: Op {:02} | Rd R{} | Rs1 R{} | Rs2 R{} | Raw {:04X}",
                    pc, op, rd, rs1, rs2, word
                );

                pc += 2;
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
