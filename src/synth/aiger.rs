//! AIGER format read/write for AIG interoperability.
//!
//! Supports both binary AIGER (compact, production) and ASCII AIGER (debugging).
//! This enables round-tripping with ABC (`abc -c "read file.aig; print_stats"`).
//!
//! # Format overview
//!
//! Binary AIGER header: `aig M I L O A\n`
//! - M = max variable index, I = inputs, L = latches (0 for us), O = outputs, A = AND nodes
//! - Inputs are implicit (variables 1..I).
//! - Outputs are listed one per line as literal values.
//! - AND nodes are delta-encoded: each node stored as two unsigned deltas.

use super::aig::{Aig, AigLit};
use std::io::{BufRead, Read, Write};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum AigerError {
    Io(std::io::Error),
    Parse(String),
}

impl From<std::io::Error> for AigerError {
    fn from(e: std::io::Error) -> Self {
        AigerError::Io(e)
    }
}

impl std::fmt::Display for AigerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AigerError::Io(e) => write!(f, "IO error: {}", e),
            AigerError::Parse(s) => write!(f, "Parse error: {}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// Delta encoding (LEB128-like, unsigned)
// ---------------------------------------------------------------------------

fn encode_delta<W: Write>(w: &mut W, value: u32) -> Result<(), AigerError> {
    let mut v = value;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            w.write_all(&[byte])?;
            return Ok(());
        }
        w.write_all(&[byte | 0x80])?;
    }
}

fn decode_delta<R: Read>(r: &mut R) -> Result<u32, AigerError> {
    let mut result = 0u32;
    let mut shift = 0u32;
    loop {
        let mut byte = [0u8; 1];
        r.read_exact(&mut byte)?;
        result |= ((byte[0] & 0x7F) as u32) << shift;
        if byte[0] & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift > 35 {
            return Err(AigerError::Parse("delta overflow".into()));
        }
    }
}

// ---------------------------------------------------------------------------
// Binary AIGER write
// ---------------------------------------------------------------------------

/// Write AIG in binary AIGER format.
///
/// Combinational only (L=0). No symbol table for now.
pub fn write_aiger<W: Write>(aig: &Aig, w: &mut W) -> Result<(), AigerError> {
    let num_i = aig.num_inputs();
    let num_a = aig.num_nodes() as u32;
    let num_o = aig.num_output_bits() as u32;
    let max_var = num_i + num_a; // M = I + A (no latches)

    // Header
    let header = format!("aig {} {} 0 {} {}\n", max_var, num_i, num_o, num_a);
    w.write_all(header.as_bytes())?;

    // Output literals (one per line, ASCII)
    for lit in aig.output_lits() {
        let line = format!("{}\n", lit.raw());
        w.write_all(line.as_bytes())?;
    }

    // AND nodes (binary delta-encoded)
    // Each AND node i has lhs = 2 * (num_i + 1 + i).
    // AIGER requires lhs > rhs0 >= rhs1.
    // We store: delta0 = lhs - rhs0, delta1 = rhs0 - rhs1.
    for (i, node) in aig.nodes().iter().enumerate() {
        let lhs = 2 * (num_i + 1 + i as u32);
        // Ensure rhs0 >= rhs1 for AIGER spec compliance
        let (rhs0, rhs1) = if node.fanin0.raw() >= node.fanin1.raw() {
            (node.fanin0.raw(), node.fanin1.raw())
        } else {
            (node.fanin1.raw(), node.fanin0.raw())
        };
        debug_assert!(
            lhs > rhs0,
            "AIGER invariant violated: lhs={} <= rhs0={}",
            lhs,
            rhs0
        );
        let delta0 = lhs - rhs0;
        let delta1 = rhs0 - rhs1;
        encode_delta(w, delta0)?;
        encode_delta(w, delta1)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Binary AIGER read
// ---------------------------------------------------------------------------

/// Read AIG from binary AIGER format.
///
/// Supports combinational AIGER only (L must be 0).
pub fn read_aiger<R: BufRead>(r: &mut R) -> Result<Aig, AigerError> {
    // Read header line
    let mut header_line = String::new();
    r.read_line(&mut header_line)?;
    let header_line = header_line.trim();

    let parts: Vec<&str> = header_line.split_whitespace().collect();
    if parts.len() < 6 {
        return Err(AigerError::Parse(format!(
            "bad header (expected 6 fields): '{}'",
            header_line
        )));
    }

    let is_binary = parts[0] == "aig";
    let _is_ascii = parts[0] == "aag";
    if !is_binary && !_is_ascii {
        return Err(AigerError::Parse(format!(
            "unknown format '{}' (expected 'aig' or 'aag')",
            parts[0]
        )));
    }

    let _max_var: u32 = parts[1]
        .parse()
        .map_err(|_| AigerError::Parse("bad M".into()))?;
    let num_i: u32 = parts[2]
        .parse()
        .map_err(|_| AigerError::Parse("bad I".into()))?;
    let num_l: u32 = parts[3]
        .parse()
        .map_err(|_| AigerError::Parse("bad L".into()))?;
    let num_o: u32 = parts[4]
        .parse()
        .map_err(|_| AigerError::Parse("bad O".into()))?;
    let num_a: u32 = parts[5]
        .parse()
        .map_err(|_| AigerError::Parse("bad A".into()))?;

    if num_l != 0 {
        return Err(AigerError::Parse(
            "latches not supported (combinational only)".into(),
        ));
    }

    let mut aig = Aig::with_inputs(num_i);

    if is_binary {
        // Read output literals (one per line, ASCII)
        let mut output_lits = Vec::with_capacity(num_o as usize);
        for _ in 0..num_o {
            let mut line = String::new();
            r.read_line(&mut line)?;
            let lit_val: u32 = line
                .trim()
                .parse()
                .map_err(|_| AigerError::Parse("bad output literal".into()))?;
            output_lits.push(AigLit::from_raw(lit_val));
        }

        // Read AND nodes (binary delta-encoded)
        for i in 0..num_a {
            let lhs = 2 * (num_i + 1 + i);
            let delta0 = decode_delta(r)?;
            let delta1 = decode_delta(r)?;
            let rhs0 = lhs - delta0;
            let rhs1 = rhs0 - delta1;
            aig.push_node_raw(AigLit::from_raw(rhs0), AigLit::from_raw(rhs1));
        }

        // Register outputs
        for (i, lit) in output_lits.into_iter().enumerate() {
            aig.push_output_raw(format!("o{}", i), lit);
        }
    } else {
        // ASCII AIGER
        read_aiger_ascii_body(r, &mut aig, num_i, num_o, num_a)?;
    }

    Ok(aig)
}

/// Read the body of an ASCII AIGER file.
fn read_aiger_ascii_body<R: BufRead>(
    r: &mut R,
    aig: &mut Aig,
    num_i: u32,
    num_o: u32,
    num_a: u32,
) -> Result<(), AigerError> {
    // Input lines (one literal per line)
    for _ in 0..num_i {
        let mut line = String::new();
        r.read_line(&mut line)?;
        // Input literals are implicit in binary format, but explicit in ASCII.
        // We already created them in with_inputs(), just verify.
    }

    // Output lines
    let mut output_lits = Vec::with_capacity(num_o as usize);
    for _ in 0..num_o {
        let mut line = String::new();
        r.read_line(&mut line)?;
        let lit_val: u32 = line
            .trim()
            .parse()
            .map_err(|_| AigerError::Parse("bad output literal".into()))?;
        output_lits.push(AigLit::from_raw(lit_val));
    }

    // AND lines: "lhs rhs0 rhs1"
    for _ in 0..num_a {
        let mut line = String::new();
        r.read_line(&mut line)?;
        let nums: Vec<u32> = line
            .split_whitespace()
            .map(|s| {
                s.parse()
                    .map_err(|_| AigerError::Parse("bad AND literal".into()))
            })
            .collect::<Result<_, _>>()?;
        if nums.len() != 3 {
            return Err(AigerError::Parse("AND line needs 3 values".into()));
        }
        // lhs = nums[0], rhs0 = nums[1], rhs1 = nums[2]
        aig.push_node_raw(AigLit::from_raw(nums[1]), AigLit::from_raw(nums[2]));
    }

    for (i, lit) in output_lits.into_iter().enumerate() {
        aig.push_output_raw(format!("o{}", i), lit);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// ASCII AIGER write (for debugging)
// ---------------------------------------------------------------------------

/// Write AIG in ASCII AIGER format.
///
/// Slower but human-readable. Useful for debugging and verification.
pub fn write_aiger_ascii<W: Write>(aig: &Aig, w: &mut W) -> Result<(), AigerError> {
    let num_i = aig.num_inputs();
    let num_a = aig.num_nodes() as u32;
    let num_o = aig.num_output_bits() as u32;
    let max_var = num_i + num_a;

    // Header
    let header = format!("aag {} {} 0 {} {}\n", max_var, num_i, num_o, num_a);
    w.write_all(header.as_bytes())?;

    // Input literals
    for i in 1..=num_i {
        let line = format!("{}\n", 2 * i);
        w.write_all(line.as_bytes())?;
    }

    // Output literals
    for lit in aig.output_lits() {
        let line = format!("{}\n", lit.raw());
        w.write_all(line.as_bytes())?;
    }

    // AND lines: "lhs rhs0 rhs1"
    for (i, node) in aig.nodes().iter().enumerate() {
        let lhs = 2 * (num_i + 1 + i as u32);
        let line = format!("{} {} {}\n", lhs, node.fanin0.raw(), node.fanin1.raw());
        w.write_all(line.as_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn aiger_roundtrip_single_and() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let mut buf = Vec::new();
        write_aiger(&aig, &mut buf).unwrap();

        let aig2 = read_aiger(&mut BufReader::new(&buf[..])).unwrap();
        assert_eq!(aig2.num_inputs(), 2);
        assert_eq!(aig2.num_nodes(), 1);
        assert_eq!(aig2.num_output_bits(), 1);
    }

    #[test]
    fn aiger_roundtrip_xor() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.xor(a, b);
        aig.add_output("y", y);

        let mut buf = Vec::new();
        write_aiger(&aig, &mut buf).unwrap();

        let aig2 = read_aiger(&mut BufReader::new(&buf[..])).unwrap();
        assert_eq!(aig2.num_inputs(), 2);
        assert_eq!(aig2.num_nodes(), aig.num_nodes());
        assert_eq!(aig2.num_output_bits(), 1);
    }

    #[test]
    fn aiger_roundtrip_multi_output() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let y1 = aig.and(a, b);
        let y2 = aig.or(b, c);
        let y3 = aig.xor(a, c);
        aig.add_output("y1", y1);
        aig.add_output("y2", y2);
        aig.add_output("y3", y3);

        let mut buf = Vec::new();
        write_aiger(&aig, &mut buf).unwrap();

        let aig2 = read_aiger(&mut BufReader::new(&buf[..])).unwrap();
        assert_eq!(aig2.num_inputs(), 3);
        assert_eq!(aig2.num_nodes(), aig.num_nodes());
        assert_eq!(aig2.num_output_bits(), 3);
    }

    #[test]
    fn aiger_ascii_roundtrip() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let mut buf = Vec::new();
        write_aiger_ascii(&aig, &mut buf).unwrap();

        let text = String::from_utf8(buf.clone()).unwrap();
        assert!(text.starts_with("aag 3 2 0 1 1\n"));

        let aig2 = read_aiger(&mut BufReader::new(&buf[..])).unwrap();
        assert_eq!(aig2.num_inputs(), 2);
        assert_eq!(aig2.num_nodes(), 1);
        assert_eq!(aig2.num_output_bits(), 1);
    }

    #[test]
    fn aiger_constant_output() {
        // Circuit where the output is a constant (no AND nodes needed)
        let mut aig = Aig::new();
        let _a = aig.add_input("a");
        aig.add_output("y", AigLit::TRUE);

        let mut buf = Vec::new();
        write_aiger(&aig, &mut buf).unwrap();

        let aig2 = read_aiger(&mut BufReader::new(&buf[..])).unwrap();
        assert_eq!(aig2.num_inputs(), 1);
        assert_eq!(aig2.num_nodes(), 0);
        assert_eq!(aig2.num_output_bits(), 1);
        // The output should be literal 1 (TRUE)
        let out_lit: Vec<_> = aig2.output_lits().collect();
        assert_eq!(out_lit[0], AigLit::TRUE);
    }
}
