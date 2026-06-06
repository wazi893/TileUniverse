//! V2 register aliases and leaf subroutine library (Sprint 182.1).
//!
//! Provides register alias constants for the V2 calling convention and
//! inline assembly snippets for common branchless operations. All
//! subroutines are leaf-only (LR is not addressable as a GPR, so
//! nested CALL is impossible).

/// Argument register 0 (R0). Also return value register.
pub const V2_ARG0: usize = 0;
/// Argument register 1 (R1).
pub const V2_ARG1: usize = 1;
/// Argument register 2 (R2).
pub const V2_ARG2: usize = 2;
/// Argument register 3 (R3).
pub const V2_ARG3: usize = 3;
/// Return value register (R0, same as ARG0).
pub const V2_RET: usize = 0;
/// Stack pointer register (R15).
pub const V2_SP: usize = 15;

/// Inline assembly: MAX(R0, R1) -> R0.
/// CMP sets C=1 when R0 < R1. MOVC fires when C=1, replacing R0 with R1.
pub const V2_STDLIB_MAX: &str = "CMP R0, R1\n    MOVC R0, R1";

/// Inline assembly: MIN(R0, R1) -> R0.
/// CMP sets C=1 when R0 < R1. MOVNC fires when C=0 (R0 >= R1), replacing R0 with R1.
pub const V2_STDLIB_MIN: &str = "CMP R0, R1\n    MOVNC R0, R1";

/// Inline assembly: CLAMP(R0, lo=R1, hi=R2) -> R0.
/// First applies max(R0, lo) via MOVC, then min(result, hi) via MOVNC.
pub const V2_STDLIB_CLAMP: &str = "CMP R0, R1\n    MOVC R0, R1\n    CMP R0, R2\n    MOVNC R0, R2";

/// Inline assembly: MEMSET4 — write R1 to [R0+0]..[R0+3].
/// R0 = base address, R1 = value. Uses indexed ST (Sprint 175).
pub const V2_STDLIB_MEMSET4: &str =
    "ST [R0+0], R1\n    ST [R0+1], R1\n    ST [R0+2], R1\n    ST [R0+3], R1";

/// Inline assembly: ABS(R0) -> R0 (unsigned semantics).
/// Computes R1 = 0 - R0, then keeps whichever is smaller (unsigned).
/// For values in the lower half of the u64 range, this is identity.
/// For values near u64::MAX (i.e., negative in two's complement), this yields -R0.
pub const V2_STDLIB_ABS: &str = "LDI R1, 0\n    SUB R1, R0\n    CMP R0, R1\n    MOVNC R0, R1";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_cpu::v2_assembler::assemble_v2;

    #[test]
    fn test_v2_stdlib_max_assembles() {
        let src = format!("LDI R0, 30\n LDI R1, 50\n {}\n HALT", V2_STDLIB_MAX);
        let prog = assemble_v2(&src).expect("MAX snippet failed to assemble");
        assert!(prog.len() <= 64);
    }

    #[test]
    fn test_v2_stdlib_min_assembles() {
        let src = format!("LDI R0, 50\n LDI R1, 30\n {}\n HALT", V2_STDLIB_MIN);
        let prog = assemble_v2(&src).expect("MIN snippet failed to assemble");
        assert!(prog.len() <= 64);
    }

    #[test]
    fn test_v2_stdlib_clamp_assembles() {
        let src = format!(
            "LDI R0, 5\n LDI R1, 10\n LDI R2, 40\n {}\n HALT",
            V2_STDLIB_CLAMP
        );
        let prog = assemble_v2(&src).expect("CLAMP snippet failed to assemble");
        assert!(prog.len() <= 64);
    }

    #[test]
    fn test_v2_stdlib_memset4_assembles() {
        let src = format!("LDI R0, 8\n LDI R1, 42\n {}\n HALT", V2_STDLIB_MEMSET4);
        let prog = assemble_v2(&src).expect("MEMSET4 snippet failed to assemble");
        assert!(prog.len() <= 64);
    }

    #[test]
    fn test_v2_stdlib_abs_assembles() {
        let src = format!("LDI R0, 5\n {}\n HALT", V2_STDLIB_ABS);
        let prog = assemble_v2(&src).expect("ABS snippet failed to assemble");
        assert!(prog.len() <= 64);
    }
}
