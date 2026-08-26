//! Delfosse–Nickerson Union-Find decoder.
//!
//! This module is a deprecated alias of [`crate::qec::decoder::UnionFindDecoder`],
//! which now implements synchronized cluster growth and peeling.

/// Delfosse–Nickerson decoder (alias of the canonical Union-Find decoder).
///
/// Use [`crate::qec::decoder::UnionFindDecoder`].
#[deprecated(
    since = "0.79.0",
    note = "Use engine::qec::UnionFindDecoder; DN is now the same growth+peel decoder"
)]
pub type UnionFindDecoderDN = crate::qec::decoder::UnionFindDecoder;

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::UnionFindDecoderDN;
    use crate::qec::codes::SurfaceCode;

    #[test]
    fn dn_alias_constructs() {
        let decoder = UnionFindDecoderDN::new(3);
        assert_eq!(decoder.distance(), 3);
    }

    #[test]
    fn dn_alias_center_x() {
        let decoder = UnionFindDecoderDN::new(3);
        let mut code = SurfaceCode::new(3);
        code.apply_x_error_at(4);
        let (x_syn, z_syn) = code.measure_syndrome();
        let (x_corr, z_corr) = decoder.decode(&x_syn, &z_syn);
        assert_eq!(x_corr, vec![4]);
        assert!(z_corr.is_empty());
        code.correct(&x_corr, &z_corr);
        let (x_after, z_after) = code.measure_syndrome();
        assert!(x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0));
        assert!(!code.has_logical_x_error());
        assert!(!code.has_logical_z_error());
    }
}
