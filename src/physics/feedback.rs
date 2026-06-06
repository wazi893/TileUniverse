use crate::field::FieldGrid;

#[derive(Debug, Clone)]
pub struct FeedbackParams {
    pub charge_to_logic_threshold: u32,
    pub heat_to_logic_threshold: u32,
    pub logic_mask: u64,
    pub clamp_value: u64,
}

impl Default for FeedbackParams {
    fn default() -> Self {
        Self {
            charge_to_logic_threshold: 1,
            heat_to_logic_threshold: 1,
            logic_mask: 1,
            clamp_value: u64::MAX,
        }
    }
}

pub(crate) fn feedback_from_charge(
    charge_src: &FieldGrid<u32>,
    logic_src: &FieldGrid<u64>,
    logic_dst: &mut FieldGrid<u64>,
    params: &FeedbackParams,
) {
    let w = charge_src.width();
    let h = charge_src.height();
    debug_assert_eq!(w, logic_src.width());
    debug_assert_eq!(h, logic_src.height());
    debug_assert_eq!(w, logic_dst.width());
    debug_assert_eq!(h, logic_dst.height());

    let threshold = params.charge_to_logic_threshold;
    let mask = params.logic_mask;
    let clamp = params.clamp_value;

    for y in 0..h {
        for x in 0..w {
            let base = logic_src.get(x, y);
            if charge_src.get(x, y) >= threshold {
                let val = base | mask;
                let clamped = if val > clamp { clamp } else { val };
                logic_dst.set(x, y, clamped);
            } else {
                logic_dst.set(x, y, base);
            }
        }
    }
}

pub(crate) fn feedback_from_heat(
    heat_src: &FieldGrid<u32>,
    logic_src: &FieldGrid<u64>,
    logic_dst: &mut FieldGrid<u64>,
    params: &FeedbackParams,
) {
    let w = heat_src.width();
    let h = heat_src.height();
    debug_assert_eq!(w, logic_src.width());
    debug_assert_eq!(h, logic_src.height());
    debug_assert_eq!(w, logic_dst.width());
    debug_assert_eq!(h, logic_dst.height());

    let threshold = params.heat_to_logic_threshold;
    let mask = params.logic_mask;
    let clamp = params.clamp_value;

    for y in 0..h {
        for x in 0..w {
            let base = logic_src.get(x, y);
            if heat_src.get(x, y) >= threshold {
                let val = base | mask;
                let clamped = if val > clamp { clamp } else { val };
                logic_dst.set(x, y, clamped);
            } else {
                logic_dst.set(x, y, base);
            }
        }
    }
}

// EPIC 40: perf-bench masked variants to reduce branching
#[cfg(feature = "perf-bench")]
#[allow(dead_code)] // May be unused depending on bench matrix; kept for parity and future modes
pub(crate) fn feedback_from_charge_masked(
    charge_src: &FieldGrid<u32>,
    logic_src: &FieldGrid<u64>,
    logic_dst: &mut FieldGrid<u64>,
    params: &FeedbackParams,
) {
    let w = charge_src.width();
    let h = charge_src.height();
    debug_assert_eq!(w, logic_src.width());
    debug_assert_eq!(h, logic_src.height());
    debug_assert_eq!(w, logic_dst.width());
    debug_assert_eq!(h, logic_dst.height());
    let threshold = params.charge_to_logic_threshold;
    let maskbits = params.logic_mask;
    let clamp = params.clamp_value;
    for y in 0..h {
        for x in 0..w {
            let base = logic_src.get(x, y);
            let set = ((charge_src.get(x, y) >= threshold) as u64).wrapping_neg();
            let selected = (base & !set) | ((base | maskbits) & set);
            let clamped = if selected > clamp { clamp } else { selected };
            logic_dst.set(x, y, clamped);
        }
    }
}

#[cfg(feature = "perf-bench")]
pub(crate) fn feedback_from_heat_masked(
    heat_src: &FieldGrid<u32>,
    logic_src: &FieldGrid<u64>,
    logic_dst: &mut FieldGrid<u64>,
    params: &FeedbackParams,
) {
    let w = heat_src.width();
    let h = heat_src.height();
    debug_assert_eq!(w, logic_src.width());
    debug_assert_eq!(h, logic_src.height());
    debug_assert_eq!(w, logic_dst.width());
    debug_assert_eq!(h, logic_dst.height());
    let threshold = params.heat_to_logic_threshold;
    let maskbits = params.logic_mask;
    let clamp = params.clamp_value;
    for y in 0..h {
        for x in 0..w {
            let base = logic_src.get(x, y);
            let set = ((heat_src.get(x, y) >= threshold) as u64).wrapping_neg();
            let selected = (base & !set) | ((base | maskbits) & set);
            let clamped = if selected > clamp { clamp } else { selected };
            logic_dst.set(x, y, clamped);
        }
    }
}

#[cfg(all(test, feature = "perf-bench"))]
mod parity_tests {
    use super::*;
    #[test]
    fn feedback_masked_equals_default() {
        let mut charge = FieldGrid::new(2, 2, 0u32);
        let mut heat = FieldGrid::new(2, 2, 0u32);
        charge.set(0, 0, 5);
        heat.set(1, 1, 7);
        let logic = FieldGrid::new(2, 2, 3u64);
        let mut out1 = FieldGrid::new(2, 2, 0u64);
        let mut out2 = FieldGrid::new(2, 2, 0u64);
        let mut out3 = FieldGrid::new(2, 2, 0u64);
        let mut out4 = FieldGrid::new(2, 2, 0u64);
        let params = FeedbackParams {
            charge_to_logic_threshold: 4,
            heat_to_logic_threshold: 6,
            logic_mask: 0xF,
            clamp_value: u64::MAX,
        };
        feedback_from_charge(&charge, &logic, &mut out1, &params);
        feedback_from_charge_masked(&charge, &logic, &mut out2, &params);
        feedback_from_heat(&heat, &logic, &mut out3, &params);
        feedback_from_heat_masked(&heat, &logic, &mut out4, &params);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out1.get(x, y), out2.get(x, y));
                assert_eq!(out3.get(x, y), out4.get(x, y));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_under_zero_fields() {
        let charge = FieldGrid::new(2, 2, 0u32);
        let heat = FieldGrid::new(2, 2, 0u32);
        let logic = FieldGrid::new(2, 2, 3u64);
        let mut out = FieldGrid::new(2, 2, 0u64);
        let params = FeedbackParams::default();
        feedback_from_charge(&charge, &logic, &mut out, &params);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.get(x, y), 3);
            }
        }
        feedback_from_heat(&heat, &logic, &mut out, &params);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.get(x, y), 3);
            }
        }
    }

    #[test]
    fn threshold_blocks_activation() {
        let mut charge = FieldGrid::new(1, 1, 0u32);
        charge.set(0, 0, 5);
        let logic = FieldGrid::new(1, 1, 0u64);
        let mut out = FieldGrid::new(1, 1, 0u64);
        let params = FeedbackParams {
            charge_to_logic_threshold: 10,
            ..Default::default()
        };
        feedback_from_charge(&charge, &logic, &mut out, &params);
        assert_eq!(out.get(0, 0), 0);
    }

    #[test]
    fn activation_sets_bits() {
        let mut heat = FieldGrid::new(1, 1, 0u32);
        heat.set(0, 0, 2);
        let logic = FieldGrid::new(1, 1, 0u64);
        let mut out = FieldGrid::new(1, 1, 0u64);
        let params = FeedbackParams {
            heat_to_logic_threshold: 1,
            logic_mask: 0xF,
            ..Default::default()
        };
        feedback_from_heat(&heat, &logic, &mut out, &params);
        assert_eq!(out.get(0, 0), 0xF);
    }

    #[test]
    fn clamp_respected() {
        let heat = FieldGrid::new(1, 1, 10u32);
        let logic = FieldGrid::new(1, 1, 0xFFFF_FFF0u64);
        let mut out = FieldGrid::new(1, 1, 0u64);
        let params = FeedbackParams {
            heat_to_logic_threshold: 1,
            logic_mask: 0xFFFF,
            clamp_value: 0xFFFF_FFF1,
            charge_to_logic_threshold: 1,
        };
        feedback_from_heat(&heat, &logic, &mut out, &params);
        assert_eq!(out.get(0, 0), 0xFFFF_FFF1);
    }
}
