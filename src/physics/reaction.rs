use crate::field::FieldGrid;

#[derive(Debug, Clone)]
pub struct ReactionParams {
    pub heat_consume: u32,
    pub charge_consume: u32,
    pub heat_yield: u32,
    pub charge_yield: u32,
    pub min_heat: u32,
    pub min_charge: u32,
    pub max_heat: u32,
    pub max_charge: u32,
}

impl Default for ReactionParams {
    fn default() -> Self {
        Self {
            heat_consume: 1,
            charge_consume: 1,
            heat_yield: 0,
            charge_yield: 0,
            min_heat: 1,
            min_charge: 1,
            max_heat: 255,
            max_charge: 255,
        }
    }
}

pub(crate) fn reaction_step(
    heat_src: &FieldGrid<u32>,
    charge_src: &FieldGrid<u32>,
    heat_dst: &mut FieldGrid<u32>,
    charge_dst: &mut FieldGrid<u32>,
    params: &ReactionParams,
) {
    let w = heat_src.width();
    let h = heat_src.height();
    debug_assert_eq!(w, charge_src.width());
    debug_assert_eq!(h, charge_src.height());
    debug_assert_eq!(w, heat_dst.width());
    debug_assert_eq!(h, heat_dst.height());
    debug_assert_eq!(w, charge_dst.width());
    debug_assert_eq!(h, charge_dst.height());

    for y in 0..h {
        for x in 0..w {
            let hval = heat_src.get(x, y);
            let cval = charge_src.get(x, y);
            if hval >= params.min_heat && cval >= params.min_charge {
                let hnew = hval
                    .saturating_sub(params.heat_consume)
                    .saturating_add(params.heat_yield)
                    .min(params.max_heat);
                let cnew = cval
                    .saturating_sub(params.charge_consume)
                    .saturating_add(params.charge_yield)
                    .min(params.max_charge);
                heat_dst.set(x, y, hnew);
                charge_dst.set(x, y, cnew);
            } else {
                heat_dst.set(x, y, hval);
                charge_dst.set(x, y, cval);
            }
        }
    }
}

// EPIC 40: perf-bench branchless variant (mask-based select)
#[cfg(feature = "perf-bench")]
pub(crate) fn reaction_step_masked(
    heat_src: &FieldGrid<u32>,
    charge_src: &FieldGrid<u32>,
    heat_dst: &mut FieldGrid<u32>,
    charge_dst: &mut FieldGrid<u32>,
    params: &ReactionParams,
) {
    let w = heat_src.width();
    let h = heat_src.height();
    debug_assert_eq!(w, charge_src.width());
    debug_assert_eq!(h, charge_src.height());
    debug_assert_eq!(w, heat_dst.width());
    debug_assert_eq!(h, heat_dst.height());
    debug_assert_eq!(w, charge_dst.width());
    debug_assert_eq!(h, charge_dst.height());

    for y in 0..h {
        for x in 0..w {
            let hval = heat_src.get(x, y);
            let cval = charge_src.get(x, y);
            let cond = (hval >= params.min_heat) & (cval >= params.min_charge);
            let mask = if cond { u32::MAX } else { 0 };
            let cand_h = hval
                .saturating_sub(params.heat_consume)
                .saturating_add(params.heat_yield)
                .min(params.max_heat);
            let cand_c = cval
                .saturating_sub(params.charge_consume)
                .saturating_add(params.charge_yield)
                .min(params.max_charge);
            let out_h = (cand_h & mask) | (hval & !mask);
            let out_c = (cand_c & mask) | (cval & !mask);
            heat_dst.set(x, y, out_h);
            charge_dst.set(x, y, out_c);
        }
    }
}

#[cfg(all(test, feature = "perf-bench"))]
mod parity_tests {
    use super::*;
    #[test]
    fn reaction_masked_equals_default() {
        let mut heat_src = FieldGrid::new(3, 3, 0u32);
        let mut charge_src = FieldGrid::new(3, 3, 0u32);
        heat_src.set(0, 0, 10);
        charge_src.set(0, 0, 10);
        heat_src.set(1, 1, 2);
        charge_src.set(1, 1, 1);
        heat_src.set(2, 2, 1);
        charge_src.set(2, 2, 5);
        let mut h1 = FieldGrid::new(3, 3, 0u32);
        let mut c1 = FieldGrid::new(3, 3, 0u32);
        let mut h2 = FieldGrid::new(3, 3, 0u32);
        let mut c2 = FieldGrid::new(3, 3, 0u32);
        let params = ReactionParams::default();
        reaction_step(&heat_src, &charge_src, &mut h1, &mut c1, &params);
        reaction_step_masked(&heat_src, &charge_src, &mut h2, &mut c2, &params);
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(h1.get(x, y), h2.get(x, y));
                assert_eq!(c1.get(x, y), c2.get(x, y));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_below_threshold() {
        let heat_src = FieldGrid::new(2, 2, 0u32);
        let charge_src = FieldGrid::new(2, 2, 0u32);
        let mut heat_dst = FieldGrid::new(2, 2, 0u32);
        let mut charge_dst = FieldGrid::new(2, 2, 0u32);
        let params = ReactionParams::default();
        reaction_step(
            &heat_src,
            &charge_src,
            &mut heat_dst,
            &mut charge_dst,
            &params,
        );
        assert_eq!(heat_dst.get(0, 0), 0);
        assert_eq!(charge_dst.get(0, 0), 0);
    }

    #[test]
    fn single_event_reaction() {
        let heat_src = FieldGrid::new(1, 1, 5u32);
        let charge_src = FieldGrid::new(1, 1, 5u32);
        let mut heat_dst = FieldGrid::new(1, 1, 0u32);
        let mut charge_dst = FieldGrid::new(1, 1, 0u32);
        let params = ReactionParams {
            heat_consume: 2,
            charge_consume: 3,
            heat_yield: 1,
            charge_yield: 2,
            min_heat: 1,
            min_charge: 1,
            max_heat: 10,
            max_charge: 10,
        };
        reaction_step(
            &heat_src,
            &charge_src,
            &mut heat_dst,
            &mut charge_dst,
            &params,
        );
        assert_eq!(heat_dst.get(0, 0), 4); // 5 - 2 + 1
        assert_eq!(charge_dst.get(0, 0), 4); // 5 - 3 + 2
    }

    #[test]
    fn clamp_respected() {
        let heat_src = FieldGrid::new(1, 1, 250u32);
        let charge_src = FieldGrid::new(1, 1, 250u32);
        let mut heat_dst = FieldGrid::new(1, 1, 0u32);
        let mut charge_dst = FieldGrid::new(1, 1, 0u32);
        let params = ReactionParams {
            heat_consume: 0,
            charge_consume: 0,
            heat_yield: 10,
            charge_yield: 10,
            min_heat: 1,
            min_charge: 1,
            max_heat: 255,
            max_charge: 255,
        };
        reaction_step(
            &heat_src,
            &charge_src,
            &mut heat_dst,
            &mut charge_dst,
            &params,
        );
        assert_eq!(heat_dst.get(0, 0), 255);
        assert_eq!(charge_dst.get(0, 0), 255);
    }
}
