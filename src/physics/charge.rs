use crate::field::FieldGrid;

#[derive(Debug, Clone)]
pub struct ChargeParams {
    pub inject_scale: u32,
    pub max_charge: u32,
    pub decay_per_step: u32,
}

impl Default for ChargeParams {
    fn default() -> Self {
        Self {
            inject_scale: 1,
            max_charge: 255,
            decay_per_step: 1,
        }
    }
}

pub(crate) fn inject_charge_from_logic(
    logic: &FieldGrid<u64>,
    charge: &mut FieldGrid<u32>,
    params: &ChargeParams,
) {
    let w = logic.width();
    let h = logic.height();
    debug_assert_eq!(w, charge.width());
    debug_assert_eq!(h, charge.height());

    for y in 0..h {
        for x in 0..w {
            let magnitude: u32 = if logic.get(x, y) != 0 { 1 } else { 0 };
            let delta = magnitude.saturating_mul(params.inject_scale);
            let val = charge
                .get(x, y)
                .saturating_add(delta)
                .min(params.max_charge);
            charge.set(x, y, val);
        }
    }
}

pub(crate) fn decay_charge(charge: &mut FieldGrid<u32>, params: &ChargeParams) {
    let w = charge.width();
    let h = charge.height();
    for y in 0..h {
        for x in 0..w {
            let val = charge.get(x, y).saturating_sub(params.decay_per_step);
            charge.set(x, y, val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_charge_from_logic_empty() {
        let logic = FieldGrid::new(2, 2, 0u64);
        let mut charge = FieldGrid::new(2, 2, 0u32);
        inject_charge_from_logic(&logic, &mut charge, &ChargeParams::default());
        assert_eq!(charge.get(0, 0), 0);
    }

    #[test]
    fn inject_charge_from_logic_simple() {
        let mut logic = FieldGrid::new(2, 2, 0u64);
        logic.set(0, 0, 1);
        let mut charge = FieldGrid::new(2, 2, 0u32);
        let params = ChargeParams::default();
        inject_charge_from_logic(&logic, &mut charge, &params);
        assert!(charge.get(0, 0) >= params.inject_scale);
        assert_eq!(charge.get(1, 1), 0);
    }

    #[test]
    fn decay_charge_to_zero() {
        let logic = FieldGrid::new(1, 1, 1u64);
        let mut charge = FieldGrid::new(1, 1, 10u32);
        let params = ChargeParams {
            inject_scale: 0,
            max_charge: 10,
            decay_per_step: 2,
        };
        inject_charge_from_logic(&logic, &mut charge, &params); // no change
        decay_charge(&mut charge, &params);
        decay_charge(&mut charge, &params);
        decay_charge(&mut charge, &params);
        assert_eq!(charge.get(0, 0), 4);
    }
}
