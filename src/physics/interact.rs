use crate::field::FieldGrid;

#[derive(Debug, Clone)]
pub struct InteractionParams {
    pub heat_affects_charge: u32,
    pub charge_affects_heat: u32,
    pub max_delta: u32,
    pub clamp: u32,
}

impl Default for InteractionParams {
    fn default() -> Self {
        Self {
            heat_affects_charge: 1,
            charge_affects_heat: 1,
            max_delta: 4,
            clamp: 255,
        }
    }
}

pub(crate) fn interact_heat_charge_step(
    heat_src: &FieldGrid<u32>,
    charge_src: &FieldGrid<u32>,
    heat_dst: &mut FieldGrid<u32>,
    charge_dst: &mut FieldGrid<u32>,
    params: &InteractionParams,
) {
    let w = heat_src.width();
    let h = heat_src.height();
    debug_assert_eq!(w, charge_src.width());
    debug_assert_eq!(h, charge_src.height());
    debug_assert_eq!(w, heat_dst.width());
    debug_assert_eq!(h, heat_dst.height());
    debug_assert_eq!(w, charge_dst.width());
    debug_assert_eq!(h, charge_dst.height());

    let clamp = params.clamp;
    for y in 0..h {
        for x in 0..w {
            let hval = heat_src.get(x, y);
            let cval = charge_src.get(x, y);
            let dh = hval
                .min(params.max_delta)
                .saturating_mul(params.heat_affects_charge);
            let dc = cval
                .min(params.max_delta)
                .saturating_mul(params.charge_affects_heat);
            let hnew = hval.saturating_add(dc).min(clamp);
            let cnew = cval.saturating_add(dh).min(clamp);
            heat_dst.set(x, y, hnew);
            charge_dst.set(x, y, cnew);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_params_zero() {
        let heat = FieldGrid::new(2, 2, 5u32);
        let charge = FieldGrid::new(2, 2, 7u32);
        let mut heat_out = FieldGrid::new(2, 2, 0u32);
        let mut charge_out = FieldGrid::new(2, 2, 0u32);
        let params = InteractionParams {
            heat_affects_charge: 0,
            charge_affects_heat: 0,
            max_delta: 0,
            clamp: 255,
        };
        interact_heat_charge_step(&heat, &charge, &mut heat_out, &mut charge_out, &params);
        assert_eq!(heat_out.get(0, 0), 5);
        assert_eq!(charge_out.get(0, 0), 7);
    }

    #[test]
    fn heat_increases_charge_and_vice_versa() {
        let heat = FieldGrid::new(1, 1, 10u32);
        let charge = FieldGrid::new(1, 1, 5u32);
        let mut heat_out = FieldGrid::new(1, 1, 0u32);
        let mut charge_out = FieldGrid::new(1, 1, 0u32);
        let params = InteractionParams {
            heat_affects_charge: 2,
            charge_affects_heat: 1,
            max_delta: 10,
            clamp: 100,
        };
        interact_heat_charge_step(&heat, &charge, &mut heat_out, &mut charge_out, &params);
        assert_eq!(charge_out.get(0, 0), 25); // heat adds to charge: min(10,10)*2 = 20; 5+20=25
        assert_eq!(heat_out.get(0, 0), 15); // charge adds to heat: min(5,10)*1 = 5; 10+5=15
    }

    #[test]
    fn clamp_respected() {
        let heat = FieldGrid::new(1, 1, 250u32);
        let charge = FieldGrid::new(1, 1, 250u32);
        let mut heat_out = FieldGrid::new(1, 1, 0u32);
        let mut charge_out = FieldGrid::new(1, 1, 0u32);
        let params = InteractionParams {
            heat_affects_charge: 10,
            charge_affects_heat: 10,
            max_delta: 10,
            clamp: 255,
        };
        interact_heat_charge_step(&heat, &charge, &mut heat_out, &mut charge_out, &params);
        assert_eq!(heat_out.get(0, 0), 255);
        assert_eq!(charge_out.get(0, 0), 255);
    }
}
