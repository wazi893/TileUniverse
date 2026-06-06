use crate::field::FieldGrid;

#[derive(Debug, Clone)]
pub struct HeatParams {
    pub inject_scale: u32,
    pub max_heat: u32,
    pub decay_per_step: u32,
}

impl Default for HeatParams {
    fn default() -> Self {
        Self {
            inject_scale: 1,
            max_heat: 255,
            decay_per_step: 1,
        }
    }
}

pub(crate) fn inject_heat_from_logic(
    logic: &FieldGrid<u64>,
    heat: &mut FieldGrid<u32>,
    params: &HeatParams,
) {
    let w = logic.width();
    let h = logic.height();
    debug_assert_eq!(w, heat.width());
    debug_assert_eq!(h, heat.height());

    for y in 0..h {
        for x in 0..w {
            let magnitude: u32 = if logic.get(x, y) != 0 { 1 } else { 0 };
            let delta = magnitude.saturating_mul(params.inject_scale);
            let val = heat.get(x, y).saturating_add(delta).min(params.max_heat);
            heat.set(x, y, val);
        }
    }
}

pub(crate) fn decay_heat(heat: &mut FieldGrid<u32>, params: &HeatParams) {
    let w = heat.width();
    let h = heat.height();
    for y in 0..h {
        for x in 0..w {
            let val = heat.get(x, y).saturating_sub(params.decay_per_step);
            heat.set(x, y, val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_heat_from_logic_empty() {
        let logic = FieldGrid::new(2, 2, 0u64);
        let mut heat = FieldGrid::new(2, 2, 0u32);
        inject_heat_from_logic(&logic, &mut heat, &HeatParams::default());
        assert_eq!(heat.get(0, 0), 0);
    }

    #[test]
    fn inject_heat_from_logic_simple() {
        let mut logic = FieldGrid::new(2, 2, 0u64);
        logic.set(0, 0, 1);
        let mut heat = FieldGrid::new(2, 2, 0u32);
        let params = HeatParams::default();
        inject_heat_from_logic(&logic, &mut heat, &params);
        assert!(heat.get(0, 0) >= params.inject_scale);
        assert_eq!(heat.get(1, 1), 0);
    }

    #[test]
    fn decay_heat_to_zero() {
        let logic = FieldGrid::new(1, 1, 1u64);
        let mut heat = FieldGrid::new(1, 1, 10u32);
        let params = HeatParams {
            inject_scale: 0,
            max_heat: 10,
            decay_per_step: 2,
        };
        inject_heat_from_logic(&logic, &mut heat, &params); // no change
        decay_heat(&mut heat, &params);
        decay_heat(&mut heat, &params);
        decay_heat(&mut heat, &params);
        assert_eq!(heat.get(0, 0), 4);
    }
}
