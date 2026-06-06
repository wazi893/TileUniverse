use crate::field::FieldGrid;

#[derive(Debug, Clone)]
pub struct CoupledParams {
    pub logic_inject_power: u32,
    pub max_power: u32,
    pub decay_per_step: u32,
    pub field_to_logic_threshold: u32,
    pub logic_mask: u64,
    pub clock_period: u64,
}

impl Default for CoupledParams {
    fn default() -> Self {
        Self {
            logic_inject_power: 1,
            max_power: 255,
            decay_per_step: 1,
            field_to_logic_threshold: 10,
            logic_mask: 0x1,
            clock_period: 4,
        }
    }
}

fn lane_active_bits(logic: u64, mask: u64) -> u32 {
    if (logic & mask) != 0 { 1 } else { 0 }
}

pub(crate) fn inject_logic_to_field(
    logic_src: &FieldGrid<u64>,
    power_src: &FieldGrid<u32>,
    power_dst: &mut FieldGrid<u32>,
    params: &CoupledParams,
) {
    let width = logic_src.width();
    let height = logic_src.height();
    debug_assert_eq!(width, power_src.width());
    debug_assert_eq!(height, power_src.height());
    debug_assert_eq!(width, power_dst.width());
    debug_assert_eq!(height, power_dst.height());

    for y in 0..height {
        for x in 0..width {
            let mut val = power_src.get(x, y).min(params.max_power);
            if lane_active_bits(logic_src.get(x, y), params.logic_mask) == 1 {
                val = val.saturating_add(params.logic_inject_power);
            }
            val = val.saturating_sub(params.decay_per_step);
            val = val.min(params.max_power);
            power_dst.set(x, y, val);
        }
    }
}

pub(crate) fn apply_field_to_logic(
    logic_src: &FieldGrid<u64>,
    power_src: &FieldGrid<u32>,
    logic_dst: &mut FieldGrid<u64>,
    params: &CoupledParams,
) {
    let width = logic_src.width();
    let height = logic_src.height();
    debug_assert_eq!(width, power_src.width());
    debug_assert_eq!(height, power_src.height());
    debug_assert_eq!(width, logic_dst.width());
    debug_assert_eq!(height, logic_dst.height());

    for y in 0..height {
        for x in 0..width {
            let base = logic_src.get(x, y);
            if power_src.get(x, y) >= params.field_to_logic_threshold {
                logic_dst.set(x, y, base | params.logic_mask);
            } else {
                logic_dst.set(x, y, base);
            }
        }
    }
}

pub(crate) fn coupled_step(
    logic_src: &FieldGrid<u64>,
    power_src: &FieldGrid<u32>,
    logic_dst: &mut FieldGrid<u64>,
    power_dst: &mut FieldGrid<u32>,
    params: &CoupledParams,
) {
    inject_logic_to_field(logic_src, power_src, power_dst, params);
    apply_field_to_logic(logic_src, power_dst, logic_dst, params);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_local_single_active() {
        let mut logic = FieldGrid::new(2, 2, 0u64);
        logic.set(0, 0, 0x1);
        let power = FieldGrid::new(2, 2, 0u32);
        let mut power_out = FieldGrid::new(2, 2, 0u32);
        let params = CoupledParams::default();
        inject_logic_to_field(&logic, &power, &mut power_out, &params);
        assert!(
            power_out.get(0, 0)
                >= params
                    .logic_inject_power
                    .saturating_sub(params.decay_per_step)
        );
        assert_eq!(power_out.get(1, 1), 0);
    }

    #[test]
    fn inject_respects_clamp() {
        let logic = FieldGrid::new(1, 1, 0x1);
        let power = FieldGrid::new(1, 1, 300u32);
        let mut power_out = FieldGrid::new(1, 1, 0u32);
        let params = CoupledParams {
            max_power: 10,
            ..Default::default()
        };
        inject_logic_to_field(&logic, &power, &mut power_out, &params);
        assert_eq!(power_out.get(0, 0), 10);
    }

    #[test]
    fn field_threshold_boosts_logic() {
        let logic = FieldGrid::new(1, 1, 0u64);
        let mut power = FieldGrid::new(1, 1, 0u32);
        power.set(0, 0, 20);
        let mut logic_out = FieldGrid::new(1, 1, 0u64);
        let params = CoupledParams {
            field_to_logic_threshold: 10,
            logic_mask: 0xF,
            ..Default::default()
        };
        apply_field_to_logic(&logic, &power, &mut logic_out, &params);
        assert_eq!(logic_out.get(0, 0), 0xF);
    }

    #[test]
    fn field_below_threshold_no_change() {
        let logic = FieldGrid::new(1, 1, 0xAAu64);
        let mut power = FieldGrid::new(1, 1, 0u32);
        power.set(0, 0, 5);
        let mut logic_out = FieldGrid::new(1, 1, 0u64);
        let params = CoupledParams {
            field_to_logic_threshold: 10,
            logic_mask: 0xF,
            ..Default::default()
        };
        apply_field_to_logic(&logic, &power, &mut logic_out, &params);
        assert_eq!(logic_out.get(0, 0), 0xAA);
    }

    #[test]
    fn coupled_step_deterministic() {
        let logic = FieldGrid::new(2, 2, 0u64);
        let power = FieldGrid::new(2, 2, 0u32);
        let mut logic_out1 = FieldGrid::new(2, 2, 0u64);
        let mut power_out1 = FieldGrid::new(2, 2, 0u32);
        let mut logic_out2 = FieldGrid::new(2, 2, 0u64);
        let mut power_out2 = FieldGrid::new(2, 2, 0u32);
        let params = CoupledParams::default();

        coupled_step(&logic, &power, &mut logic_out1, &mut power_out1, &params);
        coupled_step(&logic, &power, &mut logic_out2, &mut power_out2, &params);

        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(logic_out1.get(x, y), logic_out2.get(x, y));
                assert_eq!(power_out1.get(x, y), power_out2.get(x, y));
            }
        }
    }
}
