use crate::field::FieldGrid;

#[derive(Debug, Clone)]
pub struct FieldStepParams {
    pub power_decay: u8,
    pub power_min: u8,
    pub clock_period: u32,
}

impl Default for FieldStepParams {
    fn default() -> Self {
        Self {
            power_decay: 1,
            power_min: 0,
            clock_period: 4,
        }
    }
}

pub(crate) fn step_power_field(
    src: &FieldGrid<u8>,
    dst: &mut FieldGrid<u8>,
    params: &FieldStepParams,
) {
    let width = src.width();
    let height = src.height();
    debug_assert_eq!(width, dst.width());
    debug_assert_eq!(height, dst.height());

    for y in 0..height {
        for x in 0..width {
            let mut neighbor_max = 0u8;
            let neighbors = [
                (x as isize - 1, y as isize),
                (x as isize + 1, y as isize),
                (x as isize, y as isize - 1),
                (x as isize, y as isize + 1),
            ];
            for (nx, ny) in neighbors {
                if nx >= 0 && ny >= 0 {
                    let xu = nx as usize;
                    let yu = ny as usize;
                    if xu < width && yu < height {
                        neighbor_max = neighbor_max.max(src.get(xu, yu));
                    }
                }
            }
            let candidate = neighbor_max
                .saturating_sub(params.power_decay)
                .max(params.power_min);
            let val = src.get(x, y).max(candidate);
            dst.set(x, y, val);
        }
    }
}

pub(crate) fn step_logic_field(src: &FieldGrid<u32>, dst: &mut FieldGrid<u32>) {
    let width = src.width();
    let height = src.height();
    debug_assert_eq!(width, dst.width());
    debug_assert_eq!(height, dst.height());

    for y in 0..height {
        for x in 0..width {
            let mut propagated = 0u32;
            let neighbors = [
                (x as isize - 1, y as isize),
                (x as isize + 1, y as isize),
                (x as isize, y as isize - 1),
                (x as isize, y as isize + 1),
            ];
            for (nx, ny) in neighbors {
                if nx >= 0 && ny >= 0 {
                    let xu = nx as usize;
                    let yu = ny as usize;
                    if xu < width && yu < height {
                        propagated |= src.get(xu, yu);
                    }
                }
            }
            let val = src.get(x, y) | propagated;
            dst.set(x, y, val);
        }
    }
}

pub(crate) fn step_clock_field(
    src: &FieldGrid<u8>,
    dst: &mut FieldGrid<u8>,
    global_tick: u64,
    params: &FieldStepParams,
) {
    let width = src.width();
    let height = src.height();
    debug_assert_eq!(width, dst.width());
    debug_assert_eq!(height, dst.height());
    let period = params.clock_period.max(1);

    for y in 0..height {
        for x in 0..width {
            let phase = ((global_tick as u32) + x as u32 + y as u32) % period;
            dst.set(x, y, phase as u8);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn step_all_fields(
    logic_src: &FieldGrid<u32>,
    logic_dst: &mut FieldGrid<u32>,
    power_src: &FieldGrid<u8>,
    power_dst: &mut FieldGrid<u8>,
    clock_src: &FieldGrid<u8>,
    clock_dst: &mut FieldGrid<u8>,
    global_tick: u64,
    params: &FieldStepParams,
) {
    step_logic_field(logic_src, logic_dst);
    step_power_field(power_src, power_dst, params);
    step_clock_field(clock_src, clock_dst, global_tick, params);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_single_source_spreads_and_decays() {
        let mut src = FieldGrid::new(5, 5, 0u8);
        src.set(2, 2, 5);
        let mut dst = FieldGrid::new(5, 5, 0u8);
        let params = FieldStepParams {
            power_decay: 1,
            power_min: 0,
            clock_period: 4,
        };

        step_power_field(&src, &mut dst, &params);
        // Center should remain 5, immediate neighbors should be 4
        assert_eq!(dst.get(2, 2), 5);
        assert_eq!(dst.get(2, 1), 4);
        assert_eq!(dst.get(2, 3), 4);
        assert_eq!(dst.get(1, 2), 4);
        assert_eq!(dst.get(3, 2), 4);
    }

    #[test]
    fn power_multiple_sources_superpose() {
        let mut src = FieldGrid::new(5, 5, 0u8);
        src.set(1, 1, 5);
        src.set(3, 3, 7);
        let mut dst = FieldGrid::new(5, 5, 0u8);
        let params = FieldStepParams::default();
        step_power_field(&src, &mut dst, &params);
        // Second step to allow propagation to (2,2) via 4-neighbor paths
        let mut next = FieldGrid::new(5, 5, 0u8);
        step_power_field(&dst, &mut next, &params);
        assert_eq!(next.get(2, 2), 5); // 7 decays twice along Manhattan path to center
    }

    #[test]
    fn logic_or_propagation_monotone() {
        let mut src = FieldGrid::new(3, 3, 0u32);
        src.set(1, 1, 0xF0);
        src.set(1, 2, 0x0F);
        let mut dst = FieldGrid::new(3, 3, 0u32);
        step_logic_field(&src, &mut dst);
        assert_eq!(dst.get(1, 1), 0xF0 | 0x0F);
        assert!(dst.get(0, 1) != 0);
    }

    #[test]
    fn logic_local_source_not_overwritten() {
        let mut src = FieldGrid::new(3, 3, 0u32);
        src.set(1, 1, 0xAA);
        src.set(1, 0, 0x55);
        let mut dst = FieldGrid::new(3, 3, 0u32);
        step_logic_field(&src, &mut dst);
        assert_eq!(dst.get(1, 1), 0xAA | 0x55);
    }

    #[test]
    fn clock_phase_deterministic() {
        let src = FieldGrid::new(2, 2, 0u8);
        let mut dst1 = FieldGrid::new(2, 2, 0u8);
        let mut dst2 = FieldGrid::new(2, 2, 0u8);
        let params = FieldStepParams {
            power_decay: 1,
            power_min: 0,
            clock_period: 4,
        };
        step_clock_field(&src, &mut dst1, 5, &params);
        step_clock_field(&src, &mut dst2, 5, &params);
        assert_eq!(dst1.get(0, 0), dst2.get(0, 0));
        assert_eq!(dst1.get(1, 1), dst2.get(1, 1));
    }

    #[test]
    fn step_all_fields_equivalent() {
        let mut logic_src = FieldGrid::new(2, 2, 0u32);
        logic_src.set(0, 0, 1);
        let mut power_src = FieldGrid::new(2, 2, 0u8);
        power_src.set(1, 1, 3);
        let clock_src = FieldGrid::new(2, 2, 0u8);

        let mut logic_dst1 = FieldGrid::new(2, 2, 0u32);
        let mut power_dst1 = FieldGrid::new(2, 2, 0u8);
        let mut clock_dst1 = FieldGrid::new(2, 2, 0u8);

        let mut logic_dst2 = FieldGrid::new(2, 2, 0u32);
        let mut power_dst2 = FieldGrid::new(2, 2, 0u8);
        let mut clock_dst2 = FieldGrid::new(2, 2, 0u8);

        let params = FieldStepParams::default();
        step_logic_field(&logic_src, &mut logic_dst1);
        step_power_field(&power_src, &mut power_dst1, &params);
        step_clock_field(&clock_src, &mut clock_dst1, 1, &params);

        step_all_fields(
            &logic_src,
            &mut logic_dst2,
            &power_src,
            &mut power_dst2,
            &clock_src,
            &mut clock_dst2,
            1,
            &params,
        );

        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(logic_dst1.get(x, y), logic_dst2.get(x, y));
                assert_eq!(power_dst1.get(x, y), power_dst2.get(x, y));
                assert_eq!(clock_dst1.get(x, y), clock_dst2.get(x, y));
            }
        }
    }
}
