use crate::field::FieldGrid;

#[derive(Debug, Clone)]
pub struct DiffuseParams {
    pub spread_numerator: u32,
    pub spread_denominator: u32,
    pub decay_per_step: u32,
    pub max_value: u32,
}

impl Default for DiffuseParams {
    fn default() -> Self {
        Self {
            spread_numerator: 1,
            spread_denominator: 4,
            decay_per_step: 0,
            max_value: 255,
        }
    }
}

pub(crate) fn diffuse_step(src: &FieldGrid<u32>, dst: &mut FieldGrid<u32>, params: &DiffuseParams) {
    let w = src.width();
    let h = src.height();
    debug_assert_eq!(w, dst.width());
    debug_assert_eq!(h, dst.height());

    let spread_num = params
        .spread_numerator
        .min(params.spread_denominator.max(1));
    let spread_den = params.spread_denominator.max(1);

    for y in 0..h {
        for x in 0..w {
            let center = src.get(x, y);
            let mut sum_nb = 0u32;
            let mut count = 0u32;
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
                    if xu < w && yu < h {
                        sum_nb = sum_nb.saturating_add(src.get(xu, yu));
                        count += 1;
                    }
                }
            }
            let avg_nb = if count > 0 { sum_nb / count } else { 0 };
            let delta = avg_nb as i64 - center as i64;
            let adjusted = center as i64 + delta * spread_num as i64 / spread_den as i64;
            let decayed = adjusted - params.decay_per_step as i64;
            let clamped = decayed.clamp(0, params.max_value as i64) as u32;
            dst.set(x, y, clamped);
        }
    }
}

// EPIC 40: perf path with interior/border split to remove bounds checks in interior
#[cfg(feature = "perf-bench")]
pub(crate) fn diffuse_step_interior_split(
    src: &FieldGrid<u32>,
    dst: &mut FieldGrid<u32>,
    params: &DiffuseParams,
) {
    let w = src.width();
    let h = src.height();
    debug_assert_eq!(w, dst.width());
    debug_assert_eq!(h, dst.height());

    if w == 0 || h == 0 {
        return;
    }

    let spread_num = params
        .spread_numerator
        .min(params.spread_denominator.max(1));
    let spread_den = params.spread_denominator.max(1);

    // Top and bottom rows (borders)
    if h >= 1 {
        for &y in &[0usize, h.saturating_sub(1)] {
            if y >= h {
                continue;
            }
            for x in 0..w {
                let mut sum_nb = 0u32;
                let mut count = 0u32;
                if x > 0 {
                    sum_nb = sum_nb.saturating_add(src.get(x - 1, y));
                    count += 1;
                }
                if x + 1 < w {
                    sum_nb = sum_nb.saturating_add(src.get(x + 1, y));
                    count += 1;
                }
                if y > 0 {
                    sum_nb = sum_nb.saturating_add(src.get(x, y - 1));
                    count += 1;
                }
                if y + 1 < h {
                    sum_nb = sum_nb.saturating_add(src.get(x, y + 1));
                    count += 1;
                }
                let center = src.get(x, y);
                let avg_nb = if count > 0 { sum_nb / count } else { 0 };
                let delta = avg_nb as i64 - center as i64;
                let adjusted = center as i64 + delta * spread_num as i64 / spread_den as i64;
                let decayed = adjusted - params.decay_per_step as i64;
                let clamped = decayed.clamp(0, params.max_value as i64) as u32;
                dst.set(x, y, clamped);
            }
        }
    }

    if w < 2 || h < 2 {
        return;
    }

    // Left and right columns (excluding corners handled above)
    for y in 1..(h - 1) {
        // left column x=0
        {
            let x = 0;
            let center = src.get(x, y);
            let sum_nb = src
                .get(x + 1, y)
                .saturating_add(src.get(x, y - 1))
                .saturating_add(src.get(x, y + 1));
            let avg_nb = sum_nb / 3;
            let delta = avg_nb as i64 - center as i64;
            let adjusted = center as i64 + delta * spread_num as i64 / spread_den as i64;
            let decayed = adjusted - params.decay_per_step as i64;
            let clamped = decayed.clamp(0, params.max_value as i64) as u32;
            dst.set(x, y, clamped);
        }
        // right column x=w-1
        {
            let x = w - 1;
            let center = src.get(x, y);
            let sum_nb = src
                .get(x - 1, y)
                .saturating_add(src.get(x, y - 1))
                .saturating_add(src.get(x, y + 1));
            let avg_nb = sum_nb / 3;
            let delta = avg_nb as i64 - center as i64;
            let adjusted = center as i64 + delta * spread_num as i64 / spread_den as i64;
            let decayed = adjusted - params.decay_per_step as i64;
            let clamped = decayed.clamp(0, params.max_value as i64) as u32;
            dst.set(x, y, clamped);
        }
    }

    // Interior (1..w-1, 1..h-1) — branchless neighbor access
    if w >= 3 && h >= 3 {
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let center = src.get(x, y);
                let sum_nb = src
                    .get(x - 1, y)
                    .saturating_add(src.get(x + 1, y))
                    .saturating_add(src.get(x, y - 1))
                    .saturating_add(src.get(x, y + 1));
                let avg_nb = sum_nb / 4;
                let delta = avg_nb as i64 - center as i64;
                let adjusted = center as i64 + delta * spread_num as i64 / spread_den as i64;
                let decayed = adjusted - params.decay_per_step as i64;
                let clamped = decayed.clamp(0, params.max_value as i64) as u32;
                dst.set(x, y, clamped);
            }
        }
    }
}

#[cfg(all(test, feature = "perf-bench"))]
mod parity_tests {
    use super::*;
    #[test]
    fn diffuse_default_equals_interior_split_small() {
        let mut src = FieldGrid::new(5, 4, 0u32);
        // Seed deterministic pattern
        src.set(0, 0, 10);
        src.set(2, 1, 20);
        src.set(4, 3, 30);
        let mut dst1 = FieldGrid::new(5, 4, 0u32);
        let mut dst2 = FieldGrid::new(5, 4, 0u32);
        let params = DiffuseParams::default();
        diffuse_step(&src, &mut dst1, &params);
        diffuse_step_interior_split(&src, &mut dst2, &params);
        for y in 0..4 {
            for x in 0..5 {
                assert_eq!(dst1.get(x, y), dst2.get(x, y));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffuse_identity_when_spread_zero() {
        let src = FieldGrid::new(3, 3, 5u32);
        let mut dst = FieldGrid::new(3, 3, 0u32);
        let params = DiffuseParams {
            spread_numerator: 0,
            spread_denominator: 1,
            decay_per_step: 0,
            max_value: 10,
        };
        diffuse_step(&src, &mut dst, &params);
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(dst.get(x, y), 5);
            }
        }
    }

    #[test]
    fn diffuse_single_hot_cell_spreads() {
        let mut src = FieldGrid::new(3, 3, 0u32);
        src.set(1, 1, 10);
        let mut dst = FieldGrid::new(3, 3, 0u32);
        let params = DiffuseParams::default();
        diffuse_step(&src, &mut dst, &params);
        // Center should move toward neighbor average (0), so decrease; edges remain 0 with our edge-leak model.
        assert!(dst.get(1, 1) < 10);
        assert_eq!(dst.get(1, 0), 0);
        assert_eq!(dst.get(0, 1), 0);
    }

    #[test]
    fn diffuse_respects_decay_and_clamp() {
        let src = FieldGrid::new(1, 1, 100u32);
        let mut dst = FieldGrid::new(1, 1, 0u32);
        let params = DiffuseParams {
            spread_numerator: 1,
            spread_denominator: 1,
            decay_per_step: 50,
            max_value: 80,
        };
        diffuse_step(&src, &mut dst, &params);
        // With edge leak (no neighbors), value drops toward 0 then decays.
        assert_eq!(dst.get(0, 0), 0);
    }
}
