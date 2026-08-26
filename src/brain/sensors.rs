//! EPIC 96.5: Rich Sensory Input Extraction
//!
//! Extracts multi-field, multi-directional sensor data for configurable NNs.
//! Supports 8, 16, 32, and 64 sensor architectures.

use crate::configurable_nn::NNConfig;

/// Extract sensors for a given NN configuration
///
/// Configurations:
/// - 8 sensors: 4 heat + 4 charge (current baseline)
/// - 16 sensors: 8 heat + 8 charge (8 directions)
/// - 32 sensors: 8 heat + 8 charge + 8 food + 8 density
/// - 64 sensors: 32 × 2 (immediate + 2-step lookahead)
pub fn extract_sensors(
    heat_map: &[f32],
    charge_map: &[f32],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    config: NNConfig,
) -> Vec<f32> {
    match config.n_sensors {
        8 => extract_sensors_8(heat_map, charge_map, x, y, width, height),
        16 => extract_sensors_16(heat_map, charge_map, x, y, width, height),
        32 => extract_sensors_32(heat_map, charge_map, x, y, width, height),
        64 => extract_sensors_64(heat_map, charge_map, x, y, width, height),
        _ => panic!("Unsupported sensor count: {}", config.n_sensors),
    }
}

/// 8 sensors: 4 heat + 4 charge (N, E, S, W)
/// Current baseline architecture
fn extract_sensors_8(
    heat_map: &[f32],
    charge_map: &[f32],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut sensors = Vec::with_capacity(8);

    // 4 cardinal directions
    let dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]; // N, E, S, W

    // Heat in 4 directions
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        sensors.push(heat_map[ny * width + nx]);
    }

    // Charge in 4 directions
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        sensors.push(charge_map[ny * width + nx]);
    }

    sensors
}

/// 16 sensors: 8 heat + 8 charge (8 directions including diagonals)
fn extract_sensors_16(
    heat_map: &[f32],
    charge_map: &[f32],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut sensors = Vec::with_capacity(16);

    // 8 directions: N, NE, E, SE, S, SW, W, NW
    let dirs = [
        (0, -1),  // N
        (1, -1),  // NE
        (1, 0),   // E
        (1, 1),   // SE
        (0, 1),   // S
        (-1, 1),  // SW
        (-1, 0),  // W
        (-1, -1), // NW
    ];

    // Heat in 8 directions
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        sensors.push(heat_map[ny * width + nx]);
    }

    // Charge in 8 directions
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        sensors.push(charge_map[ny * width + nx]);
    }

    sensors
}

/// 32 sensors: 8 heat + 8 charge + 8 food + 8 density
///
/// "Food" = high heat regions (organisms seek warmth)
/// "Density" = combined heat+charge intensity (crowding detection)
fn extract_sensors_32(
    heat_map: &[f32],
    charge_map: &[f32],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut sensors = Vec::with_capacity(32);

    // 8 directions
    let dirs = [
        (0, -1),  // N
        (1, -1),  // NE
        (1, 0),   // E
        (1, 1),   // SE
        (0, 1),   // S
        (-1, 1),  // SW
        (-1, 0),  // W
        (-1, -1), // NW
    ];

    // Heat in 8 directions
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        sensors.push(heat_map[ny * width + nx]);
    }

    // Charge in 8 directions
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        sensors.push(charge_map[ny * width + nx]);
    }

    // Food (high heat = desirable) in 8 directions
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        let heat = heat_map[ny * width + nx];
        // Food = positive heat only (ignore cold regions)
        sensors.push(heat.max(0.0));
    }

    // Density (heat + charge magnitude) in 8 directions
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        let heat = heat_map[ny * width + nx];
        let charge = charge_map[ny * width + nx];
        // Density = sum of absolute values (crowding indicator)
        sensors.push(heat.abs() + charge.abs());
    }

    sensors
}

/// 64 sensors: 32 immediate + 32 lookahead (2-step)
///
/// Provides temporal prediction capability:
/// - First 32: Immediate neighborhood (8-dir heat/charge/food/density)
/// - Next 32: 2-step lookahead in same directions
fn extract_sensors_64(
    heat_map: &[f32],
    charge_map: &[f32],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut sensors = Vec::with_capacity(64);

    // 8 directions
    let dirs = [
        (0, -1),  // N
        (1, -1),  // NE
        (1, 0),   // E
        (1, 1),   // SE
        (0, 1),   // S
        (-1, 1),  // SW
        (-1, 0),  // W
        (-1, -1), // NW
    ];

    // Immediate (1-step) sensors (32 total)
    // Heat
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        sensors.push(heat_map[ny * width + nx]);
    }

    // Charge
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        sensors.push(charge_map[ny * width + nx]);
    }

    // Food
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        let heat = heat_map[ny * width + nx];
        sensors.push(heat.max(0.0));
    }

    // Density
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
        let heat = heat_map[ny * width + nx];
        let charge = charge_map[ny * width + nx];
        sensors.push(heat.abs() + charge.abs());
    }

    // Lookahead (2-step) sensors (32 total)
    // Heat (2 steps away)
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx * 2).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy * 2).rem_euclid(height as i32) as usize;
        sensors.push(heat_map[ny * width + nx]);
    }

    // Charge (2 steps away)
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx * 2).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy * 2).rem_euclid(height as i32) as usize;
        sensors.push(charge_map[ny * width + nx]);
    }

    // Food (2 steps away)
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx * 2).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy * 2).rem_euclid(height as i32) as usize;
        let heat = heat_map[ny * width + nx];
        sensors.push(heat.max(0.0));
    }

    // Density (2 steps away)
    for &(dx, dy) in &dirs {
        let nx = (x as i32 + dx * 2).rem_euclid(width as i32) as usize;
        let ny = (y as i32 + dy * 2).rem_euclid(height as i32) as usize;
        let heat = heat_map[ny * width + nx];
        let charge = charge_map[ny * width + nx];
        sensors.push(heat.abs() + charge.abs());
    }

    sensors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_world() -> (Vec<f32>, Vec<f32>) {
        // 5×5 world with simple gradient
        let heat: Vec<f32> = (0..25).map(|i| (i as f32 / 25.0) * 2.0 - 1.0).collect();
        let charge: Vec<f32> = (0..25)
            .map(|i| ((24 - i) as f32 / 25.0) * 2.0 - 1.0)
            .collect();
        (heat, charge)
    }

    #[test]
    fn test_extract_8_sensors() {
        let (heat, charge) = make_test_world();
        let sensors = extract_sensors_8(&heat, &charge, 2, 2, 5, 5);

        assert_eq!(sensors.len(), 8);
        // Should have 4 heat values + 4 charge values
        // All values should be in [-1, 1] range
        for &s in &sensors {
            assert!((-1.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn test_extract_16_sensors() {
        let (heat, charge) = make_test_world();
        let sensors = extract_sensors_16(&heat, &charge, 2, 2, 5, 5);

        assert_eq!(sensors.len(), 16);
        for &s in &sensors {
            assert!((-1.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn test_extract_32_sensors() {
        let (heat, charge) = make_test_world();
        let sensors = extract_sensors_32(&heat, &charge, 2, 2, 5, 5);

        assert_eq!(sensors.len(), 32);
        // First 16: heat + charge (can be negative)
        // Next 8: food (non-negative)
        for i in 16..24 {
            assert!(sensors[i] >= 0.0, "Food sensors should be non-negative");
        }
        // Last 8: density (non-negative)
        for i in 24..32 {
            assert!(sensors[i] >= 0.0, "Density sensors should be non-negative");
        }
    }

    #[test]
    fn test_extract_64_sensors() {
        let (heat, charge) = make_test_world();
        let sensors = extract_sensors_64(&heat, &charge, 2, 2, 5, 5);

        assert_eq!(sensors.len(), 64);
        // First 32: immediate
        // Next 32: lookahead
    }

    #[test]
    fn test_extract_via_config() {
        let (heat, charge) = make_test_world();

        let sensors_a = extract_sensors(&heat, &charge, 2, 2, 5, 5, NNConfig::CONFIG_A);
        assert_eq!(sensors_a.len(), 8);

        let sensors_b = extract_sensors(&heat, &charge, 2, 2, 5, 5, NNConfig::CONFIG_B);
        assert_eq!(sensors_b.len(), 16);

        let sensors_c = extract_sensors(&heat, &charge, 2, 2, 5, 5, NNConfig::CONFIG_C);
        assert_eq!(sensors_c.len(), 32);

        let sensors_e = extract_sensors(&heat, &charge, 2, 2, 5, 5, NNConfig::CONFIG_E);
        assert_eq!(sensors_e.len(), 64);
    }

    #[test]
    fn test_wrapping() {
        let (heat, charge) = make_test_world();

        // Test edge wrapping (0, 0) should access (4, 4) for SW direction
        let sensors = extract_sensors_8(&heat, &charge, 0, 0, 5, 5);
        assert_eq!(sensors.len(), 8);
    }
}
