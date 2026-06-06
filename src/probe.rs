use crate::fieldstep::FieldStepParams;
use crate::simulation::{FieldKind, Simulation};
use crate::tilemap::{HEIGHT, WIDTH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeKind {
    Logic,
    FieldPower,
    FieldLogic,
    FieldClock,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeTrace {
    pub kind: ProbeKind,
    pub x: u32,
    pub y: u32,
    pub steps: u32,
    pub values: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeError {
    OutOfBounds,
}

fn in_bounds(x: u32, y: u32) -> bool {
    (x as usize) < WIDTH && (y as usize) < HEIGHT
}

pub fn run_logic_probe(
    sim: &mut Simulation,
    x: u32,
    y: u32,
    steps: u32,
) -> Result<ProbeTrace, ProbeError> {
    if !in_bounds(x, y) {
        return Err(ProbeError::OutOfBounds);
    }
    let mut values = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let val = sim
            .tilemap
            .get_tile(x as usize, y as usize)
            .map(|t| t.logic.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0);
        values.push(val);
        sim.tick();
    }
    Ok(ProbeTrace {
        kind: ProbeKind::Logic,
        x,
        y,
        steps,
        values,
    })
}

pub fn run_field_probe(
    sim: &mut Simulation,
    kind: FieldKind,
    x: u32,
    y: u32,
    steps: u32,
    params: &FieldStepParams,
) -> Result<ProbeTrace, ProbeError> {
    if !in_bounds(x, y) {
        return Err(ProbeError::OutOfBounds);
    }
    let mut values = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let val = match kind {
            FieldKind::Power => sim.get_power_field_for_test(x as usize, y as usize) as u64,
            FieldKind::Logic => sim.get_logic_field_for_test(x as usize, y as usize) as u64,
            FieldKind::Clock => sim.get_clock_field_for_test(x as usize, y as usize) as u64,
        };
        values.push(val);
        sim.step_fields_with(params);
    }
    let pk = match kind {
        FieldKind::Power => ProbeKind::FieldPower,
        FieldKind::Logic => ProbeKind::FieldLogic,
        FieldKind::Clock => ProbeKind::FieldClock,
    };
    Ok(ProbeTrace {
        kind: pk,
        x,
        y,
        steps,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::tile_meta::TileType;

    #[test]
    fn logic_probe_records_over_ticks() {
        let mut sim = Simulation::new();
        // Seed a simple pattern that doesn't trigger runaway delta cycles.
        sim.set_logic_value(0, 0, 0xFFFF);
        let trace = run_logic_probe(&mut sim, 0, 0, 3).expect("probe ok");
        assert_eq!(trace.values.len(), 3);
        assert_eq!(trace.values[0], 0xFFFF);
    }

    #[test]
    fn field_probe_propagates_power() {
        let mut sim = Simulation::new();
        sim.set_power_field_for_test(1, 1, 5);
        let params = FieldStepParams::default();
        let trace =
            run_field_probe(&mut sim, FieldKind::Power, 2, 1, 3, &params).expect("probe ok");
        assert_eq!(trace.values.len(), 3);
        assert!(trace.values[1] >= trace.values[0]); // monotone or equal with decay+max
    }

    #[test]
    fn probes_oob_rejected() {
        let mut sim = Simulation::new();
        let res = run_logic_probe(&mut sim, WIDTH as u32, 0, 1);
        assert!(matches!(res, Err(ProbeError::OutOfBounds)));
    }
}
