use crate::charge::ChargeParams;
use crate::coupling::CoupledParams;
use crate::diffuse::DiffuseParams;
use crate::heat::HeatParams;
use crate::physics_interact::InteractionParams;
use crate::reaction::ReactionParams;
use crate::simulation::Simulation;
use crate::tilemap::{HEIGHT, WIDTH};

#[derive(Debug, Clone)]
pub struct Region {
    pub x0: u32,
    pub y0: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone)]
pub enum ScenarioKind {
    HeatPulse1D,
    HeatBox,
    ChargeWave,
    InteractOscillator,
    Custom,
}

#[derive(Debug, Clone)]
pub struct PhysicsScenario {
    pub kind: ScenarioKind,
    pub steps: u32,
    pub tick_stride: u32,
    pub region: Option<Region>,
    pub heat_params: Option<HeatParams>,
    pub charge_params: Option<ChargeParams>,
    pub diffuse_params: Option<DiffuseParams>,
    pub reaction_params: Option<ReactionParams>,
    pub interact_params: Option<InteractionParams>,
    pub coupling_params: Option<CoupledParams>,
}

#[derive(Debug, Clone)]
pub struct ScenarioSummary {
    pub steps: u32,
    pub region: Option<Region>,
    pub max_heat: u32,
    pub max_charge: u32,
    pub final_avg_heat: f32,
    pub final_avg_charge: f32,
}

fn region_bounds(region: &Option<Region>) -> (usize, usize, usize, usize) {
    if let Some(r) = region {
        (
            r.x0 as usize,
            r.y0 as usize,
            (r.x0 + r.w).min(WIDTH as u32) as usize,
            (r.y0 + r.h).min(HEIGHT as u32) as usize,
        )
    } else {
        (0, 0, WIDTH, HEIGHT)
    }
}

fn seed_scenario(sim: &mut Simulation, kind: &ScenarioKind) {
    match kind {
        ScenarioKind::HeatPulse1D => {
            let mid_y = HEIGHT / 2;
            for x in 0..WIDTH {
                sim.set_heat_field_for_test(x, mid_y, 10);
            }
        }
        ScenarioKind::HeatBox => {
            let x0 = WIDTH / 4;
            let y0 = HEIGHT / 4;
            let w = WIDTH / 2;
            let h = HEIGHT / 2;
            for y in y0..(y0 + h) {
                for x in x0..(x0 + w) {
                    sim.set_heat_field_for_test(x, y, 20);
                }
            }
        }
        ScenarioKind::ChargeWave => {
            let mid_x = WIDTH / 2;
            for y in 0..HEIGHT {
                sim.set_charge_field_for_test(mid_x, y, 15);
            }
        }
        ScenarioKind::InteractOscillator => {
            let x = WIDTH / 2;
            let y = HEIGHT / 2;
            sim.set_heat_field_for_test(x, y, 30);
            sim.set_charge_field_for_test(x, y, 30);
        }
        ScenarioKind::Custom => {}
    }
}

pub fn run_scenario(sim: &mut Simulation, scenario: &PhysicsScenario) -> ScenarioSummary {
    // Seed initial pattern
    seed_scenario(sim, &scenario.kind);

    let _heat_params = scenario.heat_params.clone().unwrap_or_default();
    let _charge_params = scenario.charge_params.clone().unwrap_or_default();
    let diffuse_params = scenario.diffuse_params.clone().unwrap_or_default();
    let reaction_params = scenario.reaction_params.clone().unwrap_or_default();
    let interact_params = scenario.interact_params.clone().unwrap_or_default();
    let _coupling_params = scenario.coupling_params.clone().unwrap_or_default();

    let mut max_heat = 0u32;
    let mut max_charge = 0u32;
    let (x0, y0, x1, y1) = region_bounds(&scenario.region);

    let steps = scenario.steps;
    for step in 0..steps {
        // coupling (logic -> fields) if desired via existing helpers
        sim.step_fields_with(&crate::fieldstep::FieldStepParams::default());

        // diffusion
        sim.step_heat_diffuse_n(1, &diffuse_params);
        sim.step_charge_diffuse_n(1, &diffuse_params);

        // reaction
        sim.step_reaction_once(&reaction_params);

        // interaction
        sim.step_interact_once(&interact_params);

        // collect stats at tick_stride
        if scenario.tick_stride == 0 || step % scenario.tick_stride == 0 {
            for y in y0..y1 {
                for x in x0..x1 {
                    max_heat = max_heat.max(sim.get_heat_field_for_test(x, y));
                    max_charge = max_charge.max(sim.get_charge_field_for_test(x, y));
                }
            }
        }
    }

    // Final averages
    let mut sum_heat: u64 = 0;
    let mut sum_charge: u64 = 0;
    let mut count: u64 = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            sum_heat += sim.get_heat_field_for_test(x, y) as u64;
            sum_charge += sim.get_charge_field_for_test(x, y) as u64;
            count += 1;
        }
    }
    let final_avg_heat = if count > 0 {
        sum_heat as f32 / count as f32
    } else {
        0.0
    };
    let final_avg_charge = if count > 0 {
        sum_charge as f32 / count as f32
    } else {
        0.0
    };

    ScenarioSummary {
        steps,
        region: scenario.region.clone(),
        max_heat,
        max_charge,
        final_avg_heat,
        final_avg_charge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_pulse_spreads_deterministically() {
        let mut sim = Simulation::new();
        let scenario = PhysicsScenario {
            kind: ScenarioKind::HeatPulse1D,
            steps: 5,
            tick_stride: 1,
            region: None,
            heat_params: None,
            charge_params: None,
            diffuse_params: None,
            reaction_params: None,
            interact_params: None,
            coupling_params: None,
        };
        let summary1 = run_scenario(&mut sim, &scenario);
        let mut sim2 = Simulation::new();
        let summary2 = run_scenario(&mut sim2, &scenario);
        assert_eq!(summary1.max_heat, summary2.max_heat);
    }
}
