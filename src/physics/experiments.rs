use crate::diffuse::DiffuseParams;
use crate::heat::HeatParams;
use crate::physics_interact::InteractionParams;
use crate::physics_scenarios::{PhysicsScenario, ScenarioKind, ScenarioSummary};
use crate::reaction::ReactionParams;
use crate::simulation::Simulation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentParam {
    MaxHeat,
    MaxCharge,
    SpreadNumerator,
    DecayPerStep,
    CouplingStrength,
    InteractionStrength,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range1D {
    pub start: u32,
    pub end: u32,
    pub steps: u32,
}

#[derive(Debug, Clone)]
pub enum ExperimentKind {
    Sweep1D {
        scenario: ScenarioKind,
        param: ExperimentParam,
        range: Range1D,
        tick_stride: u32,
    },
}

#[derive(Debug, Clone)]
pub struct RunResult {
    pub param_value: u32,
    pub summary: ScenarioSummary,
}

#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub kind: ExperimentKind,
    pub runs: Vec<RunResult>,
    pub max_heat_overall: u32,
    pub max_charge_overall: u32,
    pub avg_final_heat: f32,
    pub avg_final_charge: f32,
}

fn apply_param_to_scenario(s: &mut PhysicsScenario, param: ExperimentParam, value: u32) {
    match param {
        ExperimentParam::MaxHeat => {
            let mut h = s.heat_params.clone().unwrap_or_default();
            h.max_heat = value;
            s.heat_params = Some(h);
        }
        ExperimentParam::MaxCharge => {
            let mut c = s.charge_params.clone().unwrap_or_default();
            c.max_charge = value;
            s.charge_params = Some(c);
        }
        ExperimentParam::SpreadNumerator => {
            let mut d = s.diffuse_params.clone().unwrap_or_default();
            d.spread_numerator = value.min(d.spread_denominator.max(1));
            s.diffuse_params = Some(d);
        }
        ExperimentParam::DecayPerStep => {
            let mut d = s.diffuse_params.clone().unwrap_or_default();
            d.decay_per_step = value;
            s.diffuse_params = Some(d);
        }
        ExperimentParam::CouplingStrength => {
            let mut r = s.reaction_params.clone().unwrap_or_default();
            r.max_heat = value; // reuse as an example of coupling strength
            s.reaction_params = Some(r);
        }
        ExperimentParam::InteractionStrength => {
            let mut i = s.interact_params.clone().unwrap_or_default();
            i.heat_affects_charge = value;
            i.charge_affects_heat = value;
            s.interact_params = Some(i);
        }
    }
}

fn param_values(range: Range1D) -> Vec<u32> {
    let steps = range.steps.max(1);
    if steps == 1 {
        return vec![range.start];
    }
    let span = range.end.saturating_sub(range.start);
    let step_size = span / (steps - 1);
    (0..steps).map(|i| range.start + i * step_size).collect()
}

pub fn run_experiment(_sim: &mut Simulation, spec: &ExperimentKind) -> ExperimentResult {
    match spec {
        ExperimentKind::Sweep1D {
            scenario,
            param,
            range,
            tick_stride,
        } => {
            let base = PhysicsScenario {
                kind: scenario.clone(),
                steps: 10,
                tick_stride: *tick_stride,
                region: None,
                heat_params: Some(HeatParams::default()),
                charge_params: Some(crate::charge::ChargeParams::default()),
                diffuse_params: Some(DiffuseParams::default()),
                reaction_params: Some(ReactionParams::default()),
                interact_params: Some(InteractionParams::default()),
                coupling_params: None,
            };
            let mut runs = Vec::new();
            for v in param_values(*range) {
                let mut scen = base.clone();
                apply_param_to_scenario(&mut scen, *param, v);
                // fresh sim per run to keep independence
                let mut local_sim = Simulation::new();
                let summary = local_sim.run_scenario(&scen);
                runs.push(RunResult {
                    param_value: v,
                    summary,
                });
            }
            let mut max_heat_overall = 0u32;
            let mut max_charge_overall = 0u32;
            let mut sum_heat = 0f32;
            let mut sum_charge = 0f32;
            for r in &runs {
                max_heat_overall = max_heat_overall.max(r.summary.max_heat);
                max_charge_overall = max_charge_overall.max(r.summary.max_charge);
                sum_heat += r.summary.final_avg_heat;
                sum_charge += r.summary.final_avg_charge;
            }
            let count = runs.len().max(1) as f32;
            ExperimentResult {
                kind: spec.clone(),
                runs,
                max_heat_overall,
                max_charge_overall,
                avg_final_heat: sum_heat / count,
                avg_final_charge: sum_charge / count,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_values_progression() {
        let vals = param_values(Range1D {
            start: 0,
            end: 10,
            steps: 3,
        });
        assert_eq!(vals, vec![0, 5, 10]);
    }

    #[test]
    fn experiment_deterministic() {
        let mut sim1 = Simulation::new();
        let spec = ExperimentKind::Sweep1D {
            scenario: ScenarioKind::HeatPulse1D,
            param: ExperimentParam::MaxHeat,
            range: Range1D {
                start: 1,
                end: 3,
                steps: 2,
            },
            tick_stride: 1,
        };
        let res1 = run_experiment(&mut sim1, &spec);
        let mut sim2 = Simulation::new();
        let res2 = run_experiment(&mut sim2, &spec);
        assert_eq!(res1.runs.len(), res2.runs.len());
        assert_eq!(res1.max_heat_overall, res2.max_heat_overall);
    }
}
