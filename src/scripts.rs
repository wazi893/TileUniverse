use crate::charge::ChargeParams;
use crate::coupling::CoupledParams;
use crate::diffuse::DiffuseParams;
use crate::heat::HeatParams;
use crate::physics_interact::InteractionParams;
use crate::physics_scenarios::Region;
use crate::reaction::ReactionParams;
use crate::simulation::Simulation;
use crate::tile_meta::TileType;

#[derive(Clone, Copy, Debug)]
pub enum ScriptStepKind {
    SeedHeat,
    SeedCharge,
    SetRegionLogic,
    RunTicks,
    RunFieldstep,
    RunDiffuseHeat,
    RunDiffuseCharge,
    RunReaction,
    RunCoupled,
    SnapshotRegion,
    MeasureSummary,
    Label,
}

#[derive(Clone, Debug, Default)]
pub struct ScriptStepArgs {
    pub region: Option<Region>,
    pub steps: Option<u32>,
    pub heat_scale: Option<u32>,
    pub charge_scale: Option<u32>,
    pub heat_params: Option<HeatParams>,
    pub charge_params: Option<ChargeParams>,
    pub diffuse_params: Option<DiffuseParams>,
    pub reaction_params: Option<ReactionParams>,
    pub coupling_params: Option<CoupledParams>,
    pub interact_params: Option<InteractionParams>,
    pub tile_type: Option<TileType>,
    pub label: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ScriptStep {
    pub kind: ScriptStepKind,
    pub args: ScriptStepArgs,
}

#[derive(Clone, Debug)]
pub struct Script {
    pub name: String,
    pub steps: Vec<ScriptStep>,
}

#[derive(Clone, Debug)]
pub struct ScriptSnapshot {
    pub label: String,
    pub region: Region,
    pub data: Vec<Vec<u64>>,
}

#[derive(Clone, Debug)]
pub struct ScriptMeasurement {
    pub label: String,
    pub region: Region,
    pub max_heat: u32,
    pub avg_heat: f32,
    pub max_charge: u32,
    pub avg_charge: f32,
}

#[derive(Clone, Debug, Default)]
pub struct ScriptResult {
    pub snapshots: Vec<ScriptSnapshot>,
    pub measurements: Vec<ScriptMeasurement>,
}

#[derive(Debug)]
pub enum ScriptError {
    MissingRegion,
}

pub struct ScriptPreset {
    pub name: &'static str,
    pub script: Script,
}

pub fn preset_scripts() -> Vec<ScriptPreset> {
    vec![
        ScriptPreset {
            name: "pulse_heat",
            script: Script {
                name: "pulse_heat".to_string(),
                steps: vec![
                    ScriptStep {
                        kind: ScriptStepKind::SeedHeat,
                        args: ScriptStepArgs {
                            region: Some(Region {
                                x0: 10,
                                y0: 10,
                                w: 5,
                                h: 5,
                            }),
                            heat_scale: Some(50),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::RunFieldstep,
                        args: ScriptStepArgs {
                            steps: Some(5),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::SnapshotRegion,
                        args: ScriptStepArgs {
                            region: Some(Region {
                                x0: 0,
                                y0: 0,
                                w: 16,
                                h: 16,
                            }),
                            label: Some("pulse_heat_snapshot".to_string()),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::MeasureSummary,
                        args: ScriptStepArgs {
                            region: Some(Region {
                                x0: 0,
                                y0: 0,
                                w: 16,
                                h: 16,
                            }),
                            label: Some("pulse_heat_measure".to_string()),
                            ..Default::default()
                        },
                    },
                ],
            },
        },
        ScriptPreset {
            name: "react_then_diffuse",
            script: Script {
                name: "react_then_diffuse".to_string(),
                steps: vec![
                    ScriptStep {
                        kind: ScriptStepKind::SeedCharge,
                        args: ScriptStepArgs {
                            region: Some(Region {
                                x0: 5,
                                y0: 5,
                                w: 10,
                                h: 1,
                            }),
                            charge_scale: Some(30),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::RunReaction,
                        args: ScriptStepArgs {
                            steps: Some(3),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::RunDiffuseCharge,
                        args: ScriptStepArgs {
                            steps: Some(5),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::SnapshotRegion,
                        args: ScriptStepArgs {
                            region: Some(Region {
                                x0: 0,
                                y0: 0,
                                w: 16,
                                h: 16,
                            }),
                            label: Some("react_diffuse_snap".to_string()),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::MeasureSummary,
                        args: ScriptStepArgs {
                            region: Some(Region {
                                x0: 0,
                                y0: 0,
                                w: 16,
                                h: 16,
                            }),
                            label: Some("react_diffuse_meas".to_string()),
                            ..Default::default()
                        },
                    },
                ],
            },
        },
        ScriptPreset {
            name: "logic_clock_heat",
            script: Script {
                name: "logic_clock_heat".to_string(),
                steps: vec![
                    ScriptStep {
                        kind: ScriptStepKind::SetRegionLogic,
                        args: ScriptStepArgs {
                            region: Some(Region {
                                x0: 0,
                                y0: 0,
                                w: 1,
                                h: 16,
                            }),
                            tile_type: Some(TileType::ClockGlobal),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::RunTicks,
                        args: ScriptStepArgs {
                            steps: Some(3),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::SeedHeat,
                        args: ScriptStepArgs {
                            region: Some(Region {
                                x0: 0,
                                y0: 0,
                                w: 1,
                                h: 16,
                            }),
                            heat_scale: Some(20),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::RunFieldstep,
                        args: ScriptStepArgs {
                            steps: Some(2),
                            ..Default::default()
                        },
                    },
                    ScriptStep {
                        kind: ScriptStepKind::SnapshotRegion,
                        args: ScriptStepArgs {
                            region: Some(Region {
                                x0: 0,
                                y0: 0,
                                w: 8,
                                h: 16,
                            }),
                            label: Some("logic_clock_heat_snap".to_string()),
                            ..Default::default()
                        },
                    },
                ],
            },
        },
    ]
}

fn region_bounds(region: &Region) -> (usize, usize, usize, usize) {
    (
        region.x0 as usize,
        region.y0 as usize,
        (region.x0 + region.w) as usize,
        (region.y0 + region.h) as usize,
    )
}

fn measure_region(sim: &Simulation, region: &Region) -> ScriptMeasurement {
    let (x0, y0, x1, y1) = region_bounds(region);
    let mut max_heat = 0u32;
    let mut max_charge = 0u32;
    let mut sum_heat = 0f32;
    let mut sum_charge = 0f32;
    let mut count = 0f32;
    for y in y0..y1 {
        for x in x0..x1 {
            let h = sim.get_heat_field_for_test(x, y);
            let c = sim.get_charge_field_for_test(x, y);
            max_heat = max_heat.max(h);
            max_charge = max_charge.max(c);
            sum_heat += h as f32;
            sum_charge += c as f32;
            count += 1.0;
        }
    }
    ScriptMeasurement {
        label: "".to_string(),
        region: region.clone(),
        max_heat,
        avg_heat: if count > 0.0 { sum_heat / count } else { 0.0 },
        max_charge,
        avg_charge: if count > 0.0 { sum_charge / count } else { 0.0 },
    }
}

pub fn run_script(sim: &mut Simulation, script: &Script) -> Result<ScriptResult, ScriptError> {
    let mut result = ScriptResult::default();
    for step in &script.steps {
        match step.kind {
            ScriptStepKind::SeedHeat => {
                let region = step.args.region.clone().ok_or(ScriptError::MissingRegion)?;
                let scale = step
                    .args
                    .heat_scale
                    .unwrap_or(HeatParams::default().inject_scale);
                let (x0, y0, x1, y1) = region_bounds(&region);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let cur = sim.get_heat_field_for_test(x, y);
                        sim.set_heat_field_for_test(x, y, cur.saturating_add(scale));
                    }
                }
            }
            ScriptStepKind::SeedCharge => {
                let region = step.args.region.clone().ok_or(ScriptError::MissingRegion)?;
                let scale = step
                    .args
                    .charge_scale
                    .unwrap_or(ChargeParams::default().inject_scale);
                let (x0, y0, x1, y1) = region_bounds(&region);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let cur = sim.get_charge_field_for_test(x, y);
                        sim.set_charge_field_for_test(x, y, cur.saturating_add(scale));
                    }
                }
            }
            ScriptStepKind::SetRegionLogic => {
                let region = step.args.region.clone().ok_or(ScriptError::MissingRegion)?;
                let tt = step.args.tile_type.unwrap_or(TileType::Wire);
                let (x0, y0, x1, y1) = region_bounds(&region);
                for y in y0..y1 {
                    for x in x0..x1 {
                        sim.set_tile(x, y, tt);
                    }
                }
            }
            ScriptStepKind::RunTicks => {
                let steps = step.args.steps.unwrap_or(1).min(1000);
                for _ in 0..steps {
                    sim.tick();
                }
            }
            ScriptStepKind::RunFieldstep => {
                let steps = step.args.steps.unwrap_or(1).min(1000);
                let params = crate::fieldstep::FieldStepParams::default();
                sim.step_fields_n(&params, steps);
            }
            ScriptStepKind::RunDiffuseHeat => {
                let steps = step.args.steps.unwrap_or(1).min(1000);
                let params = step.args.diffuse_params.clone().unwrap_or_default();
                sim.step_heat_diffuse_n(steps, &params);
            }
            ScriptStepKind::RunDiffuseCharge => {
                let steps = step.args.steps.unwrap_or(1).min(1000);
                let params = step.args.diffuse_params.clone().unwrap_or_default();
                sim.step_charge_diffuse_n(steps, &params);
            }
            ScriptStepKind::RunReaction => {
                let steps = step.args.steps.unwrap_or(1).min(1000);
                let params = step.args.reaction_params.clone().unwrap_or_default();
                sim.step_reaction_n(steps, &params);
            }
            ScriptStepKind::RunCoupled => {
                let steps = step.args.steps.unwrap_or(1).min(1000);
                let params = step.args.coupling_params.clone().unwrap_or_default();
                sim.coupled_step_n(steps, &params);
            }
            ScriptStepKind::SnapshotRegion => {
                let region = step.args.region.clone().ok_or(ScriptError::MissingRegion)?;
                let label = step
                    .args
                    .label
                    .clone()
                    .unwrap_or_else(|| "snapshot".to_string());
                let snap = sim.snapshot_region(
                    region.x0 as usize,
                    region.y0 as usize,
                    region.w as usize,
                    region.h as usize,
                );
                result.snapshots.push(ScriptSnapshot {
                    label,
                    region,
                    data: snap,
                });
            }
            ScriptStepKind::MeasureSummary => {
                let region = step.args.region.clone().ok_or(ScriptError::MissingRegion)?;
                let mut meas = measure_region(sim, &region);
                meas.label = step
                    .args
                    .label
                    .clone()
                    .unwrap_or_else(|| "measure".to_string());
                result.measurements.push(meas);
            }
            ScriptStepKind::Label => {
                // no-op
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_empty_noop() {
        let mut sim = Simulation::new();
        let script = Script {
            name: "empty".to_string(),
            steps: vec![],
        };
        let before_heat = sim.get_heat_field_for_test(0, 0);
        let res = run_script(&mut sim, &script).unwrap();
        assert!(res.snapshots.is_empty());
        assert_eq!(sim.get_heat_field_for_test(0, 0), before_heat);
    }

    #[test]
    fn seed_heat_then_snapshot() {
        let mut sim = Simulation::new();
        let region = Region {
            x0: 0,
            y0: 0,
            w: 1,
            h: 1,
        };
        let script = Script {
            name: "seed".to_string(),
            steps: vec![
                ScriptStep {
                    kind: ScriptStepKind::SeedHeat,
                    args: ScriptStepArgs {
                        region: Some(region.clone()),
                        heat_scale: Some(5),
                        ..Default::default()
                    },
                },
                ScriptStep {
                    kind: ScriptStepKind::SnapshotRegion,
                    args: ScriptStepArgs {
                        region: Some(region.clone()),
                        ..Default::default()
                    },
                },
            ],
        };
        let res = run_script(&mut sim, &script).unwrap();
        assert_eq!(res.snapshots.len(), 1);
        assert_eq!(sim.get_heat_field_for_test(0, 0), 5);
    }

    #[test]
    fn diffuse_via_script_matches_direct() {
        let mut sim = Simulation::new();
        let _region = Region {
            x0: 1,
            y0: 1,
            w: 1,
            h: 1,
        };
        sim.set_heat_field_for_test(1, 1, 100);
        let script = Script {
            name: "diffuse".to_string(),
            steps: vec![ScriptStep {
                kind: ScriptStepKind::RunDiffuseHeat,
                args: ScriptStepArgs {
                    steps: Some(1),
                    ..Default::default()
                },
            }],
        };
        let mut sim_manual = Simulation::new();
        sim_manual.set_heat_field_for_test(1, 1, 100);
        run_script(&mut sim, &script).unwrap();
        sim_manual.step_heat_diffuse_n(1, &DiffuseParams::default());
        assert_eq!(
            sim.get_heat_field_for_test(1, 1),
            sim_manual.get_heat_field_for_test(1, 1)
        );
    }

    #[test]
    fn missing_region_errors() {
        let mut sim = Simulation::new();
        let script = Script {
            name: "bad".to_string(),
            steps: vec![ScriptStep {
                kind: ScriptStepKind::SeedHeat,
                args: ScriptStepArgs::default(),
            }],
        };
        let res = run_script(&mut sim, &script);
        assert!(matches!(res, Err(ScriptError::MissingRegion)));
    }
}
