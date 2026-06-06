use crate::circuits::{CircuitError, CircuitKind, PlacementOptions, place_circuit};
use crate::patterns::{self, FieldSelect, PatternParams};
use crate::render::{self, FieldLayerMode, ImageFormat, RenderParams};
use crate::simulation::Simulation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentCategory {
    CircuitPlacement,
}

#[derive(Debug, Clone)]
pub enum AgentStep {
    Analyze {
        fields: Vec<FieldSelect>,
        params: PatternParams,
    },
    Place {
        kind: CircuitKind,
        x: u32,
        y: u32,
        allow_overwrite: bool,
    },
    PlaceFromHeatBlobs {
        min_area: u32,
        max_count: u32,
    },
    Render {
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        mode: FieldLayerMode,
        scale: u32,
        path: String,
    },
}

#[derive(Debug, Clone)]
pub struct AgentPlan {
    pub name: &'static str,
    pub category: AgentCategory,
    pub summary: &'static str,
    pub steps: Vec<AgentStep>,
}

#[derive(Debug, Clone)]
pub struct StepResultSummary {
    pub index: usize,
    pub label: String,
    pub ok: bool,
    pub notes: String,
}

#[derive(Debug, Clone)]
pub struct AgentResult {
    pub plan_name: String,
    pub log: Vec<StepResultSummary>,
    pub error: Option<String>,
}

fn step_label(step: &AgentStep) -> String {
    match step {
        AgentStep::Analyze { .. } => "Analyze".to_string(),
        AgentStep::Place { kind, .. } => format!("Place.{:?}", kind),
        AgentStep::PlaceFromHeatBlobs { .. } => "PlaceFromHeatBlobs".to_string(),
        AgentStep::Render { .. } => "Render".to_string(),
    }
}

pub fn list_plans() -> &'static [AgentPlan] {
    static PLANS: once_cell::sync::Lazy<Vec<AgentPlan>> = once_cell::sync::Lazy::new(|| {
        vec![
            AgentPlan {
                name: "seed_ring_osc_clusters",
                category: AgentCategory::CircuitPlacement,
                summary: "Analyze heat blobs and place ring oscillators near centers.",
                steps: vec![
                    AgentStep::Analyze {
                        fields: vec![FieldSelect::Heat],
                        params: PatternParams {
                            blob_threshold: 1,
                            edge_threshold: 1,
                            min_blob_area: 1,
                            max_results: 10,
                        },
                    },
                    AgentStep::PlaceFromHeatBlobs {
                        min_area: 1,
                        max_count: 3,
                    },
                ],
            },
            AgentPlan {
                name: "repair_latch_strips",
                category: AgentCategory::CircuitPlacement,
                summary: "Inspect logic edges and place latch strips along horizontal ridges.",
                steps: vec![
                    AgentStep::Analyze {
                        fields: vec![FieldSelect::LogicField],
                        params: PatternParams {
                            blob_threshold: 1,
                            edge_threshold: 1,
                            min_blob_area: 1,
                            max_results: 10,
                        },
                    },
                    // Minimal placeholder: no automatic placement from edges yet.
                ],
            },
            AgentPlan {
                name: "autowire_bus_between_blobs",
                category: AgentCategory::CircuitPlacement,
                summary: "Connect large charge blobs with wire bus when aligned.",
                steps: vec![
                    AgentStep::Analyze {
                        fields: vec![FieldSelect::Charge],
                        params: PatternParams {
                            blob_threshold: 1,
                            edge_threshold: 1,
                            min_blob_area: 2,
                            max_results: 10,
                        },
                    },
                    // Placeholder: manual placement to keep determinism without routing
                ],
            },
        ]
    });
    &PLANS
}

pub fn run_plan(sim: &mut Simulation, plan: &AgentPlan, dry_run: bool) -> AgentResult {
    let mut log: Vec<StepResultSummary> = Vec::new();
    let mut last_detect: Option<patterns::DetectionReport> = None;

    for (i, step) in plan.steps.iter().enumerate() {
        let notes: String;
        let mut ok = true;
        match step {
            AgentStep::Analyze { fields, params } => {
                // Compute summaries for requested fields
                let rep = patterns::detect_patterns(sim, fields, params);
                notes = format!(
                    "fields={} blobs0={}",
                    fields.len(),
                    rep.field_summaries
                        .get(0)
                        .map(|s| s.blobs.len())
                        .unwrap_or(0)
                );
                last_detect = Some(rep);
            }
            AgentStep::Place {
                kind,
                x,
                y,
                allow_overwrite,
            } => {
                if dry_run {
                    notes = format!("would place {:?} at ({},{})", kind, x, y);
                } else {
                    match place_circuit(
                        sim,
                        *kind,
                        *x,
                        *y,
                        &PlacementOptions {
                            allow_overwrite: *allow_overwrite,
                        },
                    ) {
                        Ok(_) => {
                            notes = format!("placed {:?} at ({},{})", kind, x, y);
                        }
                        Err(e) => {
                            ok = false;
                            notes = format!("place error: {:?}", e);
                        }
                    }
                }
            }
            AgentStep::PlaceFromHeatBlobs {
                min_area,
                max_count,
            } => {
                // Use latest detect or compute heat-only
                let rep = if let Some(r) = &last_detect {
                    r.clone()
                } else {
                    patterns::detect_patterns(sim, &[FieldSelect::Heat], &PatternParams::default())
                };
                let mut placed = 0u32;
                let mut blobs: Vec<patterns::Blob> = Vec::new();
                for s in &rep.field_summaries {
                    if s.field == "heat" {
                        blobs.extend_from_slice(&s.blobs);
                    }
                }
                for b in blobs.iter() {
                    if b.area < *min_area {
                        continue;
                    }
                    let cx = ((b.x0 + b.x1) / 2) as i64 - 1; // center - half template
                    let cy = ((b.y0 + b.y1) / 2) as i64 - 1;
                    let ox = cx.max(0) as u32;
                    let oy = cy.max(0) as u32;
                    if dry_run {
                        placed = placed.saturating_add(1);
                    } else {
                        match place_circuit(
                            sim,
                            CircuitKind::RingOscillator,
                            ox,
                            oy,
                            &PlacementOptions {
                                allow_overwrite: false,
                            },
                        ) {
                            Ok(_) => {
                                placed = placed.saturating_add(1);
                            }
                            Err(CircuitError::OutOfBounds { .. })
                            | Err(CircuitError::Collision { .. }) => { /* skip gracefully */ }
                            Err(_) => { /* skip */ }
                        }
                    }
                    if placed >= *max_count {
                        break;
                    }
                }
                notes = format!("ring_osc placed={}", placed);
            }
            AgentStep::Render {
                x0,
                y0,
                w,
                h,
                mode,
                scale,
                path,
            } => {
                if dry_run {
                    notes = format!("would render {}x{} at ({},{})->{}", w, h, x0, y0, path);
                } else {
                    let params = RenderParams {
                        fields: *mode,
                        scale: *scale,
                        format: ImageFormat::Ppm,
                    };
                    match render::render_region(sim, *x0, *y0, *w, *h, &params) {
                        Ok(img) => match std::fs::File::create(path) {
                            Ok(mut f) => {
                                if let Err(e) = render::write_ppm(&img, &mut f) {
                                    ok = false;
                                    notes = format!("render io error: {}", e);
                                } else {
                                    notes = format!(
                                        "rendered {}x{} -> {}",
                                        img.width, img.height, path
                                    );
                                }
                            }
                            Err(e) => {
                                ok = false;
                                notes = format!("render create error: {}", e);
                            }
                        },
                        Err(e) => {
                            ok = false;
                            notes = format!("render error: {:?}", e);
                        }
                    }
                }
            }
        }
        log.push(StepResultSummary {
            index: i,
            label: step_label(step),
            ok,
            notes,
        });
        if !ok {
            return AgentResult {
                plan_name: plan.name.to_string(),
                log,
                error: Some(format!("step {} failed", i)),
            };
        }
    }
    AgentResult {
        plan_name: plan.name.to_string(),
        log,
        error: None,
    }
}

pub fn run_plan_named(
    sim: &mut Simulation,
    name: &str,
    dry_run: bool,
) -> Result<AgentResult, String> {
    let plan = list_plans()
        .iter()
        .find(|p| p.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| "Unknown agent".to_string())?;
    Ok(run_plan(sim, plan, dry_run))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_nonempty() {
        assert!(!list_plans().is_empty());
    }

    #[test]
    fn dry_run_ok() {
        let mut sim = Simulation::new();
        let plan = &list_plans()[0];
        let res = run_plan(&mut sim, plan, true);
        assert!(res.error.is_none());
        assert!(!res.log.is_empty());
    }
}
