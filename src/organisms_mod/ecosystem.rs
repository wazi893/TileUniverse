use crate::organisms::{self, OrganismKind};
use crate::render::{self, FieldLayerMode, ImageFormat, RenderParams};
use crate::simulation::Simulation;
use crate::tilemap::{HEIGHT, WIDTH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub x0: u32,
    pub y0: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimMode {
    Exclusive,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceBudgets {
    pub max_heat: u32,
    pub max_charge: u32,
    pub per_step_cap_heat: u32,
    pub per_step_cap_charge: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Thresholds {
    pub predator_heat: u32,
    pub cooperate_charge: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InteractionRules {
    pub predator_prey: bool,
    pub cooperation: bool,
    pub thresholds: Thresholds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeciesPolicy {
    pub move_axis: Option<Axis>,
    pub allow_overwrite: bool,
    pub max_instances: u32,
}

#[derive(Debug, Clone)]
pub struct SpeciesConfig {
    pub name: &'static str,
    pub kind: OrganismKind,
    pub spawn_points: Vec<(u32, u32)>,
    pub policy: SpeciesPolicy,
}

#[derive(Debug, Clone)]
pub struct EcosystemConfig {
    pub region: Region,
    pub steps: u32,
    pub species: Vec<SpeciesConfig>,
    pub budgets: ResourceBudgets,
    pub claim_mode: ClaimMode,
    pub interaction: InteractionRules,
    pub render_each: bool,
}

#[derive(Debug, Clone)]
pub struct OrganismInstance {
    pub species_id: usize,
    pub origin_x: u32,
    pub origin_y: u32,
    pub w: u16,
    pub h: u16,
}

#[derive(Debug, Default, Clone)]
pub struct EcosystemResult {
    pub steps_run: u32,
    pub log: Vec<String>,
    pub species_counts: Vec<u32>,
    pub max_heat: u32,
    pub max_charge: u32,
}

// EPIC 29: Budget tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ActionCost {
    heat: u32,
    charge: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConsumptionReason {
    None,
    Budget,
    Cap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConsumptionResult {
    applied: bool,
    clamped_to: ActionCost,
    reason: ConsumptionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BudgetTracker {
    heat_total: u32,
    charge_total: u32,
    heat_left: u32,
    charge_left: u32,
    per_step_cap_heat: u32,
    per_step_cap_charge: u32,
    step_cap_left_heat: u32,
    step_cap_left_charge: u32,
}

impl BudgetTracker {
    fn new(b: ResourceBudgets) -> Self {
        Self {
            heat_total: b.max_heat,
            charge_total: b.max_charge,
            heat_left: b.max_heat,
            charge_left: b.max_charge,
            per_step_cap_heat: b.per_step_cap_heat,
            per_step_cap_charge: b.per_step_cap_charge,
            step_cap_left_heat: b.per_step_cap_heat,
            step_cap_left_charge: b.per_step_cap_charge,
        }
    }
    fn reset_step_caps(&mut self) {
        self.step_cap_left_heat = self.per_step_cap_heat;
        self.step_cap_left_charge = self.per_step_cap_charge;
    }
    fn try_consume_peek(&self, cost: ActionCost) -> ConsumptionResult {
        let cap_heat = cost.heat.min(self.step_cap_left_heat);
        let cap_charge = cost.charge.min(self.step_cap_left_charge);
        let clamp_heat = cap_heat.min(self.heat_left);
        let clamp_charge = cap_charge.min(self.charge_left);
        let clamped = ActionCost {
            heat: clamp_heat,
            charge: clamp_charge,
        };
        if cost.heat == 0 && cost.charge == 0 {
            return ConsumptionResult {
                applied: true,
                clamped_to: clamped,
                reason: ConsumptionReason::None,
            };
        }
        if cost.heat > 0 && clamped.heat == 0 {
            return ConsumptionResult {
                applied: false,
                clamped_to: clamped,
                reason: if cap_heat == 0 {
                    ConsumptionReason::Cap
                } else {
                    ConsumptionReason::Budget
                },
            };
        }
        if cost.charge > 0 && clamped.charge == 0 {
            return ConsumptionResult {
                applied: false,
                clamped_to: clamped,
                reason: if cap_charge == 0 {
                    ConsumptionReason::Cap
                } else {
                    ConsumptionReason::Budget
                },
            };
        }
        // Discrete actions require exact match
        if clamped.heat < cost.heat || clamped.charge < cost.charge {
            return ConsumptionResult {
                applied: false,
                clamped_to: clamped,
                reason: ConsumptionReason::Cap,
            };
        }
        ConsumptionResult {
            applied: true,
            clamped_to: clamped,
            reason: ConsumptionReason::None,
        }
    }
    fn commit(&mut self, cost: ActionCost) {
        self.heat_left = self.heat_left.saturating_sub(cost.heat);
        self.charge_left = self.charge_left.saturating_sub(cost.charge);
        self.step_cap_left_heat = self.step_cap_left_heat.saturating_sub(cost.heat);
        self.step_cap_left_charge = self.step_cap_left_charge.saturating_sub(cost.charge);
    }
}

fn rects_overlap(a: (u32, u32, u32, u32), b: (u32, u32, u32, u32)) -> bool {
    let (ax0, ay0, ax1, ay1) = a;
    let (bx0, by0, bx1, by1) = b;
    !(ax1 <= bx0 || bx1 <= ax0 || ay1 <= by0 || by1 <= ay0)
}

fn bbox(inst: &OrganismInstance) -> (u32, u32, u32, u32) {
    (
        inst.origin_x,
        inst.origin_y,
        inst.origin_x + inst.w as u32,
        inst.origin_y + inst.h as u32,
    )
}

fn in_region(r: &Region, inst: &OrganismInstance) -> bool {
    let (x0, y0, x1, y1) = bbox(inst);
    x0 >= r.x0 && y0 >= r.y0 && x1 <= r.x0 + r.w && y1 <= r.y0 + r.h
}

fn heat_at(sim: &Simulation, x: u32, y: u32) -> u32 {
    let xu = x as usize;
    let yu = y as usize;
    if xu >= WIDTH || yu >= HEIGHT {
        0
    } else {
        sim.get_heat_field_for_test(xu, yu)
    }
}

pub fn run_ecosystem(sim: &mut Simulation, cfg: &EcosystemConfig) -> EcosystemResult {
    let mut res = EcosystemResult::default();
    let steps = cfg.steps.min(1000);
    // Pre-run validation
    if cfg.region.x0 as usize >= WIDTH || cfg.region.y0 as usize >= HEIGHT {
        res.log.push("config_invalid region_oob".to_string());
        return res;
    }

    // Spawn
    let mut instances: Vec<OrganismInstance> = Vec::new();
    for (sid, sp) in cfg.species.iter().enumerate() {
        for &(ox, oy) in &sp.spawn_points {
            let tmpl = organisms::list_organisms()
                .iter()
                .find(|t| t.kind == sp.kind)
                .unwrap();
            let inst = OrganismInstance {
                species_id: sid,
                origin_x: ox,
                origin_y: oy,
                w: tmpl.width,
                h: tmpl.height,
            };
            if !in_region(&cfg.region, &inst) {
                res.log
                    .push(format!("spawn_skip species={} reason=oob", sp.name));
                continue;
            }
            // claims check
            let mut collide = false;
            for other in &instances {
                if rects_overlap(bbox(&inst), bbox(other)) {
                    collide = true;
                    break;
                }
            }
            if collide && matches!(cfg.claim_mode, ClaimMode::Exclusive) {
                res.log
                    .push(format!("spawn_skip species={} reason=claimed", sp.name));
                continue;
            }
            match organisms::place_organism(sim, sp.kind, ox, oy, sp.policy.allow_overwrite) {
                Ok(()) => {
                    instances.push(inst);
                }
                Err(_) => res.log.push(format!(
                    "spawn_skip species={} reason=place_failed",
                    sp.name
                )),
            }
        }
    }
    res.species_counts = vec![0u32; cfg.species.len()];
    for inst in &instances {
        res.species_counts[inst.species_id] += 1;
    }

    // Loop
    let mut tracker = BudgetTracker::new(cfg.budgets);
    let mut frame_index: u32 = 0;
    let mut actions: Vec<(usize, Option<(u32, u32)>)> = Vec::new();
    for step in 0..steps {
        #[cfg(feature = "perf-bench")]
        let _ = step;
        let mut step_used_heat: u32 = 0;
        let mut step_used_charge: u32 = 0;
        tracker.reset_step_caps();
        // Deterministic order: (species_id, top_y, left_x)
        instances.sort_by(|a, b| {
            a.species_id
                .cmp(&b.species_id)
                .then(a.origin_y.cmp(&b.origin_y))
                .then(a.origin_x.cmp(&b.origin_x))
        });

        // Interactions + movement
        // To avoid double-borrow, compute next positions in a temp vec
        actions.clear();
        for (idx, inst) in instances.iter().enumerate() {
            let sp = &cfg.species[inst.species_id];
            // Predator placement: if heat at center >= threshold, try to place RingOscillator adjacent to the right
            if cfg.interaction.predator_prey {
                let cx = inst.origin_x + (inst.w as u32 / 2);
                let cy = inst.origin_y + (inst.h as u32 / 2);
                let h = heat_at(sim, cx, cy);
                if h >= cfg.interaction.thresholds.predator_heat {
                    let px = inst.origin_x + inst.w as u32;
                    let py = inst.origin_y; // adjacent right
                    // EPIC 29: budget check for placement (charge cost 1)
                    let cost = ActionCost { heat: 0, charge: 1 };
                    let peek = tracker.try_consume_peek(cost);
                    if !peek.applied {
                        let why = match peek.reason {
                            ConsumptionReason::Budget => "budget",
                            ConsumptionReason::Cap => "cap",
                            ConsumptionReason::None => "none",
                        };
                        #[cfg(feature = "perf-bench")]
                        let _ = why;
                        #[cfg(not(feature = "perf-bench"))]
                        res.log.push(format!("step={} species={} action=place_oscillator target=({}, {}) result=skipped(reason={})", step, sp.name, px, py, why));
                    } else {
                        let place_ok = organisms::place_organism(
                            sim,
                            OrganismKind::RingCluster,
                            px,
                            py,
                            false,
                        )
                        .is_ok();
                        if place_ok {
                            tracker.commit(cost);
                            step_used_charge = step_used_charge.saturating_add(cost.charge);
                            step_used_heat = step_used_heat.saturating_add(cost.heat);
                            #[cfg(not(feature = "perf-bench"))]
                            res.log.push(format!(
                                "step={} species={} action=place_oscillator target=({}, {}) result=ok(heat_used={} charge_used={} cap_left=({},{}) rem=({},{}))",
                                step, sp.name, px, py, cost.heat, cost.charge, tracker.step_cap_left_heat, tracker.step_cap_left_charge, tracker.heat_left, tracker.charge_left
                            ));
                        } else {
                            #[cfg(not(feature = "perf-bench"))]
                            res.log.push(format!("step={} species={} action=place_oscillator target=({}, {}) result=skipped(reason=collision_or_oob)", step, sp.name, px, py));
                        }
                    }
                }
            }

            // Movement: one cell along axis if enabled
            if let Some(axis) = sp.policy.move_axis {
                let (nx, ny) = match axis {
                    Axis::X => (inst.origin_x + 1, inst.origin_y),
                    Axis::Y => (inst.origin_x, inst.origin_y + 1),
                };
                // Check bounds and claims against current positions
                let next = OrganismInstance {
                    species_id: inst.species_id,
                    origin_x: nx,
                    origin_y: ny,
                    w: inst.w,
                    h: inst.h,
                };
                let mut blocked = false;
                if !in_region(&cfg.region, &next) {
                    blocked = true;
                }
                if matches!(cfg.claim_mode, ClaimMode::Exclusive) {
                    for (j, other) in instances.iter().enumerate() {
                        if j == idx {
                            continue;
                        }
                        if rects_overlap(bbox(&next), bbox(other)) {
                            blocked = true;
                            break;
                        }
                    }
                }
                if !blocked {
                    actions.push((idx, Some((nx, ny))));
                } else {
                    actions.push((idx, None));
                }
            } else {
                actions.push((idx, None));
            }
        }
        // Apply moves
        for (idx, opt) in actions.drain(..) {
            let sp = &cfg.species[instances[idx].species_id];
            #[cfg(feature = "perf-bench")]
            let _ = sp;
            if let Some((nx, ny)) = opt {
                let from = (instances[idx].origin_x, instances[idx].origin_y);
                #[cfg(feature = "perf-bench")]
                let _ = from;
                instances[idx].origin_x = nx;
                instances[idx].origin_y = ny;
                #[cfg(not(feature = "perf-bench"))]
                res.log.push(format!(
                    "step={} species={} action=move from=({}, {}) to=({}, {}) result=ok",
                    step, sp.name, from.0, from.1, nx, ny
                ));
            } else if cfg.species[instances[idx].species_id]
                .policy
                .move_axis
                .is_some()
            {
                #[cfg(not(feature = "perf-bench"))]
                res.log.push(format!(
                    "step={} species={} action=move result=skipped(reason=blocked)",
                    step, sp.name
                ));
            }
        }

        // Optional render snapshot with GUI-compatible meta
        if cfg.render_each {
            let p = RenderParams {
                fields: FieldLayerMode::Heat,
                scale: 2,
                format: ImageFormat::Ppm,
            };
            let img = render::render_region(
                sim,
                cfg.region.x0,
                cfg.region.y0,
                cfg.region.w,
                cfg.region.h,
                &p,
            );
            if let Ok(img) = img {
                let mut buf = Vec::new();
                if render::write_ppm(&img, &mut buf).is_ok() {
                    // FNV-1a32
                    let mut hash: u32 = 0x811C9DC5;
                    for &b in &buf {
                        hash ^= b as u32;
                        hash = hash.wrapping_mul(0x01000193);
                    }
                    #[cfg(not(feature = "perf-bench"))]
                    res.log.push(format!(
                        "frame index={} t={} w={} h={} field=heat checksum=0x{:08x}",
                        frame_index, step, img.width, img.height, hash
                    ));
                    frame_index = frame_index.saturating_add(1);
                }
            }
        }

        // EPIC 29: budget meta per step
        #[cfg(not(feature = "perf-bench"))]
        res.log.push(format!(
            "meta_budget: step={} heat_used={} charge_used={} heat_left={} charge_left={}",
            step, step_used_heat, step_used_charge, tracker.heat_left, tracker.charge_left
        ));
    }

    res.steps_run = steps;
    res
}
