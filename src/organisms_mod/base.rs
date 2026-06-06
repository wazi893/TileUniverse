use crate::circuits::{CircuitError, CircuitKind, PlacementOptions, place_circuit};
// use crate::patterns::{self, FieldSelect, PatternParams};
use crate::render::{self, FieldLayerMode, ImageFormat, RenderParams};
use crate::simulation::Simulation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrganismKind {
    HeatEmitter,
    RingCluster,
    ChargeCrawler,
    WireBundle,
    OscillatorNest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneRole {
    Core,
    Input,
    Output,
    ReplicationSeed,
    Sensor,
    Actuator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub x0: u32,
    pub y0: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoneSpec {
    pub region: Region,
    pub role: ZoneRole,
}

#[derive(Debug, Clone)]
pub struct OrganismTemplate {
    pub kind: OrganismKind,
    pub name: &'static str,
    pub width: u16,
    pub height: u16,
    pub zones: &'static [ZoneSpec],
    pub circuits: &'static [(CircuitKind, u16, u16)],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrganismError {
    UnknownOrganism,
    OutOfBounds,
    Collision,
    InvalidParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrganismParams {
    pub steps: u32,
    pub render_each: bool,
}

impl Default for OrganismParams {
    fn default() -> Self {
        Self {
            steps: 1,
            render_each: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrganismResult {
    pub name: String,
    pub steps_run: u32,
    pub actions: Vec<String>,
    pub error: Option<String>,
}

// Static templates
const HEAT_EMITTER_TMPL: OrganismTemplate = OrganismTemplate {
    kind: OrganismKind::HeatEmitter,
    name: "heat_emitter",
    width: 3,
    height: 3,
    zones: &[ZoneSpec {
        region: Region {
            x0: 1,
            y0: 1,
            w: 1,
            h: 1,
        },
        role: ZoneRole::Core,
    }],
    circuits: &[],
};

// 3 ring oscillators arranged in a small triangle around origin
const RING_CLUSTER_TMPL: OrganismTemplate = OrganismTemplate {
    kind: OrganismKind::RingCluster,
    name: "ring_cluster",
    width: 7,
    height: 5,
    zones: &[ZoneSpec {
        region: Region {
            x0: 2,
            y0: 2,
            w: 1,
            h: 1,
        },
        role: ZoneRole::Core,
    }],
    circuits: &[
        (CircuitKind::RingOscillator, 0, 0),
        (CircuitKind::RingOscillator, 4, 0),
        (CircuitKind::RingOscillator, 2, 3),
    ],
};

// A simple crawler laying wire_bus segments to the right each step
const CHARGE_CRAWLER_TMPL: OrganismTemplate = OrganismTemplate {
    kind: OrganismKind::ChargeCrawler,
    name: "charge_crawler",
    width: 8,
    height: 1,
    zones: &[ZoneSpec {
        region: Region {
            x0: 0,
            y0: 0,
            w: 8,
            h: 1,
        },
        role: ZoneRole::Actuator,
    }],
    circuits: &[],
};

const WIRE_BUNDLE_TMPL: OrganismTemplate = OrganismTemplate {
    kind: OrganismKind::WireBundle,
    name: "wire_bundle",
    width: 8,
    height: 3,
    zones: &[ZoneSpec {
        region: Region {
            x0: 0,
            y0: 1,
            w: 8,
            h: 1,
        },
        role: ZoneRole::Core,
    }],
    circuits: &[(CircuitKind::WireBus, 0, 1)],
};

const OSC_NEST_TMPL: OrganismTemplate = OrganismTemplate {
    kind: OrganismKind::OscillatorNest,
    name: "oscillator_nest",
    width: 7,
    height: 7,
    zones: &[ZoneSpec {
        region: Region {
            x0: 3,
            y0: 3,
            w: 1,
            h: 1,
        },
        role: ZoneRole::Core,
    }],
    circuits: &[
        (CircuitKind::RingOscillator, 0, 0),
        (CircuitKind::RingOscillator, 4, 0),
        (CircuitKind::RingOscillator, 0, 4),
        (CircuitKind::RingOscillator, 4, 4),
    ],
};

pub fn list_organisms() -> &'static [OrganismTemplate] {
    &[
        HEAT_EMITTER_TMPL,
        RING_CLUSTER_TMPL,
        CHARGE_CRAWLER_TMPL,
        WIRE_BUNDLE_TMPL,
        OSC_NEST_TMPL,
    ]
}

fn find_template(kind: OrganismKind) -> Option<&'static OrganismTemplate> {
    list_organisms().iter().find(|t| t.kind == kind)
}

pub fn place_organism(
    sim: &mut Simulation,
    kind: OrganismKind,
    origin_x: u32,
    origin_y: u32,
    allow_overwrite: bool,
) -> Result<(), OrganismError> {
    let t = find_template(kind).ok_or(OrganismError::UnknownOrganism)?;
    // Dry-run bounds by checking last cell
    let max_x = origin_x + t.width as u32 - 1;
    let max_y = origin_y + t.height as u32 - 1;
    if max_x as usize >= crate::tilemap::WIDTH || max_y as usize >= crate::tilemap::HEIGHT {
        return Err(OrganismError::OutOfBounds);
    }
    // Place circuits via placement API
    for (ck, lx, ly) in t.circuits {
        let x = origin_x + *lx as u32;
        let y = origin_y + *ly as u32;
        match place_circuit(sim, *ck, x, y, &PlacementOptions { allow_overwrite }) {
            Ok(_) => {}
            Err(CircuitError::OutOfBounds { .. }) => return Err(OrganismError::OutOfBounds),
            Err(CircuitError::Collision { .. }) => return Err(OrganismError::Collision),
            Err(_) => return Err(OrganismError::InvalidParams),
        }
    }
    Ok(())
}

pub fn run_organism(
    sim: &mut Simulation,
    kind: OrganismKind,
    origin_x: u32,
    origin_y: u32,
    params: &OrganismParams,
) -> OrganismResult {
    let mut actions = Vec::new();
    let steps = params.steps.min(1000);
    for i in 0..steps {
        match kind {
            OrganismKind::HeatEmitter => {
                // Seed heat at core zone center, clamp value
                let t = HEAT_EMITTER_TMPL;
                let cx = origin_x + (t.width as u32 / 2);
                let cy = origin_y + (t.height as u32 / 2);
                let val = (i + 1).min(10) as u32; // mild, deterministic
                let clamped = val.min(255);
                sim.set_heat_field_for_test(cx as usize, cy as usize, clamped);
                actions.push(format!("seed_heat x={} y={} value={}", cx, cy, clamped));
            }
            OrganismKind::RingCluster => {
                // Mild heat pulse at cluster core zone
                let t = RING_CLUSTER_TMPL;
                let cx = origin_x + (t.width as u32 / 2);
                let cy = origin_y + (t.height as u32 / 2);
                let val = 1u32;
                let clamped = val.min(255);
                sim.set_heat_field_for_test(cx as usize, cy as usize, clamped);
                actions.push(format!("seed_heat x={} y={} value={}", cx, cy, clamped));
            }
            OrganismKind::ChargeCrawler => {
                // Grow by placing a wire_bus 1 step to the right per round, reject on OOB/collision
                let x = origin_x + (i * 4) as u32; // wire_bus width = 4
                let y = origin_y;
                match place_circuit(
                    sim,
                    CircuitKind::WireBus,
                    x,
                    y,
                    &PlacementOptions {
                        allow_overwrite: false,
                    },
                ) {
                    Ok(_) => actions.push(format!("placed wire_bus at ({},{})", x, y)),
                    Err(CircuitError::OutOfBounds { .. }) => {
                        actions.push("crawler_oob".to_string());
                        break;
                    }
                    Err(CircuitError::Collision { .. }) => {
                        actions.push("crawler_collision".to_string());
                        break;
                    }
                    Err(_) => {
                        actions.push("crawler_error".to_string());
                        break;
                    }
                }
            }
            OrganismKind::WireBundle => {
                // No dynamic behavior in initial cut
                actions.push("noop".to_string());
            }
            OrganismKind::OscillatorNest => {
                // Mild alternating heat pulses within bounds
                let t = OSC_NEST_TMPL;
                let cx = origin_x + (t.width as u32 / 2);
                let cy = origin_y + (t.height as u32 / 2);
                let val = if i % 2 == 0 { 2 } else { 1 };
                sim.set_heat_field_for_test(cx as usize, cy as usize, val);
                actions.push(format!("seed_heat x={} y={} value={}", cx, cy, val));
            }
        }

        if params.render_each {
            let t = find_template(kind).unwrap();
            let rparams = RenderParams {
                fields: FieldLayerMode::Heat,
                scale: 2,
                format: ImageFormat::Ppm,
            };
            if let Ok(img) = render::render_region(
                sim,
                origin_x,
                origin_y,
                t.width as u32,
                t.height as u32,
                &rparams,
            ) {
                let path = format!("organism_{}_step_{}.ppm", t.name, i);
                if let Ok(mut f) = std::fs::File::create(&path) {
                    let _ = render::write_ppm(&img, &mut f);
                    actions.push(format!("render {}", path));
                }
            }
        }
    }

    OrganismResult {
        name: find_template(kind)
            .map(|t| t.name.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        steps_run: steps,
        actions,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_nonempty() {
        assert!(!list_organisms().is_empty());
    }

    #[test]
    fn place_ring_cluster_ok() {
        let mut sim = Simulation::new();
        place_organism(&mut sim, OrganismKind::RingCluster, 10, 10, false).unwrap();
    }

    #[test]
    fn run_heat_emitter_two_steps() {
        let mut sim = Simulation::new();
        place_organism(&mut sim, OrganismKind::HeatEmitter, 5, 5, false).unwrap();
        let res = run_organism(
            &mut sim,
            OrganismKind::HeatEmitter,
            5,
            5,
            &OrganismParams {
                steps: 2,
                render_each: false,
            },
        );
        assert!(res.error.is_none());
        assert!(res.actions.iter().any(|a| a.contains("seed_heat")));
    }
}
