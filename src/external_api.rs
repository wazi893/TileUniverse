use crate::ecosystem;
use crate::gui_api;
use crate::gui_backend::FrameFieldKind;
use crate::organisms::OrganismKind;
use crate::recorder;
use crate::simulation::Simulation;
use std::path::Path;

pub const API_VERSION: &str = "1";

fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811C9DC5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

fn contract_hash() -> String {
    // Stable signature over endpoint names + dto keys (minimal, deterministic)
    let sig = b"/api/read/run_info,/api/read/run_frame,/api/exec/place_organism,/api/exec/run_organism,/api/exec/record_organism,/api/exec/replay,/api/exec/diff,/api/exec/run_ecosystem";
    format!("{:08x}", fnv1a32(sig))
}

#[derive(Debug, Clone)]
pub struct ApiRoot {
    pub version: String,
    pub contract_hash: String,
}

#[derive(Debug, Clone)]
pub struct ApiError {
    pub code: String,
    pub message: String,
}

impl ApiError {
    pub fn new(code: &str, msg: &str) -> Self {
        Self {
            code: code.into(),
            message: msg.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GuiRunInfoDTO {
    pub frames: usize,
    pub width: u32,
    pub height: u32,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FrameMetaDTO {
    pub index: u32,
    pub t: u32,
    pub width: u32,
    pub height: u32,
    pub field: String,
    pub checksum: u32,
}

#[derive(Debug, Clone)]
pub struct FrameDTO {
    pub meta: FrameMetaDTO,
    pub ppm_base64: String,
}

#[derive(Debug, Clone)]
pub struct OkDTO {
    pub ok: bool,
}

#[derive(Debug, Clone)]
pub struct ReplayResultDTO {
    pub ok: bool,
}

#[derive(Debug, Clone)]
pub struct DiffResultDTO {
    pub mismatches: usize,
    pub length_diff: usize,
}

#[derive(Debug, Clone)]
pub struct RecordManifestDTO {
    pub run_kind: String,
    pub name: String,
    pub steps: u32,
}

#[derive(Debug, Clone)]
pub struct EcosystemSummaryDTO {
    pub steps_run: u32,
}

pub trait DeterministicEngine {
    fn api_root(&self) -> ApiRoot;
    // READ
    fn read_run_info(&self, dir: &Path) -> Result<GuiRunInfoDTO, ApiError>;
    fn read_run_frame(
        &self,
        dir: &Path,
        index: usize,
        max_bytes: usize,
    ) -> Result<FrameDTO, ApiError>;
    // EXEC
    fn exec_place_organism(
        &mut self,
        name: &str,
        x: u32,
        y: u32,
        allow_overwrite: bool,
    ) -> Result<OkDTO, ApiError>;
    fn exec_run_organism(
        &mut self,
        name: &str,
        steps: u32,
        origin_x: u32,
        origin_y: u32,
        max_steps: u32,
    ) -> Result<OkDTO, ApiError>;
    fn exec_record_organism(
        &mut self,
        name: &str,
        steps: u32,
        origin_x: u32,
        origin_y: u32,
        dir: &Path,
        max_steps: u32,
    ) -> Result<RecordManifestDTO, ApiError>;
    fn exec_replay(&self, dir: &Path) -> Result<ReplayResultDTO, ApiError>;
    fn exec_diff(&self, dir1: &Path, dir2: &Path) -> Result<DiffResultDTO, ApiError>;
    fn exec_run_ecosystem(
        &mut self,
        cfg: &ecosystem::EcosystemConfig,
        max_steps: u32,
    ) -> Result<EcosystemSummaryDTO, ApiError>;
}

pub struct Engine {
    sim: Simulation,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        Self {
            sim: Simulation::new(),
        }
    }
}

fn field_to_string(f: FrameFieldKind) -> String {
    match f {
        FrameFieldKind::Heat => "heat".into(),
    }
}

impl DeterministicEngine for Engine {
    fn api_root(&self) -> ApiRoot {
        ApiRoot {
            version: API_VERSION.into(),
            contract_hash: contract_hash(),
        }
    }

    fn read_run_info(&self, dir: &Path) -> Result<GuiRunInfoDTO, ApiError> {
        let info = gui_api::load_run(dir).map_err(|e| ApiError::new("io", &format!("{}", e)))?;
        Ok(GuiRunInfoDTO {
            frames: info.frame_count,
            width: info.width,
            height: info.height,
            fields: info.fields.iter().map(|f| field_to_string(*f)).collect(),
        })
    }

    fn read_run_frame(
        &self,
        dir: &Path,
        index: usize,
        max_bytes: usize,
    ) -> Result<FrameDTO, ApiError> {
        let frame =
            gui_api::load_frame(dir, index).map_err(|e| ApiError::new("io", &format!("{}", e)))?;
        if frame.bytes.len() > max_bytes {
            return Err(ApiError::new("limit", "frame too large"));
        }
        let meta = FrameMetaDTO {
            index: frame.meta.index,
            t: frame.meta.t,
            width: frame.meta.width,
            height: frame.meta.height,
            field: field_to_string(frame.meta.field),
            checksum: frame.meta.checksum,
        };
        let b64 = base64_like(&frame.bytes);
        Ok(FrameDTO {
            meta,
            ppm_base64: b64,
        })
    }

    fn exec_place_organism(
        &mut self,
        name: &str,
        x: u32,
        y: u32,
        allow_overwrite: bool,
    ) -> Result<OkDTO, ApiError> {
        let kind = match name.to_ascii_lowercase().as_str() {
            "heat_emitter" => Some(OrganismKind::HeatEmitter),
            "ring_cluster" => Some(OrganismKind::RingCluster),
            "charge_crawler" => Some(OrganismKind::ChargeCrawler),
            "wire_bundle" => Some(OrganismKind::WireBundle),
            "oscillator_nest" => Some(OrganismKind::OscillatorNest),
            _ => None,
        }
        .ok_or_else(|| ApiError::new("bad_request", "unknown organism"))?;
        crate::organisms::place_organism(&mut self.sim, kind, x, y, allow_overwrite)
            .map_err(|_| ApiError::new("exec", "place failed"))?;
        Ok(OkDTO { ok: true })
    }

    fn exec_run_organism(
        &mut self,
        name: &str,
        steps: u32,
        origin_x: u32,
        origin_y: u32,
        max_steps: u32,
    ) -> Result<OkDTO, ApiError> {
        if steps > max_steps {
            return Err(ApiError::new("limit", "steps too large"));
        }
        let kind = match name.to_ascii_lowercase().as_str() {
            "heat_emitter" => Some(OrganismKind::HeatEmitter),
            "ring_cluster" => Some(OrganismKind::RingCluster),
            "charge_crawler" => Some(OrganismKind::ChargeCrawler),
            "wire_bundle" => Some(OrganismKind::WireBundle),
            "oscillator_nest" => Some(OrganismKind::OscillatorNest),
            _ => None,
        }
        .ok_or_else(|| ApiError::new("bad_request", "unknown organism"))?;
        let _ = crate::organisms::run_organism(
            &mut self.sim,
            kind,
            origin_x,
            origin_y,
            &crate::organisms::OrganismParams {
                steps,
                render_each: false,
            },
        );
        Ok(OkDTO { ok: true })
    }

    fn exec_record_organism(
        &mut self,
        name: &str,
        steps: u32,
        origin_x: u32,
        origin_y: u32,
        dir: &Path,
        max_steps: u32,
    ) -> Result<RecordManifestDTO, ApiError> {
        if steps > max_steps {
            return Err(ApiError::new("limit", "steps too large"));
        }
        let kind = match name.to_ascii_lowercase().as_str() {
            "heat_emitter" => Some(OrganismKind::HeatEmitter),
            "ring_cluster" => Some(OrganismKind::RingCluster),
            "charge_crawler" => Some(OrganismKind::ChargeCrawler),
            "wire_bundle" => Some(OrganismKind::WireBundle),
            "oscillator_nest" => Some(OrganismKind::OscillatorNest),
            _ => None,
        }
        .ok_or_else(|| ApiError::new("bad_request", "unknown organism"))?;
        // Ensure placement once
        let _ = crate::organisms::place_organism(&mut self.sim, kind, origin_x, origin_y, false);
        let man = recorder::record_organism(&mut self.sim, kind, origin_x, origin_y, steps, dir)
            .map_err(|e| ApiError::new("exec", &format!("{}", e)))?;
        Ok(RecordManifestDTO {
            run_kind: man.run_kind,
            name: man.name,
            steps: man.steps,
        })
    }

    fn exec_replay(&self, dir: &Path) -> Result<ReplayResultDTO, ApiError> {
        let ok = recorder::replay_run(dir).map_err(|e| ApiError::new("exec", &format!("{}", e)))?;
        Ok(ReplayResultDTO { ok })
    }

    fn exec_diff(&self, dir1: &Path, dir2: &Path) -> Result<DiffResultDTO, ApiError> {
        let (m, len) = recorder::diff_runs(dir1, dir2)
            .map_err(|e| ApiError::new("exec", &format!("{}", e)))?;
        Ok(DiffResultDTO {
            mismatches: m,
            length_diff: len,
        })
    }

    fn exec_run_ecosystem(
        &mut self,
        cfg: &ecosystem::EcosystemConfig,
        max_steps: u32,
    ) -> Result<EcosystemSummaryDTO, ApiError> {
        if cfg.steps > max_steps {
            return Err(ApiError::new("limit", "steps too large"));
        }
        let res = ecosystem::run_ecosystem(&mut self.sim, cfg);
        Ok(EcosystemSummaryDTO {
            steps_run: res.steps_run,
        })
    }
}

fn base64_like(bytes: &[u8]) -> String {
    // Minimal URL-safe encoder sufficient for tests; not a full base64
    const ALPH: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        let b2 = bytes[i + 2];
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);
        out.push(ALPH[((n >> 18) & 63) as usize] as char);
        out.push(ALPH[((n >> 12) & 63) as usize] as char);
        out.push(ALPH[((n >> 6) & 63) as usize] as char);
        out.push(ALPH[(n & 63) as usize] as char);
        i += 3;
    }
    if i < bytes.len() {
        let rem = &bytes[i..];
        let n = if rem.len() == 1 {
            (rem[0] as u32) << 16
        } else {
            ((rem[0] as u32) << 16) | ((rem[1] as u32) << 8)
        };
        out.push(ALPH[((n >> 18) & 63) as usize] as char);
        out.push(ALPH[((n >> 12) & 63) as usize] as char);
        out.push(ALPH[((n >> 6) & 63) as usize] as char);
        out.push('_'); // padding indicator
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn api_root_has_version_and_hash() {
        let eng = Engine::new();
        let root = eng.api_root();
        assert_eq!(root.version, API_VERSION);
        assert_eq!(root.contract_hash.len(), 8);
    }

    #[test]
    fn read_and_frame_dto_roundtrip() {
        let out = Path::new("target/test_output/api_rec_demo");
        let _ = std::fs::create_dir_all("target/test_output");
        let _ = std::fs::remove_dir_all(out);
        let mut eng = Engine::new();
        let _ =
            crate::organisms::place_organism(&mut eng.sim, OrganismKind::HeatEmitter, 3, 3, false);
        let _ = recorder::record_organism(&mut eng.sim, OrganismKind::HeatEmitter, 3, 3, 1, out)
            .expect("record");
        let info = eng.read_run_info(out).expect("info");
        assert!(info.frames >= 1);
        let frame = eng.read_run_frame(out, 0, 2_000_000).expect("frame");
        assert!(frame.ppm_base64.len() > 4);
        assert_eq!(frame.meta.index, 0);
    }

    #[test]
    fn exec_run_and_replay() {
        let out = Path::new("target/test_output/api_rec_exec");
        let _ = std::fs::create_dir_all("target/test_output");
        let _ = std::fs::remove_dir_all(out);
        let mut eng = Engine::new();
        eng.exec_place_organism("heat_emitter", 2, 2, false)
            .expect("place");
        let man = eng
            .exec_record_organism("heat_emitter", 2, 2, 2, out, 10)
            .expect("rec");
        assert_eq!(man.run_kind, "organism");
        let rep = eng.exec_replay(out).expect("replay");
        assert!(rep.ok);
        let diff = eng.exec_diff(out, out).expect("diff");
        assert_eq!(diff.mismatches, 0);
    }
}
