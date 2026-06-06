use crate::organisms::OrganismKind;
use crate::recorder::read_manifest as read_record_manifest;
use crate::render::{self, FieldLayerMode, ImageFormat, RenderParams};
use crate::simulation::Simulation;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameFieldKind {
    Heat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameMeta {
    pub index: u32,
    pub t: u32,
    pub width: u32,
    pub height: u32,
    pub field: FrameFieldKind,
    pub checksum: u32,
}

#[derive(Debug, Clone)]
pub struct GuiFrameStream {
    metas: Vec<FrameMeta>,
    frames: Vec<Vec<u8>>, // PPM bytes
}

impl GuiFrameStream {
    pub fn len(&self) -> usize {
        self.metas.len()
    }
    pub fn meta(&self, index: u32) -> Option<&FrameMeta> {
        self.metas.get(index as usize)
    }
    pub fn frame_bytes(&self, index: u32) -> Option<&[u8]> {
        self.frames.get(index as usize).map(|v| v.as_slice())
    }
}

fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811C9DC5;
    for &b in bytes {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

fn parse_ppm_header(bytes: &[u8]) -> Option<(u32, u32, usize)> {
    // Expect ASCII header: P6\n<width> <height>\n255\n
    let s = match std::str::from_utf8(bytes) {
        Ok(v) => v,
        Err(_) => return None,
    };
    let mut it = s.splitn(4, '\n');
    let magic = it.next()?;
    if magic.trim() != "P6" {
        return None;
    }
    let dims = it.next()?;
    let maxval = it.next()?;
    let after = it.next()?;
    let mut parts = dims.split_whitespace();
    let w: u32 = parts.next()?.parse().ok()?;
    let h: u32 = parts.next()?.parse().ok()?;
    if maxval.trim() != "255" {
        return None;
    }
    // Find header length in bytes: up to and including third newline
    let mut idx = 0usize;
    let mut nl = 0u32;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'\n' {
            nl += 1;
            if nl == 3 {
                idx = i + 1;
                break;
            }
        }
    }
    if idx == 0 {
        return None;
    }
    let _ = after; // silence unused
    Some((w, h, idx))
}

#[derive(Debug)]
pub enum FrameError {
    Io(std::io::Error),
    InvalidManifest,
}

impl From<std::io::Error> for FrameError {
    fn from(e: std::io::Error) -> Self {
        FrameError::Io(e)
    }
}

pub fn open_recorded_frames(dir: &Path) -> Result<GuiFrameStream, FrameError> {
    let man = read_record_manifest(dir).map_err(|_| FrameError::InvalidManifest)?;
    // Collect artifacts of kind=frame in recorded order
    let mut metas = Vec::new();
    let mut frames = Vec::new();
    let mut idx: u32 = 0;
    for art in man.artifacts.iter().filter(|a| a.kind == "frame") {
        let p = dir.join(&art.path);
        let mut f = File::open(&p)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        // parse header to get width/height
        let (w, h, _off) = parse_ppm_header(&buf).ok_or(FrameError::InvalidManifest)?;
        let checksum = fnv1a32(&buf);
        metas.push(FrameMeta {
            index: idx,
            t: idx,
            width: w,
            height: h,
            field: FrameFieldKind::Heat,
            checksum,
        });
        frames.push(buf);
        idx = idx.saturating_add(1);
    }
    Ok(GuiFrameStream { metas, frames })
}

pub fn run_organism_frames(
    sim: &mut Simulation,
    organism: &str,
    origin_x: u32,
    origin_y: u32,
    steps: u32,
) -> Result<GuiFrameStream, FrameError> {
    let kind = match organism.to_ascii_lowercase().as_str() {
        "heat_emitter" => Some(OrganismKind::HeatEmitter),
        "ring_cluster" => Some(OrganismKind::RingCluster),
        "charge_crawler" => Some(OrganismKind::ChargeCrawler),
        "wire_bundle" => Some(OrganismKind::WireBundle),
        "oscillator_nest" => Some(OrganismKind::OscillatorNest),
        _ => None,
    }
    .ok_or(FrameError::InvalidManifest)?;

    // Place once
    let _ = crate::organisms::place_organism(sim, kind, origin_x, origin_y, false);
    let tmpl = crate::organisms::list_organisms()
        .iter()
        .find(|t| t.kind == kind)
        .unwrap();

    let mut metas = Vec::new();
    let mut frames = Vec::new();
    let p = RenderParams {
        fields: FieldLayerMode::Heat,
        scale: 2,
        format: ImageFormat::Ppm,
    };
    let single = crate::organisms::OrganismParams {
        steps: 1,
        render_each: false,
    };
    for i in 0..steps {
        let _ = crate::organisms::run_organism(sim, kind, origin_x, origin_y, &single);
        let img = render::render_region(
            sim,
            origin_x,
            origin_y,
            tmpl.width as u32,
            tmpl.height as u32,
            &p,
        )
        .map_err(|_| FrameError::InvalidManifest)?;
        let mut buf = Vec::new();
        render::write_ppm(&img, &mut buf).map_err(|e| FrameError::Io(e))?;
        let checksum = fnv1a32(&buf);
        metas.push(FrameMeta {
            index: i,
            t: i,
            width: img.width,
            height: img.height,
            field: FrameFieldKind::Heat,
            checksum,
        });
        frames.push(buf);
    }
    Ok(GuiFrameStream { metas, frames })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn open_recorded_frames_smoke() {
        let out = Path::new("target/test_output/gui_rec_smoke");
        let _ = std::fs::create_dir_all("target/test_output");
        let _ = std::fs::remove_dir_all(out);
        let mut sim = Simulation::new();
        let _ = crate::organisms::place_organism(&mut sim, OrganismKind::HeatEmitter, 5, 5, false);
        let _man =
            crate::recorder::record_organism(&mut sim, OrganismKind::HeatEmitter, 5, 5, 2, out)
                .expect("record");
        let stream = open_recorded_frames(out).expect("open");
        assert_eq!(stream.len(), 2);
        let m0 = stream.meta(0).unwrap();
        assert!(m0.width > 0 && m0.height > 0);
        let bytes = stream.frame_bytes(0).unwrap();
        assert!(bytes.starts_with(b"P6\n"));
    }
}
