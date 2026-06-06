use crate::gui_backend::{
    FrameFieldKind, GuiFrameStream, open_recorded_frames, run_organism_frames,
};
use crate::recorder::{RecordManifest, read_manifest};
use crate::simulation::Simulation;
use std::fs::File;
use std::io::Read;
use std::ops::Range;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct GuiRunInfo {
    pub manifest: RecordManifest,
    pub frame_count: usize,
    pub width: u32,
    pub height: u32,
    pub fields: Vec<FrameFieldKind>,
}

#[derive(Debug, Clone)]
pub struct GuiFrame {
    pub meta: crate::gui_backend::FrameMeta,
    pub bytes: Vec<u8>, // PPM bytes
}

fn parse_ppm_header(bytes: &[u8]) -> Option<(u32, u32, usize)> {
    let s = std::str::from_utf8(bytes).ok()?;
    let mut it = s.splitn(4, '\n');
    if it.next()?.trim() != "P6" {
        return None;
    }
    let dims = it.next()?;
    if it.next()?.trim() != "255" {
        return None;
    }
    let _rest = it.next();
    let mut parts = dims.split_whitespace();
    let w: u32 = parts.next()?.parse().ok()?;
    let h: u32 = parts.next()?.parse().ok()?;
    let mut idx = 0usize;
    let mut cnt = 0u32;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'\n' {
            cnt += 1;
            if cnt == 3 {
                idx = i + 1;
                break;
            }
        }
    }
    if idx == 0 {
        return None;
    }
    Some((w, h, idx))
}

pub fn load_run(dir: &Path) -> Result<GuiRunInfo, std::io::Error> {
    let man = read_manifest(dir)?;
    // count frames and parse first frame dim
    let mut count = 0usize;
    let mut width = 0u32;
    let mut height = 0u32;
    for art in &man.artifacts {
        if art.kind == "frame" {
            if count == 0 {
                let mut f = File::open(dir.join(&art.path))?;
                let mut buf = Vec::new();
                f.read_to_end(&mut buf)?;
                let (w, h, _) = parse_ppm_header(&buf)
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "bad ppm"))?;
                width = w;
                height = h;
            }
            count += 1;
        }
    }
    Ok(GuiRunInfo {
        manifest: man,
        frame_count: count,
        width,
        height,
        fields: vec![FrameFieldKind::Heat],
    })
}

pub fn load_frame(dir: &Path, index: usize) -> Result<GuiFrame, std::io::Error> {
    let stream = open_recorded_frames(dir)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "open frames"))?;
    let meta = stream
        .meta(index as u32)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "index"))?
        .clone();
    let bytes = stream
        .frame_bytes(index as u32)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "bytes"))?
        .to_vec();
    Ok(GuiFrame { meta, bytes })
}

pub fn load_frame_range(dir: &Path, range: Range<usize>) -> Result<Vec<GuiFrame>, std::io::Error> {
    let stream = open_recorded_frames(dir)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "open frames"))?;
    let mut out = Vec::new();
    for i in range {
        if let (Some(m), Some(b)) = (stream.meta(i as u32), stream.frame_bytes(i as u32)) {
            out.push(GuiFrame {
                meta: *m,
                bytes: b.to_vec(),
            });
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct GuiStream {
    pub info: GuiRunInfo,
    pub frames: GuiFrameStream,
}

pub fn stream_from_recorded(dir: &Path) -> Result<GuiStream, std::io::Error> {
    let info = load_run(dir)?;
    let frames = open_recorded_frames(dir)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "open frames"))?;
    Ok(GuiStream { info, frames })
}

pub fn preview_organism(
    sim: &mut Simulation,
    name: &str,
    steps: u32,
    origin_x: u32,
    origin_y: u32,
) -> Result<GuiStream, std::io::Error> {
    let frames = run_organism_frames(sim, name, origin_x, origin_y, steps)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "preview"))?;
    // derive info from first frame
    let meta0 = frames
        .meta(0)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "empty"))?;
    let man = read_manifest(Path::new(".")) // dummy manifest for preview, minimal
        .unwrap_or_else(|_| crate::recorder::RecordManifest {
            run_kind: "preview".into(),
            name: name.into(),
            steps,
            origin_x,
            origin_y,
            artifacts: vec![],
        });
    let info = GuiRunInfo {
        manifest: man,
        frame_count: frames.len(),
        width: meta0.width,
        height: meta0.height,
        fields: vec![FrameFieldKind::Heat],
    };
    Ok(GuiStream { info, frames })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn load_and_frame_range() {
        let out = Path::new("gui_api_rec");
        let _ = std::fs::remove_dir_all(out);
        let mut sim = Simulation::new();
        let _ = crate::organisms::place_organism(
            &mut sim,
            crate::organisms::OrganismKind::HeatEmitter,
            2,
            2,
            false,
        );
        let _ = crate::recorder::record_organism(
            &mut sim,
            crate::organisms::OrganismKind::HeatEmitter,
            2,
            2,
            3,
            out,
        )
        .expect("record");
        let info = load_run(out).expect("info");
        assert_eq!(info.frame_count, 3);
        let f0 = load_frame(out, 0).expect("frame0");
        assert!(f0.bytes.starts_with(b"P6\n"));
        let fr = load_frame_range(out, 1..3).expect("range");
        assert_eq!(fr.len(), 2);
    }
}
