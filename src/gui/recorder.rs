use crate::organisms::{self, OrganismKind};
use crate::render::{self, FieldLayerMode, ImageFormat, RenderParams};
use crate::simulation::Simulation;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct RecordArtifact {
    pub kind: String,
    pub path: String,
    pub checksum_hex: String,
}

#[derive(Debug, Clone)]
pub struct RecordManifest {
    pub run_kind: String,
    pub name: String,
    pub steps: u32,
    pub origin_x: u32,
    pub origin_y: u32,
    pub artifacts: Vec<RecordArtifact>,
}

fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811C9DC5;
    for &b in bytes {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

fn checksum_file_hex(path: &Path) -> io::Result<String> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(format!("{:08x}", fnv1a32(&buf)))
}

fn write_manifest(dir: &Path, man: &RecordManifest) -> io::Result<()> {
    let mut f = File::create(dir.join("manifest.txt"))?;
    writeln!(
        f,
        "RUN kind={} name={} steps={} origin=({}, {})",
        man.run_kind, man.name, man.steps, man.origin_x, man.origin_y
    )?;
    for a in &man.artifacts {
        writeln!(
            f,
            "ART kind={} path={} checksum={}",
            a.kind, a.path, a.checksum_hex
        )?;
    }
    Ok(())
}

pub fn read_manifest(dir: &Path) -> io::Result<RecordManifest> {
    let mut f = File::open(dir.join("manifest.txt"))?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    let mut run_kind = String::new();
    let mut name = String::new();
    let mut steps = 0u32;
    let mut origin_x = 0u32;
    let mut origin_y = 0u32;
    let mut artifacts = Vec::new();
    for line in s.lines() {
        if line.starts_with("RUN ") {
            // simple parse
            // RUN kind=... name=... steps=... origin=(x, y)
            let parts: Vec<_> = line.split_whitespace().collect();
            for p in parts.into_iter().skip(1) {
                if let Some(v) = p.strip_prefix("kind=") {
                    run_kind = v.to_string();
                }
                if let Some(v) = p.strip_prefix("name=") {
                    name = v.to_string();
                }
                if let Some(v) = p.strip_prefix("steps=") {
                    steps = v.parse().unwrap_or(0);
                }
                if let Some(v) = p.strip_prefix("origin=(") {
                    let xy: Vec<_> = v.trim_end_matches(")").split(',').collect();
                    if xy.len() == 2 {
                        origin_x = xy[0].parse().unwrap_or(0);
                        origin_y = xy[1].parse().unwrap_or(0);
                    }
                }
            }
        } else if line.starts_with("ART ") {
            // ART kind=... path=... checksum=...
            let mut kind = String::new();
            let mut path = String::new();
            let mut checksum_hex = String::new();
            for p in line.split_whitespace().skip(1) {
                if let Some(v) = p.strip_prefix("kind=") {
                    kind = v.to_string();
                }
                if let Some(v) = p.strip_prefix("path=") {
                    path = v.to_string();
                }
                if let Some(v) = p.strip_prefix("checksum=") {
                    checksum_hex = v.to_string();
                }
            }
            artifacts.push(RecordArtifact {
                kind,
                path,
                checksum_hex,
            });
        }
    }
    Ok(RecordManifest {
        run_kind,
        name,
        steps,
        origin_x,
        origin_y,
        artifacts,
    })
}

fn write_heat_snapshot(dir: &Path, name: &str, data: &[Vec<u32>]) -> io::Result<PathBuf> {
    let path = dir.join(name);
    let mut f = File::create(&path)?;
    writeln!(f, "heat snapshot:")?;
    for row in data {
        for v in row {
            write!(f, "{:08x} ", v)?;
        }
        writeln!(f)?;
    }
    Ok(path)
}

pub fn record_organism(
    sim: &mut Simulation,
    kind: OrganismKind,
    origin_x: u32,
    origin_y: u32,
    steps: u32,
    out_dir: &Path,
) -> io::Result<RecordManifest> {
    fs::create_dir_all(out_dir)?;
    let tmpl = organisms::list_organisms()
        .iter()
        .find(|t| t.kind == kind)
        .expect("template");
    let mut artifacts: Vec<RecordArtifact> = Vec::new();
    let params = crate::organisms::OrganismParams {
        steps: 1,
        render_each: false,
    };
    for i in 0..steps {
        let _res = crate::organisms::run_organism(sim, kind, origin_x, origin_y, &params);
        // Render heat frame
        let rparams = RenderParams {
            fields: FieldLayerMode::Heat,
            scale: 2,
            format: ImageFormat::Ppm,
        };
        let img = render::render_region(
            sim,
            origin_x,
            origin_y,
            tmpl.width as u32,
            tmpl.height as u32,
            &rparams,
        )
        .map_err(|_| io::Error::new(io::ErrorKind::Other, "render error"))?;
        let frame_name = format!("frame_{:04}.ppm", i);
        let frame_path = out_dir.join(&frame_name);
        {
            let mut f = File::create(&frame_path)?;
            render::write_ppm(&img, &mut f)?;
        }
        let csum = checksum_file_hex(&frame_path)?;
        artifacts.push(RecordArtifact {
            kind: "frame".to_string(),
            path: frame_name.clone(),
            checksum_hex: csum,
        });

        // Heat snapshot
        let snap =
            sim.snapshot_heat_region(origin_x, origin_y, tmpl.width as u32, tmpl.height as u32);
        let snap_name = format!("heat_{:04}.txt", i);
        let snap_path = write_heat_snapshot(out_dir, &snap_name, &snap)?;
        let csum2 = checksum_file_hex(&snap_path)?;
        artifacts.push(RecordArtifact {
            kind: "heat".to_string(),
            path: snap_name.clone(),
            checksum_hex: csum2,
        });
    }
    let man = RecordManifest {
        run_kind: "organism".to_string(),
        name: tmpl.name.to_string(),
        steps,
        origin_x,
        origin_y,
        artifacts,
    };
    write_manifest(out_dir, &man)?;
    Ok(man)
}

pub fn replay_run(dir: &Path) -> io::Result<bool> {
    let man = read_manifest(dir)?;
    let mut ok_all = true;
    for a in &man.artifacts {
        let p = dir.join(&a.path);
        let c = checksum_file_hex(&p)?;
        if c != a.checksum_hex {
            ok_all = false;
            break;
        }
    }
    Ok(ok_all)
}

pub fn record_region_with<F: FnMut(&mut Simulation, u32)>(
    sim: &mut Simulation,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    steps: u32,
    out_dir: &Path,
    mut step_hook: F,
    run_name: &str,
) -> io::Result<RecordManifest> {
    fs::create_dir_all(out_dir)?;
    let mut artifacts: Vec<RecordArtifact> = Vec::new();
    let rparams = RenderParams {
        fields: FieldLayerMode::Heat,
        scale: 2,
        format: ImageFormat::Ppm,
    };
    for i in 0..steps {
        sim.tick();
        step_hook(sim, i);
        let img = render::render_region(sim, x0, y0, w, h, &rparams)
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "render error"))?;
        let frame_name = format!("frame_{:04}.ppm", i);
        let frame_path = out_dir.join(&frame_name);
        {
            let mut f = File::create(&frame_path)?;
            render::write_ppm(&img, &mut f)?;
        }
        let csum = checksum_file_hex(&frame_path)?;
        artifacts.push(RecordArtifact {
            kind: "frame".to_string(),
            path: frame_name.clone(),
            checksum_hex: csum,
        });

        // Heat snapshot
        let snap = sim.snapshot_heat_region(x0, y0, w, h);
        let snap_name = format!("heat_{:04}.txt", i);
        let snap_path = write_heat_snapshot(out_dir, &snap_name, &snap)?;
        let csum2 = checksum_file_hex(&snap_path)?;
        artifacts.push(RecordArtifact {
            kind: "heat".to_string(),
            path: snap_name.clone(),
            checksum_hex: csum2,
        });
    }
    let man = RecordManifest {
        run_kind: "demo".to_string(),
        name: run_name.to_string(),
        steps,
        origin_x: x0,
        origin_y: y0,
        artifacts,
    };
    write_manifest(out_dir, &man)?;
    Ok(man)
}

pub fn diff_runs(dir1: &Path, dir2: &Path) -> io::Result<(usize, usize)> {
    let m1 = read_manifest(dir1)?;
    let m2 = read_manifest(dir2)?;
    let len = m1.artifacts.len().min(m2.artifacts.len());
    let mut mismatches = 0usize;
    for i in 0..len {
        let a1 = &m1.artifacts[i];
        let a2 = &m2.artifacts[i];
        if a1.kind != a2.kind {
            mismatches += 1;
            continue;
        }
        let p1 = dir1.join(&a1.path);
        let p2 = dir2.join(&a2.path);
        let c1 = checksum_file_hex(&p1)?;
        let c2 = checksum_file_hex(&p2)?;
        if c1 != c2 {
            mismatches += 1;
        }
    }
    let len1 = m1.artifacts.len();
    let len2 = m2.artifacts.len();
    let len_diff = if len1 >= len2 {
        len1 - len2
    } else {
        len2 - len1
    };
    Ok((mismatches, len_diff))
}
