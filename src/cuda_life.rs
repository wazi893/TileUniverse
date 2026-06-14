//! Conway's Game of Life on the engine's packed 1-bit tile grid.
//!
//! This runs B3/S23 directly on the same [`PackedTileGrid`](crate::cuda_tiles::PackedTileGrid)
//! representation the cellular engine uses for its 115T tiles/sec packed kernels:
//! **64 cells per `u64` word**, row-major. The GPU kernel evaluates one word
//! (64 cells) per thread in pure register bit-ops — neighbour counting via
//! bit-sliced carry-save adders — so it is the same "64 lanes per instruction"
//! technique the flagship logic-gate kernels use, specialised to Conway's rule.
//!
//! Topology is toroidal (wrap-around). Grid width must be a multiple of 64 so
//! the horizontal wrap lands on a word boundary.
//!
//! A scalar CPU reference ([`life_step_cpu`]) computes the identical rule and is
//! used to oracle-verify the GPU kernel ([`verify_gpu_matches_cpu`]). The CPU
//! path also drives the visualization's embedded frame capture, while the GPU
//! path supplies the real "billions of cell-updates/sec on the device" headline.

use crate::cuda_tiles::PackedTileGrid;

#[cfg(feature = "cuda")]
use crate::cuda_tiles::GpuPackedGrid;
#[cfg(feature = "cuda")]
use logic_fabric_core::cuda::{CudaError, CudaResult, CudaRuntime};

// ── Patterns ──────────────────────────────────────────────────────────────

/// Empty grid with width rounded as-is (caller ensures width % 64 == 0).
fn blank(width: usize, height: usize) -> PackedTileGrid {
    PackedTileGrid::new(width, height)
}

/// Stamp the canonical Gosper glider gun with its top-left at (ox, oy).
/// Emits a glider every 30 generations toward the bottom-right.
pub fn stamp_gosper_gun(grid: &mut PackedTileGrid, ox: usize, oy: usize) {
    const CELLS: &[(usize, usize)] = &[
        (0, 4), (0, 5), (1, 4), (1, 5),
        (10, 4), (10, 5), (10, 6), (11, 3), (11, 7), (12, 2), (12, 8), (13, 2),
        (13, 8), (14, 5), (15, 3), (15, 7), (16, 4), (16, 5), (16, 6), (17, 5),
        (20, 2), (20, 3), (20, 4), (21, 2), (21, 3), (21, 4), (22, 1), (22, 5),
        (24, 0), (24, 1), (24, 5), (24, 6),
        (34, 2), (34, 3), (35, 2), (35, 3),
    ];
    for &(x, y) in CELLS {
        grid.set(ox + x, oy + y, true);
    }
}

/// A single glider with its top-left at (ox, oy), travelling down-right.
pub fn stamp_glider(grid: &mut PackedTileGrid, ox: usize, oy: usize) {
    for &(x, y) in &[(1usize, 0usize), (2, 1), (0, 2), (1, 2), (2, 2)] {
        grid.set(ox + x, oy + y, true);
    }
}

/// A glider-gun grid: one gun in the upper-left streaming gliders across.
pub fn pattern_glider_gun(width: usize, height: usize) -> PackedTileGrid {
    let mut g = blank(width, height);
    stamp_gosper_gun(&mut g, 4, 4);
    g
}

/// Several guns whose glider streams crisscross and collide across the grid.
pub fn pattern_multi_gun(width: usize, height: usize) -> PackedTileGrid {
    let mut g = blank(width, height);
    for &(ox, oy) in &[(4usize, 4usize), (150, 28), (24, 150), (158, 162)] {
        if ox + 36 < width && oy + 9 < height {
            stamp_gosper_gun(&mut g, ox, oy);
        }
    }
    g
}

/// Random "primordial soup" at the given live-fraction (out of 256), deterministic.
pub fn pattern_soup(width: usize, height: usize, seed: u64, fill_num: u32) -> PackedTileGrid {
    let mut g = blank(width, height);
    let mut s = seed | 1;
    for y in 0..height {
        for x in 0..width {
            // xorshift64
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if (s & 0xff) < fill_num as u64 {
                g.set(x, y, true);
            }
        }
    }
    g
}

/// A grid of independently-seeded soup "colonies" laid out cols×rows with a
/// dead gutter between them — distinct seeds give visually distinct worlds.
pub fn pattern_colonies(
    width: usize,
    height: usize,
    cols: usize,
    rows: usize,
    gutter: usize,
    seed: u64,
) -> PackedTileGrid {
    let mut g = blank(width, height);
    let cell_w = (width - gutter * (cols + 1)) / cols;
    let cell_h = (height - gutter * (rows + 1)) / rows;
    for cy in 0..rows {
        for cx in 0..cols {
            let ox = gutter + cx * (cell_w + gutter);
            let oy = gutter + cy * (cell_h + gutter);
            let mut s = seed
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add((cy * cols + cx) as u64 * 0xD1B54A32D192ED03)
                | 1;
            for y in 0..cell_h {
                for x in 0..cell_w {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    if (s & 0xff) < 90 {
                        g.set(ox + x, oy + y, true);
                    }
                }
            }
        }
    }
    g
}

// ── CPU reference (oracle + frame capture) ─────────────────────────────────

/// One toroidal B3/S23 step. Authoritative reference for the GPU kernel.
pub fn life_step_cpu(g: &PackedTileGrid) -> PackedTileGrid {
    let w = g.width;
    let h = g.height;
    let mut out = PackedTileGrid::new(w, h);
    for y in 0..h {
        let up = if y == 0 { h - 1 } else { y - 1 };
        let dn = if y == h - 1 { 0 } else { y + 1 };
        for x in 0..w {
            let lx = if x == 0 { w - 1 } else { x - 1 };
            let rx = if x == w - 1 { 0 } else { x + 1 };
            let n = g.get(lx, up) as u32
                + g.get(x, up) as u32
                + g.get(rx, up) as u32
                + g.get(lx, y) as u32
                + g.get(rx, y) as u32
                + g.get(lx, dn) as u32
                + g.get(x, dn) as u32
                + g.get(rx, dn) as u32;
            let alive = g.get(x, y);
            if n == 3 || (alive && n == 2) {
                out.set(x, y, true);
            }
        }
    }
    out
}

/// A captured Life run: packed frames (sampled) plus per-frame population.
#[derive(Debug, Clone)]
pub struct LifeFrameRun {
    pub width: usize,
    pub height: usize,
    pub words_per_row: usize,
    pub generations: u32,
    pub sample_every: u32,
    /// Packed bit data per sampled frame (`words_per_row * height` u64 each).
    pub frames: Vec<Vec<u64>>,
    /// Generation index of each sampled frame.
    pub frame_gens: Vec<u32>,
    /// Live-cell count per sampled frame.
    pub populations: Vec<u32>,
}

/// Evolve `seed` on the CPU for `generations`, sampling a frame every
/// `sample_every` generations (frame 0 = the seed itself).
pub fn capture_life_cpu(
    seed: PackedTileGrid,
    generations: u32,
    sample_every: u32,
) -> LifeFrameRun {
    let sample_every = sample_every.max(1);
    let width = seed.width;
    let height = seed.height;
    let words_per_row = seed.words_per_row;

    let mut frames = Vec::new();
    let mut frame_gens = Vec::new();
    let mut populations = Vec::new();

    let mut cur = seed;
    for gi in 0..=generations {
        if gi % sample_every == 0 || gi == generations {
            frames.push(cur.data.clone());
            frame_gens.push(gi);
            populations.push(cur.count_active() as u32);
        }
        if gi < generations {
            cur = life_step_cpu(&cur);
        }
    }

    LifeFrameRun {
        width,
        height,
        words_per_row,
        generations,
        sample_every,
        frames,
        frame_gens,
        populations,
    }
}

// ── GPU kernel ─────────────────────────────────────────────────────────────

/// Bit-parallel packed Game of Life: one `u64` word (64 cells) per thread.
/// Neighbour counts are formed with carry-save adders into 4 bit-planes
/// (b0..b3), then B3/S23 is applied with pure bitwise ops. Toroidal wrap.
#[cfg(feature = "cuda")]
const PACKED_LIFE_KERNEL_SRC: &str = r#"
extern "C" __global__ void packed_life_kernel(
    const unsigned long long* __restrict__ in,
    unsigned long long* __restrict__ out,
    int words_per_row,
    int height)
{
    int wc  = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (wc >= words_per_row || row >= height) return;

    int up = (row == 0) ? (height - 1) : (row - 1);
    int dn = (row == height - 1) ? 0 : (row + 1);
    int lc = (wc == 0) ? (words_per_row - 1) : (wc - 1);
    int rc = (wc == words_per_row - 1) ? 0 : (wc + 1);

    unsigned long long uc = in[up  * words_per_row + wc];
    unsigned long long ul = in[up  * words_per_row + lc];
    unsigned long long ur = in[up  * words_per_row + rc];
    unsigned long long mc = in[row * words_per_row + wc];
    unsigned long long ml = in[row * words_per_row + lc];
    unsigned long long mr = in[row * words_per_row + rc];
    unsigned long long dc = in[dn  * words_per_row + wc];
    unsigned long long dl = in[dn  * words_per_row + lc];
    unsigned long long dr = in[dn  * words_per_row + rc];

    // 8 neighbour bit-planes (bit i = that neighbour of cell i)
    unsigned long long a = (uc << 1) | (ul >> 63); // up-left
    unsigned long long b =  uc;                     // up
    unsigned long long c = (uc >> 1) | (ur << 63); // up-right
    unsigned long long d = (mc << 1) | (ml >> 63); // left
    unsigned long long e = (mc >> 1) | (mr << 63); // right
    unsigned long long f = (dc << 1) | (dl >> 63); // down-left
    unsigned long long g =  dc;                     // down
    unsigned long long h = (dc >> 1) | (dr << 63); // down-right

    // bit-sliced popcount of a..h -> bit0..bit3
    unsigned long long s1 = a ^ b ^ c, c1 = (a & b) | (a & c) | (b & c);
    unsigned long long s2 = d ^ e ^ f, c2 = (d & e) | (d & f) | (e & f);
    unsigned long long s3 = g ^ h,     c3 = (g & h);

    unsigned long long bit0 = s1 ^ s2 ^ s3;
    unsigned long long cA   = (s1 & s2) | (s1 & s3) | (s2 & s3);

    unsigned long long t = c1 ^ c2 ^ c3;
    unsigned long long u = (c1 & c2) | (c1 & c3) | (c2 & c3);
    unsigned long long bit1 = t ^ cA;
    unsigned long long v = t & cA;

    unsigned long long bit2 = u ^ v;
    unsigned long long bit3 = u & v;

    // count in {2,3}  &  (count==3 || alive)
    unsigned long long inrange = bit1 & ~bit2 & ~bit3;
    out[row * words_per_row + wc] = inrange & (bit0 | mc);
}
"#;

#[cfg(feature = "cuda")]
struct LifeKernel {
    func: cudarc::driver::CudaFunction,
}

#[cfg(feature = "cuda")]
fn life_kernel(rt: &CudaRuntime) -> CudaResult<&'static cudarc::driver::CudaFunction> {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    static CACHE: std::sync::OnceLock<LifeKernel> = std::sync::OnceLock::new();
    if let Some(k) = CACHE.get() {
        return Ok(&k.func);
    }
    let opts = CompileOptions {
        arch: Some("compute_89"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(PACKED_LIFE_KERNEL_SRC, opts)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("packed_life: {e:?}")))?;
    let module = rt
        .ctx()
        .load_module(ptx)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("module load: {e:?}")))?;
    let func = module
        .load_function("packed_life_kernel")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("fn load: {e:?}")))?;
    // First writer wins; ignore the race loser.
    let _ = CACHE.set(LifeKernel { func });
    Ok(&CACHE.get().unwrap().func)
}

/// Advance the packed grid `generations` steps on the GPU (ping-pong buffers).
/// On return the current state is in `grid.grid_a`.
#[cfg(feature = "cuda")]
pub fn run_packed_life_gpu(
    rt: &CudaRuntime,
    grid: &mut GpuPackedGrid,
    generations: u32,
) -> CudaResult<()> {
    use cudarc::driver::PushKernelArg;
    let func = life_kernel(rt)?;
    let words_per_row = grid.words_per_row as i32;
    let height = grid.height as i32;

    let block = (32u32, 8u32, 1u32);
    let grid_dim = (
        (grid.words_per_row as u32).div_ceil(block.0),
        (grid.height as u32).div_ceil(block.1),
        1,
    );
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    for _ in 0..generations {
        unsafe {
            rt.stream()
                .launch_builder(func)
                .arg(&grid.grid_a)
                .arg(&grid.grid_b)
                .arg(&words_per_row)
                .arg(&height)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("packed_life: {e:?}")))?;
        }
        grid.swap_buffers();
    }
    Ok(())
}

/// Result of a GPU throughput measurement.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct LifeThroughput {
    pub width: usize,
    pub height: usize,
    pub generations: u32,
    pub elapsed_secs: f64,
    pub cell_updates: u64,
    pub cell_updates_per_sec: f64,
}

/// Measure real Game-of-Life throughput on the device: a random-soup grid of
/// `width`×`height` advanced `generations` steps, timed kernel-only (one warm-up
/// generation, then a synchronised timed region — no host transfers inside).
#[cfg(feature = "cuda")]
pub fn measure_life_throughput_gpu(
    rt: &CudaRuntime,
    width: usize,
    height: usize,
    generations: u32,
    seed: u64,
) -> CudaResult<LifeThroughput> {
    use std::time::Instant;
    let cpu = pattern_soup(width, height, seed, 90);
    let mut g = GpuPackedGrid::from_cpu(rt, &cpu)?;

    run_packed_life_gpu(rt, &mut g, 1)?; // warm up (kernel compile + caches)
    rt.synchronize()?;

    let t0 = Instant::now();
    run_packed_life_gpu(rt, &mut g, generations)?;
    rt.synchronize()?;
    let elapsed_secs = t0.elapsed().as_secs_f64();

    let cell_updates = width as u64 * height as u64 * generations as u64;
    Ok(LifeThroughput {
        width,
        height,
        generations,
        elapsed_secs,
        cell_updates,
        cell_updates_per_sec: cell_updates as f64 / elapsed_secs.max(1e-12),
    })
}

/// Evolve `seed` on the **GPU**, downloading a packed frame every
/// `sample_every` generations (frame 0 = the seed). Every returned frame is
/// the kernel's own output, so the visualization replays real device state.
#[cfg(feature = "cuda")]
pub fn capture_life_gpu(
    rt: &CudaRuntime,
    seed: PackedTileGrid,
    generations: u32,
    sample_every: u32,
) -> CudaResult<LifeFrameRun> {
    let sample_every = sample_every.max(1);
    let width = seed.width;
    let height = seed.height;
    let words_per_row = seed.words_per_row;

    let mut g = GpuPackedGrid::from_cpu(rt, &seed)?;
    let mut frames = Vec::new();
    let mut frame_gens = Vec::new();
    let mut populations = Vec::new();

    let pop = |d: &[u64]| d.iter().map(|w| w.count_ones()).sum::<u32>();

    let d0 = rt.download(&g.grid_a)?;
    populations.push(pop(&d0));
    frames.push(d0);
    frame_gens.push(0);

    let mut done = 0u32;
    while done < generations {
        let chunk = sample_every.min(generations - done);
        run_packed_life_gpu(rt, &mut g, chunk)?;
        rt.synchronize()?;
        done += chunk;
        let d = rt.download(&g.grid_a)?;
        populations.push(pop(&d));
        frames.push(d);
        frame_gens.push(done);
    }

    Ok(LifeFrameRun {
        width,
        height,
        words_per_row,
        generations,
        sample_every,
        frames,
        frame_gens,
        populations,
    })
}

/// Oracle check: GPU kernel must match the CPU reference bit-for-bit after
/// `generations` steps on a seeded soup grid.
#[cfg(feature = "cuda")]
pub fn verify_gpu_matches_cpu(
    rt: &CudaRuntime,
    width: usize,
    height: usize,
    generations: u32,
    seed: u64,
) -> CudaResult<bool> {
    let seed_grid = pattern_soup(width, height, seed, 110);

    // CPU reference
    let mut cpu = seed_grid.clone();
    for _ in 0..generations {
        cpu = life_step_cpu(&cpu);
    }

    // GPU
    let mut g = GpuPackedGrid::from_cpu(rt, &seed_grid)?;
    run_packed_life_gpu(rt, &mut g, generations)?;
    rt.synchronize()?;
    let gpu_data = rt.download(&g.grid_a)?;

    Ok(gpu_data == cpu.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blinker_oscillates() {
        // Horizontal blinker -> vertical blinker -> horizontal (period 2).
        let mut g = PackedTileGrid::new(64, 64);
        g.set(10, 10, true);
        g.set(11, 10, true);
        g.set(12, 10, true);
        let g1 = life_step_cpu(&g);
        assert!(g1.get(11, 9) && g1.get(11, 10) && g1.get(11, 11));
        assert!(!g1.get(10, 10) && !g1.get(12, 10));
        let g2 = life_step_cpu(&g1);
        assert_eq!(g2.data, g.data);
    }

    #[test]
    fn test_block_is_still_life() {
        let mut g = PackedTileGrid::new(64, 64);
        for &(x, y) in &[(5, 5), (6, 5), (5, 6), (6, 6)] {
            g.set(x, y, true);
        }
        let g1 = life_step_cpu(&g);
        assert_eq!(g1.data, g.data);
    }

    #[test]
    fn test_glider_translates() {
        // A glider returns to its shape, offset by (1,1), after 4 generations.
        let mut g = PackedTileGrid::new(128, 128);
        stamp_glider(&mut g, 10, 10);
        let mut cur = g.clone();
        for _ in 0..4 {
            cur = life_step_cpu(&cur);
        }
        let mut expected = PackedTileGrid::new(128, 128);
        stamp_glider(&mut expected, 11, 11);
        assert_eq!(cur.data, expected.data);
    }

    #[test]
    fn test_gosper_gun_population_grows() {
        // The gun is not a still life; population should change as it fires.
        let g = pattern_glider_gun(256, 128);
        let p0 = g.count_active();
        let mut cur = g;
        for _ in 0..40 {
            cur = life_step_cpu(&cur);
        }
        assert!(cur.count_active() > p0);
    }

    #[test]
    fn test_capture_frames_count() {
        let run = capture_life_cpu(pattern_soup(128, 128, 7, 100), 20, 2);
        // gens 0,2,4,...,20 -> 11 sampled frames
        assert_eq!(run.frames.len(), 11);
        assert_eq!(run.frame_gens.first(), Some(&0));
        assert_eq!(run.frame_gens.last(), Some(&20));
        assert_eq!(run.populations.len(), run.frames.len());
    }
}
