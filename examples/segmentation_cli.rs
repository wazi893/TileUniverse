//! TileAnneal Phase 3: Image Segmentation CLI
//!
//! Command-line tool for QUBO-based image segmentation.
//!
//! Usage:
//!   cargo run --release --features cuda,tileanneal-cli --example segmentation_cli -- \
//!       --input image.png --output mask.png [OPTIONS]
//!
//! Options:
//!   --input <path>       Input image (PNG/JPG, will be converted to grayscale)
//!   --output <path>      Output segmentation mask (PNG)
//!   --seeds <path>       Optional seeds image (green=foreground, red=background)
//!   --lambda <float>     Coupling strength for smoothness (default: 2.0)
//!   --seed-strength <n>  Bias strength for seed pixels (default: 4)
//!   --edge-aware         Enable edge-aware couplings
//!   --edge-alpha <float> Edge sensitivity (default: 0.05)
//!   --sweeps <n>         Number of annealing sweeps (default: 100000)
//!   --beta-max <float>   Maximum inverse temperature (default: 20.0)
//!   --quiet              Suppress progress output
//!
//! Example:
//!   cargo run --release --features cuda,tileanneal-cli --example segmentation_cli -- \
//!       --input photo.png --output mask.png --seeds scribbles.png --edge-aware

#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
use engine::cuda::CudaRuntime;
#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
use engine::cuda_tiles::{
    AdaptiveMode, AdaptiveSchedule, CouplingMode, GpuIsingGrid, IsingConfig,
    compute_weighted_energy, run_adaptive_annealing, run_ising_update_weighted,
};
#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
use image::{GrayImage, Luma};
#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
use std::path::PathBuf;
#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
#[derive(Debug)]
struct Config {
    input: PathBuf,
    output: PathBuf,
    seeds: Option<PathBuf>,
    lambda: f32,
    seed_strength: i8,
    edge_aware: bool,
    edge_alpha: f32,
    sweeps: u32,
    beta_max: f32,
    quiet: bool,
}

#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
impl Config {
    fn from_args() -> Result<Self, String> {
        let args: Vec<String> = std::env::args().collect();

        let mut config = Config {
            input: PathBuf::new(),
            output: PathBuf::new(),
            seeds: None,
            lambda: 2.0,
            seed_strength: 4,
            edge_aware: false,
            edge_alpha: 0.05,
            sweeps: 100_000,
            beta_max: 20.0,
            quiet: false,
        };

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--input" => {
                    i += 1;
                    config.input = PathBuf::from(args.get(i).ok_or("--input requires a path")?);
                }
                "--output" => {
                    i += 1;
                    config.output = PathBuf::from(args.get(i).ok_or("--output requires a path")?);
                }
                "--seeds" => {
                    i += 1;
                    config.seeds =
                        Some(PathBuf::from(args.get(i).ok_or("--seeds requires a path")?));
                }
                "--lambda" => {
                    i += 1;
                    config.lambda = args
                        .get(i)
                        .ok_or("--lambda requires a value")?
                        .parse()
                        .map_err(|_| "Invalid --lambda value")?;
                }
                "--seed-strength" => {
                    i += 1;
                    config.seed_strength = args
                        .get(i)
                        .ok_or("--seed-strength requires a value")?
                        .parse()
                        .map_err(|_| "Invalid --seed-strength value")?;
                }
                "--edge-aware" => {
                    config.edge_aware = true;
                }
                "--edge-alpha" => {
                    i += 1;
                    config.edge_alpha = args
                        .get(i)
                        .ok_or("--edge-alpha requires a value")?
                        .parse()
                        .map_err(|_| "Invalid --edge-alpha value")?;
                }
                "--sweeps" => {
                    i += 1;
                    config.sweeps = args
                        .get(i)
                        .ok_or("--sweeps requires a value")?
                        .parse()
                        .map_err(|_| "Invalid --sweeps value")?;
                }
                "--beta-max" => {
                    i += 1;
                    config.beta_max = args
                        .get(i)
                        .ok_or("--beta-max requires a value")?
                        .parse()
                        .map_err(|_| "Invalid --beta-max value")?;
                }
                "--quiet" | "-q" => {
                    config.quiet = true;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(format!("Unknown argument: {}", other));
                }
            }
            i += 1;
        }

        if config.input.as_os_str().is_empty() {
            return Err("--input is required".to_string());
        }
        if config.output.as_os_str().is_empty() {
            return Err("--output is required".to_string());
        }

        Ok(config)
    }
}

#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
fn print_help() {
    println!("TileAnneal Image Segmentation CLI");
    println!();
    println!("USAGE:");
    println!("    segmentation_cli --input <image> --output <mask> [OPTIONS]");
    println!();
    println!("REQUIRED:");
    println!("    --input <path>       Input image (PNG/JPG)");
    println!("    --output <path>      Output segmentation mask (PNG)");
    println!();
    println!("OPTIONS:");
    println!("    --seeds <path>       Seeds image (green=foreground, red=background)");
    println!("    --lambda <float>     Coupling strength (default: 2.0)");
    println!("    --seed-strength <n>  Seed bias strength 1-4 (default: 4)");
    println!("    --edge-aware         Enable edge-aware couplings");
    println!("    --edge-alpha <float> Edge sensitivity (default: 0.05)");
    println!("    --sweeps <n>         Annealing sweeps (default: 100000)");
    println!("    --beta-max <float>   Max inverse temperature (default: 20.0)");
    println!("    --quiet, -q          Suppress progress output");
    println!("    --help, -h           Show this help");
    println!();
    println!("EXAMPLE:");
    println!(
        "    segmentation_cli --input photo.png --output mask.png --seeds scribbles.png --edge-aware"
    );
}

#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
#[derive(Clone, Copy, PartialEq)]
enum SeedType {
    None,
    Foreground,
    Background,
}

#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
fn load_seeds_from_image(path: &PathBuf, width: u32, height: u32) -> Result<Vec<SeedType>, String> {
    let img = image::open(path).map_err(|e| format!("Failed to load seeds image: {}", e))?;
    let rgb = img.to_rgb8();

    if rgb.width() != width || rgb.height() != height {
        return Err(format!(
            "Seeds image size {}x{} doesn't match input image {}x{}",
            rgb.width(),
            rgb.height(),
            width,
            height
        ));
    }

    let mut seeds = vec![SeedType::None; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];

            // Green-ish = foreground (g > r && g > b && g > 100)
            if g > r.saturating_add(30) && g > b.saturating_add(30) && g > 100 {
                seeds[(y * width + x) as usize] = SeedType::Foreground;
            }
            // Red-ish = background (r > g && r > b && r > 100)
            else if r > g.saturating_add(30) && r > b.saturating_add(30) && r > 100 {
                seeds[(y * width + x) as usize] = SeedType::Background;
            }
        }
    }

    let fg_count = seeds.iter().filter(|s| **s == SeedType::Foreground).count();
    let bg_count = seeds.iter().filter(|s| **s == SeedType::Background).count();

    if fg_count == 0 && bg_count == 0 {
        eprintln!(
            "Warning: No seeds detected in seeds image. Use green for foreground, red for background."
        );
    }

    Ok(seeds)
}

#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
fn compute_bias(gray: &GrayImage, seeds: &[SeedType], seed_strength: i8) -> Vec<i8> {
    let width = gray.width() as usize;
    let height = gray.height() as usize;
    let mut h_bias = vec![0i8; width * height];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            h_bias[idx] = match seeds[idx] {
                SeedType::Foreground => seed_strength,
                SeedType::Background => -seed_strength,
                SeedType::None => {
                    // Mild bias from intensity
                    let intensity = gray.get_pixel(x as u32, y as u32)[0] as f32;
                    ((intensity - 128.0) / 128.0).round().clamp(-1.0, 1.0) as i8
                }
            };
        }
    }

    h_bias
}

#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
fn compute_edge_aware_couplings(gray: &GrayImage, lambda: f32, alpha: f32) -> (Vec<i8>, Vec<i8>) {
    let width = gray.width() as usize;
    let height = gray.height() as usize;
    let size = width * height;

    let mut j_h = vec![0i8; size]; // Horizontal
    let mut j_v = vec![0i8; size]; // Vertical

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let i_center = gray.get_pixel(x as u32, y as u32)[0] as f32;

            // Horizontal coupling
            if x + 1 < width {
                let i_right = gray.get_pixel((x + 1) as u32, y as u32)[0] as f32;
                let diff = (i_center - i_right) / 255.0;
                let coupling = lambda * (-alpha * diff * diff * 255.0 * 255.0).exp();
                j_h[idx] = coupling.round().clamp(0.0, 4.0) as i8;
            }

            // Vertical coupling
            if y + 1 < height {
                let i_down = gray.get_pixel(x as u32, (y + 1) as u32)[0] as f32;
                let diff = (i_center - i_down) / 255.0;
                let coupling = lambda * (-alpha * diff * diff * 255.0 * 255.0).exp();
                j_v[idx] = coupling.round().clamp(0.0, 4.0) as i8;
            }
        }
    }

    (j_h, j_v)
}

#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
fn save_mask(
    rt: &CudaRuntime,
    grid: &GpuIsingGrid,
    width: usize,
    height: usize,
    output_path: &PathBuf,
) -> Result<(), String> {
    let spins = rt
        .download(&grid.spins)
        .map_err(|e| format!("Failed to download spins: {}", e))?;
    let words_per_row = (width + 63) / 64;

    let mut img = GrayImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let word_idx = y * words_per_row + x / 64;
            let bit_pos = x % 64;
            let spin = if (spins[word_idx] >> bit_pos) & 1 == 1 {
                1
            } else {
                -1
            };

            // Foreground = white (255), background = black (0)
            let val = if spin == 1 { 255u8 } else { 0u8 };
            img.put_pixel(x as u32, y as u32, Luma([val]));
        }
    }

    img.save(output_path)
        .map_err(|e| format!("Failed to save output: {}", e))?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "tileanneal-cli"))]
fn main() {
    let config = match Config::from_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Use --help for usage information.");
            std::process::exit(1);
        }
    };

    if !config.quiet {
        println!("TileAnneal Image Segmentation");
        println!("==============================");
        println!("Input:  {}", config.input.display());
        println!("Output: {}", config.output.display());
        if let Some(ref seeds) = config.seeds {
            println!("Seeds:  {}", seeds.display());
        }
        println!("Lambda: {}", config.lambda);
        println!("Edge-aware: {}", config.edge_aware);
        println!("Sweeps: {}", config.sweeps);
        println!();
    }

    // Load input image
    let img = match image::open(&config.input) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Failed to load input image: {}", e);
            std::process::exit(1);
        }
    };

    let gray = img.to_luma8();
    let width = gray.width() as usize;
    let height = gray.height() as usize;

    if !config.quiet {
        println!(
            "Image size: {}x{} ({} pixels)",
            width,
            height,
            width * height
        );
    }

    // Load seeds if provided
    let seeds = if let Some(ref seeds_path) = config.seeds {
        match load_seeds_from_image(seeds_path, width as u32, height as u32) {
            Ok(s) => {
                let fg = s.iter().filter(|x| **x == SeedType::Foreground).count();
                let bg = s.iter().filter(|x| **x == SeedType::Background).count();
                if !config.quiet {
                    println!("Seeds loaded: {} foreground, {} background", fg, bg);
                }
                s
            }
            Err(e) => {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
    } else {
        if !config.quiet {
            println!("No seeds provided - using intensity-only bias");
        }
        vec![SeedType::None; width * height]
    };

    // Initialize CUDA
    let rt = match CudaRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to initialize CUDA: {:?}", e);
            std::process::exit(1);
        }
    };

    if !config.quiet {
        println!("CUDA initialized");
    }

    // Compute bias
    let h_bias = compute_bias(&gray, &seeds, config.seed_strength);

    // Compute couplings
    let (j_h, j_v) = if config.edge_aware {
        if !config.quiet {
            println!(
                "Computing edge-aware couplings (alpha={})...",
                config.edge_alpha
            );
        }
        compute_edge_aware_couplings(&gray, config.lambda, config.edge_alpha)
    } else {
        let j = config.lambda.round().clamp(1.0, 4.0) as i8;
        (vec![j; width * height], vec![j; width * height])
    };

    // Create Ising grid
    let coupling = CouplingMode::weighted(j_h, j_v);
    let ising_config = IsingConfig::new(width, height).with_seed(42);

    let mut grid = match GpuIsingGrid::new_weighted(&rt, &ising_config, coupling) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to create Ising grid: {:?}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = grid.set_bias(&rt, Some(&h_bias)) {
        eprintln!("Failed to set bias: {:?}", e);
        std::process::exit(1);
    }

    // Run annealing
    if !config.quiet {
        println!("Running annealing ({} sweeps)...", config.sweeps);
    }

    let start = Instant::now();

    // Use ThreeStage adaptive annealing
    let mut schedule = AdaptiveSchedule::new(config.sweeps)
        .with_mode(AdaptiveMode::robust())
        .with_initial_beta(0.1)
        .with_bounds(0.1, config.beta_max);

    // j_coupling = 1.0 for ferromagnetic (positive couplings = neighbors want to align)
    let result = match run_adaptive_annealing(&rt, &mut grid, &mut schedule, 1.0) {
        Ok(r) => r,
        Err(_) => {
            // Fallback to simple annealing if adaptive not available
            if !config.quiet {
                println!("Using fallback annealing...");
            }
            grid.beta = 0.1;
            let sweeps_per_step = 1000;
            let steps = config.sweeps / sweeps_per_step;
            let beta_step = (config.beta_max - 0.1) / steps as f32;

            for _ in 0..steps {
                if let Err(e) = run_ising_update_weighted(&rt, &mut grid, sweeps_per_step) {
                    eprintln!("Annealing error: {:?}", e);
                    std::process::exit(1);
                }
                grid.beta = (grid.beta + beta_step).min(config.beta_max);
            }

            // Get final energy
            let energy = compute_weighted_energy(&rt, &grid).unwrap_or(0);
            engine::cuda_tiles::AdaptiveAnnealingResult {
                best_energy: energy,
                best_cut: 0,
                best_sweep: config.sweeps,
                best_config: grid.download(&rt).unwrap(),
                final_energy: energy,
                total_sweeps: config.sweeps,
                history: vec![],
            }
        }
    };

    let elapsed = start.elapsed();

    if !config.quiet {
        println!("Annealing complete in {:.2?}", elapsed);
        println!("Final energy: {}", result.final_energy);
        println!("Best energy:  {}", result.best_energy);
        println!("Total sweeps: {}", result.total_sweeps);
        let spins_per_sec =
            (width * height) as f64 * result.total_sweeps as f64 / elapsed.as_secs_f64();
        println!(
            "Throughput:   {:.2} billion spin-updates/sec",
            spins_per_sec / 1e9
        );
    }

    // Save output mask
    if let Err(e) = save_mask(&rt, &grid, width, height, &config.output) {
        eprintln!("{}", e);
        std::process::exit(1);
    }

    if !config.quiet {
        println!();
        println!("Segmentation mask saved to: {}", config.output.display());
    }
}

#[cfg(not(all(feature = "cuda", feature = "tileanneal-cli")))]
fn main() {
    eprintln!("This example requires both 'cuda' and 'tileanneal-cli' features.");
    eprintln!(
        "Run with: cargo run --release --features cuda,tileanneal-cli --example segmentation_cli -- --help"
    );
    std::process::exit(1);
}
