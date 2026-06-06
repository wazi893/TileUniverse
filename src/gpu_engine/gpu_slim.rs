//! EPIC 105c: GPU-Native Slim Simulation
//!
//! Entire simulation lives in VRAM for maximum throughput.
//! Target: 14K x 14K grid (196M tiles) at 40B+ evals/sec
//!
//! Memory layout (per tile):
//!   - logic: 8 bytes (u64)
//!   - tile_type: 1 byte (u8)
//!   - neighbors: 16 bytes (4 x u32)
//!   - logic_next: 8 bytes (u64)
//!   Total: 33 bytes/tile in VRAM
//!
//! 8GB VRAM / 33 bytes = ~242M tiles max = 15.5K x 15.5K

#[cfg(feature = "cuda")]
use crate::cuda::{CudaError, CudaResult, CudaRuntime};

#[cfg(feature = "cuda")]
use crate::tile_meta::TileType;

#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use once_cell::sync::Lazy;

#[cfg(feature = "cuda")]
use cudarc::driver::PushKernelArg;

/// Cached kernel for slim simulation
#[cfg(feature = "cuda")]
struct SlimKernels {
    eval_fn: cudarc::driver::CudaFunction,
}

#[cfg(feature = "cuda")]
static SLIM_KERNEL_CACHE: Lazy<Mutex<Option<SlimKernels>>> = Lazy::new(|| Mutex::new(None));

/// GPU-resident simulation for maximum throughput
#[cfg(feature = "cuda")]
pub struct GpuSlimSimulation {
    /// Tile logic values
    pub logic: cudarc::driver::CudaSlice<u64>,
    /// Tile type codes
    pub tile_types: cudarc::driver::CudaSlice<u8>,
    /// Precomputed neighbor indices [left, right, up, down] per tile
    pub neighbors: cudarc::driver::CudaSlice<u32>,
    /// Double buffer for outputs
    pub logic_next: cudarc::driver::CudaSlice<u64>,
    /// Grid dimensions
    pub width: usize,
    pub height: usize,
    /// Clock state (kept on CPU - just 1 byte)
    pub global_clock: bool,
    prev_clock: bool,
}

#[cfg(feature = "cuda")]
impl GpuSlimSimulation {
    /// Create a new GPU simulation with given dimensions
    pub fn with_size(rt: &CudaRuntime, width: usize, height: usize) -> CudaResult<Self> {
        let tile_count = width * height;

        // Initialize all tiles as Wire with logic 0
        let logic_host = vec![0u64; tile_count];
        let types_host = vec![TileType::Wire as u8; tile_count];

        // Build neighbor indices
        let mut neighbors_host: Vec<u32> = Vec::with_capacity(tile_count * 4);
        for idx in 0..tile_count {
            let x = idx % width;
            let y = idx / width;
            let left = if x > 0 {
                (y * width + (x - 1)) as u32
            } else {
                u32::MAX
            };
            let right = if x + 1 < width {
                (y * width + (x + 1)) as u32
            } else {
                u32::MAX
            };
            let up = if y > 0 {
                ((y - 1) * width + x) as u32
            } else {
                u32::MAX
            };
            let down = if y + 1 < height {
                ((y + 1) * width + x) as u32
            } else {
                u32::MAX
            };
            neighbors_host.push(left);
            neighbors_host.push(right);
            neighbors_host.push(up);
            neighbors_host.push(down);
        }

        // Upload to GPU
        let logic = rt.upload(&logic_host)?;
        let tile_types = rt.upload(&types_host)?;
        let neighbors = rt.upload(&neighbors_host)?;
        let logic_next = rt.alloc_zeros::<u64>(tile_count)?;

        Ok(Self {
            logic,
            tile_types,
            neighbors,
            logic_next,
            width,
            height,
            global_clock: false,
            prev_clock: false,
        })
    }

    /// Total tile count
    #[inline]
    pub fn tile_count(&self) -> usize {
        self.width * self.height
    }

    /// Memory usage in VRAM (bytes)
    pub fn vram_usage_bytes(&self) -> usize {
        let tile_count = self.tile_count();
        let logic_bytes = tile_count * 8; // u64
        let types_bytes = tile_count * 1; // u8
        let neighbors_bytes = tile_count * 16; // 4 x u32
        let logic_next_bytes = tile_count * 8; // u64
        logic_bytes + types_bytes + neighbors_bytes + logic_next_bytes
    }

    /// Memory usage in MB
    pub fn vram_usage_mb(&self) -> f64 {
        self.vram_usage_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Set tile types from CPU (batch upload)
    pub fn set_tile_types(&mut self, rt: &CudaRuntime, types: &[u8]) -> CudaResult<()> {
        if types.len() != self.tile_count() {
            return Err(CudaError::InvalidConfig(format!(
                "Type array size {} doesn't match tile count {}",
                types.len(),
                self.tile_count()
            )));
        }
        self.tile_types = rt.upload(types)?;
        Ok(())
    }

    /// Set logic values from CPU (batch upload)
    pub fn set_logic(&mut self, rt: &CudaRuntime, logic: &[u64]) -> CudaResult<()> {
        if logic.len() != self.tile_count() {
            return Err(CudaError::InvalidConfig(format!(
                "Logic array size {} doesn't match tile count {}",
                logic.len(),
                self.tile_count()
            )));
        }
        self.logic = rt.upload(logic)?;
        Ok(())
    }

    /// Download logic values to CPU
    pub fn get_logic(&self, rt: &CudaRuntime) -> CudaResult<Vec<u64>> {
        rt.download(&self.logic)
    }

    /// Swap logic buffers after evaluation
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.logic, &mut self.logic_next);
    }

    /// Toggle clock
    pub fn tick_clock(&mut self) {
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;
    }

    /// Evaluate all tiles on GPU (single step)
    pub fn eval_step(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

        // Ensure kernel is compiled and cached
        let mut cache_guard = SLIM_KERNEL_CACHE.lock().unwrap();
        if cache_guard.is_none() {
            let opts = CompileOptions {
                arch: Some("sm_89"), // RTX 4070M is Ada Lovelace (SM 8.9)
                include_paths: vec![],
                ..Default::default()
            };

            let ptx = compile_ptx_with_opts(SLIM_TILE_EVAL_KERNEL, opts)
                .map_err(|e| CudaError::KernelCompilationFailed(format!("Slim eval: {:?}", e)))?;

            let module = rt
                .ctx()
                .load_module(ptx)
                .map_err(|e| CudaError::KernelCompilationFailed(format!("Module load: {:?}", e)))?;

            let eval_fn = module.load_function("slim_tile_eval_kernel").map_err(|e| {
                CudaError::KernelCompilationFailed(format!("Function load: {:?}", e))
            })?;

            *cache_guard = Some(SlimKernels { eval_fn });
        }

        let kernels = cache_guard.as_ref().unwrap();

        // Launch configuration
        let threads_per_block = 256u32;
        let tile_count = self.tile_count();
        let num_blocks = (tile_count as u32 + threads_per_block - 1) / threads_per_block;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let tile_count_i32 = tile_count as i32;
        let clock = if self.global_clock { 1i32 } else { 0i32 };
        let prev = if self.prev_clock { 1i32 } else { 0i32 };

        unsafe {
            rt.stream()
                .launch_builder(&kernels.eval_fn)
                .arg(&self.logic)
                .arg(&self.logic_next)
                .arg(&self.tile_types)
                .arg(&self.neighbors)
                .arg(&tile_count_i32)
                .arg(&clock)
                .arg(&prev)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("Slim eval: {:?}", e)))?;
        }

        // Swap buffers
        self.swap_buffers();
        Ok(())
    }

    /// Evaluate multiple steps (batched, single sync at end)
    pub fn eval_steps(&mut self, rt: &CudaRuntime, steps: u32) -> CudaResult<()> {
        for _ in 0..steps {
            self.tick_clock();
            self.eval_step(rt)?;
        }
        rt.synchronize()?;
        Ok(())
    }

    /// Benchmark: run N steps and return (total_evals, elapsed_secs, evals_per_sec)
    pub fn benchmark(
        &mut self,
        rt: &CudaRuntime,
        steps: u32,
        warmup: u32,
    ) -> CudaResult<(u64, f64, f64)> {
        // Warmup
        for _ in 0..warmup {
            self.tick_clock();
            self.eval_step(rt)?;
        }
        rt.synchronize()?;

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..steps {
            self.tick_clock();
            self.eval_step(rt)?;
        }
        rt.synchronize()?;
        let elapsed = start.elapsed().as_secs_f64();

        let total_evals = (self.tile_count() as u64) * (steps as u64);
        let evals_per_sec = total_evals as f64 / elapsed;

        Ok((total_evals, elapsed, evals_per_sec))
    }

    /// Calculate maximum grid size for given VRAM budget (in bytes)
    pub fn max_grid_size(vram_bytes: usize) -> (usize, usize) {
        // 33 bytes per tile: 8 (logic) + 1 (type) + 16 (neighbors) + 8 (logic_next)
        let bytes_per_tile = 33;
        let max_tiles = vram_bytes / bytes_per_tile;
        let side = (max_tiles as f64).sqrt() as usize;
        (side, side)
    }
}

/// CUDA kernel for slim tile evaluation (same logic as main kernel)
#[cfg(feature = "cuda")]
const SLIM_TILE_EVAL_KERNEL: &str = r#"
extern "C" __global__
void slim_tile_eval_kernel(
    unsigned long long* logic_in,
    unsigned long long* logic_out,
    const unsigned char* tile_types,
    const unsigned int* neighbors,
    int tile_count,
    int global_clock,
    int prev_clock
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tile_count) return;

    // Load neighbor indices
    int base = idx * 4;
    unsigned int left_idx = neighbors[base + 0];
    unsigned int right_idx = neighbors[base + 1];
    unsigned int up_idx = neighbors[base + 2];
    unsigned int down_idx = neighbors[base + 3];

    // Load neighbor values (0 if at boundary)
    unsigned long long left = (left_idx != 0xFFFFFFFF) ? logic_in[left_idx] : 0ULL;
    unsigned long long right = (right_idx != 0xFFFFFFFF) ? logic_in[right_idx] : 0ULL;
    unsigned long long up = (up_idx != 0xFFFFFFFF) ? logic_in[up_idx] : 0ULL;
    unsigned long long down = (down_idx != 0xFFFFFFFF) ? logic_in[down_idx] : 0ULL;

    // Load current value and type
    unsigned long long current = logic_in[idx];
    int tile_type = tile_types[idx];

    // Compute output based on tile type
    unsigned long long output;
    switch (tile_type) {
        case 0:  // Wire
            output = left | right | up | down;
            break;
        case 1:  // And
            output = left & right;
            break;
        case 2:  // Or
            output = left | right;
            break;
        case 3:  // Xor
            output = left ^ right;
            break;
        case 4:  // Not
            output = ~left;
            break;
        case 5:  // Latch
            output = global_clock ? left : current;
            break;
        case 6:  // Register8 (rising edge triggered, 8-bit capture)
            output = (!prev_clock && global_clock) ? (left & 0xFFULL) : current;
            break;
        case 7:  // ClockGlobal
            output = global_clock ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;
            break;
        // Arithmetic Tiles (11-17)
        case 11: output = left + right; break;  // Add
        case 12: output = left - right; break;  // Sub
        case 13: output = left * right; break;  // Mul
        case 14: output = (right != 0) ? (left / right) : 0ULL; break;  // Div
        case 15: output = (right != 0) ? (left % right) : 0ULL; break;  // Mod
        case 16: output = left << (right & 63); break;  // Shl
        case 17: output = left >> (right & 63); break;  // Shr
        // Comparison Tiles (21-26)
        case 21: output = (left < right) ? 1ULL : 0ULL; break;   // Lt
        case 22: output = (left > right) ? 1ULL : 0ULL; break;   // Gt
        case 23: output = (left == right) ? 1ULL : 0ULL; break;  // Eq
        case 24: output = (left != right) ? 1ULL : 0ULL; break;  // Neq
        case 25: output = (left <= right) ? 1ULL : 0ULL; break;  // Lte
        case 26: output = (left >= right) ? 1ULL : 0ULL; break;  // Gte
        // Routing & Special (27-30)
        case 27: output = (up != 0) ? right : left; break;  // Mux
        case 28: output = (left == 0) ? 1ULL : 0ULL; break;  // Zero
        case 29: output = (unsigned long long)(-(long long)left); break;  // Neg
        case 30: {  // Abs
            long long signed_val = (long long)left;
            output = (unsigned long long)(signed_val < 0 ? -signed_val : signed_val);
            break;
        }
        // Memory Tiles (31-33)
        case 31: output = (up != 0) ? left : current; break;  // Ram
        case 32: output = (up != 0) ? current + 1 : current; break;  // Counter
        case 33: output = current; break;  // Const
        default:
            output = current;  // Passthrough for unknown types
            break;
    }

    logic_out[idx] = output;
}
"#;

/// Calculate max tiles for VRAM budget
pub fn max_tiles_for_vram(vram_mb: usize) -> usize {
    let vram_bytes = vram_mb * 1024 * 1024;
    let bytes_per_tile = 33; // 8 + 1 + 16 + 8
    vram_bytes / bytes_per_tile
}

/// Calculate VRAM needed for grid
pub fn vram_for_grid(width: usize, height: usize) -> usize {
    let tiles = width * height;
    let bytes_per_tile = 33;
    (tiles * bytes_per_tile) / (1024 * 1024) // MB
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_calculations() {
        // 8GB VRAM
        let max = max_tiles_for_vram(8 * 1024);
        println!("8GB VRAM max tiles: {}M", max / 1_000_000);
        assert!(max > 200_000_000); // Should be > 200M

        // 14K x 14K grid
        let vram = vram_for_grid(14336, 14336);
        println!("14K x 14K needs: {} MB", vram);
        assert!(vram < 8 * 1024); // Should fit in 8GB
    }
}
