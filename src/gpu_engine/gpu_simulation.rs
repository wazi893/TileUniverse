//! EPIC 97: Fused GPU Simulation Kernel
//!
//! Zero-transfer-per-tick simulation with all state GPU-resident.
//!
//! Architecture:
//! - Phase A: sense_kernel - Each organism reads field neighbors into sensors
//! - Phase B: FP32 NN forward - Simple inline NN (later: WMMA batch forward)
//! - Phase C: act_kernel - Apply moves, update energy, check death
//!
//! All three phases execute on GPU with no PCIe transfers between ticks.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;

#[cfg(feature = "cuda")]
use crate::batched_brain::BrainWeights;
#[cfg(feature = "cuda")]
use crate::cuda::{CudaError, CudaResult, CudaRuntime};

/// GPU-resident simulation state
#[cfg(feature = "cuda")]
pub struct GpuSimulation {
    rt: CudaRuntime,

    // Organism state (GPU-resident)
    d_positions: CudaSlice<f32>, // [capacity * 2] - x,y pairs
    d_energy: CudaSlice<f32>,    // [capacity]
    d_alive: CudaSlice<i32>,     // [capacity] - 0=dead, 1=alive

    // Sensor buffer (GPU-resident, reused each tick)
    d_sensors: CudaSlice<f32>, // [capacity * 8] - 8 sensors per organism

    // Move decisions (GPU-resident, written by NN, read by act kernel)
    d_moves: CudaSlice<i32>, // [capacity] - move index 0-4

    // Environment (GPU-resident)
    d_field: CudaSlice<f32>, // [grid_h * grid_w] - resource field

    // NN weights (GPU-resident, FP32)
    d_weights_ih: CudaSlice<f32>, // [64] - input→hidden weights
    d_bias_h: CudaSlice<f32>,     // [8] - hidden bias
    d_weights_ho: CudaSlice<f32>, // [40] - hidden→output weights
    d_bias_o: CudaSlice<f32>,     // [5] - output bias

    // Kernels
    sense_kernel: CudaFunction,
    nn_kernel: CudaFunction,
    act_kernel: CudaFunction,

    // Configuration
    grid_width: i32,
    grid_height: i32,
    move_cost: f32,
    eat_gain: f32,
    capacity: usize,
    active_count: usize,
}

/// Simulation state snapshot (for CPU inspection)
#[derive(Clone, Debug)]
pub struct SimulationState {
    pub positions: Vec<f32>, // [n * 2] - x,y pairs
    pub energy: Vec<f32>,    // [n]
    pub alive: Vec<i32>,     // [n]
}

#[cfg(feature = "cuda")]
const SENSE_KERNEL: &str = r#"
extern "C" __global__ void sense_kernel(
    const float* __restrict__ positions,
    const float* __restrict__ field,
    const int* __restrict__ alive,
    float* __restrict__ sensors,
    int n_organisms,
    int grid_w,
    int grid_h
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_organisms) return;
    if (alive[tid] == 0) {
        for (int i = 0; i < 8; i++) {
            sensors[tid * 8 + i] = 0.0f;
        }
        return;
    }

    float x = positions[tid * 2];
    float y = positions[tid * 2 + 1];
    int ix = (int)x;
    int iy = (int)y;

    // 8 directions: N, S, E, W, NE, NW, SE, SW
    int dx[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    int dy[8] = {-1, 1, 0, 0, -1, -1, 1, 1};

    for (int d = 0; d < 8; d++) {
        int nx = ix + dx[d];
        int ny = iy + dy[d];

        float value = 0.0f;
        if (nx >= 0 && nx < grid_w && ny >= 0 && ny < grid_h) {
            value = field[ny * grid_w + nx] / 255.0f;
        }

        sensors[tid * 8 + d] = value * 2.0f - 1.0f;
    }
}
"#;

#[cfg(feature = "cuda")]
const NN_FORWARD_KERNEL: &str = r#"
extern "C" __global__ void nn_forward(
    const float* __restrict__ sensors,
    int* __restrict__ moves,
    const float* __restrict__ weights_ih,
    const float* __restrict__ bias_h,
    const float* __restrict__ weights_ho,
    const float* __restrict__ bias_o,
    const int* __restrict__ alive,
    int n_organisms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_organisms) return;
    if (alive[tid] == 0) {
        moves[tid] = 4;
        return;
    }

    float inp[8];
    for (int i = 0; i < 8; i++) {
        inp[i] = sensors[tid * 8 + i];
    }

    // Layer 1: Input -> Hidden (8 -> 8)
    float hidden[8];
    for (int h = 0; h < 8; h++) {
        float sum = bias_h[h];
        for (int i = 0; i < 8; i++) {
            sum += inp[i] * weights_ih[h * 8 + i];
        }
        if (sum > 3.0f) hidden[h] = 1.0f;
        else if (sum < -3.0f) hidden[h] = -1.0f;
        else hidden[h] = sum * (27.0f + sum * sum) / (27.0f + 9.0f * sum * sum);
    }

    // Layer 2: Hidden -> Output (8 -> 5)
    float output[5];
    for (int o = 0; o < 5; o++) {
        float sum = bias_o[o];
        for (int h = 0; h < 8; h++) {
            sum += hidden[h] * weights_ho[h * 5 + o];
        }
        if (sum > 3.0f) output[o] = 1.0f;
        else if (sum < -3.0f) output[o] = -1.0f;
        else output[o] = sum * (27.0f + sum * sum) / (27.0f + 9.0f * sum * sum);
    }

    // Argmax
    int best_move = 0;
    float best_val = output[0];
    for (int o = 1; o < 5; o++) {
        if (output[o] > best_val) {
            best_val = output[o];
            best_move = o;
        }
    }

    moves[tid] = best_move;
}
"#;

#[cfg(feature = "cuda")]
const ACT_KERNEL: &str = r#"
extern "C" __global__ void act_kernel(
    float* __restrict__ positions,
    float* __restrict__ energy,
    int* __restrict__ alive,
    const float* __restrict__ field,
    const int* __restrict__ moves,
    int n_organisms,
    int grid_w,
    int grid_h,
    float move_cost,
    float eat_gain
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_organisms) return;
    if (alive[tid] == 0) return;

    float x = positions[tid * 2];
    float y = positions[tid * 2 + 1];

    int mv = moves[tid];
    float dx_table[5] = {0.0f, 0.0f, 1.0f, -1.0f, 0.0f};
    float dy_table[5] = {-1.0f, 1.0f, 0.0f, 0.0f, 0.0f};

    if (mv >= 0 && mv < 5) {
        x += dx_table[mv];
        y += dy_table[mv];
    }

    x = fmaxf(0.0f, fminf(x, (float)(grid_w - 1)));
    y = fmaxf(0.0f, fminf(y, (float)(grid_h - 1)));

    positions[tid * 2] = x;
    positions[tid * 2 + 1] = y;

    float e = energy[tid] - move_cost;

    int ix = (int)x;
    int iy = (int)y;
    float food = field[iy * grid_w + ix];
    if (food > 0.0f) {
        e += eat_gain;
    }

    if (e <= 0.0f) {
        alive[tid] = 0;
        e = 0.0f;
    }

    energy[tid] = e;
}
"#;

#[cfg(feature = "cuda")]
fn compile_kernel(rt: &CudaRuntime, src: &str, name: &str) -> CudaResult<CudaFunction> {
    let ptx = compile_ptx(src)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("{}: {:?}", name, e)))?;

    let module = rt
        .ctx()
        .load_module(ptx)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("{} module: {:?}", name, e)))?;

    let kernel = module
        .load_function(name)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("{} function: {:?}", name, e)))?;

    Ok(kernel)
}

#[cfg(feature = "cuda")]
impl GpuSimulation {
    /// Create new GPU simulation with given capacity
    pub fn new(
        capacity: usize,
        grid_w: i32,
        grid_h: i32,
        move_cost: f32,
        eat_gain: f32,
    ) -> CudaResult<Self> {
        let rt = CudaRuntime::new()?;

        // Compile kernels
        let sense_kernel = compile_kernel(&rt, SENSE_KERNEL, "sense_kernel")?;
        let nn_kernel = compile_kernel(&rt, NN_FORWARD_KERNEL, "nn_forward")?;
        let act_kernel = compile_kernel(&rt, ACT_KERNEL, "act_kernel")?;

        // Allocate organism buffers
        let d_positions = rt.alloc_zeros::<f32>(capacity * 2)?;
        let d_energy = rt.alloc_zeros::<f32>(capacity)?;
        let d_alive = rt.alloc_zeros::<i32>(capacity)?;
        let d_sensors = rt.alloc_zeros::<f32>(capacity * 8)?;
        let d_moves = rt.alloc_zeros::<i32>(capacity)?;
        let d_field = rt.alloc_zeros::<f32>((grid_w * grid_h) as usize)?;

        // Allocate weight buffers
        let d_weights_ih = rt.alloc_zeros::<f32>(64)?;
        let d_bias_h = rt.alloc_zeros::<f32>(8)?;
        let d_weights_ho = rt.alloc_zeros::<f32>(40)?;
        let d_bias_o = rt.alloc_zeros::<f32>(5)?;

        Ok(Self {
            rt,
            d_positions,
            d_energy,
            d_alive,
            d_sensors,
            d_moves,
            d_field,
            d_weights_ih,
            d_bias_h,
            d_weights_ho,
            d_bias_o,
            sense_kernel,
            nn_kernel,
            act_kernel,
            grid_width: grid_w,
            grid_height: grid_h,
            move_cost,
            eat_gain,
            capacity,
            active_count: 0,
        })
    }

    /// Upload organism state
    pub fn upload_organisms(
        &mut self,
        positions: &[f32],
        energy: &[f32],
        alive: &[i32],
    ) -> CudaResult<()> {
        let n = energy.len();
        assert_eq!(positions.len(), n * 2);
        assert_eq!(alive.len(), n);
        assert!(n <= self.capacity);

        self.d_positions = self.rt.upload(positions)?;
        self.d_energy = self.rt.upload(energy)?;
        self.d_alive = self.rt.upload(alive)?;

        self.active_count = n;
        Ok(())
    }

    /// Upload field
    pub fn upload_field(&mut self, field: &[f32]) -> CudaResult<()> {
        let expected = (self.grid_width * self.grid_height) as usize;
        assert_eq!(
            field.len(),
            expected,
            "Field size mismatch: got {}, expected {}",
            field.len(),
            expected
        );
        self.d_field = self.rt.upload(field)?;
        Ok(())
    }

    /// Upload weights
    pub fn upload_weights(&mut self, weights: &BrainWeights) -> CudaResult<()> {
        // Flatten weights_ih: [h][i] -> row-major
        let mut weights_ih = vec![0.0f32; 64];
        for h in 0..8 {
            for i in 0..8 {
                weights_ih[h * 8 + i] = weights.weights_ih[h][i];
            }
        }

        // Flatten weights_ho: [h][o] -> row-major
        let mut weights_ho = vec![0.0f32; 40];
        for h in 0..8 {
            for o in 0..5 {
                weights_ho[h * 5 + o] = weights.weights_ho[h][o];
            }
        }

        self.d_weights_ih = self.rt.upload(&weights_ih)?;
        self.d_bias_h = self.rt.upload(&weights.bias_h)?;
        self.d_weights_ho = self.rt.upload(&weights_ho)?;
        self.d_bias_o = self.rt.upload(&weights.bias_o)?;

        Ok(())
    }

    /// Run one tick on GPU (zero transfers!)
    pub fn tick(&mut self) -> CudaResult<()> {
        if self.active_count == 0 {
            return Ok(());
        }

        let n = self.active_count as i32;
        let block_size = 256u32;
        let grid_size = ((self.active_count + 255) / 256) as u32;
        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Phase A: Sense
        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.sense_kernel)
                .arg(&self.d_positions)
                .arg(&self.d_field)
                .arg(&self.d_alive)
                .arg(&self.d_sensors)
                .arg(&n)
                .arg(&self.grid_width)
                .arg(&self.grid_height)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("sense: {:?}", e)))?;
        }

        // Phase B: NN Forward
        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.nn_kernel)
                .arg(&self.d_sensors)
                .arg(&self.d_moves)
                .arg(&self.d_weights_ih)
                .arg(&self.d_bias_h)
                .arg(&self.d_weights_ho)
                .arg(&self.d_bias_o)
                .arg(&self.d_alive)
                .arg(&n)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("nn: {:?}", e)))?;
        }

        // Phase C: Act
        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.act_kernel)
                .arg(&self.d_positions)
                .arg(&self.d_energy)
                .arg(&self.d_alive)
                .arg(&self.d_field)
                .arg(&self.d_moves)
                .arg(&n)
                .arg(&self.grid_width)
                .arg(&self.grid_height)
                .arg(&self.move_cost)
                .arg(&self.eat_gain)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("act: {:?}", e)))?;
        }

        Ok(())
    }

    /// Run N ticks (fire-and-forget!)
    pub fn tick_many(&mut self, n: usize) -> CudaResult<()> {
        for _ in 0..n {
            self.tick()?;
        }
        Ok(())
    }

    /// Sync state back to CPU (expensive, do sparingly)
    pub fn sync_state(&self) -> CudaResult<SimulationState> {
        let positions = self.rt.download(&self.d_positions)?;
        let energy = self.rt.download(&self.d_energy)?;
        let alive = self.rt.download(&self.d_alive)?;

        // Trim to active_count
        let positions = positions[..self.active_count * 2].to_vec();
        let energy = energy[..self.active_count].to_vec();
        let alive = alive[..self.active_count].to_vec();

        Ok(SimulationState {
            positions,
            energy,
            alive,
        })
    }

    /// Count alive organisms
    pub fn count_alive(&self) -> CudaResult<usize> {
        let state = self.sync_state()?;
        Ok(state.alive.iter().filter(|&&a| a == 1).count())
    }

    /// Synchronize GPU
    pub fn synchronize(&self) -> CudaResult<()> {
        self.rt.synchronize()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn active_count(&self) -> usize {
        self.active_count
    }
    pub fn grid_size(&self) -> (i32, i32) {
        (self.grid_width, self.grid_height)
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::batched_brain::BrainWeights;

    #[test]
    fn test_gpu_simulation_creation() {
        let sim = GpuSimulation::new(1000, 64, 64, 1.0, 5.0);
        assert!(sim.is_ok());
        let sim = sim.unwrap();
        assert_eq!(sim.capacity(), 1000);
        assert_eq!(sim.grid_size(), (64, 64));
    }

    #[test]
    fn test_upload_and_tick() {
        let mut sim = GpuSimulation::new(100, 64, 64, 1.0, 5.0).unwrap();

        // Create test organisms
        let mut positions = vec![0.0f32; 100 * 2];
        let energy = vec![100.0f32; 100];
        let alive = vec![1i32; 100];

        // Scatter organisms across grid
        for i in 0..100 {
            positions[i * 2] = (i % 64) as f32;
            positions[i * 2 + 1] = (i / 64) as f32;
        }

        // Create field with some food
        let mut field = vec![0.0f32; 64 * 64];
        for i in 0..64 {
            for j in 0..64 {
                if (i + j) % 4 == 0 {
                    field[j * 64 + i] = 255.0;
                }
            }
        }

        // Upload
        sim.upload_organisms(&positions, &energy, &alive).unwrap();
        sim.upload_field(&field).unwrap();
        sim.upload_weights(&BrainWeights::random(12345)).unwrap();

        // Run a tick
        sim.tick().unwrap();
        sim.synchronize().unwrap();

        // Check state
        let state = sim.sync_state().unwrap();
        assert_eq!(state.alive.len(), 100);

        // All should still be alive after 1 tick with 100 energy
        let alive_count = state.alive.iter().filter(|&&a| a == 1).count();
        assert_eq!(alive_count, 100);
    }
}
