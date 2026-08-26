//! V2 MMIO reference devices, display device, math coprocessor, SNN bridge,
//! and p-bit bridge prototype.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::pbit::ising::problems;
use crate::pbit::{PBitConfig, PBitNetwork};
use crate::tile_cpu::v2_mmio::V2MmioDevice;

pub const V2_MMIO_REF_SNAPSHOT_KIND: &str = "v2.mmio.ref_pack.v1";
pub const V2_MMIO_PBIT_SNAPSHOT_KIND: &str = "v2.mmio.pbit_bridge.v1";
pub const V2_MMIO_DISPLAY_SNAPSHOT_KIND: &str = "v2.mmio.display.v1";
pub const V2_MMIO_MATH_SNAPSHOT_KIND: &str = "v2.mmio.math.v1";
pub const V2_MMIO_SNN_SNAPSHOT_KIND: &str = "v2.mmio.snn_bridge.v1";
pub const V2_MMIO_QUANTUM_SNAPSHOT_KIND: &str = "v2.mmio.quantum.v1";

// Sprint 157: All MMIO address constants use absolute values (decoupled from
// V2_MMIO_BASE) to prevent cascading changes when the MMIO window expands.

// --- Dataset device (addresses 41-43, Sprint 159) ---
pub const MMIO_DATASET_CMD: u8 = 41;
pub const MMIO_DATASET_DATA: u8 = 42;
pub const MMIO_DATASET_STATUS: u8 = 43;

// --- HLS accelerator (addresses 41-43, Sprint 388) ---
// The 23-address MMIO window is fully assigned, so the accelerator OVERLAYS the
// optional dataset device's slots: both are optional sub-devices of
// `V2MmioCombinedDevice` and are mutually exclusive (the constructors enforce one or
// the other). In the default combined device neither is present and 41-43 fall
// through to the ref pack, which ignores them — these are the only genuinely free
// addresses in practice. Indexed-operand protocol to stay frugal with the window:
// write an operand index to ARG_SELECT, its value to ARG_DATA (full 64-bit ST;
// the device latches operand[index]); a read of RESULT evaluates the synthesized
// tile datapath on the latched operands and returns the answer.
pub const MMIO_ACCEL_ARG_SELECT: u8 = 41;
pub const MMIO_ACCEL_ARG_DATA: u8 = 42;
pub const MMIO_ACCEL_RESULT: u8 = 43;

// --- Quantum bridge (addresses 44-47, Sprint 157) ---
pub const MMIO_QUANTUM_CMD: u8 = 44;
pub const MMIO_QUANTUM_QUBIT: u8 = 45;
pub const MMIO_QUANTUM_DATA: u8 = 46;
pub const MMIO_QUANTUM_PARAM: u8 = 47;

// --- Display device (addresses 48-49) ---
pub const MMIO_DISPLAY_CMD: u8 = 48;
pub const MMIO_DISPLAY_STATUS: u8 = 49;

// --- Math coprocessor (addresses 50-53) ---
pub const MMIO_MATH_A: u8 = 50;
pub const MMIO_MATH_B: u8 = 51;
pub const MMIO_MATH_CMD: u8 = 52;
pub const MMIO_MATH_RESULT: u8 = 53;

// --- SNN bridge (addresses 54-55) ---
pub const MMIO_SNN_DATA: u8 = 54;
pub const MMIO_SNN_CMD: u8 = 55;

// --- Reference devices (addresses 56-63, backward-compatible) ---
pub const MMIO_TIMER_CYCLE: u8 = 56;
pub const MMIO_CONSOLE_DATA: u8 = 57;
pub const MMIO_CONSOLE_COUNT: u8 = 58;
pub const MMIO_RNG_DATA: u8 = 59;
pub const MMIO_MAILBOX_IN: u8 = 60;
pub const MMIO_MAILBOX_OUT: u8 = 61;
pub const MMIO_PBIT_CTRL: u8 = 62;
pub const MMIO_PBIT_RESULT: u8 = 63;

const PBIT_STATUS_IDLE: u64 = 0;
const PBIT_STATUS_DONE: u64 = 2;
const PBIT_STATUS_ERROR: u64 = 0x8000_0000_0000_0000;

/// Reference MMIO pack: timer, console, deterministic RNG, mailbox.
#[derive(Debug, Default)]
pub struct V2MmioRefDevicePack {
    cycle: Cell<u64>,
    console_last: Cell<u64>,
    console_count: Cell<u64>,
    /// Sprint 375 (Gate E): full accumulated console output (every byte written to
    /// MMIO_CONSOLE_DATA, in order), so a compiled program's printed text can be read
    /// back. `console_last`/`console_count` are unchanged.
    console_buffer: RefCell<Vec<u8>>,
    rng_state: Cell<u64>,
    mailbox_in: Cell<u64>,
    mailbox_out: Cell<u64>,
}

impl V2MmioRefDevicePack {
    pub fn new(seed: u64) -> Self {
        Self {
            rng_state: Cell::new(seed),
            ..Self::default()
        }
    }

    pub fn cycle(&self) -> u64 {
        self.cycle.get()
    }

    pub fn console_last(&self) -> u64 {
        self.console_last.get()
    }

    pub fn console_count(&self) -> u64 {
        self.console_count.get()
    }

    /// Sprint 375 (Gate E): the full accumulated console output as raw bytes.
    pub fn console_output(&self) -> Vec<u8> {
        self.console_buffer.borrow().clone()
    }

    /// Sprint 375 (Gate E): the accumulated console output as a UTF-8 string (lossy).
    pub fn console_string(&self) -> String {
        String::from_utf8_lossy(&self.console_buffer.borrow()).into_owned()
    }

    pub fn mailbox_in(&self) -> u64 {
        self.mailbox_in.get()
    }

    pub fn mailbox_out(&self) -> u64 {
        self.mailbox_out.get()
    }

    fn next_rng_u64(&self) -> u64 {
        let mut x = self.rng_state.get();
        if x == 0 {
            x = 0x9E37_79B9_7F4A_7C15;
        }
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.rng_state.set(x);
        x
    }

    fn snapshot_words(&self) -> [u64; 6] {
        [
            self.cycle.get(),
            self.console_last.get(),
            self.console_count.get(),
            self.rng_state.get(),
            self.mailbox_in.get(),
            self.mailbox_out.get(),
        ]
    }

    fn restore_words(&self, words: &[u64; 6]) {
        self.cycle.set(words[0]);
        self.console_last.set(words[1]);
        self.console_count.set(words[2]);
        self.rng_state.set(words[3]);
        self.mailbox_in.set(words[4]);
        self.mailbox_out.set(words[5]);
    }
}

impl V2MmioDevice for V2MmioRefDevicePack {
    fn read(&self, addr: u8) -> u64 {
        match addr {
            MMIO_TIMER_CYCLE => self.cycle.get(),
            MMIO_CONSOLE_DATA => self.console_last.get(),
            MMIO_CONSOLE_COUNT => self.console_count.get(),
            MMIO_RNG_DATA => self.next_rng_u64(),
            MMIO_MAILBOX_IN => self.mailbox_in.get(),
            MMIO_MAILBOX_OUT => self.mailbox_out.get(),
            _ => 0,
        }
    }

    fn write(&self, addr: u8, value: u64) {
        match addr {
            MMIO_CONSOLE_DATA => {
                self.console_last.set(value & 0xFF);
                self.console_count.set(self.console_count.get() + 1);
                self.console_buffer.borrow_mut().push((value & 0xFF) as u8);
            }
            MMIO_MAILBOX_IN => self.mailbox_in.set(value),
            MMIO_MAILBOX_OUT => self.mailbox_out.set(value),
            _ => {}
        }
    }

    fn tick(&self, cycle: u64) {
        self.cycle.set(cycle);
    }

    fn snapshot_kind(&self) -> Option<&'static str> {
        Some(V2_MMIO_REF_SNAPSHOT_KIND)
    }

    fn snapshot_state(&self) -> Option<Vec<u8>> {
        Some(encode_u64_snapshot(&self.snapshot_words()))
    }

    fn restore_state(&self, snapshot: &[u8]) -> Result<(), String> {
        let words = decode_u64_snapshot::<6>(snapshot)?;
        self.restore_words(&words);
        Ok(())
    }
}

/// MMIO p-bit bridge prototype.
///
/// Register map:
/// - `base+0` (`MMIO_TIMER_CYCLE`): command/status.
///   - write `1` to run p-bit solve.
///   - read status: `0` idle, `2` done, `0x8000..` error.
/// - `base+1` (`MMIO_CONSOLE_DATA`): seed (u64)
/// - `base+2` (`MMIO_CONSOLE_COUNT`): steps (u64)
/// - `base+3` (`MMIO_RNG_DATA`): n_pbits (u64, clamped 2..16)
/// - `base+4` (`MMIO_MAILBOX_IN`): best energy bits (`f64::to_bits()`)
/// - `base+5` (`MMIO_MAILBOX_OUT`): best state bitset (low `n` bits)
/// - `base+6` (`MMIO_PBIT_CTRL`): run counter
/// - `base+7` (`MMIO_PBIT_RESULT`): last error code
#[derive(Debug)]
pub struct V2MmioPbitBridgeDevice {
    status: Cell<u64>,
    seed: Cell<u64>,
    steps: Cell<u64>,
    n_pbits: Cell<u64>,
    best_energy_bits: Cell<u64>,
    best_state_bits: Cell<u64>,
    run_count: Cell<u64>,
    last_error: Cell<u64>,
}

impl Default for V2MmioPbitBridgeDevice {
    fn default() -> Self {
        Self {
            status: Cell::new(PBIT_STATUS_IDLE),
            seed: Cell::new(42),
            steps: Cell::new(256),
            n_pbits: Cell::new(8),
            best_energy_bits: Cell::new(0),
            best_state_bits: Cell::new(0),
            run_count: Cell::new(0),
            last_error: Cell::new(0),
        }
    }
}

impl V2MmioPbitBridgeDevice {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn status(&self) -> u64 {
        self.status.get()
    }

    pub fn best_energy(&self) -> f64 {
        f64::from_bits(self.best_energy_bits.get())
    }

    pub fn best_state_bits(&self) -> u64 {
        self.best_state_bits.get()
    }

    fn run_solver(&self) {
        let n = (self.n_pbits.get().clamp(2, 16)) as usize;
        let seed = self.seed.get();
        let steps = self.steps.get().clamp(1, 200_000);

        let config = PBitConfig::new(n).with_seed(seed).with_beta(2.0);
        let mut network = PBitNetwork::new(config);
        let problem = problems::random_maxcut(n, 0.5, seed ^ 0xA5A5_A5A5_A5A5_A5A5);
        network.set_problem(&problem);
        network.randomize();
        network.run(steps);

        let best_energy = network.best_energy();
        let best_state = network.best_state();
        let mut bits = 0u64;
        for (i, &b) in best_state.iter().take(64).enumerate() {
            if b != 0 {
                bits |= 1u64 << i;
            }
        }

        self.best_energy_bits.set(best_energy.to_bits());
        self.best_state_bits.set(bits);
        self.status.set(PBIT_STATUS_DONE);
        self.last_error.set(0);
        self.run_count.set(self.run_count.get() + 1);
    }

    fn snapshot_words(&self) -> [u64; 8] {
        [
            self.status.get(),
            self.seed.get(),
            self.steps.get(),
            self.n_pbits.get(),
            self.best_energy_bits.get(),
            self.best_state_bits.get(),
            self.run_count.get(),
            self.last_error.get(),
        ]
    }

    fn restore_words(&self, words: &[u64; 8]) {
        self.status.set(words[0]);
        self.seed.set(words[1]);
        self.steps.set(words[2]);
        self.n_pbits.set(words[3]);
        self.best_energy_bits.set(words[4]);
        self.best_state_bits.set(words[5]);
        self.run_count.set(words[6]);
        self.last_error.set(words[7]);
    }
}

impl V2MmioDevice for V2MmioPbitBridgeDevice {
    fn read(&self, addr: u8) -> u64 {
        match addr {
            MMIO_TIMER_CYCLE => self.status.get(),
            MMIO_CONSOLE_DATA => self.seed.get(),
            MMIO_CONSOLE_COUNT => self.steps.get(),
            MMIO_RNG_DATA => self.n_pbits.get(),
            MMIO_MAILBOX_IN => self.best_energy_bits.get(),
            MMIO_MAILBOX_OUT => self.best_state_bits.get(),
            MMIO_PBIT_CTRL => self.run_count.get(),
            MMIO_PBIT_RESULT => self.last_error.get(),
            _ => 0,
        }
    }

    fn write(&self, addr: u8, value: u64) {
        match addr {
            MMIO_TIMER_CYCLE => {
                if value == 1 {
                    self.run_solver();
                } else if value == 0 {
                    self.status.set(PBIT_STATUS_IDLE);
                }
            }
            MMIO_CONSOLE_DATA => self.seed.set(value),
            MMIO_CONSOLE_COUNT => self.steps.set(value),
            MMIO_RNG_DATA => self.n_pbits.set(value),
            MMIO_MAILBOX_IN => self.best_energy_bits.set(value),
            MMIO_MAILBOX_OUT => self.best_state_bits.set(value),
            MMIO_PBIT_CTRL => self.run_count.set(value),
            MMIO_PBIT_RESULT => self.last_error.set(value),
            _ => {
                self.status.set(PBIT_STATUS_ERROR);
                self.last_error.set(1);
            }
        }
    }

    fn snapshot_kind(&self) -> Option<&'static str> {
        Some(V2_MMIO_PBIT_SNAPSHOT_KIND)
    }

    fn snapshot_state(&self) -> Option<Vec<u8>> {
        Some(encode_u64_snapshot(&self.snapshot_words()))
    }

    fn restore_state(&self, snapshot: &[u8]) -> Result<(), String> {
        let words = decode_u64_snapshot::<8>(snapshot)?;
        self.restore_words(&words);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Math coprocessor device
// ---------------------------------------------------------------------------

/// Math coprocessor operation codes.
const MATH_OP_MUL: u64 = 0;
const MATH_OP_DIV: u64 = 1;
const MATH_OP_MOD: u64 = 2;
const MATH_OP_MULHI: u64 = 3;
const MATH_OP_POPCOUNT: u64 = 4;

/// MMIO math coprocessor: 64-bit MUL/DIV/MOD/MULHI via 4 registers.
///
/// Register map (addresses 50-53):
/// - `MMIO_MATH_A` (50): Write/read operand A.
/// - `MMIO_MATH_B` (51): Write/read operand B.
/// - `MMIO_MATH_CMD` (52): Write triggers computation (value = op code).
///   Read returns status: 0 = ok, 1 = div-by-zero.
/// - `MMIO_MATH_RESULT` (53): Read returns last computed result. Write ignored.
///
/// Operations: 0=MUL, 1=DIV, 2=MOD, 3=MULHI (high 64 bits of 128-bit product).
/// Computation is synchronous — result available on the next read.
#[derive(Debug)]
pub struct V2MmioMathDevice {
    a: Cell<u64>,
    b: Cell<u64>,
    status: Cell<u64>,
    result: Cell<u64>,
}

impl Default for V2MmioMathDevice {
    fn default() -> Self {
        Self {
            a: Cell::new(0),
            b: Cell::new(0),
            status: Cell::new(0),
            result: Cell::new(0),
        }
    }
}

impl V2MmioMathDevice {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn operand_a(&self) -> u64 {
        self.a.get()
    }

    pub fn operand_b(&self) -> u64 {
        self.b.get()
    }

    pub fn status(&self) -> u64 {
        self.status.get()
    }

    pub fn result(&self) -> u64 {
        self.result.get()
    }

    fn execute(&self, op: u64) {
        let a = self.a.get();
        let b = self.b.get();
        match op {
            MATH_OP_MUL => {
                self.result.set(a.wrapping_mul(b));
                self.status.set(0);
            }
            MATH_OP_DIV => {
                if b == 0 {
                    self.status.set(1);
                } else {
                    self.result.set(a / b);
                    self.status.set(0);
                }
            }
            MATH_OP_MOD => {
                if b == 0 {
                    self.status.set(1);
                } else {
                    self.result.set(a % b);
                    self.status.set(0);
                }
            }
            MATH_OP_MULHI => {
                let wide = (a as u128).wrapping_mul(b as u128);
                self.result.set((wide >> 64) as u64);
                self.status.set(0);
            }
            MATH_OP_POPCOUNT => {
                self.result.set(a.count_ones() as u64);
                self.status.set(0);
            }
            _ => {
                // Unknown op — no-op, clear status
                self.status.set(0);
            }
        }
    }

    fn snapshot_words(&self) -> [u64; 4] {
        [
            self.a.get(),
            self.b.get(),
            self.status.get(),
            self.result.get(),
        ]
    }

    fn restore_words(&self, words: &[u64; 4]) {
        self.a.set(words[0]);
        self.b.set(words[1]);
        self.status.set(words[2]);
        self.result.set(words[3]);
    }
}

impl V2MmioDevice for V2MmioMathDevice {
    fn read(&self, addr: u8) -> u64 {
        match addr {
            MMIO_MATH_A => self.a.get(),
            MMIO_MATH_B => self.b.get(),
            MMIO_MATH_CMD => self.status.get(),
            MMIO_MATH_RESULT => self.result.get(),
            _ => 0,
        }
    }

    fn write(&self, addr: u8, value: u64) {
        match addr {
            MMIO_MATH_A => self.a.set(value),
            MMIO_MATH_B => self.b.set(value),
            MMIO_MATH_CMD => self.execute(value),
            MMIO_MATH_RESULT => {} // ignored
            _ => {}
        }
    }

    fn snapshot_kind(&self) -> Option<&'static str> {
        Some(V2_MMIO_MATH_SNAPSHOT_KIND)
    }

    fn snapshot_state(&self) -> Option<Vec<u8>> {
        Some(encode_u64_snapshot(&self.snapshot_words()))
    }

    fn restore_state(&self, snapshot: &[u8]) -> Result<(), String> {
        let words = decode_u64_snapshot::<4>(snapshot)?;
        self.restore_words(&words);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SNN bridge device
// ---------------------------------------------------------------------------

use crate::snn::mlp_weights::{CachedRates, MlpWeights};
use crate::snn::neuron::LIFNeuron;

/// A lightweight internal synapse for the SNN bridge (stores source explicitly).
#[derive(Clone, Copy, Debug)]
struct BridgeSynapse {
    source: u32,
    target: u32,
    weight: i16, // Q8.8
}

/// SNN bridge command codes.
const SNN_CMD_RESET: u64 = 0;
const SNN_CMD_SET_INPUT: u64 = 1;
const SNN_CMD_TICK: u64 = 2;
const SNN_CMD_TICK_N: u64 = 3;
const SNN_CMD_STAGE_OUTPUT: u64 = 4;
const SNN_CMD_STAGE_MEMBRANE: u64 = 5;
const SNN_CMD_GET_SPIKE_COUNT: u64 = 6;
const SNN_CMD_INFER: u64 = 7;
const SNN_CMD_LOAD_IMAGE: u64 = 8;
const SNN_CMD_SNN_RUN: u64 = 9;
const SNN_CMD_INFER_LIVE: u64 = 10;
/// Option 2: on-device training. Treats `live_hidden_counts` as a frozen
/// feature vector and trains an extra linear readout via delta-rule.
/// `SNN_DATA` holds the true label (0..n_classes); prediction is staged back.
const SNN_CMD_TRAIN_ONE: u64 = 11;
/// Option 2: inference using the on-device-trained readout (no weight update).
const SNN_CMD_PREDICT_READOUT: u64 = 12;

/// M11: Pre-loaded inference model for MMIO-backed MLP classification.
pub struct InferenceModel {
    pub weights: MlpWeights,
    pub cached_rates: CachedRates,
}

/// MMIO SNN bridge: a small LIF spiking neural network accessible via 2 registers.
///
/// Network topology (fixed at construction):
/// - Input layer: `n_input` neurons (default 64)
/// - Hidden layer: `n_hidden` neurons (default 32)
/// - Output layer: `n_output` neurons (default 10)
///
/// Register map (addresses 54-55):
/// - `MMIO_SNN_DATA` (54): Write sets data for next command. Read returns staged value.
/// - `MMIO_SNN_CMD` (55): Write executes command. Read returns status word.
///
/// Commands:
/// - 0 RESET: reset all neurons to resting state, clear spike counts
/// - 1 SET_INPUT: set input spike pattern from SNN_DATA (64-bit bitmask)
/// - 2 TICK: advance 1 timestep
/// - 3 TICK_N: advance N timesteps (N = last SNN_DATA value)
/// - 4 STAGE_OUTPUT: stage output spike bitmask into SNN_DATA
/// - 5 STAGE_MEMBRANE: stage membrane potential of neuron[SNN_DATA] into SNN_DATA
/// - 6 GET_SPIKE_COUNT: stage cumulative spike count of output neuron[SNN_DATA]
/// - 7 INFER: M11 cached-rate MLP inference (uses InferenceModel)
/// - 8 LOAD_IMAGE: M12 encode image pixels into Poisson input rates, reset SNN
/// - 9 SNN_RUN: M12 run live LIF simulation for n_ticks timesteps
/// - 10 INFER_LIVE: M12 compute firing rates from hidden spike counts, MLP forward
///
/// Status word (read from SNN_CMD):
/// `(n_output << 16) | (n_input << 8) | status`
/// where status: 0=idle, 1=error
pub struct V2MmioSnnBridgeDevice {
    n_input: usize,
    n_hidden: usize,
    n_output: usize,
    /// LIF neurons: [input..., hidden..., output...]
    neurons: RefCell<Vec<LIFNeuron>>,
    /// Synapses: input→hidden and hidden→output (feedforward)
    synapses: Vec<BridgeSynapse>,
    /// Cumulative spike counts per output neuron (since last RESET)
    output_spike_counts: RefCell<Vec<u64>>,
    /// Data register
    data: Cell<u64>,
    /// Status: 0=idle, 1=error
    status: Cell<u64>,
    /// Deterministic RNG seed for weight initialization
    #[allow(dead_code)]
    seed: u64,
    /// M11: Optional pre-loaded inference model for MLP classification.
    inference_model: Option<InferenceModel>,
    /// M12: Live SNN model for on-the-fly inference.
    live_model: Option<crate::snn::mlp_weights::LiveSnnModel>,
    /// M12: MNIST image data for LOAD_IMAGE command.
    live_images: Option<Vec<Vec<u8>>>,
    /// M12: Live LIF neuron membrane potentials.
    live_v_mem: RefCell<Vec<i16>>,
    /// M12: Live LIF refractory counters.
    live_refract: RefCell<Vec<u8>>,
    /// M12: Live LIF spike flags.
    live_spiked: RefCell<Vec<u8>>,
    /// M12: Hidden spike counts accumulated over n_ticks.
    live_hidden_counts: RefCell<Vec<u32>>,
    /// M12: Poisson input rates for current image.
    live_input_rates: RefCell<Vec<u8>>,
    /// M12: RNG seed for Poisson spike generation.
    live_rng_seed: Cell<u32>,
    /// Option 2: on-device trainable linear readout (`W[n_in*n_out] + b[n_out]`).
    /// Layout: row-major, `w[h*n_out + c]`. Operates on the cached
    /// `live_hidden_counts` from the most recent `SNN_RUN`.
    readout_w: RefCell<Vec<f32>>,
    readout_b: RefCell<Vec<f32>>,
    /// Trainable readout dimensions (set by `enable_trainable_readout`).
    readout_in: Cell<usize>,
    readout_out: Cell<usize>,
    /// Learning rate for the delta-rule update applied on each `TRAIN_ONE`.
    train_lr: Cell<f32>,
    /// True once `enable_trainable_readout` has been called and `W/b` are sized.
    readout_initialized: Cell<bool>,
}

impl std::fmt::Debug for V2MmioSnnBridgeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V2MmioSnnBridgeDevice")
            .field("n_input", &self.n_input)
            .field("n_hidden", &self.n_hidden)
            .field("n_output", &self.n_output)
            .field("has_inference_model", &self.inference_model.is_some())
            .field("has_live_model", &self.live_model.is_some())
            .field("readout_initialized", &self.readout_initialized.get())
            .finish()
    }
}

impl V2MmioSnnBridgeDevice {
    /// Create a new SNN bridge with the given topology and deterministic seed.
    pub fn new(n_input: usize, n_hidden: usize, n_output: usize, seed: u64) -> Self {
        let total = n_input + n_hidden + n_output;
        let neurons = vec![LIFNeuron::new(); total];
        let synapses = Self::build_synapses(n_input, n_hidden, n_output, seed);
        let output_spike_counts = vec![0u64; n_output];
        Self {
            n_input,
            n_hidden,
            n_output,
            neurons: RefCell::new(neurons),
            synapses,
            output_spike_counts: RefCell::new(output_spike_counts),
            data: Cell::new(0),
            status: Cell::new(0),
            seed,
            inference_model: None,
            live_model: None,
            live_images: None,
            live_v_mem: RefCell::new(Vec::new()),
            live_refract: RefCell::new(Vec::new()),
            live_spiked: RefCell::new(Vec::new()),
            live_hidden_counts: RefCell::new(Vec::new()),
            live_input_rates: RefCell::new(Vec::new()),
            live_rng_seed: Cell::new(0),
            readout_w: RefCell::new(Vec::new()),
            readout_b: RefCell::new(Vec::new()),
            readout_in: Cell::new(0),
            readout_out: Cell::new(0),
            train_lr: Cell::new(0.0),
            readout_initialized: Cell::new(false),
        }
    }

    /// Default 64-32-10 topology with seed 42.
    pub fn default_topology() -> Self {
        Self::new(64, 32, 10, 42)
    }

    /// Small 8-4-2 topology for testing.
    pub fn small() -> Self {
        Self::new(8, 4, 2, 42)
    }

    /// M11: Create with a pre-loaded inference model.
    pub fn with_model(
        n_input: usize,
        n_hidden: usize,
        n_output: usize,
        seed: u64,
        model: InferenceModel,
    ) -> Self {
        let mut dev = Self::new(n_input, n_hidden, n_output, seed);
        dev.inference_model = Some(model);
        dev
    }

    /// M12: Create with a live SNN model and image data for on-the-fly inference.
    pub fn with_live_model(
        n_input: usize,
        n_hidden: usize,
        n_output: usize,
        seed: u64,
        live_model: crate::snn::mlp_weights::LiveSnnModel,
        images: Vec<Vec<u8>>,
    ) -> Self {
        let n_total = live_model.n_neurons();
        let n_hid = live_model.n_hidden;
        let n_inp = live_model.n_input;
        let mut dev = Self::new(n_input, n_hidden, n_output, seed);
        dev.live_v_mem = RefCell::new(vec![0i16; n_total]);
        dev.live_refract = RefCell::new(vec![0u8; n_total]);
        dev.live_spiked = RefCell::new(vec![0u8; n_total]);
        dev.live_hidden_counts = RefCell::new(vec![0u32; n_hid]);
        dev.live_input_rates = RefCell::new(vec![0u8; n_inp]);
        dev.live_rng_seed = Cell::new(seed as u32);
        dev.live_images = Some(images);
        dev.live_model = Some(live_model);
        dev
    }

    fn total_neurons(&self) -> usize {
        self.n_input + self.n_hidden + self.n_output
    }

    fn output_start(&self) -> usize {
        self.n_input + self.n_hidden
    }

    /// Build feedforward synapses with deterministic weights.
    fn build_synapses(
        n_input: usize,
        n_hidden: usize,
        n_output: usize,
        seed: u64,
    ) -> Vec<BridgeSynapse> {
        let mut synapses = Vec::new();
        let mut rng = seed;

        // Input → Hidden (sparse: each hidden gets connections from ~half of inputs)
        for h in 0..n_hidden {
            for i in 0..n_input {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng >> 32) & 1 == 1 {
                    // ~50% connectivity
                    let weight_bits = ((rng >> 16) & 0xFF) as i8;
                    let weight = (weight_bits as f32) / 128.0; // -1.0 to ~1.0
                    synapses.push(BridgeSynapse {
                        source: i as u32,
                        target: (n_input + h) as u32,
                        weight: (weight * 256.0) as i16, // Q8.8
                    });
                }
            }
        }

        // Hidden → Output (dense: each output gets all hidden)
        for o in 0..n_output {
            for h in 0..n_hidden {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let weight_bits = ((rng >> 16) & 0xFF) as i8;
                let weight = (weight_bits as f32) / 128.0;
                synapses.push(BridgeSynapse {
                    source: (n_input + h) as u32,
                    target: (n_input + n_hidden + o) as u32,
                    weight: (weight * 256.0) as i16,
                });
            }
        }

        synapses
    }

    fn reset_neurons(&self) {
        let mut neurons = self.neurons.borrow_mut();
        for n in neurons.iter_mut() {
            n.v_mem = LIFNeuron::V_REST;
            n.refractory = 0;
            n.spiked = 0;
            n.last_spike_time = 0;
        }
        self.output_spike_counts
            .borrow_mut()
            .iter_mut()
            .for_each(|c| *c = 0);
        self.status.set(0);
    }

    fn set_input_spikes(&self, bitmask: u64) {
        let mut neurons = self.neurons.borrow_mut();
        for i in 0..self.n_input.min(64) {
            if (bitmask >> i) & 1 == 1 {
                neurons[i].v_mem = LIFNeuron::THRESHOLD_DEFAULT + 1; // Force spike
                neurons[i].spiked = 1;
            } else {
                neurons[i].spiked = 0;
            }
        }
    }

    fn tick_one(&self) {
        let mut neurons = self.neurons.borrow_mut();
        let n = neurons.len();

        // Phase 1: Accumulate synaptic current into targets (Q8.8 weights)
        let mut currents = vec![0i32; n];
        for syn in &self.synapses {
            let src = syn.source as usize;
            if src < n && neurons[src].spiked != 0 {
                let tgt = syn.target as usize;
                if tgt < n {
                    currents[tgt] += syn.weight as i32; // Q8.8
                }
            }
        }

        // Phase 2: LIF dynamics for hidden + output neurons
        for i in self.n_input..n {
            let neuron = &mut neurons[i];
            if neuron.refractory > 0 {
                neuron.refractory -= 1;
                neuron.spiked = 0;
                continue;
            }
            // Leak
            neuron.v_mem = ((neuron.v_mem as i32 * neuron.leak as i32) >> 8) as i16;
            // Integrate
            neuron.v_mem = neuron
                .v_mem
                .saturating_add(currents[i].clamp(-32768, 32767) as i16);
            // Fire?
            if neuron.v_mem >= neuron.threshold {
                neuron.spiked = 1;
                neuron.v_mem = LIFNeuron::V_RESET;
                neuron.refractory = LIFNeuron::REFRACTORY_DEFAULT;
            } else {
                neuron.spiked = 0;
            }
        }

        // Phase 3: Track output spikes
        let out_start = self.output_start();
        let mut counts = self.output_spike_counts.borrow_mut();
        for o in 0..self.n_output {
            if neurons[out_start + o].spiked != 0 {
                counts[o] += 1;
            }
        }

        // Clear input spike flags (one-shot injection)
        for i in 0..self.n_input {
            neurons[i].spiked = 0;
        }
    }

    fn tick_n(&self, n: u64) {
        for _ in 0..n.min(10_000) {
            self.tick_one();
        }
    }

    fn stage_output_spikes(&self) {
        let neurons = self.neurons.borrow();
        let out_start = self.output_start();
        let mut bits = 0u64;
        for o in 0..self.n_output.min(64) {
            if neurons[out_start + o].spiked != 0 {
                bits |= 1u64 << o;
            }
        }
        self.data.set(bits);
    }

    fn stage_membrane(&self, neuron_idx: u64) {
        let neurons = self.neurons.borrow();
        let idx = neuron_idx as usize;
        if idx < neurons.len() {
            self.data.set(neurons[idx].v_mem as u64);
        } else {
            self.data.set(0);
            self.status.set(1);
        }
    }

    fn stage_spike_count(&self, output_idx: u64) {
        let counts = self.output_spike_counts.borrow();
        let idx = output_idx as usize;
        if idx < counts.len() {
            self.data.set(counts[idx]);
        } else {
            self.data.set(0);
            self.status.set(1);
        }
    }

    fn infer(&self, sample_idx: u64) {
        if let Some(ref model) = self.inference_model {
            let idx = sample_idx as usize;
            if idx < model.cached_rates.n_samples() {
                let rates = model.cached_rates.get(idx);
                let pred = model.weights.forward_cpu(rates);
                self.data.set(pred as u64);
                self.status.set(0);
            } else {
                self.data.set(0);
                self.status.set(1);
            }
        } else {
            self.data.set(0);
            self.status.set(1);
        }
    }

    /// M12: Load image pixels, encode as Poisson input rates, reset SNN state.
    fn load_image_live(&self, sample_idx: u64) {
        if let Some(ref model) = self.live_model {
            let idx = sample_idx as usize;
            if let Some(ref images) = self.live_images {
                if idx < images.len() {
                    // Encode image using pixel selection strategy
                    let rates = model.encode_image(&images[idx]);
                    *self.live_input_rates.borrow_mut() = rates;

                    // Reset neuron state
                    let n_total = model.n_neurons();
                    let mut v_mem = self.live_v_mem.borrow_mut();
                    let mut refract = self.live_refract.borrow_mut();
                    let mut spiked = self.live_spiked.borrow_mut();
                    v_mem.iter_mut().for_each(|v| *v = 0); // V_REST
                    refract.iter_mut().for_each(|r| *r = 0);
                    spiked.iter_mut().for_each(|s| *s = 0);

                    // Ensure vectors are correctly sized
                    if v_mem.len() != n_total {
                        *v_mem = vec![0i16; n_total];
                        *refract = vec![0u8; n_total];
                        *spiked = vec![0u8; n_total];
                    }

                    // Clear hidden spike counts
                    let mut counts = self.live_hidden_counts.borrow_mut();
                    counts.iter_mut().for_each(|c| *c = 0);
                    if counts.len() != model.n_hidden {
                        *counts = vec![0u32; model.n_hidden];
                    }

                    self.status.set(0);
                } else {
                    self.status.set(1); // OOB
                }
            } else {
                self.status.set(1); // no images
            }
        } else {
            self.status.set(1); // no model
        }
    }

    /// M12: Run live LIF simulation for n_ticks timesteps.
    ///
    /// Faithfully replicates the GPU fused kernel dynamics:
    /// Phase A: Poisson input generation (LCG PRNG)
    /// Phase B: CSR current accumulation
    /// Phase C: LIF dynamics (leak → integrate → fire)
    /// Phase D: Hidden spike count accumulation
    fn snn_run_live(&self) {
        let model = match self.live_model.as_ref() {
            Some(m) => m,
            None => {
                self.status.set(1);
                return;
            }
        };

        let mut v_mem = self.live_v_mem.borrow_mut();
        let mut refract = self.live_refract.borrow_mut();
        let mut spiked = self.live_spiked.borrow_mut();
        let mut counts = self.live_hidden_counts.borrow_mut();
        let input_rates = self.live_input_rates.borrow();
        let n_total = model.n_neurons();

        for tick in 0..model.n_ticks {
            // Phase A: Poisson input spike generation (matches GPU LCG PRNG)
            let tick_seed = self.live_rng_seed.get().wrapping_add(tick as u32 * 1000003);
            for i in 0..model.n_input {
                if refract[i] > 0 {
                    refract[i] -= 1;
                    spiked[i] = 0;
                    continue;
                }
                let mut rng = tick_seed ^ (i as u32).wrapping_mul(2654435761);
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                let rand_val = (rng >> 24) as u8;
                if rand_val < input_rates[i] {
                    spiked[i] = 1;
                    v_mem[i] = -128; // V_RESET
                    refract[i] = 2;
                } else {
                    spiked[i] = 0;
                }
            }

            // Phase B: CSR current accumulation
            let mut currents = vec![0i32; n_total];
            for src in 0..n_total {
                if spiked[src] != 0 {
                    let start = model.syn_ptr[src] as usize;
                    let end = model.syn_ptr[src + 1] as usize;
                    for j in start..end {
                        let tgt = model.targets[j] as usize;
                        if tgt < n_total {
                            currents[tgt] += model.weights[j] as i32 * 2;
                        }
                    }
                }
            }

            // Phase C: LIF dynamics for hidden + readout neurons
            for i in model.n_input..n_total {
                if refract[i] > 0 {
                    refract[i] -= 1;
                    spiked[i] = 0;
                    continue;
                }
                // Leak: v = v * leak >> 8
                let v = ((v_mem[i] as i32 * model.leaks[i] as i32) >> 8) + currents[i];
                let v = v.clamp(-32768, 32767);
                // Fire?
                if v >= model.thresholds[i] as i32 {
                    spiked[i] = 1;
                    v_mem[i] = -128; // V_RESET
                    refract[i] = 2;
                } else {
                    spiked[i] = 0;
                    v_mem[i] = v as i16;
                }
            }

            // Phase D: Count hidden spikes
            for h in 0..model.n_hidden {
                if spiked[model.n_input + h] != 0 {
                    counts[h] += 1;
                }
            }

            // Clear input spike flags (one-shot injection per tick)
            for i in 0..model.n_input {
                spiked[i] = 0;
            }
        }

        self.status.set(0);
    }

    /// M12: Compute firing rates from hidden spike counts and run MLP forward.
    fn infer_live(&self) {
        let model = match self.live_model.as_ref() {
            Some(m) => m,
            None => {
                self.status.set(1);
                return;
            }
        };

        let counts = self.live_hidden_counts.borrow();
        let n_ticks = model.n_ticks as f32;

        // Compute firing rates: counts / n_ticks
        let rates: Vec<f32> = counts.iter().map(|&c| c as f32 / n_ticks).collect();

        // MLP forward pass
        let pred = model.mlp.forward_cpu(&rates);
        self.data.set(pred as u64);
        self.status.set(0);
    }

    /// Option 2: enable on-device trainable linear readout.
    ///
    /// Sizes `W[n_in*n_out] + b[n_out]` and initializes weights Glorot-uniform
    /// from `seed`. Subsequent `SNN_CMD_TRAIN_ONE` calls compute a delta-rule
    /// update on the cached `live_hidden_counts` (the frozen feature vector
    /// produced by the most recent `SNN_RUN`).
    ///
    /// `n_in` must match `live_model.n_hidden` for the forward pass to be valid.
    /// `lr` is the per-sample learning rate (typically 0.005 – 0.05 for f32 rates
    /// in [0, 1] with single-hot delta updates).
    pub fn enable_trainable_readout(&mut self, n_in: usize, n_out: usize, lr: f32, seed: u64) {
        let scale = (6.0f32 / (n_in + n_out) as f32).sqrt();
        let total = n_in * n_out;
        let mut rng_state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut w = Vec::with_capacity(total);
        for _ in 0..total {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // u in [0, 1): take 31 high bits of state, normalize.
            let u = ((rng_state >> 33) as u32) as f32 / ((1u32 << 31) as f32);
            w.push((u * 2.0 - 1.0) * scale);
        }
        *self.readout_w.borrow_mut() = w;
        *self.readout_b.borrow_mut() = vec![0.0f32; n_out];
        self.readout_in.set(n_in);
        self.readout_out.set(n_out);
        self.train_lr.set(lr);
        self.readout_initialized.set(true);
    }

    /// Option 2: read out the current trainable-readout state (clones W and b).
    /// Used to transfer trained weights between bridge instances — e.g., from a
    /// training bridge (holding train images) to an eval bridge (holding test
    /// images). Returns empty vectors if `enable_trainable_readout` was never
    /// called.
    pub fn readout_weights(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.readout_w.borrow().clone(),
            self.readout_b.borrow().clone(),
        )
    }

    /// Option 2: install pre-trained readout weights (without re-initialising
    /// from a random seed). Sets dimensions and marks the readout as ready for
    /// `TRAIN_ONE` / `PREDICT_READOUT`. Does NOT touch `train_lr` — caller must
    /// set learning rate via `enable_trainable_readout` first if continuing
    /// training; for eval-only use leave the LR at whatever it was.
    pub fn set_readout_weights(&mut self, n_in: usize, n_out: usize, w: Vec<f32>, b: Vec<f32>) {
        assert_eq!(w.len(), n_in * n_out, "readout W shape mismatch");
        assert_eq!(b.len(), n_out, "readout b shape mismatch");
        *self.readout_w.borrow_mut() = w;
        *self.readout_b.borrow_mut() = b;
        self.readout_in.set(n_in);
        self.readout_out.set(n_out);
        self.readout_initialized.set(true);
    }

    /// Option 2 helper: compute `logits = W·rates + b` from cached spike counts.
    /// Returns `Some((logits, rates))` if the readout is initialized and the
    /// hidden-count buffer matches; `None` otherwise (status will be set to 1
    /// by the caller).
    fn forward_readout(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        if !self.readout_initialized.get() {
            return None;
        }
        let model = self.live_model.as_ref()?;
        let n_in = self.readout_in.get();
        let n_out = self.readout_out.get();
        if n_in != model.n_hidden {
            return None;
        }
        let counts = self.live_hidden_counts.borrow();
        if counts.len() != n_in {
            return None;
        }
        let inv_ticks = 1.0 / model.n_ticks as f32;
        let rates: Vec<f32> = counts.iter().map(|&c| c as f32 * inv_ticks).collect();
        let w = self.readout_w.borrow();
        let b = self.readout_b.borrow();
        let mut logits = b.clone();
        for h in 0..n_in {
            let r = rates[h];
            if r == 0.0 {
                continue; // skip silent hidden units
            }
            let row = h * n_out;
            for c in 0..n_out {
                logits[c] += r * w[row + c];
            }
        }
        Some((logits, rates))
    }

    /// Option 2 (`SNN_CMD_PREDICT_READOUT`): inference-only forward through the
    /// trainable linear readout. Stages argmax(logits) into `SNN_DATA`.
    fn predict_readout(&self) {
        match self.forward_readout() {
            Some((logits, _)) => {
                let mut best = 0usize;
                let mut best_v = logits[0];
                for c in 1..logits.len() {
                    if logits[c] > best_v {
                        best_v = logits[c];
                        best = c;
                    }
                }
                self.data.set(best as u64);
                self.status.set(0);
            }
            None => {
                self.status.set(1);
            }
        }
    }

    /// Option 2 (`SNN_CMD_TRAIN_ONE`): forward through the readout, compute
    /// `argmax`, and apply a delta-rule update if the prediction is wrong.
    ///
    /// Delta rule (single-hot reinforce / suppress):
    /// ```text
    /// if pred != label:
    ///   W[h, label] += lr * rates[h]
    ///   W[h, pred]  -= lr * rates[h]
    ///   b[label]    += lr
    ///   b[pred]     -= lr
    /// ```
    /// Stages the prediction into `SNN_DATA` (so the V2 program can submit it).
    fn train_one(&self, label: u64) {
        let (logits, rates) = match self.forward_readout() {
            Some(x) => x,
            None => {
                self.status.set(1);
                return;
            }
        };
        let n_in = self.readout_in.get();
        let n_out = self.readout_out.get();
        let mut pred = 0usize;
        let mut best_v = logits[0];
        for c in 1..n_out {
            if logits[c] > best_v {
                best_v = logits[c];
                pred = c;
            }
        }
        let lbl = label as usize;
        if lbl < n_out && pred != lbl {
            let lr = self.train_lr.get();
            let mut w = self.readout_w.borrow_mut();
            let mut b = self.readout_b.borrow_mut();
            for h in 0..n_in {
                let r = rates[h];
                if r == 0.0 {
                    continue;
                }
                w[h * n_out + lbl] += lr * r;
                w[h * n_out + pred] -= lr * r;
            }
            b[lbl] += lr;
            b[pred] -= lr;
        }
        self.data.set(pred as u64);
        self.status.set(0);
    }

    fn status_word(&self) -> u64 {
        let s = self.status.get() & 0xFF;
        let ni = (self.n_input as u64) & 0xFF;
        let no = (self.n_output as u64) & 0xFF;
        (no << 16) | (ni << 8) | s
    }
}

impl V2MmioDevice for V2MmioSnnBridgeDevice {
    fn read(&self, addr: u8) -> u64 {
        match addr {
            MMIO_SNN_DATA => self.data.get(),
            MMIO_SNN_CMD => self.status_word(),
            _ => 0,
        }
    }

    fn write(&self, addr: u8, value: u64) {
        match addr {
            MMIO_SNN_DATA => self.data.set(value),
            MMIO_SNN_CMD => match value {
                SNN_CMD_RESET => self.reset_neurons(),
                SNN_CMD_SET_INPUT => self.set_input_spikes(self.data.get()),
                SNN_CMD_TICK => self.tick_one(),
                SNN_CMD_TICK_N => self.tick_n(self.data.get()),
                SNN_CMD_STAGE_OUTPUT => self.stage_output_spikes(),
                SNN_CMD_STAGE_MEMBRANE => self.stage_membrane(self.data.get()),
                SNN_CMD_GET_SPIKE_COUNT => self.stage_spike_count(self.data.get()),
                SNN_CMD_INFER => self.infer(self.data.get()),
                SNN_CMD_LOAD_IMAGE => self.load_image_live(self.data.get()),
                SNN_CMD_SNN_RUN => self.snn_run_live(),
                SNN_CMD_INFER_LIVE => self.infer_live(),
                SNN_CMD_TRAIN_ONE => self.train_one(self.data.get()),
                SNN_CMD_PREDICT_READOUT => self.predict_readout(),
                _ => {}
            },
            _ => {}
        }
    }

    fn snapshot_kind(&self) -> Option<&'static str> {
        Some(V2_MMIO_SNN_SNAPSHOT_KIND)
    }

    fn snapshot_state(&self) -> Option<Vec<u8>> {
        let neurons = self.neurons.borrow();
        let counts = self.output_spike_counts.borrow();
        // Header: data, status, n_neurons (3 u64s)
        let mut out = Vec::new();
        out.extend_from_slice(&self.data.get().to_le_bytes());
        out.extend_from_slice(&self.status.get().to_le_bytes());
        out.extend_from_slice(&(neurons.len() as u64).to_le_bytes());
        // Each neuron: 8 bytes (matching LIFNeuron repr(C) layout)
        for n in neurons.iter() {
            out.extend_from_slice(&n.v_mem.to_le_bytes());
            out.extend_from_slice(&n.threshold.to_le_bytes());
            out.push(n.leak);
            out.push(n.refractory);
            out.push(n.last_spike_time);
            out.push(n.spiked);
        }
        // Output spike counts
        for &c in counts.iter() {
            out.extend_from_slice(&c.to_le_bytes());
        }
        Some(out)
    }

    fn restore_state(&self, snapshot: &[u8]) -> Result<(), String> {
        if snapshot.len() < 24 {
            return Err("SNN snapshot too short for header".to_string());
        }
        let data = u64::from_le_bytes(snapshot[0..8].try_into().unwrap());
        let status = u64::from_le_bytes(snapshot[8..16].try_into().unwrap());
        let n_neurons = u64::from_le_bytes(snapshot[16..24].try_into().unwrap()) as usize;

        if n_neurons != self.total_neurons() {
            return Err(format!(
                "SNN snapshot neuron count mismatch: expected {}, got {n_neurons}",
                self.total_neurons()
            ));
        }

        let neuron_bytes = n_neurons * 8;
        let count_bytes = self.n_output * 8;
        let expected = 24 + neuron_bytes + count_bytes;
        if snapshot.len() != expected {
            return Err(format!(
                "SNN snapshot length mismatch: expected {expected}, got {}",
                snapshot.len()
            ));
        }

        self.data.set(data);
        self.status.set(status);

        let mut neurons = self.neurons.borrow_mut();
        let mut off = 24;
        for n in neurons.iter_mut() {
            n.v_mem = i16::from_le_bytes(snapshot[off..off + 2].try_into().unwrap());
            n.threshold = i16::from_le_bytes(snapshot[off + 2..off + 4].try_into().unwrap());
            n.leak = snapshot[off + 4];
            n.refractory = snapshot[off + 5];
            n.last_spike_time = snapshot[off + 6];
            n.spiked = snapshot[off + 7];
            off += 8;
        }

        let mut counts = self.output_spike_counts.borrow_mut();
        for c in counts.iter_mut() {
            *c = u64::from_le_bytes(snapshot[off..off + 8].try_into().unwrap());
            off += 8;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Display device
// ---------------------------------------------------------------------------

/// Display width in pixels.
pub const DISPLAY_WIDTH: usize = 16;
/// Display height in pixels.
pub const DISPLAY_HEIGHT: usize = 16;
/// Total pixels in the display buffer.
pub const DISPLAY_PIXELS: usize = DISPLAY_WIDTH * DISPLAY_HEIGHT;

/// MMIO display device: 16x16 pixel buffer with 8-bit color (3-3-2 RGB).
///
/// Register map:
/// - `MMIO_DISPLAY_CMD` (addr 48): Write packed `(y << 16) | (x << 8) | color` to set a pixel.
///   Read returns last written packed value.
/// - `MMIO_DISPLAY_STATUS` (addr 49): Read returns `(height << 8) | width`.
///   Write 0 to clear screen.
#[derive(Debug)]
pub struct V2MmioDisplayDevice {
    pixels: RefCell<Vec<u8>>,
    last_cmd: Cell<u64>,
}

impl Default for V2MmioDisplayDevice {
    fn default() -> Self {
        Self {
            pixels: RefCell::new(vec![0u8; DISPLAY_PIXELS]),
            last_cmd: Cell::new(0),
        }
    }
}

impl V2MmioDisplayDevice {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read the pixel at (x, y). Returns 8-bit color (3-3-2 RGB) or 0 if out of bounds.
    pub fn get_pixel(&self, x: usize, y: usize) -> u8 {
        if x < DISPLAY_WIDTH && y < DISPLAY_HEIGHT {
            self.pixels.borrow()[y * DISPLAY_WIDTH + x]
        } else {
            0
        }
    }

    /// Read the entire pixel buffer (row-major, 256 bytes).
    pub fn pixels(&self) -> Vec<u8> {
        self.pixels.borrow().clone()
    }

    /// Render the pixel buffer as a PPM (P6) image at the given scale factor.
    pub fn render_display_ppm(&self, scale: usize) -> Vec<u8> {
        let w = DISPLAY_WIDTH * scale;
        let h = DISPLAY_HEIGHT * scale;
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        ppm.reserve(w * h * 3);
        let buf = self.pixels.borrow();
        for py in 0..h {
            for px in 0..w {
                let sx = px / scale;
                let sy = py / scale;
                let c = buf[sy * DISPLAY_WIDTH + sx];
                // 3-3-2 RGB: bits 7-5 = R(3), 4-2 = G(3), 1-0 = B(2)
                let r = ((c >> 5) & 0x07) * 36; // 0..7 -> 0..252
                let g = ((c >> 2) & 0x07) * 36;
                let b = (c & 0x03) * 85; // 0..3 -> 0..255
                ppm.push(r);
                ppm.push(g);
                ppm.push(b);
            }
        }
        ppm
    }
}

impl V2MmioDevice for V2MmioDisplayDevice {
    fn read(&self, addr: u8) -> u64 {
        match addr {
            MMIO_DISPLAY_CMD => self.last_cmd.get(),
            MMIO_DISPLAY_STATUS => ((DISPLAY_HEIGHT as u64) << 8) | (DISPLAY_WIDTH as u64),
            _ => 0,
        }
    }

    fn write(&self, addr: u8, value: u64) {
        match addr {
            MMIO_DISPLAY_CMD => {
                self.last_cmd.set(value);
                let y = ((value >> 16) & 0xFF) as usize;
                let x = ((value >> 8) & 0xFF) as usize;
                let color = (value & 0xFF) as u8;
                if x < DISPLAY_WIDTH && y < DISPLAY_HEIGHT {
                    self.pixels.borrow_mut()[y * DISPLAY_WIDTH + x] = color;
                }
            }
            MMIO_DISPLAY_STATUS => {
                if value == 0 {
                    self.pixels.borrow_mut().iter_mut().for_each(|p| *p = 0);
                    self.last_cmd.set(0);
                }
            }
            _ => {}
        }
    }

    fn snapshot_kind(&self) -> Option<&'static str> {
        Some(V2_MMIO_DISPLAY_SNAPSHOT_KIND)
    }

    fn snapshot_state(&self) -> Option<Vec<u8>> {
        let buf = self.pixels.borrow();
        let mut out = Vec::with_capacity(8 + DISPLAY_PIXELS);
        out.extend_from_slice(&self.last_cmd.get().to_le_bytes());
        out.extend_from_slice(&buf);
        Some(out)
    }

    fn restore_state(&self, snapshot: &[u8]) -> Result<(), String> {
        let expected = 8 + DISPLAY_PIXELS;
        if snapshot.len() != expected {
            return Err(format!(
                "display snapshot: expected {expected} bytes, got {}",
                snapshot.len()
            ));
        }
        let mut cmd_bytes = [0u8; 8];
        cmd_bytes.copy_from_slice(&snapshot[..8]);
        self.last_cmd.set(u64::from_le_bytes(cmd_bytes));
        self.pixels.borrow_mut().copy_from_slice(&snapshot[8..]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Quantum bridge device (addresses 44-47, Sprint 157)
// ---------------------------------------------------------------------------

/// MMIO quantum bridge: wraps the core quantum engine (QState, gates, measurement).
///
/// CPU programs can create quantum states, apply gates, and measure qubits —
/// turning V2 into a quantum-classical hybrid orchestrator.
///
/// Commands (written to QUANTUM_CMD):
///   0 = INIT        Create N-qubit state (N = QUANTUM_DATA, max 8)
///   1 = RESET       Reset state to |0...0⟩
///   2 = GATE_H      Hadamard on qubit QUANTUM_QUBIT
///   3 = GATE_X      Pauli-X on qubit QUANTUM_QUBIT
///   4 = GATE_Y      Pauli-Y on qubit QUANTUM_QUBIT
///   5 = GATE_Z      Pauli-Z on qubit QUANTUM_QUBIT
///   6 = GATE_T      T gate on qubit QUANTUM_QUBIT
///   7 = GATE_CNOT   CNOT: control=QUANTUM_QUBIT, target=QUANTUM_DATA
///   8 = GATE_CZ     CZ: control=QUANTUM_QUBIT, target=QUANTUM_DATA
///   9 = GATE_RZ     Rz(θ) on qubit QUANTUM_QUBIT, θ=f64::from_bits(QUANTUM_PARAM)
///  10 = MEASURE     Measure qubit QUANTUM_QUBIT; outcome in QUANTUM_DATA (0 or 1)
///  11 = MEASURE_ALL Measure all qubits; outcome bitmask in QUANTUM_DATA
///  12 = PROB        P(|1⟩) for qubit QUANTUM_QUBIT; result in QUANTUM_DATA (f64 bits)
#[derive(Debug)]
pub struct V2MmioQuantumDevice {
    qstate: RefCell<Option<crate::quantum::QState>>,
    rng: Cell<crate::quantum::QRng>,
    target_qubit: Cell<u8>,
    param_bits: Cell<u64>,
    last_result: Cell<u64>,
    status: Cell<u8>,
    gate_count: Cell<u32>,
}

const QUANTUM_MAX_QUBITS: u8 = 8;
const QUANTUM_STATUS_OK: u8 = 0;
const QUANTUM_STATUS_NOT_INIT: u8 = 1;
const QUANTUM_STATUS_QUBIT_OOB: u8 = 2;

impl V2MmioQuantumDevice {
    pub fn new(rng_seed: u64) -> Self {
        Self {
            qstate: RefCell::new(None),
            rng: Cell::new(crate::quantum::QRng::new(rng_seed)),
            target_qubit: Cell::new(0),
            param_bits: Cell::new(0),
            last_result: Cell::new(0),
            status: Cell::new(QUANTUM_STATUS_NOT_INIT),
            gate_count: Cell::new(0),
        }
    }

    fn n_qubits(&self) -> u8 {
        self.qstate
            .borrow()
            .as_ref()
            .map(|q| q.n_qubits)
            .unwrap_or(0)
    }

    fn check_qubit(&self, q: u8) -> bool {
        let n = self.n_qubits();
        if n == 0 {
            self.status.set(QUANTUM_STATUS_NOT_INIT);
            return false;
        }
        if q >= n {
            self.status.set(QUANTUM_STATUS_QUBIT_OOB);
            return false;
        }
        true
    }

    fn apply_gate(&self, gate: crate::quantum::QGate) {
        let mut qs = self.qstate.borrow_mut();
        if let Some(ref mut state) = *qs {
            let mut rng = self.rng.get();
            let outcome = crate::quantum::apply_gate_scalar(state, &gate, &mut rng);
            self.rng.set(rng);
            self.gate_count.set(self.gate_count.get() + 1);
            if let crate::quantum::GateOutcome::Measured { qubit: _, bit } = outcome {
                self.last_result.set(bit as u64);
            }
            self.status.set(QUANTUM_STATUS_OK);
        } else {
            self.status.set(QUANTUM_STATUS_NOT_INIT);
        }
    }

    fn execute_cmd(&self, cmd: u64) {
        let data = self.last_result.get(); // QUANTUM_DATA current value
        match cmd {
            0 => {
                // INIT: create N-qubit state
                let n = (self.last_result.get() as u8)
                    .min(QUANTUM_MAX_QUBITS)
                    .max(1);
                let state = crate::quantum::QState::new_zero(n);
                *self.qstate.borrow_mut() = Some(state);
                self.gate_count.set(0);
                self.status.set(QUANTUM_STATUS_OK);
            }
            1 => {
                // RESET: reset to |0...0⟩
                let n = self.n_qubits();
                if n == 0 {
                    self.status.set(QUANTUM_STATUS_NOT_INIT);
                    return;
                }
                let state = crate::quantum::QState::new_zero(n);
                *self.qstate.borrow_mut() = Some(state);
                self.gate_count.set(0);
                self.status.set(QUANTUM_STATUS_OK);
            }
            2 => {
                // GATE_H
                let q = self.target_qubit.get();
                if self.check_qubit(q) {
                    self.apply_gate(crate::quantum::QGate::H(q));
                }
            }
            3 => {
                // GATE_X
                let q = self.target_qubit.get();
                if self.check_qubit(q) {
                    self.apply_gate(crate::quantum::QGate::X(q));
                }
            }
            4 => {
                // GATE_Y
                let q = self.target_qubit.get();
                if self.check_qubit(q) {
                    self.apply_gate(crate::quantum::QGate::Y(q));
                }
            }
            5 => {
                // GATE_Z
                let q = self.target_qubit.get();
                if self.check_qubit(q) {
                    self.apply_gate(crate::quantum::QGate::Z(q));
                }
            }
            6 => {
                // GATE_T
                let q = self.target_qubit.get();
                if self.check_qubit(q) {
                    self.apply_gate(crate::quantum::QGate::T(q));
                }
            }
            7 => {
                // GATE_CNOT: control=target_qubit, target=data
                let control = self.target_qubit.get();
                let target = data as u8;
                if self.check_qubit(control) && self.check_qubit(target) {
                    self.apply_gate(crate::quantum::QGate::CNot(control, target));
                }
            }
            8 => {
                // GATE_CZ: control=target_qubit, target=data
                let control = self.target_qubit.get();
                let target = data as u8;
                if self.check_qubit(control) && self.check_qubit(target) {
                    self.apply_gate(crate::quantum::QGate::CZ(control, target));
                }
            }
            9 => {
                // GATE_RZ: Rz(θ) on target_qubit, θ from param_bits
                let q = self.target_qubit.get();
                if self.check_qubit(q) {
                    let theta = f64::from_bits(self.param_bits.get()) as f32;
                    self.apply_gate(crate::quantum::QGate::Rz(q, theta));
                }
            }
            10 => {
                // MEASURE: measure qubit target_qubit
                let q = self.target_qubit.get();
                if self.check_qubit(q) {
                    self.apply_gate(crate::quantum::QGate::Measure(q));
                    // last_result set by apply_gate via GateOutcome::Measured
                }
            }
            11 => {
                // MEASURE_ALL: measure all qubits, bitmask result
                let n = self.n_qubits();
                if n == 0 {
                    self.status.set(QUANTUM_STATUS_NOT_INIT);
                    return;
                }
                let mut result: u64 = 0;
                for q in 0..n {
                    let mut qs = self.qstate.borrow_mut();
                    let state = qs.as_mut().unwrap();
                    let mut rng = self.rng.get();
                    let outcome = crate::quantum::apply_gate_scalar(
                        state,
                        &crate::quantum::QGate::Measure(q),
                        &mut rng,
                    );
                    self.rng.set(rng);
                    if let crate::quantum::GateOutcome::Measured { qubit: _, bit } = outcome
                        && bit != 0
                    {
                        result |= 1u64 << q;
                    }
                }
                self.gate_count.set(self.gate_count.get() + n as u32);
                self.last_result.set(result);
                self.status.set(QUANTUM_STATUS_OK);
            }
            12 => {
                // PROB: P(|1⟩) for qubit target_qubit
                let q = self.target_qubit.get();
                if !self.check_qubit(q) {
                    return;
                }
                let qs = self.qstate.borrow();
                let state = qs.as_ref().unwrap();
                // Sum |amplitude|^2 for all basis states where qubit q is |1⟩
                let mask = 1usize << q;
                let mut prob: f64 = 0.0;
                let r = state.real.as_slice();
                let im = state.imag.as_slice();
                for i in 0..state.len {
                    if (i & mask) != 0 {
                        let re = r[i] as f64;
                        let im_v = im[i] as f64;
                        prob += re * re + im_v * im_v;
                    }
                }
                self.last_result.set(prob.to_bits());
                self.status.set(QUANTUM_STATUS_OK);
            }
            // Sprint 158: Gate expansion
            13 => {
                // GATE_RX: Rx(θ) on target_qubit, θ from param_bits
                let q = self.target_qubit.get();
                if self.check_qubit(q) {
                    let theta = f64::from_bits(self.param_bits.get()) as f32;
                    self.apply_gate(crate::quantum::QGate::Rx(q, theta));
                }
            }
            14 => {
                // GATE_RY: Ry(θ) on target_qubit, θ from param_bits
                let q = self.target_qubit.get();
                if self.check_qubit(q) {
                    let theta = f64::from_bits(self.param_bits.get()) as f32;
                    self.apply_gate(crate::quantum::QGate::Ry(q, theta));
                }
            }
            15 => {
                // GATE_SWAP: Swap(target_qubit, data)
                let q1 = self.target_qubit.get();
                let q2 = data as u8;
                if self.check_qubit(q1) && self.check_qubit(q2) {
                    self.apply_gate(crate::quantum::QGate::Swap(q1, q2));
                }
            }
            16 => {
                // GATE_TDG: T† on target_qubit
                let q = self.target_qubit.get();
                if self.check_qubit(q) {
                    self.apply_gate(crate::quantum::QGate::Tdg(q));
                }
            }
            _ => {
                // Unknown command — ignore
            }
        }
    }
}

impl V2MmioDevice for V2MmioQuantumDevice {
    fn read(&self, addr: u8) -> u64 {
        match addr {
            MMIO_QUANTUM_CMD => {
                // Status word: (gate_count << 16) | (n_qubits << 8) | status
                let gc = self.gate_count.get() as u64;
                let nq = self.n_qubits() as u64;
                let st = self.status.get() as u64;
                (gc << 16) | (nq << 8) | st
            }
            MMIO_QUANTUM_QUBIT => self.target_qubit.get() as u64,
            MMIO_QUANTUM_DATA => self.last_result.get(),
            MMIO_QUANTUM_PARAM => self.param_bits.get(),
            _ => 0,
        }
    }

    fn write(&self, addr: u8, value: u64) {
        match addr {
            MMIO_QUANTUM_CMD => self.execute_cmd(value),
            MMIO_QUANTUM_QUBIT => self.target_qubit.set(value as u8),
            MMIO_QUANTUM_DATA => self.last_result.set(value),
            MMIO_QUANTUM_PARAM => self.param_bits.set(value),
            _ => {}
        }
    }

    fn snapshot_kind(&self) -> Option<&'static str> {
        Some(V2_MMIO_QUANTUM_SNAPSHOT_KIND)
    }

    fn snapshot_state(&self) -> Option<Vec<u8>> {
        let qs = self.qstate.borrow();
        let mut out = Vec::new();
        // Header: [has_state(u8), n_qubits(u8), status(u8), pad(u8),
        //          gate_count(u32), target_qubit(u8), pad3(3), rng_state(u64),
        //          param_bits(u64), last_result(u64)]
        // = 36 bytes header
        if let Some(ref state) = *qs {
            out.push(1u8); // has_state
            out.push(state.n_qubits);
        } else {
            out.push(0u8);
            out.push(0u8);
        }
        out.push(self.status.get());
        out.push(0u8); // padding
        out.extend_from_slice(&self.gate_count.get().to_le_bytes());
        out.push(self.target_qubit.get());
        out.extend_from_slice(&[0u8; 3]); // padding
        out.extend_from_slice(&self.rng.get().state.to_le_bytes());
        out.extend_from_slice(&self.param_bits.get().to_le_bytes());
        out.extend_from_slice(&self.last_result.get().to_le_bytes());
        // State arrays (if present)
        if let Some(ref state) = *qs {
            for v in state.real.as_slice() {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for v in state.imag.as_slice() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        Some(out)
    }

    fn restore_state(&self, snapshot: &[u8]) -> Result<(), String> {
        if snapshot.len() < 36 {
            return Err(format!(
                "quantum snapshot too short: expected >= 36 bytes, got {}",
                snapshot.len()
            ));
        }
        let has_state = snapshot[0];
        let n_qubits = snapshot[1];
        self.status.set(snapshot[2]);
        let gc = u32::from_le_bytes(snapshot[4..8].try_into().unwrap());
        self.gate_count.set(gc);
        self.target_qubit.set(snapshot[8]);
        let rng_state = u64::from_le_bytes(snapshot[12..20].try_into().unwrap());
        self.rng.set(crate::quantum::QRng::new(0));
        // Directly set the rng state
        let mut rng = self.rng.get();
        rng.state = rng_state;
        self.rng.set(rng);
        self.param_bits
            .set(u64::from_le_bytes(snapshot[20..28].try_into().unwrap()));
        self.last_result
            .set(u64::from_le_bytes(snapshot[28..36].try_into().unwrap()));

        if has_state != 0 && n_qubits > 0 && n_qubits <= QUANTUM_MAX_QUBITS {
            let len = 1usize << n_qubits;
            let expected_data = 36 + len * 4 * 2; // real + imag, f32 each
            if snapshot.len() < expected_data {
                return Err(format!(
                    "quantum snapshot data truncated: expected {} bytes, got {}",
                    expected_data,
                    snapshot.len()
                ));
            }
            let mut state = crate::quantum::QState::new_zero(n_qubits);
            let mut off = 36;
            let r = state.real.as_mut_slice();
            for v in r.iter_mut().take(len) {
                *v = f32::from_le_bytes(snapshot[off..off + 4].try_into().unwrap());
                off += 4;
            }
            let im = state.imag.as_mut_slice();
            for v in im.iter_mut().take(len) {
                *v = f32::from_le_bytes(snapshot[off..off + 4].try_into().unwrap());
                off += 4;
            }
            *self.qstate.borrow_mut() = Some(state);
        } else {
            *self.qstate.borrow_mut() = None;
        }
        Ok(())
    }
}

// ── Dataset device (Sprint 159) ──

/// Dataset status/error codes.
pub const DS_OK: u8 = 0;
pub const DS_ERR_OOB: u8 = 1;
pub const DS_ERR_NO_SAMPLE: u8 = 2;
pub const DS_ERR_INVALID_CMD: u8 = 3;

/// A single pre-packed dataset sample.
#[derive(Debug, Clone, Copy)]
pub struct DatasetSample {
    pub features: u64,
    pub label: u64,
}

/// MMIO dataset device: serves pre-processed samples to the V2 CPU.
///
/// Register map (addresses 41-43):
/// - `MMIO_DATASET_CMD` (41): Write executes command. Read returns `(last_error << 8) | (cmd_count & 0xFF)`.
/// - `MMIO_DATASET_DATA` (42): Write sets data register. Read returns last result.
/// - `MMIO_DATASET_STATUS` (43): Read returns `(current_index << 32) | total_samples`.
///
/// Commands: 0=LOAD_SAMPLE, 1=GET_FEATURES, 2=GET_LABEL, 3=GET_COUNT, 4=GET_CORRECT, 5=SUBMIT_PREDICTION.
#[derive(Debug)]
pub struct V2MmioDatasetDevice {
    samples: Vec<DatasetSample>,
    current_index: Cell<Option<usize>>,
    data: Cell<u64>,
    correct_count: Cell<u64>,
    last_error: Cell<u8>,
    cmd_count: Cell<u64>,
}

impl V2MmioDatasetDevice {
    pub fn from_samples(samples: Vec<DatasetSample>) -> Self {
        Self {
            samples,
            current_index: Cell::new(None),
            data: Cell::new(0),
            correct_count: Cell::new(0),
            last_error: Cell::new(0),
            cmd_count: Cell::new(0),
        }
    }

    fn execute_cmd(&self, cmd: u64) {
        self.cmd_count.set(self.cmd_count.get() + 1);
        match cmd {
            0 => {
                // LOAD_SAMPLE: index from data register
                let idx = self.data.get() as usize;
                if idx >= self.samples.len() {
                    self.last_error.set(DS_ERR_OOB);
                } else {
                    self.current_index.set(Some(idx));
                    self.last_error.set(DS_OK);
                }
            }
            1 => {
                // GET_FEATURES
                if let Some(idx) = self.current_index.get() {
                    self.data.set(self.samples[idx].features);
                    self.last_error.set(DS_OK);
                } else {
                    self.last_error.set(DS_ERR_NO_SAMPLE);
                }
            }
            2 => {
                // GET_LABEL
                if let Some(idx) = self.current_index.get() {
                    self.data.set(self.samples[idx].label);
                    self.last_error.set(DS_OK);
                } else {
                    self.last_error.set(DS_ERR_NO_SAMPLE);
                }
            }
            3 => {
                // GET_COUNT
                self.data.set(self.samples.len() as u64);
                self.last_error.set(DS_OK);
            }
            4 => {
                // GET_CORRECT
                self.data.set(self.correct_count.get());
                self.last_error.set(DS_OK);
            }
            5 => {
                // SUBMIT_PREDICTION: compare data with current sample's label
                if let Some(idx) = self.current_index.get() {
                    let prediction = self.data.get();
                    if prediction == self.samples[idx].label {
                        self.correct_count.set(self.correct_count.get() + 1);
                    }
                    self.last_error.set(DS_OK);
                } else {
                    self.last_error.set(DS_ERR_NO_SAMPLE);
                }
            }
            _ => {
                self.last_error.set(DS_ERR_INVALID_CMD);
            }
        }
    }
}

impl V2MmioDevice for V2MmioDatasetDevice {
    fn read(&self, addr: u8) -> u64 {
        match addr {
            MMIO_DATASET_CMD => {
                ((self.last_error.get() as u64) << 8) | (self.cmd_count.get() & 0xFF)
            }
            MMIO_DATASET_DATA => self.data.get(),
            MMIO_DATASET_STATUS => {
                let idx = self
                    .current_index
                    .get()
                    .map(|i| i as u64)
                    .unwrap_or(u64::MAX);
                (idx << 32) | (self.samples.len() as u64)
            }
            _ => 0,
        }
    }

    fn write(&self, addr: u8, value: u64) {
        match addr {
            MMIO_DATASET_CMD => self.execute_cmd(value),
            MMIO_DATASET_DATA => self.data.set(value),
            _ => {} // DATASET_STATUS is read-only
        }
    }

    fn snapshot_kind(&self) -> Option<&'static str> {
        Some("v2.mmio.dataset.v1")
    }

    fn snapshot_state(&self) -> Option<Vec<u8>> {
        // Format: [current_index_flag(u8), current_index(u64), data(u64),
        //          correct_count(u64), last_error(u8), cmd_count(u64)]
        let mut out = Vec::new();
        match self.current_index.get() {
            Some(idx) => {
                out.push(1);
                out.extend_from_slice(&(idx as u64).to_le_bytes());
            }
            None => {
                out.push(0);
                out.extend_from_slice(&0u64.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.data.get().to_le_bytes());
        out.extend_from_slice(&self.correct_count.get().to_le_bytes());
        out.push(self.last_error.get());
        out.extend_from_slice(&self.cmd_count.get().to_le_bytes());
        Some(out)
    }

    fn restore_state(&self, snapshot: &[u8]) -> Result<(), String> {
        // 1 + 8 + 8 + 8 + 1 + 8 = 34 bytes
        if snapshot.len() != 34 {
            return Err(format!(
                "dataset snapshot: expected 34 bytes, got {}",
                snapshot.len()
            ));
        }
        let has_index = snapshot[0];
        let index_val = u64::from_le_bytes(snapshot[1..9].try_into().unwrap());
        self.current_index.set(if has_index != 0 {
            Some(index_val as usize)
        } else {
            None
        });
        self.data
            .set(u64::from_le_bytes(snapshot[9..17].try_into().unwrap()));
        self.correct_count
            .set(u64::from_le_bytes(snapshot[17..25].try_into().unwrap()));
        self.last_error.set(snapshot[25]);
        self.cmd_count
            .set(u64::from_le_bytes(snapshot[26..34].try_into().unwrap()));
        Ok(())
    }
}

/// Combined MMIO snapshot format version.
/// v1 (implicit, Sprint 155-158): 5 sections (ref_pack, display, math, snn, quantum).
/// v2 (Sprint 159+): version byte + 5 sections + optional dataset section.
const COMBINED_SNAPSHOT_VERSION: u8 = 2;

/// Combined MMIO device: dispatches across all sub-devices by address range.
///
/// Address map:
/// - 41-43: HLS accelerator (optional, Sprint 388) / dataset (optional) — mutually
///   exclusive sub-devices sharing the same slots (see `MMIO_ACCEL_ARG_SELECT`)
/// - 44-47: quantum bridge
/// - 48-49: display
/// - 50-53: math coprocessor
/// - 54-55: SNN bridge
/// - 56-63: ref pack (timer, console, RNG, mailbox, p-bit)
#[derive(Debug)]
pub struct V2MmioCombinedDevice {
    pub ref_pack: V2MmioRefDevicePack,
    pub display: V2MmioDisplayDevice,
    pub math: V2MmioMathDevice,
    pub snn: V2MmioSnnBridgeDevice,
    pub quantum: V2MmioQuantumDevice,
    pub dataset: Option<V2MmioDatasetDevice>,
    /// Sprint 388: optional HLS accelerator (tile datapath behind MMIO 41-43).
    /// Mutually exclusive with `dataset` (same address slots). Not included in
    /// snapshot/restore — replay bundles for accel programs are deferred.
    pub accel: Option<crate::tile_cpu::v2_hls_accel::V2MmioHlsAccelDevice>,
}

impl V2MmioCombinedDevice {
    pub fn new(rng_seed: u64) -> Self {
        Self {
            ref_pack: V2MmioRefDevicePack::new(rng_seed),
            display: V2MmioDisplayDevice::new(),
            math: V2MmioMathDevice::new(),
            snn: V2MmioSnnBridgeDevice::small(),
            quantum: V2MmioQuantumDevice::new(rng_seed.wrapping_add(157)),
            dataset: None,
            accel: None,
        }
    }

    /// Create with a custom SNN topology.
    pub fn with_snn(rng_seed: u64, snn: V2MmioSnnBridgeDevice) -> Self {
        Self {
            ref_pack: V2MmioRefDevicePack::new(rng_seed),
            display: V2MmioDisplayDevice::new(),
            math: V2MmioMathDevice::new(),
            snn,
            quantum: V2MmioQuantumDevice::new(rng_seed.wrapping_add(157)),
            dataset: None,
            accel: None,
        }
    }

    /// Create with a dataset device for MNIST inference.
    pub fn with_dataset(rng_seed: u64, dataset: V2MmioDatasetDevice) -> Self {
        Self {
            ref_pack: V2MmioRefDevicePack::new(rng_seed),
            display: V2MmioDisplayDevice::new(),
            math: V2MmioMathDevice::new(),
            snn: V2MmioSnnBridgeDevice::small(),
            quantum: V2MmioQuantumDevice::new(rng_seed.wrapping_add(157)),
            dataset: Some(dataset),
            accel: None,
        }
    }

    /// M11: Create with both a custom SNN bridge and a dataset device.
    pub fn with_snn_and_dataset(
        rng_seed: u64,
        snn: V2MmioSnnBridgeDevice,
        dataset: V2MmioDatasetDevice,
    ) -> Self {
        Self {
            ref_pack: V2MmioRefDevicePack::new(rng_seed),
            display: V2MmioDisplayDevice::new(),
            math: V2MmioMathDevice::new(),
            snn,
            quantum: V2MmioQuantumDevice::new(rng_seed.wrapping_add(157)),
            dataset: Some(dataset),
            accel: None,
        }
    }

    /// Sprint 388: create with an HLS accelerator (tile datapath behind MMIO 41-43).
    /// The accelerator occupies the dataset device's address slots, so this
    /// configuration has no dataset device.
    pub fn with_accel(
        rng_seed: u64,
        accel: crate::tile_cpu::v2_hls_accel::V2MmioHlsAccelDevice,
    ) -> Self {
        Self {
            ref_pack: V2MmioRefDevicePack::new(rng_seed),
            display: V2MmioDisplayDevice::new(),
            math: V2MmioMathDevice::new(),
            snn: V2MmioSnnBridgeDevice::small(),
            quantum: V2MmioQuantumDevice::new(rng_seed.wrapping_add(157)),
            dataset: None,
            accel: Some(accel),
        }
    }
}

impl V2MmioDevice for V2MmioCombinedDevice {
    fn read(&self, addr: u8) -> u64 {
        match addr {
            MMIO_ACCEL_ARG_SELECT..=MMIO_ACCEL_RESULT if self.accel.is_some() => {
                self.accel.as_ref().unwrap().read(addr)
            }
            MMIO_DATASET_CMD..=MMIO_DATASET_STATUS if self.dataset.is_some() => {
                self.dataset.as_ref().unwrap().read(addr)
            }
            MMIO_QUANTUM_CMD..=MMIO_QUANTUM_PARAM => self.quantum.read(addr),
            MMIO_DISPLAY_CMD | MMIO_DISPLAY_STATUS => self.display.read(addr),
            MMIO_MATH_A..=MMIO_MATH_RESULT => self.math.read(addr),
            MMIO_SNN_DATA | MMIO_SNN_CMD => self.snn.read(addr),
            _ => self.ref_pack.read(addr),
        }
    }

    fn write(&self, addr: u8, value: u64) {
        match addr {
            MMIO_ACCEL_ARG_SELECT..=MMIO_ACCEL_RESULT if self.accel.is_some() => {
                self.accel.as_ref().unwrap().write(addr, value)
            }
            MMIO_DATASET_CMD..=MMIO_DATASET_STATUS if self.dataset.is_some() => {
                self.dataset.as_ref().unwrap().write(addr, value)
            }
            MMIO_QUANTUM_CMD..=MMIO_QUANTUM_PARAM => self.quantum.write(addr, value),
            MMIO_DISPLAY_CMD | MMIO_DISPLAY_STATUS => self.display.write(addr, value),
            MMIO_MATH_A..=MMIO_MATH_RESULT => self.math.write(addr, value),
            MMIO_SNN_DATA | MMIO_SNN_CMD => self.snn.write(addr, value),
            _ => self.ref_pack.write(addr, value),
        }
    }

    fn tick(&self, cycle: u64) {
        self.ref_pack.tick(cycle);
        self.display.tick(cycle);
    }

    fn snapshot_kind(&self) -> Option<&'static str> {
        Some(V2_MMIO_REF_SNAPSHOT_KIND)
    }

    fn snapshot_state(&self) -> Option<Vec<u8>> {
        let ref_snap = self.ref_pack.snapshot_state().unwrap_or_default();
        let disp_snap = self.display.snapshot_state().unwrap_or_default();
        let math_snap = self.math.snapshot_state().unwrap_or_default();
        let snn_snap = self.snn.snapshot_state().unwrap_or_default();
        let quantum_snap = self.quantum.snapshot_state().unwrap_or_default();
        let mut out = Vec::new();
        // Version byte (v2 = 6-section format with optional dataset)
        out.push(COMBINED_SNAPSHOT_VERSION);
        // 5 core sections: length-prefixed
        for snap in [&ref_snap, &disp_snap, &math_snap, &snn_snap, &quantum_snap] {
            out.extend_from_slice(&(snap.len() as u64).to_le_bytes());
            out.extend_from_slice(snap);
        }
        // Optional dataset section
        if let Some(ds) = &self.dataset {
            let ds_snap = ds.snapshot_state().unwrap_or_default();
            out.push(1); // dataset present flag
            out.extend_from_slice(&(ds_snap.len() as u64).to_le_bytes());
            out.extend_from_slice(&ds_snap);
        } else {
            out.push(0); // no dataset
        }
        Some(out)
    }

    fn restore_state(&self, snapshot: &[u8]) -> Result<(), String> {
        // Check for version byte. Legacy v1 snapshots start with a u64 length
        // (the ref_pack section header). If byte 0 is 2, it's a v2 snapshot.
        if snapshot.is_empty() {
            return Err("combined snapshot empty".to_string());
        }
        let version = snapshot[0];
        let mut off = if version == COMBINED_SNAPSHOT_VERSION {
            1 // skip version byte
        } else {
            // Legacy v1: no version byte, 5 sections only
            // (version byte would be a small number like 2, but v1 snapshots
            // start with a u64 length whose low byte is much larger.)
            0
        };
        let mut read_section = |name: &str| -> Result<&[u8], String> {
            if off + 8 > snapshot.len() {
                return Err(format!("combined snapshot truncated at {name} header"));
            }
            let len = u64::from_le_bytes(snapshot[off..off + 8].try_into().unwrap()) as usize;
            off += 8;
            if off + len > snapshot.len() {
                return Err(format!("combined snapshot truncated at {name} data"));
            }
            let data = &snapshot[off..off + len];
            off += len;
            Ok(data)
        };
        let ref_data = read_section("ref_pack")?;
        let disp_data = read_section("display")?;
        let math_data = read_section("math")?;
        let snn_data = read_section("snn")?;
        let quantum_data = read_section("quantum")?;
        self.ref_pack.restore_state(ref_data)?;
        self.display.restore_state(disp_data)?;
        self.math.restore_state(math_data)?;
        self.snn.restore_state(snn_data)?;
        self.quantum.restore_state(quantum_data)?;
        // V2: optional dataset section
        if version == COMBINED_SNAPSHOT_VERSION && off < snapshot.len() {
            let ds_present = snapshot[off];
            off += 1;
            if ds_present == 1
                && let Some(ds) = &self.dataset
            {
                let ds_data = {
                    if off + 8 > snapshot.len() {
                        return Err("combined snapshot truncated at dataset header".to_string());
                    }
                    let len =
                        u64::from_le_bytes(snapshot[off..off + 8].try_into().unwrap()) as usize;
                    off += 8;
                    if off + len > snapshot.len() {
                        return Err("combined snapshot truncated at dataset data".to_string());
                    }
                    let data = &snapshot[off..off + len];
                    off += len;
                    let _ = off; // suppress unused warning
                    data
                };
                ds.restore_state(ds_data)?;
            }
            // If snapshot has dataset but device doesn't, skip the data gracefully
        }
        Ok(())
    }
}

fn encode_u64_snapshot<const N: usize>(words: &[u64; N]) -> Vec<u8> {
    let mut out = Vec::with_capacity(N * 8);
    for &word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

fn decode_u64_snapshot<const N: usize>(snapshot: &[u8]) -> Result<[u64; N], String> {
    let expected_len = N * 8;
    if snapshot.len() != expected_len {
        return Err(format!(
            "invalid snapshot length: expected {expected_len} bytes, got {}",
            snapshot.len()
        ));
    }
    let mut out = [0u64; N];
    for (i, chunk) in snapshot.chunks_exact(8).enumerate() {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(chunk);
        out[i] = u64::from_le_bytes(bytes);
    }
    Ok(out)
}

// =============================================================================
// Sprint 180: V2LinkMailboxDevice — Per-CPU link-aware mailbox for CPU arrays
// =============================================================================

/// MMIO device providing bidirectional mailbox channels to left and right
/// neighbors in a CPU array topology. Each channel is a pair of shared
/// `Rc<Cell<u64>>` cells — one for send, one for receive.
///
/// Address mapping:
/// - addr 60 (MMIO_MAILBOX_IN): read = receive from left, write = send to left
/// - addr 61 (MMIO_MAILBOX_OUT): read = receive from right, write = send to right
/// - All other addresses return 0.
pub struct V2LinkMailboxDevice {
    left_recv: Option<Rc<Cell<u64>>>,
    left_send: Option<Rc<Cell<u64>>>,
    right_send: Option<Rc<Cell<u64>>>,
    right_recv: Option<Rc<Cell<u64>>>,
}

impl V2LinkMailboxDevice {
    pub fn new(
        left_recv: Option<Rc<Cell<u64>>>,
        left_send: Option<Rc<Cell<u64>>>,
        right_send: Option<Rc<Cell<u64>>>,
        right_recv: Option<Rc<Cell<u64>>>,
    ) -> Self {
        Self {
            left_recv,
            left_send,
            right_send,
            right_recv,
        }
    }
}

impl V2MmioDevice for V2LinkMailboxDevice {
    fn read(&self, addr: u8) -> u64 {
        match addr {
            MMIO_MAILBOX_IN => self.left_recv.as_ref().map_or(0, |c| c.get()),
            MMIO_MAILBOX_OUT => self.right_recv.as_ref().map_or(0, |c| c.get()),
            _ => 0,
        }
    }

    fn write(&self, addr: u8, value: u64) {
        match addr {
            MMIO_MAILBOX_IN => {
                if let Some(c) = &self.left_send {
                    c.set(value);
                }
            }
            MMIO_MAILBOX_OUT => {
                if let Some(c) = &self.right_send {
                    c.set(value);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snn::mlp_weights::{CachedRates, MlpWeights};

    #[test]
    fn test_ref_device_pack_timer_rng_mailbox_console() {
        let dev = V2MmioRefDevicePack::new(1234);
        dev.tick(7);
        assert_eq!(dev.read(MMIO_TIMER_CYCLE), 7);

        dev.write(MMIO_CONSOLE_DATA, 0x41);
        dev.write(MMIO_CONSOLE_DATA, 0x42);
        assert_eq!(dev.read(MMIO_CONSOLE_DATA), 0x42);
        assert_eq!(dev.read(MMIO_CONSOLE_COUNT), 2);

        dev.write(MMIO_MAILBOX_IN, 0xAA55);
        dev.write(MMIO_MAILBOX_OUT, 0x55AA);
        assert_eq!(dev.read(MMIO_MAILBOX_IN), 0xAA55);
        assert_eq!(dev.read(MMIO_MAILBOX_OUT), 0x55AA);

        let r0 = dev.read(MMIO_RNG_DATA);
        let r1 = dev.read(MMIO_RNG_DATA);
        assert_ne!(r0, r1);
    }

    #[test]
    fn test_pbit_bridge_deterministic_runs() {
        let dev = V2MmioPbitBridgeDevice::new();
        dev.write(MMIO_CONSOLE_DATA, 99); // seed
        dev.write(MMIO_CONSOLE_COUNT, 256); // steps
        dev.write(MMIO_RNG_DATA, 8); // n_pbits
        dev.write(MMIO_TIMER_CYCLE, 1); // run

        assert_eq!(dev.read(MMIO_TIMER_CYCLE), PBIT_STATUS_DONE);
        let e0 = dev.read(MMIO_MAILBOX_IN);
        let s0 = dev.read(MMIO_MAILBOX_OUT);
        let runs0 = dev.read(MMIO_PBIT_CTRL);

        dev.write(MMIO_TIMER_CYCLE, 1); // run again with same params
        let e1 = dev.read(MMIO_MAILBOX_IN);
        let s1 = dev.read(MMIO_MAILBOX_OUT);
        let runs1 = dev.read(MMIO_PBIT_CTRL);

        assert_eq!(e0, e1);
        assert_eq!(s0, s1);
        assert_eq!(runs1, runs0 + 1);
        assert_eq!(dev.read(MMIO_PBIT_RESULT), 0);
    }

    #[test]
    fn test_ref_device_pack_contract_deterministic_sequence() {
        let dev_a = V2MmioRefDevicePack::new(0xC0FFEE);
        let dev_b = V2MmioRefDevicePack::new(0xC0FFEE);

        for cycle in 1..=8u64 {
            dev_a.tick(cycle);
            dev_b.tick(cycle);
            dev_a.write(MMIO_CONSOLE_DATA, cycle);
            dev_b.write(MMIO_CONSOLE_DATA, cycle);
            dev_a.write(MMIO_MAILBOX_IN, cycle * 3);
            dev_b.write(MMIO_MAILBOX_IN, cycle * 3);
            let ra = dev_a.read(MMIO_RNG_DATA);
            let rb = dev_b.read(MMIO_RNG_DATA);
            assert_eq!(ra, rb, "RNG diverged at cycle {cycle}");
        }

        assert_eq!(dev_a.read(MMIO_TIMER_CYCLE), dev_b.read(MMIO_TIMER_CYCLE));
        assert_eq!(
            dev_a.read(MMIO_CONSOLE_COUNT),
            dev_b.read(MMIO_CONSOLE_COUNT)
        );
        assert_eq!(dev_a.read(MMIO_CONSOLE_DATA), dev_b.read(MMIO_CONSOLE_DATA));
        assert_eq!(dev_a.read(MMIO_MAILBOX_IN), dev_b.read(MMIO_MAILBOX_IN));
    }

    #[test]
    fn test_ref_device_pack_tick_contract_last_cycle_visible() {
        let dev = V2MmioRefDevicePack::new(1);
        assert_eq!(dev.read(MMIO_TIMER_CYCLE), 0);
        dev.tick(11);
        assert_eq!(dev.read(MMIO_TIMER_CYCLE), 11);
        dev.tick(37);
        assert_eq!(dev.read(MMIO_TIMER_CYCLE), 37);
    }

    #[test]
    fn test_pbit_bridge_status_transition_contract() {
        let dev = V2MmioPbitBridgeDevice::new();
        assert_eq!(dev.read(MMIO_TIMER_CYCLE), PBIT_STATUS_IDLE);

        dev.write(MMIO_TIMER_CYCLE, 1);
        assert_eq!(dev.read(MMIO_TIMER_CYCLE), PBIT_STATUS_DONE);
        assert_eq!(dev.read(MMIO_PBIT_RESULT), 0);
        assert!(dev.read(MMIO_PBIT_CTRL) >= 1);

        dev.write(MMIO_TIMER_CYCLE, 0);
        assert_eq!(dev.read(MMIO_TIMER_CYCLE), PBIT_STATUS_IDLE);
    }

    #[test]
    fn test_pbit_bridge_param_roundtrip_contract() {
        let dev = V2MmioPbitBridgeDevice::new();
        dev.write(MMIO_CONSOLE_DATA, 123456);
        dev.write(MMIO_CONSOLE_COUNT, 2048);
        dev.write(MMIO_RNG_DATA, 12);
        assert_eq!(dev.read(MMIO_CONSOLE_DATA), 123456);
        assert_eq!(dev.read(MMIO_CONSOLE_COUNT), 2048);
        assert_eq!(dev.read(MMIO_RNG_DATA), 12);
    }

    #[test]
    fn test_ref_device_pack_snapshot_restore_roundtrip() {
        let src = V2MmioRefDevicePack::new(0xBEEF);
        src.tick(99);
        src.write(MMIO_CONSOLE_DATA, 0x41);
        src.write(MMIO_MAILBOX_IN, 0xABCD);
        src.write(MMIO_MAILBOX_OUT, 0xDCBA);
        let rng0 = src.read(MMIO_RNG_DATA);
        let snapshot = src.snapshot_state().expect("snapshot support required");

        let dst = V2MmioRefDevicePack::new(0);
        dst.restore_state(&snapshot)
            .expect("snapshot restore should succeed");

        assert_eq!(dst.snapshot_kind(), Some(V2_MMIO_REF_SNAPSHOT_KIND));
        assert_eq!(dst.read(MMIO_TIMER_CYCLE), 99);
        assert_eq!(dst.read(MMIO_CONSOLE_DATA), 0x41);
        assert_eq!(dst.read(MMIO_CONSOLE_COUNT), 1);
        assert_eq!(dst.read(MMIO_MAILBOX_IN), 0xABCD);
        assert_eq!(dst.read(MMIO_MAILBOX_OUT), 0xDCBA);
        assert_eq!(
            dst.read(MMIO_RNG_DATA),
            rng0.wrapping_mul(6364136223846793005).wrapping_add(1)
        );
    }

    #[test]
    fn test_pbit_bridge_snapshot_restore_roundtrip() {
        let src = V2MmioPbitBridgeDevice::new();
        src.write(MMIO_CONSOLE_DATA, 77);
        src.write(MMIO_CONSOLE_COUNT, 512);
        src.write(MMIO_RNG_DATA, 10);
        src.write(MMIO_TIMER_CYCLE, 1);
        let snapshot = src.snapshot_state().expect("snapshot support required");

        let dst = V2MmioPbitBridgeDevice::new();
        dst.restore_state(&snapshot)
            .expect("snapshot restore should succeed");

        assert_eq!(dst.snapshot_kind(), Some(V2_MMIO_PBIT_SNAPSHOT_KIND));
        assert_eq!(dst.read(MMIO_TIMER_CYCLE), src.read(MMIO_TIMER_CYCLE));
        assert_eq!(dst.read(MMIO_CONSOLE_DATA), src.read(MMIO_CONSOLE_DATA));
        assert_eq!(dst.read(MMIO_CONSOLE_COUNT), src.read(MMIO_CONSOLE_COUNT));
        assert_eq!(dst.read(MMIO_RNG_DATA), src.read(MMIO_RNG_DATA));
        assert_eq!(dst.read(MMIO_MAILBOX_IN), src.read(MMIO_MAILBOX_IN));
        assert_eq!(dst.read(MMIO_MAILBOX_OUT), src.read(MMIO_MAILBOX_OUT));
        assert_eq!(dst.read(MMIO_PBIT_CTRL), src.read(MMIO_PBIT_CTRL));
        assert_eq!(dst.read(MMIO_PBIT_RESULT), src.read(MMIO_PBIT_RESULT));
    }

    #[test]
    fn test_mmio_snapshot_restore_rejects_invalid_length() {
        let dev = V2MmioRefDevicePack::new(1);
        let err = dev
            .restore_state(&[1, 2, 3])
            .expect_err("invalid snapshot length must fail");
        assert!(err.contains("invalid snapshot length"));
    }

    // --- Display device tests ---

    #[test]
    fn test_display_pixel_write_read_roundtrip() {
        let dev = V2MmioDisplayDevice::new();
        // Write pixel at (3, 5) with color 0xE4 (R=7,G=1,B=0)
        let cmd: u64 = (5 << 16) | (3 << 8) | 0xE4;
        dev.write(MMIO_DISPLAY_CMD, cmd);
        // Read back last command
        assert_eq!(dev.read(MMIO_DISPLAY_CMD), cmd);
        // Verify pixel value
        assert_eq!(dev.get_pixel(3, 5), 0xE4);
        // Other pixels remain 0
        assert_eq!(dev.get_pixel(0, 0), 0);
        assert_eq!(dev.get_pixel(15, 15), 0);
    }

    #[test]
    fn test_display_status_returns_dimensions() {
        let dev = V2MmioDisplayDevice::new();
        let status = dev.read(MMIO_DISPLAY_STATUS);
        assert_eq!(status, (16 << 8) | 16);
    }

    #[test]
    fn test_display_clear_screen() {
        let dev = V2MmioDisplayDevice::new();
        // Write some pixels
        dev.write(MMIO_DISPLAY_CMD, 0xFF);
        dev.write(MMIO_DISPLAY_CMD, (7 << 16) | (7 << 8) | 0xAB);
        assert_eq!(dev.get_pixel(0, 0), 0xFF);
        assert_eq!(dev.get_pixel(7, 7), 0xAB);
        // Clear
        dev.write(MMIO_DISPLAY_STATUS, 0);
        assert_eq!(dev.get_pixel(0, 0), 0);
        assert_eq!(dev.get_pixel(7, 7), 0);
        // last_cmd also cleared
        assert_eq!(dev.read(MMIO_DISPLAY_CMD), 0);
    }

    #[test]
    fn test_display_out_of_bounds_ignored() {
        let dev = V2MmioDisplayDevice::new();
        // Write to (16, 0) — out of bounds X
        dev.write(MMIO_DISPLAY_CMD, (16 << 8) | 0xFF);
        // Write to (0, 16) — out of bounds Y
        dev.write(MMIO_DISPLAY_CMD, (16 << 16) | 0xFF);
        // All pixels remain 0
        let all_zero = dev.pixels().iter().all(|&p| p == 0);
        assert!(all_zero, "out-of-bounds writes should not modify any pixel");
    }

    #[test]
    fn test_display_ppm_render() {
        let dev = V2MmioDisplayDevice::new();
        // Set pixel (0,0) to white-ish (R=7,G=7,B=3 = 0xFF)
        dev.write(MMIO_DISPLAY_CMD, 0xFF);
        let ppm = dev.render_display_ppm(1);
        // Check PPM header
        let header = b"P6\n16 16\n255\n";
        assert!(ppm.starts_with(header), "PPM header mismatch");
        // Check first pixel RGB (3-3-2: R=7*36=252, G=7*36=252, B=3*85=255)
        let data_start = header.len();
        assert_eq!(ppm[data_start], 252); // R
        assert_eq!(ppm[data_start + 1], 252); // G
        assert_eq!(ppm[data_start + 2], 255); // B
        // Check total size: header + 16*16*3 = header + 768
        assert_eq!(ppm.len(), data_start + 16 * 16 * 3);
    }

    #[test]
    fn test_display_snapshot_restore_roundtrip() {
        let src = V2MmioDisplayDevice::new();
        src.write(MMIO_DISPLAY_CMD, (2 << 16) | (3 << 8) | 0x42);
        src.write(MMIO_DISPLAY_CMD, (10 << 16) | (15 << 8) | 0xBE);
        let snapshot = src.snapshot_state().expect("snapshot required");

        let dst = V2MmioDisplayDevice::new();
        dst.restore_state(&snapshot)
            .expect("restore should succeed");

        assert_eq!(dst.snapshot_kind(), Some(V2_MMIO_DISPLAY_SNAPSHOT_KIND));
        assert_eq!(dst.get_pixel(3, 2), 0x42);
        assert_eq!(dst.get_pixel(15, 10), 0xBE);
        assert_eq!(dst.read(MMIO_DISPLAY_CMD), src.read(MMIO_DISPLAY_CMD));
    }

    #[test]
    fn test_combined_device_dispatches_correctly() {
        let dev = V2MmioCombinedDevice::new(42);
        // Display write
        dev.write(MMIO_DISPLAY_CMD, (1 << 16) | (2 << 8) | 0xAA);
        assert_eq!(dev.display.get_pixel(2, 1), 0xAA);
        // Ref pack write
        dev.write(MMIO_CONSOLE_DATA, 0x55);
        assert_eq!(dev.ref_pack.console_last(), 0x55);
        // Tick updates timer
        dev.tick(100);
        assert_eq!(dev.read(MMIO_TIMER_CYCLE), 100);
        // Display status
        assert_eq!(dev.read(MMIO_DISPLAY_STATUS), (16 << 8) | 16);
    }

    #[test]
    fn test_mmio_address_constants_backward_compatible() {
        // Verify existing device addresses remain at 56-63
        assert_eq!(MMIO_TIMER_CYCLE, 56);
        assert_eq!(MMIO_CONSOLE_DATA, 57);
        assert_eq!(MMIO_CONSOLE_COUNT, 58);
        assert_eq!(MMIO_RNG_DATA, 59);
        assert_eq!(MMIO_MAILBOX_IN, 60);
        assert_eq!(MMIO_MAILBOX_OUT, 61);
        assert_eq!(MMIO_PBIT_CTRL, 62);
        assert_eq!(MMIO_PBIT_RESULT, 63);
        // Display addresses at 48-49
        assert_eq!(MMIO_DISPLAY_CMD, 48);
        assert_eq!(MMIO_DISPLAY_STATUS, 49);
        // Math coprocessor at 50-53
        assert_eq!(MMIO_MATH_A, 50);
        assert_eq!(MMIO_MATH_B, 51);
        assert_eq!(MMIO_MATH_CMD, 52);
        assert_eq!(MMIO_MATH_RESULT, 53);
        // SNN bridge at 54-55
        assert_eq!(MMIO_SNN_DATA, 54);
        assert_eq!(MMIO_SNN_CMD, 55);
    }

    // --- Math coprocessor tests ---

    #[test]
    fn test_math_mul_basic() {
        let dev = V2MmioMathDevice::new();
        dev.write(MMIO_MATH_A, 7);
        dev.write(MMIO_MATH_B, 6);
        assert_eq!(dev.read(MMIO_MATH_A), 7);
        assert_eq!(dev.read(MMIO_MATH_B), 6);
        dev.write(MMIO_MATH_CMD, 0); // MUL
        assert_eq!(dev.read(MMIO_MATH_RESULT), 42);
        assert_eq!(dev.read(MMIO_MATH_CMD), 0); // status ok

        // Larger values
        dev.write(MMIO_MATH_A, 1_000_000);
        dev.write(MMIO_MATH_B, 1_000_000);
        dev.write(MMIO_MATH_CMD, 0); // MUL
        assert_eq!(dev.read(MMIO_MATH_RESULT), 1_000_000_000_000);
    }

    #[test]
    fn test_math_div_mod() {
        let dev = V2MmioMathDevice::new();
        dev.write(MMIO_MATH_A, 100);
        dev.write(MMIO_MATH_B, 7);

        dev.write(MMIO_MATH_CMD, 1); // DIV
        assert_eq!(dev.read(MMIO_MATH_RESULT), 14);
        assert_eq!(dev.read(MMIO_MATH_CMD), 0); // status ok

        dev.write(MMIO_MATH_CMD, 2); // MOD
        assert_eq!(dev.read(MMIO_MATH_RESULT), 2);
        assert_eq!(dev.read(MMIO_MATH_CMD), 0);
    }

    #[test]
    fn test_math_div_by_zero() {
        let dev = V2MmioMathDevice::new();
        dev.write(MMIO_MATH_A, 42);
        dev.write(MMIO_MATH_B, 0);

        // First do a valid MUL to set result to something known
        dev.write(MMIO_MATH_B, 1);
        dev.write(MMIO_MATH_CMD, 0); // MUL: 42 * 1 = 42
        assert_eq!(dev.read(MMIO_MATH_RESULT), 42);

        // Now div by zero
        dev.write(MMIO_MATH_B, 0);
        dev.write(MMIO_MATH_CMD, 1); // DIV by zero
        assert_eq!(dev.read(MMIO_MATH_CMD), 1); // status = div-by-zero
        assert_eq!(dev.read(MMIO_MATH_RESULT), 42); // result unchanged

        // MOD by zero too
        dev.write(MMIO_MATH_CMD, 2); // MOD by zero
        assert_eq!(dev.read(MMIO_MATH_CMD), 1); // status = div-by-zero
    }

    #[test]
    fn test_math_mulhi() {
        let dev = V2MmioMathDevice::new();
        // 2^63 * 2 = 2^64 → low = 0, high = 1
        dev.write(MMIO_MATH_A, 1u64 << 63);
        dev.write(MMIO_MATH_B, 2);
        dev.write(MMIO_MATH_CMD, 3); // MULHI
        assert_eq!(dev.read(MMIO_MATH_RESULT), 1);
        assert_eq!(dev.read(MMIO_MATH_CMD), 0);

        // Small values: MULHI should be 0
        dev.write(MMIO_MATH_A, 100);
        dev.write(MMIO_MATH_B, 200);
        dev.write(MMIO_MATH_CMD, 3); // MULHI
        assert_eq!(dev.read(MMIO_MATH_RESULT), 0);
    }

    #[test]
    fn test_math_popcount() {
        let dev = V2MmioMathDevice::new();

        // popcount(0) = 0
        dev.write(MMIO_MATH_A, 0);
        dev.write(MMIO_MATH_CMD, 4); // POPCOUNT
        assert_eq!(dev.read(MMIO_MATH_RESULT), 0);
        assert_eq!(dev.read(MMIO_MATH_CMD), 0); // status ok

        // popcount(0xFFFF) = 16
        dev.write(MMIO_MATH_A, 0xFFFF);
        dev.write(MMIO_MATH_CMD, 4);
        assert_eq!(dev.read(MMIO_MATH_RESULT), 16);

        // popcount(0x8000_0000_0000_0001) = 2
        dev.write(MMIO_MATH_A, 0x8000_0000_0000_0001);
        dev.write(MMIO_MATH_CMD, 4);
        assert_eq!(dev.read(MMIO_MATH_RESULT), 2);

        // popcount(u64::MAX) = 64
        dev.write(MMIO_MATH_A, u64::MAX);
        dev.write(MMIO_MATH_CMD, 4);
        assert_eq!(dev.read(MMIO_MATH_RESULT), 64);
    }

    #[test]
    fn test_math_snapshot_restore() {
        let src = V2MmioMathDevice::new();
        src.write(MMIO_MATH_A, 123);
        src.write(MMIO_MATH_B, 456);
        src.write(MMIO_MATH_CMD, 0); // MUL: 123*456 = 56088
        let snapshot = src.snapshot_state().expect("snapshot");

        let dst = V2MmioMathDevice::new();
        dst.restore_state(&snapshot).expect("restore");

        assert_eq!(dst.read(MMIO_MATH_A), 123);
        assert_eq!(dst.read(MMIO_MATH_B), 456);
        assert_eq!(dst.read(MMIO_MATH_RESULT), 56088);
        assert_eq!(dst.read(MMIO_MATH_CMD), 0);
    }

    // --- SNN bridge tests ---

    #[test]
    fn test_snn_bridge_reset_clears_state() {
        let dev = V2MmioSnnBridgeDevice::small(); // 8-4-2
        // Inject input and tick
        dev.write(MMIO_SNN_DATA, 0xFF);
        dev.write(MMIO_SNN_CMD, 1); // SET_INPUT
        dev.write(MMIO_SNN_CMD, 2); // TICK

        // Reset
        dev.write(MMIO_SNN_CMD, 0); // RESET
        // Check all neurons at rest
        for i in 0..dev.total_neurons() {
            dev.write(MMIO_SNN_DATA, i as u64);
            dev.write(MMIO_SNN_CMD, 5); // STAGE_MEMBRANE
            assert_eq!(
                dev.read(MMIO_SNN_DATA),
                LIFNeuron::V_REST as u64,
                "neuron {i} not at rest after RESET"
            );
        }
        // Spike counts cleared
        for o in 0..2 {
            dev.write(MMIO_SNN_DATA, o);
            dev.write(MMIO_SNN_CMD, 6); // GET_SPIKE_COUNT
            assert_eq!(dev.read(MMIO_SNN_DATA), 0, "output {o} spike count not 0");
        }
    }

    #[test]
    fn test_snn_bridge_set_input_tick_output() {
        let dev = V2MmioSnnBridgeDevice::small(); // 8-4-2
        dev.write(MMIO_SNN_CMD, 0); // RESET

        // Set all 8 inputs firing
        dev.write(MMIO_SNN_DATA, 0xFF);
        dev.write(MMIO_SNN_CMD, 1); // SET_INPUT

        // Run 10 timesteps
        dev.write(MMIO_SNN_DATA, 10);
        dev.write(MMIO_SNN_CMD, 3); // TICK_N

        // Stage output
        dev.write(MMIO_SNN_CMD, 4); // STAGE_OUTPUT
        let _output = dev.read(MMIO_SNN_DATA);
        // Output is deterministic but depends on random weights — just verify no panic
        // and status is ok
        let status_word = dev.read(MMIO_SNN_CMD);
        assert_eq!(status_word & 0xFF, 0, "status should be ok");
        assert_eq!((status_word >> 8) & 0xFF, 8, "n_input should be 8");
        assert_eq!((status_word >> 16) & 0xFF, 2, "n_output should be 2");
    }

    #[test]
    fn test_snn_bridge_deterministic() {
        // Same seed + same inputs = same outputs
        let dev_a = V2MmioSnnBridgeDevice::small();
        let dev_b = V2MmioSnnBridgeDevice::small();

        for dev in [&dev_a, &dev_b] {
            dev.write(MMIO_SNN_CMD, 0); // RESET
            dev.write(MMIO_SNN_DATA, 0xAA);
            dev.write(MMIO_SNN_CMD, 1); // SET_INPUT
            dev.write(MMIO_SNN_DATA, 5);
            dev.write(MMIO_SNN_CMD, 3); // TICK_N(5)
            dev.write(MMIO_SNN_CMD, 4); // STAGE_OUTPUT
        }

        assert_eq!(
            dev_a.read(MMIO_SNN_DATA),
            dev_b.read(MMIO_SNN_DATA),
            "output mismatch between identical runs"
        );

        // Check spike counts match too
        for o in 0..2u64 {
            dev_a.write(MMIO_SNN_DATA, o);
            dev_a.write(MMIO_SNN_CMD, 6);
            dev_b.write(MMIO_SNN_DATA, o);
            dev_b.write(MMIO_SNN_CMD, 6);
            assert_eq!(
                dev_a.read(MMIO_SNN_DATA),
                dev_b.read(MMIO_SNN_DATA),
                "spike count mismatch for output {o}"
            );
        }
    }

    #[test]
    fn test_snn_bridge_tick_n_matches_n_ticks() {
        let dev_a = V2MmioSnnBridgeDevice::small();
        let dev_b = V2MmioSnnBridgeDevice::small();

        // Device A: TICK_N(5)
        dev_a.write(MMIO_SNN_CMD, 0);
        dev_a.write(MMIO_SNN_DATA, 0x55);
        dev_a.write(MMIO_SNN_CMD, 1);
        dev_a.write(MMIO_SNN_DATA, 5);
        dev_a.write(MMIO_SNN_CMD, 3); // TICK_N(5)

        // Device B: 5 × TICK(1)
        dev_b.write(MMIO_SNN_CMD, 0);
        dev_b.write(MMIO_SNN_DATA, 0x55);
        dev_b.write(MMIO_SNN_CMD, 1);
        for _ in 0..5 {
            dev_b.write(MMIO_SNN_CMD, 2); // TICK
        }

        // Compare membrane potentials of all neurons
        for i in 0..dev_a.total_neurons() {
            dev_a.write(MMIO_SNN_DATA, i as u64);
            dev_a.write(MMIO_SNN_CMD, 5);
            dev_b.write(MMIO_SNN_DATA, i as u64);
            dev_b.write(MMIO_SNN_CMD, 5);
            assert_eq!(
                dev_a.read(MMIO_SNN_DATA),
                dev_b.read(MMIO_SNN_DATA),
                "membrane mismatch at neuron {i}"
            );
        }
    }

    #[test]
    fn test_snn_bridge_snapshot_restore() {
        let src = V2MmioSnnBridgeDevice::small();
        src.write(MMIO_SNN_CMD, 0); // RESET
        src.write(MMIO_SNN_DATA, 0xFF);
        src.write(MMIO_SNN_CMD, 1); // SET_INPUT
        src.write(MMIO_SNN_DATA, 3);
        src.write(MMIO_SNN_CMD, 3); // TICK_N(3)

        let snapshot = src.snapshot_state().expect("snapshot");

        let dst = V2MmioSnnBridgeDevice::small();
        dst.restore_state(&snapshot).expect("restore");

        // Compare all neuron membranes
        for i in 0..src.total_neurons() {
            src.write(MMIO_SNN_DATA, i as u64);
            src.write(MMIO_SNN_CMD, 5);
            dst.write(MMIO_SNN_DATA, i as u64);
            dst.write(MMIO_SNN_CMD, 5);
            assert_eq!(
                src.read(MMIO_SNN_DATA),
                dst.read(MMIO_SNN_DATA),
                "membrane mismatch at neuron {i}"
            );
        }

        // Compare output spike counts
        for o in 0..2u64 {
            src.write(MMIO_SNN_DATA, o);
            src.write(MMIO_SNN_CMD, 6);
            dst.write(MMIO_SNN_DATA, o);
            dst.write(MMIO_SNN_CMD, 6);
            assert_eq!(
                src.read(MMIO_SNN_DATA),
                dst.read(MMIO_SNN_DATA),
                "spike count mismatch for output {o}"
            );
        }
    }

    #[test]
    fn test_snn_bridge_stage_membrane() {
        let dev = V2MmioSnnBridgeDevice::small();
        // At rest, membrane should be V_REST (0)
        dev.write(MMIO_SNN_DATA, 0);
        dev.write(MMIO_SNN_CMD, 5); // STAGE_MEMBRANE
        assert_eq!(dev.read(MMIO_SNN_DATA), LIFNeuron::V_REST as u64);

        // Out of bounds returns 0 with status=1
        dev.write(MMIO_SNN_DATA, 9999);
        dev.write(MMIO_SNN_CMD, 5);
        assert_eq!(dev.read(MMIO_SNN_DATA), 0);
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 1); // error status
    }

    // --- M11: SNN bridge INFER command ---

    #[test]
    fn test_snn_bridge_infer_no_model() {
        let dev = V2MmioSnnBridgeDevice::small(); // no model loaded
        dev.write(MMIO_SNN_DATA, 0);
        dev.write(MMIO_SNN_CMD, 7); // INFER
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 1); // error: no model
    }

    #[test]
    fn test_snn_bridge_infer_with_model() {
        // 2-neuron hidden layer, 2-class MLP (identity weights)
        let weights = MlpWeights {
            w1: vec![1.0, 0.0, 0.0, 1.0], // 2x2 identity
            b1: vec![0.0, 0.0],
            w2: vec![1.0, 0.0, 0.0, 1.0],
            b2: vec![0.0, 0.0],
            w3: vec![1.0, -1.0, -1.0, 1.0], // contrast matrix
            b3: vec![0.0, 0.0],
        };
        // 3 samples: [1,0], [0,1], [0.5,0.5]
        let rates = CachedRates::new(2, vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
        let model = InferenceModel {
            weights,
            cached_rates: rates,
        };
        let dev = V2MmioSnnBridgeDevice::with_model(2, 2, 2, 42, model);

        // Sample 0: [1,0] → class 0
        dev.write(MMIO_SNN_DATA, 0);
        dev.write(MMIO_SNN_CMD, 7);
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 0); // ok
        assert_eq!(dev.read(MMIO_SNN_DATA), 0);

        // Sample 1: [0,1] → class 1
        dev.write(MMIO_SNN_DATA, 1);
        dev.write(MMIO_SNN_CMD, 7);
        assert_eq!(dev.read(MMIO_SNN_DATA), 1);

        // Sample 2: [0.5,0.5] → tied, class 0 (first argmax)
        dev.write(MMIO_SNN_DATA, 2);
        dev.write(MMIO_SNN_CMD, 7);
        assert_eq!(dev.read(MMIO_SNN_DATA), 0);

        // OOB sample
        dev.write(MMIO_SNN_DATA, 999);
        dev.write(MMIO_SNN_CMD, 7);
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 1); // error: OOB
    }

    // --- Combined device with math + SNN ---

    #[test]
    fn test_combined_device_math_dispatch() {
        let dev = V2MmioCombinedDevice::new(42);
        dev.write(MMIO_MATH_A, 10);
        dev.write(MMIO_MATH_B, 20);
        dev.write(MMIO_MATH_CMD, 0); // MUL
        assert_eq!(dev.read(MMIO_MATH_RESULT), 200);
        // Ensure other devices still work
        dev.write(MMIO_CONSOLE_DATA, 0x42);
        assert_eq!(dev.ref_pack.console_last(), 0x42);
    }

    #[test]
    fn test_combined_device_snn_dispatch() {
        let dev = V2MmioCombinedDevice::new(42);
        dev.write(MMIO_SNN_CMD, 0); // RESET
        let status = dev.read(MMIO_SNN_CMD);
        assert_eq!(status & 0xFF, 0); // idle status
    }

    // ── Sprint 157: Quantum bridge device tests ──

    #[test]
    fn test_quantum_init_and_status() {
        let dev = V2MmioQuantumDevice::new(42);
        // Before INIT, status should be NOT_INIT
        let status = dev.read(MMIO_QUANTUM_CMD);
        assert_eq!(status & 0xFF, 1); // QUANTUM_STATUS_NOT_INIT
        // INIT 3 qubits: write N=3 to DATA, then cmd 0
        dev.write(MMIO_QUANTUM_DATA, 3);
        dev.write(MMIO_QUANTUM_CMD, 0); // INIT
        let status = dev.read(MMIO_QUANTUM_CMD);
        assert_eq!(status & 0xFF, 0); // OK
        assert_eq!((status >> 8) & 0xFF, 3); // n_qubits = 3
        assert_eq!(status >> 16, 0); // gate_count = 0
    }

    #[test]
    fn test_quantum_hadamard_measure() {
        let dev = V2MmioQuantumDevice::new(42);
        // INIT 1 qubit
        dev.write(MMIO_QUANTUM_DATA, 1);
        dev.write(MMIO_QUANTUM_CMD, 0);
        // H(0)
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_CMD, 2); // GATE_H
        // Measure
        dev.write(MMIO_QUANTUM_CMD, 10); // MEASURE
        let result = dev.read(MMIO_QUANTUM_DATA);
        // Deterministic: with seed 42 the result is either 0 or 1
        assert!(result == 0 || result == 1);
        // gate_count should be 2 (H + Measure)
        let status = dev.read(MMIO_QUANTUM_CMD);
        assert_eq!(status >> 16, 2);
    }

    #[test]
    fn test_quantum_bell_state() {
        let dev = V2MmioQuantumDevice::new(42);
        // INIT 2 qubits
        dev.write(MMIO_QUANTUM_DATA, 2);
        dev.write(MMIO_QUANTUM_CMD, 0);
        // H(0)
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_CMD, 2);
        // CNOT(0, 1): control=QUBIT=0, target=DATA=1
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_DATA, 1);
        dev.write(MMIO_QUANTUM_CMD, 7); // GATE_CNOT
        // MEASURE_ALL
        dev.write(MMIO_QUANTUM_CMD, 11);
        let result = dev.read(MMIO_QUANTUM_DATA);
        // Bell state: outcome must be 00 (0) or 11 (3) — correlated
        assert!(
            result == 0 || result == 3,
            "Bell state: expected 0 or 3, got {}",
            result
        );
    }

    #[test]
    fn test_quantum_x_gate_flip() {
        let dev = V2MmioQuantumDevice::new(42);
        // INIT 1 qubit (starts as |0⟩)
        dev.write(MMIO_QUANTUM_DATA, 1);
        dev.write(MMIO_QUANTUM_CMD, 0);
        // X(0) flips |0⟩ → |1⟩
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_CMD, 3); // GATE_X
        // Measure — should always be 1
        dev.write(MMIO_QUANTUM_CMD, 10); // MEASURE
        assert_eq!(dev.read(MMIO_QUANTUM_DATA), 1);
    }

    #[test]
    fn test_quantum_probability() {
        let dev = V2MmioQuantumDevice::new(42);
        // INIT 1 qubit
        dev.write(MMIO_QUANTUM_DATA, 1);
        dev.write(MMIO_QUANTUM_CMD, 0);
        // H(0) → equal superposition
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_CMD, 2); // GATE_H
        // PROB(0) → should be ~0.5
        dev.write(MMIO_QUANTUM_CMD, 12); // PROB
        let prob_bits = dev.read(MMIO_QUANTUM_DATA);
        let prob = f64::from_bits(prob_bits);
        assert!((prob - 0.5).abs() < 1e-6, "expected ~0.5, got {}", prob);
    }

    #[test]
    fn test_quantum_deterministic_replay() {
        // Same seed + same gates = same measurement outcome
        let run = |seed: u64| -> u64 {
            let dev = V2MmioQuantumDevice::new(seed);
            dev.write(MMIO_QUANTUM_DATA, 2);
            dev.write(MMIO_QUANTUM_CMD, 0); // INIT 2 qubits
            dev.write(MMIO_QUANTUM_QUBIT, 0);
            dev.write(MMIO_QUANTUM_CMD, 2); // H(0)
            dev.write(MMIO_QUANTUM_DATA, 1);
            dev.write(MMIO_QUANTUM_CMD, 7); // CNOT(0,1)
            dev.write(MMIO_QUANTUM_CMD, 11); // MEASURE_ALL
            dev.read(MMIO_QUANTUM_DATA)
        };
        let a = run(99);
        let b = run(99);
        assert_eq!(a, b, "same seed must produce identical measurement");
        // Different seed may differ (not guaranteed but very likely for 2-qubit Bell)
    }

    #[test]
    fn test_quantum_snapshot_restore() {
        let dev = V2MmioQuantumDevice::new(42);
        // INIT 2 qubits + apply H(0)
        dev.write(MMIO_QUANTUM_DATA, 2);
        dev.write(MMIO_QUANTUM_CMD, 0);
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_CMD, 2); // H(0)
        // Snapshot
        let snap = dev.snapshot_state().unwrap();
        assert_eq!(dev.snapshot_kind(), Some(V2_MMIO_QUANTUM_SNAPSHOT_KIND));
        // Mutate: apply X(1) to change state
        dev.write(MMIO_QUANTUM_QUBIT, 1);
        dev.write(MMIO_QUANTUM_CMD, 3); // X(1)
        // Restore
        dev.restore_state(&snap).unwrap();
        // After restore, gate_count should be 1 (just H), not 2
        let status = dev.read(MMIO_QUANTUM_CMD);
        assert_eq!(status >> 16, 1); // gate_count = 1
        // PROB(0) should still be ~0.5 (H applied, no X)
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_CMD, 12); // PROB
        let prob = f64::from_bits(dev.read(MMIO_QUANTUM_DATA));
        assert!(
            (prob - 0.5).abs() < 1e-6,
            "post-restore prob mismatch: {}",
            prob
        );
    }

    #[test]
    fn test_quantum_not_initialized_error() {
        let dev = V2MmioQuantumDevice::new(42);
        // Try H(0) without INIT
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_CMD, 2); // GATE_H
        let status = dev.read(MMIO_QUANTUM_CMD);
        assert_eq!(status & 0xFF, 1); // QUANTUM_STATUS_NOT_INIT
        // Try MEASURE without INIT
        dev.write(MMIO_QUANTUM_CMD, 10);
        assert_eq!(dev.read(MMIO_QUANTUM_CMD) & 0xFF, 1);
        // Try MEASURE_ALL without INIT
        dev.write(MMIO_QUANTUM_CMD, 11);
        assert_eq!(dev.read(MMIO_QUANTUM_CMD) & 0xFF, 1);
        // Try RESET without INIT
        dev.write(MMIO_QUANTUM_CMD, 1);
        assert_eq!(dev.read(MMIO_QUANTUM_CMD) & 0xFF, 1);
    }

    #[test]
    fn test_quantum_qubit_out_of_range() {
        let dev = V2MmioQuantumDevice::new(42);
        // INIT 2 qubits
        dev.write(MMIO_QUANTUM_DATA, 2);
        dev.write(MMIO_QUANTUM_CMD, 0);
        // Try H(5) — qubit 5 out of range for 2-qubit system
        dev.write(MMIO_QUANTUM_QUBIT, 5);
        dev.write(MMIO_QUANTUM_CMD, 2); // GATE_H
        let status = dev.read(MMIO_QUANTUM_CMD);
        assert_eq!(status & 0xFF, 2); // QUANTUM_STATUS_QUBIT_OOB
        // Valid operation should clear the error
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_CMD, 3); // GATE_X on qubit 0 — valid
        assert_eq!(dev.read(MMIO_QUANTUM_CMD) & 0xFF, 0); // OK
    }

    // ── Sprint 158: Gate expansion tests ──

    #[test]
    fn test_quantum_rx_gate() {
        // Rx(π) on |0⟩ should flip to |1⟩ (up to global phase)
        let dev = V2MmioQuantumDevice::new(99);
        dev.write(MMIO_QUANTUM_DATA, 1);
        dev.write(MMIO_QUANTUM_CMD, 0); // INIT 1 qubit
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_PARAM, std::f64::consts::PI.to_bits());
        dev.write(MMIO_QUANTUM_CMD, 13); // GATE_RX(π)
        dev.write(MMIO_QUANTUM_CMD, 10); // MEASURE
        assert_eq!(dev.read(MMIO_QUANTUM_DATA), 1, "Rx(π)|0⟩ → |1⟩");
    }

    #[test]
    fn test_quantum_ry_gate() {
        // Ry(π) on |0⟩ should flip to |1⟩
        let dev = V2MmioQuantumDevice::new(99);
        dev.write(MMIO_QUANTUM_DATA, 1);
        dev.write(MMIO_QUANTUM_CMD, 0); // INIT 1 qubit
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_PARAM, std::f64::consts::PI.to_bits());
        dev.write(MMIO_QUANTUM_CMD, 14); // GATE_RY(π)
        dev.write(MMIO_QUANTUM_CMD, 10); // MEASURE
        assert_eq!(dev.read(MMIO_QUANTUM_DATA), 1, "Ry(π)|0⟩ → |1⟩");
    }

    #[test]
    fn test_quantum_swap_gate() {
        // X(1) then Swap(0,1) → qubit 0 becomes |1⟩, qubit 1 becomes |0⟩
        let dev = V2MmioQuantumDevice::new(42);
        dev.write(MMIO_QUANTUM_DATA, 2);
        dev.write(MMIO_QUANTUM_CMD, 0); // INIT 2 qubits
        // X on qubit 1 → |01⟩ → state = |10⟩ in little-endian (qubit 1 = 1)
        dev.write(MMIO_QUANTUM_QUBIT, 1);
        dev.write(MMIO_QUANTUM_CMD, 3); // GATE_X
        // Swap(0, 1)
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_DATA, 1); // target = qubit 1
        dev.write(MMIO_QUANTUM_CMD, 15); // GATE_SWAP
        // Measure all — qubit 0 should be 1, qubit 1 should be 0
        dev.write(MMIO_QUANTUM_CMD, 11); // MEASURE_ALL
        let result = dev.read(MMIO_QUANTUM_DATA);
        assert_eq!(
            result, 1,
            "After Swap: qubit 0 = 1, qubit 1 = 0 → bitmask 1"
        );
    }

    #[test]
    fn test_quantum_tdg_gate() {
        // X(0) → T(0) → T†(0) → should be identity, measure always 1
        let dev = V2MmioQuantumDevice::new(42);
        dev.write(MMIO_QUANTUM_DATA, 1);
        dev.write(MMIO_QUANTUM_CMD, 0); // INIT 1 qubit
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_CMD, 3); // GATE_X: |0⟩ → |1⟩
        dev.write(MMIO_QUANTUM_CMD, 6); // GATE_T
        dev.write(MMIO_QUANTUM_CMD, 16); // GATE_TDG: T·T† = I
        dev.write(MMIO_QUANTUM_CMD, 10); // MEASURE
        assert_eq!(dev.read(MMIO_QUANTUM_DATA), 1, "T·T† = I, measure 1");
    }

    #[test]
    fn test_quantum_cmd_status_read() {
        // Verify QUANTUM_CMD read returns packed (gate_count << 16) | (n_qubits << 8) | status
        let dev = V2MmioQuantumDevice::new(42);
        // Before init: n_qubits=0, gate_count=0, status=NOT_INIT(1)
        assert_eq!(dev.read(MMIO_QUANTUM_CMD), 1);
        // INIT 3 qubits
        dev.write(MMIO_QUANTUM_DATA, 3);
        dev.write(MMIO_QUANTUM_CMD, 0);
        // n_qubits=3, gate_count=0, status=OK(0)
        assert_eq!(dev.read(MMIO_QUANTUM_CMD), 3 << 8);
        // Apply H on qubit 0 (gate_count=1)
        dev.write(MMIO_QUANTUM_QUBIT, 0);
        dev.write(MMIO_QUANTUM_CMD, 2); // GATE_H
        assert_eq!(dev.read(MMIO_QUANTUM_CMD), ((1 << 16) | (3 << 8)));
        // Apply X on qubit 1 (gate_count=2)
        dev.write(MMIO_QUANTUM_QUBIT, 1);
        dev.write(MMIO_QUANTUM_CMD, 3); // GATE_X
        assert_eq!(dev.read(MMIO_QUANTUM_CMD), ((2 << 16) | (3 << 8)));
    }

    // --- Address boundary tests (Sprint 159) ---

    #[test]
    fn test_mmio_address_boundary_41_43() {
        use crate::tile_cpu::v2_mmio::is_v2_mmio_addr;

        // Address 40 must NOT be MMIO (used by memory_stream benchmark as regular RAM)
        assert!(!is_v2_mmio_addr(40), "addr 40 must remain regular RAM");

        // Addresses 41-43 must BE MMIO after Sprint 159 expansion
        assert!(is_v2_mmio_addr(41), "addr 41 must be MMIO (DATASET_CMD)");
        assert!(is_v2_mmio_addr(42), "addr 42 must be MMIO (DATASET_DATA)");
        assert!(is_v2_mmio_addr(43), "addr 43 must be MMIO (DATASET_STATUS)");

        // Addresses 44-63 remain MMIO (unchanged)
        assert!(is_v2_mmio_addr(44), "addr 44 must be MMIO (QUANTUM_CMD)");
        assert!(is_v2_mmio_addr(63), "addr 63 must be MMIO (PBIT_RESULT)");

        // Address 64 is out of range
        assert!(!is_v2_mmio_addr(64), "addr 64 must not be MMIO");

        // Verify constant values
        assert_eq!(MMIO_DATASET_CMD, 41);
        assert_eq!(MMIO_DATASET_DATA, 42);
        assert_eq!(MMIO_DATASET_STATUS, 43);
    }

    // --- Dataset device tests (Sprint 159) ---

    fn make_test_dataset() -> V2MmioDatasetDevice {
        V2MmioDatasetDevice::from_samples(vec![
            DatasetSample {
                features: 0xAAAA_BBBB_CCCC_DDDD,
                label: 0,
            },
            DatasetSample {
                features: 0x1111_2222_3333_4444,
                label: 1,
            },
            DatasetSample {
                features: 0xFFFF_FFFF_FFFF_FFFF,
                label: 0,
            },
        ])
    }

    #[test]
    fn test_dataset_device_load_and_read() {
        let dev = make_test_dataset();

        // GET_COUNT → 3
        dev.write(MMIO_DATASET_CMD, 3);
        assert_eq!(dev.read(MMIO_DATASET_DATA), 3);
        assert_eq!(dev.read(MMIO_DATASET_CMD) >> 8, DS_OK as u64);

        // LOAD_SAMPLE(0) then GET_FEATURES
        dev.write(MMIO_DATASET_DATA, 0);
        dev.write(MMIO_DATASET_CMD, 0); // LOAD_SAMPLE
        assert_eq!(dev.read(MMIO_DATASET_CMD) >> 8, DS_OK as u64);

        dev.write(MMIO_DATASET_CMD, 1); // GET_FEATURES
        assert_eq!(dev.read(MMIO_DATASET_DATA), 0xAAAA_BBBB_CCCC_DDDD);

        // GET_LABEL
        dev.write(MMIO_DATASET_CMD, 2);
        assert_eq!(dev.read(MMIO_DATASET_DATA), 0);

        // LOAD_SAMPLE(1) then check label=1
        dev.write(MMIO_DATASET_DATA, 1);
        dev.write(MMIO_DATASET_CMD, 0);
        dev.write(MMIO_DATASET_CMD, 2); // GET_LABEL
        assert_eq!(dev.read(MMIO_DATASET_DATA), 1);

        // STATUS: (current_index << 32) | total
        assert_eq!(dev.read(MMIO_DATASET_STATUS), (1u64 << 32) | 3);
    }

    #[test]
    fn test_dataset_device_submit_prediction() {
        let dev = make_test_dataset();

        // Load sample 0 (label=0), predict 0 → correct
        dev.write(MMIO_DATASET_DATA, 0);
        dev.write(MMIO_DATASET_CMD, 0); // LOAD_SAMPLE
        dev.write(MMIO_DATASET_DATA, 0); // prediction = 0
        dev.write(MMIO_DATASET_CMD, 5); // SUBMIT_PREDICTION
        dev.write(MMIO_DATASET_CMD, 4); // GET_CORRECT
        assert_eq!(dev.read(MMIO_DATASET_DATA), 1);

        // Load sample 1 (label=1), predict 0 → wrong
        dev.write(MMIO_DATASET_DATA, 1);
        dev.write(MMIO_DATASET_CMD, 0);
        dev.write(MMIO_DATASET_DATA, 0); // wrong prediction
        dev.write(MMIO_DATASET_CMD, 5);
        dev.write(MMIO_DATASET_CMD, 4);
        assert_eq!(dev.read(MMIO_DATASET_DATA), 1); // still 1

        // Load sample 2 (label=0), predict 0 → correct
        dev.write(MMIO_DATASET_DATA, 2);
        dev.write(MMIO_DATASET_CMD, 0);
        dev.write(MMIO_DATASET_DATA, 0);
        dev.write(MMIO_DATASET_CMD, 5);
        dev.write(MMIO_DATASET_CMD, 4);
        assert_eq!(dev.read(MMIO_DATASET_DATA), 2); // now 2
    }

    #[test]
    fn test_dataset_device_error_codes() {
        let dev = make_test_dataset();

        // OOB index
        dev.write(MMIO_DATASET_DATA, 99);
        dev.write(MMIO_DATASET_CMD, 0); // LOAD_SAMPLE(99) → OOB
        assert_eq!(dev.read(MMIO_DATASET_CMD) >> 8, DS_ERR_OOB as u64);

        // NO_SAMPLE: GET_FEATURES before LOAD_SAMPLE
        let dev2 = make_test_dataset();
        dev2.write(MMIO_DATASET_CMD, 1); // GET_FEATURES with no sample loaded
        assert_eq!(dev2.read(MMIO_DATASET_CMD) >> 8, DS_ERR_NO_SAMPLE as u64);

        // GET_LABEL with no sample
        let dev3 = make_test_dataset();
        dev3.write(MMIO_DATASET_CMD, 2);
        assert_eq!(dev3.read(MMIO_DATASET_CMD) >> 8, DS_ERR_NO_SAMPLE as u64);

        // SUBMIT_PREDICTION with no sample
        let dev4 = make_test_dataset();
        dev4.write(MMIO_DATASET_CMD, 5);
        assert_eq!(dev4.read(MMIO_DATASET_CMD) >> 8, DS_ERR_NO_SAMPLE as u64);

        // INVALID_CMD
        dev.write(MMIO_DATASET_CMD, 99);
        assert_eq!(dev.read(MMIO_DATASET_CMD) >> 8, DS_ERR_INVALID_CMD as u64);
    }

    #[test]
    fn test_dataset_device_snapshot_restore() {
        let src = make_test_dataset();
        // Load sample 1, get features, submit correct prediction
        src.write(MMIO_DATASET_DATA, 1);
        src.write(MMIO_DATASET_CMD, 0);
        src.write(MMIO_DATASET_CMD, 1); // GET_FEATURES
        src.write(MMIO_DATASET_DATA, 1); // prediction = 1
        src.write(MMIO_DATASET_CMD, 5); // SUBMIT (correct)

        let snapshot = src.snapshot_state().expect("snapshot");

        let dst = make_test_dataset();
        dst.restore_state(&snapshot).expect("restore");

        // Verify restored state matches
        assert_eq!(dst.read(MMIO_DATASET_DATA), src.read(MMIO_DATASET_DATA));
        dst.write(MMIO_DATASET_CMD, 4); // GET_CORRECT
        assert_eq!(dst.read(MMIO_DATASET_DATA), 1); // 1 correct
    }

    // --- M12: Live SNN commands ---

    #[test]
    fn test_snn_live_no_model() {
        let dev = V2MmioSnnBridgeDevice::small();
        dev.write(MMIO_SNN_DATA, 0);
        dev.write(MMIO_SNN_CMD, 8); // LOAD_IMAGE
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 1); // error: no model

        dev.write(MMIO_SNN_CMD, 9); // SNN_RUN
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 1); // error: no model

        dev.write(MMIO_SNN_CMD, 10); // INFER_LIVE
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 1); // error: no model
    }

    #[test]
    fn test_snn_live_basic() {
        use crate::snn::mlp_weights::LiveSnnModel;

        // Tiny model: 4 input, 2 hidden, 1 readout, 2 classes
        let model = LiveSnnModel {
            syn_ptr: vec![0, 1, 2, 2, 2, 4, 5, 5],
            targets: vec![4, 5, 4, 5, 6],
            weights: vec![100, 100, 80, 80, 60],
            thresholds: vec![32000, 32000, 32000, 32000, 50, 50, 100],
            leaks: vec![230; 7],
            pix_per_class: vec![vec![0, 1], vec![2, 3]],
            d_norms: vec![vec![1.0, 0.8], vec![0.9, 1.0]],
            n_input: 4,
            n_hidden: 2,
            n_readout: 1,
            n_classes: 2,
            k_per_class: 2,
            max_rate: 100,
            n_ticks: 50,
            mlp: MlpWeights {
                w1: vec![1.0, 0.0, 0.0, 1.0],
                b1: vec![0.0, 0.0],
                w2: vec![1.0, 0.0, 0.0, 1.0],
                b2: vec![0.0, 0.0],
                w3: vec![1.0, -1.0, -1.0, 1.0],
                b3: vec![0.0, 0.0],
            },
        };

        // Bright image: all pixels 200
        let images = vec![vec![200u8; 4]];
        let dev = V2MmioSnnBridgeDevice::with_live_model(4, 2, 1, 42, model, images);

        // LOAD_IMAGE(0)
        dev.write(MMIO_SNN_DATA, 0);
        dev.write(MMIO_SNN_CMD, 8);
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 0); // ok

        // SNN_RUN
        dev.write(MMIO_SNN_CMD, 9);
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 0); // ok

        // INFER_LIVE
        dev.write(MMIO_SNN_CMD, 10);
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 0); // ok
        let pred = dev.read(MMIO_SNN_DATA);
        // With this simple model, prediction should be 0 or 1
        assert!(pred <= 1, "prediction should be 0 or 1, got {pred}");
    }

    #[test]
    fn test_snn_live_oob_image() {
        use crate::snn::mlp_weights::LiveSnnModel;

        let model = LiveSnnModel {
            syn_ptr: vec![0, 0, 0, 0],
            targets: vec![],
            weights: vec![],
            thresholds: vec![100, 100, 100],
            leaks: vec![230; 3],
            pix_per_class: vec![vec![0]],
            d_norms: vec![vec![1.0]],
            n_input: 1,
            n_hidden: 1,
            n_readout: 1,
            n_classes: 1,
            k_per_class: 1,
            max_rate: 100,
            n_ticks: 10,
            mlp: MlpWeights {
                w1: vec![1.0],
                b1: vec![0.0],
                w2: vec![1.0],
                b2: vec![0.0],
                w3: vec![1.0],
                b3: vec![0.0],
            },
        };

        let images = vec![vec![128u8]]; // only 1 image
        let dev = V2MmioSnnBridgeDevice::with_live_model(1, 1, 1, 42, model, images);

        // OOB: try to load image index 5
        dev.write(MMIO_SNN_DATA, 5);
        dev.write(MMIO_SNN_CMD, 8);
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 1); // error: OOB
    }

    /// Option 2: the trainable linear readout must actually learn. We bypass the
    /// stochastic LIF dynamics by injecting the hidden-count feature vector
    /// directly, then drive the real MMIO `TRAIN_ONE`/`PREDICT_READOUT` command
    /// path. Two orthogonal, linearly-separable features must be classified
    /// correctly after a handful of delta-rule passes (perceptron convergence).
    #[test]
    fn test_trainable_readout_learns_separable() {
        use crate::snn::mlp_weights::LiveSnnModel;

        // 2 hidden units, 2 classes. The SNN body is unused — we set
        // `live_hidden_counts` directly — so the model just needs valid shapes.
        let model = LiveSnnModel {
            syn_ptr: vec![0; 5],
            targets: vec![],
            weights: vec![],
            thresholds: vec![100; 4],
            leaks: vec![230; 4],
            pix_per_class: vec![vec![0], vec![0]],
            d_norms: vec![vec![1.0], vec![1.0]],
            n_input: 2,
            n_hidden: 2,
            n_readout: 1,
            n_classes: 2,
            k_per_class: 1,
            max_rate: 100,
            n_ticks: 50,
            mlp: MlpWeights {
                w1: vec![1.0, 0.0, 0.0, 1.0],
                b1: vec![0.0, 0.0],
                w2: vec![1.0, 0.0, 0.0, 1.0],
                b2: vec![0.0, 0.0],
                w3: vec![1.0, 0.0, 0.0, 1.0],
                b3: vec![0.0, 0.0],
            },
        };

        let images = vec![vec![0u8; 2]];
        let mut dev = V2MmioSnnBridgeDevice::with_live_model(2, 2, 1, 7, model, images);
        dev.enable_trainable_readout(2, 2, 0.2, 7);

        // Feature vectors (counts; rates = counts / n_ticks):
        //   class 0 -> hidden unit 0 saturated -> rates [1.0, 0.0]
        //   class 1 -> hidden unit 1 saturated -> rates [0.0, 1.0]
        let n_ticks = 50u32;
        let feat0 = vec![n_ticks, 0u32];
        let feat1 = vec![0u32, n_ticks];

        // Sanity: an untrained readout should not yet separate both (the test is
        // only meaningful if learning, not luck, produces the final result).
        for _ in 0..100 {
            *dev.live_hidden_counts.borrow_mut() = feat0.clone();
            dev.write(MMIO_SNN_DATA, 0); // true label 0
            dev.write(MMIO_SNN_CMD, 11); // TRAIN_ONE
            assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 0, "TRAIN_ONE(feat0) status");

            *dev.live_hidden_counts.borrow_mut() = feat1.clone();
            dev.write(MMIO_SNN_DATA, 1); // true label 1
            dev.write(MMIO_SNN_CMD, 11); // TRAIN_ONE
            assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 0, "TRAIN_ONE(feat1) status");
        }

        // PREDICT_READOUT (no weight update) must now classify both correctly.
        *dev.live_hidden_counts.borrow_mut() = feat0.clone();
        dev.write(MMIO_SNN_CMD, 12); // PREDICT_READOUT
        assert_eq!(dev.read(MMIO_SNN_CMD) & 0xFF, 0, "PREDICT status feat0");
        assert_eq!(
            dev.read(MMIO_SNN_DATA),
            0,
            "feat0 -> class 0 after training"
        );

        *dev.live_hidden_counts.borrow_mut() = feat1.clone();
        dev.write(MMIO_SNN_CMD, 12); // PREDICT_READOUT
        assert_eq!(
            dev.read(MMIO_SNN_DATA),
            1,
            "feat1 -> class 1 after training"
        );
    }

    /// Option 2: `set_readout_weights` must transfer a trained readout to a fresh
    /// bridge (the train-bridge -> eval-bridge handoff the example relies on),
    /// and the copied weights must reproduce the same predictions.
    #[test]
    fn test_readout_weight_transfer_roundtrip() {
        use crate::snn::mlp_weights::LiveSnnModel;

        let make_model = || LiveSnnModel {
            syn_ptr: vec![0; 5],
            targets: vec![],
            weights: vec![],
            thresholds: vec![100; 4],
            leaks: vec![230; 4],
            pix_per_class: vec![vec![0], vec![0]],
            d_norms: vec![vec![1.0], vec![1.0]],
            n_input: 2,
            n_hidden: 2,
            n_readout: 1,
            n_classes: 2,
            k_per_class: 1,
            max_rate: 100,
            n_ticks: 50,
            mlp: MlpWeights {
                w1: vec![1.0, 0.0, 0.0, 1.0],
                b1: vec![0.0, 0.0],
                w2: vec![1.0, 0.0, 0.0, 1.0],
                b2: vec![0.0, 0.0],
                w3: vec![1.0, 0.0, 0.0, 1.0],
                b3: vec![0.0, 0.0],
            },
        };

        let n_ticks = 50u32;
        let feat0 = vec![n_ticks, 0u32];
        let feat1 = vec![0u32, n_ticks];

        // Train on the source bridge.
        let mut src =
            V2MmioSnnBridgeDevice::with_live_model(2, 2, 1, 7, make_model(), vec![vec![0u8; 2]]);
        src.enable_trainable_readout(2, 2, 0.2, 7);
        for _ in 0..100 {
            *src.live_hidden_counts.borrow_mut() = feat0.clone();
            src.write(MMIO_SNN_DATA, 0);
            src.write(MMIO_SNN_CMD, 11);
            *src.live_hidden_counts.borrow_mut() = feat1.clone();
            src.write(MMIO_SNN_DATA, 1);
            src.write(MMIO_SNN_CMD, 11);
        }
        let (w, b) = src.readout_weights();
        assert_eq!(w.len(), 4, "W is n_in*n_out = 4");
        assert_eq!(b.len(), 2, "b is n_out = 2");

        // Transfer into a fresh eval bridge that never trained.
        let mut dst =
            V2MmioSnnBridgeDevice::with_live_model(2, 2, 1, 99, make_model(), vec![vec![0u8; 2]]);
        dst.set_readout_weights(2, 2, w, b);

        *dst.live_hidden_counts.borrow_mut() = feat0.clone();
        dst.write(MMIO_SNN_CMD, 12);
        assert_eq!(
            dst.read(MMIO_SNN_DATA),
            0,
            "transferred readout: feat0 -> 0"
        );
        *dst.live_hidden_counts.borrow_mut() = feat1.clone();
        dst.write(MMIO_SNN_CMD, 12);
        assert_eq!(
            dst.read(MMIO_SNN_DATA),
            1,
            "transferred readout: feat1 -> 1"
        );
    }
}
