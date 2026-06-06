use crate::simulation::Simulation;
use crate::tile_cpu::v2_components::{
    EXT_OFFSET, EXT_RD_HI, EXT_REG_INDIRECT, EXT_RS_HI, EXT_WIDE_IMM, effective_reg,
};
use crate::tile_cpu::v2_mmio_devices::{
    DS_ERR_INVALID_CMD, DS_ERR_NO_SAMPLE, DS_ERR_OOB, DS_OK, MMIO_CONSOLE_COUNT, MMIO_CONSOLE_DATA,
    MMIO_DATASET_CMD, MMIO_DATASET_DATA, MMIO_DATASET_STATUS, MMIO_MAILBOX_IN, MMIO_MAILBOX_OUT,
    MMIO_MATH_A, MMIO_MATH_B, MMIO_MATH_CMD, MMIO_MATH_RESULT, MMIO_QUANTUM_CMD, MMIO_QUANTUM_DATA,
    MMIO_QUANTUM_PARAM, MMIO_QUANTUM_QUBIT, MMIO_RNG_DATA, MMIO_SNN_CMD, MMIO_SNN_DATA,
    MMIO_TIMER_CYCLE,
};

use super::v2_builder::V2Builder;
use super::v2_execute::TileCpuV2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2IssState {
    /// Sprint 370 (Gate B.3): u32 to hold the wide 16-bit PC / return address.
    pub pc: u32,
    pub lr: u32,
    pub flag_z: bool,
    pub flag_c: bool,
    pub halted: bool,
    pub regs: [u64; 16],
    pub ram: [u64; 128],
    /// Sprint 362 (Gate A): extended main memory at addresses 128..(128+len).
    pub main_mem: Vec<u64>,
    /// Sprint 362: data-address mask (default 0x7F).
    pub mem_addr_mask: usize,
    // Sprint 157: ISS math device mirror
    pub iss_math_a: u64,
    pub iss_math_b: u64,
    pub iss_math_status: u64,
    pub iss_math_result: u64,
}

// Sprint 299: Pre-decoded instruction cache — eliminates per-step bit extraction.
#[derive(Clone, Copy, Default)]
struct PreDecoded {
    opcode: u8,         // ((word >> 11) & 0x1F)
    rd: u8,             // effective_reg(rd_lo, rd_hi) — 0..15
    rs: u8,             // effective_reg(rs_lo, rs_hi) — 0..15
    imm8: u8,           // (word & 0xFF) — raw, for sub-fn / targets / shifts
    imm_val: u16,       // wide-expanded immediate (max 16 bits)
    has_offset: bool,   // (ir_ext & EXT_OFFSET) != 0
    offset: u8,         // (ir_ext >> 8) & 0xFF
    reg_indirect: bool, // Sprint 363: (ir_ext & EXT_REG_INDIRECT) != 0 — JMPR/CALLR
}

fn build_decoded(program: &[u32]) -> Vec<PreDecoded> {
    let mut decoded = vec![PreDecoded::default(); program.len()];
    for i in 0..program.len() {
        let word = program[i];
        let rd_lo = ((word >> 8) & 0x07) as u8;
        let rs_lo = ((word >> 5) & 0x07) as u8;
        let imm8 = (word & 0xFF) as u8;
        let ir_ext = ((word >> 16) & 0xFFFF) as u16;
        let rd_hi = (ir_ext & EXT_RD_HI) != 0;
        let rs_hi = (ir_ext & EXT_RS_HI) != 0;
        let wide_imm = (ir_ext & EXT_WIDE_IMM) != 0;
        let imm_val: u16 = if wide_imm {
            let hi = ((ir_ext >> 8) & 0xFF) as u16;
            (hi << 8) | (imm8 as u16)
        } else {
            imm8 as u16
        };
        decoded[i] = PreDecoded {
            opcode: ((word >> 11) & 0x1F) as u8,
            rd: effective_reg(rd_lo, rd_hi),
            rs: effective_reg(rs_lo, rs_hi),
            imm8,
            imm_val,
            has_offset: (ir_ext & EXT_OFFSET) != 0,
            offset: ((ir_ext >> 8) & 0xFF) as u8,
            reg_indirect: (ir_ext & EXT_REG_INDIRECT) != 0,
        };
    }
    decoded
}

pub struct V2Iss {
    // Sprint 363 (Gate B): expanded to 256 entries (was 128). The PC mask
    // (default 0x7F) gates how much is reachable — legacy programs stay 7-bit;
    // `enable_extended_pc()` widens to 0xFF for up to 256 instructions.
    // Sprint 370 (Gate B.3): Vec to hold >256 instructions (wide PC).
    program: Vec<u32>,
    // Sprint 299: Pre-decoded instruction cache.
    decoded: Vec<PreDecoded>,
    /// Sprint 363/370 (Gate B): instruction-address mask. 0x7F (legacy, 128 instrs),
    /// 0xFF (extended, 256), or 0xFFFF (wide, 65536). The instruction-memory analog
    /// of Gate A's `mem_addr_mask`.
    pc_mask: u32,
    state: V2IssState,
    /// Sprint 158: When false (default), all MMIO addresses pass through to ram[].
    /// When true, MMIO addresses are intercepted by ISS device mirrors.
    iss_mmio_enabled: bool,
    // Sprint 158: ISS quantum device mirror (stored here, not V2IssState, to avoid Eq issues with f32)
    iss_qstate: Option<crate::quantum::QState>,
    iss_qrng: crate::quantum::QRng,
    iss_q_target_qubit: u8,
    iss_q_param_bits: u64,
    iss_q_last_result: u64,
    iss_q_status: u8,
    iss_q_gate_count: u32,
    // Sprint 158: ISS ref device mirrors
    iss_timer_steps: u64,
    iss_console_last: u64,
    iss_console_count: u64,
    /// Sprint 376 (Gate E.2): full accumulated console output (mirrors the physical
    /// device's `console_buffer`), so compiled-program output can be read back on the ISS.
    iss_console_buffer: Vec<u8>,
    iss_rng_state: u64,
    // Sprint 297: Split mailbox into separate recv/send fields.
    // Matches V2LinkMailboxDevice: read returns recv, write records send.
    iss_mailbox_in_recv: u64,  // mem_read(60) returns this
    iss_mailbox_in_send: u64,  // mem_write(60) records here
    iss_mailbox_out_recv: u64, // mem_read(61) returns this
    iss_mailbox_out_send: u64, // mem_write(61) records here
    // Sprint 159: ISS dataset device mirror
    iss_dataset_samples: Vec<(u64, u64)>, // (features, label) pairs
    iss_dataset_cursor: Option<usize>,
    iss_dataset_data: u64,
    iss_dataset_correct: u64,
    iss_dataset_last_error: u8,
    iss_dataset_cmd_count: u64,
    // M11: ISS SNN bridge mirror (INFER command only)
    iss_snn_data: u64,
    iss_snn_status: u64,
    iss_snn_model: Option<(crate::snn::MlpWeights, crate::snn::CachedRates)>,
    // M12: ISS live SNN mirror
    iss_live_model: Option<crate::snn::mlp_weights::LiveSnnModel>,
    iss_live_images: Vec<Vec<u8>>,
    iss_live_v_mem: Vec<i16>,
    iss_live_refract: Vec<u8>,
    iss_live_spiked: Vec<u8>,
    iss_live_hidden_counts: Vec<u32>,
    iss_live_input_rates: Vec<u8>,
    iss_live_rng_seed: u32,
}

impl V2Iss {
    pub fn from_program(program: &[u32]) -> Self {
        // Sprint 370 (Gate B.3): size the ROM to the program (min 256 for legacy
        // index safety), so the wide PC can address >256 instructions.
        let cap = program.len().max(256);
        let mut rom = vec![0u32; cap];
        rom[..program.len()].copy_from_slice(program);
        let decoded = build_decoded(&rom);
        Self {
            program: rom,
            decoded,
            pc_mask: 0x7F,
            state: V2IssState {
                pc: 0,
                lr: 0,
                flag_z: false,
                flag_c: false,
                halted: false,
                regs: [0u64; 16],
                ram: [0u64; 128],
                main_mem: Vec::new(),
                mem_addr_mask: 0x7F,
                iss_math_a: 0,
                iss_math_b: 0,
                iss_math_status: 0,
                iss_math_result: 0,
            },
            iss_mmio_enabled: false,
            iss_qstate: None,
            iss_qrng: crate::quantum::QRng::new(0x156_CAFE_u64.wrapping_add(157)),
            iss_q_target_qubit: 0,
            iss_q_param_bits: 0,
            iss_q_last_result: 0,
            iss_q_status: 1, // QUANTUM_STATUS_NOT_INIT
            iss_q_gate_count: 0,
            iss_timer_steps: 0,
            iss_console_last: 0,
            iss_console_count: 0,
            iss_console_buffer: Vec::new(),
            iss_rng_state: 0x156_CAFE, // same seed as hardware V2MmioRefDevicePack
            iss_mailbox_in_recv: 0,
            iss_mailbox_in_send: 0,
            iss_mailbox_out_recv: 0,
            iss_mailbox_out_send: 0,
            iss_dataset_samples: Vec::new(),
            iss_dataset_cursor: None,
            iss_dataset_data: 0,
            iss_dataset_correct: 0,
            iss_dataset_last_error: 0,
            iss_dataset_cmd_count: 0,
            iss_snn_data: 0,
            iss_snn_status: 0,
            iss_snn_model: None,
            iss_live_model: None,
            iss_live_images: Vec::new(),
            iss_live_v_mem: Vec::new(),
            iss_live_refract: Vec::new(),
            iss_live_spiked: Vec::new(),
            iss_live_hidden_counts: Vec::new(),
            iss_live_input_rates: Vec::new(),
            iss_live_rng_seed: 0,
        }
    }

    /// Sprint 158: Create ISS with MMIO interception enabled.
    /// Use when the hardware CPU has `.with_mmio()`.
    pub fn from_program_with_mmio(program: &[u32]) -> Self {
        let mut iss = Self::from_program(program);
        iss.iss_mmio_enabled = true;
        iss
    }

    /// Sprint 159: Create ISS with MMIO + dataset mirror.
    pub fn from_program_with_dataset(program: &[u32], samples: Vec<(u64, u64)>) -> Self {
        let mut iss = Self::from_program(program);
        iss.iss_mmio_enabled = true;
        iss.iss_dataset_samples = samples;
        iss
    }

    /// M11: Create ISS with MMIO + dataset + SNN INFER mirror.
    pub fn from_program_with_snn_model(
        program: &[u32],
        samples: Vec<(u64, u64)>,
        weights: crate::snn::MlpWeights,
        cached_rates: crate::snn::CachedRates,
    ) -> Self {
        let mut iss = Self::from_program(program);
        iss.iss_mmio_enabled = true;
        iss.iss_dataset_samples = samples;
        iss.iss_snn_model = Some((weights, cached_rates));
        iss
    }

    /// M12: Create ISS with MMIO + dataset + live SNN model.
    pub fn from_program_with_live_snn_model(
        program: &[u32],
        samples: Vec<(u64, u64)>,
        live_model: crate::snn::mlp_weights::LiveSnnModel,
        images: Vec<Vec<u8>>,
        seed: u64,
    ) -> Self {
        let n_total = live_model.n_neurons();
        let n_hid = live_model.n_hidden;
        let n_inp = live_model.n_input;
        let mut iss = Self::from_program(program);
        iss.iss_mmio_enabled = true;
        iss.iss_dataset_samples = samples;
        iss.iss_live_v_mem = vec![0i16; n_total];
        iss.iss_live_refract = vec![0u8; n_total];
        iss.iss_live_spiked = vec![0u8; n_total];
        iss.iss_live_hidden_counts = vec![0u32; n_hid];
        iss.iss_live_input_rates = vec![0u8; n_inp];
        iss.iss_live_rng_seed = seed as u32;
        iss.iss_live_images = images;
        iss.iss_live_model = Some(live_model);
        iss
    }

    pub fn state(&self) -> &V2IssState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut V2IssState {
        &mut self.state
    }

    /// Sprint 376 (Gate E.2): accumulated console output (bytes written to
    /// MMIO_CONSOLE_DATA), captured when the ISS was built with MMIO interception.
    pub fn console_output(&self) -> &[u8] {
        &self.iss_console_buffer
    }

    /// Sprint 376 (Gate E.2): accumulated console output as a UTF-8 string (lossy).
    pub fn console_string(&self) -> String {
        String::from_utf8_lossy(&self.iss_console_buffer).into_owned()
    }

    /// Sprint 362 (Gate A): allocate `words` of extended main memory and widen
    /// the data-address mask. Mirrors `TileCpuV2::configure_main_memory` exactly
    /// so the ISS and physical executor share an identical address space and
    /// equal-length main-memory vectors for differential comparison.
    pub fn configure_main_memory(&mut self, words: usize) {
        if words == 0 {
            self.state.main_mem = Vec::new();
            self.state.mem_addr_mask = 0x7F;
            return;
        }
        let total = (128 + words).next_power_of_two().max(128);
        self.state.mem_addr_mask = total - 1;
        self.state.main_mem = vec![0u64; total - 128];
    }

    /// Sprint 363 (Gate B): widen the instruction address space from 128 to 256
    /// instructions (PC mask 0x7F -> 0xFF). Programs can then occupy ROM entries
    /// 0..255 and use JMPR/CALLR (register-indirect far-jump) to reach the upper
    /// half past the 8-bit branch-immediate limit.
    pub fn enable_extended_pc(&mut self) {
        self.pc_mask = 0xFF;
    }

    /// Sprint 370 (Gate B.3): widen the instruction address space to 16 bits (PC
    /// mask 0xFFFF, up to 65,536 instructions). Mirrors the physical wide PC.
    pub fn enable_wide_pc(&mut self) {
        self.pc_mask = 0xFFFF;
    }

    /// Sprint 299: Access raw program word at a given PC.
    pub fn program_word(&self, pc: usize) -> u32 {
        self.program
            .get(pc & self.pc_mask as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Sprint 299: Access pre-decoded fields at a given PC (for verification).
    /// Returns (opcode, rd, rs, imm8, imm_val, has_offset, offset).
    pub fn decoded_at(&self, pc: usize) -> (u8, u8, u8, u8, u16, bool, u8) {
        let d = self
            .decoded
            .get(pc & self.pc_mask as usize)
            .copied()
            .unwrap_or_default();
        (
            d.opcode,
            d.rd,
            d.rs,
            d.imm8,
            d.imm_val,
            d.has_offset,
            d.offset,
        )
    }

    pub fn is_halted(&self) -> bool {
        self.state.halted
    }

    // ── Sprint 297: Mailbox accessors for fabric ──

    pub fn mailbox_in_recv(&self) -> u64 {
        self.iss_mailbox_in_recv
    }
    pub fn mailbox_in_send(&self) -> u64 {
        self.iss_mailbox_in_send
    }
    pub fn mailbox_out_recv(&self) -> u64 {
        self.iss_mailbox_out_recv
    }
    pub fn mailbox_out_send(&self) -> u64 {
        self.iss_mailbox_out_send
    }
    pub fn set_mailbox_in_recv(&mut self, v: u64) {
        self.iss_mailbox_in_recv = v;
    }
    pub fn set_mailbox_out_recv(&mut self, v: u64) {
        self.iss_mailbox_out_recv = v;
    }
    /// Reset send fields to 0 (called between epochs).
    pub fn clear_mailbox_sends(&mut self) {
        self.iss_mailbox_in_send = 0;
        self.iss_mailbox_out_send = 0;
    }

    // ── Sprint 158: ISS quantum helpers ──

    fn iss_q_n_qubits(&self) -> u8 {
        self.iss_qstate.as_ref().map(|q| q.n_qubits).unwrap_or(0)
    }

    fn iss_q_check_qubit(&mut self, q: u8) -> bool {
        let n = self.iss_q_n_qubits();
        if n == 0 {
            self.iss_q_status = 1; // NOT_INIT
            return false;
        }
        if q >= n {
            self.iss_q_status = 2; // QUBIT_OOB
            return false;
        }
        true
    }

    fn iss_q_apply_gate(&mut self, gate: crate::quantum::QGate) {
        let state = self.iss_qstate.as_mut().unwrap();
        let outcome = crate::quantum::apply_gate_scalar(state, &gate, &mut self.iss_qrng);
        self.iss_q_gate_count += 1;
        self.iss_q_status = 0;
        if let crate::quantum::GateOutcome::Measured { qubit: _, bit } = outcome {
            self.iss_q_last_result = bit as u64;
        }
    }

    /// Execute a quantum command. Mirrors V2MmioQuantumDevice::execute_cmd exactly.
    fn iss_quantum_execute_cmd(&mut self, cmd: u64) {
        let data = self.iss_q_last_result;
        match cmd {
            0 => {
                // INIT
                let n = (self.iss_q_last_result as u8).min(8).max(1);
                self.iss_qstate = Some(crate::quantum::QState::new_zero(n));
                self.iss_q_gate_count = 0;
                self.iss_q_status = 0;
            }
            1 => {
                // RESET
                let n = self.iss_q_n_qubits();
                if n == 0 {
                    self.iss_q_status = 1;
                    return;
                }
                self.iss_qstate = Some(crate::quantum::QState::new_zero(n));
                self.iss_q_gate_count = 0;
                self.iss_q_status = 0;
            }
            2 => {
                // H
                let q = self.iss_q_target_qubit;
                if self.iss_q_check_qubit(q) {
                    self.iss_q_apply_gate(crate::quantum::QGate::H(q));
                }
            }
            3 => {
                // X
                let q = self.iss_q_target_qubit;
                if self.iss_q_check_qubit(q) {
                    self.iss_q_apply_gate(crate::quantum::QGate::X(q));
                }
            }
            4 => {
                // Y
                let q = self.iss_q_target_qubit;
                if self.iss_q_check_qubit(q) {
                    self.iss_q_apply_gate(crate::quantum::QGate::Y(q));
                }
            }
            5 => {
                // Z
                let q = self.iss_q_target_qubit;
                if self.iss_q_check_qubit(q) {
                    self.iss_q_apply_gate(crate::quantum::QGate::Z(q));
                }
            }
            6 => {
                // T
                let q = self.iss_q_target_qubit;
                if self.iss_q_check_qubit(q) {
                    self.iss_q_apply_gate(crate::quantum::QGate::T(q));
                }
            }
            7 => {
                // CNOT
                let c = self.iss_q_target_qubit;
                let t = data as u8;
                if self.iss_q_check_qubit(c) && self.iss_q_check_qubit(t) {
                    self.iss_q_apply_gate(crate::quantum::QGate::CNot(c, t));
                }
            }
            8 => {
                // CZ
                let c = self.iss_q_target_qubit;
                let t = data as u8;
                if self.iss_q_check_qubit(c) && self.iss_q_check_qubit(t) {
                    self.iss_q_apply_gate(crate::quantum::QGate::CZ(c, t));
                }
            }
            9 => {
                // RZ
                let q = self.iss_q_target_qubit;
                if self.iss_q_check_qubit(q) {
                    let theta = f64::from_bits(self.iss_q_param_bits) as f32;
                    self.iss_q_apply_gate(crate::quantum::QGate::Rz(q, theta));
                }
            }
            10 => {
                // MEASURE
                let q = self.iss_q_target_qubit;
                if self.iss_q_check_qubit(q) {
                    self.iss_q_apply_gate(crate::quantum::QGate::Measure(q));
                }
            }
            11 => {
                // MEASURE_ALL
                let n = self.iss_q_n_qubits();
                if n == 0 {
                    self.iss_q_status = 1;
                    return;
                }
                let mut result: u64 = 0;
                for q in 0..n {
                    let state = self.iss_qstate.as_mut().unwrap();
                    let outcome = crate::quantum::apply_gate_scalar(
                        state,
                        &crate::quantum::QGate::Measure(q),
                        &mut self.iss_qrng,
                    );
                    if let crate::quantum::GateOutcome::Measured { qubit: _, bit } = outcome {
                        if bit != 0 {
                            result |= 1u64 << q;
                        }
                    }
                }
                self.iss_q_gate_count += n as u32;
                self.iss_q_last_result = result;
                self.iss_q_status = 0;
            }
            12 => {
                // PROB
                let q = self.iss_q_target_qubit;
                if self.iss_q_n_qubits() == 0 {
                    self.iss_q_status = 1;
                    return;
                }
                if q >= self.iss_q_n_qubits() {
                    self.iss_q_status = 2;
                    return;
                }
                let state = self.iss_qstate.as_ref().unwrap();
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
                self.iss_q_last_result = prob.to_bits();
                self.iss_q_status = 0;
            }
            13 => {
                // RX
                let q = self.iss_q_target_qubit;
                if self.iss_q_check_qubit(q) {
                    let theta = f64::from_bits(self.iss_q_param_bits) as f32;
                    self.iss_q_apply_gate(crate::quantum::QGate::Rx(q, theta));
                }
            }
            14 => {
                // RY
                let q = self.iss_q_target_qubit;
                if self.iss_q_check_qubit(q) {
                    let theta = f64::from_bits(self.iss_q_param_bits) as f32;
                    self.iss_q_apply_gate(crate::quantum::QGate::Ry(q, theta));
                }
            }
            15 => {
                // SWAP
                let q1 = self.iss_q_target_qubit;
                let q2 = data as u8;
                if self.iss_q_check_qubit(q1) && self.iss_q_check_qubit(q2) {
                    self.iss_q_apply_gate(crate::quantum::QGate::Swap(q1, q2));
                }
            }
            16 => {
                // TDG
                let q = self.iss_q_target_qubit;
                if self.iss_q_check_qubit(q) {
                    self.iss_q_apply_gate(crate::quantum::QGate::Tdg(q));
                }
            }
            _ => {}
        }
    }

    /// ISS RNG: mirrors V2MmioRefDevicePack::next_rng_u64 exactly.
    fn iss_next_rng(&mut self) -> u64 {
        let mut x = self.iss_rng_state;
        if x == 0 {
            x = 0x9E37_79B9_7F4A_7C15;
        }
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.iss_rng_state = x;
        x
    }

    // ── Sprint 159: ISS dataset command execution ──

    fn iss_dataset_execute_cmd(&mut self, cmd: u64) {
        self.iss_dataset_cmd_count += 1;
        match cmd {
            0 => {
                // LOAD_SAMPLE
                let idx = self.iss_dataset_data as usize;
                if idx >= self.iss_dataset_samples.len() {
                    self.iss_dataset_last_error = DS_ERR_OOB;
                } else {
                    self.iss_dataset_cursor = Some(idx);
                    self.iss_dataset_last_error = DS_OK;
                }
            }
            1 => {
                // GET_FEATURES
                if let Some(idx) = self.iss_dataset_cursor {
                    self.iss_dataset_data = self.iss_dataset_samples[idx].0;
                    self.iss_dataset_last_error = DS_OK;
                } else {
                    self.iss_dataset_last_error = DS_ERR_NO_SAMPLE;
                }
            }
            2 => {
                // GET_LABEL
                if let Some(idx) = self.iss_dataset_cursor {
                    self.iss_dataset_data = self.iss_dataset_samples[idx].1;
                    self.iss_dataset_last_error = DS_OK;
                } else {
                    self.iss_dataset_last_error = DS_ERR_NO_SAMPLE;
                }
            }
            3 => {
                // GET_COUNT
                self.iss_dataset_data = self.iss_dataset_samples.len() as u64;
                self.iss_dataset_last_error = DS_OK;
            }
            4 => {
                // GET_CORRECT
                self.iss_dataset_data = self.iss_dataset_correct;
                self.iss_dataset_last_error = DS_OK;
            }
            5 => {
                // SUBMIT_PREDICTION
                if let Some(idx) = self.iss_dataset_cursor {
                    let prediction = self.iss_dataset_data;
                    if prediction == self.iss_dataset_samples[idx].1 {
                        self.iss_dataset_correct += 1;
                    }
                    self.iss_dataset_last_error = DS_OK;
                } else {
                    self.iss_dataset_last_error = DS_ERR_NO_SAMPLE;
                }
            }
            _ => {
                self.iss_dataset_last_error = DS_ERR_INVALID_CMD;
            }
        }
    }

    /// M11: Execute SNN MMIO command (mirrors V2MmioSnnBridgeDevice).
    fn iss_snn_execute_cmd(&mut self, cmd: u64) {
        match cmd {
            7 => {
                // INFER: run MLP forward_cpu on cached rates[data]
                if let Some((ref weights, ref rates)) = self.iss_snn_model {
                    let idx = self.iss_snn_data as usize;
                    if idx < rates.n_samples() {
                        let pred = weights.forward_cpu(rates.get(idx));
                        self.iss_snn_data = pred as u64;
                        self.iss_snn_status = 0; // ok
                    } else {
                        self.iss_snn_data = 0;
                        self.iss_snn_status = 1; // OOB
                    }
                } else {
                    self.iss_snn_data = 0;
                    self.iss_snn_status = 1; // no model
                }
            }
            8 => {
                // LOAD_IMAGE: encode image pixels, reset SNN state
                if let Some(ref model) = self.iss_live_model {
                    let idx = self.iss_snn_data as usize;
                    if idx < self.iss_live_images.len() {
                        self.iss_live_input_rates = model.encode_image(&self.iss_live_images[idx]);
                        let n_total = model.n_neurons();
                        self.iss_live_v_mem = vec![0i16; n_total];
                        self.iss_live_refract = vec![0u8; n_total];
                        self.iss_live_spiked = vec![0u8; n_total];
                        self.iss_live_hidden_counts = vec![0u32; model.n_hidden];
                        self.iss_snn_status = 0;
                    } else {
                        self.iss_snn_status = 1; // OOB
                    }
                } else {
                    self.iss_snn_status = 1; // no model
                }
            }
            9 => {
                // SNN_RUN: live LIF simulation
                if let Some(ref model) = self.iss_live_model {
                    let n_total = model.n_neurons();
                    for tick in 0..model.n_ticks {
                        // Phase A: Poisson input
                        let tick_seed = self.iss_live_rng_seed.wrapping_add(tick as u32 * 1000003);
                        for i in 0..model.n_input {
                            if self.iss_live_refract[i] > 0 {
                                self.iss_live_refract[i] -= 1;
                                self.iss_live_spiked[i] = 0;
                                continue;
                            }
                            let mut rng = tick_seed ^ (i as u32).wrapping_mul(2654435761);
                            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                            let rand_val = (rng >> 24) as u8;
                            if rand_val < self.iss_live_input_rates[i] {
                                self.iss_live_spiked[i] = 1;
                                self.iss_live_v_mem[i] = -128;
                                self.iss_live_refract[i] = 2;
                            } else {
                                self.iss_live_spiked[i] = 0;
                            }
                        }

                        // Phase B: CSR current accumulation
                        let mut currents = vec![0i32; n_total];
                        for src in 0..n_total {
                            if self.iss_live_spiked[src] != 0 {
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

                        // Phase C: LIF dynamics
                        for i in model.n_input..n_total {
                            if self.iss_live_refract[i] > 0 {
                                self.iss_live_refract[i] -= 1;
                                self.iss_live_spiked[i] = 0;
                                continue;
                            }
                            let v = ((self.iss_live_v_mem[i] as i32 * model.leaks[i] as i32) >> 8)
                                + currents[i];
                            let v = v.clamp(-32768, 32767);
                            if v >= model.thresholds[i] as i32 {
                                self.iss_live_spiked[i] = 1;
                                self.iss_live_v_mem[i] = -128;
                                self.iss_live_refract[i] = 2;
                            } else {
                                self.iss_live_spiked[i] = 0;
                                self.iss_live_v_mem[i] = v as i16;
                            }
                        }

                        // Phase D: Count hidden spikes
                        for h in 0..model.n_hidden {
                            if self.iss_live_spiked[model.n_input + h] != 0 {
                                self.iss_live_hidden_counts[h] += 1;
                            }
                        }

                        // Clear input spikes
                        for i in 0..model.n_input {
                            self.iss_live_spiked[i] = 0;
                        }
                    }
                    self.iss_snn_status = 0;
                } else {
                    self.iss_snn_status = 1;
                }
            }
            10 => {
                // INFER_LIVE: compute rates, MLP forward
                if let Some(ref model) = self.iss_live_model {
                    let n_ticks = model.n_ticks as f32;
                    let rates: Vec<f32> = self
                        .iss_live_hidden_counts
                        .iter()
                        .map(|&c| c as f32 / n_ticks)
                        .collect();
                    let pred = model.mlp.forward_cpu(&rates);
                    self.iss_snn_data = pred as u64;
                    self.iss_snn_status = 0;
                } else {
                    self.iss_snn_data = 0;
                    self.iss_snn_status = 1;
                }
            }
            _ => {
                // Commands 0-6 (toy LIF model) not mirrored in ISS
                self.iss_snn_status = 0;
            }
        }
    }

    // ── MMIO interception ──

    /// Sprint 362 (Gate A): route a (masked) data address to scratchpad RAM
    /// (addr < 128) or extended main memory (addr >= 128).
    #[inline(always)]
    fn ram_load(&self, addr: usize) -> u64 {
        if addr < 128 {
            self.state.ram[addr]
        } else {
            self.state.main_mem[addr - 128]
        }
    }

    /// Sprint 362: store to scratchpad RAM or extended main memory.
    #[inline(always)]
    fn ram_store(&mut self, addr: usize, value: u64) {
        if addr < 128 {
            self.state.ram[addr] = value;
        } else {
            self.state.main_mem[addr - 128] = value;
        }
    }

    /// Sprint 299: Const-generic memory read — MMIO branch eliminated at compile time.
    /// Sprint 362: mask once with mem_addr_mask, then route. MMIO devices live
    /// only in 41..63 (< 128); addr >= 128 always falls through to main memory.
    #[inline(always)]
    fn mem_read_inner<const MMIO: bool>(&mut self, addr: usize) -> u64 {
        let addr = addr & self.state.mem_addr_mask;
        if MMIO && addr < 128 {
            return self.mem_read_mmio(addr);
        }
        self.ram_load(addr)
    }

    /// MMIO read dispatch — cold path, called only when MMIO is enabled.
    #[cold]
    fn mem_read_mmio(&mut self, addr: usize) -> u64 {
        let a = (addr & 0x7F) as u8;
        let mmio_val = match a {
            // Dataset (41-43)
            x if x == MMIO_DATASET_CMD && !self.iss_dataset_samples.is_empty() => Some(
                ((self.iss_dataset_last_error as u64) << 8) | (self.iss_dataset_cmd_count & 0xFF),
            ),
            x if x == MMIO_DATASET_DATA && !self.iss_dataset_samples.is_empty() => {
                Some(self.iss_dataset_data)
            }
            x if x == MMIO_DATASET_STATUS && !self.iss_dataset_samples.is_empty() => {
                let idx = self
                    .iss_dataset_cursor
                    .map(|i| i as u64)
                    .unwrap_or(u64::MAX);
                Some((idx << 32) | (self.iss_dataset_samples.len() as u64))
            }
            // Quantum (44-47)
            x if x == MMIO_QUANTUM_CMD => {
                let gc = self.iss_q_gate_count as u64;
                let nq = self.iss_q_n_qubits() as u64;
                let st = self.iss_q_status as u64;
                Some((gc << 16) | (nq << 8) | st)
            }
            x if x == MMIO_QUANTUM_QUBIT => Some(self.iss_q_target_qubit as u64),
            x if x == MMIO_QUANTUM_DATA => Some(self.iss_q_last_result),
            x if x == MMIO_QUANTUM_PARAM => Some(self.iss_q_param_bits),
            // SNN (54-55)
            x if x == MMIO_SNN_DATA
                && (self.iss_snn_model.is_some() || self.iss_live_model.is_some()) =>
            {
                Some(self.iss_snn_data)
            }
            x if x == MMIO_SNN_CMD
                && (self.iss_snn_model.is_some() || self.iss_live_model.is_some()) =>
            {
                Some(self.iss_snn_status)
            }
            // Math (50-53)
            x if x == MMIO_MATH_A => Some(self.state.iss_math_a),
            x if x == MMIO_MATH_B => Some(self.state.iss_math_b),
            x if x == MMIO_MATH_CMD => Some(self.state.iss_math_status),
            x if x == MMIO_MATH_RESULT => Some(self.state.iss_math_result),
            // Ref devices (56-63)
            x if x == MMIO_TIMER_CYCLE => Some(self.iss_timer_steps),
            x if x == MMIO_CONSOLE_DATA => Some(self.iss_console_last),
            x if x == MMIO_CONSOLE_COUNT => Some(self.iss_console_count),
            x if x == MMIO_RNG_DATA => Some(self.iss_next_rng()),
            x if x == MMIO_MAILBOX_IN => Some(self.iss_mailbox_in_recv),
            x if x == MMIO_MAILBOX_OUT => Some(self.iss_mailbox_out_recv),
            // P-bit (62-63): pass-through to RAM
            _ => None,
        };
        if let Some(v) = mmio_val {
            self.state.ram[addr & 0x7F] = v; // mirror hardware inject
            return v;
        }
        self.state.ram[addr & 0x7F]
    }

    /// Sprint 299: Const-generic memory write — MMIO branch eliminated at compile time.
    #[inline(always)]
    fn mem_write_inner<const MMIO: bool>(&mut self, addr: usize, value: u64) {
        let addr = addr & self.state.mem_addr_mask;
        if MMIO && addr < 128 {
            self.mem_write_mmio(addr, value);
            return;
        }
        self.ram_store(addr, value);
    }

    /// MMIO write dispatch — cold path.
    #[cold]
    fn mem_write_mmio(&mut self, addr: usize, value: u64) {
        let a = (addr & 0x7F) as u8;
        match a {
            // Dataset (41-43)
            x if x == MMIO_DATASET_CMD && !self.iss_dataset_samples.is_empty() => {
                self.iss_dataset_execute_cmd(value);
                return;
            }
            x if x == MMIO_DATASET_DATA && !self.iss_dataset_samples.is_empty() => {
                self.iss_dataset_data = value;
                return;
            }
            x if x == MMIO_DATASET_STATUS && !self.iss_dataset_samples.is_empty() => {
                return; // read-only
            }
            // Quantum (44-47)
            x if x == MMIO_QUANTUM_CMD => {
                self.iss_quantum_execute_cmd(value);
                return;
            }
            x if x == MMIO_QUANTUM_QUBIT => {
                self.iss_q_target_qubit = value as u8;
                return;
            }
            x if x == MMIO_QUANTUM_DATA => {
                self.iss_q_last_result = value;
                return;
            }
            x if x == MMIO_QUANTUM_PARAM => {
                self.iss_q_param_bits = value;
                return;
            }
            // SNN (54-55)
            x if x == MMIO_SNN_CMD
                && (self.iss_snn_model.is_some() || self.iss_live_model.is_some()) =>
            {
                self.iss_snn_execute_cmd(value);
                return;
            }
            x if x == MMIO_SNN_DATA
                && (self.iss_snn_model.is_some() || self.iss_live_model.is_some()) =>
            {
                self.iss_snn_data = value;
                return;
            }
            // Math (50-53)
            x if x == MMIO_MATH_A => {
                self.state.iss_math_a = value;
                return;
            }
            x if x == MMIO_MATH_B => {
                self.state.iss_math_b = value;
                return;
            }
            x if x == MMIO_MATH_CMD => {
                let ma = self.state.iss_math_a;
                let mb = self.state.iss_math_b;
                match value {
                    0 => {
                        self.state.iss_math_result = ma.wrapping_mul(mb);
                        self.state.iss_math_status = 0;
                    }
                    1 => {
                        if mb == 0 {
                            self.state.iss_math_status = 1;
                        } else {
                            self.state.iss_math_result = ma / mb;
                            self.state.iss_math_status = 0;
                        }
                    }
                    2 => {
                        if mb == 0 {
                            self.state.iss_math_status = 1;
                        } else {
                            self.state.iss_math_result = ma % mb;
                            self.state.iss_math_status = 0;
                        }
                    }
                    3 => {
                        let wide = (ma as u128).wrapping_mul(mb as u128);
                        self.state.iss_math_result = (wide >> 64) as u64;
                        self.state.iss_math_status = 0;
                    }
                    4 => {
                        self.state.iss_math_result = ma.count_ones() as u64;
                        self.state.iss_math_status = 0;
                    }
                    _ => {
                        self.state.iss_math_status = 0;
                    }
                }
                return;
            }
            x if x == MMIO_MATH_RESULT => {
                return;
            } // read-only
            // Ref devices (56-63)
            x if x == MMIO_CONSOLE_DATA => {
                self.iss_console_last = value & 0xFF;
                self.iss_console_count += 1;
                self.iss_console_buffer.push((value & 0xFF) as u8);
                return;
            }
            x if x == MMIO_MAILBOX_IN => {
                self.iss_mailbox_in_send = value;
                return;
            }
            x if x == MMIO_MAILBOX_OUT => {
                self.iss_mailbox_out_send = value;
                return;
            }
            // Timer, RNG, console_count, p-bit: writes ignored (matches hardware)
            x if x == MMIO_TIMER_CYCLE => {
                return;
            }
            x if x == MMIO_RNG_DATA => {
                return;
            }
            x if x == MMIO_CONSOLE_COUNT => {
                return;
            }
            _ => {} // P-bit (62-63): fall through to RAM
        }
        self.state.ram[addr & 0x7F] = value;
    }

    /// Execute one instruction. Dispatches to the correct MMIO specialization.
    pub fn step(&mut self) -> Option<u32> {
        if self.iss_mmio_enabled {
            self.step_inner::<true>()
        } else {
            self.step_inner::<false>()
        }
    }

    /// Sprint 299: Direct non-MMIO step — bypasses the MMIO dispatch entirely.
    /// Use when the caller statically knows MMIO is disabled.
    #[inline(always)]
    pub fn step_no_mmio(&mut self) -> Option<u32> {
        self.step_inner::<false>()
    }

    /// Sprint 299: Core step implementation with const-generic MMIO specialization
    /// and pre-decoded instruction cache.
    fn step_inner<const MMIO: bool>(&mut self) -> Option<u32> {
        if self.state.halted {
            return None;
        }
        self.iss_timer_steps += 1;

        let pc = self.state.pc & self.pc_mask;
        // Sprint 370 (Gate B.3): out-of-range fetch reads as NOP (mirrors the
        // physical program_ext .unwrap_or(0) behavior).
        let d = self.decoded.get(pc as usize).copied().unwrap_or_default();
        let imm8 = d.imm8;
        let imm_val = d.imm_val as u64;
        let rd = d.rd as usize;
        let rs = d.rs as usize;

        let a = self.state.regs[rd];
        let b = self.state.regs[rs];
        let mut next_pc = pc.wrapping_add(1) & self.pc_mask;

        let mut result = 0u64;
        let mut update_z = false;
        let mut update_c = false;
        let mut c_value = self.state.flag_c;
        let mut write_reg: Option<usize> = None;

        match d.opcode {
            0x00 => {
                // NOP
            }
            0x01 => {
                // HALT
                self.state.halted = true;
                next_pc = pc;
            }
            0x02 => {
                // MOV/MOVZ/MOVNZ/MOVC/MOVNC rd, rs  (sub_fn 0..=4)
                // Sprint 364 (Gate C): MFLR rd (sub_fn=5), MTLR rs (sub_fn=6)
                let sub_fn = imm8 & 0x1F;
                match sub_fn {
                    5 => {
                        // MFLR rd: regs[rd] = lr
                        result = self.state.lr as u64;
                        write_reg = Some(rd);
                    }
                    6 => {
                        // MTLR rs: lr = regs[rs] (low byte, masked by pc_mask)
                        self.state.lr = (b as u32) & self.pc_mask;
                    }
                    _ => {
                        result = b;
                        let cond_met = match sub_fn {
                            1 => self.state.flag_z,
                            2 => !self.state.flag_z,
                            3 => self.state.flag_c,
                            4 => !self.state.flag_c,
                            _ => true,
                        };
                        if cond_met {
                            write_reg = Some(rd);
                        }
                    }
                }
            }
            0x03 => {
                // LDI rd, imm
                result = imm_val;
                write_reg = Some(rd);
                update_z = true;
            }
            0x04 => {
                // ADD rd, rs / MUL rd, rs
                let sub_fn = imm8 & 0x1F;
                if sub_fn == 1 {
                    result = a.wrapping_mul(b);
                    write_reg = Some(rd);
                    update_z = true;
                } else {
                    result = a.wrapping_add(b);
                    write_reg = Some(rd);
                    update_z = true;
                    update_c = true;
                    c_value = a > result;
                }
            }
            0x05 => {
                // SUB rd, rs
                result = a.wrapping_sub(b);
                write_reg = Some(rd);
                update_z = true;
                update_c = true;
                c_value = result > a;
            }
            0x06 => {
                // AND rd, rs
                result = a & b;
                write_reg = Some(rd);
                update_z = true;
            }
            0x07 => {
                // OR rd, rs
                result = a | b;
                write_reg = Some(rd);
                update_z = true;
            }
            0x08 => {
                // XOR rd, rs
                result = a ^ b;
                write_reg = Some(rd);
                update_z = true;
            }
            0x09 => {
                // NOT/CLZ/CTZ/POPCNT rd
                let sub_fn = imm8 & 0x1F;
                result = match sub_fn {
                    1 => a.leading_zeros() as u64,
                    2 => a.trailing_zeros() as u64,
                    3 => a.count_ones() as u64,
                    _ => !a,
                };
                write_reg = Some(rd);
                update_z = true;
            }
            0x0A => {
                // NEG rd
                result = 0u64.wrapping_sub(b);
                write_reg = Some(rd);
                update_z = true;
                update_c = true;
                c_value = result > a;
            }
            0x0B => {
                // INC rd
                let rhs = imm8 as u64;
                result = a.wrapping_add(rhs);
                write_reg = Some(rd);
                update_z = true;
                update_c = true;
                c_value = a > result;
            }
            0x0C => {
                // DEC rd
                let rhs = imm8 as u64;
                result = a.wrapping_sub(rhs);
                write_reg = Some(rd);
                update_z = true;
                update_c = true;
                c_value = result > a;
            }
            0x0D => {
                // SHL rd, imm
                let rhs = imm8 as u64;
                result = a.wrapping_shl((rhs & 63) as u32);
                write_reg = Some(rd);
                update_z = true;
                update_c = true;
                let bit_pos = (8u64.wrapping_sub(rhs)) & 63;
                c_value = ((a >> bit_pos) & 1) != 0;
            }
            0x0E => {
                // SHR rd, imm / SRA rd, imm
                let is_arithmetic = (imm8 & 0x08) != 0;
                let shift_amount = (imm8 & 0x07) as u64;
                if is_arithmetic {
                    result = (a as i64).wrapping_shr((shift_amount & 63) as u32) as u64;
                } else {
                    result = a.wrapping_shr((shift_amount & 63) as u32);
                }
                write_reg = Some(rd);
                update_z = true;
                update_c = true;
                let bit_pos = shift_amount.wrapping_sub(1) & 63;
                c_value = ((a >> bit_pos) & 1) != 0;
            }
            0x0F => {
                // CMP rd, rs
                result = a.wrapping_sub(b);
                update_z = true;
                update_c = true;
                c_value = result > a;
            }
            0x10 => {
                // ADDI rd, imm
                result = a.wrapping_add(imm_val);
                write_reg = Some(rd);
                update_z = true;
                update_c = true;
                c_value = a > result;
            }
            0x11 => {
                // SUBI rd, imm
                result = a.wrapping_sub(imm_val);
                write_reg = Some(rd);
                update_z = true;
                update_c = true;
                c_value = result > a;
            }
            0x12 => {
                // ANDI rd, imm
                result = a & imm_val;
                write_reg = Some(rd);
                update_z = true;
            }
            0x13 => {
                // ORI rd, imm
                result = a | imm_val;
                write_reg = Some(rd);
                update_z = true;
            }
            0x14 => {
                // XORI rd, imm
                result = a ^ imm_val;
                write_reg = Some(rd);
                update_z = true;
            }
            0x15 => {
                // RET
                next_pc = self.state.lr & self.pc_mask;
            }
            0x16 => {
                // LD rd, [rs] / LD rd, [rs+offset]
                // Sprint 362: address masked in mem_read_inner (mem_addr_mask).
                let addr = if d.has_offset {
                    b.wrapping_add(d.offset as u64) as usize
                } else {
                    b as usize
                };
                result = self.mem_read_inner::<MMIO>(addr);
                write_reg = Some(rd);
                update_z = true;
            }
            0x17 => {
                // ST [rs], rd / ST [rs+offset], rd
                let addr = if d.has_offset {
                    b.wrapping_add(d.offset as u64) as usize
                } else {
                    b as usize
                };
                self.mem_write_inner::<MMIO>(addr, a);
            }
            0x18 => {
                // LDB rd, [imm8] / LDB rd, [rs+offset]
                let addr = if d.has_offset {
                    b.wrapping_add(d.offset as u64) as usize
                } else {
                    imm8 as usize
                };
                result = self.mem_read_inner::<MMIO>(addr);
                write_reg = Some(rd);
                update_z = true;
            }
            0x19 => {
                // STB [imm8], rd / STB [rs+offset], rd
                let addr = if d.has_offset {
                    b.wrapping_add(d.offset as u64) as usize
                } else {
                    imm8 as usize
                };
                self.mem_write_inner::<MMIO>(addr, a);
            }
            0x1A => {
                // JMP imm / JMPR rd (Sprint 363: register-indirect far-jump).
                // Sprint 371 (Gate B.3.1): the immediate target is `imm_val` (16-bit
                // wide-expanded). For non-wide JMP, `imm_val == imm8` → unchanged.
                next_pc = if d.reg_indirect {
                    (a as u32) & self.pc_mask
                } else {
                    (imm_val as u32) & self.pc_mask
                };
            }
            0x1B => {
                // JZ imm / JZ.W (Sprint 374): wide-expanded `imm_val` target.
                if self.state.flag_z {
                    next_pc = (imm_val as u32) & self.pc_mask;
                }
            }
            0x1C => {
                // JNZ imm / JNZ.W (Sprint 374).
                if !self.state.flag_z {
                    next_pc = (imm_val as u32) & self.pc_mask;
                }
            }
            0x1D => {
                // JC imm / JC.W (Sprint 374).
                if self.state.flag_c {
                    next_pc = (imm_val as u32) & self.pc_mask;
                }
            }
            0x1E => {
                // JNC imm / JNC.W (Sprint 374).
                if !self.state.flag_c {
                    next_pc = (imm_val as u32) & self.pc_mask;
                }
            }
            0x1F => {
                // CALL imm / CALLR rd (Sprint 363: register-indirect call).
                // Sprint 371 (Gate B.3.1): the immediate target is `imm_val` (16-bit
                // wide-expanded). For non-wide CALL, `imm_val == imm8` → unchanged.
                self.state.lr = next_pc & self.pc_mask;
                next_pc = if d.reg_indirect {
                    (a as u32) & self.pc_mask
                } else {
                    (imm_val as u32) & self.pc_mask
                };
            }
            _ => {}
        }

        if let Some(reg) = write_reg {
            self.state.regs[reg] = result;
        }
        if update_z {
            self.state.flag_z = result == 0;
        }
        if update_c {
            self.state.flag_c = c_value;
        }

        self.state.pc = next_pc & self.pc_mask;
        Some(self.program[pc as usize])
    }
}

#[allow(dead_code)]
fn opcode_uses_rs(opcode: u8) -> bool {
    matches!(
        opcode,
        0x02 | // MOV
        0x04 | 0x05 | 0x06 | 0x07 | 0x08 | // ADD/SUB/AND/OR/XOR
        0x0F | // CMP
        0x16 | // LD
        0x17 // ST
    )
}

#[derive(Debug, Clone, Copy)]
pub struct V2DiffConfig {
    pub max_ticks: u64,
}

impl Default for V2DiffConfig {
    fn default() -> Self {
        Self { max_ticks: 512 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V2DiffStats {
    pub ticks: u64,
    pub retired: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2DiffMismatch {
    pub tick: u64,
    pub retired: u64,
    pub reason: String,
}

impl std::fmt::Display for V2DiffMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tick={} retired={} reason={}",
            self.tick, self.retired, self.reason
        )
    }
}

impl std::error::Error for V2DiffMismatch {}

pub fn run_v2_differential(
    program: &[u32],
    config: V2DiffConfig,
) -> Result<V2DiffStats, V2DiffMismatch> {
    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(program)
        .with_rom_size(128)
        .with_ram_size(128)
        .build(&mut sim);
    run_v2_differential_with(program, config, &cpu, &mut sim)
}

pub fn run_v2_differential_with(
    program: &[u32],
    config: V2DiffConfig,
    cpu: &TileCpuV2,
    sim: &mut Simulation,
) -> Result<V2DiffStats, V2DiffMismatch> {
    let mut iss = V2Iss::from_program(program);
    run_v2_differential_core(&mut iss, config, cpu, sim)
}

/// Sprint 158: Differential testing with MMIO on both sides.
/// Builds hardware CPU with V2MmioCombinedDevice and ISS with MMIO enabled.
pub fn run_v2_differential_mmio(
    program: &[u32],
    config: V2DiffConfig,
) -> Result<V2DiffStats, V2DiffMismatch> {
    use crate::tile_cpu::v2_mmio::V2MmioHandle;
    use crate::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
    use std::rc::Rc;

    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let combined = Rc::new(V2MmioCombinedDevice::new(0x156_CAFE));
    let mmio_handle = V2MmioHandle::from_rc(combined);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(program)
        .with_rom_size(128)
        .with_ram_size(128)
        .with_mmio(mmio_handle)
        .build(&mut sim);

    let mut iss = V2Iss::from_program_with_mmio(program);
    run_v2_differential_core(&mut iss, config, &cpu, &mut sim)
}

/// Sprint 159: Differential testing with dataset device on both sides.
/// Pre-loads RAM cells (e.g. weight masks) before execution.
pub fn run_v2_differential_dataset(
    program: &[u32],
    config: V2DiffConfig,
    samples: Vec<crate::tile_cpu::v2_mmio_devices::DatasetSample>,
    ram_preload: &[(usize, u64)],
) -> Result<V2DiffStats, V2DiffMismatch> {
    use crate::tile_cpu::v2_mmio::V2MmioHandle;
    use crate::tile_cpu::v2_mmio_devices::{V2MmioCombinedDevice, V2MmioDatasetDevice};
    use std::rc::Rc;

    let dataset = V2MmioDatasetDevice::from_samples(samples.clone());
    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let combined = Rc::new(V2MmioCombinedDevice::with_dataset(0x156_CAFE, dataset));
    let mmio_handle = V2MmioHandle::from_rc(combined);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(program)
        .with_rom_size(128)
        .with_ram_size(128)
        .with_mmio(mmio_handle)
        .build(&mut sim);

    // Pre-load RAM in both hardware and ISS
    for &(addr, val) in ram_preload {
        cpu.write_ram(&mut sim, addr, val);
    }

    let iss_samples: Vec<(u64, u64)> = samples.iter().map(|s| (s.features, s.label)).collect();
    let mut iss = V2Iss::from_program_with_dataset(program, iss_samples);
    for &(addr, val) in ram_preload {
        iss.state_mut().ram[addr & 0x7F] = val;
    }

    run_v2_differential_core(&mut iss, config, &cpu, &mut sim)
}

fn run_v2_differential_core(
    iss: &mut V2Iss,
    config: V2DiffConfig,
    cpu: &TileCpuV2,
    sim: &mut Simulation,
) -> Result<V2DiffStats, V2DiffMismatch> {
    let mut retired = 0u64;

    for tick in 1..=config.max_ticks {
        let _ = cpu.step(sim);

        if cpu.last_stage_x_valid() {
            retired += 1;
            if iss.step().is_none() {
                return Err(V2DiffMismatch {
                    tick,
                    retired,
                    reason: "CPU retired an instruction after ISS halted".to_string(),
                });
            }

            let cpu_state = snapshot_cpu_state(cpu, sim);
            let iss_state = iss.state().clone();
            if let Some(reason) = diff_states(&cpu_state, &iss_state) {
                return Err(V2DiffMismatch {
                    tick,
                    retired,
                    reason,
                });
            }
        }

        if cpu.is_halted() && iss.is_halted() {
            return Ok(V2DiffStats {
                ticks: tick,
                retired,
            });
        }
    }

    Err(V2DiffMismatch {
        tick: config.max_ticks,
        retired,
        reason: format!(
            "timeout: cpu_halted={} iss_halted={} after {} ticks",
            cpu.is_halted(),
            iss.is_halted(),
            config.max_ticks
        ),
    })
}

fn snapshot_cpu_state(cpu: &TileCpuV2, sim: &Simulation) -> V2IssState {
    let regs = std::array::from_fn(|i| cpu.read_reg(sim, i));
    let ram = std::array::from_fn(|i| cpu.read_ram(sim, i));
    // Sprint 362: snapshot extended main memory for differential comparison.
    let main_mem = (0..cpu.main_mem_len())
        .map(|i| cpu.read_ram(sim, 128 + i))
        .collect();
    V2IssState {
        // Sprint 369 (Gate B.2): mask by the CPU's physical PC width (0xFF in the
        // extended address space) so PC>=128 is not aliased down to 7 bits.
        pc: cpu.read_pc(sim) & cpu.pc_phys_mask(),
        lr: cpu.read_lr() & cpu.pc_phys_mask(),
        flag_z: cpu.read_flag_z(sim),
        flag_c: cpu.read_flag_c(sim),
        halted: cpu.is_halted(),
        regs,
        ram,
        main_mem,
        mem_addr_mask: cpu.mem_addr_mask(),
        // Hardware math state not readable from CPU snapshot — zero-fill
        // (diff_states skips these fields)
        iss_math_a: 0,
        iss_math_b: 0,
        iss_math_status: 0,
        iss_math_result: 0,
    }
}

fn diff_states(cpu: &V2IssState, iss: &V2IssState) -> Option<String> {
    if cpu.pc != iss.pc {
        return Some(format!("pc mismatch: cpu={} iss={}", cpu.pc, iss.pc));
    }
    if cpu.lr != iss.lr {
        return Some(format!("lr mismatch: cpu={} iss={}", cpu.lr, iss.lr));
    }
    if cpu.flag_z != iss.flag_z {
        return Some(format!(
            "flag_z mismatch: cpu={} iss={}",
            cpu.flag_z, iss.flag_z
        ));
    }
    if cpu.flag_c != iss.flag_c {
        return Some(format!(
            "flag_c mismatch: cpu={} iss={}",
            cpu.flag_c, iss.flag_c
        ));
    }
    if cpu.halted != iss.halted {
        return Some(format!(
            "halted mismatch: cpu={} iss={}",
            cpu.halted, iss.halted
        ));
    }

    for i in 0..16usize {
        if cpu.regs[i] != iss.regs[i] {
            return Some(format!(
                "reg mismatch r{}: cpu=0x{:016X} iss=0x{:016X}",
                i, cpu.regs[i], iss.regs[i]
            ));
        }
    }
    for i in 0..128usize {
        if cpu.ram[i] != iss.ram[i] {
            return Some(format!(
                "ram mismatch [{}]: cpu=0x{:016X} iss=0x{:016X}",
                i, cpu.ram[i], iss.ram[i]
            ));
        }
    }
    // Sprint 362: extended main-memory comparison.
    let n = cpu.main_mem.len().max(iss.main_mem.len());
    for i in 0..n {
        let c = cpu.main_mem.get(i).copied().unwrap_or(0);
        let s = iss.main_mem.get(i).copied().unwrap_or(0);
        if c != s {
            return Some(format!(
                "main_mem mismatch [{}] (addr {}): cpu=0x{:016X} iss=0x{:016X}",
                i,
                128 + i,
                c,
                s
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_cpu::v2_components::{
        v2_add, v2_addi, v2_and, v2_andi, v2_call, v2_cmp, v2_dec, v2_halt, v2_inc, v2_jc, v2_jmp,
        v2_jnc, v2_jnz, v2_jz, v2_ld, v2_ldb, v2_ldi, v2_mov, v2_neg, v2_nop, v2_not, v2_or,
        v2_ori, v2_ret, v2_shl, v2_shr, v2_st, v2_stb, v2_sub, v2_subi, v2_with_reg_ext, v2_xor,
        v2_xori,
    };

    #[derive(Clone)]
    struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            let state = if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            };
            Self { state }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        fn next_u8(&mut self) -> u8 {
            self.next_u64() as u8
        }

        fn gen_bool(&mut self) -> bool {
            (self.next_u64() & 1) != 0
        }

        fn gen_range(&mut self, upper: usize) -> usize {
            if upper == 0 {
                0
            } else {
                (self.next_u64() as usize) % upper
            }
        }
    }

    fn gen_forward_target(rng: &mut XorShift64, i: usize, len: usize) -> u8 {
        if i + 1 >= len {
            i.min(63) as u8
        } else {
            (i + 1 + rng.gen_range(len - (i + 1))).min(63) as u8
        }
    }

    fn gen_program(seed: u64) -> Vec<u32> {
        let mut rng = XorShift64::new(seed);
        let len = 8 + rng.gen_range(24);
        let mut prog = vec![v2_nop(); len];

        for (i, slot) in prog.iter_mut().enumerate().take(len.saturating_sub(1)) {
            let rd = (rng.next_u8() & 0x07) as u8;
            let rs = (rng.next_u8() & 0x07) as u8;
            let rd_hi = rng.gen_bool();
            let rs_hi = rng.gen_bool();
            let imm = rng.next_u8();

            let op = rng.gen_range(28);
            let instr = match op {
                0 => v2_nop(),
                1 => v2_with_reg_ext(v2_mov(rd, rs), rd_hi, rs_hi),
                2 => v2_with_reg_ext(v2_ldi(rd, imm), rd_hi, false),
                3 => v2_with_reg_ext(v2_add(rd, rs), rd_hi, rs_hi),
                4 => v2_with_reg_ext(v2_sub(rd, rs), rd_hi, rs_hi),
                5 => v2_with_reg_ext(v2_and(rd, rs), rd_hi, rs_hi),
                6 => v2_with_reg_ext(v2_or(rd, rs), rd_hi, rs_hi),
                7 => v2_with_reg_ext(v2_xor(rd, rs), rd_hi, rs_hi),
                8 => v2_with_reg_ext(v2_not(rd), rd_hi, false),
                9 => v2_with_reg_ext(v2_neg(rd), rd_hi, rd_hi), // NEG uses Tree B for rd operand
                10 => v2_with_reg_ext(v2_inc(rd), rd_hi, false),
                11 => v2_with_reg_ext(v2_dec(rd), rd_hi, false),
                12 => {
                    let amount = 1 + (rng.next_u8() % 7);
                    v2_with_reg_ext(v2_shl(rd, amount), rd_hi, false)
                }
                13 => {
                    let amount = 1 + (rng.next_u8() % 7);
                    v2_with_reg_ext(v2_shr(rd, amount), rd_hi, false)
                }
                14 => v2_with_reg_ext(v2_cmp(rd, rs), rd_hi, rs_hi),
                15 => v2_with_reg_ext(v2_addi(rd, imm), rd_hi, false),
                16 => v2_with_reg_ext(v2_subi(rd, imm), rd_hi, false),
                17 => v2_with_reg_ext(v2_andi(rd, imm), rd_hi, false),
                18 => v2_with_reg_ext(v2_ori(rd, imm), rd_hi, false),
                19 => v2_with_reg_ext(v2_xori(rd, imm), rd_hi, false),
                20 => v2_with_reg_ext(v2_ld(rd, rs), rd_hi, rs_hi),
                21 => v2_with_reg_ext(v2_st(rs, rd), rd_hi, rs_hi),
                22 => v2_with_reg_ext(v2_ldb(rd, imm), rd_hi, false),
                23 => v2_with_reg_ext(v2_stb(rd, imm), rd_hi, false),
                24 => v2_jz(gen_forward_target(&mut rng, i, len)),
                25 => v2_jnz(gen_forward_target(&mut rng, i, len)),
                26 => v2_jc(gen_forward_target(&mut rng, i, len)),
                27 => v2_jnc(gen_forward_target(&mut rng, i, len)),
                _ => v2_nop(),
            };
            *slot = instr;
        }

        prog[len - 1] = v2_halt();
        prog
    }

    #[test]
    fn test_v2_diff_simple_program() {
        let program = [
            v2_ldi(0, 10),
            v2_ldi(1, 3),
            v2_add(0, 1),
            v2_subi(0, 2),
            v2_stb(0, 7),
            v2_ldb(2, 7),
            v2_halt(),
        ];
        let res = run_v2_differential(&program, V2DiffConfig { max_ticks: 128 });
        assert!(res.is_ok(), "simple differential failed: {:?}", res.err());
    }

    /// Sprint 362 (Gate A): the extended main-memory address space (addresses
    /// 128..) must work end-to-end — a program processes a 200-element array
    /// (impossible in the 128-cell scratchpad), and the physical executor and
    /// ISS agree bit-for-bit (including main memory) at every cycle.
    #[test]
    fn test_v2_main_memory_array_sum_differential() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::{V2Builder, assemble_v2};

        const N: u64 = 200; // > 128 scratchpad cells; addresses reach 327 (> u8)
        // Fill main memory mem[128+i] = i for i in 0..N, then sum it.
        let src = r#"
            LDI R1, 128        ; base address of main memory
            LDI R2, 0          ; i
            LDI R3, 200        ; N
            LDI R5, 0          ; sum
        fill:
            CMP R2, R3
            JZ sum_start
            MOV R4, R1
            ADD R4, R2         ; addr = 128 + i
            ST [R4], R2        ; mem[128+i] = i
            INC R2
            JMP fill
        sum_start:
            LDI R2, 0
        sum:
            CMP R2, R3
            JZ done
            MOV R4, R1
            ADD R4, R2
            LD R6, [R4]        ; R6 = mem[128+i]
            ADD R5, R6         ; sum += R6
            INC R2
            JMP sum
        done:
            HALT
        "#;
        let program = assemble_v2(src).expect("assemble main-memory program");

        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_main_memory(256) // covers addresses 128..383
            .build(&mut sim);

        let mut iss = V2Iss::from_program(&program);
        iss.configure_main_memory(256);

        // Lockstep differential — diffs registers, RAM, AND main memory each cycle.
        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 20_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "main-memory differential mismatch: {:?}",
            res.err()
        );

        // Ground truth: sum(0..N) = N*(N-1)/2.
        let expected_sum = N * (N - 1) / 2;
        assert_eq!(
            cpu.read_reg(&sim, 5),
            expected_sum,
            "R5 should hold the array sum"
        );
        // Spot-check stored contents across the scratchpad/main-memory boundary
        // and past the 255 (u8) line that the old address mask would have aliased.
        for k in [0usize, 1, 71, 128, 199] {
            assert_eq!(
                cpu.read_ram(&sim, 128 + k),
                k as u64,
                "mem[128+{k}] should equal {k}"
            );
        }
    }

    /// Sprint 363 (Gate B foundation): the extended instruction space (256
    /// instructions) and the register-indirect far-jump (JMPR) must work in the
    /// ISS. We place work in the UPPER half (instruction 200, only reachable with
    /// extended PC) and reach it by computing the target in a register and JMPR-ing
    /// to it — dynamic dispatch that a static imm8 jump cannot express.
    #[test]
    fn test_v2_iss_gate_b_extended_capacity_and_jmpr() {
        use crate::tile_cpu::v2_components::{v2_halt, v2_jmpr, v2_ldi, v2_nop};

        let mut prog = vec![v2_nop(); 210];
        prog[0] = v2_ldi(1, 200); // R1 = 200 (computed target address)
        prog[1] = v2_jmpr(1); // PC = R1 = 200 (register-indirect far-jump)
        prog[200] = v2_ldi(0, 42); // landing pad in the upper half
        prog[201] = v2_halt();

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc(); // 0x7F -> 0xFF: 256 instructions reachable

        for _ in 0..1000 {
            if iss.state().halted {
                break;
            }
            iss.step();
        }
        assert!(iss.state().halted, "program should halt");
        assert_eq!(
            iss.state().regs[0],
            42,
            "R0 must be set by the upper-half instruction at index 200"
        );
        assert_eq!(
            iss.state().pc,
            201,
            "PC should rest at the HALT (instr 201)"
        );
    }

    /// Sprint 363: register-indirect CALL (CALLR) + RET across the 128 boundary —
    /// a function pointer to a subroutine in the upper half, returning correctly.
    #[test]
    fn test_v2_iss_gate_b_callr_ret() {
        use crate::tile_cpu::v2_components::{v2_add, v2_callr, v2_halt, v2_ldi, v2_nop, v2_ret};

        let mut prog = vec![v2_nop(); 190];
        prog[0] = v2_ldi(0, 10); // R0 = 10
        prog[1] = v2_ldi(1, 5); // R1 = 5
        prog[2] = v2_ldi(3, 180); // R3 = 180 (function pointer into upper half)
        prog[3] = v2_callr(3); // LR = 4; PC = R3 = 180
        prog[4] = v2_halt(); // return lands here
        prog[180] = v2_add(0, 1); // subroutine body: R0 = R0 + R1 = 15
        prog[181] = v2_ret(); // PC = LR = 4

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc();

        for _ in 0..1000 {
            if iss.state().halted {
                break;
            }
            iss.step();
        }
        assert!(iss.state().halted, "program should halt");
        assert_eq!(iss.state().regs[0], 15, "subroutine computed R0 + R1");
        assert_eq!(iss.state().pc, 4, "RET returned to instruction 4 (HALT)");
    }

    // ================================================================
    // Sprint 364 (Gate C) — ABI: MFLR / MTLR + nested calls + recursion
    // ================================================================

    /// Gate C primitive: MTLR writes LR from a register, MFLR reads LR
    /// into a register. Verifies the LR↔reg bridge in isolation (no CALL).
    #[test]
    fn test_v2_iss_gate_c_mflr_mtlr_roundtrip() {
        use crate::tile_cpu::v2_components::{v2_halt, v2_ldi, v2_mflr, v2_mtlr, v2_nop};

        let mut prog = vec![v2_nop(); 16];
        prog[0] = v2_ldi(0, 42); // R0 = 42
        prog[1] = v2_mtlr(0); // lr = R0 = 42
        prog[2] = v2_mflr(1); // R1 = lr = 42
        prog[3] = v2_halt();

        let mut iss = V2Iss::from_program(&prog);
        for _ in 0..100 {
            if iss.state().halted {
                break;
            }
            iss.step();
        }
        assert!(iss.state().halted);
        assert_eq!(iss.state().lr, 42, "MTLR wrote LR");
        assert_eq!(iss.state().regs[1], 42, "MFLR read LR back");
        assert_eq!(iss.state().regs[0], 42, "source untouched");
    }

    /// Gate C: nested call. The outer function must save LR via MFLR
    /// before issuing its own CALL (which clobbers LR), and restore via
    /// MTLR before its own RET. Without the save, the outer RET would
    /// re-enter the inner subroutine and loop forever.
    #[test]
    fn test_v2_iss_gate_c_nested_call_saves_lr() {
        use crate::tile_cpu::v2_components::{
            v2_call, v2_halt, v2_ldi, v2_mflr, v2_mtlr, v2_nop, v2_ret, v2_with_reg_ext,
        };

        let mut prog = vec![v2_nop(); 64];
        // main:
        prog[0] = v2_call(16); // LR = 1, PC = 16 (outer)
        prog[1] = v2_halt();
        // outer (16):
        prog[16] = v2_with_reg_ext(v2_mflr(0), true, false); // R8 = LR = 1  (callee-saved reg)
        prog[17] = v2_call(32); // LR = 18, PC = 32 (inner)
        prog[18] = v2_with_reg_ext(v2_mtlr(0), false, true); // lr = R8 = 1
        prog[19] = v2_ret(); // PC = 1 (HALT in main)
        // inner (32): leaf function — sets a return value and returns.
        prog[32] = v2_ldi(0, 99); // R0 = 99 (return value)
        prog[33] = v2_ret(); // PC = 18 (outer epilogue)

        let mut iss = V2Iss::from_program(&prog);
        for _ in 0..1000 {
            if iss.state().halted {
                break;
            }
            iss.step();
        }
        assert!(iss.state().halted, "should halt at main+1");
        assert_eq!(iss.state().pc, 1, "outer RET returned to main");
        assert_eq!(iss.state().regs[0], 99, "inner's return value visible");
        assert_eq!(iss.state().regs[8], 1, "R8 still holds the saved LR");
    }

    /// Gate C capability bar: recursive factorial(5) = 120. Five stack
    /// frames deep, with LR saved/restored to the stack between calls.
    /// Stack lives in main_mem (Gate A) — the first multi-frame program.
    #[test]
    fn test_v2_iss_gate_c_factorial_recursion() {
        use crate::tile_cpu::v2_components::{
            v2_call, v2_cmp, v2_dec, v2_halt, v2_inc, v2_jnz, v2_ld, v2_ldi, v2_mflr, v2_mov,
            v2_mtlr, v2_mul, v2_nop, v2_ret, v2_st, v2_with_reg_ext,
        };

        // ABI for this routine:
        //   R0 = input n / output n!
        //   R8 = saved LR        (callee-saved)
        //   R9 = saved input n   (callee-saved across the recursive CALL)
        //   R15 (SP) = stack pointer (empty-descending into main_mem)
        //
        // factorial(n):
        //   if n == 0: return 1
        //   tmp = n
        //   r0 = factorial(n-1)
        //   return tmp * r0
        //
        // Layout: main at 0..6, factorial at 16..40.

        let mut prog = vec![v2_nop(); 64];
        // main:
        prog[0] = v2_with_reg_ext(v2_ldi(7, 200), true, false); // SP (R15) = 200
        prog[1] = v2_ldi(0, 5); // R0 = 5  (call factorial(5))
        prog[2] = v2_call(16); // -> factorial
        prog[3] = v2_halt();

        // factorial (16):
        //   CMP R0, R0  ? we need to test R0 == 0. Use a zero-compare via SUB-ish.
        // Simpler: use CMP R0, R1 where R1 = 0.
        prog[16] = v2_ldi(1, 0); // R1 = 0
        prog[17] = v2_cmp(0, 1); // Z=1 iff R0 == 0
        prog[18] = v2_jnz(22); // if R0 != 0, recurse
        prog[19] = v2_ldi(0, 1); // base case: R0 = 1
        prog[20] = v2_ret(); // return 1
        prog[21] = v2_nop();
        // recursive case (22):
        prog[22] = v2_with_reg_ext(v2_mflr(0), true, false); // R8 = LR (save return addr)
        prog[23] = v2_with_reg_ext(v2_mov(1, 0), true, false); // R9 = R0 (save n)
        // push R8 (LR) onto stack: ST [SP], R8; DEC SP
        prog[24] = v2_with_reg_ext(v2_st(7, 0), true, true); // ST [R15], R8
        prog[25] = v2_with_reg_ext(v2_dec(7), true, false); // SP--
        // push R9 (n): ST [SP], R9; DEC SP
        prog[26] = v2_with_reg_ext(v2_st(7, 1), true, true); // ST [R15], R9
        prog[27] = v2_with_reg_ext(v2_dec(7), true, false); // SP--
        // R0 = n - 1, recursive call
        prog[28] = v2_dec(0); // R0--
        prog[29] = v2_call(16); // R0 = factorial(R0)
        // pop R9 (saved n): INC SP; LD R9, [SP]
        prog[30] = v2_with_reg_ext(v2_inc(7), true, false); // SP++
        prog[31] = v2_with_reg_ext(v2_ld(1, 7), true, true);
        // pop R8 (saved LR): INC SP; LD R8, [SP]
        prog[32] = v2_with_reg_ext(v2_inc(7), true, false);
        prog[33] = v2_with_reg_ext(v2_ld(0, 7), true, true);
        // R0 = R9 * R0  (n * factorial(n-1))
        prog[34] = v2_with_reg_ext(v2_mul(0, 1), false, true); // R0 = R0 * R9
        // restore LR and return
        prog[35] = v2_with_reg_ext(v2_mtlr(0), false, true); // lr = R8
        prog[36] = v2_ret();

        let mut iss = V2Iss::from_program(&prog);
        iss.configure_main_memory(256); // stack lives in main_mem (Gate A)

        for _ in 0..20_000 {
            if iss.state().halted {
                break;
            }
            iss.step();
        }
        assert!(iss.state().halted, "factorial should halt");
        assert_eq!(
            iss.state().regs[0],
            120,
            "factorial(5) = 120 via 5-deep recursion + stack-saved LR"
        );
        // SP must be back to its starting value (balanced push/pop)
        assert_eq!(iss.state().regs[15], 200, "stack balanced");
    }

    /// Sprint 364 (Gate C, physical): MFLR is wired into the physical executor.
    /// LR is sourced from its actual Register8 tile (not the software mirror),
    /// then routed to Rd through the standard wb_data injection path (the same
    /// pattern MUL uses). Lockstep ISS↔physical differential proves both sides
    /// agree on Rd and LR every cycle — i.e., MFLR commits via real physical
    /// tiles, not via software-side bookkeeping.
    #[test]
    fn test_v2_iss_gate_c_mflr_physical_differential() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::v2_components::{
            v2_call, v2_halt, v2_ldi, v2_mflr, v2_nop, v2_ret, v2_with_reg_ext,
        };

        // Two MFLR variants exercised:
        //   - R3 (low bank, R0-R7): wb_data_mux Const-swap injection path.
        //   - R8 (high bank, R8-R15): high_reg_wb_data_const injection path.
        let mut prog = vec![v2_nop(); 32];
        prog[0] = v2_ldi(0, 0); // R0 = 0 (clear)
        prog[1] = v2_call(8); // LR = 2, PC = 8
        prog[2] = v2_call(12); // LR = 3, PC = 12
        prog[3] = v2_halt();
        // Subroutine A: low-bank MFLR target.
        prog[8] = v2_mflr(3); // R3 = LR (= 2)
        prog[9] = v2_ret(); // PC = LR = 2
        // Subroutine B: high-bank MFLR target.
        prog[12] = v2_with_reg_ext(v2_mflr(0), true, false); // R8 = LR (= 3)
        prog[13] = v2_ret(); // PC = LR = 3

        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .build(&mut sim);

        let mut iss = V2Iss::from_program(&prog);

        // Lockstep: physical and ISS must agree on registers + LR + flags at
        // every retire. If MFLR's wb_data injection lands the wrong value in
        // the physical Rd, this fails on the cycle MFLR retires.
        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 5_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "physical MFLR differential mismatch: {:?}",
            res.err()
        );

        // Sanity: physical R3 == 2 (LR captured by first CALL), R8 == 3 (LR captured
        // by second CALL). Reads are from the actual physical Register8 tiles.
        assert_eq!(cpu.read_reg(&sim, 3), 2, "R3 holds LR from CALL #1");
        assert_eq!(cpu.read_reg(&sim, 8), 3, "R8 holds LR from CALL #2");
        assert!(cpu.is_halted(), "program should halt at prog[3]");
    }

    /// Sprint 369 (Gate B.2): PHYSICAL fetch + execution of instructions at PC>=128.
    /// The capability bar — a program physically executing past instruction 127.
    ///
    /// Built in the authoritative config (`max_authority`, 16-layer grid) + the new
    /// `with_extended_pc()`. A function body lives at instruction 200 (only reachable
    /// with the 8-bit PC), reached via CALLR with a function pointer in R3. The body
    /// runs two physically-decoded instructions (ADD then INC, exercising the 200->201
    /// fall-through that the 7-bit mask would have wrapped to 72), then RETs across the
    /// 128 boundary. Lockstep physical<->ISS: every retire must match bit-for-bit.
    #[test]
    fn test_v2_iss_gate_b2_far_call_physical_differential() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_components::{
            v2_add, v2_callr, v2_halt, v2_inc, v2_ldi, v2_nop, v2_ret,
        };

        let mut prog = vec![v2_nop(); 210];
        prog[0] = v2_ldi(0, 10); // R0 = 10
        prog[1] = v2_ldi(1, 5); // R1 = 5
        prog[2] = v2_ldi(3, 200); // R3 = 200 (function pointer into the upper half)
        prog[3] = v2_callr(3); // LR = 4; PC = R3 = 200 (register-indirect far-call)
        prog[4] = v2_halt(); // return lands here
        // Subroutine body at PC>=128 — physically fetched from program_ext and
        // decoded/executed through real tiles (IR injected via the Super-Mux Const path).
        prog[200] = v2_add(0, 1); // R0 = R0 + R1 = 15
        prog[201] = v2_inc(0); // R0 = 16  (proves 200->201 fall-through, no 7-bit wrap)
        prog[202] = v2_ret(); // PC = LR = 4

        // Authoritative config: same grid + synth authority as the golden parity test.
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 5_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "Gate B.2 far-call physical differential mismatch: {:?}",
            res.err()
        );

        // The upper-half subroutine ran physically: R0 = (10 + 5) + 1 = 16.
        assert_eq!(
            cpu.read_reg(&sim, 0),
            16,
            "R0 must be computed by instructions physically executed at PC 200/201"
        );
        assert!(cpu.is_halted(), "program should halt at prog[4]");
    }

    /// Sprint 370 (Gate B.3): PHYSICAL-WIDE PC register — execution past the 256
    /// instruction line with the PC physically held and incremented in a Register64.
    ///
    /// Built with `with_wide_pc()` (Register64 PC/LR, 16-bit mask) + `max_authority`.
    /// A function body lives at instruction 300 (> 256, only reachable with the wide
    /// PC), reached via CALLR. The body runs three instructions (ADD, INC, INC),
    /// exercising the physical 300→301→302 fall-through — each increment carried by
    /// the physical pc_plus_one adder + Register64 capture, NOT software. We assert
    /// `pc_override_mismatches() == 0`: the physical wide PC was authoritative for
    /// every fall-through cycle (zero software fallbacks).
    #[test]
    fn test_v2_iss_gate_b3_wide_pc_physical_differential() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_components::{
            v2_add, v2_addi, v2_callr, v2_halt, v2_inc, v2_ldi, v2_nop, v2_ret,
        };

        let mut prog = vec![v2_nop(); 310];
        prog[0] = v2_ldi(0, 10); // R0 = 10
        prog[1] = v2_ldi(1, 5); // R1 = 5
        prog[2] = v2_ldi(3, 0); // R3 = 0
        prog[3] = v2_addi(3, 200); // R3 = 200
        prog[4] = v2_addi(3, 100); // R3 = 300 (function pointer, > 256)
        prog[5] = v2_callr(3); // LR = 6; PC = R3 = 300
        prog[6] = v2_halt(); // return lands here
        // Subroutine body past the 256 line — physically fetched + sequenced.
        prog[300] = v2_add(0, 1); // R0 = R0 + R1 = 15
        prog[301] = v2_inc(0); // R0 = 16  (physical fall-through 300->301)
        prog[302] = v2_inc(0); // R0 = 17  (physical fall-through 301->302)
        prog[303] = v2_ret(); // PC = LR = 6

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_wide_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_wide_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 5_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "Gate B.3 wide-PC physical differential mismatch: {:?}",
            res.err()
        );

        // The body at instruction 300+ ran: R0 = (10 + 5) + 1 + 1 = 17.
        assert_eq!(
            cpu.read_reg(&sim, 0),
            17,
            "R0 must be computed by instructions physically executed at PC 300/301/302"
        );
        assert!(cpu.is_halted(), "program should halt at prog[6]");

        // Physical authority proof: every fall-through cycle trusted the physical
        // Register64 PC — zero software-fallback overrides.
        assert_eq!(
            cpu.pc_override_mismatches(),
            0,
            "wide PC fall-through must be physically authoritative (no software fallback)"
        );
        assert!(
            cpu.pc_override_checks() > 0,
            "the wide-PC physical fall-through path must have been exercised"
        );
    }

    /// Sprint 371 (Gate B.3.1): WIDE-IMMEDIATE CALL — `CALL.W 300` reaches a
    /// subroutine past the 256 line in a single instruction, with no register load
    /// (contrast the B.3 test, which needed three instructions to build the target
    /// in R3 before `CALLR`). The far target rides the wide immediate; LR is captured
    /// physically as for any CALL. After the call, the body's 300→301→302 fall-through
    /// is physically authoritative (`pc_override_mismatches() == 0`).
    #[test]
    fn test_v2_iss_gate_b31_call_w_physical_differential() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_components::{
            v2_add, v2_call_w, v2_halt, v2_inc, v2_ldi, v2_nop, v2_ret,
        };

        let mut prog = vec![v2_nop(); 310];
        prog[0] = v2_ldi(0, 10); // R0 = 10
        prog[1] = v2_ldi(1, 5); // R1 = 5
        prog[2] = v2_call_w(300); // LR = 3; PC = 300 — one instruction, no reg load
        prog[3] = v2_halt(); // return lands here
        // Subroutine body past the 256 line — physically fetched + sequenced.
        prog[300] = v2_add(0, 1); // R0 = R0 + R1 = 15
        prog[301] = v2_inc(0); // R0 = 16  (physical fall-through 300->301)
        prog[302] = v2_inc(0); // R0 = 17  (physical fall-through 301->302)
        prog[303] = v2_ret(); // PC = LR = 3

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_wide_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_wide_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 5_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "Gate B.3.1 CALL.W physical differential mismatch: {:?}",
            res.err()
        );

        assert_eq!(
            cpu.read_reg(&sim, 0),
            17,
            "R0 must be computed by the subroutine reached via CALL.W 300"
        );
        assert!(cpu.is_halted(), "program should halt at prog[3]");
        assert_eq!(
            cpu.pc_override_mismatches(),
            0,
            "wide PC fall-through must be physically authoritative (no software fallback)"
        );
        assert!(
            cpu.pc_override_checks() > 0,
            "the wide-PC physical fall-through path must have been exercised"
        );
    }

    /// Sprint 371 (Gate B.3.1): WIDE-IMMEDIATE JUMP — `JMP.W 300` transfers control
    /// directly to instruction 300 (no register, no condition), the body fall-throughs
    /// physically (300→301), then `JMP.W 2` transfers back to the low range to halt.
    /// Exercises wide-immediate jumps in both directions (low→far, far→low).
    #[test]
    fn test_v2_iss_gate_b31_jmp_w_physical_differential() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_components::{v2_halt, v2_inc, v2_jmp_w, v2_ldi, v2_nop};

        let mut prog = vec![v2_nop(); 310];
        prog[0] = v2_ldi(0, 40); // R0 = 40
        prog[1] = v2_jmp_w(300); // PC = 300 — direct far jump, single instruction
        prog[2] = v2_halt(); // return target (far → low)
        prog[300] = v2_inc(0); // R0 = 41
        prog[301] = v2_inc(0); // R0 = 42  (physical fall-through 300->301)
        prog[302] = v2_jmp_w(2); // jump back to the low range to halt

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_wide_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_wide_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 5_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "Gate B.3.1 JMP.W physical differential mismatch: {:?}",
            res.err()
        );

        assert_eq!(
            cpu.read_reg(&sim, 0),
            42,
            "R0 must be computed at instructions 300/301 reached via JMP.W 300"
        );
        assert!(cpu.is_halted(), "program should halt at prog[2]");
        assert_eq!(
            cpu.pc_override_mismatches(),
            0,
            "wide PC fall-through after JMP.W must be physically authoritative"
        );
    }

    /// Sprint 372 (Gate D.1): COMPILER BACKEND — a recursive function written as
    /// **structured high-level code** (not hand-asm) compiles to V2 assembly,
    /// assembles, and runs **on the physical tile CPU** with results matching the
    /// ISS bit-for-bit. The compiler did register allocation, stack-frame
    /// construction, the calling convention (return via the physical MFLR/JMPR
    /// idiom — no MTLR), and control-flow lowering. `fact(5) = 120` via 5-deep
    /// recursion, every instruction physically executed (extended-PC config so the
    /// JMPR return target path is physical).
    #[test]
    fn test_v2_compiler_factorial_physical_differential() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_compiler::{
            Func, Program, Stmt, call, compile_to_words, int, lt, mul, sub, var,
        };
        use crate::tile_cpu::v2_components::v2_nop;

        // fact(n) = n < 2 ? 1 : n * fact(n-1);  main() = fact(5).
        let fact = Func {
            name: "fact".to_string(),
            params: vec!["n".to_string()],
            body: vec![Stmt::If(
                lt(var("n"), int(2)),
                vec![Stmt::Return(int(1))],
                vec![Stmt::Return(mul(
                    var("n"),
                    call("fact", vec![sub(var("n"), int(1))]),
                ))],
            )],
        };
        let main = Func {
            name: "main".to_string(),
            params: vec![],
            body: vec![Stmt::Return(call("fact", vec![int(5)]))],
        };
        let program = Program::new("main", vec![main, fact]);

        // Compile to machine words and pad to the 128-word ROM.
        let mut prog = compile_to_words(&program).expect("compile failed");
        assert!(
            prog.len() <= 128,
            "compiled program exceeds ROM ({} words)",
            prog.len()
        );
        prog.resize(128, v2_nop());

        // Same authoritative config as the gate differentials; extended PC so the
        // JMPR-based return target is computed in the physical commit path.
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc();

        let res = run_v2_differential_core(
            &mut iss,
            V2DiffConfig { max_ticks: 200_000 },
            &cpu,
            &mut sim,
        );
        assert!(
            res.is_ok(),
            "Gate D.1 compiled-factorial physical differential mismatch: {:?}",
            res.err()
        );

        // The compiled program computed fact(5) = 120 in tiles.
        assert_eq!(
            cpu.read_reg(&sim, 0),
            120,
            "R0 must hold the compiled fact(5) result"
        );
        assert!(cpu.is_halted(), "compiled program should halt");
    }

    /// Sprint 373 (Gate D.2): TEXT FRONT END — a C-like **source string** is lexed,
    /// parsed to the D.1 AST, compiled, assembled, and run **on the physical tile
    /// CPU** bit-for-bit with the ISS. Proves the whole toolchain — text source →
    /// tiles — end to end. `fact(5) = 120` from source.
    #[test]
    fn test_v2_parser_factorial_physical_differential() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_components::v2_nop;
        use crate::tile_cpu::v2_parser::compile_source;

        let src = r#"
            int fact(int n) {
                if (n < 2) { return 1; }
                return n * fact(n - 1);
            }
            int main() { return fact(5); }
        "#;
        let mut prog = compile_source(src).expect("compile_source failed");
        assert!(
            prog.len() <= 128,
            "program exceeds ROM ({} words)",
            prog.len()
        );
        prog.resize(128, v2_nop());

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc();

        let res = run_v2_differential_core(
            &mut iss,
            V2DiffConfig { max_ticks: 200_000 },
            &cpu,
            &mut sim,
        );
        assert!(
            res.is_ok(),
            "Gate D.2 source-factorial physical differential mismatch: {:?}",
            res.err()
        );
        assert_eq!(
            cpu.read_reg(&sim, 0),
            120,
            "R0 must hold fact(5) compiled from source text"
        );
        assert!(cpu.is_halted(), "compiled-from-source program should halt");
    }

    /// Sprint 374 (Gate D.3): SCALE PAST 128 INSTRUCTIONS. A compiled program of
    /// > 128 instructions — three linear-recursion functions + `main` summing them —
    /// runs **on the physical tile CPU** bit-exact with the ISS. The later functions
    /// sit above address 127 (physically fetched via the B.2 extended path) and use
    /// **wide calls + wide conditional branches** (`CALL.W`/`JZ.W`/`JMP.W`). Small
    /// arguments keep the run fast; program *size*, not runtime, is what exceeds 128.
    /// Result: `a(3) + b(3) + c(3) = 9`.
    #[test]
    fn test_v2_compiler_wide_program_physical_differential() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_compiler::{
            Func, Program, Stmt, add, call, compile_to_words, int, lt, sub, var,
        };
        use crate::tile_cpu::v2_components::v2_nop;

        // count(n) = n<1 ? 0 : 1 + count(n-1)  (returns n via linear recursion).
        let counter = |name: &str| Func {
            name: name.to_string(),
            params: vec!["n".to_string()],
            body: vec![Stmt::If(
                lt(var("n"), int(1)),
                vec![Stmt::Return(int(0))],
                vec![Stmt::Return(add(
                    int(1),
                    call(name, vec![sub(var("n"), int(1))]),
                ))],
            )],
        };
        // Sum a(3)+b(3)+c(3)+d(3)+e(3) = 15 — five functions so the program comfortably
        // exceeds 128 words even with the Perf-track call-overhead reductions.
        let main = Func {
            name: "main".to_string(),
            params: vec![],
            body: vec![Stmt::Return(add(
                add(
                    add(
                        add(call("a", vec![int(3)]), call("b", vec![int(3)])),
                        call("c", vec![int(3)]),
                    ),
                    call("d", vec![int(3)]),
                ),
                call("e", vec![int(3)]),
            ))],
        };
        let program = Program::new(
            "main",
            vec![
                main,
                counter("a"),
                counter("b"),
                counter("c"),
                counter("d"),
                counter("e"),
            ],
        );

        let mut prog = compile_to_words(&program).expect("compile failed");
        assert!(
            prog.len() > 128,
            "scale test must exceed 128 words (got {})",
            prog.len()
        );
        assert!(
            prog.len() <= 256,
            "stay within the extended (8-bit) PC range"
        );
        let rom_words = prog.len();
        prog.resize(rom_words.max(128), v2_nop());

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc();

        let res = run_v2_differential_core(
            &mut iss,
            V2DiffConfig { max_ticks: 400_000 },
            &cpu,
            &mut sim,
        );
        assert!(
            res.is_ok(),
            "Gate D.3 wide-program physical differential mismatch: {:?}",
            res.err()
        );
        assert_eq!(
            cpu.read_reg(&sim, 0),
            15,
            "R0 must hold a(3)+b(3)+c(3)+d(3)+e(3) computed across the 128 line"
        );
        assert!(cpu.is_halted(), "wide compiled program should halt");
    }

    /// DIAGNOSTIC (Sprint 374): isolate whether high-register INC/DEC + PUSH/POP at
    /// PC>=128 stay ISS<->physical consistent. (Probes the wide-program SP mismatch.)
    /// Assembled so high registers get the EXT_RD_HI bit.
    #[test]
    fn test_v2_diag_high_reg_ops_extended_pc() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::assemble_v2;

        let src = r#"
            LDI R8, 0
            LDI SP, 100
            LDI R3, 200
            CALLR R3
            HALT
            .org 200
            INC R8
            INC R8
            ST [SP], R8
            DEC SP
            INC SP
            RET
        "#;
        let prog = assemble_v2(src).expect("assemble failed");

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 5_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "high-reg extended-PC diagnostic mismatch: {:?}",
            res.err()
        );
        assert_eq!(cpu.read_reg(&sim, 8), 2, "R8 must be 2");
        assert_eq!(cpu.read_reg(&sim, 15), 100, "R15 must be 100");
    }

    /// DIAGNOSTIC (Sprint 374): the compiler's full return idiom (MFLR / PUSH / POP /
    /// CALL.W / conditional `.W` branch / JMPR) inside a **recursive function placed
    /// entirely above PC 128**. This is the structure the wide-program differential
    /// fails on (SP drift). `rec(3) = 3`.
    #[test]
    fn test_v2_diag_recursion_above_128() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::assemble_v2;

        let src = r#"
            LDI SP, 100
            LDI R0, 3
            CALL.W rec
            HALT
            .org 118
        rec:
            MFLR R5
            PUSH R5
            PUSH R0
            LDI R1, 1
            CMP R0, R1
            JC.W base
            DEC R0
            CALL.W rec
            POP R1
            INC R0
            JMP.W done
        base:
            POP R1
            LDI R0, 0
        done:
            POP R5
            JMPR R5
        "#;
        let prog = assemble_v2(src).expect("assemble failed");

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 20_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "recursion-above-128 diagnostic mismatch: {:?}",
            res.err()
        );
        assert_eq!(cpu.read_reg(&sim, 0), 3, "rec(3) must be 3");
    }

    /// DIAGNOSTIC (Sprint 374): straight-line fall-through that **crosses the 127->128
    /// boundary** (low ROM bank -> extended program_ext). Neither prior diagnostic
    /// covers this transition; the wide-program differential has a function straddling
    /// 128. Expect R0 = 5 after five INCs spanning 125..129.
    #[test]
    fn test_v2_diag_boundary_crossing_fallthrough() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::assemble_v2;

        let src = r#"
            LDI R0, 0
            LDI R3, 125
            CALLR R3
            HALT
            .org 125
            INC R0
            INC R0
            INC R0
            INC R0
            INC R0
            RET
        "#;
        let prog = assemble_v2(src).expect("assemble failed");

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 5_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "boundary-crossing fall-through mismatch (likely the wide-program bug): {:?}",
            res.err()
        );
        assert_eq!(
            cpu.read_reg(&sim, 0),
            5,
            "R0 must be 5 across the 127->128 boundary"
        );
    }

    /// DIAGNOSTIC (Sprint 374): instrumented lockstep on the failing wide a+b+c
    /// program — reports the exact PC / instruction where ISS and physical first
    /// diverge, with a ring buffer of preceding retired instructions.
    #[test]
    #[ignore = "diagnostic scaffold: prints the first ISS/physical divergence; kept for future debugging"]
    fn test_v2_diag_wide_program_trace() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::disassemble_v2_word;
        use crate::tile_cpu::v2_compiler::{
            Func, Program, Stmt, add, call, compile_to_words, int, lt, sub, var,
        };

        let counter = |name: &str| Func {
            name: name.to_string(),
            params: vec!["n".to_string()],
            body: vec![Stmt::If(
                lt(var("n"), int(1)),
                vec![Stmt::Return(int(0))],
                vec![Stmt::Return(add(
                    int(1),
                    call(name, vec![sub(var("n"), int(1))]),
                ))],
            )],
        };
        let main = Func {
            name: "main".to_string(),
            params: vec![],
            body: vec![Stmt::Return(add(
                add(call("a", vec![int(3)]), call("b", vec![int(3)])),
                call("c", vec![int(3)]),
            ))],
        };
        let program = Program::new("main", vec![main, counter("a"), counter("b"), counter("c")]);
        let prog = compile_to_words(&program).expect("compile failed");

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc();

        let mut ring: std::collections::VecDeque<String> = std::collections::VecDeque::new();
        let mut retired = 0u64;
        for tick in 1..=400_000u64 {
            let pc_exec = cpu.read_pc(&sim) as usize; // PC of the instruction about to retire
            let _ = cpu.step(&mut sim);
            if cpu.last_stage_x_valid() {
                retired += 1;
                let iss_pc_before = iss.state().pc as usize;
                iss.step();
                let csp = cpu.read_reg(&sim, 15);
                let isp = iss.state().regs[15];
                let cpc = cpu.read_pc(&sim);
                let ipc = iss.state().pc;
                let word = prog.get(pc_exec).copied().unwrap_or(0);
                let line = format!(
                    "retired={retired} tick={tick} execPC~{pc_exec}({}) | cpu: pc={cpc} sp={csp} | iss: pc={ipc} sp={isp}",
                    disassemble_v2_word(word)
                );
                ring.push_back(line);
                if ring.len() > 16 {
                    ring.pop_front();
                }
                if csp != isp || cpc != ipc {
                    let dump: Vec<String> = ring.iter().cloned().collect();
                    panic!(
                        "FIRST DIVERGENCE (iss_pc_before={iss_pc_before}):\n{}",
                        dump.join("\n")
                    );
                }
            }
            if cpu.is_halted() && iss.is_halted() {
                break;
            }
        }
        eprintln!("no divergence; final R0 cpu={}", cpu.read_reg(&sim, 0));
    }

    /// Sprint 374: MINIMAL REPRO of the extended-PC high->low return bug. A routine
    /// at PC 200 (high bank) `RET`s back to PC 3 (low bank) which is a register-writing
    /// instruction. The physical writeback for the instruction(s) at the low target is
    /// corrupted (B.2's own test returned to HALT, so it never exercised this). After
    /// the fix this must pass: R0 = 2.
    #[test]
    fn test_v2_high_to_low_return_writeback() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_components::{v2_callr, v2_halt, v2_inc, v2_ldi, v2_nop, v2_ret};

        let mut prog = vec![v2_nop(); 210];
        prog[0] = v2_ldi(0, 0); // R0 = 0
        prog[1] = v2_ldi(3, 200); // R3 = 200
        prog[2] = v2_callr(3); // -> 200 (low->high), LR = 3
        prog[3] = v2_inc(0); // return target (real writeback): R0 = 1
        prog[4] = v2_inc(0); // R0 = 2
        prog[5] = v2_halt();
        prog[200] = v2_ret(); // -> 3 (high->low return)

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        let mut iss = V2Iss::from_program(&prog);
        iss.enable_extended_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 5_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "high->low return writeback mismatch: {:?}",
            res.err()
        );
        assert_eq!(
            cpu.read_reg(&sim, 0),
            2,
            "R0 must be 2 after the high->low return"
        );
    }

    /// DIAGNOSTIC (Sprint 375 / Gate E de-risk): does MMIO (STB to the console device)
    /// work on the max_authority + extended_pc + 16-layer build the compiler targets?
    /// (The existing MMIO differential uses a plain 4-layer grid.) Writes 'H' then 'I'
    /// to the console and checks the device state, ISS<->physical lockstep.
    #[test]
    fn test_v2_diag_mmio_console_extended_pc() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_components::{v2_halt, v2_ldi, v2_nop, v2_stb};
        use crate::tile_cpu::v2_mmio::V2MmioHandle;
        use crate::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
        use std::rc::Rc;

        let mut prog = vec![v2_nop(); 10];
        prog[0] = v2_ldi(0, 72); // 'H'
        prog[1] = v2_stb(0, 57); // STB [57], R0  (MMIO_CONSOLE_DATA)
        prog[2] = v2_ldi(0, 73); // 'I'
        prog[3] = v2_stb(0, 57);
        prog[4] = v2_halt();

        let combined = Rc::new(V2MmioCombinedDevice::new(0xCAFE));
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_mmio(V2MmioHandle::from_rc(combined.clone()))
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program_with_mmio(&prog);
        iss.enable_extended_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 2_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "MMIO console differential mismatch: {:?}",
            res.err()
        );
        assert_eq!(
            combined.ref_pack.console_last(),
            73,
            "last console byte must be 'I'"
        );
        assert_eq!(combined.ref_pack.console_count(), 2, "two console writes");
    }

    /// Sprint 375 (Gate E): a C-like program compiled **from source** prints text via
    /// the `putchar` builtin, running on the **physical tile CPU** — the console device
    /// captures "HI", ISS<->physical bit-exact. The first compiled program with visible
    /// output computing in tiles.
    #[test]
    fn test_v2_compiler_putchar_hi_physical() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_mmio::V2MmioHandle;
        use crate::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
        use crate::tile_cpu::v2_parser::compile_source;
        use std::rc::Rc;

        let src = r#"
            int main() {
                putchar(72);  // 'H'
                putchar(73);  // 'I'
                return 0;
            }
        "#;
        let prog = compile_source(src).expect("compile_source failed");
        assert!(
            prog.len() <= 128,
            "demo should fit the low ROM ({} words)",
            prog.len()
        );

        let combined = Rc::new(V2MmioCombinedDevice::new(0xE));
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_mmio(V2MmioHandle::from_rc(combined.clone()))
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program_with_mmio(&prog);
        iss.enable_extended_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 50_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "putchar demo differential mismatch: {:?}",
            res.err()
        );
        assert_eq!(
            combined.ref_pack.console_string(),
            "HI",
            "the compiled program must have printed HI to the console (in tiles)"
        );
    }

    /// Sprint 375 (Gate E): a richer compiled demo — a helper function plus a counting
    /// `while` loop that prints digits '1'..'5', running on the physical tile CPU.
    /// Exercises putchar with control flow + a call. Console reads "12345".
    #[test]
    fn test_v2_compiler_putchar_loop_physical() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_mmio::V2MmioHandle;
        use crate::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
        use crate::tile_cpu::v2_parser::compile_source;
        use std::rc::Rc;

        // print_digit(d) prints the ASCII digit for 0..9 ('0' == 48).
        let src = r#"
            int print_digit(int d) {
                putchar(48 + d);
                return 0;
            }
            int main() {
                int i = 1;
                while (i <= 5) {
                    print_digit(i);
                    i = i + 1;
                }
                return 0;
            }
        "#;
        let prog = compile_source(src).expect("compile_source failed");

        let combined = Rc::new(V2MmioCombinedDevice::new(0xE));
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_mmio(V2MmioHandle::from_rc(combined.clone()))
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program_with_mmio(&prog);
        iss.enable_extended_pc();

        let res =
            run_v2_differential_core(&mut iss, V2DiffConfig { max_ticks: 80_000 }, &cpu, &mut sim);
        assert!(
            res.is_ok(),
            "putchar loop differential mismatch: {:?}",
            res.err()
        );
        assert_eq!(
            combined.ref_pack.console_string(),
            "12345",
            "the compiled loop must have printed 12345"
        );
    }

    /// Sprint 376 (Gate E.2): COMPUTE AND PRINT. A program computes the squares 1..5
    /// and prints them as decimal numbers via the auto-included `print_int` (decimal
    /// formatting written in the language: `div10`/`mod10` repeated subtraction). Runs
    /// on the physical tile CPU; console reads "1 4 9 16 25 ".
    #[test]
    fn test_v2_compiler_squares_physical() {
        use crate::simulation::Simulation;
        use crate::tile_cpu::V2Builder;
        use crate::tile_cpu::V2SynthConfig;
        use crate::tile_cpu::v2_mmio::V2MmioHandle;
        use crate::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
        use crate::tile_cpu::v2_parser::compile_source;
        use std::rc::Rc;

        let src = r#"
            int main() {
                int i = 1;
                while (i <= 5) {
                    print_int(i * i);
                    putchar(32);
                    i = i + 1;
                }
                return 0;
            }
        "#;
        let prog = compile_source(src).expect("compile_source failed");

        let combined = Rc::new(V2MmioCombinedDevice::new(0xE));
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&prog)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_mmio(V2MmioHandle::from_rc(combined.clone()))
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        let mut iss = V2Iss::from_program_with_mmio(&prog);
        iss.enable_extended_pc();

        let res = run_v2_differential_core(
            &mut iss,
            V2DiffConfig { max_ticks: 200_000 },
            &cpu,
            &mut sim,
        );
        assert!(
            res.is_ok(),
            "squares demo differential mismatch: {:?}",
            res.err()
        );
        assert_eq!(
            combined.ref_pack.console_string(),
            "1 4 9 16 25 ",
            "the compiled program must compute and print the squares"
        );
        // The ISS captured the same output (its console buffer mirrors the device).
        assert_eq!(iss.console_string(), "1 4 9 16 25 ");
    }

    /// Sprint 363: legacy programs (default PC mask 0x7F) must be unchanged — a
    /// jump to 200 without extended PC wraps to 200 & 0x7F = 72 (7-bit behavior).
    #[test]
    fn test_v2_iss_gate_b_legacy_pc_mask_preserved() {
        use crate::tile_cpu::v2_components::{v2_halt, v2_jmp, v2_ldi, v2_nop};

        let mut prog = vec![v2_nop(); 210];
        prog[0] = v2_jmp(200); // without extended PC: 200 & 0x7F = 72
        prog[72] = v2_ldi(0, 7);
        prog[73] = v2_halt();
        // (instruction 200 is never reached under the legacy 7-bit mask)

        let mut iss = V2Iss::from_program(&prog); // pc_mask defaults to 0x7F
        for _ in 0..1000 {
            if iss.state().halted {
                break;
            }
            iss.step();
        }
        assert!(iss.state().halted, "should halt at 73");
        assert_eq!(iss.state().regs[0], 7, "legacy 7-bit wrap reached instr 72");
        assert_eq!(iss.state().pc, 73);
    }

    #[test]
    fn test_v2_diff_mixed_bank_program() {
        let program = [
            v2_with_reg_ext(v2_ldi(0, 30), true, false), // R8 = 30
            v2_ldi(1, 12),                               // R1 = 12
            v2_with_reg_ext(v2_sub(0, 1), true, false),  // R8 = 18
            v2_with_reg_ext(v2_add(1, 0), false, true),  // R1 = 30
            v2_with_reg_ext(v2_stb(1, 9), false, false), // RAM[9] = R1
            v2_with_reg_ext(v2_ldb(2, 9), true, false),  // R10 = RAM[9]
            v2_halt(),
        ];
        let res = run_v2_differential(&program, V2DiffConfig { max_ticks: 160 });
        assert!(
            res.is_ok(),
            "mixed-bank differential failed: {:?}",
            res.err()
        );
    }

    #[test]
    fn test_v2_diff_control_flow_program() {
        let mut program = vec![v2_nop(); 32];
        program[0] = v2_ldi(0, 1);
        program[1] = v2_ldi(1, 1);
        program[2] = v2_cmp(0, 1);
        program[3] = v2_jz(6);
        program[4] = v2_ldi(2, 0xAA); // skipped
        program[5] = v2_jmp(9);
        program[6] = v2_call(12);
        program[7] = v2_jnc(10);
        program[8] = v2_halt(); // skipped by jnc
        program[9] = v2_halt();
        program[10] = v2_halt();
        program[12] = v2_inc(0);
        program[13] = v2_ret();

        let res = run_v2_differential(&program, V2DiffConfig { max_ticks: 256 });
        assert!(
            res.is_ok(),
            "control-flow differential failed: {:?}",
            res.err()
        );
    }

    #[test]
    fn test_v2_diff_seeded_random_corpus() {
        for seed in 1u64..=64u64 {
            let program = gen_program(seed);
            let res = run_v2_differential(&program, V2DiffConfig { max_ticks: 384 });
            if let Err(err) = &res {
                println!("seed {} program: {:#X?}", seed, program);
                println!("seed {} mismatch: {:?}", seed, err);
            }
            assert!(
                res.is_ok(),
                "seed {} differential failed: {:?}",
                seed,
                res.err()
            );
        }
    }

    #[test]
    #[ignore]
    fn test_v2_diff_seeded_random_corpus_10k_soak() {
        for seed in 1u64..=10_000u64 {
            let program = gen_program(seed);
            let res = run_v2_differential(&program, V2DiffConfig { max_ticks: 512 });
            assert!(
                res.is_ok(),
                "seed {} differential failed: {:?}",
                seed,
                res.err()
            );
        }
    }

    // ── Sprint 157: ISS math device mirror tests ──

    #[test]
    fn test_iss_math_device_mul() {
        // ISS should produce same MUL result as hardware math device
        let program = [
            v2_ldi(0, 7),  // R0 = 7
            v2_stb(0, 50), // MATH_A = 7
            v2_ldi(0, 6),  // R0 = 6
            v2_stb(0, 51), // MATH_B = 6
            v2_ldi(0, 0),  // R0 = 0 (MUL op)
            v2_stb(0, 52), // MATH_CMD = MUL
            v2_ldb(1, 53), // R1 = MATH_RESULT (should be 42)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..8 {
            iss.step();
        }
        assert_eq!(iss.state().regs[1], 42, "ISS math MUL: 7*6 should be 42");
        assert_eq!(iss.state().iss_math_status, 0, "status should be OK");
    }

    #[test]
    fn test_iss_math_device_div_by_zero() {
        // ISS should mirror status=1 for division by zero
        let program = [
            v2_ldi(0, 10), // R0 = 10
            v2_stb(0, 50), // MATH_A = 10
            v2_ldi(0, 0),  // R0 = 0
            v2_stb(0, 51), // MATH_B = 0
            v2_ldi(0, 1),  // R0 = 1 (DIV op)
            v2_stb(0, 52), // MATH_CMD = DIV
            v2_ldb(1, 52), // R1 = MATH_CMD (reads status)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..8 {
            iss.step();
        }
        assert_eq!(iss.state().regs[1], 1, "ISS math DIV/0: status should be 1");
        assert_eq!(iss.state().iss_math_status, 1);
    }

    #[test]
    fn test_iss_math_popcount() {
        // ISS should produce same POPCOUNT result as hardware math device
        let program = [
            v2_ldi(0, 0xFF), // R0 = 0xFF (8 bits set)
            v2_stb(0, 50),   // MATH_A = 0xFF
            v2_ldi(0, 4),    // R0 = 4 (POPCOUNT op)
            v2_stb(0, 52),   // MATH_CMD = POPCOUNT
            v2_ldb(1, 53),   // R1 = MATH_RESULT (should be 8)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..6 {
            iss.step();
        }
        assert_eq!(
            iss.state().regs[1],
            8,
            "ISS math POPCOUNT: 0xFF should have 8 bits"
        );
        assert_eq!(iss.state().iss_math_status, 0, "status should be OK");
    }

    #[test]
    fn test_iss_differential_factorial() {
        // Sprint 158: Use run_v2_differential_mmio for MMIO-aware differential.
        use crate::tile_cpu::v2_assembler::assemble_v2;
        use crate::tile_cpu::v2_showcase::SHOWCASE_FACTORIAL_TABLE;

        let words = assemble_v2(SHOWCASE_FACTORIAL_TABLE.source).unwrap();
        let config = V2DiffConfig { max_ticks: 10000 };
        let result = run_v2_differential_mmio(&words, config);
        assert!(
            result.is_ok(),
            "factorial differential with ISS math: {:?}",
            result.err()
        );
    }

    // ── Sprint 158: ISS MMIO safety tests ──

    #[test]
    fn test_iss_mmio_disabled_passthrough() {
        // With MMIO disabled (default), STB to addr 50 should write to ram[50]
        let program = [
            v2_ldi(0, 42),
            v2_stb(0, 50), // addr 50 = MMIO_MATH_A, but MMIO disabled
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program(&program);
        for _ in 0..3 {
            iss.step();
        }
        // Without MMIO, value goes to ram[50], not iss_math_a
        assert_eq!(iss.state().ram[50], 42, "ram[50] should have the value");
        assert_eq!(iss.state().iss_math_a, 0, "iss_math_a should be untouched");
    }

    #[test]
    fn test_iss_mmio_enabled_intercept() {
        // With MMIO enabled, STB to addr 50 should intercept to iss_math_a
        let program = [
            v2_ldi(0, 42),
            v2_stb(0, 50), // addr 50 = MMIO_MATH_A, MMIO enabled
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..3 {
            iss.step();
        }
        // With MMIO, value goes to iss_math_a, not ram[50]
        assert_eq!(
            iss.state().iss_math_a,
            42,
            "iss_math_a should have the value"
        );
        assert_eq!(iss.state().ram[50], 0, "ram[50] should be untouched");
    }

    // ── Sprint 158: ISS quantum mirror tests ──

    #[test]
    fn test_iss_quantum_h_measure() {
        // ISS standalone: INIT 1 qubit, H, measure → valid 0 or 1
        let program = [
            v2_ldi(0, 1),
            v2_stb(0, 46), // QUANTUM_DATA = 1 (1 qubit)
            v2_ldi(0, 0),
            v2_stb(0, 44), // QUANTUM_CMD = INIT
            v2_stb(0, 45), // QUANTUM_QUBIT = 0
            v2_ldi(0, 2),
            v2_stb(0, 44), // QUANTUM_CMD = H
            v2_ldi(0, 10),
            v2_stb(0, 44), // QUANTUM_CMD = MEASURE
            v2_ldb(1, 46), // R1 = QUANTUM_DATA (result)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..11 {
            iss.step();
        }
        let result = iss.state().regs[1];
        assert!(result <= 1, "H+measure should give 0 or 1, got {}", result);
    }

    #[test]
    fn test_iss_quantum_bell_state() {
        // ISS standalone: INIT 2, H(0), CNOT(0,1), MEASURE_ALL → 0 or 3
        let program = [
            v2_ldi(0, 2),
            v2_stb(0, 46), // QUANTUM_DATA = 2
            v2_ldi(0, 0),
            v2_stb(0, 44), // INIT(2)
            v2_stb(0, 45), // QUANTUM_QUBIT = 0
            v2_ldi(0, 2),
            v2_stb(0, 44), // H(q0)
            v2_ldi(0, 1),
            v2_stb(0, 46), // QUANTUM_DATA = 1 (target for CNOT)
            v2_ldi(0, 0),
            v2_stb(0, 45), // QUANTUM_QUBIT = 0 (control)
            v2_ldi(0, 7),
            v2_stb(0, 44), // CNOT(0, 1)
            v2_ldi(0, 11),
            v2_stb(0, 44), // MEASURE_ALL
            v2_ldb(1, 46), // R1 = result
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..17 {
            iss.step();
        }
        let result = iss.state().regs[1];
        assert!(
            result == 0 || result == 3,
            "Bell state should give 0 or 3, got {}",
            result
        );
    }

    #[test]
    fn test_iss_differential_deutsch_jozsa() {
        use crate::tile_cpu::v2_assembler::assemble_v2;
        use crate::tile_cpu::v2_showcase::SHOWCASE_DEUTSCH_JOZSA;
        let words = assemble_v2(SHOWCASE_DEUTSCH_JOZSA.source).unwrap();
        let config = V2DiffConfig { max_ticks: 10000 };
        let result = run_v2_differential_mmio(&words, config);
        assert!(result.is_ok(), "DJ differential: {:?}", result.err());
    }

    #[test]
    fn test_iss_differential_grover() {
        use crate::tile_cpu::v2_assembler::assemble_v2;
        use crate::tile_cpu::v2_showcase::SHOWCASE_GROVER_SEARCH;
        let words = assemble_v2(SHOWCASE_GROVER_SEARCH.source).unwrap();
        let config = V2DiffConfig { max_ticks: 10000 };
        let result = run_v2_differential_mmio(&words, config);
        assert!(result.is_ok(), "Grover differential: {:?}", result.err());
    }

    #[test]
    fn test_iss_differential_bv() {
        use crate::tile_cpu::v2_assembler::assemble_v2;
        use crate::tile_cpu::v2_showcase::SHOWCASE_BERNSTEIN_VAZIRANI;
        let words = assemble_v2(SHOWCASE_BERNSTEIN_VAZIRANI.source).unwrap();
        let config = V2DiffConfig { max_ticks: 10000 };
        let result = run_v2_differential_mmio(&words, config);
        assert!(result.is_ok(), "BV differential: {:?}", result.err());
    }

    // ── Sprint 158: ISS ref device mirror tests ──

    #[test]
    fn test_iss_ref_device_console() {
        // Write two bytes to console, verify count
        let program = [
            v2_ldi(0, 65), // 'A'
            v2_stb(0, 57), // CONSOLE_DATA = 65
            v2_ldi(0, 66), // 'B'
            v2_stb(0, 57), // CONSOLE_DATA = 66
            v2_ldb(1, 58), // R1 = CONSOLE_COUNT
            v2_ldb(2, 57), // R2 = CONSOLE_DATA (last byte)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..7 {
            iss.step();
        }
        assert_eq!(iss.state().regs[1], 2, "console count should be 2");
        assert_eq!(
            iss.state().regs[2],
            66,
            "console last byte should be 'B' (66)"
        );
    }

    #[test]
    fn test_iss_ref_device_rng() {
        // Two reads from RNG should return different non-zero values
        let program = [
            v2_ldb(0, 59), // R0 = RNG
            v2_ldb(1, 59), // R1 = RNG (second read)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..3 {
            iss.step();
        }
        let r0 = iss.state().regs[0];
        let r1 = iss.state().regs[1];
        assert_ne!(r0, 0, "RNG first read should be non-zero");
        assert_ne!(r1, 0, "RNG second read should be non-zero");
        assert_ne!(r0, r1, "RNG reads should differ");
    }

    #[test]
    fn test_iss_ref_device_mailbox() {
        // Sprint 297: Mailbox split — write goes to _send, read returns _recv.
        // Single-CPU: write addr 60 then read addr 60 returns 0 (no loopback).
        // This matches V2LinkMailboxDevice hardware (Rc<Cell> are separate).
        let program = [
            v2_ldi(0, 42),
            v2_stb(0, 60), // MAILBOX_IN send = 42
            v2_ldb(1, 60), // R1 = MAILBOX_IN recv (0, no neighbor)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..4 {
            iss.step();
        }
        // No loopback: read returns recv (0), not the just-written send value.
        assert_eq!(
            iss.state().regs[1],
            0,
            "mailbox_in recv should be 0 (no loopback)"
        );
        // But the send field should have captured the write.
        assert_eq!(iss.mailbox_in_send(), 42, "mailbox_in send should be 42");
    }

    // --- ISS dataset mirror tests (Sprint 159) ---

    #[test]
    fn test_iss_dataset_mirror() {
        // ISS standalone: load sample 0, get features, verify match
        let samples = vec![(0xDEAD_BEEF_u64, 1u64), (0xCAFE_BABE, 0)];
        let program = [
            v2_ldi(0, 0),  // R0 = 0 (sample index)
            v2_stb(0, 42), // DATASET_DATA = 0
            v2_ldi(0, 0),  // R0 = 0 (LOAD_SAMPLE cmd)
            v2_stb(0, 41), // DATASET_CMD = LOAD_SAMPLE
            v2_ldi(0, 1),  // R0 = 1 (GET_FEATURES cmd)
            v2_stb(0, 41), // DATASET_CMD = GET_FEATURES
            v2_ldb(1, 42), // R1 = DATASET_DATA (features)
            v2_ldi(0, 2),  // R0 = 2 (GET_LABEL cmd)
            v2_stb(0, 41), // DATASET_CMD = GET_LABEL
            v2_ldb(2, 42), // R2 = DATASET_DATA (label)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_dataset(&program, samples);
        for _ in 0..11 {
            iss.step();
        }
        assert_eq!(iss.state().regs[1], 0xDEAD_BEEF, "features should match");
        assert_eq!(iss.state().regs[2], 1, "label should be 1");
    }

    #[test]
    fn test_iss_dataset_error_codes() {
        use crate::tile_cpu::v2_mmio_devices::{DS_ERR_INVALID_CMD, DS_ERR_NO_SAMPLE, DS_ERR_OOB};

        let samples = vec![(0xAA, 0)];
        let mut iss = V2Iss::from_program_with_dataset(&[v2_halt()], samples);

        // GET_FEATURES before LOAD_SAMPLE → NO_SAMPLE
        iss.iss_dataset_execute_cmd(1);
        assert_eq!(iss.iss_dataset_last_error, DS_ERR_NO_SAMPLE);

        // LOAD_SAMPLE with OOB index
        iss.iss_dataset_data = 99;
        iss.iss_dataset_execute_cmd(0);
        assert_eq!(iss.iss_dataset_last_error, DS_ERR_OOB);

        // Unknown command
        iss.iss_dataset_execute_cmd(99);
        assert_eq!(iss.iss_dataset_last_error, DS_ERR_INVALID_CMD);

        // Valid load
        iss.iss_dataset_data = 0;
        iss.iss_dataset_execute_cmd(0);
        assert_eq!(iss.iss_dataset_last_error, 0); // DS_OK
        iss.iss_dataset_execute_cmd(1); // GET_FEATURES
        assert_eq!(iss.iss_dataset_data, 0xAA);
    }

    #[test]
    fn test_iss_dataset_randomized_commands() {
        // Run randomized command sequences, compare ISS vs hardware output
        use crate::tile_cpu::v2_mmio::V2MmioDevice;
        use crate::tile_cpu::v2_mmio_devices::{DatasetSample, V2MmioDatasetDevice};

        let samples_hw = vec![
            DatasetSample {
                features: 0x1234,
                label: 0,
            },
            DatasetSample {
                features: 0x5678,
                label: 1,
            },
            DatasetSample {
                features: 0x9ABC,
                label: 0,
            },
        ];
        let samples_iss: Vec<(u64, u64)> =
            samples_hw.iter().map(|s| (s.features, s.label)).collect();

        let hw = V2MmioDatasetDevice::from_samples(samples_hw);
        let mut iss = V2Iss::from_program_with_dataset(&[v2_halt()], samples_iss);

        // 60 command sequences generated from a simple LCG
        let mut rng = 12345u64;
        for _ in 0..60 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let cmd = (rng >> 16) % 8; // 0..7, includes invalid commands
            let data_val = (rng >> 32) % 5;

            // Set data on both sides
            hw.write(MMIO_DATASET_DATA, data_val);
            iss.iss_dataset_data = data_val;

            // Execute command on both sides
            hw.write(MMIO_DATASET_CMD, cmd);
            iss.iss_dataset_execute_cmd(cmd);

            // Compare DATA register
            assert_eq!(
                hw.read(MMIO_DATASET_DATA),
                iss.iss_dataset_data,
                "DATA diverged after cmd={cmd} data={data_val}"
            );
            // Compare error code
            assert_eq!(
                hw.read(MMIO_DATASET_CMD) >> 8,
                iss.iss_dataset_last_error as u64,
                "error diverged after cmd={cmd} data={data_val}"
            );
        }
    }

    // --- ISS SNN INFER mirror tests (M11) ---

    #[test]
    fn test_iss_snn_infer_basic() {
        use crate::snn::mlp_weights::{CachedRates, MlpWeights};

        // Build a tiny 2-2-2-2 identity MLP: input→output passthrough, argmax
        let weights = MlpWeights {
            w1: vec![1.0, 0.0, 0.0, 1.0], // 2x2 identity
            b1: vec![0.0, 0.0],
            w2: vec![1.0, 0.0, 0.0, 1.0], // 2x2 identity
            b2: vec![0.0, 0.0],
            w3: vec![1.0, -1.0, -1.0, 1.0], // contrast
            b3: vec![0.0, 0.0],
        };
        // Two samples: sample 0 → class 0, sample 1 → class 1
        let rates = CachedRates::new(2, vec![1.0, 0.0, 0.0, 1.0]);
        let samples = vec![(0u64, 0u64), (0, 1)]; // labels for submit_prediction

        // Program: INFER sample 0, read prediction
        let program = [
            v2_ldi(0, 0),  // R0 = 0 (sample index)
            v2_stb(0, 54), // SNN_DATA = 0
            v2_ldi(1, 7),  // R1 = 7 (INFER cmd)
            v2_stb(1, 55), // SNN_CMD = INFER
            v2_ldb(2, 54), // R2 = SNN_DATA (prediction for sample 0)
            // Now infer sample 1
            v2_ldi(0, 1),  // R0 = 1
            v2_stb(0, 54), // SNN_DATA = 1
            v2_stb(1, 55), // SNN_CMD = INFER (R1 still 7)
            v2_ldb(3, 54), // R3 = SNN_DATA (prediction for sample 1)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_snn_model(&program, samples, weights, rates);
        for _ in 0..10 {
            iss.step();
        }
        assert_eq!(iss.state().regs[2], 0, "sample 0 should predict class 0");
        assert_eq!(iss.state().regs[3], 1, "sample 1 should predict class 1");
    }

    #[test]
    fn test_iss_snn_infer_no_model() {
        // Without a model, SNN commands fall through to RAM
        let program = [
            v2_ldi(0, 42), // R0 = 42
            v2_stb(0, 54), // SNN_DATA (falls through to RAM[54])
            v2_ldi(1, 7),  // R1 = 7
            v2_stb(1, 55), // SNN_CMD (falls through to RAM[55])
            v2_ldb(2, 54), // R2 = RAM[54] = 42 (no INFER executed)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..6 {
            iss.step();
        }
        // Without model, SNN_DATA/CMD are not intercepted → writes go to RAM
        assert_eq!(iss.state().regs[2], 42, "should read back raw RAM value");
    }

    #[test]
    fn test_iss_snn_infer_oob() {
        use crate::snn::mlp_weights::{CachedRates, MlpWeights};

        let weights = MlpWeights {
            w1: vec![1.0, 0.0, 0.0, 1.0],
            b1: vec![0.0, 0.0],
            w2: vec![1.0, 0.0, 0.0, 1.0],
            b2: vec![0.0, 0.0],
            w3: vec![1.0, -1.0, -1.0, 1.0],
            b3: vec![0.0, 0.0],
        };
        // Only 1 sample
        let rates = CachedRates::new(2, vec![1.0, 0.0]);
        let samples = vec![(0u64, 0u64)];

        // Try to infer sample 5 (OOB)
        let program = [
            v2_ldi(0, 5),  // R0 = 5 (OOB)
            v2_stb(0, 54), // SNN_DATA = 5
            v2_ldi(1, 7),  // R1 = 7 (INFER)
            v2_stb(1, 55), // SNN_CMD = INFER
            v2_ldb(2, 54), // R2 = SNN_DATA (should be 0)
            v2_ldb(3, 55), // R3 = SNN_CMD (status = 1 error)
            v2_halt(),
        ];
        let mut iss = V2Iss::from_program_with_snn_model(&program, samples, weights, rates);
        for _ in 0..7 {
            iss.step();
        }
        assert_eq!(iss.state().regs[2], 0, "OOB prediction should be 0");
        assert_eq!(iss.state().regs[3], 1, "OOB status should be 1 (error)");
    }

    #[test]
    fn test_iss_snn_infer_parity_with_direct() {
        use crate::snn::mlp_weights::{CachedRates, MlpWeights};

        // Build a slightly more complex MLP
        let weights = MlpWeights {
            w1: vec![1.0, 0.5, -0.3, 0.0, 0.0, 1.0, 0.2, -0.5, 0.0, 0.0, 0.0, 1.0],
            b1: vec![0.0, 0.0, 0.0, 0.0],
            w2: vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5],
            b2: vec![0.0, 0.0],
            w3: vec![1.0, -1.0, -1.0, 1.0],
            b3: vec![0.0, 0.0],
        };
        // 5 samples with 3 hidden neurons each
        let n_hid = 3;
        let sample_rates = vec![
            1.0, 0.0, 0.0, // sample 0: class 0
            0.0, 1.0, 0.0, // sample 1
            0.0, 0.0, 1.0, // sample 2
            0.5, 0.5, 0.0, // sample 3
            0.0, 0.5, 0.5, // sample 4
        ];
        let rates = CachedRates::new(n_hid, sample_rates);

        // Get direct predictions
        let direct: Vec<usize> = (0..5).map(|i| weights.forward_cpu(rates.get(i))).collect();

        // Run each through ISS and compare
        for sample_idx in 0..5u8 {
            let samples = vec![(0u64, 0u64); 5]; // labels don't matter for this test
            let program = [
                v2_ldi(0, sample_idx),
                v2_stb(0, 54), // SNN_DATA = idx
                v2_ldi(1, 7),  // INFER
                v2_stb(1, 55), // SNN_CMD = INFER
                v2_ldb(2, 54), // R2 = prediction
                v2_halt(),
            ];
            let mut iss = V2Iss::from_program_with_snn_model(
                &program,
                samples,
                weights.clone(),
                rates.clone(),
            );
            for _ in 0..6 {
                iss.step();
            }
            assert_eq!(
                iss.state().regs[2],
                direct[sample_idx as usize] as u64,
                "ISS prediction for sample {sample_idx} should match direct forward_cpu"
            );
        }
    }

    // --- M12: ISS live SNN mirror tests ---

    #[test]
    fn test_iss_snn_live_basic() {
        use crate::snn::mlp_weights::{LiveSnnModel, MlpWeights};

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
            n_ticks: 20,
            mlp: MlpWeights {
                w1: vec![1.0, 0.0, 0.0, 1.0],
                b1: vec![0.0, 0.0],
                w2: vec![1.0, 0.0, 0.0, 1.0],
                b2: vec![0.0, 0.0],
                w3: vec![1.0, -1.0, -1.0, 1.0],
                b3: vec![0.0, 0.0],
            },
        };

        let images = vec![vec![200u8; 4]];
        let samples = vec![(0u64, 0u64)];

        // Program: LOAD_IMAGE(0) → SNN_RUN → INFER_LIVE → read prediction
        let program = [
            v2_ldi(0, 0),  // R0 = 0 (sample idx)
            v2_stb(0, 54), // SNN_DATA = 0
            v2_ldi(1, 8),  // LOAD_IMAGE
            v2_stb(1, 55), // SNN_CMD = 8
            v2_ldi(1, 9),  // SNN_RUN
            v2_stb(1, 55), // SNN_CMD = 9
            v2_ldi(1, 10), // INFER_LIVE
            v2_stb(1, 55), // SNN_CMD = 10
            v2_ldb(2, 54), // R2 = prediction
            v2_halt(),
        ];

        let mut iss = V2Iss::from_program_with_live_snn_model(&program, samples, model, images, 42);
        for _ in 0..10 {
            iss.step();
        }
        assert!(iss.state().halted);
        let pred = iss.state().regs[2];
        assert!(pred <= 1, "prediction should be 0 or 1, got {pred}");
    }

    #[test]
    fn test_iss_snn_live_no_model() {
        // Without a live model, commands 8-10 should return error
        let program = [
            v2_ldi(0, 0),
            v2_stb(0, 54), // SNN_DATA
            v2_ldi(1, 8),
            v2_stb(1, 55), // SNN_CMD = LOAD_IMAGE → no live model
            v2_halt(),
        ];
        // Use from_program_with_mmio (no SNN model at all → cmds not intercepted)
        // This is expected — the ISS only intercepts SNN if model is present
        let mut iss = V2Iss::from_program_with_mmio(&program);
        for _ in 0..5 {
            iss.step();
        }
        assert!(iss.state().halted);
    }

    #[test]
    fn test_iss_snn_live_parity_with_device() {
        use crate::snn::mlp_weights::{LiveSnnModel, MlpWeights};
        use crate::tile_cpu::v2_mmio::V2MmioDevice;
        use crate::tile_cpu::v2_mmio_devices::{
            MMIO_SNN_CMD, MMIO_SNN_DATA, V2MmioSnnBridgeDevice,
        };

        // Build identical model for both ISS and device
        let model = LiveSnnModel {
            syn_ptr: vec![0, 1, 1, 3, 4, 4],
            targets: vec![3, 3, 4, 4],
            weights: vec![80, 90, 70, 60],
            thresholds: vec![32000, 32000, 40, 40, 100],
            leaks: vec![230; 5],
            pix_per_class: vec![vec![0], vec![1]],
            d_norms: vec![vec![1.0], vec![1.0]],
            n_input: 2,
            n_hidden: 2,
            n_readout: 1,
            n_classes: 2,
            k_per_class: 1,
            max_rate: 100,
            n_ticks: 30,
            mlp: MlpWeights {
                w1: vec![1.0, 0.0, 0.0, 1.0],
                b1: vec![0.0, 0.0],
                w2: vec![1.0, 0.0, 0.0, 1.0],
                b2: vec![0.0, 0.0],
                w3: vec![1.0, -1.0, -1.0, 1.0],
                b3: vec![0.0, 0.0],
            },
        };

        let images = vec![vec![255u8, 0], vec![0u8, 255]];
        let seed = 42u64;

        // Run through ISS
        let program = [
            v2_ldi(0, 0),  // sample idx
            v2_stb(0, 54), // SNN_DATA
            v2_ldi(1, 8),  // LOAD_IMAGE
            v2_stb(1, 55), // SNN_CMD
            v2_ldi(1, 9),  // SNN_RUN
            v2_stb(1, 55),
            v2_ldi(1, 10), // INFER_LIVE
            v2_stb(1, 55),
            v2_ldb(2, 54), // prediction
            v2_halt(),
        ];

        let mut iss = V2Iss::from_program_with_live_snn_model(
            &program,
            vec![(0, 0)],
            model.clone(),
            images.clone(),
            seed,
        );
        for _ in 0..10 {
            iss.step();
        }
        let iss_pred = iss.state().regs[2];

        // Run through device
        let dev = V2MmioSnnBridgeDevice::with_live_model(2, 2, 1, seed, model, images);
        dev.write(MMIO_SNN_DATA, 0);
        dev.write(MMIO_SNN_CMD, 8); // LOAD_IMAGE
        dev.write(MMIO_SNN_CMD, 9); // SNN_RUN
        dev.write(MMIO_SNN_CMD, 10); // INFER_LIVE
        let dev_pred = dev.read(MMIO_SNN_DATA);

        assert_eq!(
            iss_pred, dev_pred,
            "ISS live prediction ({iss_pred}) should match device ({dev_pred})"
        );
    }
}
