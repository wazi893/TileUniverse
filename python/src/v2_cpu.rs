use engine::simulation::Simulation;
use engine::tile_cpu::{
    TileCpuV2, V2Builder, V2SynthConfig, assemble_v2, capture_v2_ground_truth_jsonl,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// PyV2Cpu — Python-facing wrapper for the V2 tile CPU
// ---------------------------------------------------------------------------

/// A tile-fabric CPU with 32-bit ISA, 16 registers, 128-entry ROM, 128-cell RAM.
///
/// Construct from assembly source or raw u32 program words.
///
/// Example:
///     cpu = V2Cpu.from_asm("LDI R0, 42\\nHALT")
///     cpu.run(100)
///     assert cpu.reg(0) == 42
#[pyclass(unsendable, name = "V2Cpu")]
pub struct PyV2Cpu {
    sim: RefCell<Simulation>,
    cpu: TileCpuV2,
    synth_mode: &'static str,
}

#[pymethods]
impl PyV2Cpu {
    /// Assemble source code and build a CPU with the resulting program.
    #[staticmethod]
    #[pyo3(signature = (source, *, rom_size=128, ram_size=128, synth="auto"))]
    fn from_asm(source: &str, rom_size: usize, ram_size: usize, synth: &str) -> PyResult<Self> {
        let program = assemble_v2(source)
            .map_err(|e| PyValueError::new_err(format!("Assembly error: {}", e)))?;
        Self::build_from_program(&program, rom_size, ram_size, synth)
    }

    /// Build a CPU from raw u32 instruction words.
    #[staticmethod]
    #[pyo3(signature = (program, *, rom_size=128, ram_size=128, synth="auto"))]
    fn from_program(
        program: Vec<u32>,
        rom_size: usize,
        ram_size: usize,
        synth: &str,
    ) -> PyResult<Self> {
        Self::build_from_program(&program, rom_size, ram_size, synth)
    }

    /// Run up to `max_cycles` instructions or until HALT.
    #[pyo3(signature = (max_cycles=10000))]
    fn run(&self, max_cycles: u64) {
        let mut sim = self.sim.borrow_mut();
        self.cpu.run(&mut sim, max_cycles);
    }

    /// Execute exactly one instruction.
    fn step(&self) {
        let mut sim = self.sim.borrow_mut();
        self.cpu.run(&mut sim, 1);
    }

    /// True if the CPU has executed a HALT instruction.
    #[getter]
    fn halted(&self) -> bool {
        self.cpu.is_halted()
    }

    /// Current program counter (wide PC: 0-65535 on wide-PC builds).
    #[getter]
    fn pc(&self) -> u32 {
        let sim = self.sim.borrow();
        self.cpu.read_pc(&sim)
    }

    /// Read a single register (0-15). Returns 64-bit value.
    fn reg(&self, index: usize) -> PyResult<u64> {
        if index >= 16 {
            return Err(PyValueError::new_err("Register index must be 0-15"));
        }
        let sim = self.sim.borrow();
        Ok(self.cpu.read_reg(&sim, index))
    }

    /// Read all 16 registers as a list.
    fn regs(&self) -> Vec<u64> {
        let sim = self.sim.borrow();
        (0..16).map(|i| self.cpu.read_reg(&sim, i)).collect()
    }

    /// Read a single RAM cell (0-127). Returns 64-bit value.
    fn ram(&self, addr: usize) -> PyResult<u64> {
        if addr >= 128 {
            return Err(PyValueError::new_err("RAM address must be 0-127"));
        }
        let sim = self.sim.borrow();
        Ok(self.cpu.read_ram(&sim, addr))
    }

    /// Read all 128 RAM cells as a list.
    fn ram_dump(&self) -> Vec<u64> {
        let sim = self.sim.borrow();
        (0..128).map(|i| self.cpu.read_ram(&sim, i)).collect()
    }

    /// Read CPU flags: {"zero": bool, "carry": bool}.
    fn flags(&self) -> std::collections::HashMap<String, bool> {
        let sim = self.sim.borrow();
        let mut m = std::collections::HashMap::new();
        m.insert("zero".to_string(), self.cpu.read_flag_z(&sim));
        m.insert("carry".to_string(), self.cpu.read_flag_c(&sim));
        m
    }

    /// Total cycles executed so far.
    #[getter]
    fn cycle_count(&self) -> u64 {
        self.cpu.read_cycle_count()
    }

    /// Active synth mode: "auto", "auto_separate", or "off".
    #[getter]
    fn synth_mode(&self) -> &str {
        self.synth_mode
    }

    /// Hybrid assist counters accumulated so far.
    fn assist_counters(&self) -> std::collections::HashMap<String, u64> {
        let counters = self.cpu.read_hybrid_assist_counters();
        let mut m = std::collections::HashMap::new();
        m.insert(
            "stage_f_bank_switches".to_string(),
            counters.stage_f_bank_switches,
        );
        m.insert(
            "stage_f_mixed_dual_capture".to_string(),
            counters.stage_f_mixed_dual_capture,
        );
        m.insert(
            "stage_x_mixed_software".to_string(),
            counters.stage_x_mixed_software,
        );
        m.insert(
            "ram_high_bank_read_swaps".to_string(),
            counters.ram_high_bank_read_swaps,
        );
        m.insert(
            "rom_upper_bank_group_select".to_string(),
            counters.rom_upper_bank_group_select,
        );
        m
    }

    /// Run up to `max_cycles` and return a trace of execution.
    ///
    /// Returns a V2Trace with per-cycle snapshots of PC, registers, and flags.
    #[pyo3(signature = (max_cycles=10000))]
    fn run_traced(&self, max_cycles: u64) -> PyV2Trace {
        let mut sim = self.sim.borrow_mut();
        let mut pcs = Vec::new();
        let mut regs = Vec::new();
        let mut flag_z = Vec::new();
        let mut flag_c = Vec::new();

        for _ in 0..max_cycles {
            if self.cpu.is_halted() {
                break;
            }
            // Snapshot before step
            let pc = self.cpu.read_pc(&sim);
            let r: Vec<u64> = (0..16).map(|i| self.cpu.read_reg(&sim, i)).collect();
            let z = self.cpu.read_flag_z(&sim);
            let c = self.cpu.read_flag_c(&sim);
            pcs.push(pc);
            regs.push(r);
            flag_z.push(z);
            flag_c.push(c);
            self.cpu.run(&mut sim, 1);
        }

        PyV2Trace {
            pcs,
            regs,
            flag_z,
            flag_c,
        }
    }

    fn __repr__(&self) -> String {
        let sim = self.sim.borrow();
        format!(
            "V2Cpu(pc={}, halted={}, cycles={}, synth='{}')",
            self.cpu.read_pc(&sim),
            self.cpu.is_halted(),
            self.cpu.read_cycle_count(),
            self.synth_mode,
        )
    }
}

impl PyV2Cpu {
    fn build_from_program(
        program: &[u32],
        rom_size: usize,
        ram_size: usize,
        synth: &str,
    ) -> PyResult<Self> {
        if rom_size != 64 && rom_size != 128 {
            return Err(PyValueError::new_err("rom_size must be 64 or 128"));
        }
        if ram_size != 64 && ram_size != 128 {
            return Err(PyValueError::new_err("ram_size must be 64 or 128"));
        }
        let synth_mode = parse_synth_mode(synth)?;
        let mut sim = Simulation::with_size_layered(128, 384, 4);
        let mut builder = V2Builder::new()
            .with_origin(0, 0)
            .with_program(program)
            .with_rom_size(rom_size)
            .with_ram_size(ram_size);
        if let Some(config) = synth_mode.config() {
            builder = builder.with_synth_blocks(config);
        }
        let cpu = builder.build(&mut sim);
        Ok(PyV2Cpu {
            sim: RefCell::new(sim),
            cpu,
            synth_mode: synth_mode.name(),
        })
    }
}

#[derive(Clone, Copy)]
enum PySynthMode {
    Auto,
    AutoSeparate,
    Off,
}

impl PySynthMode {
    fn parse(mode: &str) -> PyResult<Self> {
        match mode.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "auto_separate" | "separate" => Ok(Self::AutoSeparate),
            "off" | "none" => Ok(Self::Off),
            _ => Err(PyValueError::new_err(
                "synth must be 'auto', 'auto_separate', or 'off'",
            )),
        }
    }

    fn config(self) -> Option<V2SynthConfig> {
        match self {
            Self::Auto => Some(V2SynthConfig::auto()),
            Self::AutoSeparate => Some(V2SynthConfig::auto_separate()),
            Self::Off => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::AutoSeparate => "auto_separate",
            Self::Off => "off",
        }
    }
}

fn parse_synth_mode(mode: &str) -> PyResult<PySynthMode> {
    PySynthMode::parse(mode)
}

// ---------------------------------------------------------------------------
// PyV2Trace — execution trace returned by run_traced()
// ---------------------------------------------------------------------------

/// Per-cycle execution trace from a V2 CPU run.
#[pyclass(unsendable, name = "V2Trace")]
pub struct PyV2Trace {
    pcs: Vec<u32>,
    regs: Vec<Vec<u64>>,
    flag_z: Vec<bool>,
    flag_c: Vec<bool>,
}

#[pymethods]
impl PyV2Trace {
    /// List of PC values at each cycle.
    #[getter]
    fn pcs(&self) -> Vec<u32> {
        self.pcs.clone()
    }

    /// Number of cycles in the trace.
    #[getter]
    fn num_cycles(&self) -> usize {
        self.pcs.len()
    }

    /// Register snapshot at a given cycle. Returns list of 16 values.
    fn regs_at(&self, cycle: usize) -> PyResult<Vec<u64>> {
        self.regs.get(cycle).cloned().ok_or_else(|| {
            PyValueError::new_err(format!(
                "cycle {} out of range (0..{})",
                cycle,
                self.pcs.len()
            ))
        })
    }

    /// Values of a single register across all cycles.
    fn reg_trace(&self, reg: usize) -> PyResult<Vec<u64>> {
        if reg >= 16 {
            return Err(PyValueError::new_err("Register index must be 0-15"));
        }
        Ok(self.regs.iter().map(|r| r[reg]).collect())
    }

    /// Zero flag at each cycle.
    #[getter]
    fn zero_flags(&self) -> Vec<bool> {
        self.flag_z.clone()
    }

    /// Carry flag at each cycle.
    #[getter]
    fn carry_flags(&self) -> Vec<bool> {
        self.flag_c.clone()
    }

    fn __repr__(&self) -> String {
        format!("V2Trace(cycles={})", self.pcs.len())
    }

    fn __len__(&self) -> usize {
        self.pcs.len()
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Assemble `source`, run it to completion, and return the ground-truth trace
/// as a JSONL string: a header record (schema version, program words, initial
/// register file, cycle/retired counts) followed by one machine-readable
/// record per retired instruction (PC, flags, register/RAM write deltas, MMIO
/// events). Deltas fold over the header's initial_regs to absolute state.
///
/// Raises ValueError on assembly errors, RuntimeError if the program does not
/// halt within `max_cycles`.
#[pyfunction]
#[pyo3(signature = (source, *, max_cycles=10000))]
fn v2_ground_truth_jsonl(source: &str, max_cycles: u64) -> PyResult<String> {
    let program =
        assemble_v2(source).map_err(|e| PyValueError::new_err(format!("Assembly error: {}", e)))?;
    capture_v2_ground_truth_jsonl(&program, max_cycles)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyV2Cpu>()?;
    m.add_class::<PyV2Trace>()?;
    m.add_function(wrap_pyfunction!(v2_ground_truth_jsonl, m)?)?;
    Ok(())
}
