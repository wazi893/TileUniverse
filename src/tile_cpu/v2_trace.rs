//! V2 instruction trace data model and deterministic serialization helpers.

use crate::tile_cpu::disassemble_v2_word;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V2TraceRegWrite {
    pub reg: u8,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V2TraceMemEvent {
    pub addr: u8,
    pub value: u64,
}

/// Column order for the flat CSV export ([`V2TraceLog::to_csv`] /
/// [`V2TraceEntry::to_csv_record`]). List-valued columns (`reg_writes`,
/// `ram_writes`, `mmio_reads`, `mmio_writes`) are flattened into a single
/// `;`-separated cell of `KEY=VALUE` pairs with decimal values; the optional
/// hybrid-evaluation counters are emitted as their decimal value or an empty
/// cell when absent.
pub const V2_TRACE_CSV_HEADER: &str = "cycle,retired,pc,ir_low,ir_ext,word,asm,flag_z,flag_c,reg_writes,ram_writes,mmio_reads,mmio_writes,stage_f_bank_switches,stage_f_mixed_dual_capture,stage_x_mixed_software,ram_high_bank_read_swaps,rom_upper_bank_group_select";

/// Schema version for the machine-readable JSONL trace dump.
///
/// Bump this whenever the field set or per-entry encoding of
/// [`V2TraceEntry::to_json`] changes so downstream interpretability tooling can
/// detect format drift before parsing bodies. Mirrors the manifest-versioning
/// convention in `v2_replay` (`V2_REPLAY_SCHEMA_VERSION`).
pub const V2_TRACE_SCHEMA_VERSION: u32 = 1;

/// Schema name embedded in the JSONL header ([`V2TraceLog::jsonl_header`]).
pub const V2_TRACE_SCHEMA_NAME: &str = "tileuniverse-v2-trace";

/// Ordered field list of every [`V2TraceEntry::to_json`] object, in emission
/// order. Kept in lockstep with `to_json`'s key order so the JSONL header is an
/// exact contract for downstream parsers.
pub const V2_TRACE_ENTRY_FIELDS: [&str; 18] = [
    "cycle",
    "retired",
    "pc",
    "ir_low",
    "ir_ext",
    "word",
    "asm",
    "flag_z",
    "flag_c",
    "reg_writes",
    "ram_writes",
    "mmio_reads",
    "mmio_writes",
    "stage_f_bank_switches",
    "stage_f_mixed_dual_capture",
    "stage_x_mixed_software",
    "ram_high_bank_read_swaps",
    "rom_upper_bank_group_select",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2TraceEntry {
    pub cycle: u64,
    pub retired: u64,
    /// Sprint 370 (Gate B.3): u32 to hold the wide PC.
    pub pc: u32,
    pub ir_low: u8,
    pub ir_ext: u16,
    pub word: u32,
    pub asm: String,
    pub flag_z: bool,
    pub flag_c: bool,
    pub reg_writes: Vec<V2TraceRegWrite>,
    pub ram_writes: Vec<V2TraceMemEvent>,
    pub mmio_reads: Vec<V2TraceMemEvent>,
    pub mmio_writes: Vec<V2TraceMemEvent>,
    pub stage_f_bank_switches: Option<u64>,
    pub stage_f_mixed_dual_capture: Option<u64>,
    pub stage_x_mixed_software: Option<u64>,
    pub ram_high_bank_read_swaps: Option<u64>,
    /// Sprint 187: ROM upper bank group selection (Banks 4-7) when PC >= 64.
    pub rom_upper_bank_group_select: Option<u64>,
}

impl V2TraceEntry {
    pub fn to_line(&self) -> String {
        let regs = fmt_reg_writes(&self.reg_writes);
        let ram = fmt_mem_events(&self.ram_writes);
        let mmio_r = fmt_mem_events(&self.mmio_reads);
        let mmio_w = fmt_mem_events(&self.mmio_writes);
        let asm = escape_trace_text(&self.asm);
        let z = if self.flag_z { 1 } else { 0 };
        let c = if self.flag_c { 1 } else { 0 };
        let hybrid = match (
            self.stage_f_bank_switches,
            self.stage_f_mixed_dual_capture,
            self.stage_x_mixed_software,
            self.ram_high_bank_read_swaps,
            self.rom_upper_bank_group_select,
        ) {
            (Some(f_switch), Some(f_mixed), Some(x_mixed), Some(ram_swap), Some(rom_upper)) => {
                format!(
                    "{{f_switch={f_switch},f_mixed={f_mixed},x_mixed={x_mixed},ram_hi_rd={ram_swap},rom_upper={rom_upper}}}"
                )
            }
            _ => "-".to_string(),
        };

        format!(
            "cycle={} retired={} pc=0x{:02X} word=0x{:08X} ir_low=0x{:02X} ir_ext=0x{:04X} asm=\"{}\" z={} c={} regs=[{}] ram=[{}] mmio_r=[{}] mmio_w=[{}] hybrid={}",
            self.cycle,
            self.retired,
            self.pc,
            self.word,
            self.ir_low,
            self.ir_ext,
            asm,
            z,
            c,
            regs,
            ram,
            mmio_r,
            mmio_w,
            hybrid
        )
    }

    /// Deterministic single-line JSON object for this trace entry.
    ///
    /// Companion to [`V2TraceEntry::to_line`]: where `to_line` is tuned for
    /// human reading, this emits a machine-parseable record for downstream
    /// tooling (e.g. interpretability ground-truth dumps). All numeric fields
    /// are decimal integers, booleans are `true`/`false`, and the optional
    /// hybrid-evaluation counters are emitted as their value or `null`. `u64`
    /// values are written in full decimal precision; consumers parsing with a
    /// 53-bit float JSON reader should treat them as integers.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"cycle\":{},\"retired\":{},\"pc\":{},\"ir_low\":{},\"ir_ext\":{},\"word\":{},\"asm\":\"{}\",\"flag_z\":{},\"flag_c\":{},\"reg_writes\":{},\"ram_writes\":{},\"mmio_reads\":{},\"mmio_writes\":{},\"stage_f_bank_switches\":{},\"stage_f_mixed_dual_capture\":{},\"stage_x_mixed_software\":{},\"ram_high_bank_read_swaps\":{},\"rom_upper_bank_group_select\":{}}}",
            self.cycle,
            self.retired,
            self.pc,
            self.ir_low,
            self.ir_ext,
            self.word,
            json_escape(&self.asm),
            self.flag_z,
            self.flag_c,
            json_reg_writes(&self.reg_writes),
            json_mem_events(&self.ram_writes),
            json_mem_events(&self.mmio_reads),
            json_mem_events(&self.mmio_writes),
            json_opt(self.stage_f_bank_switches),
            json_opt(self.stage_f_mixed_dual_capture),
            json_opt(self.stage_x_mixed_software),
            json_opt(self.ram_high_bank_read_swaps),
            json_opt(self.rom_upper_bank_group_select),
        )
    }

    /// Parse a single JSON object produced by [`V2TraceEntry::to_json`] back
    /// into a `V2TraceEntry`.
    ///
    /// Inverse of [`V2TraceEntry::to_json`]: `from_json(&e.to_json())` yields a
    /// value equal to `e` for every entry. Field lookup is key-based (order
    /// independent), so it tolerates any well-formed emission of the schema.
    /// Numeric fields are range-checked against their storage width; an
    /// out-of-range or missing field, a type mismatch, or trailing garbage all
    /// produce an `Err` rather than a silent truncation.
    pub fn from_json(input: &str) -> Result<Self, String> {
        let mut parser = JsonParser::new(input);
        let obj = parser.parse_value()?.into_object()?;
        parser.finish()?;

        Ok(Self {
            cycle: obj.u64("cycle")?,
            retired: obj.u64("retired")?,
            pc: obj.u32("pc")?,
            ir_low: obj.u8("ir_low")?,
            ir_ext: obj.u16("ir_ext")?,
            word: obj.u32("word")?,
            asm: obj.string("asm")?,
            flag_z: obj.bool("flag_z")?,
            flag_c: obj.bool("flag_c")?,
            reg_writes: obj.reg_writes("reg_writes")?,
            ram_writes: obj.mem_events("ram_writes")?,
            mmio_reads: obj.mem_events("mmio_reads")?,
            mmio_writes: obj.mem_events("mmio_writes")?,
            stage_f_bank_switches: obj.opt_u64("stage_f_bank_switches")?,
            stage_f_mixed_dual_capture: obj.opt_u64("stage_f_mixed_dual_capture")?,
            stage_x_mixed_software: obj.opt_u64("stage_x_mixed_software")?,
            ram_high_bank_read_swaps: obj.opt_u64("ram_high_bank_read_swaps")?,
            rom_upper_bank_group_select: obj.opt_u64("rom_upper_bank_group_select")?,
        })
    }

    /// Deterministic single CSV record (no trailing newline) for this entry,
    /// matching the column order of [`V2_TRACE_CSV_HEADER`].
    ///
    /// Companion to [`V2TraceEntry::to_json`]: where the JSON record nests the
    /// register/memory events as arrays, this flattens each list column into a
    /// single `;`-separated `KEY=VALUE` cell so the trace loads directly as a
    /// flat table (e.g. a pandas/numpy frame for interpretability probing). All
    /// numeric fields are decimal integers, flags are `0`/`1`, and absent
    /// hybrid counters become empty cells. Only the free-text `asm` column can
    /// require RFC 4180 quoting; the synthesized cells never contain commas.
    pub fn to_csv_record(&self) -> String {
        let z = if self.flag_z { 1 } else { 0 };
        let c = if self.flag_c { 1 } else { 0 };
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.cycle,
            self.retired,
            self.pc,
            self.ir_low,
            self.ir_ext,
            self.word,
            csv_escape(&self.asm),
            z,
            c,
            csv_reg_writes(&self.reg_writes),
            csv_mem_events(&self.ram_writes),
            csv_mem_events(&self.mmio_reads),
            csv_mem_events(&self.mmio_writes),
            csv_opt(self.stage_f_bank_switches),
            csv_opt(self.stage_f_mixed_dual_capture),
            csv_opt(self.stage_x_mixed_software),
            csv_opt(self.ram_high_bank_read_swaps),
            csv_opt(self.rom_upper_bank_group_select),
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct V2TraceLog {
    entries: Vec<V2TraceEntry>,
}

impl V2TraceLog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn push(&mut self, entry: V2TraceEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[V2TraceEntry] {
        &self.entries
    }

    pub fn to_lines(&self) -> String {
        self.entries
            .iter()
            .map(V2TraceEntry::to_line)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Deterministic JSON Lines dump: one [`V2TraceEntry::to_json`] object per
    /// line, in cycle order. Suitable as a labeled ground-truth trace for
    /// downstream analysis tooling.
    pub fn to_jsonl(&self) -> String {
        self.entries
            .iter()
            .map(V2TraceEntry::to_json)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extract the ground-truth stream of MMIO writes to a single output port.
    ///
    /// Returns `(cycle, value)` pairs in execution order — one per write to
    /// `port` across the whole trace — giving interpretability tooling a clean
    /// per-step "OUT" label sequence for a program without re-scanning the full
    /// per-cycle records. Multiple writes to `port` within one cycle are kept
    /// in their original order, each tagged with that cycle.
    pub fn mmio_write_stream(&self, port: u8) -> Vec<(u64, u64)> {
        self.entries
            .iter()
            .flat_map(|e| {
                e.mmio_writes
                    .iter()
                    .filter(move |w| w.addr == port)
                    .map(move |w| (e.cycle, w.value))
            })
            .collect()
    }

    /// Extract the ground-truth value series written to a single register.
    ///
    /// Companion to [`V2TraceLog::mmio_write_stream`] for the datapath side:
    /// returns `(cycle, value)` pairs for every write to `reg`, in execution
    /// order, so a probe target's evolution over the run is available as a
    /// labeled sequence. Cycles where `reg` is not written are omitted.
    pub fn reg_write_stream(&self, reg: u8) -> Vec<(u64, u64)> {
        self.entries
            .iter()
            .flat_map(|e| {
                e.reg_writes
                    .iter()
                    .filter(move |w| w.reg == reg)
                    .map(move |w| (e.cycle, w.value))
            })
            .collect()
    }

    /// Reconstruct the **absolute** architectural state after each retired step
    /// by folding the per-cycle delta events ([`V2TraceEntry::reg_writes`] and
    /// [`V2TraceEntry::ram_writes`]) over an initial machine state.
    ///
    /// The trace itself records *deltas* (the writes a cycle performed), which
    /// is compact but awkward for interpretability ground truth, where a probe
    /// typically wants the full state vector — all 16 registers, the 128-word
    /// RAM, the flags, and the PC — present at every step. This helper produces
    /// exactly that: one [`V2TraceState`] per entry, in cycle order, holding the
    /// state *after* the entry's writes are applied. Flags and PC are copied
    /// straight from the entry (they are already absolute).
    ///
    /// `initial_regs` / `initial_ram` seed the fold; pass the machine's reset
    /// state (all zeros) via [`V2TraceLog::reconstruct_states_from_zero`] for
    /// the common case. Out-of-range register or RAM indices in the trace are
    /// ignored defensively rather than panicking.
    pub fn reconstruct_states(
        &self,
        initial_regs: [u64; 16],
        initial_ram: [u64; 128],
    ) -> Vec<V2TraceState> {
        let mut regs = initial_regs;
        let mut ram = initial_ram;
        let mut out = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            for w in &entry.reg_writes {
                if let Some(slot) = regs.get_mut(w.reg as usize) {
                    *slot = w.value;
                }
            }
            for w in &entry.ram_writes {
                if let Some(slot) = ram.get_mut(w.addr as usize) {
                    *slot = w.value;
                }
            }
            out.push(V2TraceState {
                cycle: entry.cycle,
                retired: entry.retired,
                pc: entry.pc,
                flag_z: entry.flag_z,
                flag_c: entry.flag_c,
                regs,
                ram,
            });
        }
        out
    }

    /// [`V2TraceLog::reconstruct_states`] seeded from the machine reset state
    /// (registers and RAM all zero), matching the V2 ISS power-on state.
    pub fn reconstruct_states_from_zero(&self) -> Vec<V2TraceState> {
        self.reconstruct_states([0u64; 16], [0u64; 128])
    }

    /// Deterministic JSON Lines dump of the reconstructed absolute states: one
    /// [`V2TraceState::to_json`] object per line, in cycle order. Suitable as a
    /// labeled ground-truth dump where each line is the full machine state at a
    /// step. See [`V2TraceLog::reconstruct_states`] for the seed arguments.
    pub fn to_state_jsonl(&self, initial_regs: [u64; 16], initial_ram: [u64; 128]) -> String {
        self.reconstruct_states(initial_regs, initial_ram)
            .iter()
            .map(V2TraceState::to_json)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Parse a JSON Lines dump produced by [`V2TraceLog::to_jsonl`] back into a
    /// `V2TraceLog`.
    ///
    /// Inverse of [`V2TraceLog::to_jsonl`]: `from_jsonl(&log.to_jsonl())` yields
    /// a value equal to `log`. Blank lines (including the empty string from an
    /// empty log) are skipped; any line that fails to parse aborts with an
    /// `Err` naming the 1-based line number.
    pub fn from_jsonl(input: &str) -> Result<Self, String> {
        let mut log = Self::new();
        for (idx, line) in input.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let entry =
                V2TraceEntry::from_json(line).map_err(|e| format!("line {}: {e}", idx + 1))?;
            log.push(entry);
        }
        Ok(log)
    }

    /// Deterministic CSV document: the [`V2_TRACE_CSV_HEADER`] row followed by
    /// one [`V2TraceEntry::to_csv_record`] per entry, in cycle order. Suitable
    /// as a flat, tabular ground-truth trace for downstream analysis tooling
    /// (one row per retired step, scalar architectural state per column).
    pub fn to_csv(&self) -> String {
        let mut out = String::from(V2_TRACE_CSV_HEADER);
        for entry in &self.entries {
            out.push('\n');
            out.push_str(&entry.to_csv_record());
        }
        out
    }

    /// Deterministic self-describing header line for a JSONL trace dump.
    ///
    /// A single JSON object carrying the schema name, [`V2_TRACE_SCHEMA_VERSION`],
    /// the ordered per-entry field list ([`V2_TRACE_ENTRY_FIELDS`]), and the
    /// record count. A downstream interpretability reader can validate the field
    /// contract and record count before parsing bodies. Emitted as the first line
    /// of [`to_jsonl_with_header`](Self::to_jsonl_with_header).
    pub fn jsonl_header(&self) -> String {
        let fields = V2_TRACE_ENTRY_FIELDS
            .iter()
            .map(|f| format!("\"{f}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":\"{}\",\"version\":{},\"fields\":[{}],\"count\":{}}}",
            V2_TRACE_SCHEMA_NAME,
            V2_TRACE_SCHEMA_VERSION,
            fields,
            self.entries.len()
        )
    }

    /// Self-describing JSON Lines dump: [`jsonl_header`](Self::jsonl_header) on the
    /// first line, followed by the [`to_jsonl`](Self::to_jsonl) body. For an empty
    /// log the result is the header line alone (no trailing newline), so the output
    /// is always at least the validatable header.
    pub fn to_jsonl_with_header(&self) -> String {
        let header = self.jsonl_header();
        if self.entries.is_empty() {
            header
        } else {
            format!("{}\n{}", header, self.to_jsonl())
        }
    }
    /// Parse a multi-line trace block produced by [`V2TraceLog::to_lines`] back
    /// into a structured log. Blank lines (e.g. a trailing newline) are
    /// ignored, so `from_lines(&log.to_lines()) == Ok(log)`.
    pub fn from_lines(text: &str) -> Result<Self, V2TraceParseError> {
        let mut log = Self::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            log.push(V2TraceEntry::from_line(line)?);
        }
        Ok(log)
    }
}

/// Absolute architectural snapshot reconstructed at a single retired step.
///
/// Produced by [`V2TraceLog::reconstruct_states`]: where [`V2TraceEntry`]
/// records the *delta* of a cycle, this holds the complete state *after* that
/// cycle — the full register file, RAM, flags, and PC — for downstream
/// interpretability ground-truth tooling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2TraceState {
    pub cycle: u64,
    pub retired: u64,
    /// Sprint 370 (Gate B.3): u32 to hold the wide PC.
    pub pc: u32,
    pub flag_z: bool,
    pub flag_c: bool,
    pub regs: [u64; 16],
    pub ram: [u64; 128],
}

impl V2TraceState {
    /// Deterministic single-line JSON object for this absolute snapshot.
    ///
    /// Mirrors the conventions of [`V2TraceEntry::to_json`]: decimal integers,
    /// `true`/`false` booleans, and the register/RAM files as decimal arrays in
    /// index order. `u64` values are written in full decimal precision.
    pub fn to_json(&self) -> String {
        let regs = self
            .regs
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let ram = self
            .ram
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"cycle\":{},\"retired\":{},\"pc\":{},\"flag_z\":{},\"flag_c\":{},\"regs\":[{}],\"ram\":[{}]}}",
            self.cycle, self.retired, self.pc, self.flag_z, self.flag_c, regs, ram
        )
    }
}

/// A single decoded ROM row: its load address, raw 32-bit word, and disassembly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2TraceProgramRow {
    pub addr: u32,
    pub word: u32,
    pub asm: String,
}

/// Program-image header for a self-describing trace artifact.
///
/// A bare [`V2TraceLog`] records PC/flags/registers/OUT per step but carries no
/// context about *which program* produced it. `V2TraceProgram` bundles the ROM
/// image (raw words + disassembly) and a few scalar ground-truth labels so a
/// downstream interpretability consumer receives *program + per-step trace* as a
/// single deterministic document (see [`V2TraceDocument`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct V2TraceProgram {
    /// Optional human-readable label for ground-truth dumps (e.g. the case name).
    pub label: Option<String>,
    /// ROM base address the program image is loaded at.
    pub origin: u32,
    /// Configured ROM capacity in words (>= program length); metadata for the
    /// consumer, independent of how many rows are actually present.
    pub rom_size: usize,
    /// Decoded program rows in address order.
    pub rows: Vec<V2TraceProgramRow>,
}

impl V2TraceProgram {
    /// Build a header from a program image loaded at `origin`, disassembling
    /// each word with [`disassemble_v2_word`]. `rom_size` records the configured
    /// ROM capacity in words (typically `>= words.len()`).
    pub fn from_words(label: Option<String>, origin: u32, rom_size: usize, words: &[u32]) -> Self {
        let rows = words
            .iter()
            .enumerate()
            .map(|(i, &word)| V2TraceProgramRow {
                addr: origin + i as u32,
                word,
                asm: disassemble_v2_word(word),
            })
            .collect();
        Self {
            label,
            origin,
            rom_size,
            rows,
        }
    }

    /// Human-readable header block: a `; program …` summary comment followed by
    /// one `0xADDR: 0xWORD  ASM` line per row, in address order.
    pub fn to_text(&self) -> String {
        let label = match &self.label {
            Some(l) => format!("\"{}\"", escape_trace_text(l)),
            None => "-".to_string(),
        };
        let mut out = format!(
            "; program label={} origin=0x{:02X} rom_size={} words={}",
            label,
            self.origin,
            self.rom_size,
            self.rows.len()
        );
        for row in &self.rows {
            out.push_str(&format!(
                "\n0x{:02X}: 0x{:08X}  {}",
                row.addr,
                row.word,
                escape_trace_text(&row.asm)
            ));
        }
        out
    }

    /// Deterministic single-line JSON object for this program header.
    ///
    /// Identifiable in a mixed [`V2TraceDocument::to_jsonl`] stream by its
    /// `"rows"` key (per-step entries carry a `"cycle"` key instead). `label` is
    /// emitted as a JSON string or `null`; addresses and words are decimal
    /// integers.
    pub fn to_json(&self) -> String {
        let rows = self
            .rows
            .iter()
            .map(|r| {
                format!(
                    "{{\"addr\":{},\"word\":{},\"asm\":\"{}\"}}",
                    r.addr,
                    r.word,
                    json_escape(&r.asm)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"label\":{},\"origin\":{},\"rom_size\":{},\"rows\":[{}]}}",
            json_opt_str(self.label.as_deref()),
            self.origin,
            self.rom_size,
            rows
        )
    }
}

/// A self-describing trace artifact: a program header plus its per-step trace.
///
/// This is the top-level ground-truth export for the interpretability testbed —
/// serializing it yields *program + PC/flags/registers/OUT per step* in one
/// deterministic blob, so a consumer never has to reunite a trace with the ROM
/// that produced it out of band.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct V2TraceDocument {
    pub program: V2TraceProgram,
    pub log: V2TraceLog,
}

impl V2TraceDocument {
    pub fn new(program: V2TraceProgram, log: V2TraceLog) -> Self {
        Self { program, log }
    }

    /// Human-readable text: the [`V2TraceProgram::to_text`] header block, a blank
    /// separator line, then the [`V2TraceLog::to_lines`] step trace.
    pub fn to_text(&self) -> String {
        format!("{}\n\n{}", self.program.to_text(), self.log.to_lines())
    }

    /// Deterministic JSON Lines dump: the first line is the program header
    /// (identifiable by its `"rows"` key), followed by one per-step entry object
    /// (each identifiable by its `"cycle"` key) in cycle order.
    pub fn to_jsonl(&self) -> String {
        let log = self.log.to_jsonl();
        if log.is_empty() {
            self.program.to_json()
        } else {
            format!("{}\n{}", self.program.to_json(), log)
        }
    }
}

fn json_opt_str(value: Option<&str>) -> String {
    match value {
        Some(s) => format!("\"{}\"", json_escape(s)),
        None => "null".to_string(),
    }
}

/// Error returned when a serialized trace line cannot be parsed back into a
/// [`V2TraceEntry`]. The message names the offending field for debuggability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2TraceParseError(pub String);

impl std::fmt::Display for V2TraceParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v2 trace parse error: {}", self.0)
    }
}

impl std::error::Error for V2TraceParseError {}

impl V2TraceEntry {
    /// Parse a single line produced by [`V2TraceEntry::to_line`] back into a
    /// structured entry. This is the exact inverse of `to_line`: for any
    /// entry `e`, `V2TraceEntry::from_line(&e.to_line()) == Ok(e)`.
    ///
    /// Useful for differential / interpretability tooling that consumes the
    /// deterministic text trace as ground-truth per-step records and needs to
    /// recover the structured fields (PC, flags, register/RAM/MMIO events).
    pub fn from_line(line: &str) -> Result<Self, V2TraceParseError> {
        let line = line.trim_end_matches(['\n', '\r']);

        // The `asm` field is the only one that may contain spaces or quotes, so
        // isolate it first: everything before `asm="` is a fixed-format prefix
        // and everything after its closing quote is a fixed-format suffix.
        const ASM_MARKER: &str = "asm=\"";
        let asm_start = line
            .find(ASM_MARKER)
            .ok_or_else(|| V2TraceParseError("missing asm field".to_string()))?;
        let prefix = line[..asm_start].trim_end();
        let after_open = &line[asm_start + ASM_MARKER.len()..];

        // Decode the quoted asm, honoring the escapes from `escape_trace_text`,
        // and record the byte offset just past the closing (unescaped) quote.
        let mut asm = String::new();
        let mut suffix_start = None;
        let mut chars = after_open.char_indices();
        while let Some((idx, ch)) = chars.next() {
            match ch {
                '\\' => {
                    let (_, esc) = chars.next().ok_or_else(|| {
                        V2TraceParseError("trailing backslash in asm".to_string())
                    })?;
                    match esc {
                        '\\' => asm.push('\\'),
                        '"' => asm.push('"'),
                        'n' => asm.push('\n'),
                        'r' => asm.push('\r'),
                        't' => asm.push('\t'),
                        other => {
                            return Err(V2TraceParseError(format!(
                                "invalid escape '\\{other}' in asm"
                            )));
                        }
                    }
                }
                '"' => {
                    suffix_start = Some(idx + ch.len_utf8());
                    break;
                }
                _ => asm.push(ch),
            }
        }
        let suffix_start =
            suffix_start.ok_or_else(|| V2TraceParseError("unterminated asm quote".to_string()))?;
        let suffix = after_open[suffix_start..].trim_start();

        // Prefix: cycle / retired / pc / word / ir_low / ir_ext (no spaces).
        let pre: Vec<&str> = prefix.split_whitespace().collect();
        if pre.len() != 6 {
            return Err(V2TraceParseError(format!(
                "expected 6 prefix fields, got {}",
                pre.len()
            )));
        }
        let cycle = parse_dec_u64(strip_kv(pre[0], "cycle=")?)?;
        let retired = parse_dec_u64(strip_kv(pre[1], "retired=")?)?;
        let pc = parse_hex(strip_kv(pre[2], "pc=")?)? as u32;
        let word = parse_hex(strip_kv(pre[3], "word=")?)? as u32;
        let ir_low = parse_hex(strip_kv(pre[4], "ir_low=")?)? as u8;
        let ir_ext = parse_hex(strip_kv(pre[5], "ir_ext=")?)? as u16;

        // Suffix: z / c / regs / ram / mmio_r / mmio_w / hybrid (no spaces).
        let post: Vec<&str> = suffix.split_whitespace().collect();
        if post.len() != 7 {
            return Err(V2TraceParseError(format!(
                "expected 7 suffix fields, got {}",
                post.len()
            )));
        }
        let flag_z = parse_flag(strip_kv(post[0], "z=")?)?;
        let flag_c = parse_flag(strip_kv(post[1], "c=")?)?;
        let reg_writes = parse_reg_writes(parse_bracketed(post[2], "regs=")?)?;
        let ram_writes = parse_mem_events(parse_bracketed(post[3], "ram=")?)?;
        let mmio_reads = parse_mem_events(parse_bracketed(post[4], "mmio_r=")?)?;
        let mmio_writes = parse_mem_events(parse_bracketed(post[5], "mmio_w=")?)?;
        let hybrid = strip_kv(post[6], "hybrid=")?;

        let (
            stage_f_bank_switches,
            stage_f_mixed_dual_capture,
            stage_x_mixed_software,
            ram_high_bank_read_swaps,
            rom_upper_bank_group_select,
        ) = if hybrid == "-" {
            (None, None, None, None, None)
        } else {
            let body = hybrid
                .strip_prefix('{')
                .and_then(|b| b.strip_suffix('}'))
                .ok_or_else(|| V2TraceParseError(format!("malformed hybrid '{hybrid}'")))?;
            let parts: Vec<&str> = body.split(',').collect();
            if parts.len() != 5 {
                return Err(V2TraceParseError(format!(
                    "expected 5 hybrid fields, got {}",
                    parts.len()
                )));
            }
            (
                Some(parse_dec_u64(strip_kv(parts[0], "f_switch=")?)?),
                Some(parse_dec_u64(strip_kv(parts[1], "f_mixed=")?)?),
                Some(parse_dec_u64(strip_kv(parts[2], "x_mixed=")?)?),
                Some(parse_dec_u64(strip_kv(parts[3], "ram_hi_rd=")?)?),
                Some(parse_dec_u64(strip_kv(parts[4], "rom_upper=")?)?),
            )
        };

        Ok(V2TraceEntry {
            cycle,
            retired,
            pc,
            ir_low,
            ir_ext,
            word,
            asm,
            flag_z,
            flag_c,
            reg_writes,
            ram_writes,
            mmio_reads,
            mmio_writes,
            stage_f_bank_switches,
            stage_f_mixed_dual_capture,
            stage_x_mixed_software,
            ram_high_bank_read_swaps,
            rom_upper_bank_group_select,
        })
    }
}

fn strip_kv<'a>(token: &'a str, key: &str) -> Result<&'a str, V2TraceParseError> {
    token
        .strip_prefix(key)
        .ok_or_else(|| V2TraceParseError(format!("expected field '{key}', got '{token}'")))
}

fn parse_dec_u64(raw: &str) -> Result<u64, V2TraceParseError> {
    raw.parse::<u64>()
        .map_err(|e| V2TraceParseError(format!("invalid integer '{raw}': {e}")))
}

fn parse_hex(raw: &str) -> Result<u64, V2TraceParseError> {
    let digits = raw
        .strip_prefix("0x")
        .ok_or_else(|| V2TraceParseError(format!("expected 0x-prefixed hex, got '{raw}'")))?;
    u64::from_str_radix(digits, 16)
        .map_err(|e| V2TraceParseError(format!("invalid hex '{raw}': {e}")))
}

fn parse_flag(raw: &str) -> Result<bool, V2TraceParseError> {
    match raw {
        "0" => Ok(false),
        "1" => Ok(true),
        other => Err(V2TraceParseError(format!("invalid flag '{other}'"))),
    }
}

fn parse_bracketed<'a>(token: &'a str, key: &str) -> Result<&'a str, V2TraceParseError> {
    let body = strip_kv(token, key)?;
    body.strip_prefix('[')
        .and_then(|b| b.strip_suffix(']'))
        .ok_or_else(|| {
            V2TraceParseError(format!("expected '[...]' for field '{key}', got '{token}'"))
        })
}

fn parse_reg_writes(body: &str) -> Result<Vec<V2TraceRegWrite>, V2TraceParseError> {
    if body.is_empty() {
        return Ok(Vec::new());
    }
    body.split(',')
        .map(|item| {
            let (reg, value) = item
                .split_once('=')
                .ok_or_else(|| V2TraceParseError(format!("malformed reg write '{item}'")))?;
            let reg = reg
                .strip_prefix('R')
                .ok_or_else(|| V2TraceParseError(format!("expected 'R' prefix in '{item}'")))?
                .parse::<u8>()
                .map_err(|e| V2TraceParseError(format!("invalid reg index '{item}': {e}")))?;
            Ok(V2TraceRegWrite {
                reg,
                value: parse_hex(value)?,
            })
        })
        .collect()
}

fn parse_mem_events(body: &str) -> Result<Vec<V2TraceMemEvent>, V2TraceParseError> {
    if body.is_empty() {
        return Ok(Vec::new());
    }
    body.split(',')
        .map(|item| {
            let (addr, value) = item
                .split_once('=')
                .ok_or_else(|| V2TraceParseError(format!("malformed mem event '{item}'")))?;
            Ok(V2TraceMemEvent {
                addr: parse_hex(addr)? as u8,
                value: parse_hex(value)?,
            })
        })
        .collect()
}

/// Flatten register writes into a single CSV cell: `;`-separated `R<reg>=<value>`
/// pairs with decimal values. Empty input yields an empty cell.
fn csv_reg_writes(items: &[V2TraceRegWrite]) -> String {
    items
        .iter()
        .map(|w| format!("R{}={}", w.reg, w.value))
        .collect::<Vec<_>>()
        .join(";")
}

/// Flatten memory events into a single CSV cell: `;`-separated `<addr>=<value>`
/// pairs with decimal values. Empty input yields an empty cell.
fn csv_mem_events(items: &[V2TraceMemEvent]) -> String {
    items
        .iter()
        .map(|w| format!("{}={}", w.addr, w.value))
        .collect::<Vec<_>>()
        .join(";")
}

/// Optional hybrid counter as a CSV cell: decimal value, or empty when absent.
fn csv_opt(value: Option<u64>) -> String {
    match value {
        Some(n) => n.to_string(),
        None => String::new(),
    }
}

/// RFC 4180 field escaping: a field containing a comma, double quote, CR, or LF
/// is wrapped in double quotes with any interior quote doubled. Other fields are
/// returned unchanged.
fn csv_escape(input: &str) -> String {
    if input.contains([',', '"', '\n', '\r']) {
        let mut out = String::with_capacity(input.len() + 2);
        out.push('"');
        for ch in input.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
        out
    } else {
        input.to_string()
    }
}

fn fmt_reg_writes(items: &[V2TraceRegWrite]) -> String {
    items
        .iter()
        .map(|w| format!("R{}=0x{:016X}", w.reg, w.value))
        .collect::<Vec<_>>()
        .join(",")
}

fn fmt_mem_events(items: &[V2TraceMemEvent]) -> String {
    items
        .iter()
        .map(|w| format!("0x{:02X}=0x{:016X}", w.addr, w.value))
        .collect::<Vec<_>>()
        .join(",")
}

fn json_reg_writes(items: &[V2TraceRegWrite]) -> String {
    let inner = items
        .iter()
        .map(|w| format!("{{\"reg\":{},\"value\":{}}}", w.reg, w.value))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{inner}]")
}

fn json_mem_events(items: &[V2TraceMemEvent]) -> String {
    let inner = items
        .iter()
        .map(|w| format!("{{\"addr\":{},\"value\":{}}}", w.addr, w.value))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{inner}]")
}

fn json_opt(value: Option<u64>) -> String {
    match value {
        Some(n) => n.to_string(),
        None => "null".to_string(),
    }
}

/// JSON string-body escaping (RFC 8259): the same control characters handled by
/// [`escape_trace_text`] plus `\u00XX` for any remaining C0 control character.
fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn escape_trace_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Minimal JSON reader (module-private)
//
// A small std-only recursive-descent parser scoped to reading back the trace
// records emitted by `V2TraceEntry::to_json`. It is intentionally not a general
// JSON library: it accepts the subset this module produces (objects, arrays,
// strings, booleans, `null`, and non-negative decimal integers) and rejects
// everything else. Keeping it self-contained avoids pulling `serde` into the
// core engine crate just for the round-trip loader.
// ---------------------------------------------------------------------------

/// A parsed JSON value from the trace subset.
enum Json {
    Null,
    Bool(bool),
    Num(u64),
    Str(String),
    Arr(Vec<Json>),
    Obj(JsonObject),
}

impl Json {
    fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "boolean",
            Json::Num(_) => "integer",
            Json::Str(_) => "string",
            Json::Arr(_) => "array",
            Json::Obj(_) => "object",
        }
    }

    fn into_object(self) -> Result<JsonObject, String> {
        match self {
            Json::Obj(obj) => Ok(obj),
            other => Err(format!("expected object, got {}", other.kind())),
        }
    }

    fn as_object(&self) -> Result<&JsonObject, String> {
        match self {
            Json::Obj(obj) => Ok(obj),
            other => Err(format!("expected object, got {}", other.kind())),
        }
    }
}

/// A JSON object as an ordered list of key/value pairs, with typed field
/// accessors that range-check against the trace schema's storage widths.
struct JsonObject {
    fields: Vec<(String, Json)>,
}

impl JsonObject {
    fn get(&self, key: &str) -> Result<&Json, String> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
            .ok_or_else(|| format!("missing field \"{key}\""))
    }

    fn u64(&self, key: &str) -> Result<u64, String> {
        match self.get(key)? {
            Json::Num(n) => Ok(*n),
            other => Err(format!(
                "field \"{key}\": expected integer, got {}",
                other.kind()
            )),
        }
    }

    fn u32(&self, key: &str) -> Result<u32, String> {
        let n = self.u64(key)?;
        u32::try_from(n).map_err(|_| format!("field \"{key}\": {n} out of range for u32"))
    }

    fn u16(&self, key: &str) -> Result<u16, String> {
        let n = self.u64(key)?;
        u16::try_from(n).map_err(|_| format!("field \"{key}\": {n} out of range for u16"))
    }

    fn u8(&self, key: &str) -> Result<u8, String> {
        let n = self.u64(key)?;
        u8::try_from(n).map_err(|_| format!("field \"{key}\": {n} out of range for u8"))
    }

    fn bool(&self, key: &str) -> Result<bool, String> {
        match self.get(key)? {
            Json::Bool(b) => Ok(*b),
            other => Err(format!(
                "field \"{key}\": expected boolean, got {}",
                other.kind()
            )),
        }
    }

    fn string(&self, key: &str) -> Result<String, String> {
        match self.get(key)? {
            Json::Str(s) => Ok(s.clone()),
            other => Err(format!(
                "field \"{key}\": expected string, got {}",
                other.kind()
            )),
        }
    }

    fn opt_u64(&self, key: &str) -> Result<Option<u64>, String> {
        match self.get(key)? {
            Json::Null => Ok(None),
            Json::Num(n) => Ok(Some(*n)),
            other => Err(format!(
                "field \"{key}\": expected integer or null, got {}",
                other.kind()
            )),
        }
    }

    fn reg_writes(&self, key: &str) -> Result<Vec<V2TraceRegWrite>, String> {
        self.array(key)?
            .iter()
            .map(|v| {
                let o = v.as_object()?;
                Ok(V2TraceRegWrite {
                    reg: o.u8("reg")?,
                    value: o.u64("value")?,
                })
            })
            .collect()
    }

    fn mem_events(&self, key: &str) -> Result<Vec<V2TraceMemEvent>, String> {
        self.array(key)?
            .iter()
            .map(|v| {
                let o = v.as_object()?;
                Ok(V2TraceMemEvent {
                    addr: o.u8("addr")?,
                    value: o.u64("value")?,
                })
            })
            .collect()
    }

    fn array(&self, key: &str) -> Result<&[Json], String> {
        match self.get(key)? {
            Json::Arr(items) => Ok(items),
            other => Err(format!(
                "field \"{key}\": expected array, got {}",
                other.kind()
            )),
        }
    }
}

struct JsonParser {
    chars: Vec<char>,
    pos: usize,
}

impl JsonParser {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, want: char) -> Result<(), String> {
        self.skip_ws();
        match self.bump() {
            Some(c) if c == want => Ok(()),
            other => Err(format!("expected '{want}', got {other:?}")),
        }
    }

    /// Consume trailing whitespace and require end-of-input (no trailing junk).
    fn finish(&mut self) -> Result<(), String> {
        self.skip_ws();
        match self.peek() {
            None => Ok(()),
            Some(c) => Err(format!("trailing characters after value (found {c:?})")),
        }
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(Json::Str(self.parse_string()?)),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c.is_ascii_digit() => self.parse_number(),
            other => Err(format!("unexpected token {other:?}")),
        }
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.expect('{')?;
        let mut fields = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(Json::Obj(JsonObject { fields }));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.expect(':')?;
            let value = self.parse_value()?;
            fields.push((key, value));
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some('}') => break,
                other => return Err(format!("expected ',' or '}}' in object, got {other:?}")),
            }
        }
        Ok(Json::Obj(JsonObject { fields }))
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some(']') => break,
                other => return Err(format!("expected ',' or ']' in array, got {other:?}")),
            }
        }
        Ok(Json::Arr(items))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err("unterminated string".to_string()),
                Some('"') => break,
                Some('\\') => {
                    let esc = self.bump().ok_or("unterminated escape sequence")?;
                    match esc {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'b' => out.push('\u{08}'),
                        'f' => out.push('\u{0C}'),
                        'u' => out.push(self.parse_unicode_escape()?),
                        other => return Err(format!("invalid escape '\\{other}'")),
                    }
                }
                Some(c) => out.push(c),
            }
        }
        Ok(out)
    }

    fn parse_unicode_escape(&mut self) -> Result<char, String> {
        let mut code: u32 = 0;
        for _ in 0..4 {
            let c = self.bump().ok_or("truncated \\u escape")?;
            let digit = c
                .to_digit(16)
                .ok_or_else(|| format!("invalid hex digit '{c}' in \\u escape"))?;
            code = code * 16 + digit;
        }
        char::from_u32(code).ok_or_else(|| format!("invalid unicode code point U+{code:04X}"))
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let mut value: u64 = 0;
        let mut digits = 0;
        while let Some(c) = self.peek() {
            if let Some(d) = c.to_digit(10) {
                value = value
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(d as u64))
                    .ok_or("integer overflow parsing number")?;
                digits += 1;
                self.pos += 1;
            } else {
                break;
            }
        }
        if digits == 0 {
            return Err("expected digit".to_string());
        }
        Ok(Json::Num(value))
    }

    fn parse_bool(&mut self) -> Result<Json, String> {
        if self.consume_literal("true") {
            Ok(Json::Bool(true))
        } else if self.consume_literal("false") {
            Ok(Json::Bool(false))
        } else {
            Err("invalid literal (expected true/false)".to_string())
        }
    }

    fn parse_null(&mut self) -> Result<Json, String> {
        if self.consume_literal("null") {
            Ok(Json::Null)
        } else {
            Err("invalid literal (expected null)".to_string())
        }
    }

    fn consume_literal(&mut self, lit: &str) -> bool {
        let end = self.pos + lit.chars().count();
        if end <= self.chars.len() && self.chars[self.pos..end].iter().copied().eq(lit.chars()) {
            self.pos = end;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_trace_entry_line_format_stable() {
        let entry = V2TraceEntry {
            cycle: 7,
            retired: 3,
            pc: 12,
            ir_low: 0x2A,
            ir_ext: 0x0001,
            word: 0x0001_182A,
            asm: "LDI R8, 42".to_string(),
            flag_z: false,
            flag_c: true,
            reg_writes: vec![V2TraceRegWrite { reg: 8, value: 42 }],
            ram_writes: vec![V2TraceMemEvent {
                addr: 0x21,
                value: 0xAA,
            }],
            mmio_reads: Vec::new(),
            mmio_writes: vec![V2TraceMemEvent {
                addr: 0x40,
                value: 0xAA,
            }],
            stage_f_bank_switches: Some(5),
            stage_f_mixed_dual_capture: Some(2),
            stage_x_mixed_software: Some(2),
            ram_high_bank_read_swaps: Some(4),
            rom_upper_bank_group_select: Some(1),
        };

        assert_eq!(
            entry.to_line(),
            "cycle=7 retired=3 pc=0x0C word=0x0001182A ir_low=0x2A ir_ext=0x0001 asm=\"LDI R8, 42\" z=0 c=1 regs=[R8=0x000000000000002A] ram=[0x21=0x00000000000000AA] mmio_r=[] mmio_w=[0x40=0x00000000000000AA] hybrid={f_switch=5,f_mixed=2,x_mixed=2,ram_hi_rd=4,rom_upper=1}"
        );
    }

    #[test]
    fn test_v2_trace_log_lines_join() {
        let mut log = V2TraceLog::new();
        log.push(V2TraceEntry {
            cycle: 1,
            retired: 0,
            pc: 0,
            ir_low: 0,
            ir_ext: 0,
            word: 0,
            asm: "NOP".to_string(),
            flag_z: false,
            flag_c: false,
            reg_writes: Vec::new(),
            ram_writes: Vec::new(),
            mmio_reads: Vec::new(),
            mmio_writes: Vec::new(),
            stage_f_bank_switches: None,
            stage_f_mixed_dual_capture: None,
            stage_x_mixed_software: None,
            ram_high_bank_read_swaps: None,
            rom_upper_bank_group_select: None,
        });
        log.push(V2TraceEntry {
            cycle: 2,
            retired: 1,
            pc: 1,
            ir_low: 0,
            ir_ext: 0,
            word: 0x0000_0800,
            asm: "HALT".to_string(),
            flag_z: false,
            flag_c: false,
            reg_writes: Vec::new(),
            ram_writes: Vec::new(),
            mmio_reads: Vec::new(),
            mmio_writes: Vec::new(),
            stage_f_bank_switches: None,
            stage_f_mixed_dual_capture: None,
            stage_x_mixed_software: None,
            ram_high_bank_read_swaps: None,
            rom_upper_bank_group_select: None,
        });

        let lines = log.to_lines();
        let split: Vec<&str> = lines.lines().collect();
        assert_eq!(split.len(), 2);
        assert!(split[0].contains("asm=\"NOP\""));
        assert!(split[1].contains("asm=\"HALT\""));
    }

    #[test]
    fn test_v2_trace_entry_json_format_stable() {
        let entry = V2TraceEntry {
            cycle: 7,
            retired: 3,
            pc: 12,
            ir_low: 0x2A,
            ir_ext: 0x0001,
            word: 0x0001_182A,
            asm: "LDI R8, 42".to_string(),
            flag_z: false,
            flag_c: true,
            reg_writes: vec![V2TraceRegWrite { reg: 8, value: 42 }],
            ram_writes: vec![V2TraceMemEvent {
                addr: 0x21,
                value: 0xAA,
            }],
            mmio_reads: Vec::new(),
            mmio_writes: vec![V2TraceMemEvent {
                addr: 0x40,
                value: 0xAA,
            }],
            stage_f_bank_switches: Some(5),
            stage_f_mixed_dual_capture: Some(2),
            stage_x_mixed_software: Some(2),
            ram_high_bank_read_swaps: Some(4),
            rom_upper_bank_group_select: Some(1),
        };

        assert_eq!(
            entry.to_json(),
            "{\"cycle\":7,\"retired\":3,\"pc\":12,\"ir_low\":42,\"ir_ext\":1,\"word\":71722,\"asm\":\"LDI R8, 42\",\"flag_z\":false,\"flag_c\":true,\"reg_writes\":[{\"reg\":8,\"value\":42}],\"ram_writes\":[{\"addr\":33,\"value\":170}],\"mmio_reads\":[],\"mmio_writes\":[{\"addr\":64,\"value\":170}],\"stage_f_bank_switches\":5,\"stage_f_mixed_dual_capture\":2,\"stage_x_mixed_software\":2,\"ram_high_bank_read_swaps\":4,\"rom_upper_bank_group_select\":1}"
        );
    }

    #[test]
    fn test_v2_trace_log_jsonl_nulls_and_lines() {
        let mut log = V2TraceLog::new();
        for (cycle, asm) in [(1u64, "NOP"), (2, "HALT")] {
            log.push(V2TraceEntry {
                cycle,
                retired: cycle - 1,
                pc: (cycle - 1) as u32,
                ir_low: 0,
                ir_ext: 0,
                word: 0,
                asm: asm.to_string(),
                flag_z: false,
                flag_c: false,
                reg_writes: Vec::new(),
                ram_writes: Vec::new(),
                mmio_reads: Vec::new(),
                mmio_writes: Vec::new(),
                stage_f_bank_switches: None,
                stage_f_mixed_dual_capture: None,
                stage_x_mixed_software: None,
                ram_high_bank_read_swaps: None,
                rom_upper_bank_group_select: None,
            });
        }

        let jsonl = log.to_jsonl();
        let split: Vec<&str> = jsonl.lines().collect();
        assert_eq!(split.len(), 2);
        // None hybrid counters serialize as JSON null; empty event vecs as [].
        assert!(split[0].contains("\"stage_f_bank_switches\":null"));
        assert!(split[0].contains("\"reg_writes\":[]"));
        assert!(split[0].contains("\"asm\":\"NOP\""));
        assert!(split[1].contains("\"asm\":\"HALT\""));
    }

    #[test]
    fn test_v2_trace_log_mmio_and_reg_write_streams() {
        let mut log = V2TraceLog::new();
        // Cycle 1: write 0xAA to port 61, and R3=1. Cycle 2: no port-61 write,
        // R3=2. Cycle 3: two writes to port 61 (0xBB then 0xCC) in one cycle.
        log.push(V2TraceEntry {
            cycle: 1,
            retired: 0,
            pc: 0,
            ir_low: 0,
            ir_ext: 0,
            word: 0,
            asm: "A".to_string(),
            flag_z: false,
            flag_c: false,
            reg_writes: vec![V2TraceRegWrite { reg: 3, value: 1 }],
            ram_writes: Vec::new(),
            mmio_reads: Vec::new(),
            mmio_writes: vec![V2TraceMemEvent {
                addr: 61,
                value: 0xAA,
            }],
            stage_f_bank_switches: None,
            stage_f_mixed_dual_capture: None,
            stage_x_mixed_software: None,
            ram_high_bank_read_swaps: None,
            rom_upper_bank_group_select: None,
        });
        log.push(V2TraceEntry {
            cycle: 2,
            retired: 1,
            pc: 1,
            ir_low: 0,
            ir_ext: 0,
            word: 0,
            asm: "B".to_string(),
            flag_z: false,
            flag_c: false,
            reg_writes: vec![V2TraceRegWrite { reg: 3, value: 2 }],
            ram_writes: Vec::new(),
            mmio_reads: Vec::new(),
            mmio_writes: vec![V2TraceMemEvent {
                addr: 40,
                value: 0x99,
            }],
            stage_f_bank_switches: None,
            stage_f_mixed_dual_capture: None,
            stage_x_mixed_software: None,
            ram_high_bank_read_swaps: None,
            rom_upper_bank_group_select: None,
        });
        log.push(V2TraceEntry {
            cycle: 3,
            retired: 2,
            pc: 2,
            ir_low: 0,
            ir_ext: 0,
            word: 0,
            asm: "C".to_string(),
            flag_z: false,
            flag_c: false,
            reg_writes: Vec::new(),
            ram_writes: Vec::new(),
            mmio_reads: Vec::new(),
            mmio_writes: vec![
                V2TraceMemEvent {
                    addr: 61,
                    value: 0xBB,
                },
                V2TraceMemEvent {
                    addr: 61,
                    value: 0xCC,
                },
            ],
            stage_f_bank_switches: None,
            stage_f_mixed_dual_capture: None,
            stage_x_mixed_software: None,
            ram_high_bank_read_swaps: None,
            rom_upper_bank_group_select: None,
        });

        // Port 61 stream: two writes in cycle 3 preserve their in-cycle order.
        assert_eq!(
            log.mmio_write_stream(61),
            vec![(1, 0xAA), (3, 0xBB), (3, 0xCC)]
        );
        // Port with no writes yields an empty stream.
        assert_eq!(log.mmio_write_stream(7), Vec::<(u64, u64)>::new());
        // Register series omits the cycle where R3 is not written (cycle 3).
        assert_eq!(log.reg_write_stream(3), vec![(1, 1), (2, 2)]);
        assert_eq!(log.reg_write_stream(8), Vec::<(u64, u64)>::new());
    }

    #[test]
    fn test_v2_trace_json_escape() {
        // asm text with a quote and backslash must produce valid JSON escapes.
        let entry = V2TraceEntry {
            cycle: 0,
            retired: 0,
            pc: 0,
            ir_low: 0,
            ir_ext: 0,
            word: 0,
            asm: "x=\"a\"\\b".to_string(),
            flag_z: false,
            flag_c: false,
            reg_writes: Vec::new(),
            ram_writes: Vec::new(),
            mmio_reads: Vec::new(),
            mmio_writes: Vec::new(),
            stage_f_bank_switches: None,
            stage_f_mixed_dual_capture: None,
            stage_x_mixed_software: None,
            ram_high_bank_read_swaps: None,
            rom_upper_bank_group_select: None,
        };
        assert!(entry.to_json().contains("\"asm\":\"x=\\\"a\\\"\\\\b\""));
    }

    /// Helper: a bare entry carrying only the fields a reconstruction test cares
    /// about (deltas, flags, pc); everything else defaulted/empty.
    fn delta_entry(
        cycle: u64,
        pc: u32,
        flag_z: bool,
        flag_c: bool,
        reg_writes: Vec<V2TraceRegWrite>,
        ram_writes: Vec<V2TraceMemEvent>,
    ) -> V2TraceEntry {
        V2TraceEntry {
            cycle,
            retired: cycle,
            pc,
            ir_low: 0,
            ir_ext: 0,
            word: 0,
            asm: String::new(),
            flag_z,
            flag_c,
            reg_writes,
            ram_writes,
            mmio_reads: Vec::new(),
            mmio_writes: Vec::new(),
            stage_f_bank_switches: None,
            stage_f_mixed_dual_capture: None,
            stage_x_mixed_software: None,
            ram_high_bank_read_swaps: None,
            rom_upper_bank_group_select: None,
        }
    }

    /// Helper: a fully-populated entry exercising every field of the JSON
    /// schema, used as the round-trip fixture and as the `..` base for the
    /// narrower reader tests.
    fn sample_entry() -> V2TraceEntry {
        V2TraceEntry {
            cycle: 7,
            retired: 3,
            pc: 12,
            ir_low: 0x2A,
            ir_ext: 0x0001,
            word: 0x0001_182A,
            asm: "LDI R8, 42".to_string(),
            flag_z: false,
            flag_c: true,
            reg_writes: vec![
                V2TraceRegWrite { reg: 8, value: 42 },
                V2TraceRegWrite {
                    reg: 15,
                    value: u64::MAX,
                },
            ],
            ram_writes: vec![V2TraceMemEvent {
                addr: 0x21,
                value: 0xAA,
            }],
            mmio_reads: Vec::new(),
            mmio_writes: vec![V2TraceMemEvent {
                addr: 0x40,
                value: 0xAA,
            }],
            stage_f_bank_switches: Some(5),
            stage_f_mixed_dual_capture: Some(2),
            stage_x_mixed_software: Some(2),
            ram_high_bank_read_swaps: Some(4),
            rom_upper_bank_group_select: Some(1),
        }
    }

    #[test]
    fn test_reconstruct_states_folds_deltas_into_absolute() {
        let mut log = V2TraceLog::new();
        // Step 1: R0=5, RAM[2]=9, flags z=0 c=1.
        log.push(delta_entry(
            1,
            0,
            false,
            true,
            vec![V2TraceRegWrite { reg: 0, value: 5 }],
            vec![V2TraceMemEvent { addr: 2, value: 9 }],
        ));
        // Step 2: R3=7 (R0 unchanged), no RAM write, flags z=1 c=0.
        log.push(delta_entry(
            2,
            1,
            true,
            false,
            vec![V2TraceRegWrite { reg: 3, value: 7 }],
            Vec::new(),
        ));
        // Step 3: overwrite R0=42.
        log.push(delta_entry(
            3,
            2,
            false,
            false,
            vec![V2TraceRegWrite { reg: 0, value: 42 }],
            Vec::new(),
        ));

        let states = log.reconstruct_states_from_zero();
        assert_eq!(states.len(), 3);

        // After step 1: R0=5, RAM[2]=9, carry set, others zero.
        assert_eq!(states[0].regs[0], 5);
        assert_eq!(states[0].ram[2], 9);
        assert!(!states[0].flag_z && states[0].flag_c);

        // After step 2: R0 persists at 5, R3=7, RAM[2] persists, flags updated.
        assert_eq!(states[1].regs[0], 5);
        assert_eq!(states[1].regs[3], 7);
        assert_eq!(states[1].ram[2], 9);
        assert!(states[1].flag_z && !states[1].flag_c);

        // After step 3: R0 overwritten to 42, R3 persists.
        assert_eq!(states[2].regs[0], 42);
        assert_eq!(states[2].regs[3], 7);
        assert_eq!(states[2].pc, 2);
    }

    #[test]
    fn test_reconstruct_states_honors_initial_seed() {
        let mut log = V2TraceLog::new();
        log.push(delta_entry(
            1,
            0,
            false,
            false,
            vec![V2TraceRegWrite { reg: 1, value: 100 }],
            Vec::new(),
        ));
        let mut init_regs = [0u64; 16];
        init_regs[0] = 11; // untouched register must survive from the seed.
        let mut init_ram = [0u64; 128];
        init_ram[5] = 99; // untouched RAM word must survive from the seed.

        let states = log.reconstruct_states(init_regs, init_ram);
        assert_eq!(states[0].regs[0], 11);
        assert_eq!(states[0].regs[1], 100);
        assert_eq!(states[0].ram[5], 99);
    }

    #[test]
    fn test_trace_state_json_format_stable() {
        let mut regs = [0u64; 16];
        regs[0] = 42;
        regs[15] = 7;
        let mut ram = [0u64; 128];
        ram[1] = 9;
        let state = V2TraceState {
            cycle: 3,
            retired: 3,
            pc: 12,
            flag_z: false,
            flag_c: true,
            regs,
            ram,
        };
        let json = state.to_json();
        assert!(
            json.starts_with(
                "{\"cycle\":3,\"retired\":3,\"pc\":12,\"flag_z\":false,\"flag_c\":true,"
            )
        );
        // Register array is decimal, index-ordered, length 16.
        assert!(json.contains("\"regs\":[42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7]"));
        // RAM array is decimal, length 128, with the single non-zero word.
        assert!(json.contains("\"ram\":[0,9,0,"));
        // 7 top-level fields (6 commas) + 16 regs (15) + 128 ram words (127).
        assert_eq!(json.matches(',').count(), 6 + 15 + 127);
    }

    #[test]
    fn test_to_state_jsonl_one_line_per_step() {
        let mut log = V2TraceLog::new();
        log.push(delta_entry(
            1,
            0,
            false,
            false,
            vec![V2TraceRegWrite { reg: 0, value: 1 }],
            Vec::new(),
        ));
        log.push(delta_entry(
            2,
            1,
            true,
            false,
            vec![V2TraceRegWrite { reg: 0, value: 2 }],
            Vec::new(),
        ));
        let jsonl = log.to_state_jsonl([0u64; 16], [0u64; 128]);
        let split: Vec<&str> = jsonl.lines().collect();
        assert_eq!(split.len(), 2);
        assert!(split[0].contains("\"regs\":[1,"));
        assert!(split[1].contains("\"regs\":[2,"));
    }

    #[test]
    fn test_v2_trace_entry_json_round_trip() {
        let entry = sample_entry();
        let parsed = V2TraceEntry::from_json(&entry.to_json()).expect("round-trip parse");
        assert_eq!(parsed, entry);
    }

    #[test]
    fn test_v2_trace_entry_json_round_trip_with_nulls() {
        let entry = V2TraceEntry {
            asm: "NOP".to_string(),
            reg_writes: Vec::new(),
            ram_writes: Vec::new(),
            mmio_writes: Vec::new(),
            stage_f_bank_switches: None,
            stage_f_mixed_dual_capture: None,
            stage_x_mixed_software: None,
            ram_high_bank_read_swaps: None,
            rom_upper_bank_group_select: None,
            ..sample_entry()
        };
        let parsed = V2TraceEntry::from_json(&entry.to_json()).expect("round-trip parse");
        assert_eq!(parsed, entry);
    }

    #[test]
    fn test_v2_trace_entry_json_round_trip_escapes() {
        // Control chars, quotes, and backslashes must survive to_json -> from_json.
        let entry = V2TraceEntry {
            asm: "x=\"a\"\\b\tc\n\u{01}".to_string(),
            ..sample_entry()
        };
        let parsed = V2TraceEntry::from_json(&entry.to_json()).expect("round-trip parse");
        assert_eq!(parsed, entry);
        assert_eq!(parsed.asm, entry.asm);
    }

    #[test]
    fn test_v2_trace_log_jsonl_round_trip() {
        let mut log = V2TraceLog::new();
        log.push(sample_entry());
        log.push(V2TraceEntry {
            cycle: 8,
            asm: "HALT".to_string(),
            reg_writes: Vec::new(),
            stage_f_bank_switches: None,
            stage_f_mixed_dual_capture: None,
            stage_x_mixed_software: None,
            ram_high_bank_read_swaps: None,
            rom_upper_bank_group_select: None,
            ..sample_entry()
        });

        let parsed = V2TraceLog::from_jsonl(&log.to_jsonl()).expect("round-trip parse");
        assert_eq!(parsed, log);
    }

    #[test]
    fn test_v2_trace_log_jsonl_empty_round_trip() {
        let log = V2TraceLog::new();
        let parsed = V2TraceLog::from_jsonl(&log.to_jsonl()).expect("empty round-trip");
        assert!(parsed.is_empty());
        assert_eq!(parsed, log);
    }

    #[test]
    fn test_v2_trace_from_jsonl_skips_blank_lines() {
        let entry = sample_entry();
        let text = format!("\n{}\n\n", entry.to_json());
        let parsed = V2TraceLog::from_jsonl(&text).expect("parse with blank lines");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed.entries()[0], entry);
    }

    #[test]
    fn test_v2_trace_from_json_rejects_bad_input() {
        // Out-of-range value for a narrow field.
        let bad_width = r#"{"cycle":0,"retired":0,"pc":4294967296,"ir_low":0,"ir_ext":0,"word":0,"asm":"","flag_z":false,"flag_c":false,"reg_writes":[],"ram_writes":[],"mmio_reads":[],"mmio_writes":[],"stage_f_bank_switches":null,"stage_f_mixed_dual_capture":null,"stage_x_mixed_software":null,"ram_high_bank_read_swaps":null,"rom_upper_bank_group_select":null}"#;
        assert!(V2TraceEntry::from_json(bad_width).is_err());

        // Missing a required field.
        assert!(V2TraceEntry::from_json(r#"{"cycle":0}"#).is_err());

        // Trailing garbage after a complete object.
        let trailing = format!("{} x", sample_entry().to_json());
        assert!(V2TraceEntry::from_json(&trailing).is_err());

        // A blank line does not parse as an entry via from_json.
        let bad_line = V2TraceLog::from_jsonl("{\"cycle\":0}");
        assert!(bad_line.is_err());
    }

    #[test]
    fn test_v2_trace_entry_csv_record_stable() {
        let entry = sample_entry();

        // asm contains a comma, so it must be RFC 4180 quoted; list columns are
        // flattened to `;`-separated decimal KEY=VALUE cells.
        assert_eq!(
            entry.to_csv_record(),
            "7,3,12,42,1,71722,\"LDI R8, 42\",0,1,R8=42;R15=18446744073709551615,33=170,,64=170,5,2,2,4,1"
        );
    }

    #[test]
    fn test_v2_trace_log_csv_header_and_rows() {
        let mut log = V2TraceLog::new();
        for (cycle, asm) in [(1u64, "NOP"), (2, "HALT")] {
            log.push(V2TraceEntry {
                cycle,
                retired: cycle - 1,
                pc: (cycle - 1) as u32,
                ir_low: 0,
                ir_ext: 0,
                word: 0,
                asm: asm.to_string(),
                flag_z: false,
                flag_c: false,
                reg_writes: Vec::new(),
                ram_writes: Vec::new(),
                mmio_reads: Vec::new(),
                mmio_writes: Vec::new(),
                stage_f_bank_switches: None,
                stage_f_mixed_dual_capture: None,
                stage_x_mixed_software: None,
                ram_high_bank_read_swaps: None,
                rom_upper_bank_group_select: None,
            });
        }

        let csv = log.to_csv();
        let split: Vec<&str> = csv.lines().collect();
        // Header row + one row per entry.
        assert_eq!(split.len(), 3);
        assert_eq!(split[0], V2_TRACE_CSV_HEADER);
        // Header column count matches every record's field count.
        let cols = split[0].split(',').count();
        assert_eq!(split[1].split(',').count(), cols);
        // Empty list columns and absent counters serialize as empty cells.
        assert_eq!(split[1], "1,0,0,0,0,0,NOP,0,0,,,,,,,,,");
        assert!(split[2].starts_with("2,1,1,0,0,0,HALT,0,0,"));
    }

    #[test]
    fn test_v2_trace_jsonl_header_stable() {
        let mut log = V2TraceLog::new();
        for cycle in [1u64, 2] {
            log.push(V2TraceEntry {
                cycle,
                retired: cycle - 1,
                pc: (cycle - 1) as u32,
                ir_low: 0,
                ir_ext: 0,
                word: 0,
                asm: "NOP".to_string(),
                flag_z: false,
                flag_c: false,
                reg_writes: Vec::new(),
                ram_writes: Vec::new(),
                mmio_reads: Vec::new(),
                mmio_writes: Vec::new(),
                stage_f_bank_switches: None,
                stage_f_mixed_dual_capture: None,
                stage_x_mixed_software: None,
                ram_high_bank_read_swaps: None,
                rom_upper_bank_group_select: None,
            });
        }

        assert_eq!(
            log.jsonl_header(),
            "{\"schema\":\"tileuniverse-v2-trace\",\"version\":1,\"fields\":[\"cycle\",\"retired\",\"pc\",\"ir_low\",\"ir_ext\",\"word\",\"asm\",\"flag_z\",\"flag_c\",\"reg_writes\",\"ram_writes\",\"mmio_reads\",\"mmio_writes\",\"stage_f_bank_switches\",\"stage_f_mixed_dual_capture\",\"stage_x_mixed_software\",\"ram_high_bank_read_swaps\",\"rom_upper_bank_group_select\"],\"count\":2}"
        );

        // Header line then one body line per entry; body is byte-identical to to_jsonl.
        let with_header = log.to_jsonl_with_header();
        let mut lines = with_header.lines();
        assert_eq!(lines.next().unwrap(), log.jsonl_header());
        let body: Vec<&str> = lines.collect();
        assert_eq!(body.join("\n"), log.to_jsonl());
        assert_eq!(body.len(), 2);
    }

    #[test]
    fn test_v2_trace_jsonl_header_empty_log_is_header_only() {
        let log = V2TraceLog::new();
        // Empty log: count=0 and the dump is exactly the (validatable) header, no newline.
        assert!(log.jsonl_header().contains("\"count\":0"));
        assert_eq!(log.to_jsonl_with_header(), log.jsonl_header());
    }

    #[test]
    fn test_v2_trace_entry_fields_match_json_key_order() {
        // Guard: the header field list must stay in lockstep with to_json's key
        // order, so downstream parsers keyed on the header stay correct.
        let entry = V2TraceEntry {
            cycle: 1,
            retired: 0,
            pc: 0,
            ir_low: 0,
            ir_ext: 0,
            word: 0,
            asm: "NOP".to_string(),
            flag_z: false,
            flag_c: false,
            reg_writes: Vec::new(),
            ram_writes: Vec::new(),
            mmio_reads: Vec::new(),
            mmio_writes: Vec::new(),
            stage_f_bank_switches: None,
            stage_f_mixed_dual_capture: None,
            stage_x_mixed_software: None,
            ram_high_bank_read_swaps: None,
            rom_upper_bank_group_select: None,
        };
        let json = entry.to_json();
        // Extract top-level keys in order (all keys are followed by `":`).
        let keys: Vec<String> = json
            .match_indices("\":")
            .map(|(idx, _)| {
                let prefix = &json[..idx];
                let start = prefix.rfind('"').unwrap() + 1;
                prefix[start..].to_string()
            })
            .collect();
        assert_eq!(keys, V2_TRACE_ENTRY_FIELDS.to_vec());
    }

    fn sample_entries() -> Vec<V2TraceEntry> {
        vec![
            // Fully-populated entry with hybrid counters and an escaped asm.
            V2TraceEntry {
                cycle: 7,
                retired: 3,
                pc: 0x1234,
                ir_low: 0x2A,
                ir_ext: 0x0001,
                word: 0x0001_182A,
                asm: "LDI R8, \"x\"\t; tab\\back".to_string(),
                flag_z: false,
                flag_c: true,
                reg_writes: vec![
                    V2TraceRegWrite { reg: 8, value: 42 },
                    V2TraceRegWrite {
                        reg: 15,
                        value: 0xDEAD_BEEF_CAFE_F00D,
                    },
                ],
                ram_writes: vec![V2TraceMemEvent {
                    addr: 0x21,
                    value: 0xAA,
                }],
                mmio_reads: vec![V2TraceMemEvent {
                    addr: 0x41,
                    value: 0x01,
                }],
                mmio_writes: vec![V2TraceMemEvent {
                    addr: 0x40,
                    value: 0xAA,
                }],
                stage_f_bank_switches: Some(5),
                stage_f_mixed_dual_capture: Some(2),
                stage_x_mixed_software: Some(2),
                ram_high_bank_read_swaps: Some(4),
                rom_upper_bank_group_select: Some(1),
            },
            // Minimal entry: empty event lists, empty asm, no hybrid counters.
            V2TraceEntry {
                cycle: 1,
                retired: 0,
                pc: 0,
                ir_low: 0,
                ir_ext: 0,
                word: 0,
                asm: String::new(),
                flag_z: true,
                flag_c: false,
                reg_writes: Vec::new(),
                ram_writes: Vec::new(),
                mmio_reads: Vec::new(),
                mmio_writes: Vec::new(),
                stage_f_bank_switches: None,
                stage_f_mixed_dual_capture: None,
                stage_x_mixed_software: None,
                ram_high_bank_read_swaps: None,
                rom_upper_bank_group_select: None,
            },
        ]
    }

    #[test]
    fn test_v2_trace_entry_line_roundtrip() {
        for entry in sample_entries() {
            let line = entry.to_line();
            let parsed = V2TraceEntry::from_line(&line)
                .unwrap_or_else(|e| panic!("failed to parse '{line}': {e}"));
            assert_eq!(parsed, entry);
            // Re-serializing the parsed entry must reproduce the exact line.
            assert_eq!(parsed.to_line(), line);
        }
    }

    #[test]
    fn test_v2_trace_log_lines_roundtrip() {
        let mut log = V2TraceLog::new();
        for entry in sample_entries() {
            log.push(entry);
        }
        let text = log.to_lines();
        assert_eq!(V2TraceLog::from_lines(&text).unwrap(), log);
        // A trailing newline (common when written to a file) is tolerated.
        let with_newline = format!("{text}\n");
        assert_eq!(V2TraceLog::from_lines(&with_newline).unwrap(), log);
        // Empty input parses to an empty log (inverse of an empty to_lines()).
        assert_eq!(V2TraceLog::from_lines("").unwrap(), V2TraceLog::new());
    }

    #[test]
    fn test_v2_trace_from_line_rejects_malformed() {
        assert!(V2TraceEntry::from_line("not a trace line").is_err());
        // Missing the trailing hybrid field.
        assert!(
            V2TraceEntry::from_line(
                "cycle=1 retired=0 pc=0x00 word=0x00000000 ir_low=0x00 ir_ext=0x0000 asm=\"NOP\" z=0 c=0 regs=[] ram=[] mmio_r=[] mmio_w=[]"
            )
            .is_err()
        );
    }

    #[test]
    fn test_v2_trace_csv_escape() {
        // An asm field with a quote and embedded comma must be quoted with the
        // interior quote doubled per RFC 4180.
        let entry = V2TraceEntry {
            asm: "x,\"y\"".to_string(),
            ..sample_entry()
        };
        assert!(entry.to_csv_record().contains(",\"x,\"\"y\"\"\","));
    }

    #[test]
    fn test_v2_trace_program_from_words_disassembles() {
        // Program image [NOP, NOP] loaded at origin 4: rows carry sequential
        // addresses and the exact disassembly of each word.
        let prog = V2TraceProgram::from_words(Some("nop_pair".to_string()), 4, 16, &[0, 0]);
        assert_eq!(prog.origin, 4);
        assert_eq!(prog.rom_size, 16);
        assert_eq!(prog.rows.len(), 2);
        assert_eq!(prog.rows[0].addr, 4);
        assert_eq!(prog.rows[1].addr, 5);
        assert_eq!(prog.rows[0].asm, disassemble_v2_word(0));
        assert_eq!(prog.rows[0].asm, "NOP");
    }

    #[test]
    fn test_v2_trace_program_text_and_json_stable() {
        // A label containing a quote must escape in both text and JSON forms.
        let prog = V2TraceProgram {
            label: Some("case\"x".to_string()),
            origin: 0,
            rom_size: 4,
            rows: vec![V2TraceProgramRow {
                addr: 0,
                word: 0,
                asm: "NOP".to_string(),
            }],
        };
        assert_eq!(
            prog.to_text(),
            "; program label=\"case\\\"x\" origin=0x00 rom_size=4 words=1\n0x00: 0x00000000  NOP"
        );
        assert_eq!(
            prog.to_json(),
            "{\"label\":\"case\\\"x\",\"origin\":0,\"rom_size\":4,\"rows\":[{\"addr\":0,\"word\":0,\"asm\":\"NOP\"}]}"
        );
    }

    #[test]
    fn test_v2_trace_program_none_label_json_null() {
        let prog = V2TraceProgram::from_words(None, 0, 1, &[0]);
        assert!(prog.to_json().starts_with("{\"label\":null,"));
        assert!(prog.to_text().starts_with("; program label=- origin=0x00"));
    }

    #[test]
    fn test_v2_trace_document_jsonl_structure() {
        // Document JSONL: line 1 is the program header (has "rows", no "cycle");
        // remaining lines are per-step entries (have "cycle", no "rows").
        let prog = V2TraceProgram::from_words(Some("doc".to_string()), 0, 2, &[0, 0]);
        let mut log = V2TraceLog::new();
        for cycle in [1u64, 2] {
            log.push(V2TraceEntry {
                cycle,
                retired: cycle - 1,
                pc: (cycle - 1) as u32,
                ir_low: 0,
                ir_ext: 0,
                word: 0,
                asm: "NOP".to_string(),
                flag_z: false,
                flag_c: false,
                reg_writes: Vec::new(),
                ram_writes: Vec::new(),
                mmio_reads: Vec::new(),
                mmio_writes: Vec::new(),
                stage_f_bank_switches: None,
                stage_f_mixed_dual_capture: None,
                stage_x_mixed_software: None,
                ram_high_bank_read_swaps: None,
                rom_upper_bank_group_select: None,
            });
        }
        let doc = V2TraceDocument::new(prog, log);
        let jsonl = doc.to_jsonl();
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("\"rows\":["));
        assert!(!lines[0].contains("\"cycle\":"));
        assert!(lines[1].contains("\"cycle\":1"));
        assert!(lines[2].contains("\"cycle\":2"));
        assert!(!lines[1].contains("\"rows\":"));
    }

    #[test]
    fn test_v2_trace_document_empty_log_jsonl_is_header_only() {
        let prog = V2TraceProgram::from_words(None, 0, 1, &[0]);
        let doc = V2TraceDocument::new(prog, V2TraceLog::new());
        let jsonl = doc.to_jsonl();
        // No trailing newline when the log is empty; just the header line.
        assert!(!jsonl.ends_with('\n'));
        assert_eq!(jsonl.lines().count(), 1);
        assert!(jsonl.contains("\"rows\":["));
    }
}
