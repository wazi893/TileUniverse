//! V2 instruction trace data model and deterministic serialization helpers.

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
}
