use std::fmt::Write as FmtWrite;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::simulation::Simulation;
use crate::tile_meta::TileType;
use crate::tilemap::{HEIGHT, WIDTH};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlueprintTile {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub tile_type: TileType,
    pub logic: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blueprint {
    pub width: u32,
    pub height: u32,
    pub num_layers: u32,
    pub tiles: Vec<BlueprintTile>,
}

#[derive(Debug)]
pub enum BlueprintError {
    Io(std::io::Error),
    Parse(String),
}

impl From<std::io::Error> for BlueprintError {
    fn from(e: std::io::Error) -> Self {
        BlueprintError::Io(e)
    }
}

fn tile_type_to_str(t: TileType) -> &'static str {
    match t {
        TileType::Wire => "Wire",
        TileType::And => "And",
        TileType::Or => "Or",
        TileType::Xor => "Xor",
        TileType::Not => "Not",
        TileType::Latch => "Latch",
        TileType::Register8 => "Register8",
        TileType::ClockGlobal => "ClockGlobal",
        TileType::VmSpawner => "VmSpawner",
        TileType::VmStatus => "VmStatus",
        TileType::QDemo => "QDemo",
        // EPIC 103: Arithmetic Tiles
        TileType::Add => "Add",
        TileType::Sub => "Sub",
        TileType::Mul => "Mul",
        TileType::Div => "Div",
        TileType::Mod => "Mod",
        TileType::Shl => "Shl",
        TileType::Shr => "Shr",
        // EPIC 103: Comparison Tiles
        TileType::Lt => "Lt",
        TileType::Gt => "Gt",
        TileType::Eq => "Eq",
        TileType::Neq => "Neq",
        TileType::Lte => "Lte",
        TileType::Gte => "Gte",
        // EPIC 103: Routing & Special
        TileType::Mux => "Mux",
        TileType::Zero => "Zero",
        TileType::Neg => "Neg",
        TileType::Abs => "Abs",
        // EPIC 104: Memory Tiles
        TileType::Ram => "Ram",
        TileType::Counter => "Counter",
        TileType::Const => "Const",
        // Wire Crossing Tiles
        TileType::Cross => "Cross",
        TileType::WireH => "WireH",
        TileType::WireV => "WireV",
        // Unidirectional Wires
        TileType::WireDown => "WireDown",
        TileType::WireRight => "WireRight",
        TileType::WireUp => "WireUp",
        TileType::WireLeft => "WireLeft",
        // Component System
        TileType::ComponentOutput => "ComponentOutput",
        // Bus Architecture
        TileType::BusInterface => "BusInterface",
        // Memory Controller
        TileType::MemoryPort => "MemoryPort",
        // EPIC 116: CPU Tiles
        TileType::CpuHead => "CpuHead",
        TileType::Register => "Register",
        TileType::Console => "Console",
        // CPU Building Blocks
        TileType::Decoder3to8 => "Decoder3to8",
        TileType::Mux8to1 => "Mux8to1",
        TileType::Mux4to1 => "Mux4to1",
        TileType::Demux1to8 => "Demux1to8",
        TileType::RegEnable => "RegEnable",
        TileType::ProgramCounter => "ProgramCounter",
        // SPRINT 66: Evolutionary Selection
        TileType::Selector => "Selector",
        // Ising Mode Tiles
        TileType::IsingNode => "IsingNode",
        TileType::IsingBias => "IsingBias",
        // Multi-Clock Domains
        TileType::ClockDivider => "ClockDivider",
        TileType::Synchronizer => "Synchronizer",
        // Phase 1: Fully Tile-Based CPU
        TileType::AddCarry => "AddCarry",
        TileType::BitSelect => "BitSelect",
        // Wire Crossing Tiles (unidirectional)
        TileType::WireCross => "WireCross",
        TileType::WireCrossVert => "WireCrossVert",
        TileType::VBusIn => "VBusIn",
        TileType::VBusOut => "VBusOut",
        // Multi-Layer Via Tiles
        TileType::ViaUp => "ViaUp",
        TileType::ViaDown => "ViaDown",
        // Sprint 82
        TileType::SubBorrow => "SubBorrow",
        TileType::Mux16to1 => "Mux16to1",
        // Sprint 127: 64-bit Scaling Primitives
        TileType::Register64 => "Register64",
        TileType::CarryDetect => "CarryDetect",
        TileType::Decoder6to64 => "Decoder6to64",
        // Sprint 160: Weighted Via Tiles
        TileType::WeightedViaUp => "WeightedViaUp",
        TileType::WeightedViaDown => "WeightedViaDown",
        // Sprint 183: Threshold Via Tiles
        TileType::ThresholdViaUp => "ThresholdViaUp",
        TileType::ThresholdViaDown => "ThresholdViaDown",
    }
}

pub fn parse_tile_type_token(tok: &str) -> Option<TileType> {
    match tok.to_ascii_lowercase().as_str() {
        "wire" => Some(TileType::Wire),
        "and" => Some(TileType::And),
        "or" => Some(TileType::Or),
        "xor" => Some(TileType::Xor),
        "not" => Some(TileType::Not),
        "latch" => Some(TileType::Latch),
        "register8" | "reg8" => Some(TileType::Register8),
        "clockglobal" | "clock" | "clk" => Some(TileType::ClockGlobal),
        "vmspawner" | "vm_spawn" | "vmspn" => Some(TileType::VmSpawner),
        "vmstatus" | "vm_stat" | "vmst" => Some(TileType::VmStatus),
        "qdemo" => Some(TileType::QDemo),
        // EPIC 103: Arithmetic Tiles
        "add" => Some(TileType::Add),
        "sub" => Some(TileType::Sub),
        "mul" => Some(TileType::Mul),
        "div" => Some(TileType::Div),
        "mod" => Some(TileType::Mod),
        "shl" | "shiftleft" => Some(TileType::Shl),
        "shr" | "shiftright" => Some(TileType::Shr),
        // EPIC 103: Comparison Tiles
        "lt" | "lessthan" => Some(TileType::Lt),
        "gt" | "greaterthan" => Some(TileType::Gt),
        "eq" | "equal" => Some(TileType::Eq),
        "neq" | "notequal" => Some(TileType::Neq),
        "lte" | "lessequal" => Some(TileType::Lte),
        "gte" | "greaterequal" => Some(TileType::Gte),
        // EPIC 103: Routing & Special
        "mux" | "multiplex" => Some(TileType::Mux),
        "zero" | "iszero" => Some(TileType::Zero),
        "neg" | "negate" => Some(TileType::Neg),
        "abs" | "absolute" => Some(TileType::Abs),
        // EPIC 104: Memory Tiles
        "ram" | "memory" => Some(TileType::Ram),
        "counter" | "cnt" => Some(TileType::Counter),
        "const" | "constant" => Some(TileType::Const),
        // Wire Crossing Tiles
        "cross" | "crossing" => Some(TileType::Cross),
        "wireh" | "hwire" | "horizontal" => Some(TileType::WireH),
        "wirev" | "vwire" | "vertical" => Some(TileType::WireV),
        // EPIC 116: CPU Tiles
        "cpuhead" | "cpu" => Some(TileType::CpuHead),
        "register" | "reg" => Some(TileType::Register),
        "console" | "term" => Some(TileType::Console),
        // CPU Building Blocks
        "decoder3to8" | "decoder" | "dec3to8" => Some(TileType::Decoder3to8),
        "mux8to1" | "mux8" => Some(TileType::Mux8to1),
        "mux4to1" | "mux4" => Some(TileType::Mux4to1),
        "demux1to8" | "demux8" | "demux" => Some(TileType::Demux1to8),
        "regenable" | "regen" | "regena" => Some(TileType::RegEnable),
        "programcounter" | "pc" | "progcnt" => Some(TileType::ProgramCounter),
        "selector" | "sel" => Some(TileType::Selector),
        // Unidirectional Wires
        "wiredown" | "wdown" => Some(TileType::WireDown),
        "wireright" | "wright" => Some(TileType::WireRight),
        "wireup" | "wup" => Some(TileType::WireUp),
        "wireleft" | "wleft" => Some(TileType::WireLeft),
        "componentoutput" | "compout" | "cout" => Some(TileType::ComponentOutput),
        // Bus Architecture
        "businterface" | "bus_interface" | "bus" => Some(TileType::BusInterface),
        // Memory Controller
        "memoryport" | "memory_port" | "memport" => Some(TileType::MemoryPort),
        // Ising Mode Tiles
        "isingnode" | "ising" | "pbit" => Some(TileType::IsingNode),
        "isingbias" | "bias" | "field" => Some(TileType::IsingBias),
        // Multi-Clock Domains
        "clockdivider" | "clock_divider" | "clkdiv" => Some(TileType::ClockDivider),
        "synchronizer" | "sync" | "cdc" => Some(TileType::Synchronizer),
        // Wire Crossing Tiles (unidirectional)
        "wirecross" | "wcross" | "xwire" => Some(TileType::WireCross),
        "wirecrossvert" | "wcrossvert" | "xwirev" => Some(TileType::WireCrossVert),
        "vbusin" | "vbus_in" => Some(TileType::VBusIn),
        "vbusout" | "vbus_out" => Some(TileType::VBusOut),
        "viaup" | "via_up" => Some(TileType::ViaUp),
        "viadown" | "via_down" => Some(TileType::ViaDown),
        // Sprint 127: 64-bit Scaling Primitives
        "register64" | "reg64" => Some(TileType::Register64),
        "carrydetect" | "carry_detect" | "carrydet" => Some(TileType::CarryDetect),
        "decoder6to64" | "dec6to64" | "decoder64" => Some(TileType::Decoder6to64),
        _ => None,
    }
}

impl Blueprint {
    pub fn new(width: u32, height: u32) -> Blueprint {
        Blueprint {
            width,
            height,
            num_layers: 1,
            tiles: Vec::new(),
        }
    }

    pub fn save_to_writer<W: Write>(&self, mut w: W) -> Result<(), BlueprintError> {
        if self.num_layers > 1 {
            // V2 format: includes layer count and z coordinates
            writeln!(w, "LF-BLUEPRINT 2")?;
            writeln!(w, "SIZE {} {} {}", self.width, self.height, self.num_layers)?;
            for t in &self.tiles {
                let typ = tile_type_to_str(t.tile_type);
                let mut logic_str = String::new();
                match t.logic {
                    Some(v) => {
                        write!(&mut logic_str, "0x{:016X}", v).unwrap();
                    }
                    None => logic_str.push('-'),
                }
                writeln!(w, "TILE {} {} {} {} {}", t.x, t.y, t.z, typ, logic_str)?;
            }
        } else {
            // V1 format: backward compatible, no z coordinate
            writeln!(w, "LF-BLUEPRINT 1")?;
            writeln!(w, "SIZE {} {}", self.width, self.height)?;
            for t in &self.tiles {
                let typ = tile_type_to_str(t.tile_type);
                let mut logic_str = String::new();
                match t.logic {
                    Some(v) => {
                        write!(&mut logic_str, "0x{:016X}", v).unwrap();
                    }
                    None => logic_str.push('-'),
                }
                writeln!(w, "TILE {} {} {} {}", t.x, t.y, typ, logic_str)?;
            }
        }
        Ok(())
    }

    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), BlueprintError> {
        let mut f = std::fs::File::create(path)?;
        self.save_to_writer(&mut f)
    }

    pub fn load_from_reader<R: BufRead>(mut r: R) -> Result<Blueprint, BlueprintError> {
        let mut line = String::new();

        // Read first non-empty, non-comment line: magic
        let magic = read_next_line(&mut r, &mut line)?;
        let version = match magic.as_deref() {
            Some("LF-BLUEPRINT 1") => 1,
            Some("LF-BLUEPRINT 2") => 2,
            _ => {
                return Err(BlueprintError::Parse(
                    "invalid or missing magic".to_string(),
                ));
            }
        };

        // Read SIZE line
        line.clear();
        let size_line = read_next_line(&mut r, &mut line)?
            .ok_or_else(|| BlueprintError::Parse("missing SIZE line".to_string()))?;
        let mut size_parts = size_line.split_whitespace();
        if size_parts.next() != Some("SIZE") {
            return Err(BlueprintError::Parse("expected SIZE line".to_string()));
        }
        let w_str = size_parts
            .next()
            .ok_or_else(|| BlueprintError::Parse("missing width".to_string()))?;
        let h_str = size_parts
            .next()
            .ok_or_else(|| BlueprintError::Parse("missing height".to_string()))?;
        let num_layers: u32 = if version >= 2 {
            let l_str = size_parts
                .next()
                .ok_or_else(|| BlueprintError::Parse("missing layers in v2 SIZE".to_string()))?;
            l_str
                .parse()
                .map_err(|_| BlueprintError::Parse("bad layers".to_string()))?
        } else {
            1 // V1: force num_layers=1
        };
        if size_parts.next().is_some() {
            return Err(BlueprintError::Parse("extra tokens in SIZE".to_string()));
        }
        let width: u32 = w_str
            .parse()
            .map_err(|_| BlueprintError::Parse("bad width".to_string()))?;
        let height: u32 = h_str
            .parse()
            .map_err(|_| BlueprintError::Parse("bad height".to_string()))?;

        let mut tiles: Vec<BlueprintTile> = Vec::new();

        // Parse remaining TILE lines
        loop {
            line.clear();
            match read_next_line(&mut r, &mut line)? {
                Some(l) => {
                    let mut it = l.split_whitespace();
                    let op = it.next().unwrap_or("");
                    if op != "TILE" {
                        return Err(BlueprintError::Parse("unknown opcode".to_string()));
                    }
                    let x_s = it
                        .next()
                        .ok_or_else(|| BlueprintError::Parse("missing x".to_string()))?;
                    let y_s = it
                        .next()
                        .ok_or_else(|| BlueprintError::Parse("missing y".to_string()))?;
                    // V2 has z coordinate between y and type
                    let (z, ty_s) = if version >= 2 {
                        let z_s = it
                            .next()
                            .ok_or_else(|| BlueprintError::Parse("missing z".to_string()))?;
                        let z: u32 = z_s
                            .parse()
                            .map_err(|_| BlueprintError::Parse("bad z".to_string()))?;
                        let ty = it
                            .next()
                            .ok_or_else(|| BlueprintError::Parse("missing type".to_string()))?;
                        (z, ty)
                    } else {
                        let ty = it
                            .next()
                            .ok_or_else(|| BlueprintError::Parse("missing type".to_string()))?;
                        (0u32, ty)
                    };
                    let lg_s = it
                        .next()
                        .ok_or_else(|| BlueprintError::Parse("missing logic".to_string()))?;
                    if it.next().is_some() {
                        return Err(BlueprintError::Parse("extra tokens in TILE".to_string()));
                    }
                    let x: u32 = x_s
                        .parse()
                        .map_err(|_| BlueprintError::Parse("bad x".to_string()))?;
                    let y: u32 = y_s
                        .parse()
                        .map_err(|_| BlueprintError::Parse("bad y".to_string()))?;
                    let tile_type = parse_tile_type_token(ty_s)
                        .ok_or_else(|| BlueprintError::Parse("unknown tile type".to_string()))?;
                    let logic = if lg_s == "-" {
                        None
                    } else {
                        let s = lg_s
                            .strip_prefix("0x")
                            .or_else(|| lg_s.strip_prefix("0X"))
                            .ok_or_else(|| {
                                BlueprintError::Parse("logic must be 0x... or -".to_string())
                            })?;
                        if s.is_empty() || s.len() > 16 {
                            return Err(BlueprintError::Parse(
                                "invalid logic hex length".to_string(),
                            ));
                        }
                        let v = u64::from_str_radix(s, 16)
                            .map_err(|_| BlueprintError::Parse("invalid logic hex".to_string()))?;
                        Some(v)
                    };
                    if x >= width || y >= height {
                        return Err(BlueprintError::Parse("oob coordinate".to_string()));
                    }
                    if z >= num_layers {
                        return Err(BlueprintError::Parse("oob layer coordinate".to_string()));
                    }
                    tiles.push(BlueprintTile {
                        x,
                        y,
                        z,
                        tile_type,
                        logic,
                    });
                }
                None => break,
            }
        }

        Ok(Blueprint {
            width,
            height,
            num_layers,
            tiles,
        })
    }

    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Blueprint, BlueprintError> {
        let f = std::fs::File::open(path)?;
        let br = BufReader::new(f);
        Blueprint::load_from_reader(br)
    }

    // EPIC 2: Export from a Simulation snapshot into a pure Blueprint
    pub fn from_simulation(sim: &Simulation) -> Blueprint {
        let w = sim.tilemap.width;
        let h = sim.tilemap.height;
        let nl = sim.tilemap.num_layers;
        let mut bp = Blueprint {
            width: w as u32,
            height: h as u32,
            num_layers: nl as u32,
            tiles: Vec::new(),
        };
        for idx in 0..sim.tilemap.tile_count() {
            let (x, y, z) = sim.tilemap.coords_from_idx(idx);
            let tile = &sim.tilemap.tiles[idx];
            let tt = tile.meta.tile_type;
            let val = sim.tilemap.value(idx);
            let logic = if val != 0 { Some(val) } else { None };
            bp.tiles.push(BlueprintTile {
                x: x as u32,
                y: y as u32,
                z: z as u32,
                tile_type: tt,
                logic,
            });
        }
        bp
    }

    // EPIC 3: Apply Blueprint into Simulation (pre-validate, then reset and apply)
    pub fn apply_to_simulation(&self, sim: &mut Simulation) -> Result<(), BlueprintError> {
        // 1) Pre-validation: ensure all coordinates are in-bounds
        for t in &self.tiles {
            let x = t.x as usize;
            let y = t.y as usize;
            if x >= WIDTH || y >= HEIGHT {
                return Err(BlueprintError::Parse("OOB coordinate".to_string()));
            }
        }

        // 2) Safe point: we can now reset the simulation
        *sim = Simulation::new();

        // 3) Apply tile types
        for t in &self.tiles {
            let x = t.x as usize;
            let y = t.y as usize;
            sim.set_tile(x, y, t.tile_type);
        }

        // 4) Apply logic bits (additive)
        for t in &self.tiles {
            if let Some(v) = t.logic {
                let x = t.x as usize;
                let y = t.y as usize;
                let _ = sim.set_logic_value(x, y, v);
            }
        }

        Ok(())
    }
}

fn read_next_line<R: BufRead>(
    r: &mut R,
    buf: &mut String,
) -> Result<Option<String>, BlueprintError> {
    let mut s = String::new();
    loop {
        buf.clear();
        let n = r.read_line(buf)?;
        if n == 0 {
            return Ok(None);
        }
        let t = buf.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        s.push_str(t);
        return Ok(Some(s));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::tile_meta::TileMeta;

    #[test]
    fn roundtrip_single_tile() {
        let mut bp = Blueprint::new(64, 64);
        bp.tiles.push(BlueprintTile {
            x: 10,
            y: 20,
            tile_type: TileType::Or,
            logic: Some(0xDEAD_BEEF_DEAD_BEEF),
            z: 0,
        });

        let mut buf = Vec::new();
        bp.save_to_writer(&mut buf).expect("save ok");

        let read = Blueprint::load_from_reader(BufReader::new(&buf[..])).expect("load ok");
        assert_eq!(bp, read);
    }

    #[test]
    fn roundtrip_multiple_tiles_mixed_logic() {
        let mut bp = Blueprint::new(32, 32);
        bp.tiles.push(BlueprintTile {
            x: 1,
            y: 1,
            z: 0,
            tile_type: TileType::Wire,
            logic: None,
        });
        bp.tiles.push(BlueprintTile {
            x: 2,
            y: 2,
            z: 0,
            tile_type: TileType::And,
            logic: Some(0x0),
        });
        bp.tiles.push(BlueprintTile {
            x: 3,
            y: 3,
            z: 0,
            tile_type: TileType::Register8,
            logic: Some(0xFFFF),
        });
        bp.tiles.push(BlueprintTile {
            x: 4,
            y: 4,
            z: 0,
            tile_type: TileType::ClockGlobal,
            logic: None,
        });

        let mut buf = Vec::new();
        bp.save_to_writer(&mut buf).expect("save ok");
        let read = Blueprint::load_from_reader(BufReader::new(&buf[..])).expect("load ok");
        assert_eq!(bp, read);
    }

    #[test]
    fn parse_rejects_bad_magic() {
        let s = b"NOT-A-BLUEPRINT\nSIZE 10 10\n";
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        match res {
            Err(BlueprintError::Parse(_)) => {}
            _ => panic!("expected parse error"),
        }
    }

    #[test]
    fn parse_rejects_bad_tile_line() {
        let s = b"LF-BLUEPRINT 1\nSIZE 10 10\nTILE 1 2 Unknown -\n";
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        match res {
            Err(BlueprintError::Parse(_)) => {}
            _ => panic!("expected parse error"),
        }

        let s2 = b"LF-BLUEPRINT 1\nSIZE 10 10\nTILE 1 a Wire -\n"; // invalid int
        let res2 = Blueprint::load_from_reader(BufReader::new(&s2[..]));
        match res2 {
            Err(BlueprintError::Parse(_)) => {}
            _ => panic!("expected parse error"),
        }

        let s3 = b"LF-BLUEPRINT 1\nSIZE 10 10\nTILE 1 2 Wire 0xZZZ\n"; // bad hex
        let res3 = Blueprint::load_from_reader(BufReader::new(&s3[..]));
        match res3 {
            Err(BlueprintError::Parse(_)) => {}
            _ => panic!("expected parse error"),
        }
    }

    // EPIC 5: Additional negative/positive tests
    #[test]
    fn parse_empty_file_fails() {
        let s = b"";
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(matches!(res, Err(BlueprintError::Parse(_))));
    }

    #[test]
    fn parse_bad_magic_fails() {
        let s = b"LF-OTHER 1\nSIZE 2 2\n";
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(matches!(res, Err(BlueprintError::Parse(_))));
    }

    #[test]
    fn parse_duplicate_magic_fails() {
        let s = b"LF-BLUEPRINT 1\nSIZE 2 2\nLF-BLUEPRINT 1\n";
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(matches!(res, Err(BlueprintError::Parse(_))));
    }

    #[test]
    fn parse_missing_size_fails() {
        let s = b"LF-BLUEPRINT 1\n# no size\n";
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(matches!(res, Err(BlueprintError::Parse(_))));
    }

    #[test]
    fn parse_bad_size_format_fails() {
        let s = b"LF-BLUEPRINT 1\nSIZE 2 X\n";
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(matches!(res, Err(BlueprintError::Parse(_))));
    }

    #[test]
    fn parse_invalid_tile_line_fails() {
        let s = b"LF-BLUEPRINT 1\nSIZE 2 2\nTIL 0 0 Or -\n"; // bad opcode
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(matches!(res, Err(BlueprintError::Parse(_))));
    }

    #[test]
    fn parse_bad_hex_fails_epic5() {
        let s = b"LF-BLUEPRINT 1\nSIZE 2 2\nTILE 0 0 Or 0xQQ\n"; // malformed hex
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(matches!(res, Err(BlueprintError::Parse(_))));
    }

    #[test]
    fn parse_oob_coords_fails() {
        let s = b"LF-BLUEPRINT 1\nSIZE 2 2\nTILE 2 0 Or -\n"; // x == width is OOB
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(matches!(res, Err(BlueprintError::Parse(_))));
    }

    #[test]
    fn parse_extra_tokens_fails() {
        let s = b"LF-BLUEPRINT 1\nSIZE 2 2 9\n"; // extra token in SIZE
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(matches!(res, Err(BlueprintError::Parse(_))));
        let s2 = b"LF-BLUEPRINT 1\nSIZE 2 2\nTILE 0 0 Or - EXTRA\n"; // extra token in TILE
        let res2 = Blueprint::load_from_reader(BufReader::new(&s2[..]));
        assert!(matches!(res2, Err(BlueprintError::Parse(_))));
    }

    #[test]
    fn parse_valid_minimal_blueprint_succeeds() {
        let s = b"LF-BLUEPRINT 1\nSIZE 2 2\n";
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(res.is_ok());
        let bp = res.unwrap();
        assert_eq!(bp.width, 2);
        assert_eq!(bp.height, 2);
        assert!(bp.tiles.is_empty());
    }

    #[test]
    fn parse_valid_mixed_tiles_succeeds() {
        let s = b"LF-BLUEPRINT 1\nSIZE 3 3\nTILE 0 0 or -\nTILE 1 1 AND 0x0001\nTILE 2 2 ClockGlobal 0xFFFF\n";
        let res = Blueprint::load_from_reader(BufReader::new(&s[..]));
        assert!(res.is_ok());
        let bp = res.unwrap();
        assert_eq!(bp.tiles.len(), 3);
    }

    #[test]
    fn export_empty_simulation() {
        let sim = Simulation::new();
        let bp = Blueprint::from_simulation(&sim);
        assert_eq!(bp.width, WIDTH as u32);
        assert_eq!(bp.height, HEIGHT as u32);
        assert_eq!(bp.tiles.len(), WIDTH * HEIGHT);
        for t in &bp.tiles {
            assert_eq!(t.tile_type, TileType::Wire);
            assert_eq!(t.logic, None);
        }
    }

    #[test]
    fn export_with_set_tiles() {
        let mut sim = Simulation::new();
        sim.set_tile(1, 1, TileType::Or);
        sim.set_tile(2, 2, TileType::And);

        let bp = Blueprint::from_simulation(&sim);
        // Helper to fetch tile from bp by coords
        let get = |x: usize, y: usize| -> &BlueprintTile {
            let idx = y * WIDTH + x;
            &bp.tiles[idx]
        };
        assert_eq!(get(1, 1).tile_type, TileType::Or);
        assert_eq!(get(2, 2).tile_type, TileType::And);
    }

    #[test]
    fn export_with_logic_bits() {
        let mut sim = Simulation::new();
        sim.set_tile(3, 3, TileType::Wire);
        // Write non-zero logic so it appears as Some(..)
        assert!(sim.set_logic_value(3, 3, 0xFFFF));

        let bp = Blueprint::from_simulation(&sim);
        let idx = 3 + 3 * WIDTH;
        assert_eq!(bp.tiles[idx].logic, Some(0xFFFF));
        // And a nearby zero remains None
        let idx2 = 4 + 3 * WIDTH;
        assert_eq!(bp.tiles[idx2].logic, None);
    }

    #[test]
    fn row_major_ordering_is_deterministic() {
        let sim = Simulation::new();
        let bp = Blueprint::from_simulation(&sim);
        for (i, t) in bp.tiles.iter().enumerate() {
            let x = i % WIDTH;
            let y = i / WIDTH;
            assert_eq!(t.x as usize, x);
            assert_eq!(t.y as usize, y);
        }
    }

    #[test]
    fn apply_empty_blueprint() {
        let bp = Blueprint::new(WIDTH as u32, HEIGHT as u32);
        let mut sim = Simulation::new();
        // Seed sim with a non-default to ensure reset happens
        sim.set_tile(1, 1, TileType::And);
        bp.apply_to_simulation(&mut sim).expect("apply ok");
        // After apply, default tiles should be Wire (as Simulation::new), unless overridden
        assert_eq!(
            sim.tilemap.get_tile(1, 1).unwrap().meta.tile_type,
            TileType::Wire
        );
    }

    #[test]
    fn apply_basic_tiles() {
        let mut bp = Blueprint::new(WIDTH as u32, HEIGHT as u32);
        bp.tiles.push(BlueprintTile {
            x: 2,
            y: 3,
            z: 0,
            tile_type: TileType::Or,
            logic: None,
        });
        bp.tiles.push(BlueprintTile {
            x: 4,
            y: 5,
            z: 0,
            tile_type: TileType::And,
            logic: None,
        });

        let mut sim = Simulation::new();
        bp.apply_to_simulation(&mut sim).expect("apply ok");
        assert_eq!(
            sim.tilemap.get_tile(2, 3).unwrap().meta.tile_type,
            TileType::Or
        );
        assert_eq!(
            sim.tilemap.get_tile(4, 5).unwrap().meta.tile_type,
            TileType::And
        );
    }

    #[test]
    fn apply_logic_bits() {
        let mut bp = Blueprint::new(WIDTH as u32, HEIGHT as u32);
        bp.tiles.push(BlueprintTile {
            x: 10,
            y: 10,
            z: 0,
            tile_type: TileType::Wire,
            logic: Some(0xABCD),
        });
        let mut sim = Simulation::new();
        bp.apply_to_simulation(&mut sim).expect("apply ok");
        let v = sim.tilemap.value_at(10, 10).unwrap();
        assert_eq!(v, 0xABCD);
    }

    #[test]
    fn apply_rejects_oob_coordinates_and_leaves_sim_untouched() {
        let mut bp = Blueprint::new(WIDTH as u32, HEIGHT as u32);
        // OOB x
        bp.tiles.push(BlueprintTile {
            x: (WIDTH as u32) + 1,
            y: 0,
            z: 0,
            tile_type: TileType::Or,
            logic: None,
        });
        let mut sim = Simulation::new();
        // Seed sim with distinctive value to ensure untouched on error
        sim.set_tile(0, 0, TileType::And);
        let res = bp.apply_to_simulation(&mut sim);
        assert!(matches!(res, Err(BlueprintError::Parse(_))));
        // Should remain as seeded (And), not reset
        assert_eq!(
            sim.tilemap.get_tile(0, 0).unwrap().meta.tile_type,
            TileType::And
        );
    }

    #[test]
    fn invertibility_roundtrip() {
        let mut sim = Simulation::new();
        // Build some state
        sim.set_tile(1, 1, TileType::Or);
        sim.set_tile(2, 2, TileType::And);
        assert!(sim.set_logic_value(1, 1, 0x1111));
        assert!(sim.set_logic_value(2, 2, 0x2222));

        let bp1 = Blueprint::from_simulation(&sim);

        let mut sim2 = Simulation::new();
        bp1.apply_to_simulation(&mut sim2).expect("apply ok");
        let bp2 = Blueprint::from_simulation(&sim2);

        assert_eq!(bp1, bp2);
    }
}
