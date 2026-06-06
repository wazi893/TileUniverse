//! CHUNGUS 3 Compiler (ADDM)
//!
//! Transforms high-level instructions (Instruction Enum) into a physical layout
//! of tiles (Sparse Tile Map).

use std::collections::HashMap;

// --- ISA Definition ---
#[derive(Clone, Debug)]
pub enum Instruction {
    /// A source node that provides a constant value (or initial value).
    /// Args: (Value, Tag)
    Source(u32, u32),

    /// An Addition Operation.
    /// Args: (Input1 Node ID, Input2 Node ID)
    Add(usize, usize),

    /// A Multiplication Operation.
    Mul(usize, usize),

    /// A Merge Node (Loop). Priority: Input1 (West), then Input2 (North).
    /// Args: (Input1 Node ID, Input2 Node ID)
    /// A Merge Node (Loop). Priority: Input1 (West), then Input2 (North).
    Merge(usize, usize),

    /// A static pixel/wall for games/physics.
    /// Args: (RelX, RelY, ColorTag)
    Pixel(i32, i32, u32), // RelX, RelY, Color/Value

    /// A stochastic scatter pin (Galton Board).
    /// Args: (RelX, RelY)
    Scatter(i32, i32), // RelX, RelY (Stochastic Pin)

    /// An Adder Unit placed at a relative position.
    /// Args: (RelX, RelY)
    AddUnit(i32, i32), // RelX, RelY (Adder Unit)

    /// A Multiplier Unit placed at a relative position.
    /// Args: (RelX, RelY)
    MulUnit(i32, i32), // RelX, RelY (Multiplier Unit)
}

// --- Tile Types ---
pub const TYPE_WIRE: u8 = 0;
pub const TYPE_ADD: u8 = 60;
pub const TYPE_MUL: u8 = 61;
pub const TYPE_TOKEN: u8 = 62;
pub const TYPE_OPTICAL: u8 = 63;
pub const TYPE_MERGE: u8 = 66;
pub const TYPE_SOURCE: u8 = 67;
pub const TYPE_WIRE_W: u8 = 70;
pub const TYPE_WIRE_N: u8 = 71;
pub const TYPE_WIRE_S: u8 = 72;
pub const TYPE_SCATTER: u8 = 75;

// --- Compiler ---
pub struct Compiler {
    instructions: Vec<Instruction>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    /// Adds an instruction and returns its Node ID.
    pub fn add_instr(&mut self, instr: Instruction) -> usize {
        self.instructions.push(instr);
        self.instructions.len() - 1
    }

    /// Compile the graph into a sparse tile map and initial states.
    /// Returns: (TileMap, InitialStates)
    /// TileMap: HashMap<(x, y), type_id>
    /// InitialStates: HashMap<(x, y), state_val>
    pub fn compile(
        &self,
        start_x: u32,
        start_y: u32,
    ) -> (HashMap<(u32, u32), u8>, HashMap<(u32, u32), u32>) {
        let mut tiles = HashMap::new();
        let mut states = HashMap::new();

        // Simple "Waterfall" Layout Strategy
        // Just place nodes in a chaotic column for now?
        // No, let's try 1D Horizontal placement with spacing, and route wires.
        // Actually, for Fibonacci, manual placement via a builder might be better.
        // But let's try auto-placement:
        // x = start_x + node_id * 5
        // y = start_y

        let mut node_positions = HashMap::new();

        // 1. Place Functional Units
        for (id, instr) in self.instructions.iter().enumerate() {
            let px = start_x + (id as u32) * 6; // Spacing 6
            let py = start_y;

            node_positions.insert(id, (px, py));

            match instr {
                Instruction::Source(val, tag) => {
                    tiles.insert((px, py), TYPE_TOKEN);
                    // Initial State
                    let state = (*tag << 16) | (0x80 << 8) | (*val & 0xFF);
                    states.insert((px, py), state);
                }
                Instruction::Add(_, _) => {
                    tiles.insert((px, py), TYPE_ADD);
                    // Inputs for ADD are North (px, py-1) and South (px, py+1)
                    // Outputs East (px+1, py)
                }
                Instruction::Merge(_, _) => {
                    tiles.insert((px, py), TYPE_MERGE);
                }
                Instruction::Mul(_, _) => {
                    tiles.insert((px, py), TYPE_MUL);
                }
                Instruction::Pixel(rx, ry, color) => {
                    // Manual placement relative to start
                    let abs_x = (start_x as i32 + rx) as u32;
                    let abs_y = (start_y as i32 + ry) as u32;
                    tiles.insert((abs_x, abs_y), TYPE_TOKEN);
                    // Value = 255 (Wall/Static), Meta = Valid (0x80), Tag = Color
                    let state = (*color << 16) | (0x80 << 8) | 255;
                    states.insert((abs_x, abs_y), state);
                }
                Instruction::Scatter(rx, ry) => {
                    let abs_x = (start_x as i32 + rx) as u32;
                    let abs_y = (start_y as i32 + ry) as u32;
                    tiles.insert((abs_x, abs_y), TYPE_SCATTER);
                    // No initial state needed, or maybe nice color?
                    // Let's set state to give it a specific color (e.g., White Pin)
                    // Tag = 7 (White)
                    let state = (7 << 16) | (0x80 << 8) | 255;
                    states.insert((abs_x, abs_y), state);
                }
                Instruction::AddUnit(rx, ry) => {
                    let abs_x = (start_x as i32 + rx) as u32;
                    let abs_y = (start_y as i32 + ry) as u32;
                    tiles.insert((abs_x, abs_y), TYPE_ADD);
                    states.insert((abs_x, abs_y), 0); // Empty initially
                }
                Instruction::MulUnit(rx, ry) => {
                    let abs_x = (start_x as i32 + rx) as u32;
                    let abs_y = (start_y as i32 + ry) as u32;
                    tiles.insert((abs_x, abs_y), TYPE_MUL);
                    states.insert((abs_x, abs_y), 0);
                }
            }
        }

        // 2. Route Wires
        for (id, instr) in self.instructions.iter().enumerate() {
            let (target_x, target_y) = node_positions[&id];

            match instr {
                Instruction::Source(_, _) => {
                    // Output is East (target_x+1, target_y)
                }
                Instruction::Add(src1, src2) => {
                    // Logic: ADD Inputs are at (N, S).
                    // Avoid Collision: Route to (target_x-1, input_y) and pull from West.

                    // Route 1: Src1 -> ADD.North (target_x, target_y - 1)
                    // Approach at (target_x - 1, target_y - 1)
                    self.route(
                        &mut tiles,
                        self.get_output_port(*src1, &node_positions),
                        (target_x - 1, target_y - 1),
                    );
                    // Turn Corner: Must flow East into the stub. Use TYPE_TOKEN.
                    tiles.insert((target_x - 1, target_y - 1), TYPE_TOKEN);

                    // Input stub (North of ADD, flows South)
                    tiles.insert((target_x, target_y - 1), TYPE_WIRE_S);

                    // Route 2: Src2 -> ADD.South (target_x, target_y + 1)
                    // Approach at (target_x - 1, target_y + 1)
                    self.route(
                        &mut tiles,
                        self.get_output_port(*src2, &node_positions),
                        (target_x - 1, target_y + 1),
                    );
                    // Turn Corner: Must flow East into the stub. Use TYPE_TOKEN.
                    tiles.insert((target_x - 1, target_y + 1), TYPE_TOKEN);

                    // Input stub (South of ADD, flows North)
                    tiles.insert((target_x, target_y + 1), TYPE_WIRE_N);

                    // Output stub
                    tiles.insert((target_x + 1, target_y), TYPE_TOKEN);
                }
                Instruction::Merge(src1, src2) => {
                    // Merge Inputs: West (src1), North (src2)
                    let p1 = self.get_output_port(*src1, &node_positions);
                    let p2 = self.get_output_port(*src2, &node_positions);

                    // Route src1 -> Merge.West (target_x-1, target_y)
                    self.route(&mut tiles, p1, (target_x - 1, target_y));
                    tiles.insert((target_x - 1, target_y), TYPE_TOKEN);

                    // Route src2 -> Merge.North (target_x, target_y - 1)
                    // Safe to route directly to (target_x, target_y - 1) as we come from North/West usually.
                    // But to be consistent with "Approach", maybe?
                    // Nah, North input is accessible from Y-1.
                    self.route(&mut tiles, p2, (target_x, target_y - 1));
                    // For Merge North input, we feed FROM North (S->N flow?? No).
                    // We feed into Merge.North. So we are AT North.
                    // Merge reads from North neighbor.
                    // So we must hold the value. TYPE_WIRE_S (N->S) holds value and pushes South?
                    // Yes. And Merge consumes it.
                    tiles.insert((target_x, target_y - 1), TYPE_WIRE_S); // Explicitly set to WIRE_S

                    tiles.insert((target_x + 1, target_y), TYPE_TOKEN);
                }
                _ => {}
            }
        }

        (tiles, states)
    }

    fn get_output_port(&self, id: usize, positions: &HashMap<usize, (u32, u32)>) -> (u32, u32) {
        let (px, py) = positions[&id];
        match self.instructions[id] {
            Instruction::Source(_, _) => (px, py), // Source is the token itself
            Instruction::Merge(_, _) => (px + 1, py), // Merge output is East
            Instruction::Add(_, _) => (px + 1, py), // Add output is East
            _ => (px + 1, py),
        }
    }

    // Simple Manhattan Router
    // Dogleg Router to avoid collisions on Y=20
    fn route(&self, tiles: &mut HashMap<(u32, u32), u8>, start: (u32, u32), end: (u32, u32)) {
        // Channel Y = 10.
        // Path: Start -> (Start.x, 10) -> (End.x, 10) -> End.

        let (sx, sy) = start;
        let (ex, _ey) = end;
        let channel_y = 10;

        // Segment 1: Start -> Channel (Vertical)
        self.route_segment(tiles, start, (sx, channel_y));
        tiles.insert(
            (sx, channel_y),
            if channel_y > sy {
                TYPE_WIRE_S
            } else {
                TYPE_WIRE_N
            },
        );

        // Segment 2: Across Channel (Horizontal)
        self.route_segment(tiles, (sx, channel_y), (ex, channel_y));
        tiles.insert(
            (ex, channel_y),
            if ex > sx { TYPE_TOKEN } else { TYPE_WIRE_W },
        );

        // Segment 3: Channel -> End (Vertical)
        self.route_segment(tiles, (ex, channel_y), end);
    }

    // Low-level Manhattan Segment (Generic)
    fn route_segment(
        &self,
        tiles: &mut HashMap<(u32, u32), u8>,
        start: (u32, u32),
        end: (u32, u32),
    ) {
        let (mut cur_x, mut cur_y) = start;
        let (end_x, end_y) = end;

        // X First
        let x_type = if end_x > cur_x {
            TYPE_TOKEN
        } else {
            TYPE_WIRE_W
        };

        while cur_x != end_x {
            if cur_x < end_x {
                cur_x += 1;
            } else {
                cur_x -= 1;
            }
            if (cur_x, cur_y) != end {
                tiles.insert((cur_x, cur_y), x_type); // Overwrite safe?
            }
        }

        // Y Second
        let y_type = if end_y > cur_y {
            TYPE_WIRE_S
        } else {
            TYPE_WIRE_N
        };

        while cur_y != end_y {
            if cur_y < end_y {
                cur_y += 1;
            } else {
                cur_y -= 1;
            }
            if (cur_x, cur_y) != end {
                tiles.insert((cur_x, cur_y), y_type);
            }
        }
    }
}
