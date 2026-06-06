//! EPIC 110: Tile Primitives
//!
//! Reusable tile-based components for building CPUs:
//! - Half Adder, Full Adder, Ripple Adder
//! - Registers, Muxes, Decoders
//!
//! Signal convention:
//! - Horizontal: Data flows left → right
//! - Vertical: Control signals from up/down
//! - Carry propagation: Down

use crate::slim_simulation::SlimSimulation;
use crate::tile_meta::TileType;
use std::collections::HashMap;

/// A single tile placement within a component
#[derive(Clone, Debug)]
pub struct TilePlacement {
    pub rel_x: i32,
    pub rel_y: i32,
    pub tile_type: TileType,
    pub initial_value: u64,
}

impl TilePlacement {
    pub fn new(rel_x: i32, rel_y: i32, tile_type: TileType) -> Self {
        Self {
            rel_x,
            rel_y,
            tile_type,
            initial_value: 0,
        }
    }

    pub fn with_value(mut self, value: u64) -> Self {
        self.initial_value = value;
        self
    }
}

/// A reusable tile-based component
#[derive(Clone, Debug)]
pub struct TileComponent {
    /// Tile placements relative to origin (0,0)
    pub tiles: Vec<TilePlacement>,
    /// Named input positions (relative coordinates)
    pub inputs: HashMap<String, (i32, i32)>,
    /// Named output positions (relative coordinates)
    pub outputs: HashMap<String, (i32, i32)>,
    /// Bounding box width
    pub width: i32,
    /// Bounding box height
    pub height: i32,
    /// Ticks from input change to output stable
    pub propagation_delay: u32,
}

impl TileComponent {
    /// Create a new empty component
    pub fn new() -> Self {
        Self {
            tiles: Vec::new(),
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            width: 0,
            height: 0,
            propagation_delay: 1,
        }
    }

    /// Add a tile to the component
    pub fn add_tile(&mut self, rel_x: i32, rel_y: i32, tile_type: TileType) -> &mut Self {
        self.tiles.push(TilePlacement::new(rel_x, rel_y, tile_type));
        self.update_bounds(rel_x, rel_y);
        self
    }

    /// Add a tile with initial value
    pub fn add_tile_with_value(
        &mut self,
        rel_x: i32,
        rel_y: i32,
        tile_type: TileType,
        value: u64,
    ) -> &mut Self {
        self.tiles
            .push(TilePlacement::new(rel_x, rel_y, tile_type).with_value(value));
        self.update_bounds(rel_x, rel_y);
        self
    }

    /// Define a named input position
    pub fn add_input(&mut self, name: &str, rel_x: i32, rel_y: i32) -> &mut Self {
        self.inputs.insert(name.to_string(), (rel_x, rel_y));
        self
    }

    /// Define a named output position
    pub fn add_output(&mut self, name: &str, rel_x: i32, rel_y: i32) -> &mut Self {
        self.outputs.insert(name.to_string(), (rel_x, rel_y));
        self
    }

    /// Set propagation delay
    pub fn with_delay(mut self, ticks: u32) -> Self {
        self.propagation_delay = ticks;
        self
    }

    fn update_bounds(&mut self, x: i32, y: i32) {
        self.width = self.width.max(x + 1);
        self.height = self.height.max(y + 1);
    }

    // =========================================================================
    // PRIMITIVE CONSTRUCTORS
    // =========================================================================

    /// Half Adder: 2 inputs (A, B) → 2 outputs (Sum, Carry)
    ///
    /// Minimal layout with just logic gates.
    /// Caller must place Const tiles at input positions for A and B.
    ///
    /// Layout (3x2):
    /// ```text
    ///        col 0     col 1     col 2
    ///   row 0: [Const A] [XOR]    [Const B]   <- Sum = A ^ B
    ///   row 1: [Const A] [AND]    [Const B]   <- Carry = A & B
    ///
    ///   A and B must be duplicated because XOR/AND read from left/right.
    /// ```
    ///
    /// Note: The Const tiles ARE part of the component. To change inputs,
    /// set values on the Const tiles at (0,0), (0,1) for A and (2,0), (2,1) for B.
    ///
    /// Propagation: 1 tick
    pub fn half_adder() -> Self {
        let mut comp = Self::new();

        // Column 0: A input (duplicated for both rows)
        comp.add_tile(0, 0, TileType::Const); // A for XOR
        comp.add_tile(0, 1, TileType::Const); // A for AND

        // Column 1: Logic gates
        comp.add_tile(1, 0, TileType::Xor); // Sum = left(A) ^ right(B)
        comp.add_tile(1, 1, TileType::And); // Carry = left(A) & right(B)

        // Column 2: B input (duplicated for both rows)
        comp.add_tile(2, 0, TileType::Const); // B for XOR
        comp.add_tile(2, 1, TileType::Const); // B for AND

        // Inputs: A at column 0, B at column 2
        comp.add_input("A0", 0, 0); // A for XOR row
        comp.add_input("A1", 0, 1); // A for AND row
        comp.add_input("B0", 2, 0); // B for XOR row
        comp.add_input("B1", 2, 1); // B for AND row

        // Outputs: Read from gate tiles
        comp.add_output("Sum", 1, 0); // XOR tile value
        comp.add_output("Carry", 1, 1); // AND tile value

        comp.propagation_delay = 1;
        comp
    }

    /// Full Adder: 3 inputs (A, B, Cin) → 2 outputs (Sum, Cout)
    ///
    /// Sum = A XOR B XOR Cin
    /// Cout = (A AND B) OR ((A XOR B) AND Cin)
    ///
    /// Layout (3x5) - all inputs as Const tiles:
    /// ```text
    ///        col 0      col 1     col 2
    ///   row 0: [Const A]  [XOR1]   [Const B]    <- P = A ^ B
    ///   row 1: [Const P]  [XOR2]   [Const Cin]  <- Sum = P ^ Cin
    ///   row 2: [Const A]  [AND1]   [Const B]    <- G = A & B
    ///   row 3: [Const P]  [AND2]   [Const Cin]  <- C2 = P & Cin
    ///   row 4: [Const G]  [OR]     [Const C2]   <- Cout = G | C2
    /// ```
    ///
    /// Requires multi-stage evaluation - use `eval_full_adder()` helper.
    ///
    /// Propagation: 3 stages
    pub fn full_adder() -> Self {
        let mut comp = Self::new();

        // Row 0: First XOR (P = A ^ B)
        comp.add_tile(0, 0, TileType::Const); // A
        comp.add_tile(1, 0, TileType::Xor); // XOR1 → P
        comp.add_tile(2, 0, TileType::Const); // B

        // Row 1: Second XOR (Sum = P ^ Cin)
        comp.add_tile(0, 1, TileType::Const); // P (copied from XOR1)
        comp.add_tile(1, 1, TileType::Xor); // XOR2 → Sum
        comp.add_tile(2, 1, TileType::Const); // Cin

        // Row 2: First AND (G = A & B)
        comp.add_tile(0, 2, TileType::Const); // A
        comp.add_tile(1, 2, TileType::And); // AND1 → G
        comp.add_tile(2, 2, TileType::Const); // B

        // Row 3: Second AND (C2 = P & Cin)
        comp.add_tile(0, 3, TileType::Const); // P (copied from XOR1)
        comp.add_tile(1, 3, TileType::And); // AND2 → C2
        comp.add_tile(2, 3, TileType::Const); // Cin

        // Row 4: OR (Cout = G | C2)
        comp.add_tile(0, 4, TileType::Const); // G (copied from AND1)
        comp.add_tile(1, 4, TileType::Or); // OR → Cout
        comp.add_tile(2, 4, TileType::Const); // C2 (copied from AND2)

        // Primary inputs (A, B, Cin locations)
        comp.add_input("A0", 0, 0);
        comp.add_input("A2", 0, 2);
        comp.add_input("B0", 2, 0);
        comp.add_input("B2", 2, 2);
        comp.add_input("Cin1", 2, 1);
        comp.add_input("Cin3", 2, 3);

        // Intermediate copy destinations
        comp.add_input("P1", 0, 1); // Copy XOR1 output here
        comp.add_input("P3", 0, 3); // Copy XOR1 output here
        comp.add_input("G4", 0, 4); // Copy AND1 output here
        comp.add_input("C2_4", 2, 4); // Copy AND2 output here

        // Outputs
        comp.add_output("XOR1", 1, 0); // P = A ^ B
        comp.add_output("Sum", 1, 1); // Sum = P ^ Cin
        comp.add_output("AND1", 1, 2); // G = A & B
        comp.add_output("AND2", 1, 3); // C2 = P & Cin
        comp.add_output("Cout", 1, 4); // Cout = G | C2

        comp.propagation_delay = 3; // 3 stages
        comp
    }

    /// 8-bit Register using RAM tiles
    ///
    /// Layout (8x1):
    /// ```text
    ///   [RAM][RAM][RAM][RAM][RAM][RAM][RAM][RAM]
    ///
    ///   Data in: from left (directly connected to left neighbors)
    ///   Write enable: from up (directly connected to up neighbors)
    ///   Data out: to right
    /// ```
    ///
    /// Propagation: 1 tick
    pub fn register_8bit() -> Self {
        let mut comp = Self::new();

        // 8 RAM tiles in a row
        for bit in 0..8 {
            comp.add_tile(bit, 0, TileType::Ram);
        }

        // Inputs: data comes from left of each bit, enable from up
        for bit in 0..8 {
            comp.add_input(&format!("D{}", bit), bit - 1, 0);
        }
        comp.add_input("WE", 0, -1); // Write enable (shared, but enters at bit 0)

        // Outputs: data goes to right
        for bit in 0..8 {
            comp.add_output(&format!("Q{}", bit), bit + 1, 0);
        }

        comp.propagation_delay = 1;
        comp
    }

    /// 8-bit 2-to-1 Multiplexer
    ///
    /// Layout (8x1):
    /// ```text
    ///   [MUX][MUX][MUX][MUX][MUX][MUX][MUX][MUX]
    ///
    ///   Select: from up
    ///   Input A: from left  (selected when sel=0)
    ///   Input B: from right (selected when sel=1)
    ///   Output: value selected
    /// ```
    ///
    /// Mux tile semantics: output = (up != 0) ? right : left
    /// - sel=0: output = left (A)
    /// - sel=1: output = right (B)
    ///
    /// Propagation: 1 tick
    pub fn mux_2to1_8bit() -> Self {
        let mut comp = Self::new();

        // 8 Mux tiles in a row
        for bit in 0..8 {
            comp.add_tile(bit, 0, TileType::Mux);
        }

        // Select signal from up
        comp.add_input("Sel", 0, -1);

        // Input A from left, Input B from right (for each bit)
        for bit in 0..8 {
            comp.add_input(&format!("A{}", bit), bit - 1, 0);
            comp.add_input(&format!("B{}", bit), bit + 1, 0);
        }

        // Output goes down
        for bit in 0..8 {
            comp.add_output(&format!("Y{}", bit), bit, 1);
        }

        comp.propagation_delay = 1;
        comp
    }

    /// 2-to-4 Decoder
    ///
    /// Converts 2-bit input to one-hot 4-bit output:
    ///   00 → 0001
    ///   01 → 0010
    ///   10 → 0100
    ///   11 → 1000
    ///
    /// Layout (4x3):
    /// ```text
    ///   [NOT]─S0─     [NOT]─S1─
    ///     │             │
    ///   [AND] ~S1&~S0 [AND] ~S1&S0 [AND] S1&~S0 [AND] S1&S0
    /// ```
    ///
    /// Propagation: 2 ticks
    pub fn decoder_2to4() -> Self {
        let mut comp = Self::new();

        // NOT gates for inverted select lines
        comp.add_tile(0, 0, TileType::Not); // ~S0
        comp.add_tile(2, 0, TileType::Not); // ~S1

        // AND gates for each output
        comp.add_tile(0, 2, TileType::And); // Out0: ~S1 & ~S0
        comp.add_tile(1, 2, TileType::And); // Out1: ~S1 & S0
        comp.add_tile(2, 2, TileType::And); // Out2: S1 & ~S0
        comp.add_tile(3, 2, TileType::And); // Out3: S1 & S0

        // Wires to route signals
        comp.add_tile(0, 1, TileType::Wire); // Route ~S0
        comp.add_tile(2, 1, TileType::Wire); // Route ~S1
        comp.add_tile(1, 1, TileType::Wire); // Route S0
        comp.add_tile(3, 1, TileType::Wire); // Route S1

        // Inputs
        comp.add_input("S0", -1, 0);
        comp.add_input("S1", 1, 0);

        // Outputs
        comp.add_output("Out0", 0, 3);
        comp.add_output("Out1", 1, 3);
        comp.add_output("Out2", 2, 3);
        comp.add_output("Out3", 3, 3);

        comp.propagation_delay = 2;
        comp
    }

    /// 8-bit Ripple Carry Adder
    ///
    /// Chain of 8 full adders with carry propagation.
    ///
    /// Layout: 8 full adders stacked vertically, carry ripples down.
    ///
    /// Propagation: 16 ticks (worst case carry propagation)
    pub fn ripple_adder_8bit() -> Self {
        let mut comp = Self::new();

        // Each full adder is approximately 4 tiles wide, 3 tiles tall
        // Stack them vertically with carry chain

        let fa_height = 4;

        for bit in 0..8 {
            let y_offset = bit * fa_height;

            // Simplified full adder per bit:
            // XOR for partial sum
            comp.add_tile(0, y_offset, TileType::Xor); // A XOR B
            comp.add_tile(1, y_offset, TileType::Xor); // (A XOR B) XOR Cin = Sum

            // AND + OR for carry
            comp.add_tile(0, y_offset + 1, TileType::And); // A AND B
            comp.add_tile(1, y_offset + 1, TileType::And); // (A XOR B) AND Cin
            comp.add_tile(2, y_offset + 1, TileType::Or); // Carry out

            // Routing wires
            comp.add_tile(0, y_offset + 2, TileType::Wire); // Route B
            comp.add_tile(2, y_offset + 2, TileType::Wire); // Route carry down

            // Carry chain wire to next bit
            if bit < 7 {
                comp.add_tile(2, y_offset + 3, TileType::Wire);
            }

            // Inputs for this bit
            comp.add_input(&format!("A{}", bit), -1, y_offset);
            comp.add_input(&format!("B{}", bit), -1, y_offset + 2);

            // Output for this bit
            comp.add_output(&format!("S{}", bit), 2, y_offset);
        }

        // Carry in at bottom
        comp.add_input("Cin", 2, -1);
        // Carry out at top
        comp.add_output("Cout", 2, 8 * fa_height - 1);

        comp.propagation_delay = 16; // Worst case carry propagation
        comp
    }
}

impl Default for TileComponent {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PLACEMENT API
// =============================================================================

/// Place a component into a SlimSimulation at the given origin
pub fn place_component(sim: &mut SlimSimulation, comp: &TileComponent, origin: (i32, i32)) {
    let (ox, oy) = origin;

    for tile in &comp.tiles {
        let x = (ox + tile.rel_x) as usize;
        let y = (oy + tile.rel_y) as usize;

        if x < sim.width() && y < sim.height() {
            sim.set_tile(x, y, tile.tile_type);
            if tile.initial_value != 0 {
                sim.set_logic_value(x, y, tile.initial_value);
            }
        }
    }
}

/// Set a value at a component's named input
pub fn set_input(
    sim: &mut SlimSimulation,
    comp: &TileComponent,
    origin: (i32, i32),
    input_name: &str,
    value: u64,
) -> bool {
    if let Some(&(rx, ry)) = comp.inputs.get(input_name) {
        let x = (origin.0 + rx) as usize;
        let y = (origin.1 + ry) as usize;
        if x < sim.width() && y < sim.height() {
            sim.set_logic_value(x, y, value);
            return true;
        }
    }
    false
}

/// Get a value from a component's named output
pub fn get_output(
    sim: &SlimSimulation,
    comp: &TileComponent,
    origin: (i32, i32),
    output_name: &str,
) -> Option<u64> {
    if let Some(&(rx, ry)) = comp.outputs.get(output_name) {
        let x = (origin.0 + rx) as usize;
        let y = (origin.1 + ry) as usize;
        if x < sim.width() && y < sim.height() {
            return Some(sim.get_logic_at(x, y));
        }
    }
    None
}

/// Evaluate a full adder component (multi-stage)
///
/// Sets inputs A, B, Cin and runs the evaluation stages with proper
/// intermediate value propagation.
///
/// Returns (Sum, Cout)
pub fn eval_full_adder(
    sim: &mut SlimSimulation,
    _comp: &TileComponent,
    origin: (i32, i32),
    a: u64,
    b: u64,
    cin: u64,
) -> (u64, u64) {
    let (ox, oy) = (origin.0 as usize, origin.1 as usize);

    // Stage 1: Set primary inputs (A, B, Cin)
    sim.set_logic_value(ox, oy, a); // A at row 0
    sim.set_logic_value(ox, oy + 2, a); // A at row 2
    sim.set_logic_value(ox + 2, oy, b); // B at row 0
    sim.set_logic_value(ox + 2, oy + 2, b); // B at row 2
    sim.set_logic_value(ox + 2, oy + 1, cin); // Cin at row 1
    sim.set_logic_value(ox + 2, oy + 3, cin); // Cin at row 3

    // Tick to compute XOR1 (P) and AND1 (G)
    sim.eval_full_grid();

    // Read intermediate values
    let p = sim.get_logic_at(ox + 1, oy); // XOR1 output = P
    let g = sim.get_logic_at(ox + 1, oy + 2); // AND1 output = G

    // Stage 2: Copy P to rows 1 and 3
    sim.set_logic_value(ox, oy + 1, p); // P for XOR2
    sim.set_logic_value(ox, oy + 3, p); // P for AND2

    // Tick to compute XOR2 (Sum) and AND2 (C2)
    sim.eval_full_grid();

    // Read Sum and C2
    let sum = sim.get_logic_at(ox + 1, oy + 1); // XOR2 output = Sum
    let c2 = sim.get_logic_at(ox + 1, oy + 3); // AND2 output = C2

    // Stage 3: Copy G and C2 for OR
    sim.set_logic_value(ox, oy + 4, g); // G for OR
    sim.set_logic_value(ox + 2, oy + 4, c2); // C2 for OR

    // Tick to compute OR (Cout)
    sim.eval_full_grid();

    let cout = sim.get_logic_at(ox + 1, oy + 4); // OR output = Cout

    (sum, cout)
}

/// Connect two points with a wire path (simple horizontal then vertical routing)
pub fn connect_wire(sim: &mut SlimSimulation, from: (i32, i32), to: (i32, i32)) {
    let (x1, y1) = from;
    let (x2, y2) = to;

    // Horizontal segment
    let x_start = x1.min(x2);
    let x_end = x1.max(x2);
    for x in x_start..=x_end {
        let ux = x as usize;
        let uy = y1 as usize;
        if ux < sim.width() && uy < sim.height() {
            sim.set_tile(ux, uy, TileType::Wire);
        }
    }

    // Vertical segment
    let y_start = y1.min(y2);
    let y_end = y1.max(y2);
    for y in y_start..=y_end {
        let ux = x2 as usize;
        let uy = y as usize;
        if ux < sim.width() && uy < sim.height() {
            sim.set_tile(ux, uy, TileType::Wire);
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn run_ticks(sim: &mut SlimSimulation, n: usize) {
        for _ in 0..n {
            sim.eval_full_grid();
        }
    }

    /// Helper to set half adder inputs (A and B are duplicated internally)
    fn set_half_adder_inputs(sim: &mut SlimSimulation, origin: (i32, i32), a: u64, b: u64) {
        let (ox, oy) = (origin.0 as usize, origin.1 as usize);
        // A is at columns 0, rows 0 and 1
        sim.set_logic_value(ox, oy, a);
        sim.set_logic_value(ox, oy + 1, a);
        // B is at column 2, rows 0 and 1
        sim.set_logic_value(ox + 2, oy, b);
        sim.set_logic_value(ox + 2, oy + 1, b);
    }

    #[test]
    fn test_half_adder_truth_table() {
        // Half adder truth table:
        // A B | Sum Carry
        // 0 0 |  0    0
        // 0 1 |  1    0
        // 1 0 |  1    0
        // 1 1 |  0    1

        let test_cases = [
            (0u64, 0u64, 0u64, 0u64),          // A=0, B=0 → Sum=0, Carry=0
            (0, u64::MAX, u64::MAX, 0),        // A=0, B=1 → Sum=1, Carry=0
            (u64::MAX, 0, u64::MAX, 0),        // A=1, B=0 → Sum=1, Carry=0
            (u64::MAX, u64::MAX, 0, u64::MAX), // A=1, B=1 → Sum=0, Carry=1
        ];

        let ha = TileComponent::half_adder();

        for (a, b, expected_sum, expected_carry) in test_cases {
            let mut sim = SlimSimulation::with_size(16, 16);
            let origin = (4, 4);

            // Component layout (placed at origin 4,4):
            //   (4,4)=Const_A  (5,4)=XOR  (6,4)=Const_B
            //   (4,5)=Const_A  (5,5)=AND  (6,5)=Const_B

            place_component(&mut sim, &ha, origin);
            set_half_adder_inputs(&mut sim, origin, a, b);

            // Run simulation (1 tick propagation)
            run_ticks(&mut sim, 2);

            // Read outputs from gate tiles
            let sum = sim.get_logic_at(5, 4); // XOR at origin + (1,0)
            let carry = sim.get_logic_at(5, 5); // AND at origin + (1,1)

            assert_eq!(
                sum, expected_sum,
                "Half adder A={}, B={}: expected Sum={}, got {}",
                a, b, expected_sum, sum
            );
            assert_eq!(
                carry, expected_carry,
                "Half adder A={}, B={}: expected Carry={}, got {}",
                a, b, expected_carry, carry
            );
        }
    }

    #[test]
    fn test_full_adder_all_inputs() {
        // Full adder truth table:
        // A B Cin | Sum Cout
        // 0 0  0  |  0   0
        // 0 0  1  |  1   0
        // 0 1  0  |  1   0
        // 0 1  1  |  0   1
        // 1 0  0  |  1   0
        // 1 0  1  |  0   1
        // 1 1  0  |  0   1
        // 1 1  1  |  1   1

        let test_cases: [(u64, u64, u64, u64, u64); 8] = [
            (0, 0, 0, 0, 0),
            (0, 0, u64::MAX, u64::MAX, 0),
            (0, u64::MAX, 0, u64::MAX, 0),
            (0, u64::MAX, u64::MAX, 0, u64::MAX),
            (u64::MAX, 0, 0, u64::MAX, 0),
            (u64::MAX, 0, u64::MAX, 0, u64::MAX),
            (u64::MAX, u64::MAX, 0, 0, u64::MAX),
            (u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX),
        ];

        let fa = TileComponent::full_adder();

        for (a, b, cin, expected_sum, expected_cout) in test_cases {
            let mut sim = SlimSimulation::with_size(16, 16);
            let origin = (4, 4);

            place_component(&mut sim, &fa, origin);

            // Use the eval_full_adder helper which handles multi-stage evaluation
            let (sum, cout) = eval_full_adder(&mut sim, &fa, origin, a, b, cin);

            assert_eq!(
                sum,
                expected_sum,
                "Full adder A={}, B={}, Cin={}: expected Sum={}, got {}",
                if a != 0 { 1 } else { 0 },
                if b != 0 { 1 } else { 0 },
                if cin != 0 { 1 } else { 0 },
                if expected_sum != 0 { 1 } else { 0 },
                if sum != 0 { 1 } else { 0 }
            );
            assert_eq!(
                cout,
                expected_cout,
                "Full adder A={}, B={}, Cin={}: expected Cout={}, got {}",
                if a != 0 { 1 } else { 0 },
                if b != 0 { 1 } else { 0 },
                if cin != 0 { 1 } else { 0 },
                if expected_cout != 0 { 1 } else { 0 },
                if cout != 0 { 1 } else { 0 }
            );
        }
    }

    #[test]
    fn test_register_store_recall() {
        // Test a single RAM tile's behavior
        // RAM: output = (up != 0) ? left : current
        // - Write enable from UP
        // - Data from LEFT
        // - Holds value when WE is low

        let mut sim = SlimSimulation::with_size(16, 16);

        // Place one RAM tile at (5, 5)
        sim.set_tile(5, 5, TileType::Ram);

        // Place Const tiles for inputs
        sim.set_tile(4, 5, TileType::Const); // Data input (left of RAM)
        sim.set_tile(5, 4, TileType::Const); // Write enable (above RAM)

        // Test: Write value 0xFF
        sim.set_logic_value(4, 5, u64::MAX); // Data = 1
        sim.set_logic_value(5, 4, u64::MAX); // WE = 1

        sim.eval_full_grid();

        let stored = sim.get_logic_at(5, 5);
        assert_eq!(stored, u64::MAX, "RAM should store MAX when WE=1");

        // Now disable write and change data input
        sim.set_logic_value(5, 4, 0); // WE = 0
        sim.set_logic_value(4, 5, 0); // Data = 0 (but should not be stored)

        sim.eval_full_grid();

        let held = sim.get_logic_at(5, 5);
        assert_eq!(held, u64::MAX, "RAM should hold value when WE=0");

        // Enable write with new value
        sim.set_logic_value(5, 4, u64::MAX); // WE = 1
        sim.set_logic_value(4, 5, 0); // Data = 0

        sim.eval_full_grid();

        let updated = sim.get_logic_at(5, 5);
        assert_eq!(updated, 0, "RAM should update to 0 when WE=1 and Data=0");
    }

    #[test]
    fn test_mux_selection() {
        // Test a single Mux tile's behavior
        // Mux: output = (up != 0) ? right : left
        // - Selector from UP
        // - Input A from LEFT
        // - Input B from RIGHT
        // When sel=1, output=right(B). When sel=0, output=left(A).

        let mut sim = SlimSimulation::with_size(16, 16);

        // Place one Mux tile at (5, 5)
        sim.set_tile(5, 5, TileType::Mux);

        // Place Const tiles for inputs
        sim.set_tile(4, 5, TileType::Const); // Input A (left of Mux)
        sim.set_tile(6, 5, TileType::Const); // Input B (right of Mux)
        sim.set_tile(5, 4, TileType::Const); // Selector (above Mux)

        // Test 1: Sel=0 → output = left (A)
        sim.set_logic_value(4, 5, u64::MAX); // A = 1
        sim.set_logic_value(6, 5, 0); // B = 0
        sim.set_logic_value(5, 4, 0); // Sel = 0

        sim.eval_full_grid();

        let out1 = sim.get_logic_at(5, 5);
        assert_eq!(out1, u64::MAX, "Mux with Sel=0 should output A (left)");

        // Test 2: Sel=1 → output = right (B)
        sim.set_logic_value(4, 5, u64::MAX); // A = 1
        sim.set_logic_value(6, 5, 0); // B = 0
        sim.set_logic_value(5, 4, u64::MAX); // Sel = 1

        sim.eval_full_grid();

        let out2 = sim.get_logic_at(5, 5);
        assert_eq!(out2, 0, "Mux with Sel=1 should output B (right)");

        // Test 3: Swap values, Sel=0
        sim.set_logic_value(4, 5, 0); // A = 0
        sim.set_logic_value(6, 5, u64::MAX); // B = 1
        sim.set_logic_value(5, 4, 0); // Sel = 0

        sim.eval_full_grid();

        let out3 = sim.get_logic_at(5, 5);
        assert_eq!(out3, 0, "Mux with Sel=0 should output A=0");

        // Test 4: Swap values, Sel=1
        sim.set_logic_value(4, 5, 0); // A = 0
        sim.set_logic_value(6, 5, u64::MAX); // B = 1
        sim.set_logic_value(5, 4, u64::MAX); // Sel = 1

        sim.eval_full_grid();

        let out4 = sim.get_logic_at(5, 5);
        assert_eq!(out4, u64::MAX, "Mux with Sel=1 should output B=1");
    }

    #[test]
    fn test_component_dimensions() {
        let ha = TileComponent::half_adder();
        assert_eq!(ha.width, 3, "Half adder should be 3 wide");
        assert_eq!(ha.height, 2, "Half adder should be 2 tall");
        assert_eq!(ha.propagation_delay, 1);

        let fa = TileComponent::full_adder();
        assert_eq!(fa.width, 3, "Full adder should be 3 wide");
        assert_eq!(fa.height, 5, "Full adder should be 5 tall");
        assert_eq!(fa.propagation_delay, 3);

        let reg = TileComponent::register_8bit();
        assert_eq!(reg.width, 8, "8-bit register should be 8 wide");
        assert_eq!(reg.propagation_delay, 1);

        let mux = TileComponent::mux_2to1_8bit();
        assert_eq!(mux.width, 8, "8-bit mux should be 8 wide");
        assert_eq!(mux.propagation_delay, 1);
    }
}
