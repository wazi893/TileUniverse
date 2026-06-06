// Logic Fabric - CHUNGUS 2 Helpers

pub const C2_HEAD: u32 = 50;
pub const C2_REG: u32 = 51;

/// Directions for Neighbors
pub enum Direction {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
}

/// Helper to get Register Index from Direction relative to CPU
pub fn dir_to_reg(d: Direction) -> u8 {
    match d {
        Direction::N => 0,
        Direction::NE => 1,
        Direction::E => 2,
        Direction::SE => 3,
        Direction::S => 4,
        Direction::SW => 5,
        Direction::W => 6,
        Direction::NW => 7,
    }
}

/// Helper to get Offset (dx, dy) for a Direction
pub fn dir_offset(d: Direction) -> (i32, i32) {
    match d {
        Direction::N => (0, -1),
        Direction::NE => (1, -1),
        Direction::E => (1, 0),
        Direction::SE => (1, 1),
        Direction::S => (0, 1),
        Direction::SW => (-1, 1),
        Direction::W => (-1, 0),
        Direction::NW => (-1, -1),
    }
}
