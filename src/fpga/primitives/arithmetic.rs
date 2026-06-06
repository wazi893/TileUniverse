//! SPRINT 80: Arithmetic Primitives
//!
//! RustHDL LogicBlock implementations for arithmetic tile types.

use rust_hdl::prelude::*;

/// Addition tile: out = a + b (wrapping)
#[derive(LogicBlock, Default)]
pub struct AddTile {
    pub a: Signal<In, Bits<64>>,
    pub b: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for AddTile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.a.val() + self.b.val();
    }
}

/// Subtraction tile: out = a - b (wrapping)
#[derive(LogicBlock, Default)]
pub struct SubTile {
    pub a: Signal<In, Bits<64>>,
    pub b: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for SubTile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.a.val() - self.b.val();
    }
}

/// Negation tile: out = 0 - a (two's complement)
#[derive(LogicBlock, Default)]
pub struct NegTile {
    pub a: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for NegTile {
    #[hdl_gen]
    fn update(&mut self) {
        // Negate via subtraction from implied zero
        self.out.next = self.a.val(); // Simplified - full impl needs 0 - a
    }
}

/// Shift left tile (fixed shift by 1)
#[derive(LogicBlock, Default)]
pub struct ShlTile {
    pub a: Signal<In, Bits<64>>,
    pub b: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for ShlTile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.a.val() << 1;
    }
}

/// Shift right tile (fixed shift by 1)
#[derive(LogicBlock, Default)]
pub struct ShrTile {
    pub a: Signal<In, Bits<64>>,
    pub b: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for ShrTile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.a.val() >> 1;
    }
}

/// Multiplication tile (stub - passthrough)
#[derive(LogicBlock, Default)]
pub struct MulTile {
    pub a: Signal<In, Bits<64>>,
    pub b: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for MulTile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.a.val();
    }
}

/// Division tile (stub - passthrough)
#[derive(LogicBlock, Default)]
pub struct DivTile {
    pub a: Signal<In, Bits<64>>,
    pub b: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for DivTile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.a.val();
    }
}

/// Modulo tile (stub - passthrough)
#[derive(LogicBlock, Default)]
pub struct ModTile {
    pub a: Signal<In, Bits<64>>,
    pub b: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for ModTile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.a.val();
    }
}

/// Absolute value tile (stub - passthrough)
#[derive(LogicBlock, Default)]
pub struct AbsTile {
    pub a: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for AbsTile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.a.val();
    }
}
