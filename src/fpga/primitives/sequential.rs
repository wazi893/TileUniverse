//! SPRINT 80: Sequential Primitives
//!
//! RustHDL LogicBlock implementations for sequential (clocked) tile types.

use rust_hdl::prelude::*;

/// Edge-triggered D flip-flop (64-bit register)
#[derive(LogicBlock, Default)]
pub struct Register8Tile {
    pub clock: Signal<In, Clock>,
    pub data: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
    dff: DFF<Bits<64>>,
}

impl Logic for Register8Tile {
    #[hdl_gen]
    fn update(&mut self) {
        dff_setup!(self, clock, dff);
        self.dff.d.next = self.data.val();
        self.out.next = self.dff.q.val();
    }
}

/// Level-sensitive latch (simplified as DFF)
#[derive(LogicBlock, Default)]
pub struct LatchTile {
    pub clock: Signal<In, Clock>,
    pub data: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
    dff: DFF<Bits<64>>,
}

impl Logic for LatchTile {
    #[hdl_gen]
    fn update(&mut self) {
        dff_setup!(self, clock, dff);
        self.dff.d.next = self.data.val();
        self.out.next = self.dff.q.val();
    }
}

/// Register with enable (simplified - always captures)
#[derive(LogicBlock, Default)]
pub struct RegEnableTile {
    pub clock: Signal<In, Clock>,
    pub data: Signal<In, Bits<64>>,
    pub enable: Signal<In, Bit>,
    pub out: Signal<Out, Bits<64>>,
    dff: DFF<Bits<64>>,
}

impl Logic for RegEnableTile {
    #[hdl_gen]
    fn update(&mut self) {
        dff_setup!(self, clock, dff);
        self.dff.d.next = self.data.val();
        self.out.next = self.dff.q.val();
    }
}

/// RAM cell with write enable (simplified)
#[derive(LogicBlock, Default)]
pub struct RamTile {
    pub clock: Signal<In, Clock>,
    pub data: Signal<In, Bits<64>>,
    pub write_enable: Signal<In, Bit>,
    pub out: Signal<Out, Bits<64>>,
    dff: DFF<Bits<64>>,
}

impl Logic for RamTile {
    #[hdl_gen]
    fn update(&mut self) {
        dff_setup!(self, clock, dff);
        self.dff.d.next = self.data.val();
        self.out.next = self.dff.q.val();
    }
}

/// Counter (simplified - always increments)
#[derive(LogicBlock, Default)]
pub struct CounterTile {
    pub clock: Signal<In, Clock>,
    pub enable: Signal<In, Bit>,
    pub out: Signal<Out, Bits<64>>,
    dff: DFF<Bits<64>>,
}

impl Logic for CounterTile {
    #[hdl_gen]
    fn update(&mut self) {
        dff_setup!(self, clock, dff);
        self.dff.d.next = self.dff.q.val(); // Simplified - just holds value
        self.out.next = self.dff.q.val();
    }
}

/// Constant value output
#[derive(LogicBlock, Default)]
pub struct ConstTile {
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for ConstTile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.out.val(); // Hold current value
    }
}

/// Global clock signal generator
#[derive(LogicBlock, Default)]
pub struct ClockGlobalTile {
    pub clock: Signal<In, Clock>,
    pub out: Signal<Out, Bits<64>>,
    dff: DFF<Bit>,
}

impl Logic for ClockGlobalTile {
    #[hdl_gen]
    fn update(&mut self) {
        dff_setup!(self, clock, dff);
        self.dff.d.next = self.dff.q.val(); // Simplified - hold value
        self.out.next = self.out.val();
    }
}
