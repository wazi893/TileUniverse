//! SPRINT 80: CPU Building Block Primitives
//!
//! RustHDL LogicBlock implementations for CPU-specific tile types.

use rust_hdl::prelude::*;

/// 3-to-8 decoder (simplified)
#[derive(LogicBlock, Default)]
pub struct Decoder3to8Tile {
    pub data_in: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for Decoder3to8Tile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.data_in.val();
    }
}

/// 8-to-1 multiplexer (simplified)
#[derive(LogicBlock, Default)]
pub struct Mux8to1Tile {
    pub data: Signal<In, Bits<64>>,
    pub select: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for Mux8to1Tile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.data.val();
    }
}

/// 1-to-8 demultiplexer (simplified)
#[derive(LogicBlock, Default)]
pub struct Demux1to8Tile {
    pub data: Signal<In, Bits<64>>,
    pub select: Signal<In, Bits<64>>,
    pub out: Signal<Out, Bits<64>>,
}

impl Logic for Demux1to8Tile {
    #[hdl_gen]
    fn update(&mut self) {
        self.out.next = self.data.val();
    }
}

/// Program counter (simplified - just captures input)
#[derive(LogicBlock, Default)]
pub struct ProgramCounterTile {
    pub clock: Signal<In, Clock>,
    pub load_value: Signal<In, Bits<64>>,
    pub jump: Signal<In, Bit>,
    pub out: Signal<Out, Bits<64>>,
    dff: DFF<Bits<64>>,
}

impl Logic for ProgramCounterTile {
    #[hdl_gen]
    fn update(&mut self) {
        dff_setup!(self, clock, dff);
        self.dff.d.next = self.load_value.val();
        self.out.next = self.dff.q.val();
    }
}
