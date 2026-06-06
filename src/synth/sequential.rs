//! Sequential circuit wrapper for the synth pipeline.
//!
//! Composes a synthesized combinational export (next-state logic) with
//! software-maintained state feedback. The existing synth pipeline handles
//! combinational synthesis; this module adds the sequential wrapper.
//!
//! State bits live as a `Vec<bool>` (not physical Register8 tiles). Each tick:
//! 1. Drive state-feedback Const inputs from the state array
//! 2. Drive external inputs
//! 3. Propagate combinational logic to convergence
//! 4. Read next-state from combinational outputs
//! 5. Update the state array
//!
//! This avoids grid resizing, via breakage, and clock management complexity.
//! The combinational export grid is used exactly as-is.

use crate::synth::export::{SynthExport, evaluate_exported, export_to_simulation};
use crate::synth::placement::{PlaceConfig, place_mapped_netlist};
use crate::synth::routing::{RouteConfig, route_placed_netlist};
use crate::synth::{
    Aig, CellLibrary, SynthConfig, materialize_inversions, synthesize as synth_compile,
};

/// A sequential circuit: combinational next-state logic + software-maintained state.
pub struct SequentialCircuit {
    pub num_state_bits: usize,
    pub num_inputs: usize,
    pub num_outputs: usize,
    /// The combinational synth export (used as-is, no grid modifications).
    export: Option<SynthExport>,
    /// Current state bits (LSB first).
    state: Vec<bool>,
    /// Current cycle count.
    num_cycles: u64,
    /// For shift register: no synth export, just software logic.
    is_shift_register: bool,
}

/// Error type for sequential circuit construction.
#[derive(Debug)]
pub enum SeqError {
    Synth(String),
}

impl std::fmt::Display for SeqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeqError::Synth(msg) => write!(f, "Sequential synth error: {}", msg),
        }
    }
}

impl SequentialCircuit {
    /// Build a sequential circuit from an AIG where the first `num_latches`
    /// inputs are state feedback and the first `num_latches` outputs are
    /// next-state. Remaining inputs/outputs are external.
    pub fn from_aig(aig: &Aig, num_latches: usize) -> Result<Self, SeqError> {
        let total_inputs = aig.num_inputs() as usize;
        let total_outputs = aig.output_lits().count();

        if num_latches > total_inputs {
            return Err(SeqError::Synth(format!(
                "num_latches ({}) > AIG inputs ({})",
                num_latches, total_inputs
            )));
        }
        if num_latches > total_outputs {
            return Err(SeqError::Synth(format!(
                "num_latches ({}) > AIG outputs ({})",
                num_latches, total_outputs
            )));
        }

        // Synthesize the combinational AIG.
        let lib = CellLibrary::tile_native();
        let synth_result = synth_compile(aig, &lib, &SynthConfig::default());
        let materialized = materialize_inversions(&synth_result.netlist, &lib);
        let place_config = PlaceConfig::standalone();
        let route_config = RouteConfig::standalone();
        let placed = place_mapped_netlist(&materialized, &lib, &place_config)
            .map_err(|e| SeqError::Synth(format!("placement: {}", e)))?;
        let routed = route_placed_netlist(&materialized, &placed, &route_config)
            .map_err(|e| SeqError::Synth(format!("routing: {}", e)))?;
        let export = export_to_simulation(&routed, &materialized);

        let num_ext_inputs = total_inputs - num_latches;
        let num_ext_outputs = total_outputs - num_latches;

        Ok(SequentialCircuit {
            num_state_bits: num_latches,
            num_inputs: num_ext_inputs,
            num_outputs: num_ext_outputs,
            export: Some(export),
            state: vec![false; num_latches],
            num_cycles: 0,
            is_shift_register: false,
        })
    }

    /// Build an N-bit binary counter (no external inputs, output = state).
    ///
    /// AIG: n inputs (state), n outputs (next-state).
    /// The visible count is read via `read_state()`.
    pub fn counter(bits: u32) -> Result<Self, SeqError> {
        if bits == 0 || bits > 8 {
            return Err(SeqError::Synth("counter bits must be 1-8".to_string()));
        }
        let n = bits as usize;

        let mut aig = Aig::new();
        let mut state_inputs = Vec::with_capacity(n);
        for i in 0..n {
            state_inputs.push(aig.add_input(&format!("s{}", i)));
        }

        // Increment logic:
        //   next[0] = NOT state[0]
        //   next[k] = state[k] XOR carry_in[k]
        //   carry_in[1] = state[0]
        //   carry_in[k] = AND(state[0], ..., state[k-1])
        let next0 = aig.not(state_inputs[0]);
        aig.add_output("ns0", next0);

        if n > 1 {
            let mut carry = state_inputs[0];
            for k in 1..n {
                let xor_val = aig.xor(state_inputs[k], carry);
                aig.add_output(&format!("ns{}", k), xor_val);
                if k < n - 1 {
                    carry = aig.and(carry, state_inputs[k]);
                }
            }
        }

        Self::from_aig(&aig, n)
    }

    /// Build an N-bit shift register (1-bit serial input, output = state).
    ///
    /// Shift logic is purely wiring, so no synth pipeline is used.
    /// next[k] = current[k+1] for k < n-1, next[n-1] = serial_in.
    pub fn shift_register(bits: u32) -> Result<Self, SeqError> {
        if bits == 0 || bits > 8 {
            return Err(SeqError::Synth(
                "shift register bits must be 1-8".to_string(),
            ));
        }
        let n = bits as usize;

        Ok(SequentialCircuit {
            num_state_bits: n,
            num_inputs: 1,
            num_outputs: 0,
            export: None,
            state: vec![false; n],
            num_cycles: 0,
            is_shift_register: true,
        })
    }

    /// Read current state as a vector of bools (LSB first).
    pub fn read_state(&self) -> Vec<bool> {
        self.state.clone()
    }

    /// Read state as an integer (for convenience).
    pub fn state_value(&self) -> u64 {
        let mut val = 0u64;
        for (i, &b) in self.state.iter().enumerate() {
            if b {
                val |= 1 << i;
            }
        }
        val
    }

    /// Set state directly (LSB first).
    pub fn set_state(&mut self, state: &[bool]) {
        assert_eq!(state.len(), self.num_state_bits);
        self.state = state.to_vec();
    }

    /// Set state from an integer value.
    pub fn set_state_value(&mut self, val: u64) {
        for i in 0..self.num_state_bits {
            self.state[i] = (val >> i) & 1 != 0;
        }
    }

    /// Perform one clock cycle. Returns external outputs (empty for counter/shift).
    pub fn tick(&mut self, inputs: &[bool]) -> Vec<bool> {
        assert_eq!(
            inputs.len(),
            self.num_inputs,
            "wrong number of external inputs"
        );

        if self.is_shift_register {
            self.tick_shift_register(inputs)
        } else {
            self.tick_synth(inputs)
        }
    }

    fn tick_synth(&mut self, ext_inputs: &[bool]) -> Vec<bool> {
        let export = self.export.as_mut().expect("synth export required");
        let n = self.num_state_bits;

        // Build combined input vector: [state_feedback..., ext_inputs...]
        let mut all_inputs = Vec::with_capacity(n + ext_inputs.len());
        all_inputs.extend_from_slice(&self.state);
        all_inputs.extend_from_slice(ext_inputs);

        // Evaluate combinational logic using the synth export.
        let all_outputs = evaluate_exported(export, &all_inputs);

        // First n outputs are next-state, rest are external outputs.
        let next_state = &all_outputs[..n];
        let ext_outputs = all_outputs[n..].to_vec();

        // Update state.
        self.state = next_state.to_vec();
        self.num_cycles += 1;
        ext_outputs
    }

    fn tick_shift_register(&mut self, inputs: &[bool]) -> Vec<bool> {
        let n = self.num_state_bits;
        let serial_in = inputs[0];

        // Shift: next[k] = current[k+1], next[n-1] = serial_in.
        let mut next = Vec::with_capacity(n);
        for k in 0..n {
            if k < n - 1 {
                next.push(self.state[k + 1]);
            } else {
                next.push(serial_in);
            }
        }
        self.state = next;
        self.num_cycles += 1;
        vec![]
    }

    /// Tick with no external inputs (for circuits like counters).
    pub fn tick_no_inputs(&mut self) -> Vec<bool> {
        self.tick(&[])
    }

    /// Current cycle count.
    pub fn cycle_count(&self) -> u64 {
        self.num_cycles
    }

    /// Grid dimensions for inspection (None for shift register).
    pub fn grid_info(&self) -> Option<(usize, usize, usize)> {
        self.export
            .as_ref()
            .map(|e| (e.sim.width(), e.sim.height(), e.sim.num_layers()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn state_to_u8(state: &[bool]) -> u8 {
        let mut val = 0u8;
        for (i, &b) in state.iter().enumerate() {
            if b {
                val |= 1 << i;
            }
        }
        val
    }

    #[test]
    fn test_counter_aig_correctness() {
        // Verify the increment AIG produces correct truth table.
        use crate::synth::PlaceConfig;
        use crate::synth::export::evaluate_exported;
        use crate::synth::mapping::evaluate_aig;
        use crate::synth::routing::RouteConfig;

        let n = 3usize;
        let mut aig = Aig::new();
        let mut state_inputs = Vec::with_capacity(n);
        for i in 0..n {
            state_inputs.push(aig.add_input(&format!("s{}", i)));
        }
        let next0 = aig.not(state_inputs[0]);
        aig.add_output("ns0", next0);
        let mut carry = state_inputs[0];
        for k in 1..n {
            let xor_val = aig.xor(state_inputs[k], carry);
            aig.add_output(&format!("ns{}", k), xor_val);
            if k < n - 1 {
                carry = aig.and(carry, state_inputs[k]);
            }
        }

        // Test AIG evaluation directly.
        for combo in 0..8u32 {
            let inputs: Vec<bool> = (0..3).map(|i| (combo >> i) & 1 != 0).collect();
            let outputs = evaluate_aig(&aig, &inputs);
            let expected = (combo + 1) % 8;
            let got_bits: Vec<bool> = (0..3).map(|i| (expected >> i) & 1 != 0).collect();
            assert_eq!(
                outputs, got_bits,
                "AIG eval: input={} expected_next={} got={:?}",
                combo, expected, outputs
            );
        }

        // Test synth export evaluation.
        let lib = CellLibrary::tile_native();
        let synth_result = crate::synth::synthesize(&aig, &lib, &SynthConfig::default());
        let materialized = crate::synth::materialize_inversions(&synth_result.netlist, &lib);
        let place_config = PlaceConfig::standalone();
        let route_config = RouteConfig::standalone();
        let placed = crate::synth::place_mapped_netlist(&materialized, &lib, &place_config)
            .expect("placement");
        let routed = crate::synth::route_placed_netlist(&materialized, &placed, &route_config)
            .expect("routing");
        let mut export = crate::synth::export_to_simulation(&routed, &materialized);

        for combo in 0..8u32 {
            let inputs: Vec<bool> = (0..3).map(|i| (combo >> i) & 1 != 0).collect();
            let outputs = evaluate_exported(&mut export, &inputs);
            let expected = (combo + 1) % 8;
            let got_bits: Vec<bool> = (0..3).map(|i| (expected >> i) & 1 != 0).collect();
            assert_eq!(
                outputs, got_bits,
                "Synth export: input={} expected_next={} got={:?}",
                combo, expected, outputs
            );
        }
    }

    #[test]
    fn test_sequential_counter_3bit() {
        let mut circuit = SequentialCircuit::counter(3).expect("3-bit counter");
        assert_eq!(circuit.num_state_bits, 3);
        assert_eq!(circuit.num_inputs, 0);
        assert_eq!(circuit.num_outputs, 0);

        // Initial state should be 0.
        assert_eq!(circuit.state_value(), 0);

        // Tick through 0→1→2→...→7→0.
        for expected in 1..=8u8 {
            circuit.tick_no_inputs();
            let got = circuit.state_value();
            assert_eq!(
                got,
                (expected % 8) as u64,
                "cycle {}: expected {}, got {}",
                expected,
                expected % 8,
                got,
            );
        }

        // Verify wrap: one more tick should give 1.
        circuit.tick_no_inputs();
        assert_eq!(circuit.state_value(), 1);
    }

    #[test]
    fn test_sequential_counter_8bit() {
        let mut circuit = SequentialCircuit::counter(8).expect("8-bit counter");
        assert_eq!(circuit.num_state_bits, 8);

        for expected in 0..=255u16 {
            assert_eq!(
                circuit.state_value(),
                expected as u64,
                "before tick: expected {}, got {}",
                expected,
                circuit.state_value(),
            );
            circuit.tick_no_inputs();
        }
        // After 256 ticks, should wrap to 0.
        assert_eq!(circuit.state_value(), 0);
    }

    #[test]
    fn test_sequential_shift_register_4bit() {
        let mut circuit = SequentialCircuit::shift_register(4).expect("4-bit shift register");
        assert_eq!(circuit.num_state_bits, 4);
        assert_eq!(circuit.num_inputs, 1);

        // Shift in: 1, 0, 1, 1
        let serial_bits = [true, false, true, true];
        for &bit in &serial_bits {
            circuit.tick(&[bit]);
        }

        // After shifting in 1,0,1,1:
        //   cycle 1: [0,0,0,0] → [0,0,0,1] (serial=1 enters at MSB)
        //   cycle 2: [0,0,0,1] → [0,0,1,0] (serial=0)
        //   cycle 3: [0,0,1,0] → [0,1,0,1] (serial=1)
        //   cycle 4: [0,1,0,1] → [1,0,1,1] (serial=1)
        assert_eq!(circuit.read_state(), vec![true, false, true, true]);
    }

    #[test]
    fn test_sequential_set_state() {
        let mut circuit = SequentialCircuit::counter(3).expect("3-bit counter");

        // Set state to 5 (101 binary).
        circuit.set_state(&[true, false, true]);
        assert_eq!(circuit.state_value(), 5);

        // Tick: should go 5→6→7→0→1.
        circuit.tick_no_inputs();
        assert_eq!(circuit.state_value(), 6);
        circuit.tick_no_inputs();
        assert_eq!(circuit.state_value(), 7);
        circuit.tick_no_inputs();
        assert_eq!(circuit.state_value(), 0);
        circuit.tick_no_inputs();
        assert_eq!(circuit.state_value(), 1);
    }

    #[test]
    fn test_sequential_with_external_inputs() {
        let mut circuit = SequentialCircuit::shift_register(4).expect("4-bit shift register");

        // All zeros initially.
        assert_eq!(circuit.read_state(), vec![false; 4]);

        // Shift in alternating bits for 8 cycles: 1,0,1,0,1,0,1,0
        for cycle in 0..8u32 {
            let bit = cycle % 2 == 0;
            circuit.tick(&[bit]);
        }

        // Last 4 bits shifted in (cycles 4,5,6,7): true, false, true, false
        // Position 0 = oldest of last 4 = cycle 4 = true
        assert_eq!(circuit.read_state(), vec![true, false, true, false]);
    }
}
