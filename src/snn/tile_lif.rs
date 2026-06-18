//! Quantized integrate-and-fire neuron as a logic datapath — the path toward a
//! LIF membrane that computes IN TILES.
//!
//! [`TileNeuron`](super::tile_neuron::TileNeuron) made the *binary* fire decision
//! (`>=T of k inputs`) a single ThresholdVia tile. A full spiking neuron also has
//! an integer **membrane** that accumulates input across ticks and resets on
//! spike — state plus a small datapath, not one via. This module builds the
//! combinational heart of that datapath as an AIG, which the synth pipeline maps
//! to real tiles (the same route `TileNeuron::fire_aig` takes).
//!
//! ## The step (integrate-and-fire; leak is the next feature)
//!
//! For a `bits`-wide unsigned membrane and a constant `threshold`:
//! ```text
//! sum     = v_mem + input                 (saturating to the bit width)
//! spike   = sum >= threshold
//! v_mem'  = if spike { 0 } else { sum }    (reset-to-zero on fire)
//! ```
//! This is integrate-and-fire — the leak-free core of LIF. Leak (a multiply of
//! `v_mem` by a `<1` factor each tick) is the next increment; the reference
//! [`LIFNeuron`](super::neuron::LIFNeuron) carries it. The *sequential* feedback
//! (a tile register holding `v_mem` between ticks) is a later step; here the
//! step is a pure combinational function `(v_mem, input) -> (v_mem', spike)`,
//! exhaustively checkable, which is exactly what synthesizes to tiles.

use crate::synth::aig::{Aig, AigLit};
use crate::synth::alphafabric::{AnnealConfig, Circuit, PlacementEnv, anneal};
use crate::synth::export::{SynthExport, evaluate_exported, export_to_simulation};
use crate::synth::routing::{RouteConfig, route_placed_netlist};

/// A quantized integrate-and-fire membrane step.
#[derive(Clone, Copy, Debug)]
pub struct QuantLif {
    /// Membrane bit width (unsigned). The membrane saturates at `2^bits - 1`.
    pub bits: u32,
    /// Firing threshold: spikes when the (saturated) membrane `>= threshold`.
    pub threshold: u64,
    /// Leak: each tick the membrane decays to `v_mem - (v_mem >> leak_shift)`
    /// (≈ a factor `1 - 2^-leak_shift`) before integrating input. `0` = no leak
    /// (pure integrate-and-fire). This power-of-two-style leak keeps the datapath
    /// a shift + subtract — the tiny first form of LIF's `v_mem * leak`.
    pub leak_shift: u32,
    /// Arbitrary fixed-point leak: when `Some(q)`, the membrane retains a factor
    /// `q/256` each tick — `leaked = (v_mem * q) >> 8` (Q0.8). `q = 256` is no
    /// leak (factor 1.0), `q = 192` is 0.75, `q = 128` is 0.5. This is the *full*
    /// LIF leak (any factor in `[0, 1]`, not just `1 - 2^-k`); the datapath is a
    /// constant shift-and-add multiply. Supersedes [`leak_shift`](Self::leak_shift)
    /// when set.
    pub leak_q8: Option<u16>,
}

impl QuantLif {
    /// Construct a step. Panics unless `1 <= bits <= 16` and
    /// `1 <= threshold <= 2^bits - 1`.
    pub fn new(bits: u32, threshold: u64) -> Self {
        assert!((1..=16).contains(&bits), "bits must be 1..=16");
        let max = (1u64 << bits) - 1;
        assert!(
            (1..=max).contains(&threshold),
            "threshold must be 1..=2^bits-1"
        );
        QuantLif {
            bits,
            threshold,
            leak_shift: 0,
            leak_q8: None,
        }
    }

    /// Set the leak shift: each tick the membrane decays by
    /// `v_mem >> leak_shift` (≈ factor `1 - 2^-leak_shift`). `0` = no leak.
    pub fn with_leak_shift(mut self, leak_shift: u32) -> Self {
        self.leak_shift = leak_shift;
        self
    }

    /// Set an arbitrary Q0.8 leak: the membrane retains a factor `q/256` each
    /// tick (`leaked = (v_mem * q) >> 8`). `q` ranges `0..=256` (256 = no leak).
    /// Supersedes any `leak_shift`. Panics unless `q <= 256`.
    pub fn with_leak_q8(mut self, q: u16) -> Self {
        assert!(q <= 256, "leak_q8 retention factor must be 0..=256 (q/256)");
        self.leak_q8 = Some(q);
        self
    }

    fn max(&self) -> u64 {
        (1u64 << self.bits) - 1
    }

    /// Leaked membrane. With `leak_q8 = Some(q)`: `(v_mem * q) >> 8` (Q0.8 factor
    /// `q/256`). Otherwise `v_mem - (v_mem >> leak_shift)`, or `v_mem` if no leak.
    fn leak(&self, v_mem: u64) -> u64 {
        if let Some(q) = self.leak_q8 {
            ((v_mem * q as u64) >> 8).min(self.max())
        } else if self.leak_shift == 0 {
            v_mem
        } else {
            v_mem - (v_mem >> self.leak_shift)
        }
    }

    /// Software reference for one step: `(v_mem, input) -> (v_mem', spike)`.
    pub fn step_reference(&self, v_mem: u64, input: u64) -> (u64, bool) {
        let max = self.max();
        let sum = (self.leak(v_mem) + input).min(max); // leak, then integrate (saturating)
        let spike = sum >= self.threshold;
        let v_out = if spike { 0 } else { sum };
        (v_out, spike)
    }

    /// Run the membrane over an input sequence from initial `v0`, threading the
    /// state — the multi-tick LIF dynamics as a `(v_mem', spike)` trace.
    pub fn run_reference(&self, v0: u64, inputs: &[u64]) -> Vec<(u64, bool)> {
        let mut v = v0;
        inputs
            .iter()
            .map(|&inp| {
                let (vn, spike) = self.step_reference(v, inp);
                v = vn;
                (vn, spike)
            })
            .collect()
    }

    /// Leak `v`, integrate `current` (saturating), and compare the result to the
    /// threshold. Returns the saturated membrane `sat` and the raw
    /// `over_threshold` bit — WITHOUT applying the reset, so callers that gate the
    /// spike (e.g. a refractory period) can decide the reset themselves. Shared by
    /// `membrane_update_aig` and [`RefractoryLif`].
    fn leak_integrate_threshold_aig(
        &self,
        aig: &mut Aig,
        v: &[AigLit],
        current: &[AigLit],
    ) -> (Vec<AigLit>, AigLit) {
        let w = self.bits as usize;
        // Leak. Q0.8 path: leaked = (v * q) >> 8 as a constant shift-and-add
        // multiply (add v<<i for each set bit i of q), then drop the low 8 bits.
        // Power-of-two path: leaked = v - (v >> leak_shift) = v + ~(v>>k) + 1.
        let leaked: Vec<AigLit> = if let Some(q) = self.leak_q8 {
            let pw = w + 9; // product fits in w+9 bits (q <= 256 => shift <= 8)
            let mut acc = vec![AigLit::FALSE; pw];
            for i in 0..=8u32 {
                if (q >> i) & 1 == 1 {
                    let shifted: Vec<AigLit> = (0..pw)
                        .map(|b| {
                            let bi = b as i64 - i as i64;
                            if bi >= 0 && (bi as usize) < w {
                                v[bi as usize]
                            } else {
                                AigLit::FALSE
                            }
                        })
                        .collect();
                    let (s, _) = ripple_add(aig, &acc, &shifted, AigLit::FALSE);
                    acc = s;
                }
            }
            (0..w).map(|b| acc[b + 8]).collect() // >> 8
        } else if self.leak_shift == 0 {
            v.to_vec()
        } else {
            let k = self.leak_shift as usize;
            let shifted: Vec<AigLit> = (0..w)
                .map(|i| if i + k < w { v[i + k] } else { AigLit::FALSE })
                .collect();
            let not_shifted: Vec<AigLit> = shifted.iter().map(|&s| aig.not(s)).collect();
            let (diff, _) = ripple_add(aig, v, &not_shifted, AigLit::TRUE);
            diff
        };
        // sum = leaked + current, saturating on carry-out.
        let (sum, carry) = ripple_add(aig, &leaked, current, AigLit::FALSE);
        let sat: Vec<AigLit> = sum.iter().map(|&s| aig.or(s, carry)).collect();
        // over_threshold = sat >= threshold  <=>  carry_out(sat + (2^w - T)).
        let comp_const = (1u64 << self.bits) - self.threshold;
        let const_bits = const_bus(&comp_const, w);
        let (_, over) = ripple_add(aig, &sat, &const_bits, AigLit::FALSE);
        (sat, over)
    }

    /// Membrane update gates into `aig`: leak `v`, integrate `current`
    /// (saturating), compare to threshold, reset on spike. Returns the next
    /// membrane bits and the spike. Shared by `step_aig` and `neuron_step_aig`.
    fn membrane_update_aig(
        &self,
        aig: &mut Aig,
        v: &[AigLit],
        current: &[AigLit],
    ) -> (Vec<AigLit>, AigLit) {
        let (sat, spike) = self.leak_integrate_threshold_aig(aig, v, current);
        // v_out = spike ? 0 : sat.
        let not_spike = aig.not(spike);
        let v_out: Vec<AigLit> = sat.iter().map(|&s| aig.and(s, not_spike)).collect();
        (v_out, spike)
    }

    /// The membrane step as an AIG. Inputs: bus `v` then `inp` (each `bits`, LSB
    /// first). Outputs: next `v_out` (`bits`) then `spike`.
    pub fn step_aig(&self) -> Aig {
        let mut aig = Aig::new();
        let v = aig.add_input_bus("v", self.bits);
        let inp = aig.add_input_bus("inp", self.bits);
        let (v_out, spike) = self.membrane_update_aig(&mut aig, &v, &inp);
        aig.add_output_bus("v_out", &v_out);
        aig.add_output("spike", spike);
        aig
    }

    /// A full neuron step as an AIG: inputs bus `v` (`bits`) then `k` synapse
    /// spikes; the synaptic current `Σ x_i·w_i` drives the membrane. Outputs next
    /// `v` (`bits`) then `spike`. The `v`-first layout suits `SequentialCircuit`
    /// (its first `bits` inputs/outputs are state feedback / next-state).
    pub fn neuron_step_aig(&self, synapses: &Synapses) -> Aig {
        assert_eq!(
            synapses.out_bits, self.bits,
            "synapse current width must equal membrane bits"
        );
        let mut aig = Aig::new();
        let v = aig.add_input_bus("v", self.bits);
        let spikes: Vec<AigLit> = (0..synapses.weights.len())
            .map(|i| aig.add_input(&format!("s{i}")))
            .collect();
        let current = synapses.sum_aig(&mut aig, &spikes);
        let (v_out, spike) = self.membrane_update_aig(&mut aig, &v, &current);
        aig.add_output_bus("v_out", &v_out);
        aig.add_output("spike", spike);
        aig
    }
}

/// A leaky integrate-and-fire membrane with an **absolute refractory period**:
/// after a spike, firing is suppressed for `refractory` ticks regardless of how
/// much current arrives. This is the last classic piece of the LIF model — and,
/// in tiles, it means the neuron carries a second piece of state (a refractory
/// countdown) alongside the membrane, both threaded between ticks.
///
/// The step is a pure combinational function of the *combined* state:
/// ```text
/// (sat, over) = leak+integrate+threshold(v_mem, input)
/// spike       = over AND (refr == 0)          (refractory gates the fire)
/// v_mem'      = if spike { 0 } else { sat }
/// refr'       = if spike { R } else { max(refr-1, 0) }   (countdown)
/// ```
/// The membrane keeps leaking/integrating during the refractory window; only the
/// *output spike* is gated, so the instant the countdown reaches 0 a still-charged
/// membrane fires again. State layout (LSB-first): `v_mem` (`bits`) then `refr`
/// (`refr_bits`); the single output is `v_mem' ++ refr' ++ spike`.
#[derive(Clone, Copy, Debug)]
pub struct RefractoryLif {
    /// The underlying leaky integrate-and-fire membrane.
    pub lif: QuantLif,
    /// Refractory period `R`: ticks to suppress firing after a spike. `0`
    /// recovers a plain [`QuantLif`] (the countdown is always 0 ⇒ never gated).
    pub refractory: u64,
}

impl RefractoryLif {
    /// Construct. `refractory` is the number of ticks firing is suppressed after
    /// a spike (`0` = no refractory period).
    pub fn new(lif: QuantLif, refractory: u64) -> Self {
        RefractoryLif { lif, refractory }
    }

    /// Bits needed to hold the refractory countdown (values `0..=R`).
    pub fn refr_bits(&self) -> u32 {
        if self.refractory == 0 {
            1
        } else {
            64 - self.refractory.leading_zeros()
        }
    }

    /// Software reference for one step over the combined state:
    /// `(v_mem, refr, input) -> (v_mem', refr', spike)`.
    pub fn step_reference(&self, v_mem: u64, refr: u64, input: u64) -> (u64, u64, bool) {
        let max = (1u64 << self.lif.bits) - 1;
        let sat = (self.lif.leak(v_mem) + input).min(max);
        let over = sat >= self.lif.threshold;
        let spike = over && refr == 0;
        let v_out = if spike { 0 } else { sat };
        let refr_out = if spike {
            self.refractory
        } else {
            refr.saturating_sub(1)
        };
        (v_out, refr_out, spike)
    }

    /// Run the refractory membrane over an input sequence from initial state,
    /// threading both the membrane and the countdown — the spike train with
    /// suppression gaps after each fire.
    pub fn run_reference(&self, v0: u64, refr0: u64, inputs: &[u64]) -> Vec<(u64, u64, bool)> {
        let mut v = v0;
        let mut r = refr0;
        inputs
            .iter()
            .map(|&inp| {
                let (vn, rn, spike) = self.step_reference(v, r, inp);
                v = vn;
                r = rn;
                (vn, rn, spike)
            })
            .collect()
    }

    /// The refractory datapath gates into `aig` given the synaptic `current` bus:
    /// leak+integrate+threshold, gate the spike by `refr == 0`, reset the
    /// membrane on fire, and update the countdown (set to `R` on fire, else
    /// decrement saturating at 0). Returns `(v_out, refr_out, spike)`. Shared by
    /// `step_aig` (current = direct input) and `neuron_step_aig` (current = the
    /// synaptic sum), so the membrane-only and full-neuron forms are one datapath.
    fn refractory_update_aig(
        &self,
        aig: &mut Aig,
        v: &[AigLit],
        refr: &[AigLit],
        current: &[AigLit],
    ) -> (Vec<AigLit>, Vec<AigLit>, AigLit) {
        let rb = self.refr_bits() as usize;
        // Leak + integrate + threshold (no reset yet — the spike is gated below).
        let (sat, over) = self.lif.leak_integrate_threshold_aig(aig, v, current);
        // can_fire = (refr == 0) = NOR of the countdown bits.
        let mut refr_nonzero = AigLit::FALSE;
        for &r in refr {
            refr_nonzero = aig.or(refr_nonzero, r);
        }
        let can_fire = aig.not(refr_nonzero);
        let spike = aig.and(over, can_fire);
        let not_spike = aig.not(spike);
        // v_out = spike ? 0 : sat.
        let v_out: Vec<AigLit> = sat.iter().map(|&s| aig.and(s, not_spike)).collect();
        // countdown = (refr - 1) masked to 0 when refr == 0:
        //   refr + (2^rb - 1) underflows to all-ones at refr==0, which the
        //   refr_nonzero mask clears — so countdown = max(refr-1, 0).
        let all_ones = const_bus(&((1u64 << rb) - 1), rb);
        let (dec, _) = ripple_add(aig, refr, &all_ones, AigLit::FALSE);
        let countdown: Vec<AigLit> = dec.iter().map(|&d| aig.and(d, refr_nonzero)).collect();
        // refr_out = spike ? R : countdown  (R is a constant bus).
        let r_const = const_bus(&self.refractory, rb);
        let refr_out: Vec<AigLit> = (0..rb)
            .map(|i| {
                let from_r = aig.and(spike, r_const[i]);
                let from_count = aig.and(not_spike, countdown[i]);
                aig.or(from_r, from_count)
            })
            .collect();
        (v_out, refr_out, spike)
    }

    /// The refractory membrane step as an AIG. Inputs (LSB-first): bus `v`
    /// (`bits`), bus `refr` (`refr_bits`), then bus `inp` (`bits`). Outputs:
    /// next `v_out` (`bits`), next `refr_out` (`refr_bits`), then `spike`.
    pub fn step_aig(&self) -> Aig {
        let mut aig = Aig::new();
        let v = aig.add_input_bus("v", self.lif.bits);
        let refr = aig.add_input_bus("refr", self.refr_bits());
        let inp = aig.add_input_bus("inp", self.lif.bits);
        let (v_out, refr_out, spike) = self.refractory_update_aig(&mut aig, &v, &refr, &inp);
        aig.add_output_bus("v_out", &v_out);
        aig.add_output_bus("refr_out", &refr_out);
        aig.add_output("spike", spike);
        aig
    }

    /// The FULL neuron step as an AIG — synapses + leak + refractory in one
    /// next-state circuit. Inputs (LSB-first): bus `v` (`bits`), bus `refr`
    /// (`refr_bits`), then `k` synapse spikes; the synaptic current `Σ xᵢ·wᵢ`
    /// drives the membrane. Outputs: next `v_out` (`bits`), next `refr_out`
    /// (`refr_bits`), then `spike`. The `v`-then-`refr` state layout matches
    /// [`TileRefractoryNeuron`]'s feedback.
    pub fn neuron_step_aig(&self, synapses: &Synapses) -> Aig {
        assert_eq!(
            synapses.out_bits, self.lif.bits,
            "synapse current width must equal membrane bits"
        );
        let mut aig = Aig::new();
        let v = aig.add_input_bus("v", self.lif.bits);
        let refr = aig.add_input_bus("refr", self.refr_bits());
        let spikes: Vec<AigLit> = (0..synapses.weights.len())
            .map(|i| aig.add_input(&format!("s{i}")))
            .collect();
        let current = synapses.sum_aig(&mut aig, &spikes);
        let (v_out, refr_out, spike) = self.refractory_update_aig(&mut aig, &v, &refr, &current);
        aig.add_output_bus("v_out", &v_out);
        aig.add_output_bus("refr_out", &refr_out);
        aig.add_output("spike", spike);
        aig
    }

    /// Software reference for the full neuron step: synaptic current from
    /// `spikes`, then the refractory membrane update.
    /// `(v_mem, refr, spikes) -> (v_mem', refr', spike)`.
    pub fn neuron_step_reference(
        &self,
        v_mem: u64,
        refr: u64,
        synapses: &Synapses,
        spikes: &[bool],
    ) -> (u64, u64, bool) {
        let current = synapses.current_reference(spikes);
        self.step_reference(v_mem, refr, current)
    }
}

/// Ripple-carry adder: returns (`w` sum bits LSB-first, carry-out).
fn ripple_add(aig: &mut Aig, a: &[AigLit], b: &[AigLit], cin: AigLit) -> (Vec<AigLit>, AigLit) {
    let mut carry = cin;
    let mut sum = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        let axb = aig.xor(a[i], b[i]);
        let s = aig.xor(axb, carry);
        let ab = aig.and(a[i], b[i]);
        let c2 = aig.and(carry, axb);
        carry = aig.or(ab, c2);
        sum.push(s);
    }
    (sum, carry)
}

/// `w`-bit constant as AIG literals, LSB first.
fn const_bus(value: &u64, w: usize) -> Vec<AigLit> {
    (0..w)
        .map(|i| {
            if (value >> i) & 1 == 1 {
                AigLit::TRUE
            } else {
                AigLit::FALSE
            }
        })
        .collect()
}

/// The synaptic front-end of a neuron: `k` binary input spikes, each scaled by a
/// constant weight, summed (and saturated) into one membrane input current.
///
/// A real spiking neuron integrates `current = Σ w_i · x_i` over its synapses;
/// for binary spikes `x_i`, `w_i · x_i` is just `x_i ? w_i : 0`, so the whole
/// sum is a gate-and-add of constant weights — a small datapath that maps to
/// tiles. Composes with [`QuantLif`]: its `current` is the `input` of a step.
#[derive(Clone, Debug)]
pub struct Synapses {
    /// One constant weight per synapse.
    pub weights: Vec<u64>,
    /// Output current bit width; the summed current saturates at `2^out_bits-1`.
    pub out_bits: u32,
}

impl Synapses {
    /// Construct. Panics if there are no synapses or `out_bits` is not `1..=16`.
    pub fn new(weights: Vec<u64>, out_bits: u32) -> Self {
        assert!(!weights.is_empty(), "need at least one synapse");
        assert!((1..=16).contains(&out_bits), "out_bits must be 1..=16");
        Synapses { weights, out_bits }
    }

    /// Software reference: `min(Σ x_i·w_i, 2^out_bits-1)`.
    pub fn current_reference(&self, spikes: &[bool]) -> u64 {
        let max = (1u64 << self.out_bits) - 1;
        let sum: u64 = spikes
            .iter()
            .zip(&self.weights)
            .map(|(&x, &w)| if x { w } else { 0 })
            .sum();
        sum.min(max)
    }

    /// Build the synaptic sum into `aig` from spike literals, returning the
    /// `out_bits`-wide saturated current. Shared by `current_aig` and the
    /// neuron's `neuron_step_aig` so the front-end composes with the membrane.
    fn sum_aig(&self, aig: &mut Aig, spikes: &[AigLit]) -> Vec<AigLit> {
        // Accumulate in a width wide enough to hold the full (unsaturated) sum.
        let sum_w: u64 = self.weights.iter().sum();
        let acc_bits = (64 - sum_w.max(1).leading_zeros()).max(self.out_bits) as usize;
        let mut acc: Vec<AigLit> = vec![AigLit::FALSE; acc_bits];
        for (i, &w) in self.weights.iter().enumerate() {
            // gated weight = spike_i ? w : 0  (w's 1-bits become spike_i).
            let gated: Vec<AigLit> = (0..acc_bits)
                .map(|b| {
                    if (w >> b) & 1 == 1 {
                        spikes[i]
                    } else {
                        AigLit::FALSE
                    }
                })
                .collect();
            let (s, _c) = ripple_add(aig, &acc, &gated, AigLit::FALSE);
            acc = s;
        }
        // Saturate to out_bits: overflow = OR of any bit at/above out_bits.
        let ob = self.out_bits as usize;
        let mut overflow = AigLit::FALSE;
        for b in acc.iter().skip(ob) {
            overflow = aig.or(overflow, *b);
        }
        acc[..ob].iter().map(|&a| aig.or(a, overflow)).collect()
    }

    /// The synaptic sum as a standalone AIG. Inputs: `k` spikes `s0..`. Output:
    /// bus `current` (`out_bits`, LSB first, saturating).
    pub fn current_aig(&self) -> Aig {
        let mut aig = Aig::new();
        let spikes: Vec<AigLit> = (0..self.weights.len())
            .map(|i| aig.add_input(&format!("s{i}")))
            .collect();
        let current = self.sum_aig(&mut aig, &spikes);
        aig.add_output_bus("current", &current);
        aig
    }
}

/// A complete LIF neuron computing in tiles: a synaptic front-end ([`Synapses`])
/// feeding a leaky integrate-and-fire membrane ([`QuantLif`]). The combined
/// next-state circuit (synapse sum → membrane step) is placed + routed + exported
/// to the tile simulator ONCE, then evaluated each tick with the `bits`-wide
/// membrane fed back. The per-tick compute is physical; the membrane state is
/// threaded in software.
///
/// Uses the AlphaFabric routing config (`max_z = 3`, no single-layer crossings) —
/// the same one that exports correctly elsewhere — rather than the standalone
/// router, which can't route these ~50-gate neuron circuits.
pub struct TileLifNeuron {
    lif: QuantLif,
    synapses: Synapses,
    export: SynthExport,
    state: u64,
}

impl TileLifNeuron {
    /// Build the neuron. `synapses.out_bits` must equal `lif.bits`.
    pub fn new(lif: QuantLif, synapses: Synapses) -> Self {
        let circuit = Circuit::from_aig("tile_lif_neuron", lif.neuron_step_aig(&synapses));
        let mut env = PlacementEnv::new(&circuit).expect("neuron placement env builds");
        let rc = RouteConfig {
            prefer_horizontal_first: true,
            no_crossings: true,
            max_z: 3,
        };
        // Cheap path first: try the baseline (row-major) placement. Most neurons
        // route as-is. Only when that fails do we pay for AlphaFabric's
        // route-validated SA placer to FIND a routable layout — the
        // NeuroAlphaFabric convergence (the optimized placer laying out a
        // spiking-neuron circuit). anneal leaves env holding its best layout.
        let routed = match route_placed_netlist(&circuit.netlist, &env.placed(), &rc) {
            Ok(r) => r,
            Err(_) => {
                let cfg = AnnealConfig {
                    route_validated_best: true,
                    iterations: 1200,
                    ..AnnealConfig::default()
                };
                anneal(&mut env, &cfg);
                route_placed_netlist(&circuit.netlist, &env.placed(), &rc)
                    .expect("neuron routes after route-validated SA placement")
            }
        };
        let export = export_to_simulation(&routed, &circuit.netlist);
        TileLifNeuron {
            lif,
            synapses,
            export,
            state: 0,
        }
    }

    /// Advance one tick with input spikes on each synapse; returns whether the
    /// neuron fired. The next-state compute runs on the tile simulator.
    pub fn tick(&mut self, spikes: &[bool]) -> bool {
        let bits = self.lif.bits as usize;
        // inputs: membrane bus `v` (state) then the synapse spikes.
        let mut ins: Vec<bool> = (0..bits).map(|i| (self.state >> i) & 1 == 1).collect();
        ins.extend_from_slice(spikes);
        let out = evaluate_exported(&mut self.export, &ins);
        // outputs: next membrane bus `v_out` then the spike.
        let v_next = (0..bits).fold(0u64, |acc, i| acc | ((out[i] as u64) << i));
        let spike = out[bits];
        self.state = v_next;
        spike
    }

    /// Current membrane potential (the threaded state).
    pub fn membrane(&self) -> u64 {
        self.state
    }

    /// Reset the membrane to rest (0) — e.g. before classifying a fresh input.
    pub fn reset(&mut self) {
        self.state = 0;
    }

    /// Software reference for the same neuron over a spike sequence: the
    /// `(membrane', spike)` trace, for differential checking against `tick`.
    pub fn run_reference(&self, v0: u64, spike_seq: &[Vec<bool>]) -> Vec<(u64, bool)> {
        let mut v = v0;
        spike_seq
            .iter()
            .map(|spikes| {
                let current = self.synapses.current_reference(spikes);
                let (vn, spike) = self.lif.step_reference(v, current);
                v = vn;
                (vn, spike)
            })
            .collect()
    }
}

/// The complete LIF neuron on tiles, **all features at once**: weighted synapses
/// + leaky integrate-and-fire (power-of-two or full Q0.8 leak) + an absolute
/// refractory period. Where [`TileLifNeuron`] threads only the membrane,
/// this threads the *combined* state — membrane `v_mem` AND the refractory
/// countdown `refr` — between ticks, with the per-tick next-state computed on the
/// tile simulator. This is the full LIF model as a single physical object.
///
/// Same build strategy as [`TileLifNeuron`]: try the cheap baseline (row-major)
/// placement first, falling back to AlphaFabric's route-validated SA placer only
/// when the neuron doesn't route as-is.
pub struct TileRefractoryNeuron {
    rlif: RefractoryLif,
    synapses: Synapses,
    export: SynthExport,
    v_state: u64,
    refr_state: u64,
}

impl TileRefractoryNeuron {
    /// Build the neuron. `synapses.out_bits` must equal `rlif.lif.bits`.
    pub fn new(rlif: RefractoryLif, synapses: Synapses) -> Self {
        let circuit = Circuit::from_aig("tile_refractory_neuron", rlif.neuron_step_aig(&synapses));
        let mut env = PlacementEnv::new(&circuit).expect("neuron placement env builds");
        let rc = RouteConfig {
            prefer_horizontal_first: true,
            no_crossings: true,
            max_z: 3,
        };
        let routed = match route_placed_netlist(&circuit.netlist, &env.placed(), &rc) {
            Ok(r) => r,
            Err(_) => {
                let cfg = AnnealConfig {
                    route_validated_best: true,
                    iterations: 1200,
                    ..AnnealConfig::default()
                };
                anneal(&mut env, &cfg);
                route_placed_netlist(&circuit.netlist, &env.placed(), &rc)
                    .expect("neuron routes after route-validated SA placement")
            }
        };
        let export = export_to_simulation(&routed, &circuit.netlist);
        TileRefractoryNeuron {
            rlif,
            synapses,
            export,
            v_state: 0,
            refr_state: 0,
        }
    }

    /// Advance one tick with input spikes on each synapse; returns whether the
    /// neuron fired. Both the membrane and the refractory countdown are threaded
    /// as state; the next-state compute runs on the tile simulator.
    pub fn tick(&mut self, spikes: &[bool]) -> bool {
        let bits = self.rlif.lif.bits as usize;
        let rb = self.rlif.refr_bits() as usize;
        // inputs: membrane bus `v`, refractory bus `refr`, then the synapse spikes.
        let mut ins: Vec<bool> = (0..bits).map(|i| (self.v_state >> i) & 1 == 1).collect();
        ins.extend((0..rb).map(|i| (self.refr_state >> i) & 1 == 1));
        ins.extend_from_slice(spikes);
        let out = evaluate_exported(&mut self.export, &ins);
        // outputs: next `v_out`, next `refr_out`, then the spike.
        let v_next = (0..bits).fold(0u64, |acc, i| acc | ((out[i] as u64) << i));
        let refr_next = (0..rb).fold(0u64, |acc, i| acc | ((out[bits + i] as u64) << i));
        let spike = out[bits + rb];
        self.v_state = v_next;
        self.refr_state = refr_next;
        spike
    }

    /// Current membrane potential (threaded state).
    pub fn membrane(&self) -> u64 {
        self.v_state
    }

    /// Current refractory countdown (threaded state).
    pub fn refractory(&self) -> u64 {
        self.refr_state
    }

    /// Reset both the membrane and the refractory countdown to rest.
    pub fn reset(&mut self) {
        self.v_state = 0;
        self.refr_state = 0;
    }

    /// Software reference for the same neuron over a spike sequence: the
    /// `(membrane', refr', spike)` trace, for differential checking against `tick`.
    pub fn run_reference(
        &self,
        v0: u64,
        refr0: u64,
        spike_seq: &[Vec<bool>],
    ) -> Vec<(u64, u64, bool)> {
        let mut v = v0;
        let mut r = refr0;
        spike_seq
            .iter()
            .map(|spikes| {
                let (vn, rn, spike) = self
                    .rlif
                    .neuron_step_reference(v, r, &self.synapses, spikes);
                v = vn;
                r = rn;
                (vn, rn, spike)
            })
            .collect()
    }
}

/// A small spiking LAYER: `N` independent LIF neurons (each a [`TileLifNeuron`]
/// — its own small, routable next-state circuit on tiles) reading one shared
/// input spike vector and producing an `N`-bit output spike vector each tick.
///
/// Keeping the neurons as separate tile circuits (rather than one monolithic
/// next-state AIG) is both the natural neuromorphic layout and what actually
/// routes: a combined `N`-neuron circuit overruns the standalone router, while
/// `N` small per-neuron circuits each place and route cleanly and scale.
pub struct TileLifLayer {
    neurons: Vec<TileLifNeuron>,
}

impl TileLifLayer {
    /// Build the layer: one neuron per [`Synapses`], all sharing `lif` params.
    /// All synapses must share the same input count and have `out_bits==lif.bits`.
    pub fn new(lif: QuantLif, synapses: Vec<Synapses>) -> Self {
        assert!(!synapses.is_empty(), "layer needs at least one neuron");
        let k = synapses[0].weights.len();
        assert!(
            synapses
                .iter()
                .all(|s| s.weights.len() == k && s.out_bits == lif.bits),
            "all neurons must share input count and have out_bits == lif.bits"
        );
        let neurons = synapses
            .into_iter()
            .map(|s| TileLifNeuron::new(lif, s))
            .collect();
        TileLifLayer { neurons }
    }

    /// Number of neurons.
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Whether the layer has no neurons (always false post-construction).
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Advance one tick with the shared input spike vector; returns each neuron's
    /// output spike. Each neuron's next-state runs on its own tile circuit.
    pub fn tick(&mut self, inputs: &[bool]) -> Vec<bool> {
        self.neurons.iter_mut().map(|n| n.tick(inputs)).collect()
    }

    /// Reset all neuron membranes to rest.
    pub fn reset(&mut self) {
        for n in &mut self.neurons {
            n.reset();
        }
    }

    /// Classify an input pattern: hold it for `ticks` steps (a simple rate code
    /// for binary features) and return the per-neuron output spike counts plus
    /// the argmax (the predicted class). Resets the membranes first.
    pub fn classify(&mut self, pattern: &[bool], ticks: usize) -> (Vec<u32>, usize) {
        self.reset();
        let mut counts = vec![0u32; self.neurons.len()];
        for _ in 0..ticks {
            for (j, fired) in self.tick(pattern).into_iter().enumerate() {
                counts[j] += fired as u32;
            }
        }
        let argmax = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .map(|(j, _)| j)
            .unwrap_or(0);
        (counts, argmax)
    }

    /// Software reference: per tick, each neuron's output spike (membranes start
    /// at 0), for differential checking against [`tick`](Self::tick).
    pub fn run_reference(&self, spike_seq: &[Vec<bool>]) -> Vec<Vec<bool>> {
        let per_neuron: Vec<Vec<(u64, bool)>> = self
            .neurons
            .iter()
            .map(|n| n.run_reference(0, spike_seq))
            .collect();
        (0..spike_seq.len())
            .map(|t| per_neuron.iter().map(|trace| trace[t].1).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::mapping::evaluate_aig;

    /// Pack a value into `bits` LSB-first booleans.
    fn bits_of(v: u64, bits: u32) -> Vec<bool> {
        (0..bits).map(|i| (v >> i) & 1 == 1).collect()
    }

    /// Unpack LSB-first booleans into a value.
    fn val_of(bits: &[bool]) -> u64 {
        bits.iter()
            .enumerate()
            .fold(0u64, |acc, (i, &b)| acc | ((b as u64) << i))
    }

    /// Place `circuit` so it routes LEGALLY under the corrected (layer-
    /// transition-exclusive) router: try the cheap row-major baseline, and on
    /// failure fall back to AlphaFabric's route-validated SA placer to FIND a
    /// routable layout. Mirrors the production `TileLifNeuron::new` fallback, so
    /// the "computes in tiles" proofs hold under the corrected router — not just
    /// against the dense baseline, whose only routes used the now-illegal
    /// phantom layer transitions.
    fn routable_env(circuit: &Circuit) -> PlacementEnv<'_> {
        let mut env = PlacementEnv::new(circuit).expect("placement env builds");
        let rc = RouteConfig {
            prefer_horizontal_first: true,
            no_crossings: true,
            max_z: 3,
        };
        if route_placed_netlist(&circuit.netlist, &env.placed(), &rc).is_err() {
            let cfg = AnnealConfig {
                route_validated_best: true,
                iterations: 1200,
                ..AnnealConfig::default()
            };
            anneal(&mut env, &cfg);
        }
        env
    }

    #[test]
    fn lif_step_aig_matches_reference_4bit() {
        // Exhaustive over a small membrane: 16 x 16 = 256 cases. Fast.
        let lif = QuantLif::new(4, 10);
        let aig = lif.step_aig();
        let w = lif.bits as usize;
        for v in 0..16u64 {
            for inp in 0..16u64 {
                let mut ins = bits_of(v, 4);
                ins.extend(bits_of(inp, 4));
                let out = evaluate_aig(&aig, &ins);
                // outputs: v_out[0..w] then spike
                let v_out = val_of(&out[0..w]);
                let spike = out[w];
                let (ref_v, ref_spike) = lif.step_reference(v, inp);
                assert_eq!(
                    (v_out, spike),
                    (ref_v, ref_spike),
                    "mismatch at v={v} inp={inp}"
                );
            }
        }
    }

    #[test]
    fn lif_step_reference_integrate_and_fire() {
        // Sanity on the reference itself: accumulate then fire+reset.
        let lif = QuantLif::new(4, 10);
        let (v1, s1) = lif.step_reference(0, 6);
        assert_eq!((v1, s1), (6, false), "6 < 10: integrate, no spike");
        let (v2, s2) = lif.step_reference(v1, 6);
        assert_eq!((v2, s2), (0, true), "12 >= 10: spike and reset");
        // Saturation: 15 + 15 caps at 15 (>= 10 so it fires and resets).
        let (v3, s3) = lif.step_reference(15, 15);
        assert_eq!((v3, s3), (0, true));
    }

    /// The physical-authority oracle: the LIF step, synthesized to REAL TILES,
    /// computes correctly. `verify_physical` places + routes + exports the AIG to
    /// the tile simulator and checks the truth table tile == AIG; combined with
    /// `lif_step_aig_matches_reference_4bit` (AIG == reference), this gives
    /// tile == reference — the LIF membrane step computing in tiles.
    #[test]
    fn lif_step_computes_in_tiles() {
        let lif = QuantLif::new(4, 10);
        let circuit = Circuit::from_aig("lif_step", lif.step_aig());
        assert!(
            circuit.mapping_is_correct(),
            "mapped netlist must equal the AIG"
        );
        let env = routable_env(&circuit);
        assert!(
            env.verify_physical(),
            "LIF integrate-and-fire step must compute correctly on real tiles"
        );
    }

    #[test]
    fn leaky_lif_reference_decays() {
        // leak_shift=1 => leaked = v - v/2. 10 -> leaked 5 (no input, no spike).
        let lif = QuantLif::new(4, 10).with_leak_shift(1);
        assert_eq!(lif.step_reference(10, 0), (5, false), "10 decays to 5");
        // 8 leaks to 4, +6 = 10 >= 10 => spike + reset.
        assert_eq!(lif.step_reference(8, 6), (0, true));
    }

    #[test]
    fn leaky_lif_aig_matches_reference_4bit() {
        // Exhaustive 4-bit with leak (factor ~0.5). The AIG's shift+subtract leak
        // must match the reference for every (v, input).
        let lif = QuantLif::new(4, 10).with_leak_shift(1);
        let aig = lif.step_aig();
        let w = lif.bits as usize;
        for v in 0..16u64 {
            for inp in 0..16u64 {
                let mut ins = bits_of(v, 4);
                ins.extend(bits_of(inp, 4));
                let out = evaluate_aig(&aig, &ins);
                let got = (val_of(&out[0..w]), out[w]);
                assert_eq!(
                    got,
                    lif.step_reference(v, inp),
                    "leak mismatch v={v} inp={inp}"
                );
            }
        }
    }

    #[test]
    fn leaky_lif_computes_in_tiles() {
        // The leaky membrane step computes on real tiles (tile == AIG == reference).
        let lif = QuantLif::new(4, 10).with_leak_shift(1);
        let circuit = Circuit::from_aig("leaky_lif_step", lif.step_aig());
        assert!(circuit.mapping_is_correct());
        let env = routable_env(&circuit);
        assert!(
            env.verify_physical(),
            "leaky LIF step must compute correctly on real tiles"
        );
    }

    #[test]
    fn synaptic_current_aig_matches_reference() {
        // 3 synapses whose weights can sum past out_bits, exercising saturation.
        let syn = Synapses::new(vec![6, 7, 5], 4); // sum up to 18 > 15 -> saturates
        let aig = syn.current_aig();
        let ob = syn.out_bits as usize;
        for pat in 0..(1u64 << 3) {
            let spikes: Vec<bool> = (0..3).map(|i| (pat >> i) & 1 == 1).collect();
            let out = evaluate_aig(&aig, &spikes);
            let got = val_of(&out[0..ob]);
            assert_eq!(got, syn.current_reference(&spikes), "pat={pat:03b}");
        }
    }

    #[test]
    fn lif_multitick_spike_train_on_tiles() {
        // Drive the membrane over a sequence by evaluating the EXPORTED tile
        // circuit each tick and feeding v_mem back: multi-tick LIF dynamics
        // computed on real tiles (per-tick compute is physical; the membrane
        // state is threaded in software until a tile register lands). Proves the
        // neuron's behavior over TIME on the fabric, not just one step.
        use crate::synth::export::{evaluate_exported, export_to_simulation};

        let lif = QuantLif::new(4, 10).with_leak_shift(1);
        let w = lif.bits as usize;
        let circuit = Circuit::from_aig("lif_step", lif.step_aig());
        // Find a layout that routes legally under the corrected router (baseline,
        // else route-validated SA) — same fallback the production neuron uses.
        let env = routable_env(&circuit);
        let placed = env.placed();
        let rc = RouteConfig {
            prefer_horizontal_first: true,
            no_crossings: true,
            max_z: 3,
        };
        let routed = route_placed_netlist(&circuit.netlist, &placed, &rc).expect("routes");
        let mut export = export_to_simulation(&routed, &circuit.netlist);

        let inputs = [5u64, 5, 5, 5, 5, 5, 5, 5];
        let reference = lif.run_reference(0, &inputs);
        let mut v = 0u64;
        for (t, &inp) in inputs.iter().enumerate() {
            let mut ins = bits_of(v, 4);
            ins.extend(bits_of(inp, 4));
            let out = evaluate_exported(&mut export, &ins);
            assert!(export.last_converged, "tick {t} did not converge on tiles");
            let v_next = val_of(&out[0..w]);
            let spike = out[w];
            assert_eq!(
                (v_next, spike),
                reference[t],
                "tile dynamics diverge from reference at tick {t}"
            );
            v = v_next;
        }
        // The leaky membrane builds sub-threshold then fires periodically.
        assert!(
            reference.iter().any(|&(_, s)| s),
            "expected at least one spike"
        );
    }

    #[test]
    fn synaptic_current_computes_in_tiles() {
        use crate::synth::alphafabric::{Circuit, PlacementEnv};
        let syn = Synapses::new(vec![6, 7, 5], 4);
        let circuit = Circuit::from_aig("synapses", syn.current_aig());
        assert!(circuit.mapping_is_correct());
        let env = PlacementEnv::new(&circuit).expect("placement env builds");
        assert!(
            env.verify_physical(),
            "synaptic weighted sum must compute correctly on real tiles"
        );
    }

    #[test]
    fn tile_lif_neuron_ticks_on_tiles() {
        // A complete neuron — synapse front-end + leaky membrane — ticking on
        // tiles via SequentialCircuit (next-state synthesized + evaluated on the
        // tile simulator each tick, membrane threaded as state). The spike train
        // must match the software neuron reference over the sequence.
        let lif = QuantLif::new(4, 10).with_leak_shift(1);
        let synapses = Synapses::new(vec![3, 4], 4); // 2 synapses, current fits in 4 bits
        let mut neuron = TileLifNeuron::new(lif, synapses);

        let seq: Vec<Vec<bool>> = vec![
            vec![true, true],  // current 7
            vec![true, false], // current 3
            vec![true, true],
            vec![false, true],
            vec![true, true],
            vec![true, true],
        ];
        let reference = neuron.run_reference(0, &seq);
        for (t, spikes) in seq.iter().enumerate() {
            let spike = neuron.tick(spikes);
            assert_eq!(
                (neuron.membrane(), spike),
                reference[t],
                "neuron tile dynamics diverge at tick {t}"
            );
        }
        assert!(reference.iter().any(|&(_, s)| s), "neuron should fire");
    }

    #[test]
    fn tile_lif_layer_ticks_on_tiles() {
        // A small spiking layer: independent neurons reading one shared input
        // vector, each its own tile circuit, all ticking together. The per-neuron
        // output spikes must match independent software neurons over the sequence.
        // Uses baseline-routable weights so the layer build is fast (no SA);
        // neuron_routes_via_sa_placement separately proves the SA fallback.
        let lif = QuantLif::new(4, 10).with_leak_shift(1);
        let synapses = vec![
            Synapses::new(vec![3, 4], 4),
            Synapses::new(vec![3, 4], 4),
            Synapses::new(vec![3, 4], 4),
        ];
        let mut layer = TileLifLayer::new(lif, synapses);
        assert_eq!(layer.len(), 3);

        let seq: Vec<Vec<bool>> = vec![
            vec![true, true],
            vec![true, false],
            vec![true, true],
            vec![false, true],
            vec![true, true],
            vec![true, true],
        ];
        let reference = layer.run_reference(&seq);
        for (t, inputs) in seq.iter().enumerate() {
            let spikes = layer.tick(inputs);
            assert_eq!(
                spikes, reference[t],
                "layer tile dynamics diverge at tick {t}"
            );
        }
        // The neurons fire (the layer is alive), and not all identically.
        assert!(
            reference.iter().flatten().any(|&s| s),
            "layer should produce spikes"
        );
    }

    #[test]
    fn neuron_routes_via_sa_placement() {
        // The convergence proof: weights [5,2] make a neuron circuit that is
        // UNROUTABLE under baseline row-major placement, so TileLifNeuron falls
        // back to AlphaFabric's route-validated SA placer to find a routable
        // layout. That this builds AND ticks correctly proves the optimized
        // placer hardens SNN routability (NeuroAlphaFabric, Stage 2). One SA run,
        // so this is the slowest tile_lif test — still well under a minute.
        let lif = QuantLif::new(4, 10).with_leak_shift(1);
        let mut neuron = TileLifNeuron::new(lif, Synapses::new(vec![5, 2], 4));
        let seq: Vec<Vec<bool>> = vec![vec![true, true]; 4];
        let reference = neuron.run_reference(0, &seq);
        for (t, spikes) in seq.iter().enumerate() {
            let spike = neuron.tick(spikes);
            assert_eq!(
                (neuron.membrane(), spike),
                reference[t],
                "SA-placed neuron diverges at tick {t}"
            );
        }
    }

    #[test]
    fn tile_spiking_classifier_separates_two_classes() {
        // A tiny end-to-end spiking classifier ON TILES — the MNIST-in-tiles
        // pipeline on a synthetic 2-class task (MNIST data files aren't present
        // in-test; the pipeline is identical: features → rate-coded input spikes
        // → tile spiking layer → argmax over output spike counts). leak_shift=2
        // gives steady-state membrane ≈ 4·current, so each output neuron fires
        // only when its own class input is active.
        let lif = QuantLif::new(4, 10).with_leak_shift(2);
        // Class-0 neuron reads input 0; class-1 neuron reads input 1.
        let mut layer = TileLifLayer::new(
            lif,
            vec![Synapses::new(vec![4, 0], 4), Synapses::new(vec![0, 4], 4)],
        );
        let ticks = 9;
        let pat_a = vec![true, false]; // class 0
        let pat_b = vec![false, true]; // class 1

        let (counts_a, class_a) = layer.classify(&pat_a, ticks);
        let (counts_b, class_b) = layer.classify(&pat_b, ticks);
        assert_eq!(class_a, 0, "pattern A → class 0 (counts {counts_a:?})");
        assert_eq!(class_b, 1, "pattern B → class 1 (counts {counts_b:?})");
        // Real separation: each class neuron fired, the other stayed silent.
        assert!(
            counts_a[0] > 0 && counts_a[1] == 0,
            "class-0 selective: {counts_a:?}"
        );
        assert!(
            counts_b[1] > 0 && counts_b[0] == 0,
            "class-1 selective: {counts_b:?}"
        );

        // Oracle: the tile layer's spike counts match a software-reference layer.
        let seq_a: Vec<Vec<bool>> = vec![pat_a.clone(); ticks];
        let ref_a = layer.run_reference(&seq_a);
        let ref_counts_a: Vec<u32> = (0..2)
            .map(|j| ref_a.iter().filter(|t| t[j]).count() as u32)
            .collect();
        assert_eq!(
            counts_a, ref_counts_a,
            "tile counts must match software reference"
        );
    }

    #[test]
    fn refractory_reference_suppresses_after_spike() {
        // No leak, threshold 10, refractory R=3. Drive constant input 6.
        // tick0: 0+6=6  (<10, no spike)              refr 0
        // tick1: 6+6=12 (>=10) FIRE, reset, refr<-3  refr 3
        // tick2: 0+6=6  (<10) — and refr=3 anyway    refr 2
        // tick3: 6+6=12 (>=10) but refr=2 SUPPRESSED  refr 1
        // tick4: 12 sat   >=10 but refr=1 SUPPRESSED  refr 0
        // tick5: 12 (sat) >=10, refr=0 FIRE again
        let rlif = RefractoryLif::new(QuantLif::new(4, 10), 3);
        let trace = rlif.run_reference(0, 0, &[6, 6, 6, 6, 6, 6]);
        let spikes: Vec<bool> = trace.iter().map(|&(_, _, s)| s).collect();
        assert_eq!(
            spikes,
            vec![false, true, false, false, false, true],
            "refractory must suppress firing for R=3 ticks after a spike: {trace:?}"
        );
        // Without refractory the same drive fires far more often (every other tick).
        let plain = RefractoryLif::new(QuantLif::new(4, 10), 0);
        let plain_spikes = plain.run_reference(0, 0, &[6, 6, 6, 6, 6, 6]);
        let n_plain = plain_spikes.iter().filter(|&&(_, _, s)| s).count();
        assert!(n_plain > 2, "no-refractory neuron fires more: {n_plain}");
    }

    #[test]
    fn refractory_aig_matches_reference_exhaustive() {
        // Exhaustive over the full combined state + input: v(16) x refr(0..=R) x
        // input(16). R=3 -> refr_bits=2 (4 values) -> 16*4*16 = 1024 evals. Fast.
        let rlif = RefractoryLif::new(QuantLif::new(4, 10).with_leak_shift(1), 3);
        let aig = rlif.step_aig();
        let w = rlif.lif.bits as usize;
        let rb = rlif.refr_bits() as usize;
        for v in 0..16u64 {
            for refr in 0..(1u64 << rb) {
                for inp in 0..16u64 {
                    let mut ins = bits_of(v, 4);
                    ins.extend(bits_of(refr, rb as u32));
                    ins.extend(bits_of(inp, 4));
                    let out = evaluate_aig(&aig, &ins);
                    let v_out = val_of(&out[0..w]);
                    let refr_out = val_of(&out[w..w + rb]);
                    let spike = out[w + rb];
                    assert_eq!(
                        (v_out, refr_out, spike),
                        rlif.step_reference(v, refr, inp),
                        "refractory AIG mismatch at v={v} refr={refr} inp={inp}"
                    );
                }
            }
        }
    }

    #[test]
    fn refractory_multitick_spike_train_on_tiles() {
        // The refractory membrane ticking on REAL TILES: export the combined-state
        // step circuit and thread BOTH the membrane and the refractory countdown
        // between ticks, comparing the spike train to the software reference. The
        // suppression gap after each fire is computed in tiles == reference.
        use crate::synth::alphafabric::{Circuit, PlacementEnv};
        use crate::synth::export::{evaluate_exported, export_to_simulation};
        use crate::synth::routing::{RouteConfig, route_placed_netlist};

        let rlif = RefractoryLif::new(QuantLif::new(4, 10), 3);
        let w = rlif.lif.bits as usize;
        let rb = rlif.refr_bits() as usize;
        let circuit = Circuit::from_aig("refractory_lif", rlif.step_aig());
        assert!(
            circuit.mapping_is_correct(),
            "mapped netlist must equal the AIG"
        );
        let env = PlacementEnv::new(&circuit).expect("env builds");
        let rc = RouteConfig {
            prefer_horizontal_first: true,
            no_crossings: true,
            max_z: 3,
        };
        let routed = route_placed_netlist(&circuit.netlist, &env.placed(), &rc).expect("routes");
        let mut export = export_to_simulation(&routed, &circuit.netlist);

        let inputs = [6u64, 6, 6, 6, 6, 6, 6, 6];
        let reference = rlif.run_reference(0, 0, &inputs);
        let (mut v, mut r) = (0u64, 0u64);
        for (t, &inp) in inputs.iter().enumerate() {
            let mut ins = bits_of(v, 4);
            ins.extend(bits_of(r, rb as u32));
            ins.extend(bits_of(inp, 4));
            let out = evaluate_exported(&mut export, &ins);
            assert!(export.last_converged, "tick {t} did not converge on tiles");
            let v_next = val_of(&out[0..w]);
            let r_next = val_of(&out[w..w + rb]);
            let spike = out[w + rb];
            assert_eq!(
                (v_next, r_next, spike),
                reference[t],
                "refractory tile dynamics diverge from reference at tick {t}"
            );
            v = v_next;
            r = r_next;
        }
        // The train both fires and is silenced — a refractory gap really happens.
        let spikes: Vec<bool> = reference.iter().map(|&(_, _, s)| s).collect();
        assert!(spikes.iter().any(|&s| s), "expected the neuron to fire");
        assert!(
            spikes.iter().any(|&s| !s),
            "expected refractory suppression"
        );
    }

    #[test]
    fn q8_leak_reference_is_fixed_point_factor() {
        // q/256 retention: 0.75 (q=192), 0.5 (q=128), ~0.996 (q=255), 1.0 (q=256).
        let f = |q: u16, v: u64| QuantLif::new(8, 200).with_leak_q8(q).leak(v);
        assert_eq!(f(192, 8), 6, "0.75 * 8 = 6"); // (8*192)>>8
        assert_eq!(f(192, 12), 9, "0.75 * 12 = 9"); // (12*192)>>8
        assert_eq!(f(128, 10), 5, "0.5 * 10 = 5");
        assert_eq!(f(256, 9), 9, "factor 1.0 is identity");
        assert_eq!(f(0, 15), 0, "factor 0 fully leaks");
        // Arbitrary factor, not reachable by any leak_shift (1 - 2^-k):
        assert_eq!(f(180, 10), 7, "0.703 * 10 = 7"); // (10*180)>>8 = 1800>>8
    }

    #[test]
    fn q8_leak_aig_matches_reference_exhaustive() {
        // The constant shift-and-add multiply leak must match the reference for
        // every (v, input). Two arbitrary factors that no power-of-two leak gives.
        for q in [192u16, 180] {
            let lif = QuantLif::new(4, 10).with_leak_q8(q);
            let aig = lif.step_aig();
            let w = lif.bits as usize;
            for v in 0..16u64 {
                for inp in 0..16u64 {
                    let mut ins = bits_of(v, 4);
                    ins.extend(bits_of(inp, 4));
                    let out = evaluate_aig(&aig, &ins);
                    let got = (val_of(&out[0..w]), out[w]);
                    assert_eq!(
                        got,
                        lif.step_reference(v, inp),
                        "Q0.8 leak mismatch q={q} v={v} inp={inp}"
                    );
                }
            }
        }
    }

    #[test]
    fn q8_leak_computes_in_tiles() {
        // The full fixed-point leak membrane step computes on real tiles
        // (tile == AIG == reference) — arbitrary leak factor, not power-of-2.
        use crate::synth::alphafabric::{Circuit, PlacementEnv};
        let lif = QuantLif::new(4, 10).with_leak_q8(192);
        let circuit = Circuit::from_aig("q8_leak_step", lif.step_aig());
        assert!(circuit.mapping_is_correct());
        let env = PlacementEnv::new(&circuit).expect("placement env builds");
        assert!(
            env.verify_physical(),
            "Q0.8 leak step must compute correctly on real tiles"
        );
    }

    #[test]
    fn refractory_neuron_aig_matches_reference_exhaustive() {
        // The UNIFIED next-state circuit (synapses + leak + refractory) matches
        // the composed reference over the full small state x input space:
        // v(16) x refr(0..=R) x spike-patterns(2^k). k=2, R=2 -> 16*4*4 = 256.
        let rlif = RefractoryLif::new(QuantLif::new(4, 10).with_leak_shift(1), 2);
        let synapses = Synapses::new(vec![3, 4], 4);
        let aig = rlif.neuron_step_aig(&synapses);
        let w = rlif.lif.bits as usize;
        let rb = rlif.refr_bits() as usize;
        let k = synapses.weights.len();
        for v in 0..16u64 {
            for refr in 0..(1u64 << rb) {
                for pat in 0..(1u64 << k) {
                    let spikes: Vec<bool> = (0..k).map(|i| (pat >> i) & 1 == 1).collect();
                    let mut ins = bits_of(v, 4);
                    ins.extend(bits_of(refr, rb as u32));
                    ins.extend(spikes.iter().copied());
                    let out = evaluate_aig(&aig, &ins);
                    let got = (val_of(&out[0..w]), val_of(&out[w..w + rb]), out[w + rb]);
                    assert_eq!(
                        got,
                        rlif.neuron_step_reference(v, refr, &synapses, &spikes),
                        "unified neuron AIG mismatch v={v} refr={refr} pat={pat:02b}"
                    );
                }
            }
        }
    }

    #[test]
    fn tile_refractory_neuron_ticks_on_tiles() {
        // The FULL LIF neuron — weighted synapses + leak + absolute refractory —
        // ticking on tiles, threading BOTH the membrane and the refractory
        // countdown as state. The spike train (fires AND refractory gaps even
        // while the membrane is over threshold) must match the software neuron
        // reference over the sequence, with real synaptic input.
        // One synapse (weight 6, always spiking ⇒ current 6) keeps the circuit
        // baseline-routable so the build needs no SA — fast. The 2-synapse path
        // is covered exhaustively by refractory_neuron_aig_matches_reference.
        let rlif = RefractoryLif::new(QuantLif::new(4, 10).with_leak_shift(1), 3);
        let synapses = Synapses::new(vec![6], 4);
        let mut neuron = TileRefractoryNeuron::new(rlif, synapses);

        let seq: Vec<Vec<bool>> = vec![vec![true]; 8]; // current 6 each tick
        let reference = neuron.run_reference(0, 0, &seq);
        for (t, spikes) in seq.iter().enumerate() {
            let spike = neuron.tick(spikes);
            assert_eq!(
                (neuron.membrane(), neuron.refractory(), spike),
                reference[t],
                "unified neuron tile dynamics diverge at tick {t}"
            );
        }
        let fired: Vec<bool> = reference.iter().map(|&(_, _, s)| s).collect();
        assert!(fired.iter().any(|&s| s), "the neuron should fire");
        assert!(
            fired.iter().any(|&s| !s),
            "refractory should suppress some over-threshold ticks"
        );
    }
}
