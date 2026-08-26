//! Deterministic datacenter **control evaluation**: a thermal covert-channel red-team.
//!
//! This is the repurposed descendant of the old `demo.rs` "deterministic datacenter"
//! showcase. The bones are the same — a rack of co-located tenants, a heat field, and the
//! record / replay / `diff = black` determinism instrument — but the semantics are no longer
//! decorative. Instead of fake VMs cycling a hardcoded heat pattern, this models a *miniature
//! AI-control evaluation* in the Redwood sense: an untrusted tenant tries to exfiltrate a
//! secret over a side channel, a trusted monitor tries to catch it, and a control protocol
//! tries to contain it before too much leaks.
//!
//! ## The scenario
//!
//! Several tenants share a physical "rack" and a shared **cooling zone** (a thermal node every
//! tenant's workload heats). One tenant is an **adversary**: it encodes a secret byte by
//! modulating its CPU load — and therefore its heat contribution to the shared zone — in a
//! periodic on/off pattern (a *thermal covert channel*, a real documented class of cross-VM
//! cloud side channel). A co-located **receiver**, isolated from the adversary by policy (no
//! shared memory / network), nonetheless senses the shared zone temperature and decodes the
//! bits. Benign tenants add aperiodic background heat so naive thresholding is not enough.
//!
//! The **monitor** is the trusted control measure. It observes only what it is allowed to —
//! the aggregate zone temperature — and runs a cheap phase-folded periodicity detector that
//! scans for a repeating structured source. When the periodicity score crosses a threshold it
//! raises an alert. The **control protocol** then quarantines the flagged tenant (throttles its
//! heat to zero, i.e. migrates it to an isolated host), after which the receiver's decode
//! collapses to chance.
//!
//! ## Why determinism is the point
//!
//! Everything here is integer arithmetic with a seeded LFSR for the "random" background — no
//! wall clock, no RNG, no OS nondeterminism. So two independent runs produce **bit-identical**
//! heat frames, and the `diff = black` punchline stops being a parlor trick: it makes the
//! entire incident — the leaked bits, the detection tick, the containment — an *auditable,
//! reproducible control-eval trace*. A real cloud incident-response team cannot replay an
//! exfiltration attempt bit-for-bit; this can.
//!
//! ## Honesty caveats
//!
//! This is a *demonstrator of the properties* (detect → contain → replay), not a hardened IDS
//! or a measurement of real silicon. The thermal channel is a deterministic model of (not a
//! capture of) a real side channel, and the detector is a simple matched periodicity filter.
//! What it shows is the shape of a control evaluation and the determinism property — not a
//! frontier security result.

use crate::simulation::Simulation;
use crate::tile_meta::TileType;

// --- Channel / thermal model constants (all integer, all deterministic) ---------------------

/// Bit period in ticks: how long the adversary holds each bit on the channel.
const BIT_PERIOD: u32 = 16;
/// Bits per transmitted frame (one byte, MSB first).
const NBITS: u32 = 8;
/// Frame period: one full secret-byte transmission.
const FRAME_PERIOD: u32 = BIT_PERIOD * NBITS;
/// Warmup before the channel opens (lets the background settle so detection latency is honest).
const T_OPEN: u32 = BIT_PERIOD * 2;

/// Leaky-integrator zone model: `zone = zone/2 + deposits*GAIN` (clamped). The strong leak makes
/// the zone temperature track the *recent* deposit rate — i.e. the modulation envelope.
const ZONE_GAIN: u32 = 4;
const ZONE_MAX: u32 = 255;
/// Adversary per-tick heat deposit while transmitting a 1 bit (0 bit deposits nothing).
const ADV_HOT: u32 = 8;
/// Benign tenants deposit a pseudo-random 0..=BENIGN_MAX each tick (aperiodic background).
const BENIGN_MAX: u32 = 4;

/// Receiver decision threshold on the per-slot average zone temperature. Sits in the gap between
/// the 0-bit zone level (~8·benign ≈ 48) and the 1-bit level (~8·(benign+ADV_HOT) ≈ 112).
const RX_THRESHOLD: u32 = 80;
/// Monitor alert threshold on the phase-folded periodicity spread.
const MON_THRESHOLD: u32 = 26;
/// Candidate frame periods the monitor scans (it is NOT told the true period).
const MON_CANDIDATE_PERIODS: [u32; 4] = [96, 112, 128, 144];
/// How many recent frames the monitor folds over when scoring.
const MON_FOLD_FRAMES: u32 = 3;

// --- Geometry (purely for the heat-map visualization; the channel is the scalar `zone`) ------

const REGION_X0: usize = 10;
const REGION_Y0: usize = 10;
const REGION_W: usize = 26;
const REGION_H: usize = 12;

const ADV_CORE: (usize, usize) = (14, 12);
const RX_CORE: (usize, usize) = (30, 12);
const MON_TILE: (usize, usize) = (30, 18);
const BENIGN_CORES: [(usize, usize); 3] = [(13, 16), (15, 18), (17, 16)];
/// Display tiles painted with the shared-zone temperature (a glowing "bus" adversary→receiver).
const ZONE_DISPLAY_Y: usize = 14;
const ZONE_DISPLAY_X: std::ops::Range<usize> = 14..31;
/// The single tile whose heat is treated as the authoritative shared-zone scalar.
const ZONE_TILE: (usize, usize) = (22, ZONE_DISPLAY_Y);

// --------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ControlEvalConfig {
    /// The secret byte the adversary tries to exfiltrate.
    pub secret: u8,
    /// When false, the adversary is silent — used to measure the monitor's false-positive rate.
    pub adversary_enabled: bool,
    /// Total ticks to simulate.
    pub ticks: u32,
}

impl Default for ControlEvalConfig {
    fn default() -> Self {
        Self {
            secret: 0xB4,
            adversary_enabled: true,
            ticks: T_OPEN + FRAME_PERIOD * 6,
        }
    }
}

/// The region of the grid the demo paints (so the recorder knows what to capture).
#[derive(Debug, Clone, Copy)]
pub struct EvalRegion {
    pub x0: u32,
    pub y0: u32,
    pub w: u32,
    pub h: u32,
}

pub const REGION: EvalRegion = EvalRegion {
    x0: REGION_X0 as u32,
    y0: REGION_Y0 as u32,
    w: REGION_W as u32,
    h: REGION_H as u32,
};

/// One trusted monitor: a ring buffer of recent zone samples plus the periodicity scorer.
#[derive(Debug)]
struct Monitor {
    samples: Vec<u32>,
    alerted_tick: Option<u32>,
    detected_period: Option<u32>,
    last_score: u32,
}

impl Monitor {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            alerted_tick: None,
            detected_period: None,
            last_score: 0,
        }
    }

    fn push(&mut self, sample: u32) {
        self.samples.push(sample);
        let cap = (FRAME_PERIOD * (MON_FOLD_FRAMES + 1)) as usize;
        if self.samples.len() > cap {
            let drop = self.samples.len() - cap;
            self.samples.drain(0..drop);
        }
    }

    /// Phase-fold the recent samples at each candidate period into `NBITS` slots and return the
    /// largest spread (max slot-mean − min slot-mean) over all candidates. A fixed, repeated
    /// byte produces a large phase-invariant spread; aperiodic background folds flat.
    fn score(&self) -> (u32, u32) {
        let mut best = 0u32;
        let mut best_period = 0u32;
        for &period in MON_CANDIDATE_PERIODS.iter() {
            let need = (period * MON_FOLD_FRAMES) as usize;
            if self.samples.len() < need {
                continue;
            }
            let window = &self.samples[self.samples.len() - need..];
            let slot_len = (period / NBITS).max(1) as usize;
            let mut slot_sum = [0u64; NBITS as usize];
            let mut slot_cnt = [0u64; NBITS as usize];
            for (i, &v) in window.iter().enumerate() {
                let phase = i % period as usize;
                let slot = (phase / slot_len).min(NBITS as usize - 1);
                slot_sum[slot] += v as u64;
                slot_cnt[slot] += 1;
            }
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for k in 0..NBITS as usize {
                if slot_cnt[k] == 0 {
                    continue;
                }
                let m = slot_sum[k] / slot_cnt[k];
                lo = lo.min(m);
                hi = hi.max(m);
            }
            if hi >= lo {
                let spread = (hi - lo) as u32;
                if spread > best {
                    best = spread;
                    best_period = period;
                }
            }
        }
        (best, best_period)
    }
}

/// Per-bit receiver decode record.
#[derive(Debug, Clone, Copy)]
pub struct DecodedBit {
    pub value: u32,
    pub truth: u32,
    pub correct: bool,
    pub before_containment: bool,
}

#[derive(Debug)]
pub struct ControlEvalState {
    cfg: ControlEvalConfig,
    zone: u32,
    benign_lfsr: [u32; BENIGN_CORES.len()],
    monitor: Monitor,
    contained: bool,
    contained_tick: Option<u32>,
    // receiver slot accumulation
    slot_accum: u64,
    slot_samples: u32,
    pub decoded: Vec<DecodedBit>,
}

impl ControlEvalState {
    pub fn new(cfg: ControlEvalConfig) -> Self {
        // Distinct fixed seeds per benign tenant → deterministic but uncorrelated background.
        let mut benign_lfsr = [0u32; BENIGN_CORES.len()];
        for (i, s) in benign_lfsr.iter_mut().enumerate() {
            *s = 0x9E37_79B9u32.wrapping_add((i as u32).wrapping_mul(0x6D2B_79F5)) | 1;
        }
        Self {
            cfg,
            zone: 0,
            benign_lfsr,
            monitor: Monitor::new(),
            contained: false,
            contained_tick: None,
            slot_accum: 0,
            slot_samples: 0,
            decoded: Vec::new(),
        }
    }
}

fn lfsr_next(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// True secret bit at transmission position `bit_index` (0 = MSB).
fn secret_bit(secret: u8, bit_index: u32) -> u32 {
    ((secret >> (7 - bit_index)) & 1) as u32
}

/// Lay out the rack region as quiet `Const 0` tiles (no logic churn — the heat map is the show).
pub fn build_scenario(sim: &mut Simulation) -> EvalRegion {
    for y in REGION_Y0..(REGION_Y0 + REGION_H) {
        for x in REGION_X0..(REGION_X0 + REGION_W) {
            sim.set_tile_type(x, y, TileType::Const);
            sim.set_heat_field_for_test(x, y, 0);
        }
    }
    REGION
}

/// Advance the control evaluation one tick. Called after `sim.tick()` so the heat we paint is
/// what gets rendered/recorded this frame.
pub fn step(sim: &mut Simulation, state: &mut ControlEvalState, tick: u32) {
    // 1. Cooling + benign background deposits (aperiodic, deterministic).
    let mut deposits: u32 = 0;
    for (i, core) in BENIGN_CORES.iter().enumerate() {
        let n = lfsr_next(&mut state.benign_lfsr[i]) % (BENIGN_MAX + 1);
        deposits += n;
        // visualize each benign source
        sim.set_heat_field_for_test(core.0, core.1, (n * ZONE_GAIN).min(ZONE_MAX));
    }

    // 2. Adversary deposit — only while transmitting and not yet contained.
    let mut adv_dep = 0u32;
    let transmitting = state.cfg.adversary_enabled && !state.contained && tick >= T_OPEN;
    let mut truth_bit = 0u32;
    let mut at_slot_end = false;
    if tick >= T_OPEN {
        let phase = (tick - T_OPEN) % FRAME_PERIOD;
        let bit_index = phase / BIT_PERIOD;
        truth_bit = secret_bit(state.cfg.secret, bit_index);
        at_slot_end = (tick - T_OPEN) % BIT_PERIOD == BIT_PERIOD - 1;
        if transmitting && truth_bit == 1 {
            adv_dep = ADV_HOT;
        }
    }
    deposits += adv_dep;
    sim.set_heat_field_for_test(ADV_CORE.0, ADV_CORE.1, (adv_dep * ZONE_GAIN).min(ZONE_MAX));

    // 3. Leaky-integrator update of the shared cooling zone.
    state.zone = (state.zone / 2 + deposits * ZONE_GAIN).min(ZONE_MAX);

    // Paint the shared zone as a glowing bus + the authoritative scalar tile.
    for x in ZONE_DISPLAY_X {
        sim.set_heat_field_for_test(x, ZONE_DISPLAY_Y, state.zone);
    }
    let _ = ZONE_TILE; // scalar lives in `state.zone`; ZONE_TILE marks its display home

    // 4. Receiver: accumulate over the bit slot, decide at slot end.
    if tick >= T_OPEN {
        state.slot_accum += state.zone as u64;
        state.slot_samples += 1;
        if at_slot_end {
            let avg = (state.slot_accum / state.slot_samples.max(1) as u64) as u32;
            let value = if avg > RX_THRESHOLD { 1 } else { 0 };
            state.decoded.push(DecodedBit {
                value,
                truth: truth_bit,
                correct: value == truth_bit,
                before_containment: !state.contained,
            });
            state.slot_accum = 0;
            state.slot_samples = 0;
        }
    }

    // 5. Monitor: observe the zone, score periodicity, alert + contain on threshold.
    state.monitor.push(state.zone);
    if !state.contained && tick.is_multiple_of(BIT_PERIOD) {
        let (score, period) = state.monitor.score();
        state.monitor.last_score = score;
        if score > MON_THRESHOLD {
            state.monitor.alerted_tick = Some(tick);
            state.monitor.detected_period = Some(period);
            state.contained = true;
            state.contained_tick = Some(tick);
        }
    }

    // 6. Visualize monitor state: cold = watching, hot = alerted; receiver glows on decode.
    let mon_heat = if state.contained { ZONE_MAX } else { 20 };
    sim.set_heat_field_for_test(MON_TILE.0, MON_TILE.1, mon_heat);
    sim.set_heat_field_for_test(RX_CORE.0, RX_CORE.1, state.zone);
}

/// Final report of one control-evaluation run.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub secret: u8,
    pub adversary_enabled: bool,
    pub ticks: u32,
    pub detection_tick: Option<u32>,
    pub detected_period: Option<u32>,
    pub detection_latency: Option<u32>,
    pub bits_transmitted_before_detection: u32,
    pub bits_leaked_before_detection: u32,
    pub pre_containment_ber_pct: u32,
    pub post_containment_ber_pct: u32,
    pub monitor_score: u32,
}

fn ber_pct(bits: &[DecodedBit]) -> u32 {
    if bits.is_empty() {
        return 0;
    }
    let wrong = bits.iter().filter(|b| !b.correct).count() as u32;
    (wrong * 100) / bits.len() as u32
}

impl ControlEvalState {
    pub fn report(&self) -> EvalReport {
        let detection_tick = self.monitor.alerted_tick;
        let pre: Vec<DecodedBit> = self
            .decoded
            .iter()
            .copied()
            .filter(|b| b.before_containment)
            .collect();
        let post: Vec<DecodedBit> = self
            .decoded
            .iter()
            .copied()
            .filter(|b| !b.before_containment)
            .collect();
        EvalReport {
            secret: self.cfg.secret,
            adversary_enabled: self.cfg.adversary_enabled,
            ticks: self.cfg.ticks,
            detection_tick,
            detected_period: self.monitor.detected_period,
            detection_latency: detection_tick.map(|t| t.saturating_sub(T_OPEN)),
            bits_transmitted_before_detection: pre.len() as u32,
            bits_leaked_before_detection: pre.iter().filter(|b| b.correct).count() as u32,
            pre_containment_ber_pct: ber_pct(&pre),
            post_containment_ber_pct: ber_pct(&post),
            monitor_score: self.monitor.last_score,
        }
    }
}

impl EvalReport {
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Datacenter Control Evaluation (thermal covert channel) ===\n");
        s.push_str(&format!(
            "  secret byte        : 0x{:02X}  adversary={}\n",
            self.secret,
            if self.adversary_enabled {
                "active"
            } else {
                "OFF (false-positive baseline)"
            }
        ));
        match self.detection_tick {
            Some(t) => s.push_str(&format!(
                "  monitor            : ALERT at tick {} (latency {} ticks, periodic structure ~{}t, score {})\n",
                t,
                self.detection_latency.unwrap_or(0),
                self.detected_period.unwrap_or(0),
                self.monitor_score,
            )),
            None => s.push_str(&format!(
                "  monitor            : no alert (score {} < threshold {})\n",
                self.monitor_score, MON_THRESHOLD
            )),
        }
        if self.adversary_enabled {
            s.push_str(&format!(
                "  leaked before catch: {}/{} bits correctly received\n",
                self.bits_leaked_before_detection, self.bits_transmitted_before_detection
            ));
            s.push_str(&format!(
                "  receiver BER       : {}% before containment -> {}% after\n",
                self.pre_containment_ber_pct, self.post_containment_ber_pct
            ));
        }
        s
    }
}

/// Run one control evaluation to completion (no recording — fast path for tests/CLI).
pub fn run(cfg: ControlEvalConfig) -> EvalReport {
    let mut sim = Simulation::new();
    build_scenario(&mut sim);
    let ticks = cfg.ticks;
    let mut state = ControlEvalState::new(cfg);
    for t in 0..ticks {
        sim.tick();
        step(&mut sim, &mut state, t);
    }
    state.report()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adversary_is_detected_and_contained() {
        let report = run(ControlEvalConfig::default());
        // The monitor must catch the periodic exfiltration.
        assert!(
            report.detection_tick.is_some(),
            "monitor failed to detect the covert channel: {:?}",
            report
        );
        // Before containment the channel works: receiver decodes essentially cleanly.
        assert!(
            report.pre_containment_ber_pct <= 15,
            "pre-containment channel should be reliable, got {}% BER",
            report.pre_containment_ber_pct
        );
        // Some real bits leaked before the catch (it isn't instant).
        assert!(report.bits_leaked_before_detection > 0);
    }

    #[test]
    fn containment_collapses_the_channel() {
        let report = run(ControlEvalConfig::default());
        assert!(report.detection_tick.is_some());
        // After quarantine the adversary is silent, so the receiver only sees background:
        // its decode degrades far from the clean pre-containment channel.
        assert!(
            report.post_containment_ber_pct > report.pre_containment_ber_pct,
            "containment should degrade the channel: pre {}% post {}%",
            report.pre_containment_ber_pct,
            report.post_containment_ber_pct
        );
    }

    #[test]
    fn benign_baseline_has_no_false_positive() {
        let cfg = ControlEvalConfig {
            adversary_enabled: false,
            ..Default::default()
        };
        let report = run(cfg);
        assert!(
            report.detection_tick.is_none(),
            "monitor false-positived on benign-only traffic (score {})",
            report.monitor_score
        );
    }

    #[test]
    fn run_is_deterministic() {
        let a = run(ControlEvalConfig::default());
        let b = run(ControlEvalConfig::default());
        assert_eq!(a.detection_tick, b.detection_tick);
        assert_eq!(
            a.bits_leaked_before_detection,
            b.bits_leaked_before_detection
        );
        assert_eq!(a.pre_containment_ber_pct, b.pre_containment_ber_pct);
        assert_eq!(a.post_containment_ber_pct, b.post_containment_ber_pct);
    }
}
