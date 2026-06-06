//! STDP (Spike-Timing-Dependent Plasticity) Learning Rules
//!
//! Implements:
//! - Standard STDP: Weight updates based on pre/post spike timing
//! - R-STDP: Reward-modulated STDP for reinforcement learning
//! - Eligibility traces for delayed reward signals

use super::config::STDPConfig;
use super::synapse::Synapse;

/// Compute exponential decay for STDP window
///
/// Returns the decay factor for a given time difference and time constant.
/// Uses fixed-point approximation: exp(-dt/tau) ≈ (256 - dt*256/tau) / 256
#[inline]
fn exp_decay_fixed(dt: u8, tau: u8) -> i8 {
    if dt == 0 || tau == 0 {
        return 127; // Maximum (no decay)
    }

    // Approximate exp(-dt/tau) in Q0.7 format (0-127)
    // For dt < tau: result ≈ 127 * (1 - dt/tau)
    let decay = (dt as u16 * 127) / tau as u16;
    (127 - decay.min(127) as i8).max(0)
}

/// Apply standard STDP weight update
///
/// Updates the synapse weight based on the timing of pre and post synaptic spikes:
/// - Post after pre (positive dt) → LTP (Long-Term Potentiation)
/// - Pre after post (negative dt) → LTD (Long-Term Depression)
///
/// # Arguments
/// * `synapse` - Synapse to update
/// * `pre_spike_time` - Tick when presynaptic neuron fired
/// * `post_spike_time` - Tick when postsynaptic neuron fired
/// * `config` - STDP parameters
pub fn apply_stdp(
    synapse: &mut Synapse,
    pre_spike_time: u8,
    post_spike_time: u8,
    config: &STDPConfig,
) {
    // Compute time difference (handling wrap-around)
    let dt = post_spike_time.wrapping_sub(pre_spike_time);

    // Determine if this is LTP or LTD based on timing
    // Note: we use wrapping arithmetic, so check if dt is "small" (post after pre)
    // or "large" (pre after post, wrapping around)
    let dw = if dt < 128 {
        // Post after pre → LTP (potentiation)
        // dt is the actual positive difference
        let decay = exp_decay_fixed(dt, config.tau_plus);
        ((config.a_plus as i16 * decay as i16) >> 7) as i8
    } else {
        // Pre after post → LTD (depression)
        // dt wrapped, actual negative difference is (256 - dt)
        let actual_dt = 256u16.saturating_sub(dt as u16) as u8;
        let decay = exp_decay_fixed(actual_dt, config.tau_minus);
        -((config.a_minus as i16 * decay as i16) >> 7) as i8
    };

    // Apply weight update with bounds
    let new_weight =
        (synapse.weight as i16 + dw as i16).clamp(config.w_min as i16, config.w_max as i16) as i8;
    synapse.weight = new_weight;
}

/// Compute STDP eligibility trace (for R-STDP)
///
/// Returns the raw eligibility value without applying it to the weight.
/// This should be stored and modulated by a reward signal later.
///
/// # Returns
/// Eligibility value in Q1.7 format (can be positive or negative)
pub fn compute_eligibility(pre_spike_time: u8, post_spike_time: u8, config: &STDPConfig) -> i8 {
    let dt = post_spike_time.wrapping_sub(pre_spike_time);

    if dt < 128 {
        // Post after pre → positive eligibility (LTP candidate)
        let decay = exp_decay_fixed(dt, config.tau_plus);
        ((config.a_plus as i16 * decay as i16) >> 7) as i8
    } else {
        // Pre after post → negative eligibility (LTD candidate)
        let actual_dt = 256u16.saturating_sub(dt as u16) as u8;
        let decay = exp_decay_fixed(actual_dt, config.tau_minus);
        -((config.a_minus as i16 * decay as i16) >> 7) as i8
    }
}

/// Apply reward-modulated STDP (R-STDP)
///
/// The weight change is proportional to both the STDP eligibility and the reward signal.
/// This enables reinforcement learning with delayed rewards.
///
/// # Arguments
/// * `synapse` - Synapse to update
/// * `pre_spike_time` - Tick when presynaptic neuron fired
/// * `post_spike_time` - Tick when postsynaptic neuron fired
/// * `reward` - Global reward signal (positive = reward, negative = punishment)
/// * `config` - STDP parameters
pub fn apply_rstdp(
    synapse: &mut Synapse,
    pre_spike_time: u8,
    post_spike_time: u8,
    reward: i8,
    config: &STDPConfig,
) {
    // Compute eligibility trace
    let eligibility = compute_eligibility(pre_spike_time, post_spike_time, config);

    // Modulate by reward signal
    // dw = eligibility * reward / 128 (scaling to keep in range)
    let dw = ((eligibility as i16 * reward as i16) >> 7) as i8;

    // Apply weight update with bounds
    let new_weight =
        (synapse.weight as i16 + dw as i16).clamp(config.w_min as i16, config.w_max as i16) as i8;
    synapse.weight = new_weight;
}

/// Eligibility trace for a single synapse
///
/// Accumulates eligibility over time for delayed reward modulation.
#[derive(Clone, Copy, Debug, Default)]
pub struct EligibilityTrace {
    /// Current eligibility value (Q1.7)
    pub value: i8,
    /// Tick of last update
    pub last_update: u8,
}

impl EligibilityTrace {
    /// Create a new eligibility trace
    pub fn new() -> Self {
        Self::default()
    }

    /// Update eligibility with a spike pair
    pub fn update(&mut self, pre_time: u8, post_time: u8, config: &STDPConfig, current_tick: u8) {
        // First, decay existing eligibility
        self.decay(current_tick, config);

        // Then add new eligibility from spike pair
        let new_eligibility = compute_eligibility(pre_time, post_time, config);
        self.value = self.value.saturating_add(new_eligibility);
        self.last_update = current_tick;
    }

    /// Decay eligibility over time
    pub fn decay(&mut self, current_tick: u8, config: &STDPConfig) {
        let dt = current_tick.wrapping_sub(self.last_update);
        if dt > 0 {
            // Apply decay: value = value * decay^dt
            // Approximate with single decay step for each tick elapsed
            for _ in 0..dt.min(10) {
                // Cap iterations to avoid slowdown
                self.value = ((self.value as i16 * config.eligibility_decay as i16) >> 8) as i8;
            }
            self.last_update = current_tick;
        }
    }

    /// Apply accumulated eligibility with reward
    pub fn apply_reward(&self, synapse: &mut Synapse, reward: i8, config: &STDPConfig) {
        let dw = ((self.value as i16 * reward as i16) >> 7) as i8;
        let new_weight = (synapse.weight as i16 + dw as i16)
            .clamp(config.w_min as i16, config.w_max as i16) as i8;
        synapse.weight = new_weight;
    }

    /// Reset the trace
    pub fn reset(&mut self) {
        self.value = 0;
        self.last_update = 0;
    }
}

/// Batch STDP processor for a population of synapses
///
/// Efficiently processes STDP updates for all synapses after a simulation step.
#[derive(Clone, Debug)]
pub struct STDPProcessor {
    /// STDP configuration
    pub config: STDPConfig,
    /// Eligibility traces (indexed by synapse ID)
    pub traces: Vec<EligibilityTrace>,
}

impl STDPProcessor {
    /// Create a new STDP processor
    pub fn new(config: STDPConfig, n_synapses: usize) -> Self {
        Self {
            config,
            traces: vec![EligibilityTrace::new(); n_synapses],
        }
    }

    /// Process STDP for a single synapse
    pub fn process_synapse(
        &mut self,
        synapse_idx: usize,
        synapse: &mut Synapse,
        pre_spike_time: Option<u8>,
        post_spike_time: Option<u8>,
        current_tick: u8,
    ) {
        if let (Some(pre), Some(post)) = (pre_spike_time, post_spike_time) {
            if synapse_idx < self.traces.len() {
                self.traces[synapse_idx].update(pre, post, &self.config, current_tick);
            }
            apply_stdp(synapse, pre, post, &self.config);
        }
    }

    /// Apply reward signal to all eligibility traces
    pub fn apply_reward(&mut self, synapses: &mut [Synapse], reward: i8) {
        for (i, synapse) in synapses.iter_mut().enumerate() {
            if i < self.traces.len() {
                self.traces[i].apply_reward(synapse, reward, &self.config);
            }
        }
    }

    /// Decay all eligibility traces
    pub fn decay_all(&mut self, current_tick: u8) {
        for trace in &mut self.traces {
            trace.decay(current_tick, &self.config);
        }
    }

    /// Reset all traces
    pub fn reset(&mut self) {
        for trace in &mut self.traces {
            trace.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_decay() {
        // At dt=0, should be maximum
        assert_eq!(exp_decay_fixed(0, 20), 127);

        // At dt=tau, should be ~37% (e^-1 ≈ 0.37)
        // With our approximation: 127 * (1 - 20/20) = 0... our approximation is linear
        let decay = exp_decay_fixed(20, 20);
        assert!(decay < 10, "Decay at tau should be small: {}", decay);

        // At dt >> tau, should be ~0
        let decay_far = exp_decay_fixed(100, 20);
        assert_eq!(decay_far, 0);
    }

    #[test]
    fn test_stdp_ltp() {
        let config = STDPConfig::default();
        let mut synapse = Synapse::excitatory(0, 0, 0); // Start at weight 0

        // Post fires 5 ticks after pre → LTP
        apply_stdp(&mut synapse, 10, 15, &config);

        assert!(
            synapse.weight > 0,
            "LTP should increase weight: {}",
            synapse.weight
        );
    }

    #[test]
    fn test_stdp_ltd() {
        let config = STDPConfig::default();
        let mut synapse = Synapse::excitatory(0, 50, 0); // Start at weight 50

        // Post fires 5 ticks BEFORE pre (pre at 20, post at 15)
        // Using wrapping: dt = 15 - 20 = 251 (wraps), which is > 128, so LTD
        apply_stdp(&mut synapse, 20, 15, &config);

        assert!(
            synapse.weight < 50,
            "LTD should decrease weight: {}",
            synapse.weight
        );
    }

    #[test]
    fn test_stdp_bounds() {
        let config = STDPConfig {
            w_min: -50,
            w_max: 50,
            ..STDPConfig::default()
        };

        let mut synapse = Synapse::excitatory(0, 45, 0);

        // Multiple LTP should be bounded
        for i in 0..10 {
            apply_stdp(&mut synapse, i, i + 2, &config);
        }

        assert!(
            synapse.weight <= 50,
            "Weight should be bounded: {}",
            synapse.weight
        );
    }

    #[test]
    fn test_rstdp_positive_reward() {
        let config = STDPConfig::default();
        let mut synapse = Synapse::excitatory(0, 0, 0);

        // Positive eligibility + positive reward → increase
        apply_rstdp(&mut synapse, 10, 15, 64, &config); // reward = 64 (positive)

        assert!(
            synapse.weight > 0,
            "Positive eligibility + reward should increase weight"
        );
    }

    #[test]
    fn test_rstdp_negative_reward() {
        let config = STDPConfig::default();
        let mut synapse = Synapse::excitatory(0, 50, 0);

        // Positive eligibility + negative reward → decrease
        apply_rstdp(&mut synapse, 10, 15, -64, &config); // reward = -64 (punishment)

        assert!(
            synapse.weight < 50,
            "Positive eligibility + punishment should decrease weight"
        );
    }

    #[test]
    fn test_eligibility_trace_decay() {
        let config = STDPConfig {
            eligibility_decay: 128, // 0.5 per tick
            ..STDPConfig::default()
        };

        let mut trace = EligibilityTrace {
            value: 100,
            last_update: 0,
        };

        trace.decay(1, &config);
        assert_eq!(trace.value, 50); // 100 * 0.5

        trace.decay(2, &config);
        assert_eq!(trace.value, 25); // 50 * 0.5
    }

    #[test]
    fn test_eligibility_trace_update() {
        let config = STDPConfig::default();
        let mut trace = EligibilityTrace::new();

        // Add positive eligibility
        trace.update(10, 15, &config, 15);
        assert!(trace.value > 0, "Trace should be positive: {}", trace.value);

        // Add negative eligibility
        trace.update(25, 20, &config, 25);
        // Value should have decayed + added negative
        // Result depends on decay rate
    }

    #[test]
    fn test_stdp_processor() {
        let config = STDPConfig::default();
        let mut processor = STDPProcessor::new(config, 2);

        let mut synapses = vec![Synapse::excitatory(0, 0, 0), Synapse::excitatory(1, 0, 0)];

        // Process LTP for synapse 0
        processor.process_synapse(0, &mut synapses[0], Some(10), Some(15), 15);
        assert!(synapses[0].weight > 0);

        // Apply reward to all
        processor.apply_reward(&mut synapses, 64);
    }
}
