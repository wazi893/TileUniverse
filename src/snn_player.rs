//! Self-contained "Spiking neural network" HTML player.
//!
//! Captures a real LIF spiking-network simulation from [`crate::snn`]: a layered
//! leaky-integrate-and-fire network ([`SNNNetwork`]) is driven by a bursting
//! input stimulus and stepped tick-by-tick, recording which neurons spike and
//! every neuron's membrane potential. The network's synapses (excitatory /
//! inhibitory) are captured for the connectivity graph. The player animates the
//! firing network with a synced spike raster plot.
//!
//! Deterministic (seeded, fixed-point integer dynamics) and CPU-only.

use crate::snn::{SNNConfig, SNNNetwork};
use std::fs::{create_dir_all, write};
use std::io;
use std::path::Path;

const TEMPLATE: &str = include_str!("snn_player_template.html");
const MARKER: &str = "/*__SNN_BUNDLE__*/null";

/// V_RESET in volts (Q8.8 -128 = -0.5); used to normalise membrane potential.
const V_RESET: f64 = -0.5;

struct SnnDataset {
    name: String,
    neurons: Vec<(f64, f64, usize)>, // (x, y, layer)
    layers: Vec<(usize, usize, usize)>,
    synapses: Vec<(usize, usize, i32, u8)>, // (pre, post, weight*100, excitatory)
    ticks: usize,
    spikes: Vec<Vec<usize>>,
    volt: Vec<u8>, // ticks * n, row-major (tick outer)
    n: usize,
    spike_total: usize,
}

/// A capture configuration.
pub struct SnnConfig {
    pub name: String,
    pub n_inputs: usize,
    pub hidden: Vec<usize>,
    pub n_outputs: usize,
    pub ticks: usize,
    pub seed: u64,
    /// If true use channel-structured connectivity, else simple feedforward.
    pub structured: bool,
}

/// Default switchable demos.
pub fn default_configs() -> Vec<SnnConfig> {
    vec![
        SnnConfig {
            name: "Feedforward · 16→40→32→12".into(),
            n_inputs: 16,
            hidden: vec![40, 32],
            n_outputs: 12,
            ticks: 320,
            seed: 0x5EED,
            structured: false,
        },
        SnnConfig {
            name: "Deep cascade · 12→36→28→20→10".into(),
            n_inputs: 12,
            hidden: vec![36, 28, 20],
            n_outputs: 10,
            ticks: 320,
            seed: 0xBEE5,
            structured: false,
        },
    ]
}

fn neuron_ref(net: &SNNNetwork, g: usize) -> &crate::snn::LIFNeuron {
    let cpu = net.topology.neuron_to_cpu[g];
    let (start, _) = net.topology.cpu_to_neurons[cpu];
    &net.populations[cpu].neurons[g - start]
}

fn capture(cfg: &SnnConfig) -> SnnDataset {
    let config = SNNConfig::new(cfg.n_inputs, cfg.hidden.clone(), cfg.n_outputs);
    let mut net = SNNNetwork::new(config);
    if cfg.structured {
        net.build_connectivity(cfg.seed);
    } else {
        net.build_simple_connectivity(cfg.seed);
    }
    let n = net.n_neurons();
    let layers = net.topology.layers.clone();

    // Layered positions.
    let _max_size = layers.iter().map(|&(_, s, e)| e - s).max().unwrap_or(1) as f64;
    let mut neurons = vec![(0.0, 0.0, 0usize); n];
    for &(id, s, e) in &layers {
        let size = (e - s) as f64;
        for (k, g) in (s..e).enumerate() {
            let y = k as f64 - (size - 1.0) / 2.0;
            neurons[g] = (id as f64, y, id);
        }
    }

    // Synapses (single/few CPU: targets are local to the source's CPU).
    let mut synapses = Vec::new();
    for cpu in 0..net.topology.n_cpus {
        let (start, _) = net.topology.cpu_to_neurons[cpu];
        for (src_local, syns) in net.synapses[cpu].local.iter().enumerate() {
            let pre = start + src_local;
            for syn in syns {
                let post = start + syn.target as usize;
                if post >= n {
                    continue;
                }
                let w = (syn.weight_f32() * 100.0).round() as i32;
                synapses.push((pre, post, w, syn.is_excitatory() as u8));
            }
        }
    }

    // Per-neuron threshold for voltage normalisation.
    let thr: Vec<f64> = (0..n)
        .map(|g| neuron_ref(&net, g).threshold_f32() as f64)
        .collect();

    let mut spikes = Vec::with_capacity(cfg.ticks);
    let mut volt = Vec::with_capacity(cfg.ticks * n);
    let mut spike_total = 0usize;

    let rates = vec![0u8; cfg.n_inputs];
    let mut rates = rates;
    for tick in 0..cfg.ticks {
        // Bursting drive: periodic strong pulse, otherwise moderate baseline.
        let burst = (tick % 24) < 6;
        let base = if burst { 210u8 } else { 95u8 };
        for (i, r) in rates.iter_mut().enumerate() {
            // slight spatial variation so waves aren't perfectly uniform
            *r = base.saturating_sub(((i * 7) % 40) as u8);
        }
        net.set_inputs(&rates);
        net.step();

        let mut fired = Vec::new();
        for g in 0..n {
            let neuron = neuron_ref(&net, g);
            if neuron.fired() {
                fired.push(g);
            }
            let v = neuron.v_mem_f32() as f64;
            let norm = ((v - V_RESET) / (thr[g] - V_RESET)).clamp(0.0, 1.0);
            volt.push((norm * 255.0).round() as u8);
        }
        spike_total += fired.len();
        spikes.push(fired);
    }

    SnnDataset {
        name: cfg.name.clone(),
        neurons,
        layers,
        synapses,
        ticks: cfg.ticks,
        spikes,
        volt,
        n,
        spike_total,
    }
}

/// Build the populated player HTML for the given configs.
pub fn build_snn_html(configs: &[SnnConfig]) -> Result<String, String> {
    if configs.is_empty() {
        return Err("no configs".into());
    }
    let datasets: Vec<SnnDataset> = configs.iter().map(capture).collect();
    let mut json = String::from("{\"datasets\":[");
    for (i, d) in datasets.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&dataset_json(d));
    }
    json.push_str("],\"default\":0}");

    let n = TEMPLATE.matches(MARKER).count();
    if n != 1 {
        return Err(format!("template marker count = {n}, expected 1"));
    }
    Ok(TEMPLATE.replace(MARKER, &json))
}

/// Capture the default demos and write the player to `path`.
pub fn export_default_snn_player(path: impl AsRef<Path>) -> Result<(), String> {
    let html = build_snn_html(&default_configs())?;
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        create_dir_all(parent).map_err(|e: io::Error| e.to_string())?;
    }
    write(path, html).map_err(|e: io::Error| e.to_string())
}

fn dataset_json(d: &SnnDataset) -> String {
    let neurons = d
        .neurons
        .iter()
        .map(|(x, y, l)| format!("[{x},{y},{l}]"))
        .collect::<Vec<_>>()
        .join(",");
    let layers = d
        .layers
        .iter()
        .map(|(a, b, c)| format!("[{a},{b},{c}]"))
        .collect::<Vec<_>>()
        .join(",");
    let syn = d
        .synapses
        .iter()
        .map(|(p, q, w, e)| format!("[{p},{q},{w},{e}]"))
        .collect::<Vec<_>>()
        .join(",");
    let spk = d
        .spikes
        .iter()
        .map(|s| {
            format!(
                "[{}]",
                s.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"name\":\"{}\",\"n\":{},\"layers\":[{}],\"neurons\":[{}],\"synapses\":[{}],\"ticks\":{},\"spikeTotal\":{},\"spikes\":[{}],\"volt\":\"{}\"}}",
        json_escape(&d.name),
        d.n,
        layers,
        neurons,
        syn,
        d.ticks,
        d.spike_total,
        spk,
        base64(&d.volt)
    )
}

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let nn = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[((nn >> 18) & 63) as usize] as char);
        out.push(T[((nn >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[((nn >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(nn & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_produces_spikes() {
        let cfg = &default_configs()[0];
        let d = capture(cfg);
        assert_eq!(d.spikes.len(), d.ticks);
        assert_eq!(d.volt.len(), d.ticks * d.n);
        assert!(!d.synapses.is_empty(), "no synapses");
        // The driven network must actually spike.
        assert!(d.spike_total > 50, "too few spikes: {}", d.spike_total);
        // Synapse endpoints in range.
        for &(p, q, _, _) in &d.synapses {
            assert!(p < d.n && q < d.n);
        }
    }

    #[test]
    fn test_build_html_embeds_bundle() {
        let html = build_snn_html(&default_configs()[..1]).expect("build");
        assert!(!html.contains(MARKER));
        assert!(html.contains("\"datasets\":["));
        assert!(html.contains("\"synapses\":["));
        assert!(html.contains("\"volt\":\""));
    }
}
