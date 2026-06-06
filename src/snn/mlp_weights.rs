//! MLP weight snapshot with CPU inference and binary serialization.
//!
//! This module is intentionally not gated behind the `cuda` feature so that
//! the tile_cpu MMIO SNN bridge can perform CPU-side MLP inference without
//! requiring a GPU.

use std::io::{self, Read, Write};

/// Magic bytes for MLP weight checkpoint files.
const MAGIC: [u8; 4] = *b"MLPw";
/// Current checkpoint format version.
const VERSION: u32 = 1;

/// Magic bytes for cached hidden-rate files.
const RATES_MAGIC: [u8; 4] = *b"MLPr";

/// 3-layer MLP weight snapshot.
///
/// Weight layout matches the GPU kernels: `W[input_idx * out_dim + output_idx]`
/// (row = input neuron, col = output neuron).
#[derive(Debug, Clone)]
pub struct MlpWeights {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub w3: Vec<f32>,
    pub b3: Vec<f32>,
}

impl MlpWeights {
    /// CPU forward pass: hidden_rates → predicted class (0..n_classes-1).
    ///
    /// Architecture: `rates[n_hid] → ReLU(W1+b1) → ReLU(W2+b2) → W3+b3 → argmax`
    ///
    /// Weight layout: `W[in_idx * out_dim + out_idx]` (matches GPU kernels).
    pub fn forward_cpu(&self, hidden_rates: &[f32]) -> usize {
        let n_hid = hidden_rates.len();
        let mlp_h1 = self.b1.len();
        let mlp_h2 = self.b2.len();
        let n_classes = self.b3.len();

        debug_assert_eq!(self.w1.len(), n_hid * mlp_h1);
        debug_assert_eq!(self.w2.len(), mlp_h1 * mlp_h2);
        debug_assert_eq!(self.w3.len(), mlp_h2 * n_classes);

        // Layer 1: z1 = rates @ W1 + b1, a1 = relu(z1)
        let mut a1 = vec![0.0f32; mlp_h1];
        for j in 0..mlp_h1 {
            let mut sum = self.b1[j];
            for h in 0..n_hid {
                sum += hidden_rates[h] * self.w1[h * mlp_h1 + j];
            }
            a1[j] = sum.max(0.0);
        }

        // Layer 2: z2 = a1 @ W2 + b2, a2 = relu(z2)
        let mut a2 = vec![0.0f32; mlp_h2];
        for k in 0..mlp_h2 {
            let mut sum = self.b2[k];
            for j in 0..mlp_h1 {
                sum += a1[j] * self.w2[j * mlp_h2 + k];
            }
            a2[k] = sum.max(0.0);
        }

        // Layer 3: logits = a2 @ W3 + b3, argmax
        let mut best_c = 0;
        let mut best_v = f32::NEG_INFINITY;
        for c in 0..n_classes {
            let mut sum = self.b3[c];
            for k in 0..mlp_h2 {
                sum += a2[k] * self.w3[k * n_classes + c];
            }
            if sum > best_v {
                best_v = sum;
                best_c = c;
            }
        }

        best_c
    }

    /// Inferred dimensions: (n_hid, mlp_h1, mlp_h2, n_classes).
    pub fn dims(&self) -> (usize, usize, usize, usize) {
        let mlp_h1 = self.b1.len();
        let mlp_h2 = self.b2.len();
        let n_classes = self.b3.len();
        let n_hid = if mlp_h1 > 0 {
            self.w1.len() / mlp_h1
        } else {
            0
        };
        (n_hid, mlp_h1, mlp_h2, n_classes)
    }

    /// Save weights to a binary checkpoint file.
    ///
    /// Format: magic(4) + version(4) + 4 dims(4 each) + 6 weight vectors (f32 LE).
    pub fn save(&self, path: &str) -> io::Result<()> {
        let (n_hid, mlp_h1, mlp_h2, n_classes) = self.dims();
        let mut f = std::fs::File::create(path)?;
        f.write_all(&MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(n_hid as u32).to_le_bytes())?;
        f.write_all(&(mlp_h1 as u32).to_le_bytes())?;
        f.write_all(&(mlp_h2 as u32).to_le_bytes())?;
        f.write_all(&(n_classes as u32).to_le_bytes())?;

        fn write_vec(f: &mut std::fs::File, v: &[f32]) -> io::Result<()> {
            for &x in v {
                f.write_all(&x.to_le_bytes())?;
            }
            Ok(())
        }
        write_vec(&mut f, &self.w1)?;
        write_vec(&mut f, &self.b1)?;
        write_vec(&mut f, &self.w2)?;
        write_vec(&mut f, &self.b2)?;
        write_vec(&mut f, &self.w3)?;
        write_vec(&mut f, &self.b3)?;
        Ok(())
    }

    /// Load weights from a binary checkpoint file.
    pub fn load(path: &str) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;

        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not an MLP weight checkpoint",
            ));
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let ver = u32::from_le_bytes(buf4);
        if ver != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported checkpoint version {ver}"),
            ));
        }

        let read_u32 = |f: &mut std::fs::File| -> io::Result<u32> {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let n_hid = read_u32(&mut f)? as usize;
        let mlp_h1 = read_u32(&mut f)? as usize;
        let mlp_h2 = read_u32(&mut f)? as usize;
        let n_classes = read_u32(&mut f)? as usize;

        fn read_vec(f: &mut std::fs::File, n: usize) -> io::Result<Vec<f32>> {
            let mut v = vec![0.0f32; n];
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
            f.read_exact(bytes)?;
            // Convert from LE if needed (no-op on LE platforms)
            #[cfg(target_endian = "big")]
            for x in v.iter_mut() {
                *x = f32::from_le_bytes(x.to_le_bytes());
            }
            Ok(v)
        }

        Ok(Self {
            w1: read_vec(&mut f, n_hid * mlp_h1)?,
            b1: read_vec(&mut f, mlp_h1)?,
            w2: read_vec(&mut f, mlp_h1 * mlp_h2)?,
            b2: read_vec(&mut f, mlp_h2)?,
            w3: read_vec(&mut f, mlp_h2 * n_classes)?,
            b3: read_vec(&mut f, n_classes)?,
        })
    }
}

/// Magic bytes for live SNN model checkpoint files.
const LIVE_MAGIC: [u8; 4] = *b"SNNl";

/// Pre-computed hidden firing rates for a set of samples.
///
/// Each sample has `n_hid` f32 values representing the mean firing rate
/// of each hidden neuron over the simulation period.
#[derive(Debug, Clone)]
pub struct CachedRates {
    pub n_hid: usize,
    pub rates: Vec<f32>, // [n_samples * n_hid], row-major
}

impl CachedRates {
    pub fn new(n_hid: usize, rates: Vec<f32>) -> Self {
        debug_assert_eq!(rates.len() % n_hid, 0);
        Self { n_hid, rates }
    }

    pub fn n_samples(&self) -> usize {
        if self.n_hid == 0 {
            0
        } else {
            self.rates.len() / self.n_hid
        }
    }

    /// Get hidden rates for sample `idx`.
    pub fn get(&self, idx: usize) -> &[f32] {
        let start = idx * self.n_hid;
        &self.rates[start..start + self.n_hid]
    }

    /// Save to binary file.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(&RATES_MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(self.n_samples() as u32).to_le_bytes())?;
        f.write_all(&(self.n_hid as u32).to_le_bytes())?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.rates.as_ptr() as *const u8, self.rates.len() * 4)
        };
        f.write_all(bytes)?;
        Ok(())
    }

    /// Load from binary file.
    pub fn load(path: &str) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;

        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if magic != RATES_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a cached rates file",
            ));
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let ver = u32::from_le_bytes(buf4);
        if ver != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported rates version {ver}"),
            ));
        }

        f.read_exact(&mut buf4)?;
        let n_samples = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let n_hid = u32::from_le_bytes(buf4) as usize;

        let total = n_samples * n_hid;
        let mut rates = vec![0.0f32; total];
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(rates.as_mut_ptr() as *mut u8, total * 4) };
        f.read_exact(bytes)?;
        Ok(Self { n_hid, rates })
    }
}

/// Full SNN + MLP model for live CPU inference (no GPU required).
///
/// Contains everything needed to replicate GPU training's inference path:
/// - CSR synapse connectivity with trained weights (i8)
/// - Per-neuron thresholds and leaks
/// - Pixel selection indices per class
/// - Encoding parameters (K, N_CLASSES, MAX_RATE, TICKS)
/// - MLP weights for readout
#[derive(Debug, Clone)]
pub struct LiveSnnModel {
    // CSR synapse topology + trained weights
    pub syn_ptr: Vec<u32>,
    pub targets: Vec<u16>,
    pub weights: Vec<i8>,

    // Per-neuron parameters
    pub thresholds: Vec<i16>,
    pub leaks: Vec<u8>,

    // Pixel selection (from training)
    pub pix_per_class: Vec<Vec<usize>>,
    pub d_norms: Vec<Vec<f32>>,

    // Architecture constants
    pub n_input: usize,
    pub n_hidden: usize,
    pub n_readout: usize,
    pub n_classes: usize,
    pub k_per_class: usize,
    pub max_rate: u32,
    pub n_ticks: usize,

    // MLP readout weights
    pub mlp: MlpWeights,
}

impl LiveSnnModel {
    /// Total number of neurons (input + hidden + readout).
    pub fn n_neurons(&self) -> usize {
        self.n_input + self.n_hidden + self.n_readout
    }

    /// Total number of synapses.
    pub fn n_synapses(&self) -> usize {
        self.targets.len()
    }

    /// Encode an image into Poisson input rates.
    ///
    /// Replicates the `encode_image()` closure from M10 training:
    /// `rates[c*K+i] = (img[pix] * MAX_RATE / 255) as u8`
    pub fn encode_image(&self, img: &[u8]) -> Vec<u8> {
        let mut rates = vec![0u8; self.n_input];
        for c in 0..self.n_classes {
            for (i, &pix) in self.pix_per_class[c].iter().enumerate() {
                if pix < img.len() {
                    rates[c * self.k_per_class + i] =
                        ((img[pix] as u32 * self.max_rate) / 255) as u8;
                }
            }
        }
        rates
    }

    /// Save to binary checkpoint file.
    ///
    /// Format: magic(4) + version(4) + architecture header + CSR + neurons + pixels + MLP
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(&LIVE_MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;

        // Architecture header (10 u32s)
        for &v in &[
            self.n_input as u32,
            self.n_hidden as u32,
            self.n_readout as u32,
            self.n_classes as u32,
            self.k_per_class as u32,
            self.max_rate,
            self.n_ticks as u32,
            self.n_neurons() as u32,
            self.n_synapses() as u32,
            0u32, // reserved
        ] {
            f.write_all(&v.to_le_bytes())?;
        }

        // CSR: syn_ptr
        for &v in &self.syn_ptr {
            f.write_all(&v.to_le_bytes())?;
        }
        // CSR: targets (u16)
        for &v in &self.targets {
            f.write_all(&v.to_le_bytes())?;
        }
        // CSR: weights (i8)
        f.write_all(unsafe {
            std::slice::from_raw_parts(self.weights.as_ptr() as *const u8, self.weights.len())
        })?;

        // Per-neuron thresholds (i16 LE)
        for &v in &self.thresholds {
            f.write_all(&v.to_le_bytes())?;
        }
        // Per-neuron leaks (u8)
        f.write_all(&self.leaks)?;

        // Pixel indices: [n_classes][k_per_class] as u32
        for class_pixels in &self.pix_per_class {
            for &pix in class_pixels {
                f.write_all(&(pix as u32).to_le_bytes())?;
            }
        }
        // D_norms: [n_classes][k_per_class] as f32
        for class_norms in &self.d_norms {
            for &d in class_norms {
                f.write_all(&d.to_le_bytes())?;
            }
        }

        // MLP weights (embedded, no separate magic/version)
        let (n_hid, mlp_h1, mlp_h2, n_cls) = self.mlp.dims();
        for &v in &[n_hid as u32, mlp_h1 as u32, mlp_h2 as u32, n_cls as u32] {
            f.write_all(&v.to_le_bytes())?;
        }
        fn write_f32_vec(f: &mut std::fs::File, v: &[f32]) -> io::Result<()> {
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
            f.write_all(bytes)
        }
        write_f32_vec(&mut f, &self.mlp.w1)?;
        write_f32_vec(&mut f, &self.mlp.b1)?;
        write_f32_vec(&mut f, &self.mlp.w2)?;
        write_f32_vec(&mut f, &self.mlp.b2)?;
        write_f32_vec(&mut f, &self.mlp.w3)?;
        write_f32_vec(&mut f, &self.mlp.b3)?;

        Ok(())
    }

    /// Load from binary checkpoint file.
    pub fn load(path: &str) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;

        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if magic != LIVE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a live SNN model checkpoint",
            ));
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let ver = u32::from_le_bytes(buf4);
        if ver != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported live model version {ver}"),
            ));
        }

        let read_u32 = |f: &mut std::fs::File| -> io::Result<u32> {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };

        let n_input = read_u32(&mut f)? as usize;
        let n_hidden = read_u32(&mut f)? as usize;
        let n_readout = read_u32(&mut f)? as usize;
        let n_classes = read_u32(&mut f)? as usize;
        let k_per_class = read_u32(&mut f)? as usize;
        let max_rate = read_u32(&mut f)?;
        let n_ticks = read_u32(&mut f)? as usize;
        let n_neurons = read_u32(&mut f)? as usize;
        let n_synapses = read_u32(&mut f)? as usize;
        let _reserved = read_u32(&mut f)?;

        // CSR: syn_ptr [n_neurons + 1]
        let mut syn_ptr = vec![0u32; n_neurons + 1];
        for v in syn_ptr.iter_mut() {
            *v = read_u32(&mut f)?;
        }

        // CSR: targets [n_synapses] as u16
        let mut targets = vec![0u16; n_synapses];
        {
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(targets.as_mut_ptr() as *mut u8, n_synapses * 2)
            };
            f.read_exact(bytes)?;
        }

        // CSR: weights [n_synapses] as i8
        let mut weights = vec![0i8; n_synapses];
        {
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(weights.as_mut_ptr() as *mut u8, n_synapses)
            };
            f.read_exact(bytes)?;
        }

        // Per-neuron thresholds [n_neurons] as i16
        let mut thresholds = vec![0i16; n_neurons];
        {
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(thresholds.as_mut_ptr() as *mut u8, n_neurons * 2)
            };
            f.read_exact(bytes)?;
        }

        // Per-neuron leaks [n_neurons] as u8
        let mut leaks = vec![0u8; n_neurons];
        f.read_exact(&mut leaks)?;

        // Pixel indices [n_classes][k_per_class] as u32
        let mut pix_per_class = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            let mut class_pix = Vec::with_capacity(k_per_class);
            for _ in 0..k_per_class {
                class_pix.push(read_u32(&mut f)? as usize);
            }
            pix_per_class.push(class_pix);
        }

        // D_norms [n_classes][k_per_class] as f32
        let mut d_norms = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            let mut class_norms = vec![0.0f32; k_per_class];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        class_norms.as_mut_ptr() as *mut u8,
                        k_per_class * 4,
                    )
                };
                f.read_exact(bytes)?;
            }
            d_norms.push(class_norms);
        }

        // MLP weights (embedded)
        let mlp_n_hid = read_u32(&mut f)? as usize;
        let mlp_h1 = read_u32(&mut f)? as usize;
        let mlp_h2 = read_u32(&mut f)? as usize;
        let mlp_n_cls = read_u32(&mut f)? as usize;

        fn read_f32_vec(f: &mut std::fs::File, n: usize) -> io::Result<Vec<f32>> {
            let mut v = vec![0.0f32; n];
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
            f.read_exact(bytes)?;
            Ok(v)
        }

        let mlp = MlpWeights {
            w1: read_f32_vec(&mut f, mlp_n_hid * mlp_h1)?,
            b1: read_f32_vec(&mut f, mlp_h1)?,
            w2: read_f32_vec(&mut f, mlp_h1 * mlp_h2)?,
            b2: read_f32_vec(&mut f, mlp_h2)?,
            w3: read_f32_vec(&mut f, mlp_h2 * mlp_n_cls)?,
            b3: read_f32_vec(&mut f, mlp_n_cls)?,
        };

        Ok(Self {
            syn_ptr,
            targets,
            weights,
            thresholds,
            leaks,
            pix_per_class,
            d_norms,
            n_input,
            n_hidden,
            n_readout,
            n_classes,
            k_per_class,
            max_rate,
            n_ticks,
            mlp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_weights() -> MlpWeights {
        // Tiny 2-2-2-2 network for testing
        MlpWeights {
            w1: vec![1.0, 0.0, 0.0, 1.0], // 2x2 identity
            b1: vec![0.0, 0.0],
            w2: vec![1.0, 0.0, 0.0, 1.0], // 2x2 identity
            b2: vec![0.0, 0.0],
            w3: vec![1.0, -1.0, -1.0, 1.0], // 2x2 contrast
            b3: vec![0.0, 0.0],
        }
    }

    #[test]
    fn test_forward_cpu_basic() {
        let w = test_weights();
        // Input [1.0, 0.0] → through identity layers → logits [1.0, -1.0] → argmax 0
        assert_eq!(w.forward_cpu(&[1.0, 0.0]), 0);
        // Input [0.0, 1.0] → through identity layers → logits [-1.0, 1.0] → argmax 1
        assert_eq!(w.forward_cpu(&[0.0, 1.0]), 1);
    }

    #[test]
    fn test_forward_cpu_relu() {
        let w = MlpWeights {
            w1: vec![1.0, 0.0, 0.0, 1.0],
            b1: vec![-0.5, 0.0], // Bias shifts first neuron negative for small inputs
            w2: vec![1.0, 0.0, 0.0, 1.0],
            b2: vec![0.0, 0.0],
            w3: vec![1.0, 0.0, 0.0, 1.0],
            b3: vec![0.0, 0.0],
        };
        // Input [0.3, 1.0]:
        //   L1: z1 = [0.3-0.5, 1.0] = [-0.2, 1.0], a1 = [0.0, 1.0] (ReLU clips)
        //   L2: z2 = [0.0, 1.0], a2 = [0.0, 1.0]
        //   L3: logits = [0.0, 1.0] → argmax 1
        assert_eq!(w.forward_cpu(&[0.3, 1.0]), 1);
    }

    #[test]
    fn test_dims() {
        let w = test_weights();
        assert_eq!(w.dims(), (2, 2, 2, 2));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let w = test_weights();
        let path = std::env::temp_dir()
            .join("test_mlp_weights.bin")
            .to_str()
            .unwrap()
            .to_string();
        w.save(&path).unwrap();
        let w2 = MlpWeights::load(&path).unwrap();
        assert_eq!(w.w1, w2.w1);
        assert_eq!(w.b1, w2.b1);
        assert_eq!(w.w2, w2.w2);
        assert_eq!(w.b2, w2.b2);
        assert_eq!(w.w3, w2.w3);
        assert_eq!(w.b3, w2.b3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cached_rates_save_load() {
        let rates = CachedRates::new(3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(rates.n_samples(), 2);
        assert_eq!(rates.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(rates.get(1), &[4.0, 5.0, 6.0]);

        let path = std::env::temp_dir()
            .join("test_cached_rates.bin")
            .to_str()
            .unwrap()
            .to_string();
        rates.save(&path).unwrap();
        let r2 = CachedRates::load(&path).unwrap();
        assert_eq!(r2.n_hid, 3);
        assert_eq!(r2.rates, rates.rates);
        let _ = std::fs::remove_file(&path);
    }

    fn test_live_model() -> LiveSnnModel {
        // Tiny 4-input, 2-hidden, 1-readout, 2-class model
        LiveSnnModel {
            syn_ptr: vec![0, 2, 3, 5, 6, 7, 7, 7], // 7 neurons, varying fan-out
            targets: vec![4, 5, 5, 4, 5, 6, 6],
            weights: vec![100, -50, 80, 60, -30, 40, 50],
            thresholds: vec![32000, 32000, 32000, 32000, 200, 200, 400],
            leaks: vec![230; 7],
            pix_per_class: vec![vec![0, 1], vec![2, 3]],
            d_norms: vec![vec![1.0, 0.8], vec![0.9, 1.0]],
            n_input: 4,
            n_hidden: 2,
            n_readout: 1,
            n_classes: 2,
            k_per_class: 2,
            max_rate: 100,
            n_ticks: 10,
            mlp: test_weights(),
        }
    }

    #[test]
    fn test_live_model_save_load_roundtrip() {
        let model = test_live_model();
        let path = std::env::temp_dir()
            .join("test_live_snn_model.bin")
            .to_str()
            .unwrap()
            .to_string();
        model.save(&path).unwrap();
        let m2 = LiveSnnModel::load(&path).unwrap();

        assert_eq!(m2.syn_ptr, model.syn_ptr);
        assert_eq!(m2.targets, model.targets);
        assert_eq!(m2.weights, model.weights);
        assert_eq!(m2.thresholds, model.thresholds);
        assert_eq!(m2.leaks, model.leaks);
        assert_eq!(m2.pix_per_class, model.pix_per_class);
        assert_eq!(m2.d_norms, model.d_norms);
        assert_eq!(m2.n_input, model.n_input);
        assert_eq!(m2.n_hidden, model.n_hidden);
        assert_eq!(m2.n_readout, model.n_readout);
        assert_eq!(m2.n_classes, model.n_classes);
        assert_eq!(m2.k_per_class, model.k_per_class);
        assert_eq!(m2.max_rate, model.max_rate);
        assert_eq!(m2.n_ticks, model.n_ticks);
        assert_eq!(m2.mlp.w1, model.mlp.w1);
        assert_eq!(m2.mlp.b1, model.mlp.b1);
        assert_eq!(m2.mlp.w3, model.mlp.w3);
        assert_eq!(m2.mlp.b3, model.mlp.b3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_live_model_encode_image() {
        let model = test_live_model();
        // Image with 4 pixels: [255, 128, 0, 200]
        let img = vec![255u8, 128, 0, 200];
        let rates = model.encode_image(&img);
        assert_eq!(rates.len(), 4); // n_input = 4
        // class 0: pix [0,1] → rates[0] = 255*100/255=100, rates[1] = 128*100/255=50
        assert_eq!(rates[0], 100);
        assert_eq!(rates[1], 50);
        // class 1: pix [2,3] → rates[2] = 0*100/255=0, rates[3] = 200*100/255=78
        assert_eq!(rates[2], 0);
        assert_eq!(rates[3], 78);
    }

    #[test]
    fn test_live_model_n_neurons() {
        let model = test_live_model();
        assert_eq!(model.n_neurons(), 7); // 4 + 2 + 1
        assert_eq!(model.n_synapses(), 7);
    }
}
