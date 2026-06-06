//! MNIST DVS Classifier - End-to-End Example
//!
//! Demonstrates DVS encoding + STDP learning for MNIST digit classification.
//!
//! # Architecture
//!
//! 1. **DVS Encoder**: Converts static images to sparse temporal events
//! 2. **Rate Coding**: Maps events to firing rates (2 neurons per pixel: ON/OFF)
//! 3. **SNN**: 1568 → 128 → 10 feedforward network with STDP
//! 4. **Winner-Take-All**: Output neuron with highest spike count = predicted digit
//!
//! # Expected Performance
//!
//! - Training: 1000-5000 images, 5-10 epochs
//! - Accuracy: >85% on test set
//! - DVS Sparsity: 5-15% (10-100 events per frame)
//!
//! # Usage
//!
//! ```bash
//! # Download MNIST first (see docs/MILESTONE_1_PLAN.md)
//! cargo run --release --example mnist_dvs_classifier -- \
//!     --images data/train-images-idx3-ubyte \
//!     --labels data/train-labels-idx1-ubyte \
//!     --train-size 1000 \
//!     --epochs 5
//! ```

use std::time::Instant;

use engine::snn::{DvsRetina, MnistDataset, SNNConfig, SNNNetwork};

/// Training metrics for one epoch
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_accuracy: f32,
    pub test_accuracy: f32,
    pub avg_spikes_per_image: f32,
    pub avg_dvs_sparsity: f32,
    pub epoch_time_ms: u128,
}

impl EpochMetrics {
    pub fn log(&self) {
        println!(
            "Epoch {}: train={:.1}%, test={:.1}%, spikes={:.1}, sparsity={:.1}%, time={}ms",
            self.epoch,
            self.train_accuracy * 100.0,
            self.test_accuracy * 100.0,
            self.avg_spikes_per_image,
            self.avg_dvs_sparsity * 100.0,
            self.epoch_time_ms
        );
    }
}

/// Train one epoch on MNIST dataset
fn train_epoch(
    network: &mut SNNNetwork,
    retina: &mut DvsRetina,
    dataset: &MnistDataset,
    ticks_per_image: u32,
) -> (f32, f32, f32) {
    let mut correct = 0;
    let mut total = 0;
    let mut total_spikes = 0u64;
    let mut total_sparsity = 0.0f32;

    for (image, label) in dataset.images.iter().zip(&dataset.labels) {
        // Reset output spike counts
        network.reset_output_counts();

        // DVS encoding
        let events = retina.process_frame(image);
        let sparsity = retina.calculate_sparsity(&events);
        let rates = retina.encode_to_rates(&events);

        total_sparsity += sparsity;

        // Debug first image
        if total == 0 {
            let active_rates = rates.iter().filter(|&&r| r > 0).count();
            let max_rate = rates.iter().max().copied().unwrap_or(0);
            let sum_rates: u64 = rates.iter().map(|&r| r as u64).sum();
            println!(
                "  [Image 0 DVS] Events: {}, Active rates: {}/{}, Max rate: {}, Sum: {}",
                events.len(),
                active_rates,
                rates.len(),
                max_rate,
                sum_rates
            );
        }

        // Present for N ticks
        for tick in 0..ticks_per_image {
            network.set_inputs(&rates);
            network.step();

            // Debug first image only
            if total == 0 && tick < 5 {
                let spikes = network.spike_count();
                let max_v: Vec<i16> = network
                    .populations
                    .iter()
                    .map(|p| p.neurons.iter().map(|n| n.v_mem).max().unwrap_or(0))
                    .collect();
                println!(
                    "  [Image 0, Tick {}] Spikes: {}, Max V per layer: {:?}",
                    tick, spikes, max_v
                );
            }
        }

        // Count total spikes across network (for monitoring)
        let image_spikes = network.spike_count() as u64;
        total_spikes += image_spikes;

        if total == 0 {
            println!(
                "  [Image 0] Total spikes: {}, Final output counts: {:?}",
                image_spikes, network.output_counts
            );
        }

        // Read output (winner-take-all)
        let prediction = network.get_action();

        // Reward signal: use strong rewards like working example (+127/-100)
        let reward = if prediction == *label as usize {
            127
        } else {
            -100
        };

        network.set_reward(reward);
        // Try channeled learning (respects channel structure)
        network.apply_learning_channeled(prediction);

        if prediction == *label as usize {
            correct += 1;
        }
        total += 1;

        // Use current image as baseline for next image (temporal difference between frames)
        // This creates sparse events only where images differ
        retina.set_baseline(image);
    }

    let accuracy = correct as f32 / total as f32;
    let avg_spikes = total_spikes as f32 / total as f32;
    let avg_sparsity = total_sparsity / total as f32;

    (accuracy, avg_spikes, avg_sparsity)
}

/// Evaluate accuracy on test set (no learning)
fn evaluate(
    network: &mut SNNNetwork,
    retina: &mut DvsRetina,
    dataset: &MnistDataset,
    ticks_per_image: u32,
) -> f32 {
    let mut correct = 0;
    let mut total = 0;

    for (image, label) in dataset.images.iter().zip(&dataset.labels) {
        network.reset_output_counts();

        // DVS encoding
        let events = retina.process_frame(image);
        let rates = retina.encode_to_rates(&events);

        // Present for N ticks
        for _ in 0..ticks_per_image {
            network.set_inputs(&rates);
            network.step();
        }

        // Read output
        let prediction = network.get_action();

        if prediction == *label as usize {
            correct += 1;
        }
        total += 1;

        retina.set_baseline(image);
    }

    correct as f32 / total as f32
}

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();

    let images_path = args
        .iter()
        .position(|a| a == "--images")
        .and_then(|i| args.get(i + 1))
        .expect("Missing --images path");

    let labels_path = args
        .iter()
        .position(|a| a == "--labels")
        .and_then(|i| args.get(i + 1))
        .expect("Missing --labels path");

    let train_size: usize = args
        .iter()
        .position(|a| a == "--train-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let epochs: usize = args
        .iter()
        .position(|a| a == "--epochs")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let ticks_per_image: u32 = args
        .iter()
        .position(|a| a == "--ticks")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    println!("=== MNIST DVS Classifier ===");
    println!("Loading dataset from:");
    println!("  Images: {}", images_path);
    println!("  Labels: {}", labels_path);

    // Load MNIST dataset
    let dataset =
        MnistDataset::load(images_path, labels_path).expect("Failed to load MNIST dataset");

    println!("Dataset loaded: {} samples", dataset.len());

    // Split into train/test
    let (train_dataset, test_dataset) = dataset.split(train_size);

    println!("Train: {} samples", train_dataset.len());
    println!("Test: {} samples", test_dataset.len());

    // Create DVS retina encoder
    let mut retina = DvsRetina::new(28, 28, 20);

    // Create SNN network with DVS-optimized config
    let config = SNNConfig::for_dvs_mnist();
    let mut network = SNNNetwork::new(config.clone());

    println!("\nNetwork architecture:");
    println!(
        "  Input: {} neurons (28×28×2 DVS channels)",
        config.n_inputs
    );
    println!("  Hidden: {:?} neurons", config.hidden_layers);
    println!("  Output: {} neurons (digits 0-9)", config.n_outputs);
    println!("  Total: {} neurons", config.total_neurons());
    println!("  Connection prob: {:.1}%", config.connection_prob * 100.0);

    // Build connectivity
    println!("\nBuilding connectivity...");
    let seed = 42;
    network.build_connectivity(seed); // Use channel-aligned connectivity (like working example)
    println!("Connectivity built (seed={})", seed);

    println!("\nTraining parameters:");
    println!("  Epochs: {}", epochs);
    println!("  Ticks per image: {}", ticks_per_image);
    println!("  DVS threshold: 20");

    // Training loop
    println!("\n=== Training ===\n");

    let mut metrics_history = Vec::new();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();

        // Reset DVS baseline to mid-gray between epochs for consistent encoding
        retina.reset();

        // Train
        let (train_acc, avg_spikes, avg_sparsity) =
            train_epoch(&mut network, &mut retina, &train_dataset, ticks_per_image);

        // Evaluate on test set (reset DVS baseline first)
        retina.reset();
        let test_acc = evaluate(&mut network, &mut retina, &test_dataset, ticks_per_image);

        let epoch_time = epoch_start.elapsed().as_millis();

        let metrics = EpochMetrics {
            epoch: epoch + 1,
            train_accuracy: train_acc,
            test_accuracy: test_acc,
            avg_spikes_per_image: avg_spikes,
            avg_dvs_sparsity: avg_sparsity,
            epoch_time_ms: epoch_time,
        };

        metrics.log();
        metrics_history.push(metrics);
    }

    // Final summary
    println!("\n=== Training Complete ===");
    if let Some(final_metrics) = metrics_history.last() {
        println!(
            "Final train accuracy: {:.1}%",
            final_metrics.train_accuracy * 100.0
        );
        println!(
            "Final test accuracy: {:.1}%",
            final_metrics.test_accuracy * 100.0
        );
        println!(
            "Average DVS sparsity: {:.1}%",
            final_metrics.avg_dvs_sparsity * 100.0
        );

        if final_metrics.test_accuracy >= 0.85 {
            println!("\n✓ SUCCESS: Achieved >85% accuracy target!");
        } else {
            println!("\n✗ Target not reached. Try:");
            println!("  - Increase --train-size (more data)");
            println!("  - Increase --epochs (more learning)");
            println!("  - Tune DVS threshold or network size");
        }
    }
}
