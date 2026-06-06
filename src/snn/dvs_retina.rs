//! DVS (Dynamic Vision Sensor) Retina Encoder
//!
//! Converts static images into sparse spike events through temporal difference encoding.
//! Mimics biological retinal processing by detecting changes in brightness.
//!
//! # Design
//!
//! Static images don't generate DVS events (no temporal change). We create synthetic
//! temporal contrast by comparing the current frame against a "previous state"
//! (initialized to mid-gray 128).
//!
//! - **ON events**: Pixel brightened (delta > threshold) → spike from ON neuron
//! - **OFF events**: Pixel darkened (delta < -threshold) → spike from OFF neuron
//! - **Rate encoding**: Spike probability proportional to |delta|, clamped to [0, 255]
//!
//! # Example
//!
//! ```ignore
//! use engine::snn::dvs_retina::{DvsRetina, Polarity};
//!
//! let mut retina = DvsRetina::new(28, 28, 20);
//! let frame = vec![128u8; 28 * 28];  // Mid-gray frame
//!
//! // Process frame to get sparse events
//! let events = retina.process_frame(&frame);
//! assert_eq!(events.len(), 0);  // No change → no events
//!
//! // Convert events to rate-coded inputs for SNN
//! let rates = retina.encode_to_rates(&events);
//! assert_eq!(rates.len(), 28 * 28 * 2);  // ON + OFF per pixel
//! ```

/// DVS retina encoder that converts frames to sparse spike events
#[derive(Debug, Clone)]
pub struct DvsRetina {
    /// Frame width in pixels
    width: usize,
    /// Frame height in pixels
    height: usize,
    /// Previous frame for temporal difference (initialized to mid-gray 128)
    prev_frame: Vec<u8>,
    /// Minimum brightness change to trigger event (typical: 20-30)
    threshold: u8,
    /// Bias toward ON vs OFF events (1.0 = balanced, >1.0 = more ON)
    #[allow(dead_code)]
    on_polarity_bias: f32,
}

/// DVS event representing a single pixel brightness change
#[derive(Debug, Clone, PartialEq)]
pub struct DvsEvent {
    /// X coordinate (column)
    pub x: usize,
    /// Y coordinate (row)
    pub y: usize,
    /// ON (brighter) or OFF (darker)
    pub polarity: Polarity,
    /// Magnitude of change |delta| in [0, 255]
    pub magnitude: u8,
}

/// Event polarity: ON (brightening) or OFF (darkening)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// Brightness increase (positive delta)
    On,
    /// Brightness decrease (negative delta)
    Off,
}

impl DvsRetina {
    /// Create a new DVS retina encoder
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `threshold` - Minimum |delta| to trigger event (typical: 20-30)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let retina = DvsRetina::new(28, 28, 25);
    /// ```
    pub fn new(width: usize, height: usize, threshold: u8) -> Self {
        Self {
            width,
            height,
            prev_frame: vec![128; width * height], // Mid-gray baseline
            threshold,
            on_polarity_bias: 1.0,
        }
    }

    /// Create with custom ON/OFF bias
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `threshold` - Event threshold
    /// * `on_bias` - Bias toward ON events (1.0 = balanced, >1.0 = more ON)
    pub fn with_bias(width: usize, height: usize, threshold: u8, on_bias: f32) -> Self {
        Self {
            width,
            height,
            prev_frame: vec![128; width * height],
            threshold,
            on_polarity_bias: on_bias,
        }
    }

    /// Process a frame and generate DVS events
    ///
    /// Compares current frame against previous frame and generates events
    /// for pixels where |delta| > threshold.
    ///
    /// # Arguments
    ///
    /// * `frame` - Grayscale image as flat array (row-major, length = width * height)
    ///
    /// # Returns
    ///
    /// Vector of DVS events (typically 5-15% of pixels for MNIST)
    ///
    /// # Panics
    ///
    /// Panics if frame.len() != width * height
    pub fn process_frame(&mut self, frame: &[u8]) -> Vec<DvsEvent> {
        assert_eq!(
            frame.len(),
            self.width * self.height,
            "Frame size mismatch: expected {}x{} = {}, got {}",
            self.width,
            self.height,
            self.width * self.height,
            frame.len()
        );

        let mut events = Vec::new();

        for (idx, (&curr, &prev)) in frame.iter().zip(&self.prev_frame).enumerate() {
            let delta = (curr as i16) - (prev as i16);

            if delta.abs() > self.threshold as i16 {
                let x = idx % self.width;
                let y = idx / self.width;

                let (polarity, magnitude) = if delta > 0 {
                    // Brightness increase → ON event
                    (Polarity::On, delta as u8)
                } else {
                    // Brightness decrease → OFF event
                    (Polarity::Off, (-delta) as u8)
                };

                events.push(DvsEvent {
                    x,
                    y,
                    polarity,
                    magnitude,
                });
            }
        }

        // Update previous frame for next comparison
        self.prev_frame.copy_from_slice(frame);
        events
    }

    /// Convert DVS events to rate-coded spike inputs for SNN
    ///
    /// Layout: [pixel_0_ON, pixel_0_OFF, pixel_1_ON, pixel_1_OFF, ...]
    ///
    /// For a 28×28 image, this produces 1568 inputs (784 pixels × 2).
    ///
    /// # Arguments
    ///
    /// * `events` - DVS events from process_frame()
    ///
    /// # Returns
    ///
    /// Rate-coded input vector of length width * height * 2, where each value
    /// is in [0, 255] representing spike probability per tick
    pub fn encode_to_rates(&self, events: &[DvsEvent]) -> Vec<u8> {
        let n_neurons = self.width * self.height * 2; // ON + OFF per pixel
        let mut rates = vec![0u8; n_neurons];

        for event in events {
            let pixel_idx = event.y * self.width + event.x;
            let neuron_idx = pixel_idx * 2
                + match event.polarity {
                    Polarity::On => 0,
                    Polarity::Off => 1,
                };

            // Rate is proportional to magnitude (already in [0, 255])
            rates[neuron_idx] = event.magnitude;
        }

        rates
    }

    /// Reset previous frame to mid-gray baseline
    ///
    /// Call this between image sequences to clear temporal state.
    pub fn reset(&mut self) {
        self.prev_frame.fill(128);
    }

    /// Set baseline frame without generating events
    ///
    /// Useful for initializing the retina to a specific starting state.
    ///
    /// # Panics
    ///
    /// Panics if frame.len() != width * height
    pub fn set_baseline(&mut self, frame: &[u8]) {
        assert_eq!(frame.len(), self.width * self.height);
        self.prev_frame.copy_from_slice(frame);
    }

    /// Get frame dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Get threshold value
    pub fn threshold(&self) -> u8 {
        self.threshold
    }

    /// Set new threshold value
    pub fn set_threshold(&mut self, threshold: u8) {
        self.threshold = threshold;
    }

    /// Calculate sparsity of events as percentage of total pixels
    ///
    /// # Returns
    ///
    /// Sparsity in [0.0, 1.0] (0.1 = 10% of pixels had events)
    pub fn calculate_sparsity(&self, events: &[DvsEvent]) -> f32 {
        let total_pixels = self.width * self.height;
        events.len() as f32 / total_pixels as f32
    }

    /// Get statistics about event distribution
    pub fn event_statistics(&self, events: &[DvsEvent]) -> EventStatistics {
        let total = events.len();
        if total == 0 {
            return EventStatistics {
                total_events: 0,
                on_events: 0,
                off_events: 0,
                avg_magnitude: 0.0,
                sparsity: 0.0,
            };
        }

        let on_count = events.iter().filter(|e| e.polarity == Polarity::On).count();
        let off_count = total - on_count;

        let total_magnitude: u32 = events.iter().map(|e| e.magnitude as u32).sum();
        let avg_magnitude = total_magnitude as f32 / total as f32;

        EventStatistics {
            total_events: total,
            on_events: on_count,
            off_events: off_count,
            avg_magnitude,
            sparsity: self.calculate_sparsity(events),
        }
    }
}

/// Statistics about DVS events
#[derive(Debug, Clone, PartialEq)]
pub struct EventStatistics {
    /// Total number of events
    pub total_events: usize,
    /// Number of ON (brightening) events
    pub on_events: usize,
    /// Number of OFF (darkening) events
    pub off_events: usize,
    /// Average event magnitude
    pub avg_magnitude: f32,
    /// Fraction of pixels with events [0.0, 1.0]
    pub sparsity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_retina() {
        let retina = DvsRetina::new(28, 28, 20);
        assert_eq!(retina.dimensions(), (28, 28));
        assert_eq!(retina.threshold(), 20);
        assert_eq!(retina.prev_frame.len(), 28 * 28);
        assert!(retina.prev_frame.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_uniform_frame_no_events() {
        let mut retina = DvsRetina::new(10, 10, 20);
        let frame = vec![128u8; 100]; // Same as baseline

        let events = retina.process_frame(&frame);
        assert_eq!(events.len(), 0, "Uniform frame should generate no events");
    }

    #[test]
    fn test_full_contrast_generates_events() {
        let mut retina = DvsRetina::new(10, 10, 20);

        // Black frame after gray baseline → all OFF events
        let black_frame = vec![0u8; 100];
        let events = retina.process_frame(&black_frame);

        assert_eq!(events.len(), 100, "All pixels should generate events");
        assert!(
            events.iter().all(|e| e.polarity == Polarity::Off),
            "All events should be OFF (darkening)"
        );

        // White frame after black → all ON events
        retina.reset();
        let white_frame = vec![255u8; 100];
        let events2 = retina.process_frame(&white_frame);

        assert_eq!(events2.len(), 100);
        assert!(events2.iter().all(|e| e.polarity == Polarity::On));
    }

    #[test]
    fn test_threshold_filtering() {
        let mut retina = DvsRetina::new(5, 5, 50);

        // Small change (delta=30) below threshold
        let frame = vec![128u8 + 30; 25];
        let events = retina.process_frame(&frame);
        assert_eq!(
            events.len(),
            0,
            "Changes below threshold should not generate events"
        );

        // Large change (delta=100) above threshold
        retina.reset();
        let frame2 = vec![128u8 + 100; 25];
        let events2 = retina.process_frame(&frame2);
        assert_eq!(events2.len(), 25, "All pixels should exceed threshold");
    }

    #[test]
    fn test_event_coordinates() {
        let mut retina = DvsRetina::new(3, 3, 20);

        // Only center pixel changes
        let mut frame = vec![128u8; 9];
        frame[4] = 200; // Center pixel (x=1, y=1)

        let events = retina.process_frame(&frame);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].x, 1);
        assert_eq!(events[0].y, 1);
        assert_eq!(events[0].polarity, Polarity::On);
    }

    #[test]
    fn test_magnitude_proportional() {
        let mut retina = DvsRetina::new(2, 2, 10);

        let mut frame = vec![128u8; 4];
        frame[0] = 128 + 50; // delta = 50
        frame[1] = 128 + 100; // delta = 100

        let events = retina.process_frame(&frame);
        assert_eq!(events.len(), 2);

        let event_50 = events.iter().find(|e| e.x == 0).unwrap();
        let event_100 = events.iter().find(|e| e.x == 1).unwrap();

        assert_eq!(event_50.magnitude, 50);
        assert_eq!(event_100.magnitude, 100);
    }

    #[test]
    fn test_encode_to_rates() {
        let retina = DvsRetina::new(2, 2, 20);

        let events = vec![
            DvsEvent {
                x: 0,
                y: 0,
                polarity: Polarity::On,
                magnitude: 100,
            },
            DvsEvent {
                x: 1,
                y: 1,
                polarity: Polarity::Off,
                magnitude: 150,
            },
        ];

        let rates = retina.encode_to_rates(&events);

        assert_eq!(rates.len(), 2 * 2 * 2); // 4 pixels × 2 (ON/OFF)
        assert_eq!(rates[0], 100); // Pixel (0,0) ON
        assert_eq!(rates[1], 0); // Pixel (0,0) OFF
        assert_eq!(rates[6], 0); // Pixel (1,1) ON
        assert_eq!(rates[7], 150); // Pixel (1,1) OFF
    }

    #[test]
    fn test_reset() {
        let mut retina = DvsRetina::new(5, 5, 20);

        // Change baseline
        let frame = vec![200u8; 25];
        let _ = retina.process_frame(&frame);
        assert_eq!(retina.prev_frame[0], 200);

        // Reset to mid-gray
        retina.reset();
        assert!(retina.prev_frame.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_sparsity_calculation() {
        let retina = DvsRetina::new(10, 10, 20);

        let events = vec![
            DvsEvent {
                x: 0,
                y: 0,
                polarity: Polarity::On,
                magnitude: 100,
            },
            DvsEvent {
                x: 1,
                y: 1,
                polarity: Polarity::Off,
                magnitude: 100,
            },
        ];

        let sparsity = retina.calculate_sparsity(&events);
        assert!((sparsity - 0.02).abs() < 0.001); // 2/100 = 0.02
    }

    #[test]
    fn test_event_statistics() {
        let retina = DvsRetina::new(10, 10, 20);

        let events = vec![
            DvsEvent {
                x: 0,
                y: 0,
                polarity: Polarity::On,
                magnitude: 100,
            },
            DvsEvent {
                x: 1,
                y: 0,
                polarity: Polarity::On,
                magnitude: 50,
            },
            DvsEvent {
                x: 2,
                y: 0,
                polarity: Polarity::Off,
                magnitude: 200,
            },
        ];

        let stats = retina.event_statistics(&events);
        assert_eq!(stats.total_events, 3);
        assert_eq!(stats.on_events, 2);
        assert_eq!(stats.off_events, 1);
        assert!((stats.avg_magnitude - 116.666).abs() < 0.01); // (100+50+200)/3
        assert!((stats.sparsity - 0.03).abs() < 0.001);
    }

    #[test]
    #[should_panic(expected = "Frame size mismatch")]
    fn test_wrong_frame_size_panics() {
        let mut retina = DvsRetina::new(10, 10, 20);
        let wrong_size = vec![128u8; 50]; // Should be 100
        let _ = retina.process_frame(&wrong_size);
    }

    #[test]
    fn test_set_baseline() {
        let mut retina = DvsRetina::new(3, 3, 20);
        let custom_baseline = vec![200u8; 9];

        retina.set_baseline(&custom_baseline);
        assert_eq!(retina.prev_frame[0], 200);

        // Process same frame → no events
        let events = retina.process_frame(&custom_baseline);
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_typical_mnist_sparsity() {
        // Simulate typical MNIST: static digit on uniform background
        // Sparse events occur only at edges during frame transitions
        let mut retina = DvsRetina::new(28, 28, 20);

        // First frame: uniform gray background
        let frame1 = vec![128u8; 28 * 28];
        retina.set_baseline(&frame1);

        // Second frame: vertical bar (10 pixels wide) on same gray background
        let mut frame2 = vec![128u8; 28 * 28];
        for y in 0..28 {
            for x in 9..19 {
                frame2[y * 28 + x] = 200; // Brightened region (delta = 72)
            }
        }

        let events = retina.process_frame(&frame2);
        let sparsity = retina.calculate_sparsity(&events);

        // Expect sparse events: only the bar region changed (280 pixels out of 784 = 35.7%)
        // This matches biological DVS behavior: events only where change occurred
        assert!(
            sparsity > 0.25,
            "MNIST-like pattern should have >25% sparsity (edges), got {}",
            sparsity
        );
        assert!(
            sparsity < 0.50,
            "Sparsity should be <50% (only changed region), got {}",
            sparsity
        );
    }
}
