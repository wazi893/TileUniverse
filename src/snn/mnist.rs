//! MNIST Dataset Loader
//!
//! Direct IDX format parser for MNIST dataset without external dependencies.
//! Download MNIST from: http://yann.lecun.com/exdb/mnist/
//!
//! # Format
//!
//! - Images: IDX3-UBYTE (magic 0x00000803)
//! - Labels: IDX1-UBYTE (magic 0x00000801)
//!
//! # Example
//!
//! ```ignore
//! use engine::snn::MnistDataset;
//!
//! let dataset = MnistDataset::load(
//!     "data/train-images-idx3-ubyte",
//!     "data/train-labels-idx1-ubyte"
//! ).unwrap();
//!
//! for (image, label) in dataset.batches(32) {
//!     // Train on batch...
//! }
//! ```

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// MNIST dataset with images and labels
#[derive(Debug, Clone)]
pub struct MnistDataset {
    pub images: Vec<Vec<u8>>,
    pub labels: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// Iterator over MNIST batches
pub struct BatchIterator<'a> {
    dataset: &'a MnistDataset,
    batch_size: usize,
    offset: usize,
}

/// Single batch of MNIST images and labels
pub struct Batch<'a> {
    pub images: &'a [Vec<u8>],
    pub labels: &'a [u8],
}

impl MnistDataset {
    /// Load MNIST dataset from IDX files
    ///
    /// # Arguments
    ///
    /// * `images_path` - Path to IDX3-UBYTE images file
    /// * `labels_path` - Path to IDX1-UBYTE labels file
    ///
    /// # Returns
    ///
    /// Result with loaded dataset or IO error
    pub fn load<P: AsRef<Path>>(images_path: P, labels_path: P) -> io::Result<Self> {
        let images = Self::load_images(images_path)?;
        let labels = Self::load_labels(labels_path)?;

        if images.len() != labels.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Image count ({}) != label count ({})",
                    images.len(),
                    labels.len()
                ),
            ));
        }

        let (width, height) = if !images.is_empty() {
            let pixels = images[0].len();
            // MNIST is 28×28 = 784 pixels
            let side = (pixels as f32).sqrt() as usize;
            (side, side)
        } else {
            (0, 0)
        };

        Ok(Self {
            images,
            labels,
            width,
            height,
        })
    }

    /// Load IDX3-UBYTE images file
    fn load_images<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<u8>>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        if buffer.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Images file too short for header",
            ));
        }

        // Parse header
        let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        if magic != 0x0000_0803 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid images magic number: 0x{:08x}", magic),
            ));
        }

        let count = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;
        let rows = u32::from_be_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]) as usize;
        let cols = u32::from_be_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]) as usize;

        let image_size = rows * cols;
        let expected_size = 16 + count * image_size;

        if buffer.len() != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Images file size mismatch: expected {}, got {}",
                    expected_size,
                    buffer.len()
                ),
            ));
        }

        // Extract images
        let mut images = Vec::with_capacity(count);
        for i in 0..count {
            let start = 16 + i * image_size;
            let end = start + image_size;
            images.push(buffer[start..end].to_vec());
        }

        Ok(images)
    }

    /// Load IDX1-UBYTE labels file
    fn load_labels<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        if buffer.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Labels file too short for header",
            ));
        }

        // Parse header
        let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        if magic != 0x0000_0801 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid labels magic number: 0x{:08x}", magic),
            ));
        }

        let count = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;

        if buffer.len() != 8 + count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Labels file size mismatch: expected {}, got {}",
                    8 + count,
                    buffer.len()
                ),
            ));
        }

        Ok(buffer[8..].to_vec())
    }

    /// Split dataset into training and testing sets
    ///
    /// # Arguments
    ///
    /// * `train_size` - Number of samples for training set
    ///
    /// # Returns
    ///
    /// Tuple of (train_dataset, test_dataset)
    pub fn split(self, train_size: usize) -> (Self, Self) {
        let train_images = self.images[..train_size].to_vec();
        let train_labels = self.labels[..train_size].to_vec();

        let test_images = self.images[train_size..].to_vec();
        let test_labels = self.labels[train_size..].to_vec();

        let train = Self {
            images: train_images,
            labels: train_labels,
            width: self.width,
            height: self.height,
        };

        let test = Self {
            images: test_images,
            labels: test_labels,
            width: self.width,
            height: self.height,
        };

        (train, test)
    }

    /// Create batch iterator
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of samples per batch
    ///
    /// # Returns
    ///
    /// Iterator over batches
    pub fn batches(&self, batch_size: usize) -> BatchIterator<'_> {
        BatchIterator {
            dataset: self,
            batch_size,
            offset: 0,
        }
    }

    /// Get single image by index
    pub fn get_image(&self, index: usize) -> Option<&Vec<u8>> {
        self.images.get(index)
    }

    /// Get single label by index
    pub fn get_label(&self, index: usize) -> Option<u8> {
        self.labels.get(index).copied()
    }

    /// Get dataset size
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Calculate label distribution
    pub fn label_distribution(&self) -> [usize; 10] {
        let mut counts = [0usize; 10];
        for &label in &self.labels {
            if (label as usize) < 10 {
                counts[label as usize] += 1;
            }
        }
        counts
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Batch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.dataset.images.len() {
            return None;
        }

        let end = (self.offset + self.batch_size).min(self.dataset.images.len());
        let batch = Batch {
            images: &self.dataset.images[self.offset..end],
            labels: &self.dataset.labels[self.offset..end],
        };

        self.offset = end;
        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split() {
        // Create mock dataset
        let images = vec![vec![0u8; 784]; 100];
        let labels = (0..100).map(|i| (i % 10) as u8).collect();

        let dataset = MnistDataset {
            images,
            labels,
            width: 28,
            height: 28,
        };

        let (train, test) = dataset.split(80);

        assert_eq!(train.len(), 80);
        assert_eq!(test.len(), 20);
        assert_eq!(train.width, 28);
        assert_eq!(test.height, 28);
    }

    #[test]
    fn test_batch_iterator() {
        // Create mock dataset
        let images = vec![vec![0u8; 784]; 100];
        let labels = vec![0u8; 100];

        let dataset = MnistDataset {
            images,
            labels,
            width: 28,
            height: 28,
        };

        let mut batch_count = 0;
        let mut total_samples = 0;

        for batch in dataset.batches(32) {
            batch_count += 1;
            total_samples += batch.images.len();

            // Check batch size
            if batch_count < 4 {
                assert_eq!(batch.images.len(), 32);
            } else {
                // Last batch
                assert_eq!(batch.images.len(), 4);
            }
        }

        assert_eq!(batch_count, 4);
        assert_eq!(total_samples, 100);
    }

    #[test]
    fn test_label_distribution() {
        let images = vec![vec![0u8; 784]; 100];
        let labels = (0..100).map(|i| (i % 10) as u8).collect();

        let dataset = MnistDataset {
            images,
            labels,
            width: 28,
            height: 28,
        };

        let dist = dataset.label_distribution();

        // Should have 10 samples of each digit
        for count in dist.iter() {
            assert_eq!(*count, 10);
        }
    }

    #[test]
    fn test_get_image_label() {
        let images = vec![vec![1u8, 2, 3], vec![4u8, 5, 6]];
        let labels = vec![7u8, 8];

        let dataset = MnistDataset {
            images,
            labels,
            width: 1,
            height: 3,
        };

        assert_eq!(dataset.get_image(0).unwrap(), &vec![1u8, 2, 3]);
        assert_eq!(dataset.get_image(1).unwrap(), &vec![4u8, 5, 6]);
        assert_eq!(dataset.get_image(2), None);

        assert_eq!(dataset.get_label(0).unwrap(), 7);
        assert_eq!(dataset.get_label(1).unwrap(), 8);
        assert_eq!(dataset.get_label(2), None);
    }

    #[test]
    fn test_empty_dataset() {
        let dataset = MnistDataset {
            images: vec![],
            labels: vec![],
            width: 0,
            height: 0,
        };

        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);

        let mut batch_iter = dataset.batches(10);
        assert!(batch_iter.next().is_none());
    }
}
