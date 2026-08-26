//! Grid Allocator for Optimal CPU Placement
//!
//! Handles placement of multiple TILE-8 CPUs in a simulation grid with
//! optimal spacing to avoid tile overlap.
//!
//! # CPU Dimensions
//!
//! Each PhysicalCpu occupies:
//! - Width: 16 tiles
//! - Height: 30 tiles
//!
//! # Spacing Strategy
//!
//! To avoid overlap, we use:
//! - Horizontal spacing: 26 tiles (16 CPU width + 10 gap)
//! - Vertical spacing: 32 tiles (30 CPU height + 2 gap)
//!
//! # Example
//!
//! ```ignore
//! use engine::tile8::grid_allocator::GridAllocator;
//!
//! let allocator = GridAllocator::new(512, 512);
//! let placements = allocator.allocate_square(100)?;  // 10×10 grid
//!
//! for (cpu_id, origin) in placements {
//!     grid.place_cpu(cpu_id, origin, &mut sim);
//! }
//! ```

use std::fmt;

/// Error types for grid allocation
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationError {
    /// Grid is too small to fit requested CPUs
    GridTooSmall {
        required_width: usize,
        required_height: usize,
        available_width: usize,
        available_height: usize,
    },

    /// Invalid number of CPUs requested
    InvalidCpuCount { requested: usize, message: String },
}

impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocationError::GridTooSmall {
                required_width,
                required_height,
                available_width,
                available_height,
            } => write!(
                f,
                "Grid too small: need {}×{} but only have {}×{}",
                required_width, required_height, available_width, available_height
            ),
            AllocationError::InvalidCpuCount { requested, message } => {
                write!(f, "Invalid CPU count {}: {}", requested, message)
            }
        }
    }
}

impl std::error::Error for AllocationError {}

/// CPU placement result: (cpu_id, (origin_x, origin_y))
pub type CpuPlacement = (usize, (usize, usize));

/// Grid allocator for optimal CPU placement
pub struct GridAllocator {
    /// Grid width in tiles
    grid_width: usize,

    /// Grid height in tiles
    grid_height: usize,

    /// CPU width in tiles (default: 16)
    cpu_width: usize,

    /// CPU height in tiles (default: 30)
    cpu_height: usize,

    /// Horizontal spacing between CPUs in tiles (default: 26)
    /// This includes CPU width + gap
    h_spacing: usize,

    /// Vertical spacing between CPUs in tiles (default: 32)
    /// This includes CPU height + gap
    v_spacing: usize,
}

impl GridAllocator {
    /// Create a new grid allocator
    ///
    /// # Arguments
    /// * `grid_width` - Total grid width in tiles
    /// * `grid_height` - Total grid height in tiles
    ///
    /// # Example
    /// ```ignore
    /// let allocator = GridAllocator::new(512, 512);
    /// ```
    pub fn new(grid_width: usize, grid_height: usize) -> Self {
        Self {
            grid_width,
            grid_height,
            cpu_width: 16,
            cpu_height: 30,
            h_spacing: 26, // 16 + 10 gap
            v_spacing: 32, // 30 + 2 gap
        }
    }

    /// Create allocator with custom CPU dimensions and spacing
    ///
    /// # Arguments
    /// * `grid_width` - Total grid width
    /// * `grid_height` - Total grid height
    /// * `cpu_width` - CPU width in tiles
    /// * `cpu_height` - CPU height in tiles
    /// * `h_spacing` - Horizontal spacing (CPU width + gap)
    /// * `v_spacing` - Vertical spacing (CPU height + gap)
    pub fn with_spacing(
        grid_width: usize,
        grid_height: usize,
        cpu_width: usize,
        cpu_height: usize,
        h_spacing: usize,
        v_spacing: usize,
    ) -> Self {
        Self {
            grid_width,
            grid_height,
            cpu_width,
            cpu_height,
            h_spacing,
            v_spacing,
        }
    }

    /// Allocate CPUs in a square grid (N = rows × rows)
    ///
    /// # Arguments
    /// * `num_cpus` - Total number of CPUs (must be perfect square)
    ///
    /// # Returns
    /// Vector of (cpu_id, (x, y)) placements
    ///
    /// # Errors
    /// - `InvalidCpuCount` if num_cpus is not a perfect square
    /// - `GridTooSmall` if grid cannot fit the requested layout
    ///
    /// # Example
    /// ```ignore
    /// // 100 CPUs in 10×10 grid
    /// let placements = allocator.allocate_square(100)?;
    /// ```
    pub fn allocate_square(&self, num_cpus: usize) -> Result<Vec<CpuPlacement>, AllocationError> {
        // Check if num_cpus is a perfect square
        let side = (num_cpus as f64).sqrt() as usize;
        if side * side != num_cpus {
            return Err(AllocationError::InvalidCpuCount {
                requested: num_cpus,
                message: format!("{} is not a perfect square", num_cpus),
            });
        }

        self.allocate_grid(side, side)
    }

    /// Allocate CPUs in a rectangular grid
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Returns
    /// Vector of (cpu_id, (x, y)) placements
    ///
    /// # Example
    /// ```ignore
    /// // 100 CPUs in 10×10 grid
    /// let placements = allocator.allocate_grid(10, 10)?;
    ///
    /// // 64 CPUs in 8×8 grid
    /// let placements = allocator.allocate_grid(8, 8)?;
    /// ```
    pub fn allocate_grid(
        &self,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<CpuPlacement>, AllocationError> {
        // Calculate required grid size
        let required_width = cols * self.h_spacing;
        let required_height = rows * self.v_spacing;

        // Check if grid is large enough
        if required_width > self.grid_width || required_height > self.grid_height {
            return Err(AllocationError::GridTooSmall {
                required_width,
                required_height,
                available_width: self.grid_width,
                available_height: self.grid_height,
            });
        }

        // Calculate starting offset to center the CPU grid
        let x_offset = (self.grid_width - required_width) / 2;
        let y_offset = (self.grid_height - required_height) / 2;

        // Generate placements
        let mut placements = Vec::with_capacity(rows * cols);
        let mut cpu_id = 0;

        for row in 0..rows {
            for col in 0..cols {
                let x = x_offset + col * self.h_spacing;
                let y = y_offset + row * self.v_spacing;
                placements.push((cpu_id, (x, y)));
                cpu_id += 1;
            }
        }

        Ok(placements)
    }

    /// Allocate CPUs for a specific qubit count
    ///
    /// Calculates the number of CPUs needed for N qubits:
    /// - Each CPU handles 128 complex amplitudes (7 qubits)
    /// - For N qubits: need 2^(N-7) CPUs
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    ///
    /// # Returns
    /// Vector of CPU placements for the given qubit count
    ///
    /// # Example
    /// ```ignore
    /// // 15 qubits = 2^(15-7) = 256 CPUs
    /// let placements = allocator.allocate_for_qubits(15)?;
    /// ```
    pub fn allocate_for_qubits(
        &self,
        num_qubits: usize,
    ) -> Result<Vec<CpuPlacement>, AllocationError> {
        if num_qubits < 7 {
            return Err(AllocationError::InvalidCpuCount {
                requested: 0,
                message: format!("Need at least 7 qubits, got {}", num_qubits),
            });
        }

        let num_cpus: usize = 1 << (num_qubits - 7); // 2^(N-7)

        // Try to make it as square as possible
        let side = (num_cpus as f64).sqrt().ceil() as usize;
        let rows = side;
        let cols = num_cpus.div_ceil(rows); // Ceiling division

        self.allocate_grid(rows, cols)
    }

    /// Get maximum number of CPUs that can fit in the grid
    ///
    /// # Returns
    /// Maximum CPUs that fit in a square layout
    pub fn max_cpus(&self) -> usize {
        let max_cols = self.grid_width / self.h_spacing;
        let max_rows = self.grid_height / self.v_spacing;
        max_cols * max_rows
    }

    /// Get maximum qubits supported by this grid
    ///
    /// # Returns
    /// Maximum number of qubits based on grid capacity
    pub fn max_qubits(&self) -> usize {
        let max_cpus = self.max_cpus();
        if max_cpus == 0 {
            return 0;
        }
        // max_cpus = 2^(N-7), so N = log2(max_cpus) + 7
        7 + (max_cpus as f64).log2() as usize
    }

    /// Validate a placement doesn't overlap with others
    ///
    /// # Arguments
    /// * `placements` - List of CPU placements to validate
    ///
    /// # Returns
    /// true if all placements are valid (no overlaps)
    pub fn validate_placements(&self, placements: &[CpuPlacement]) -> bool {
        for (i, (_, (x1, y1))) in placements.iter().enumerate() {
            for (j, (_, (x2, y2))) in placements.iter().enumerate() {
                if i >= j {
                    continue; // Skip self and already-checked pairs
                }

                // Check for overlap
                let x_overlap = (x1 + self.cpu_width > *x2) && (x2 + self.cpu_width > *x1);
                let y_overlap = (y1 + self.cpu_height > *y2) && (y2 + self.cpu_height > *y1);

                if x_overlap && y_overlap {
                    return false; // Overlap detected
                }
            }
        }

        true // No overlaps
    }

    /// Get grid utilization percentage
    ///
    /// # Arguments
    /// * `num_cpus` - Number of CPUs placed
    ///
    /// # Returns
    /// Percentage of grid area used by CPUs (0.0 to 100.0)
    pub fn utilization(&self, num_cpus: usize) -> f64 {
        let cpu_area = num_cpus * self.cpu_width * self.cpu_height;
        let grid_area = self.grid_width * self.grid_height;

        if grid_area == 0 {
            return 0.0;
        }

        (cpu_area as f64 / grid_area as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_creation() {
        let allocator = GridAllocator::new(512, 512);
        assert_eq!(allocator.grid_width, 512);
        assert_eq!(allocator.grid_height, 512);
        assert_eq!(allocator.cpu_width, 16);
        assert_eq!(allocator.cpu_height, 30);
    }

    #[test]
    fn test_allocate_4_cpus() {
        let allocator = GridAllocator::new(512, 512);
        let placements = allocator.allocate_square(4).unwrap();

        assert_eq!(placements.len(), 4);

        // Should be 2×2 grid
        // Check CPU IDs are sequential
        assert_eq!(placements[0].0, 0);
        assert_eq!(placements[1].0, 1);
        assert_eq!(placements[2].0, 2);
        assert_eq!(placements[3].0, 3);

        // Validate no overlaps
        assert!(allocator.validate_placements(&placements));
    }

    #[test]
    fn test_allocate_100_cpus() {
        let allocator = GridAllocator::new(512, 512);
        let placements = allocator.allocate_square(100).unwrap();

        assert_eq!(placements.len(), 100);

        // Should be 10×10 grid
        // Required space: 10 × 26 = 260 width, 10 × 32 = 320 height
        // Both fit in 512×512

        // Validate no overlaps
        assert!(allocator.validate_placements(&placements));

        // Check utilization
        let util = allocator.utilization(100);
        assert!(util > 10.0 && util < 50.0); // Reasonable utilization
    }

    #[test]
    fn test_allocate_grid_rectangular() {
        let allocator = GridAllocator::new(512, 512);

        // 8 rows × 12 cols = 96 CPUs
        let placements = allocator.allocate_grid(8, 12).unwrap();
        assert_eq!(placements.len(), 96);
        assert!(allocator.validate_placements(&placements));
    }

    #[test]
    fn test_allocate_for_qubits() {
        let allocator = GridAllocator::new(512, 512);

        // 9 qubits = 2^(9-7) = 4 CPUs (perfect square: 2×2)
        let placements = allocator.allocate_for_qubits(9).unwrap();
        assert_eq!(placements.len(), 4);

        // 10 qubits = 2^(10-7) = 8 CPUs (not square: 3×3 = 9 CPUs allocated)
        let placements = allocator.allocate_for_qubits(10).unwrap();
        assert!(
            placements.len() >= 8,
            "Expected at least 8 CPUs, got {}",
            placements.len()
        );

        // 11 qubits = 2^(11-7) = 16 CPUs (perfect square: 4×4)
        let placements = allocator.allocate_for_qubits(11).unwrap();
        assert_eq!(placements.len(), 16);
    }

    #[test]
    fn test_max_cpus() {
        let allocator = GridAllocator::new(512, 512);

        // Grid: 512×512
        // H spacing: 26, V spacing: 32
        // Max cols: 512 / 26 = 19
        // Max rows: 512 / 32 = 16
        // Max CPUs: 19 × 16 = 304

        let max_cpus = allocator.max_cpus();
        assert_eq!(max_cpus, 304);
    }

    #[test]
    fn test_max_qubits() {
        let allocator = GridAllocator::new(512, 512);

        let max_qubits = allocator.max_qubits();
        // max_cpus = 304
        // max_qubits = 7 + log2(304) ≈ 7 + 8 = 15
        assert!((14..=16).contains(&max_qubits));
    }

    #[test]
    fn test_grid_too_small() {
        // Very small grid
        let allocator = GridAllocator::new(50, 50);

        // Try to allocate 100 CPUs (needs 260×320)
        let result = allocator.allocate_square(100);
        assert!(result.is_err());

        match result {
            Err(AllocationError::GridTooSmall { .. }) => {
                // Expected error
            }
            _ => panic!("Expected GridTooSmall error"),
        }
    }

    #[test]
    fn test_not_perfect_square() {
        let allocator = GridAllocator::new(512, 512);

        // 99 is not a perfect square
        let result = allocator.allocate_square(99);
        assert!(result.is_err());

        match result {
            Err(AllocationError::InvalidCpuCount { requested, .. }) => {
                assert_eq!(requested, 99);
            }
            _ => panic!("Expected InvalidCpuCount error"),
        }
    }

    #[test]
    fn test_validate_overlapping_placements() {
        let allocator = GridAllocator::new(512, 512);

        // Create overlapping placements manually
        let bad_placements = vec![
            (0, (0, 0)),
            (1, (10, 10)), // This overlaps with (0, 0)
        ];

        assert!(!allocator.validate_placements(&bad_placements));
    }

    #[test]
    fn test_validate_non_overlapping_placements() {
        let allocator = GridAllocator::new(512, 512);

        // Create non-overlapping placements
        let good_placements = vec![
            (0, (0, 0)),
            (1, (26, 0)), // Properly spaced
            (2, (0, 32)), // Properly spaced vertically
        ];

        assert!(allocator.validate_placements(&good_placements));
    }

    #[test]
    fn test_utilization() {
        let allocator = GridAllocator::new(512, 512);

        // 100 CPUs, each 16×30 = 480 tiles
        // Total: 48,000 tiles
        // Grid: 512×512 = 262,144 tiles
        // Utilization: 48,000 / 262,144 ≈ 18.3%

        let util = allocator.utilization(100);
        assert!(util > 18.0 && util < 19.0);
    }

    #[test]
    fn test_large_grid_12k() {
        // Test with full 12,288×12,288 grid (from Phase 1)
        let allocator = GridAllocator::new(12288, 12288);

        let max_cpus = allocator.max_cpus();
        // 12288 / 26 = 472 cols
        // 12288 / 32 = 384 rows
        // Total: 181,248 CPUs!

        assert!(max_cpus > 180000);

        let max_qubits = allocator.max_qubits();
        // log2(181248) ≈ 17.5
        // max_qubits ≈ 7 + 17 = 24 qubits
        assert!(max_qubits >= 24);

        println!("12k grid can fit {} CPUs ({} qubits)", max_cpus, max_qubits);
    }
}
