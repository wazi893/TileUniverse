#[derive(Clone, Debug)]
pub(crate) struct FieldGrid<T: Copy> {
    width: usize,
    height: usize,
    data: Vec<T>, // row-major contiguous storage
}

impl<T: Copy> FieldGrid<T> {
    pub(crate) fn new(width: usize, height: usize, default: T) -> Self {
        Self {
            width,
            height,
            data: vec![default; width.saturating_mul(height)],
        }
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn height(&self) -> usize {
        self.height
    }

    pub(crate) fn index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width && y < self.height, "index oob");
        y * self.width + x
    }

    pub(crate) fn get(&self, x: usize, y: usize) -> T {
        let idx = self.index(x, y);
        self.data[idx]
    }

    pub(crate) fn set(&mut self, x: usize, y: usize, value: T) {
        let idx = self.index(x, y);
        self.data[idx] = value;
    }

    pub(crate) fn clear(&mut self, value: T) {
        self.data.fill(value);
    }

    #[allow(dead_code)]
    pub(crate) fn resize(&mut self, width: usize, height: usize, default: T) {
        if width == self.width && height == self.height {
            self.clear(default);
            return;
        }

        let mut new_data = vec![default; width.saturating_mul(height)];
        let copy_width = width.min(self.width);
        let copy_height = height.min(self.height);
        for y in 0..copy_height {
            let src_start = y * self.width;
            let dst_start = y * width;
            let src_end = src_start + copy_width;
            let dst_end = dst_start + copy_width;
            new_data[dst_start..dst_end].copy_from_slice(&self.data[src_start..src_end]);
        }

        self.width = width;
        self.height = height;
        self.data = new_data;
    }
}

#[cfg(test)]
mod tests {
    use super::FieldGrid;

    #[test]
    fn new_grid_has_size_and_default_values() {
        let grid = FieldGrid::<u8>::new(4, 3, 7);
        assert_eq!(grid.width(), 4);
        assert_eq!(grid.height(), 3);
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(grid.get(x, y), 7);
            }
        }
    }

    #[test]
    fn set_and_get_round_trip() {
        let mut grid = FieldGrid::<u32>::new(2, 2, 0);
        grid.set(0, 0, 1);
        grid.set(1, 1, 2);
        assert_eq!(grid.get(0, 0), 1);
        assert_eq!(grid.get(1, 1), 2);
        assert_eq!(grid.get(1, 0), 0);
    }

    #[test]
    fn clear_resets_all_values() {
        let mut grid = FieldGrid::<u8>::new(3, 2, 0);
        grid.set(0, 0, 5);
        grid.set(2, 1, 9);
        grid.clear(1);
        for y in 0..2 {
            for x in 0..3 {
                assert_eq!(grid.get(x, y), 1);
            }
        }
    }

    #[test]
    fn resize_preserves_overlap_and_sets_new_cells() {
        let mut grid = FieldGrid::<u8>::new(2, 2, 0);
        grid.set(0, 0, 1);
        grid.set(1, 1, 2);
        grid.resize(3, 3, 9);
        assert_eq!(grid.width(), 3);
        assert_eq!(grid.height(), 3);
        assert_eq!(grid.get(0, 0), 1);
        assert_eq!(grid.get(1, 1), 2);
        assert_eq!(grid.get(2, 2), 9);
    }
}
