use std::cell::Cell;

/// Sprint 274B: Non-atomic dirty bitset for single-threaded simulation.
/// Uses Cell<u64> for interior mutability without atomic RMW overhead.
/// Hierarchical L1 summary enables O(active_segments) draining.
pub struct DirtyBitset {
    pub segments: Vec<Cell<u64>>,
    /// L1 summary: 1 bit per L0 segment. If L1 bit is clear, L0 word is guaranteed clean.
    summary_l1: Vec<Cell<u64>>,
}

impl DirtyBitset {
    pub fn new(tile_count: usize) -> Self {
        let words = (tile_count + 63) / 64;
        let segments: Vec<Cell<u64>> = (0..words).map(|_| Cell::new(0)).collect();
        let l1_words = (words + 63) / 64;
        let summary_l1: Vec<Cell<u64>> = (0..l1_words).map(|_| Cell::new(0)).collect();
        Self {
            segments,
            summary_l1,
        }
    }

    #[inline]
    pub fn mark_dirty(&self, idx: usize) {
        let seg = idx / 64;
        let bit = idx % 64;
        if let Some(word) = self.segments.get(seg) {
            word.set(word.get() | (1u64 << bit));
            let l1_seg = seg / 64;
            let l1_bit = seg % 64;
            if let Some(l1_word) = self.summary_l1.get(l1_seg) {
                l1_word.set(l1_word.get() | (1u64 << l1_bit));
            }
        }
    }

    pub fn take_dirty_batch(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.fill_into_internal(&mut out);
        out.into_iter().map(|x| x as usize).collect()
    }

    pub fn mark_all_dirty(&self, count: usize) {
        let full_words = count / 64;
        for i in 0..full_words {
            self.segments[i].set(u64::MAX);
        }
        let remainder = count % 64;
        if remainder > 0 && full_words < self.segments.len() {
            let mask = (1u64 << remainder) - 1;
            self.segments[full_words].set(mask);
        }
        let used_segs = (count + 63) / 64;
        let full_l1 = used_segs / 64;
        for i in 0..full_l1 {
            self.summary_l1[i].set(u64::MAX);
        }
        let l1_remainder = used_segs % 64;
        if l1_remainder > 0 && full_l1 < self.summary_l1.len() {
            let mask = (1u64 << l1_remainder) - 1;
            self.summary_l1[full_l1].set(mask);
        }
    }

    pub fn fill_into(&self, out: &mut Vec<u32>) {
        self.fill_into_internal(out);
    }

    fn fill_into_internal(&self, out: &mut Vec<u32>) {
        out.clear();
        for (l1_idx, l1_cell) in self.summary_l1.iter().enumerate() {
            let mut l1_word = l1_cell.get();
            if l1_word == 0 {
                continue;
            }
            l1_cell.set(0);
            while l1_word != 0 {
                let l0_offset = l1_word.trailing_zeros() as usize;
                l1_word &= l1_word - 1;
                let seg_idx = l1_idx * 64 + l0_offset;
                if seg_idx >= self.segments.len() {
                    break;
                }
                let w = self.segments[seg_idx].get();
                if w == 0 {
                    continue;
                }
                self.segments[seg_idx].set(0);
                let base = (seg_idx * 64) as u32;
                let mut bits = w;
                while bits != 0 {
                    let tz = bits.trailing_zeros();
                    out.push(base + tz);
                    bits &= bits - 1;
                }
            }
        }
    }

    pub fn fill_into_scoped(&self, scope: &[u32], out: &mut Vec<u32>) {
        out.clear();
        for &idx in scope {
            let seg = idx as usize / 64;
            let bit = idx as usize % 64;
            let mask = 1u64 << bit;
            if let Some(word) = self.segments.get(seg) {
                let old = word.get();
                if (old & mask) != 0 {
                    word.set(old & !mask);
                    out.push(idx);
                }
            }
        }
    }

    #[inline]
    pub fn is_dirty_and_clear(&self, idx: usize) -> bool {
        let seg = idx / 64;
        let bit = idx % 64;
        let mask = 1u64 << bit;
        if let Some(word) = self.segments.get(seg) {
            let old = word.get();
            if (old & mask) != 0 {
                word.set(old & !mask);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn fill_into_masked(&self, scope_mask: &[u64], out: &mut Vec<u32>) {
        out.clear();
        for (l1_idx, l1_cell) in self.summary_l1.iter().enumerate() {
            let l1_word = l1_cell.get();
            if l1_word == 0 {
                continue;
            }
            // Sprint 277: Track which L1 bits still have dirty L0 segments
            // so we can clear stale summary bits at the end.
            let mut l1_remaining = l1_word;
            let mut bits = l1_word;
            while bits != 0 {
                let l0_offset = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let seg_idx = l1_idx * 64 + l0_offset;
                if seg_idx >= self.segments.len() {
                    break;
                }
                let scope = scope_mask.get(seg_idx).copied().unwrap_or(0);
                if scope == 0 {
                    continue;
                }
                let dirty = self.segments[seg_idx].get();
                let dirty_in_scope = dirty & scope;
                if dirty_in_scope == 0 {
                    continue;
                }
                let remaining = dirty & !dirty_in_scope;
                self.segments[seg_idx].set(remaining);
                // If the L0 segment is now empty, clear its L1 summary bit.
                if remaining == 0 {
                    l1_remaining &= !(1u64 << l0_offset);
                }
                let base = (seg_idx * 64) as u32;
                let mut m = dirty_in_scope;
                while m != 0 {
                    let tz = m.trailing_zeros();
                    out.push(base + tz);
                    m &= m - 1;
                }
            }
            // Sprint 277: Repair L1 summary — clear bits for emptied segments.
            if l1_remaining != l1_word {
                l1_cell.set(l1_remaining);
            }
        }
    }

    /// Sprint 333: Sparse segment drain — only checks segments in the precomputed
    /// in-scope list. Each entry is (segment_index, scope_mask_word). O(in_scope_segments)
    /// instead of O(L1_bits × L0_segments).
    pub fn fill_into_sparse(&self, in_scope_segments: &[(u32, u64)], out: &mut Vec<u32>) {
        out.clear();
        for &(seg_idx_u32, scope) in in_scope_segments {
            let seg_idx = seg_idx_u32 as usize;
            if seg_idx >= self.segments.len() {
                continue;
            }
            // Check L1 summary first — skip if entire L1 group is clean.
            let l1_idx = seg_idx / 64;
            let l1_bit = 1u64 << (seg_idx % 64);
            if let Some(l1_cell) = self.summary_l1.get(l1_idx) {
                if l1_cell.get() & l1_bit == 0 {
                    continue;
                }
            } else {
                continue;
            }

            let dirty = self.segments[seg_idx].get();
            let dirty_in_scope = dirty & scope;
            if dirty_in_scope == 0 {
                continue;
            }
            let remaining = dirty & !dirty_in_scope;
            self.segments[seg_idx].set(remaining);
            // Repair L1 if segment emptied.
            if remaining == 0 {
                let l1_idx = seg_idx / 64;
                if let Some(l1_cell) = self.summary_l1.get(l1_idx) {
                    l1_cell.set(l1_cell.get() & !l1_bit);
                }
            }
            let base = (seg_idx * 64) as u32;
            let mut m = dirty_in_scope;
            while m != 0 {
                let tz = m.trailing_zeros();
                out.push(base + tz);
                m &= m - 1;
            }
        }
    }

    /// Sprint 333: Check if any dirty bits exist in the sparse segment list.
    /// Returns true if at least one in-scope segment has dirty bits.
    pub fn has_dirty_in_sparse(&self, in_scope_segments: &[(u32, u64)]) -> bool {
        for &(seg_idx_u32, scope) in in_scope_segments {
            let seg_idx = seg_idx_u32 as usize;
            if seg_idx >= self.segments.len() {
                continue;
            }
            let l1_idx = seg_idx / 64;
            let l1_bit = 1u64 << (seg_idx % 64);
            if let Some(l1_cell) = self.summary_l1.get(l1_idx) {
                if l1_cell.get() & l1_bit == 0 {
                    continue;
                }
            } else {
                continue;
            }
            if self.segments[seg_idx].get() & scope != 0 {
                return true;
            }
        }
        false
    }
}
