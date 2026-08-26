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

    /// Sprint 384: Clear all dirty bits inside `scope_mask` WITHOUT extracting
    /// their indices. Same traversal as `fill_into_masked` minus the per-bit
    /// push loop — used to drain informationless residue after the JIT settle
    /// (every in-scope tile was just evaluated unconditionally). Returns the
    /// number of bits cleared.
    pub fn clear_masked(&self, scope_mask: &[u64]) -> u32 {
        let mut cleared = 0u32;
        for (l1_idx, l1_cell) in self.summary_l1.iter().enumerate() {
            let l1_word = l1_cell.get();
            if l1_word == 0 {
                continue;
            }
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
                cleared += dirty_in_scope.count_ones();
                let remaining = dirty & !dirty_in_scope;
                self.segments[seg_idx].set(remaining);
                if remaining == 0 {
                    l1_remaining &= !(1u64 << l0_offset);
                }
            }
            if l1_remaining != l1_word {
                l1_cell.set(l1_remaining);
            }
        }
        cleared
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
    /// Sprint 384: Sparse clear — clears dirty bits in the given (segment, scope)
    /// pairs without extracting indices. The sparse counterpart of `clear_masked`,
    /// visiting only in-scope segments (S333 pattern) instead of every dirty
    /// segment in the grid. Returns the number of bits cleared.
    pub fn clear_sparse(&self, in_scope_segments: &[(u32, u64)]) -> u32 {
        let mut cleared = 0u32;
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
            let dirty = self.segments[seg_idx].get();
            let dirty_in_scope = dirty & scope;
            if dirty_in_scope == 0 {
                continue;
            }
            cleared += dirty_in_scope.count_ones();
            let remaining = dirty & !dirty_in_scope;
            self.segments[seg_idx].set(remaining);
            if remaining == 0 {
                if let Some(l1_cell) = self.summary_l1.get(l1_idx) {
                    l1_cell.set(l1_cell.get() & !l1_bit);
                }
            }
        }
        cleared
    }

    /// Sprint 384: Diagnostic — count (dirty_segments, dirty_bits) across the
    /// whole bitset. Not for hot paths.
    pub fn debug_counts(&self) -> (u32, u32) {
        let mut segs = 0u32;
        let mut bits = 0u32;
        for seg in &self.segments {
            let d = seg.get();
            if d != 0 {
                segs += 1;
                bits += d.count_ones();
            }
        }
        (segs, bits)
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// `take_dirty_batch` drains the whole bitset in globally-ascending index
    /// order (L1 words ascending → L0 segments ascending → bits ascending).
    /// The hierarchical traversal must agree with a flat reference set, or the
    /// sparse evaluator silently drops tiles.
    fn drain_sorted(b: &DirtyBitset) -> Vec<usize> {
        b.take_dirty_batch()
    }

    #[test]
    fn new_is_empty() {
        let b = DirtyBitset::new(500);
        assert!(drain_sorted(&b).is_empty());
        assert_eq!(b.debug_counts(), (0, 0));
        assert!(!b.has_dirty_in_sparse(&[(0, u64::MAX), (7, u64::MAX)]));
    }

    #[test]
    fn mark_and_drain_single_then_clears() {
        let b = DirtyBitset::new(500);
        b.mark_dirty(5);
        assert_eq!(b.debug_counts(), (1, 1));
        assert_eq!(drain_sorted(&b), vec![5]);
        // Draining is consuming: a second drain is empty and the L1 summary is repaired.
        assert!(drain_sorted(&b).is_empty());
        assert_eq!(b.debug_counts(), (0, 0));
    }

    #[test]
    fn drain_is_ascending_across_l1_words_and_clears() {
        // 9000 tiles → 141 L0 segments → 3 L1 words, exercising the second/third
        // L1 word (segment indices ≥ 64 live under L1 word 1+).
        let b = DirtyBitset::new(9000);
        let marked = [0usize, 1, 63, 64, 130, 4095, 4096, 4097, 8191, 8999];
        for &i in &marked {
            b.mark_dirty(i);
        }
        let got = drain_sorted(&b);
        let mut want: Vec<usize> = marked.to_vec();
        want.sort_unstable();
        assert_eq!(got, want, "hierarchical drain must match the flat set");
        // Strictly ascending output.
        assert!(got.windows(2).all(|w| w[0] < w[1]));
        // Fully consumed: no phantom dirty bits left in either level.
        assert!(drain_sorted(&b).is_empty());
        assert_eq!(b.debug_counts(), (0, 0));
    }

    #[test]
    fn mark_dirty_out_of_range_is_ignored() {
        let b = DirtyBitset::new(100); // 2 L0 segments
        b.mark_dirty(5_000); // beyond capacity — must be a silent no-op
        assert_eq!(b.debug_counts(), (0, 0));
        assert!(drain_sorted(&b).is_empty());
    }

    #[test]
    fn mark_all_dirty_marks_exact_range_with_remainder() {
        // 200 is not a multiple of 64, so this exercises the partial-word mask.
        let b = DirtyBitset::new(200);
        b.mark_all_dirty(200);
        assert_eq!(drain_sorted(&b), (0..200).collect::<Vec<_>>());

        // A partial count must mark exactly [0, count) and nothing above it.
        let b2 = DirtyBitset::new(200);
        b2.mark_all_dirty(100);
        assert_eq!(b2.debug_counts().1, 100);
        assert_eq!(drain_sorted(&b2), (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn is_dirty_and_clear_consumes_bit() {
        let b = DirtyBitset::new(200);
        b.mark_dirty(7);
        assert!(b.is_dirty_and_clear(7));
        assert!(!b.is_dirty_and_clear(7)); // already consumed
        assert!(!b.is_dirty_and_clear(8)); // never marked
        // The stale L1 summary bit must not resurrect a phantom on a later drain.
        assert!(drain_sorted(&b).is_empty());
    }

    #[test]
    fn fill_into_scoped_drains_only_scope() {
        let b = DirtyBitset::new(200);
        for &i in &[3u32, 10, 70] {
            b.mark_dirty(i as usize);
        }
        let mut out = Vec::new();
        b.fill_into_scoped(&[3, 70], &mut out);
        assert_eq!(out, vec![3, 70]); // follows scope order, only in-scope bits
        // The unscoped index survives and drains afterwards.
        assert_eq!(drain_sorted(&b), vec![10]);
    }

    #[test]
    fn fill_into_masked_respects_mask_and_repairs_l1() {
        let b = DirtyBitset::new(200); // 4 L0 segments
        for &i in &[1usize, 2, 65] {
            b.mark_dirty(i);
        }
        // Seg 0: only bit 1 in scope; seg 1: full scope; segs 2,3: empty scope.
        let mut mask = vec![0u64; 4];
        mask[0] = 1u64 << 1;
        mask[1] = u64::MAX;
        let mut out = Vec::new();
        b.fill_into_masked(&mask, &mut out);
        assert_eq!(out, vec![1, 65]); // bit 2 of seg 0 was out of scope
        // Seg 1 emptied (L1 repaired); seg 0 keeps bit 2.
        assert_eq!(b.debug_counts(), (1, 1));
        assert_eq!(drain_sorted(&b), vec![2]);
    }

    #[test]
    fn clear_masked_counts_and_drops_without_extract() {
        let b = DirtyBitset::new(200);
        for &i in &[1usize, 2, 65] {
            b.mark_dirty(i);
        }
        // Scope only bits 1,2 of seg 0.
        let mut mask = vec![0u64; 4];
        mask[0] = (1u64 << 1) | (1u64 << 2);
        assert_eq!(b.clear_masked(&mask), 2);
        // 65 survives; the cleared bits are gone, L1 for seg 0 repaired.
        assert_eq!(drain_sorted(&b), vec![65]);

        // Full mask clears everything and reports the full count.
        let b2 = DirtyBitset::new(200);
        for &i in &[1usize, 2, 65] {
            b2.mark_dirty(i);
        }
        assert_eq!(b2.clear_masked(&vec![u64::MAX; 4]), 3);
        assert_eq!(b2.debug_counts(), (0, 0));
    }

    #[test]
    fn sparse_drain_and_clear_repair_l1() {
        // bit 1 → seg 0; bit 130 → seg 2, bit 2.
        let scope = [(0u32, u64::MAX), (2u32, u64::MAX)];

        let b = DirtyBitset::new(300);
        b.mark_dirty(1);
        b.mark_dirty(130);
        let mut out = Vec::new();
        b.fill_into_sparse(&scope, &mut out);
        assert_eq!(out, vec![1, 130]);
        assert_eq!(b.debug_counts(), (0, 0));
        assert!(drain_sorted(&b).is_empty()); // L1 repaired, no phantom

        let b2 = DirtyBitset::new(300);
        b2.mark_dirty(1);
        b2.mark_dirty(130);
        assert_eq!(b2.clear_sparse(&scope), 2);
        assert_eq!(b2.debug_counts(), (0, 0));
    }

    #[test]
    fn has_dirty_in_sparse_honors_segment_and_mask() {
        let b = DirtyBitset::new(300);
        b.mark_dirty(130); // seg 2, bit 2
        assert!(b.has_dirty_in_sparse(&[(2, u64::MAX)]));
        assert!(!b.has_dirty_in_sparse(&[(0, u64::MAX)])); // wrong segment
        assert!(!b.has_dirty_in_sparse(&[(2, !(1u64 << 2))])); // bit masked out
        assert!(!b.has_dirty_in_sparse(&[(9_999, u64::MAX)])); // out-of-range segment
    }

    #[test]
    fn hierarchical_drain_matches_reference_over_wide_spread() {
        // Deterministic (no RNG) spread of indices across many segments and all
        // three L1 words; cross-check the hierarchical drain against a flat set.
        let tile_count = 9000usize;
        let b = DirtyBitset::new(tile_count);
        let mut reference = BTreeSet::new();
        for i in 0..1500u64 {
            // Knuth multiplicative hash → well-distributed but fully reproducible.
            let idx = (i.wrapping_mul(2654435761) % tile_count as u64) as usize;
            b.mark_dirty(idx);
            reference.insert(idx);
        }
        assert_eq!(b.debug_counts().1, reference.len() as u32);
        let got = drain_sorted(&b);
        let want: Vec<usize> = reference.into_iter().collect();
        assert_eq!(got, want);
        assert!(got.windows(2).all(|w| w[0] < w[1]));
        assert!(drain_sorted(&b).is_empty());
    }
}
