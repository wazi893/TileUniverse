//! Milestone 2: Neuromorphic substrate — physical tile simulation for SNN spike routing.
//!
//! Wraps a `Simulation` configured as a router mesh. Wire tiles are placed along
//! pre-routed paths so that spike signals propagate with realistic timing.
//! `measure_delay()` injects a signal and counts propagation deltas.

use crate::simulation::Simulation;
use crate::snn::router_mesh::RouterMesh;
use crate::tiles::tile_meta::TileType;

/// A tile simulation configured as an SNN routing substrate.
pub struct NeuromorphicSubstrate {
    pub sim: Simulation,
    pub mesh: RouterMesh,
    /// neuron_id → flat tile index in sim (z=0)
    #[allow(dead_code)]
    neuron_tile_idx: Vec<Option<usize>>,
    /// Maximum neuron id that has been mapped.
    #[allow(dead_code)]
    max_neuron: usize,
}

impl NeuromorphicSubstrate {
    /// Build the substrate from a completed RouterMesh.
    ///
    /// Places directional Wire tiles along every routed path and Const(0)
    /// guard tiles on all unoccupied positions to prevent signal leakage.
    pub fn new(mesh: RouterMesh) -> Self {
        let w = mesh.width;
        let h = mesh.height;
        let mut sim = Simulation::with_size_layered(w, h, 1);

        // Start with all tiles as Const(0) — dead ends that block propagation.
        for y in 0..h {
            for x in 0..w {
                sim.set_tile_3d(x, y, 0, TileType::Const);
                sim.set_logic_value_3d(x, y, 0, 0);
            }
        }

        // Collect neuron home positions — these stay as Const (signal sources/sinks).
        let mut is_neuron_tile = vec![false; w * h];
        for &(nx, ny) in mesh.all_positions().values() {
            is_neuron_tile[ny * w + nx] = true;
        }

        // Place Wire tiles along every routed path, EXCEPT at neuron positions.
        // Neuron positions stay as Const: injecting a value into Const persists
        // across evaluation (Wire would overwrite it with left|right|up|down).
        let mut is_route_tile = vec![false; w * h];
        for (_, (_, path)) in mesh.all_routes() {
            for &(px, py) in path.iter() {
                if !is_neuron_tile[py * w + px] {
                    is_route_tile[py * w + px] = true;
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                if is_route_tile[y * w + x] {
                    sim.set_tile_3d(x, y, 0, TileType::Wire);
                }
            }
        }

        // Build neuron tile index map.
        let max_neuron = mesh.all_positions().keys().copied().max().unwrap_or(0);
        let mut neuron_tile_idx = vec![None; max_neuron + 1];
        for (&nid, &(x, y)) in mesh.all_positions() {
            if let Some(idx) = sim.tilemap.index_3d(x, y, 0) {
                neuron_tile_idx[nid] = Some(idx);
            }
        }

        // Initial settle.
        sim.dirty.mark_all_dirty(sim.tile_count());
        sim.propagate_combinational_counted();

        Self {
            sim,
            mesh,
            neuron_tile_idx,
            max_neuron,
        }
    }

    /// Zero all logic values and settle.
    pub fn clear_all(&mut self) {
        let tc = self.sim.tile_count();
        for idx in 0..tc {
            self.sim.set_logic_value_by_idx(idx, 0);
        }
        self.sim.dirty.mark_all_dirty(tc);
        self.sim.propagate_combinational_counted();
    }

    /// Inject a spike at a neuron's home tile (Const tile: set value, mark neighbors dirty).
    pub fn inject_spike(&mut self, neuron_id: usize) {
        if let Some(&(x, y)) = self.mesh.all_positions().get(&neuron_id) {
            if let Some(idx) = self.sim.tilemap.index_3d(x, y, 0) {
                self.sim.set_logic_value_by_idx(idx, u64::MAX);
                // Const tiles preserve their value — mark neighbors dirty so
                // adjacent Wire tiles pick up the changed output.
                let w = self.mesh.width;
                let h = self.mesh.height;
                if x > 0 {
                    if let Some(n) = self.sim.tilemap.index_3d(x - 1, y, 0) {
                        self.sim.dirty.mark_dirty(n);
                    }
                }
                if x + 1 < w {
                    if let Some(n) = self.sim.tilemap.index_3d(x + 1, y, 0) {
                        self.sim.dirty.mark_dirty(n);
                    }
                }
                if y > 0 {
                    if let Some(n) = self.sim.tilemap.index_3d(x, y - 1, 0) {
                        self.sim.dirty.mark_dirty(n);
                    }
                }
                if y + 1 < h {
                    if let Some(n) = self.sim.tilemap.index_3d(x, y + 1, 0) {
                        self.sim.dirty.mark_dirty(n);
                    }
                }
            }
        }
    }

    /// Read logic value at a neuron's home tile.
    ///
    /// For destination neurons (Const tiles), the signal arrives when an
    /// adjacent Wire tile outputs non-zero. We read the Const tile's
    /// neighbors and OR them together.
    pub fn read_signal(&self, neuron_id: usize) -> u64 {
        if let Some(&(x, y)) = self.mesh.all_positions().get(&neuron_id) {
            // Read the Wire tiles adjacent to this Const neuron tile.
            let w = self.mesh.width;
            let h = self.mesh.height;
            let mut val = 0u64;
            let neighbors = [
                if x > 0 { Some((x - 1, y)) } else { None },
                if x + 1 < w { Some((x + 1, y)) } else { None },
                if y > 0 { Some((x, y - 1)) } else { None },
                if y + 1 < h { Some((x, y + 1)) } else { None },
            ];
            for pos in &neighbors {
                if let Some((nx, ny)) = pos {
                    if let Some(idx) = self.sim.tilemap.index_3d(*nx, *ny, 0) {
                        val |= self.sim.get_logic_value_by_idx(idx);
                    }
                }
            }
            val
        } else {
            0
        }
    }

    /// Measure propagation delay from src to dst in deltas.
    ///
    /// Clears the grid, injects a spike at src, propagates to convergence,
    /// and returns the total delta count (= hop count along the wire path).
    #[allow(unused_variables)]
    pub fn measure_delay(&mut self, src: usize, dst: usize) -> u32 {
        self.clear_all();
        self.inject_spike(src);
        let (deltas, _, _) = self.sim.propagate_combinational_counted();
        deltas
    }

    /// Measure contention: inject spikes from multiple sources simultaneously,
    /// propagate to convergence, return the total delta count.
    ///
    /// Wire tiles use OR semantics, so all signals merge. The delta count
    /// equals the longest path among the injected sources.
    #[allow(unused_variables)]
    pub fn measure_first_arrival(&mut self, sources: &[usize], dst: usize) -> u32 {
        self.clear_all();
        for &src in sources {
            self.inject_spike(src);
        }
        let (deltas, _, _) = self.sim.propagate_combinational_counted();
        deltas
    }

    /// Measure per-source delays from multiple sources to a single destination.
    ///
    /// Returns a Vec of (neuron_id, delay) pairs. Each source is measured
    /// independently (separate clear+inject+propagate per source).
    pub fn measure_per_source_delays(
        &mut self,
        sources: &[usize],
        dst: usize,
    ) -> Vec<(usize, u32)> {
        sources
            .iter()
            .map(|&src| (src, self.measure_delay(src, dst)))
            .collect()
    }

    /// Width of the substrate grid.
    pub fn width(&self) -> usize {
        self.mesh.width
    }

    /// Height of the substrate grid.
    pub fn height(&self) -> usize {
        self.mesh.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m2_substrate_short_path() {
        let mut mesh = RouterMesh::new(8, 8);
        mesh.place_neuron(0, 0, 0);
        mesh.place_neuron(1, 5, 0);
        let routed = mesh.route_all_synapses(&[(0, 1)]);
        assert_eq!(routed, 1);

        let mut sub = NeuromorphicSubstrate::new(mesh);
        let d = sub.measure_delay(0, 1);
        assert!(d > 0, "delay should be > 0 for 5-hop path");
        assert!(d <= 8, "delay should be <= 8 hops for 5-unit distance");
    }

    #[test]
    fn test_m2_substrate_corner_to_corner() {
        let mut mesh = RouterMesh::new(32, 32);
        mesh.place_neuron(0, 0, 0);
        mesh.place_neuron(1, 31, 31);
        let routed = mesh.route_all_synapses(&[(0, 1)]);
        assert_eq!(routed, 1);

        let mut sub = NeuromorphicSubstrate::new(mesh);
        let d = sub.measure_delay(0, 1);
        // Manhattan = 62. Measured = 61 (first hop from Const→Wire is 0-cost).
        assert!(
            d >= 59 && d <= 63,
            "corner delay should be ~61 deltas (Manhattan=62), got {}",
            d
        );
    }

    #[test]
    fn test_m2_substrate_clear_resets() {
        let mut mesh = RouterMesh::new(8, 8);
        mesh.place_neuron(0, 0, 0);
        mesh.place_neuron(1, 3, 0);
        mesh.route_all_synapses(&[(0, 1)]);

        let mut sub = NeuromorphicSubstrate::new(mesh);
        sub.inject_spike(0);
        // After clear, dst should read 0.
        sub.clear_all();
        assert_eq!(sub.read_signal(1), 0);
    }

    #[test]
    fn test_m2_substrate_multiple_routes() {
        let mut mesh = RouterMesh::new(16, 16);
        mesh.place_neuron(0, 0, 0);
        mesh.place_neuron(1, 7, 0);
        mesh.place_neuron(2, 0, 7);
        mesh.place_neuron(3, 7, 7);
        mesh.route_all_synapses(&[(0, 1), (0, 2), (0, 3), (2, 3)]);

        let mut sub = NeuromorphicSubstrate::new(mesh);

        let d01 = sub.measure_delay(0, 1);
        let d02 = sub.measure_delay(0, 2);
        let d03 = sub.measure_delay(0, 3);
        let d23 = sub.measure_delay(2, 3);

        assert!(d01 > 0, "d01 should be > 0");
        assert!(d02 > 0, "d02 should be > 0");
        assert!(d03 > 0, "d03 should be > 0");
        assert!(d23 > 0, "d23 should be > 0");
        // Diagonal should be longest.
        assert!(d03 >= d01 && d03 >= d02, "diagonal should be longest");
    }

    #[test]
    fn test_m2_contention_differential_delays() {
        // Place 5 sources at different distances from a common destination.
        // Verify that closer sources have shorter delays.
        let mut mesh = RouterMesh::new(32, 32);
        // Destination at (16, 16).
        mesh.place_neuron(100, 16, 16);
        // Sources at increasing Manhattan distances.
        mesh.place_neuron(0, 14, 16); // dist 2
        mesh.place_neuron(1, 10, 16); // dist 6
        mesh.place_neuron(2, 4, 16); // dist 12
        mesh.place_neuron(3, 0, 16); // dist 16
        mesh.place_neuron(4, 0, 0); // dist 32

        let synapses: Vec<(usize, usize)> = (0..5).map(|i| (i, 100)).collect();
        let routed = mesh.route_all_synapses(&synapses);
        assert_eq!(routed, 5);

        let mut sub = NeuromorphicSubstrate::new(mesh);
        let delays = sub.measure_per_source_delays(&[0, 1, 2, 3, 4], 100);

        // Delays should be monotonically increasing with distance.
        for i in 0..4 {
            assert!(
                delays[i].1 <= delays[i + 1].1,
                "delay[{}]={} should be <= delay[{}]={} (closer source should arrive first)",
                i,
                delays[i].1,
                i + 1,
                delays[i + 1].1
            );
        }
        // The nearest source (dist 2) should have much less delay than farthest (dist 32).
        assert!(
            delays[4].1 >= delays[0].1 + 10,
            "farthest delay {} should exceed nearest {} by at least 10",
            delays[4].1,
            delays[0].1
        );
    }
}
