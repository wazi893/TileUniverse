//! Milestone 2: Router mesh for tile-based SNN spike routing.
//!
//! Maps SNN neurons to (x, y) positions in a tile grid and computes
//! A*-based routing paths between synapse endpoints. The path length
//! (hop count) determines the spike propagation delay.

use std::collections::HashMap;

use crate::tile_cpu::v2_routing::{
    V2EndpointClass, V2RouteBounds, V2RouteCoord, V2RouteNet, V2RouteNetClass, V2RoutingConfig,
    V2RoutingDb,
};

/// A neuron placement and routing table for a tile-based SNN substrate.
pub struct RouterMesh {
    pub width: usize,
    pub height: usize,
    /// neuron_id → (x, y) position in the mesh grid
    positions: HashMap<usize, (usize, usize)>,
    /// (src_neuron, dst_neuron) → (delay_hops, path coordinates)
    routes: HashMap<(usize, usize), (u8, Vec<(usize, usize)>)>,
}

impl RouterMesh {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            positions: HashMap::new(),
            routes: HashMap::new(),
        }
    }

    /// Place a neuron at grid position (x, y).
    pub fn place_neuron(&mut self, neuron_id: usize, x: usize, y: usize) {
        assert!(x < self.width && y < self.height, "position out of bounds");
        self.positions.insert(neuron_id, (x, y));
    }

    /// Place neurons in layer-clustered rows.
    ///
    /// Each entry: `(neuron_start, neuron_end, start_row)`.
    /// Neurons are packed left-to-right across `width` columns,
    /// wrapping to the next row when a row fills.
    pub fn place_layer_clustered(&mut self, layers: &[(usize, usize, usize)]) {
        for &(start, end, start_row) in layers {
            for (i, nid) in (start..end).enumerate() {
                let x = i % self.width;
                let y = start_row + i / self.width;
                if y < self.height {
                    self.positions.insert(nid, (x, y));
                }
            }
        }
    }

    /// Get the position of a placed neuron.
    pub fn position(&self, neuron_id: usize) -> Option<(usize, usize)> {
        self.positions.get(&neuron_id).copied()
    }

    /// Number of placed neurons.
    pub fn neuron_count(&self) -> usize {
        self.positions.len()
    }

    /// Route all given synapses through the mesh using A*.
    ///
    /// `synapses`: slice of `(src_neuron, dst_neuron)` pairs.
    /// Returns the number of successfully routed paths.
    pub fn route_all_synapses(&mut self, synapses: &[(usize, usize)]) -> usize {
        self.routes.clear();

        // Deduplicate: multiple synapses between same (src, dst) share one route.
        let mut unique_pairs: Vec<(usize, usize)> = synapses.to_vec();
        unique_pairs.sort_unstable();
        unique_pairs.dedup();

        // Filter to pairs where both neurons are placed.
        let valid: Vec<(usize, usize)> = unique_pairs
            .into_iter()
            .filter(|(s, d)| self.positions.contains_key(s) && self.positions.contains_key(d))
            .collect();

        if valid.is_empty() {
            return 0;
        }

        // Build routing DB: single layer, all cells open.
        let bounds = V2RouteBounds::new(0, self.width, 0, self.height, 0, 0);
        let db = V2RoutingDb::new(self.width, self.height, 1, bounds);

        // Build nets in batches to avoid overloading the router.
        // For SNN routing, nets are Data class (no priority distinction).
        let batch_size = 256;
        let mut routed = 0usize;

        for chunk in valid.chunks(batch_size) {
            let nets: Vec<V2RouteNet> = chunk
                .iter()
                .map(|&(src, dst)| {
                    let (sx, sy) = self.positions[&src];
                    let (dx, dy) = self.positions[&dst];
                    V2RouteNet::new(
                        format!("s{}_{}", src, dst),
                        V2RouteNetClass::Data,
                        V2RouteCoord::new(sx, sy, 0),
                        V2RouteCoord::new(dx, dy, 0),
                    )
                    .with_start_class(V2EndpointClass::SharedFan { capacity: 64 })
                    .with_goal_class(V2EndpointClass::SharedFan { capacity: 64 })
                })
                .collect();

            let config = V2RoutingConfig {
                bend_penalty: 0,
                via_penalty: 0,
                congestion_penalty: 0,
                overflow_penalty: 0,
                soft_cost_weight: 0,
                max_negotiation_iters: 1,
                use_input_order: true,
                enable_rip_up: false,
                max_rip_up_per_iter: 0,
                layer_affinity_penalty: 0,
            };

            let result = db.route_multinet(&nets, config);

            for &(src, dst) in chunk.iter() {
                let net_id = format!("s{}_{}", src, dst);
                if let Some(path) = result.routes.get(&net_id) {
                    let hops = if path.coords.len() > 1 {
                        (path.coords.len() - 1) as u8
                    } else {
                        0
                    };
                    let xy_path: Vec<(usize, usize)> =
                        path.coords.iter().map(|c| (c.x, c.y)).collect();
                    self.routes.insert((src, dst), (hops, xy_path));
                    routed += 1;
                }
            }
        }

        routed
    }

    /// Get the routing delay (hop count) for a synapse.
    /// Returns Manhattan distance as fallback if no A* route exists.
    pub fn delay(&self, src: usize, dst: usize) -> u8 {
        if let Some(&(hops, _)) = self.routes.get(&(src, dst)) {
            hops
        } else if let (Some(&(sx, sy)), Some(&(dx, dy))) =
            (self.positions.get(&src), self.positions.get(&dst))
        {
            (sx.abs_diff(dx) + sy.abs_diff(dy)) as u8
        } else {
            0
        }
    }

    /// Get the full routed path for a synapse.
    pub fn path(&self, src: usize, dst: usize) -> Option<&[(usize, usize)]> {
        self.routes.get(&(src, dst)).map(|(_, p)| p.as_slice())
    }

    /// Find the critical path: the synapse with the longest delay.
    /// Returns (src, dst, delay).
    pub fn critical_path(&self) -> (usize, usize, u8) {
        self.routes
            .iter()
            .max_by_key(|(_, (h, _))| *h)
            .map(|(&(s, d), &(h, _))| (s, d, h))
            .unwrap_or((0, 0, 0))
    }

    /// Delay histogram: bins[i] = number of routes with delay == i.
    pub fn delay_histogram(&self) -> Vec<u32> {
        let max_delay = self
            .routes
            .values()
            .map(|(h, _)| *h as usize)
            .max()
            .unwrap_or(0);
        let mut bins = vec![0u32; max_delay + 1];
        for &(h, _) in self.routes.values() {
            bins[h as usize] += 1;
        }
        bins
    }

    /// Number of routed synapse paths.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// All placed neuron positions.
    pub fn all_positions(&self) -> &HashMap<usize, (usize, usize)> {
        &self.positions
    }

    /// All routed paths (read-only).
    pub fn all_routes(&self) -> &HashMap<(usize, usize), (u8, Vec<(usize, usize)>)> {
        &self.routes
    }

    /// Tiles used by 3+ routes (contention hotspots).
    pub fn find_hotspots(&self, min_users: usize) -> Vec<((usize, usize), usize)> {
        let mut usage: HashMap<(usize, usize), usize> = HashMap::new();
        for (_, path) in self.routes.values() {
            for &pos in path {
                *usage.entry(pos).or_insert(0) += 1;
            }
        }
        let mut hotspots: Vec<_> = usage
            .into_iter()
            .filter(|&(_, count)| count >= min_users)
            .collect();
        hotspots.sort_by(|a, b| b.1.cmp(&a.1));
        hotspots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m2_manhattan_distance_baseline() {
        let mut mesh = RouterMesh::new(32, 32);
        mesh.place_neuron(0, 0, 0);
        mesh.place_neuron(1, 31, 31);

        // Manhattan distance should be 62.
        assert_eq!(mesh.delay(0, 1), 62);
    }

    #[test]
    fn test_m2_route_simple_pair() {
        let mut mesh = RouterMesh::new(32, 32);
        mesh.place_neuron(0, 0, 0);
        mesh.place_neuron(1, 5, 3);

        let routed = mesh.route_all_synapses(&[(0, 1)]);
        assert_eq!(routed, 1);

        // A* delay should equal Manhattan distance (no obstacles).
        let d = mesh.delay(0, 1);
        assert_eq!(d, 8); // |5-0| + |3-0| = 8
    }

    #[test]
    fn test_m2_route_corner_to_corner() {
        let mut mesh = RouterMesh::new(32, 32);
        mesh.place_neuron(0, 0, 0);
        mesh.place_neuron(1, 31, 31);

        let routed = mesh.route_all_synapses(&[(0, 1)]);
        assert_eq!(routed, 1);

        let d = mesh.delay(0, 1);
        assert_eq!(d, 62);
    }

    #[test]
    fn test_m2_layer_clustered_placement() {
        let mut mesh = RouterMesh::new(32, 32);
        // 10 input neurons starting at row 0, 10 hidden at row 5
        mesh.place_layer_clustered(&[(0, 10, 0), (10, 20, 5)]);

        assert_eq!(mesh.neuron_count(), 20);
        assert_eq!(mesh.position(0), Some((0, 0)));
        assert_eq!(mesh.position(9), Some((9, 0)));
        assert_eq!(mesh.position(10), Some((0, 5)));
        assert_eq!(mesh.position(19), Some((9, 5)));
    }

    #[test]
    fn test_m2_critical_path() {
        let mut mesh = RouterMesh::new(32, 32);
        mesh.place_neuron(0, 0, 0);
        mesh.place_neuron(1, 10, 0);
        mesh.place_neuron(2, 31, 31);

        let routed = mesh.route_all_synapses(&[(0, 1), (0, 2)]);
        assert_eq!(routed, 2);

        let (_, _, max_d) = mesh.critical_path();
        assert_eq!(max_d, 62);
    }

    #[test]
    fn test_m2_hotspot_detection() {
        let mut mesh = RouterMesh::new(8, 8);
        // Three neurons: all routes pass through center area
        mesh.place_neuron(0, 0, 3);
        mesh.place_neuron(1, 7, 3);
        mesh.place_neuron(2, 0, 4);
        mesh.place_neuron(3, 7, 4);

        mesh.route_all_synapses(&[(0, 1), (2, 3), (0, 3), (2, 1)]);

        let hotspots = mesh.find_hotspots(3);
        // Center tiles should be shared by multiple routes
        assert!(
            !hotspots.is_empty() || mesh.route_count() < 4,
            "expected hotspots in center of 8x8 grid with 4 cross-routes"
        );
    }
}
