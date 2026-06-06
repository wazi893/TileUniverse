//! Cross-CPU Spike Routing
//!
//! Routes spikes between CPUs in a multi-CPU SNN grid using the
//! quantum router's direct tile access pattern.

use std::collections::HashMap;

use super::synapse::CrossCpuSynapse;

/// Spike packet for cross-CPU transmission (4 bytes)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct SpikePacket {
    /// Source CPU ID (supports up to 65536 CPUs)
    pub source_cpu: u16,
    /// Source neuron index within CPU (0-255)
    pub source_neuron: u8,
    /// Tick when spike occurred (for STDP timing)
    pub timestamp: u8,
}

impl SpikePacket {
    pub fn new(source_cpu: u16, source_neuron: u8, timestamp: u8) -> Self {
        Self {
            source_cpu,
            source_neuron,
            timestamp,
        }
    }

    /// Check if this is a valid spike (non-zero source)
    pub fn is_valid(&self) -> bool {
        // A spike from CPU 0, neuron 0 at time 0 is ambiguous
        // We use a convention: invalid packets have all zeros
        self.source_cpu != 0 || self.source_neuron != 0 || self.timestamp != 0
    }
}

/// Ring buffer for spike packets
#[derive(Clone, Debug)]
pub struct SpikeBuffer {
    /// Fixed-size buffer for spikes
    pub buffer: Vec<SpikePacket>,
    /// Head index (next write position)
    pub head: usize,
    /// Tail index (next read position)
    pub tail: usize,
    /// Number of spikes in buffer
    pub count: usize,
}

impl SpikeBuffer {
    /// Create a new spike buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![SpikePacket::default(); capacity],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    /// Push a spike to the buffer
    pub fn push(&mut self, spike: SpikePacket) -> bool {
        if self.count >= self.buffer.len() {
            return false; // Buffer full
        }

        self.buffer[self.head] = spike;
        self.head = (self.head + 1) % self.buffer.len();
        self.count += 1;
        true
    }

    /// Pop a spike from the buffer
    pub fn pop(&mut self) -> Option<SpikePacket> {
        if self.count == 0 {
            return None;
        }

        let spike = self.buffer[self.tail];
        self.tail = (self.tail + 1) % self.buffer.len();
        self.count -= 1;
        Some(spike)
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.count = 0;
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.count >= self.buffer.len()
    }

    /// Get all spikes without removing them
    pub fn peek_all(&self) -> Vec<SpikePacket> {
        let mut result = Vec::with_capacity(self.count);
        let mut idx = self.tail;
        for _ in 0..self.count {
            result.push(self.buffer[idx]);
            idx = (idx + 1) % self.buffer.len();
        }
        result
    }

    /// Drain all spikes from the buffer
    pub fn drain(&mut self) -> Vec<SpikePacket> {
        let result = self.peek_all();
        self.clear();
        result
    }
}

/// Spike router for a single CPU
///
/// Manages incoming and outgoing spike buffers and cross-CPU synapse routing.
#[derive(Clone, Debug)]
pub struct SpikeRouter {
    /// CPU ID this router belongs to
    pub cpu_id: u16,

    /// Incoming spike buffer (from other CPUs)
    pub incoming: SpikeBuffer,

    /// Outgoing spike buffer (to other CPUs)
    pub outgoing: SpikeBuffer,

    /// Cross-CPU synapse lookup: source neuron -> list of cross-CPU synapses
    pub cross_cpu_synapses: HashMap<u8, Vec<CrossCpuSynapse>>,

    /// List of CPUs this router needs to read from
    pub source_cpus: Vec<u16>,

    /// List of CPUs this router sends to
    pub target_cpus: Vec<u16>,
}

impl SpikeRouter {
    /// Create a new spike router for a CPU
    pub fn new(cpu_id: u16, buffer_size: usize) -> Self {
        Self {
            cpu_id,
            incoming: SpikeBuffer::new(buffer_size),
            outgoing: SpikeBuffer::new(buffer_size),
            cross_cpu_synapses: HashMap::new(),
            source_cpus: Vec::new(),
            target_cpus: Vec::new(),
        }
    }

    /// Add a cross-CPU synapse
    pub fn add_cross_cpu_synapse(&mut self, source_neuron: u8, synapse: CrossCpuSynapse) {
        self.cross_cpu_synapses
            .entry(source_neuron)
            .or_default()
            .push(synapse);

        // Track target CPU
        let target = synapse.target_cpu as u16;
        if !self.target_cpus.contains(&target) {
            self.target_cpus.push(target);
        }
    }

    /// Add a source CPU (CPU we need to read spikes from)
    pub fn add_source_cpu(&mut self, cpu_id: u16) {
        if !self.source_cpus.contains(&cpu_id) {
            self.source_cpus.push(cpu_id);
        }
    }

    /// Queue a local spike for outgoing transmission
    pub fn queue_outgoing(&mut self, neuron: u8, timestamp: u8) {
        let spike = SpikePacket::new(self.cpu_id, neuron, timestamp);
        self.outgoing.push(spike);
    }

    /// Queue an incoming spike from another CPU
    pub fn queue_incoming(&mut self, spike: SpikePacket) {
        self.incoming.push(spike);
    }

    /// Get cross-CPU synapses for a source neuron
    pub fn get_synapses(&self, source_neuron: u8) -> Option<&Vec<CrossCpuSynapse>> {
        self.cross_cpu_synapses.get(&source_neuron)
    }

    /// Clear all buffers (for next tick)
    pub fn clear_buffers(&mut self) {
        self.incoming.clear();
        self.outgoing.clear();
    }

    /// Number of cross-CPU synapses
    pub fn n_cross_cpu_synapses(&self) -> usize {
        self.cross_cpu_synapses.values().map(|v| v.len()).sum()
    }
}

/// Multi-CPU spike routing manager
///
/// Coordinates spike routing across all CPUs in the SNN grid.
#[derive(Clone, Debug)]
pub struct SpikeRoutingManager {
    /// Routers for each CPU
    pub routers: Vec<SpikeRouter>,

    /// Current simulation tick
    pub tick: u8,

    /// Total spikes routed this tick
    pub spikes_routed: u64,

    /// Total cross-CPU spikes ever
    pub total_cross_cpu_spikes: u64,
}

impl SpikeRoutingManager {
    /// Create a new routing manager for `n_cpus` CPUs
    pub fn new(n_cpus: usize, buffer_size: usize) -> Self {
        let routers = (0..n_cpus)
            .map(|i| SpikeRouter::new(i as u16, buffer_size))
            .collect();

        Self {
            routers,
            tick: 0,
            spikes_routed: 0,
            total_cross_cpu_spikes: 0,
        }
    }

    /// Add a cross-CPU synapse to the appropriate router
    pub fn add_synapse(&mut self, synapse: CrossCpuSynapse) {
        let source_cpu = synapse.source_cpu as usize;
        if source_cpu < self.routers.len() {
            self.routers[source_cpu].add_cross_cpu_synapse(synapse.source_neuron, synapse);

            // Also record that target CPU needs to read from source CPU
            let target_cpu = synapse.target_cpu as usize;
            if target_cpu < self.routers.len() {
                self.routers[target_cpu].add_source_cpu(synapse.source_cpu as u16);
            }
        }
    }

    /// Queue a spike from a neuron
    pub fn queue_spike(&mut self, cpu_id: usize, neuron: u8) {
        if cpu_id < self.routers.len() {
            self.routers[cpu_id].queue_outgoing(neuron, self.tick);
        }
    }

    /// Execute spike routing phase
    ///
    /// This is called after all CPUs have completed their local updates.
    /// Uses barrier synchronization pattern from quantum router.
    pub fn route_spikes(&mut self) {
        self.spikes_routed = 0;

        // Phase 1: Collect all outgoing spikes
        let mut all_spikes: Vec<(u16, Vec<SpikePacket>)> = Vec::new();
        for router in &mut self.routers {
            let spikes = router.outgoing.drain();
            if !spikes.is_empty() {
                all_spikes.push((router.cpu_id, spikes));
            }
        }

        // Phase 2: Collect all routing decisions (to avoid borrow conflicts)
        let mut routing_decisions: Vec<(usize, SpikePacket)> = Vec::new();

        for (source_cpu, spikes) in &all_spikes {
            for spike in spikes {
                // Look up cross-CPU synapses for this spike
                if let Some(synapses) =
                    self.routers[*source_cpu as usize].get_synapses(spike.source_neuron)
                {
                    for syn in synapses {
                        let target_cpu = syn.target_cpu as usize;
                        if target_cpu < self.routers.len() {
                            routing_decisions.push((target_cpu, *spike));
                        }
                    }
                }
            }
        }

        // Phase 3: Apply routing decisions
        for (target_cpu, spike) in routing_decisions {
            self.routers[target_cpu].queue_incoming(spike);
            self.spikes_routed += 1;
            self.total_cross_cpu_spikes += 1;
        }
    }

    /// Get incoming spikes for a CPU
    pub fn get_incoming(&self, cpu_id: usize) -> Vec<SpikePacket> {
        if cpu_id < self.routers.len() {
            self.routers[cpu_id].incoming.peek_all()
        } else {
            Vec::new()
        }
    }

    /// Clear all incoming buffers (after processing)
    pub fn clear_incoming(&mut self) {
        for router in &mut self.routers {
            router.incoming.clear();
        }
    }

    /// Advance to next tick
    pub fn next_tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
    }

    /// Number of CPUs
    pub fn n_cpus(&self) -> usize {
        self.routers.len()
    }

    /// Total number of cross-CPU synapses
    pub fn total_cross_cpu_synapses(&self) -> usize {
        self.routers.iter().map(|r| r.n_cross_cpu_synapses()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_buffer() {
        let mut buffer = SpikeBuffer::new(4);

        assert!(buffer.is_empty());

        // Push spikes
        assert!(buffer.push(SpikePacket::new(0, 1, 10)));
        assert!(buffer.push(SpikePacket::new(0, 2, 11)));
        assert_eq!(buffer.count, 2);

        // Pop spike
        let spike = buffer.pop().unwrap();
        assert_eq!(spike.source_neuron, 1);
        assert_eq!(spike.timestamp, 10);

        // Push more
        buffer.push(SpikePacket::new(0, 3, 12));
        buffer.push(SpikePacket::new(0, 4, 13));

        assert_eq!(buffer.count, 3);

        // Fill buffer
        buffer.push(SpikePacket::new(0, 5, 14));
        assert!(buffer.is_full());

        // Can't push more
        assert!(!buffer.push(SpikePacket::new(0, 6, 15)));
    }

    #[test]
    fn test_spike_router() {
        let mut router = SpikeRouter::new(0, 64);

        // Add cross-CPU synapse
        let syn = CrossCpuSynapse::new(0, 5, 1, 10, 50, 1);
        router.add_cross_cpu_synapse(5, syn);

        assert_eq!(router.n_cross_cpu_synapses(), 1);
        assert!(router.get_synapses(5).is_some());
        assert!(router.get_synapses(0).is_none());
    }

    #[test]
    fn test_routing_manager() {
        let mut manager = SpikeRoutingManager::new(4, 32);

        // Add cross-CPU synapse: CPU 0 neuron 1 -> CPU 2 neuron 5
        manager.add_synapse(CrossCpuSynapse::new(0, 1, 2, 5, 50, 0));

        // Queue spike from CPU 0 neuron 1
        manager.queue_spike(0, 1);

        // Route spikes
        manager.route_spikes();

        // Check spike arrived at CPU 2
        let incoming = manager.get_incoming(2);
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].source_neuron, 1);
    }

    #[test]
    fn test_spike_packet_size() {
        assert_eq!(std::mem::size_of::<SpikePacket>(), 4);
    }
}
