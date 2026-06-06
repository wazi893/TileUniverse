//! Magic State Supply for Fault-Tolerant Computing
//!
//! Bridges the QEC logical qubit layer with the magic state distillation
//! infrastructure in `qram::distillation` and `qram::magic_budget`.
//!
//! # Design Principles
//!
//! 1. **Throughput-aware**: Not just "how many states" but "states per cycle"
//! 2. **Latency-aware**: Factory has startup time and pipeline depth
//! 3. **Inventory tracking**: Buffer smooths bursty consumption
//! 4. **Resource accounting**: Every consumption returns cost delta
//!
//! # Example
//!
//! ```ignore
//! use engine::qec::MagicSupply;
//!
//! // Create supply for target 10^-10 error rate
//! let mut supply = MagicSupply::new(1e-3, 1e-10);
//!
//! // Consume magic states for T-gates
//! let result = supply.consume();
//! println!("State error: {:.2e}", result.error_rate);
//! println!("Delay: {} cycles", result.delay_cycles);
//! ```

use std::fmt;

use crate::qram::{DistillationFactory, DistillationProtocol, FactoryResources, MagicStateBudget};

use super::ft_types::ResourceDelta;

// ============================================================================
// Magic State Result
// ============================================================================

/// Result of requesting a magic state from the supply.
#[derive(Clone, Debug)]
pub struct MagicStateResult {
    /// Error rate of the distilled magic state
    pub error_rate: f64,
    /// Delay if factory was saturated (cycles waited)
    pub delay_cycles: usize,
    /// Resource cost of this consumption
    pub resource_cost: ResourceDelta,
    /// Was this served from inventory (no production delay)?
    pub from_inventory: bool,
}

impl MagicStateResult {
    /// Create result for immediately available state
    pub fn immediate(error_rate: f64) -> Self {
        Self {
            error_rate,
            delay_cycles: 0,
            resource_cost: ResourceDelta::zero(),
            from_inventory: true,
        }
    }

    /// Create result for freshly produced state
    pub fn produced(error_rate: f64, production_cycles: usize, qubits: usize) -> Self {
        Self {
            error_rate,
            delay_cycles: production_cycles,
            resource_cost: ResourceDelta {
                qubits,
                cycles: production_cycles,
                magic_states: 1,
            },
            from_inventory: false,
        }
    }
}

// ============================================================================
// Magic Supply
// ============================================================================

/// Magic state supply with throughput and latency awareness.
///
/// This is the bridge between logical circuit execution and the
/// magic state distillation infrastructure.
#[derive(Clone, Debug)]
pub struct MagicSupply {
    /// Physical error rate of input states
    physical_error: f64,
    /// Target error rate for output states
    target_error: f64,
    /// Distillation factory configuration
    factory: DistillationFactory,
    /// States produced per cycle (throughput)
    throughput: f64,
    /// Cycles to produce one state (latency/pipeline depth)
    latency: usize,
    /// Current inventory of pre-distilled states
    inventory: usize,
    /// Physical qubits dedicated to factory
    factory_footprint: usize,
    /// Total states consumed
    total_consumed: usize,
    /// Total delay accumulated (cycles waited for production)
    total_delay: usize,
    /// Running budget for resource tracking
    budget: MagicStateBudget,
}

impl MagicSupply {
    /// Create magic supply for target error rate.
    ///
    /// Automatically configures distillation levels to achieve target.
    pub fn new(physical_error: f64, target_error: f64) -> Self {
        let factory = DistillationFactory::for_target_error(physical_error, target_error);
        let resources = factory.resources_for(1);

        // Estimate throughput and latency from factory resources
        // Throughput: 1 / cycles_per_state (states per cycle)
        let cycles_per_state = resources.total_cycles.max(1);
        let throughput = 1.0 / cycles_per_state as f64;
        let latency = cycles_per_state;
        let factory_footprint = resources.total_qubits;

        Self {
            physical_error,
            target_error,
            factory,
            throughput,
            latency,
            inventory: 0,
            factory_footprint,
            total_consumed: 0,
            total_delay: 0,
            budget: MagicStateBudget::new(),
        }
    }

    /// Create with explicit factory configuration.
    pub fn with_factory(factory: DistillationFactory, physical_error: f64) -> Self {
        let target_error = factory.achieved_error_rate();
        let resources = factory.resources_for(1);
        let cycles_per_state = resources.total_cycles.max(1);

        Self {
            physical_error,
            target_error,
            factory,
            throughput: 1.0 / cycles_per_state as f64,
            latency: cycles_per_state,
            inventory: 0,
            factory_footprint: resources.total_qubits,
            total_consumed: 0,
            total_delay: 0,
            budget: MagicStateBudget::new(),
        }
    }

    /// Create with pre-filled inventory (for testing or warm start).
    pub fn with_inventory(mut self, inventory: usize) -> Self {
        self.inventory = inventory;
        self
    }

    /// Create with custom throughput (multiple factories in parallel).
    pub fn with_throughput(mut self, states_per_cycle: f64) -> Self {
        self.throughput = states_per_cycle;
        // More throughput means more qubits
        let factories = (states_per_cycle * self.latency as f64).ceil() as usize;
        self.factory_footprint *= factories.max(1);
        self
    }

    /// Consume one magic state for T-gate.
    ///
    /// Returns information about the state including any delay.
    pub fn consume(&mut self) -> MagicStateResult {
        self.total_consumed += 1;
        self.budget.direct_t_gates += 1;

        if self.inventory > 0 {
            // Serve from inventory (no production delay)
            self.inventory -= 1;
            MagicStateResult::immediate(self.target_error)
        } else {
            // Need to wait for production
            let delay = self.latency;
            self.total_delay += delay;
            MagicStateResult::produced(self.target_error, delay, self.factory_footprint)
        }
    }

    /// Consume multiple magic states (e.g., for Toffoli).
    pub fn consume_n(&mut self, n: usize) -> Vec<MagicStateResult> {
        (0..n).map(|_| self.consume()).collect()
    }

    /// Consume states for a Toffoli gate (7 T-gates by default).
    pub fn consume_toffoli(&mut self) -> Vec<MagicStateResult> {
        self.budget.toffoli_count += 1;
        self.consume_n(7)
    }

    /// Add states to inventory (pre-production).
    pub fn produce(&mut self, n: usize) {
        self.inventory += n;
    }

    /// Check if supply can sustain a given consumption rate.
    ///
    /// Returns true if throughput >= demand.
    pub fn can_sustain_rate(&self, states_per_cycle: f64) -> bool {
        self.throughput >= states_per_cycle
    }

    /// Number of factories needed to sustain a rate.
    pub fn factories_for_rate(&self, states_per_cycle: f64) -> usize {
        if self.throughput <= 0.0 {
            return usize::MAX;
        }
        (states_per_cycle / self.throughput).ceil() as usize
    }

    /// Get output error rate.
    pub fn error_rate(&self) -> f64 {
        self.target_error
    }

    /// Get throughput (states per cycle).
    pub fn throughput(&self) -> f64 {
        self.throughput
    }

    /// Get latency (cycles to produce one state).
    pub fn latency(&self) -> usize {
        self.latency
    }

    /// Get current inventory.
    pub fn inventory(&self) -> usize {
        self.inventory
    }

    /// Get factory footprint (physical qubits).
    pub fn factory_footprint(&self) -> usize {
        self.factory_footprint
    }

    /// Get total states consumed.
    pub fn total_consumed(&self) -> usize {
        self.total_consumed
    }

    /// Get total delay accumulated.
    pub fn total_delay(&self) -> usize {
        self.total_delay
    }

    /// Get running budget.
    pub fn budget(&self) -> &MagicStateBudget {
        &self.budget
    }

    /// Get factory resources for n states.
    pub fn resources_for(&self, n: usize) -> FactoryResources {
        self.factory.resources_for(n)
    }

    /// Get distillation overhead ratio.
    pub fn overhead_ratio(&self) -> f64 {
        self.factory.total_overhead() as f64
    }

    /// Estimate total cost for a circuit with given T-count.
    pub fn estimate_cost(&self, t_count: usize, t_depth: usize) -> MagicSupplyCost {
        // Total magic states needed
        let magic_states = t_count;

        // Factory cycles: depends on T-depth and throughput
        // If we can parallelize T-gates, we need enough throughput
        let t_per_layer = if t_depth > 0 {
            t_count as f64 / t_depth as f64
        } else {
            t_count as f64
        };

        // Factories needed to match T-depth parallelism
        let factories_needed = self.factories_for_rate(t_per_layer / self.latency as f64);

        // Total cycles: max of T-depth and production time
        let production_cycles = if self.throughput > 0.0 {
            (t_count as f64 / (self.throughput * factories_needed as f64)).ceil() as usize
        } else {
            t_count * self.latency
        };

        // Factory qubits
        let factory_qubits = factories_needed * self.factory_footprint;

        MagicSupplyCost {
            magic_states,
            factory_qubits,
            factory_cycles: production_cycles,
            factories_needed,
            overhead_ratio: self.overhead_ratio(),
            bottleneck: if production_cycles > t_depth {
                SupplyBottleneck::FactoryThroughput
            } else {
                SupplyBottleneck::TDepth
            },
        }
    }

    /// Reset consumption tracking.
    pub fn reset(&mut self) {
        self.total_consumed = 0;
        self.total_delay = 0;
        self.budget = MagicStateBudget::new();
    }
}

impl fmt::Display for MagicSupply {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Magic State Supply:")?;
        writeln!(f, "  Physical error:  {:.2e}", self.physical_error)?;
        writeln!(f, "  Target error:    {:.2e}", self.target_error)?;
        writeln!(f, "  Throughput:      {:.3} states/cycle", self.throughput)?;
        writeln!(f, "  Latency:         {} cycles", self.latency)?;
        writeln!(f, "  Factory qubits:  {}", self.factory_footprint)?;
        writeln!(f, "  Inventory:       {}", self.inventory)?;
        writeln!(f, "  Total consumed:  {}", self.total_consumed)?;
        writeln!(f, "  Total delay:     {} cycles", self.total_delay)?;
        Ok(())
    }
}

// ============================================================================
// Supply Cost Estimation
// ============================================================================

/// Cost estimate for magic state supply.
#[derive(Clone, Debug)]
pub struct MagicSupplyCost {
    /// Total magic states needed
    pub magic_states: usize,
    /// Physical qubits for factories
    pub factory_qubits: usize,
    /// Cycles for factory operation
    pub factory_cycles: usize,
    /// Number of parallel factories needed
    pub factories_needed: usize,
    /// Distillation overhead ratio
    pub overhead_ratio: f64,
    /// What limits execution speed
    pub bottleneck: SupplyBottleneck,
}

impl MagicSupplyCost {
    /// Spacetime volume for factory
    pub fn spacetime_volume(&self) -> usize {
        self.factory_qubits * self.factory_cycles
    }
}

impl fmt::Display for MagicSupplyCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Magic Supply Cost:")?;
        writeln!(f, "  Magic states:    {}", self.magic_states)?;
        writeln!(f, "  Factory qubits:  {}", self.factory_qubits)?;
        writeln!(f, "  Factory cycles:  {}", self.factory_cycles)?;
        writeln!(f, "  Factories:       {}", self.factories_needed)?;
        writeln!(f, "  Overhead ratio:  {:.1}:1", self.overhead_ratio)?;
        writeln!(f, "  Bottleneck:      {:?}", self.bottleneck)?;
        Ok(())
    }
}

/// What limits magic state supply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupplyBottleneck {
    /// T-depth limits execution (factory can keep up)
    TDepth,
    /// Factory throughput limits execution (need more factories)
    FactoryThroughput,
}

// ============================================================================
// Factory Pipeline (Phase 2.1)
// ============================================================================

/// A factory pipeline that simulates time-aware magic state production.
///
/// Unlike the simple `MagicSupply::consume()` which assumes instant production
/// or full latency, this models a proper pipeline where:
/// - Factory has a warm-up phase before first state is available
/// - Factory continuously produces states at a given throughput
/// - Requests queue up if demand exceeds supply
/// - Backpressure is tracked and reported
///
/// # Warm-up Phase (Phase 2.1)
///
/// Real distillation factories don't produce states instantly. The warm-up
/// period models the time before the first state exits the pipeline:
/// - `warmup_cycles`: Cycles before first state is available
/// - Production only starts after warm-up completes
/// - This is separate from `latency` (pipeline depth for throughput calculation)
///
/// # Example
///
/// ```ignore
/// let mut pipeline = FactoryPipeline::new(0.1, 100, 50) // 0.1 states/cycle, 100 latency
///     .with_warmup(200);  // 200 cycles before first state
/// pipeline.tick_to(250);  // Advance past warmup
/// let result = pipeline.request(3);
/// println!("Got {} now, {} pending", result.delivered, result.pending);
/// ```
#[derive(Clone, Debug)]
pub struct FactoryPipeline {
    /// States produced per cycle
    throughput: f64,
    /// Cycles from request to delivery (pipeline depth)
    latency: usize,
    /// Warm-up cycles before first state available (Phase 2.1)
    warmup_cycles: usize,
    /// Current simulation cycle
    current_cycle: usize,
    /// Accumulated production (fractional states)
    accumulated_production: f64,
    /// Available states in buffer
    available: usize,
    /// Pending requests (not yet fulfillable)
    pending_requests: usize,
    /// Total states delivered
    total_delivered: usize,
    /// Total stall cycles (backpressure)
    total_stall_cycles: usize,
    /// Stall events for analysis
    stall_events: Vec<StallEvent>,
    /// Factory footprint (physical qubits)
    factory_qubits: usize,
    /// Output error rate
    error_rate: f64,
    /// Has warm-up completed?
    warmup_complete: bool,
}

/// Result of requesting magic states from pipeline.
#[derive(Clone, Debug)]
pub struct PipelineRequest {
    /// States delivered immediately
    pub delivered: usize,
    /// States that couldn't be delivered (pending)
    pub pending: usize,
    /// Cycles to wait for all requested states (0 if all delivered)
    pub wait_cycles: usize,
    /// Was there backpressure (demand > supply)?
    pub backpressure: bool,
}

/// A stall event when demand exceeded supply.
#[derive(Clone, Debug)]
pub struct StallEvent {
    /// Cycle when stall occurred
    pub cycle: usize,
    /// States requested
    pub requested: usize,
    /// States available
    pub available: usize,
    /// Cycles stalled
    pub stall_cycles: usize,
}

impl FactoryPipeline {
    /// Create a new factory pipeline.
    ///
    /// # Arguments
    /// * `throughput` - States produced per cycle
    /// * `latency` - Pipeline depth (cycles from request to delivery)
    /// * `factory_qubits` - Physical qubit footprint
    pub fn new(throughput: f64, latency: usize, factory_qubits: usize) -> Self {
        Self {
            throughput,
            latency,
            warmup_cycles: 0, // No warmup by default (instant start)
            current_cycle: 0,
            accumulated_production: 0.0,
            available: 0,
            pending_requests: 0,
            total_delivered: 0,
            total_stall_cycles: 0,
            stall_events: Vec::new(),
            factory_qubits,
            error_rate: 1e-10,     // Default, can be set
            warmup_complete: true, // No warmup = already complete
        }
    }

    /// Create from MagicSupply configuration.
    pub fn from_supply(supply: &MagicSupply) -> Self {
        Self::new(
            supply.throughput(),
            supply.latency(),
            supply.factory_footprint(),
        )
        .with_error_rate(supply.error_rate())
        // By default, warmup = latency (first state takes latency cycles)
        .with_warmup(supply.latency())
    }

    /// Set the output error rate.
    pub fn with_error_rate(mut self, error_rate: f64) -> Self {
        self.error_rate = error_rate;
        self
    }

    /// Set warm-up cycles before first state is available.
    ///
    /// This models the startup delay of real distillation factories.
    /// During warm-up, no states are produced.
    pub fn with_warmup(mut self, cycles: usize) -> Self {
        self.warmup_cycles = cycles;
        self.warmup_complete = cycles == 0;
        self
    }

    /// Pre-fill the available buffer (warm start).
    ///
    /// If buffer is pre-filled, warm-up is considered complete.
    pub fn with_initial_buffer(mut self, n: usize) -> Self {
        self.available = n;
        if n > 0 {
            self.warmup_complete = true;
        }
        self
    }

    /// Check if warm-up is complete.
    pub fn is_warmed_up(&self) -> bool {
        self.warmup_complete
    }

    /// Get warm-up cycles configured.
    pub fn warmup_cycles(&self) -> usize {
        self.warmup_cycles
    }

    /// Cycles remaining until first state can be produced.
    pub fn warmup_remaining(&self) -> usize {
        if self.warmup_complete {
            0
        } else {
            self.warmup_cycles.saturating_sub(self.current_cycle)
        }
    }

    /// Advance simulation to a specific cycle.
    ///
    /// Produces states according to throughput for all elapsed cycles.
    /// No production occurs during warm-up phase.
    pub fn tick_to(&mut self, target_cycle: usize) {
        if target_cycle <= self.current_cycle {
            return;
        }

        // Check if warm-up completes during this tick
        let production_start_cycle = if self.warmup_complete {
            self.current_cycle
        } else if target_cycle >= self.warmup_cycles {
            self.warmup_complete = true;
            self.warmup_cycles
        } else {
            // Still in warm-up, no production
            self.current_cycle = target_cycle;
            return;
        };

        // Calculate production for cycles after warm-up
        let production_cycles = target_cycle - production_start_cycle;
        if production_cycles > 0 {
            self.accumulated_production += self.throughput * production_cycles as f64;

            // Convert accumulated production to discrete states
            let new_states = self.accumulated_production.floor() as usize;
            self.accumulated_production -= new_states as f64;
            self.available += new_states;
        }

        // Service any pending requests
        if self.pending_requests > 0 && self.available > 0 {
            let serviced = self.pending_requests.min(self.available);
            self.available -= serviced;
            self.pending_requests -= serviced;
            self.total_delivered += serviced;
        }

        self.current_cycle = target_cycle;
    }

    /// Request magic states from the pipeline.
    ///
    /// Returns immediately available states and queues the rest.
    /// Use `wait_cycles` to determine how long to stall if needed.
    pub fn request(&mut self, n: usize) -> PipelineRequest {
        if n == 0 {
            return PipelineRequest {
                delivered: 0,
                pending: 0,
                wait_cycles: 0,
                backpressure: false,
            };
        }

        // Deliver what we have
        let delivered = n.min(self.available);
        self.available -= delivered;
        self.total_delivered += delivered;

        let pending = n - delivered;
        let backpressure = pending > 0;

        // Calculate wait cycles for pending states
        let wait_cycles = if pending > 0 {
            // Account for warm-up remaining first
            let warmup_wait = self.warmup_remaining();

            // How many cycles until we have `pending` more states?
            let production_cycles = if self.throughput > 0.0 {
                ((pending as f64 - self.accumulated_production) / self.throughput).ceil() as usize
            } else {
                usize::MAX
            };

            // Total wait = warmup remaining + production time (at least latency)
            warmup_wait + production_cycles.max(self.latency)
        } else {
            0
        };

        // Record stall event if backpressure occurred
        if backpressure {
            self.pending_requests += pending;
            self.total_stall_cycles += wait_cycles;
            self.stall_events.push(StallEvent {
                cycle: self.current_cycle,
                requested: n,
                available: delivered,
                stall_cycles: wait_cycles,
            });
        }

        PipelineRequest {
            delivered,
            pending,
            wait_cycles,
            backpressure,
        }
    }

    /// Request states and block (advance time) until all are delivered.
    ///
    /// This simulates a circuit that MUST have the states before proceeding.
    pub fn request_blocking(&mut self, n: usize) -> PipelineRequest {
        let result = self.request(n);
        if result.wait_cycles > 0 {
            self.tick_to(self.current_cycle + result.wait_cycles);
            // Now service the pending request
            if self.pending_requests > 0 {
                let serviced = self.pending_requests.min(self.available);
                self.available -= serviced;
                self.pending_requests -= serviced;
                self.total_delivered += serviced;
            }
        }
        result
    }

    /// Analyze demand timeline and compute backpressure report.
    ///
    /// Takes a demand timeline (from Phase 2.2's ResourceSummary) and
    /// simulates whether the factory can keep up.
    ///
    /// Note: Uses the pipeline's initial buffer state for warmup.
    pub fn analyze_demand(&self, demand_bins: &[super::ft_types::DemandBin]) -> BackpressureReport {
        let mut sim = self.clone();
        // Reset counters but preserve initial buffer (for warmup simulation)
        let initial_buffer = sim.available;
        sim.reset();
        sim.available = initial_buffer;

        let mut total_demand = 0;
        let mut unmet_demand = 0;
        let mut stall_bins = 0;

        for bin in demand_bins {
            // Advance to bin start
            sim.tick_to(bin.start_cycle);

            // Request states for this bin
            let demand = bin.magic_states;
            total_demand += demand;

            let result = sim.request(demand);
            if result.backpressure {
                unmet_demand += result.pending;
                stall_bins += 1;
            }

            // Advance to bin end
            sim.tick_to(bin.end_cycle);
        }

        let sustained = unmet_demand == 0;
        let utilization = if total_demand > 0 {
            sim.total_delivered as f64 / total_demand as f64
        } else {
            1.0
        };

        BackpressureReport {
            total_demand,
            total_delivered: sim.total_delivered,
            unmet_demand,
            total_stall_cycles: sim.total_stall_cycles,
            stall_bins,
            total_bins: demand_bins.len(),
            sustained,
            utilization,
            recommended_throughput: if !sustained {
                // Estimate required throughput
                let total_cycles: usize = demand_bins.iter().map(|b| b.duration()).sum();
                if total_cycles > 0 {
                    total_demand as f64 / total_cycles as f64
                } else {
                    self.throughput * 2.0
                }
            } else {
                self.throughput
            },
            spacetime_volume: sim.spacetime_volume(),
        }
    }

    /// Reset the pipeline state (including warm-up).
    pub fn reset(&mut self) {
        self.current_cycle = 0;
        self.accumulated_production = 0.0;
        self.available = 0;
        self.pending_requests = 0;
        self.total_delivered = 0;
        self.total_stall_cycles = 0;
        self.stall_events.clear();
        self.warmup_complete = self.warmup_cycles == 0;
    }

    /// Reset but preserve warm-up state (for mid-simulation analysis).
    pub fn reset_counters_only(&mut self) {
        self.pending_requests = 0;
        self.total_delivered = 0;
        self.total_stall_cycles = 0;
        self.stall_events.clear();
    }

    // Accessors

    pub fn throughput(&self) -> f64 {
        self.throughput
    }

    pub fn latency(&self) -> usize {
        self.latency
    }

    pub fn current_cycle(&self) -> usize {
        self.current_cycle
    }

    pub fn available(&self) -> usize {
        self.available
    }

    pub fn pending_requests(&self) -> usize {
        self.pending_requests
    }

    pub fn total_delivered(&self) -> usize {
        self.total_delivered
    }

    pub fn total_stall_cycles(&self) -> usize {
        self.total_stall_cycles
    }

    pub fn stall_events(&self) -> &[StallEvent] {
        &self.stall_events
    }

    pub fn factory_qubits(&self) -> usize {
        self.factory_qubits
    }

    pub fn error_rate(&self) -> f64 {
        self.error_rate
    }

    /// Spacetime volume: factory qubits × current cycle.
    ///
    /// This represents the total qubit-cycles consumed by this factory
    /// from cycle 0 to the current cycle.
    pub fn spacetime_volume(&self) -> usize {
        self.factory_qubits * self.current_cycle
    }

    /// Spacetime cost per magic state produced.
    ///
    /// Returns the average qubit-cycles per state delivered.
    pub fn spacetime_per_state(&self) -> f64 {
        if self.total_delivered > 0 {
            self.spacetime_volume() as f64 / self.total_delivered as f64
        } else {
            0.0
        }
    }
}

impl fmt::Display for FactoryPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Factory Pipeline:")?;
        writeln!(f, "  Throughput:      {:.3} states/cycle", self.throughput)?;
        writeln!(f, "  Latency:         {} cycles", self.latency)?;
        if self.warmup_cycles > 0 {
            writeln!(
                f,
                "  Warm-up:         {} cycles ({})",
                self.warmup_cycles,
                if self.warmup_complete {
                    "complete"
                } else {
                    "in progress"
                }
            )?;
        }
        writeln!(f, "  Factory qubits:  {}", self.factory_qubits)?;
        writeln!(f, "  Current cycle:   {}", self.current_cycle)?;
        writeln!(f, "  Available:       {}", self.available)?;
        writeln!(f, "  Pending:         {}", self.pending_requests)?;
        writeln!(f, "  Total delivered: {}", self.total_delivered)?;
        writeln!(
            f,
            "  Spacetime vol:   {} qubit-cycles",
            self.spacetime_volume()
        )?;
        writeln!(f, "  Total stalls:    {} cycles", self.total_stall_cycles)?;
        Ok(())
    }
}

/// Report from backpressure analysis.
#[derive(Clone, Debug)]
pub struct BackpressureReport {
    /// Total magic states demanded
    pub total_demand: usize,
    /// Total states actually delivered
    pub total_delivered: usize,
    /// Demand that couldn't be met
    pub unmet_demand: usize,
    /// Total cycles lost to stalls
    pub total_stall_cycles: usize,
    /// Number of bins with stalls
    pub stall_bins: usize,
    /// Total number of bins analyzed
    pub total_bins: usize,
    /// Could factory sustain demand?
    pub sustained: bool,
    /// Effective utilization (delivered/demand)
    pub utilization: f64,
    /// Recommended throughput to avoid stalls
    pub recommended_throughput: f64,
    /// Factory spacetime volume (qubit-cycles consumed)
    pub spacetime_volume: usize,
}

impl fmt::Display for BackpressureReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Backpressure Report:")?;
        writeln!(f, "  Demand:          {} states", self.total_demand)?;
        writeln!(f, "  Delivered:       {} states", self.total_delivered)?;
        writeln!(f, "  Unmet:           {} states", self.unmet_demand)?;
        writeln!(f, "  Stall cycles:    {}", self.total_stall_cycles)?;
        writeln!(
            f,
            "  Stall bins:      {}/{}",
            self.stall_bins, self.total_bins
        )?;
        writeln!(
            f,
            "  Sustained:       {}",
            if self.sustained { "Yes" } else { "NO" }
        )?;
        writeln!(f, "  Utilization:     {:.1}%", self.utilization * 100.0)?;
        writeln!(
            f,
            "  Spacetime vol:   {} qubit-cycles",
            self.spacetime_volume
        )?;
        if !self.sustained {
            writeln!(
                f,
                "  Recommended:     {:.3} states/cycle",
                self.recommended_throughput
            )?;
        }
        Ok(())
    }
}

// ============================================================================
// Factory Configuration Helpers
// ============================================================================

/// Create MagicSupply with optimized factory selection.
pub fn optimized_supply(physical_error: f64, target_error: f64) -> MagicSupply {
    let factory = DistillationFactory::optimized(physical_error, target_error);
    MagicSupply::with_factory(factory, physical_error)
}

/// Create MagicSupply using 15-to-1 protocol with specified levels.
pub fn fifteen_to_one_supply(physical_error: f64, levels: usize) -> MagicSupply {
    let protocol = DistillationProtocol::fifteen_to_one();
    let protocols: Vec<_> = (0..levels).map(|_| protocol.clone()).collect();
    let factory = DistillationFactory::new(physical_error, protocols);
    MagicSupply::with_factory(factory, physical_error)
}

/// Estimate required factory configuration for a circuit.
pub fn estimate_factory_config(
    t_count: usize,
    t_depth: usize,
    physical_error: f64,
    target_error: f64,
    max_latency_cycles: usize,
) -> FactoryConfig {
    let supply = MagicSupply::new(physical_error, target_error);

    // States needed per T-layer
    let t_per_layer = if t_depth > 0 {
        (t_count + t_depth - 1) / t_depth
    } else {
        t_count
    };

    // Required throughput to meet latency target
    let required_throughput = if max_latency_cycles > 0 {
        t_count as f64 / max_latency_cycles as f64
    } else {
        t_per_layer as f64 / supply.latency() as f64
    };

    let factories_needed = supply.factories_for_rate(required_throughput);
    let total_qubits = factories_needed * supply.factory_footprint();
    let actual_throughput = factories_needed as f64 * supply.throughput();
    let actual_latency = (t_count as f64 / actual_throughput).ceil() as usize;

    FactoryConfig {
        factory_count: factories_needed,
        total_qubits,
        throughput: actual_throughput,
        latency_cycles: actual_latency,
        error_rate: supply.error_rate(),
    }
}

/// Factory configuration recommendation.
#[derive(Clone, Debug)]
pub struct FactoryConfig {
    /// Number of parallel factories
    pub factory_count: usize,
    /// Total physical qubits for factories
    pub total_qubits: usize,
    /// Aggregate throughput (states per cycle)
    pub throughput: f64,
    /// Expected latency to complete all T-gates
    pub latency_cycles: usize,
    /// Output error rate
    pub error_rate: f64,
}

impl fmt::Display for FactoryConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Factory Configuration:")?;
        writeln!(f, "  Factories:       {}", self.factory_count)?;
        writeln!(f, "  Total qubits:    {}", self.total_qubits)?;
        writeln!(f, "  Throughput:      {:.3} states/cycle", self.throughput)?;
        writeln!(f, "  Latency:         {} cycles", self.latency_cycles)?;
        writeln!(f, "  Error rate:      {:.2e}", self.error_rate)?;
        Ok(())
    }
}

// ============================================================================
// Factory Fleet (Phase 2.1b)
// ============================================================================

/// Strategy for distributing magic state requests across factories.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FleetStrategy {
    /// Round-robin: distribute requests evenly across factories
    #[default]
    RoundRobin,
    /// Least-loaded: send to factory with most available states
    LeastLoaded,
    /// Fill-first: use one factory until exhausted, then move to next
    FillFirst,
}

/// A fleet of factory pipelines for high-throughput magic state production.
///
/// FactoryFleet manages multiple FactoryPipeline instances to achieve
/// aggregate throughput higher than a single factory can provide.
///
/// # Use Cases
///
/// - **High T-count circuits**: Circuits with thousands of T-gates need
///   multiple factories running in parallel to avoid becoming factory-bound.
/// - **Heterogeneous factories**: Different factories can target different
///   error rates or have different throughput characteristics.
/// - **Redundancy**: If one factory stalls, others can pick up slack.
///
/// # Example
///
/// ```ignore
/// // Create fleet of 4 identical factories
/// let factory = FactoryPipeline::new(0.1, 100, 1000).with_warmup(100);
/// let mut fleet = FactoryFleet::homogeneous(factory, 4);
///
/// // Advance time and request states
/// fleet.tick_to(200);  // All factories warm up
/// let result = fleet.request(10);  // Distributed across factories
/// println!("Got {} now, {} pending", result.delivered, result.pending);
/// ```
#[derive(Clone, Debug)]
pub struct FactoryFleet {
    /// Individual factory pipelines
    factories: Vec<FactoryPipeline>,
    /// Distribution strategy
    strategy: FleetStrategy,
    /// Next factory index for round-robin
    next_factory: usize,
    /// Total states delivered across all factories
    total_delivered: usize,
    /// Total stall cycles (max across concurrent requests)
    total_stall_cycles: usize,
}

impl FactoryFleet {
    /// Create a fleet from existing pipelines.
    pub fn new(factories: Vec<FactoryPipeline>) -> Self {
        Self {
            factories,
            strategy: FleetStrategy::default(),
            next_factory: 0,
            total_delivered: 0,
            total_stall_cycles: 0,
        }
    }

    /// Create a homogeneous fleet (N copies of the same factory).
    pub fn homogeneous(template: FactoryPipeline, count: usize) -> Self {
        let factories = (0..count).map(|_| template.clone()).collect();
        Self::new(factories)
    }

    /// Create a fleet from a MagicSupply with specified factory count.
    pub fn from_supply(supply: &MagicSupply, count: usize) -> Self {
        let template = FactoryPipeline::from_supply(supply);
        Self::homogeneous(template, count)
    }

    /// Set the distribution strategy.
    pub fn with_strategy(mut self, strategy: FleetStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Number of factories in the fleet.
    pub fn factory_count(&self) -> usize {
        self.factories.len()
    }

    /// Aggregate throughput (sum of all factories).
    pub fn aggregate_throughput(&self) -> f64 {
        self.factories.iter().map(|f| f.throughput()).sum()
    }

    /// Total physical qubits across all factories.
    pub fn total_qubits(&self) -> usize {
        self.factories.iter().map(|f| f.factory_qubits()).sum()
    }

    /// Maximum latency across all factories.
    pub fn max_latency(&self) -> usize {
        self.factories
            .iter()
            .map(|f| f.latency())
            .max()
            .unwrap_or(0)
    }

    /// Check if all factories are warmed up.
    pub fn all_warmed_up(&self) -> bool {
        self.factories.iter().all(|f| f.is_warmed_up())
    }

    /// Number of factories currently warmed up.
    pub fn warmed_up_count(&self) -> usize {
        self.factories.iter().filter(|f| f.is_warmed_up()).count()
    }

    /// Total available states across all factories.
    pub fn total_available(&self) -> usize {
        self.factories.iter().map(|f| f.available()).sum()
    }

    /// Advance all factories to a specific cycle.
    pub fn tick_to(&mut self, target_cycle: usize) {
        for factory in &mut self.factories {
            factory.tick_to(target_cycle);
        }
    }

    /// Request magic states using the configured strategy.
    ///
    /// Distributes the request across factories according to strategy.
    pub fn request(&mut self, n: usize) -> FleetRequest {
        if n == 0 {
            return FleetRequest::empty();
        }

        match self.strategy {
            FleetStrategy::RoundRobin => self.request_round_robin(n),
            FleetStrategy::LeastLoaded => self.request_least_loaded(n),
            FleetStrategy::FillFirst => self.request_fill_first(n),
        }
    }

    /// Round-robin distribution.
    fn request_round_robin(&mut self, n: usize) -> FleetRequest {
        let mut delivered = 0;
        let mut pending = 0;
        let mut max_wait = 0;
        let mut factory_requests = Vec::with_capacity(self.factories.len());

        let num_factories = self.factories.len();
        if num_factories == 0 {
            return FleetRequest {
                delivered: 0,
                pending: n,
                wait_cycles: usize::MAX,
                backpressure: true,
                factory_requests,
            };
        }

        // Distribute evenly with remainder going to first factories
        let base = n / num_factories;
        let remainder = n % num_factories;

        for (i, factory) in self.factories.iter_mut().enumerate() {
            let request_count = base + if i < remainder { 1 } else { 0 };
            if request_count > 0 {
                let result = factory.request(request_count);
                delivered += result.delivered;
                pending += result.pending;
                max_wait = max_wait.max(result.wait_cycles);
                factory_requests.push((i, result));
            }
        }

        self.total_delivered += delivered;
        if max_wait > 0 {
            self.total_stall_cycles += max_wait;
        }

        FleetRequest {
            delivered,
            pending,
            wait_cycles: max_wait,
            backpressure: pending > 0,
            factory_requests,
        }
    }

    /// Least-loaded distribution (greedy fill from most available).
    fn request_least_loaded(&mut self, n: usize) -> FleetRequest {
        let mut delivered = 0;
        let mut pending = n;
        let mut max_wait = 0;
        let mut factory_requests = Vec::new();

        // Sort factories by available (descending)
        let mut indices: Vec<_> = (0..self.factories.len()).collect();
        indices.sort_by_key(|&i| std::cmp::Reverse(self.factories[i].available()));

        for &i in &indices {
            if pending == 0 {
                break;
            }
            let available = self.factories[i].available();
            let take = pending.min(available);
            if take > 0 {
                let result = self.factories[i].request(take);
                delivered += result.delivered;
                pending -= result.delivered;
                max_wait = max_wait.max(result.wait_cycles);
                factory_requests.push((i, result));
            }
        }

        // If still pending, distribute remaining across factories
        if pending > 0 {
            for &i in &indices {
                if pending == 0 {
                    break;
                }
                let result = self.factories[i].request(pending);
                delivered += result.delivered;
                pending = result.pending;
                max_wait = max_wait.max(result.wait_cycles);
                factory_requests.push((i, result));
            }
        }

        self.total_delivered += delivered;
        if max_wait > 0 {
            self.total_stall_cycles += max_wait;
        }

        FleetRequest {
            delivered,
            pending,
            wait_cycles: max_wait,
            backpressure: pending > 0,
            factory_requests,
        }
    }

    /// Fill-first distribution (use one factory until exhausted).
    fn request_fill_first(&mut self, n: usize) -> FleetRequest {
        let mut delivered = 0;
        let mut pending = n;
        let mut max_wait = 0;
        let mut factory_requests = Vec::new();

        for (i, factory) in self.factories.iter_mut().enumerate() {
            if pending == 0 {
                break;
            }
            let result = factory.request(pending);
            delivered += result.delivered;
            pending = result.pending;
            max_wait = max_wait.max(result.wait_cycles);
            factory_requests.push((i, result));
        }

        self.total_delivered += delivered;
        if max_wait > 0 {
            self.total_stall_cycles += max_wait;
        }

        FleetRequest {
            delivered,
            pending,
            wait_cycles: max_wait,
            backpressure: pending > 0,
            factory_requests,
        }
    }

    /// Request and block until all states are delivered.
    pub fn request_blocking(&mut self, n: usize) -> FleetRequest {
        let result = self.request(n);
        if result.wait_cycles > 0 && result.wait_cycles < usize::MAX {
            // Advance all factories
            let current = self
                .factories
                .first()
                .map(|f| f.current_cycle())
                .unwrap_or(0);
            self.tick_to(current + result.wait_cycles);
        }
        result
    }

    /// Analyze demand timeline across the fleet.
    pub fn analyze_demand(
        &self,
        demand_bins: &[super::ft_types::DemandBin],
    ) -> FleetBackpressureReport {
        let mut fleet_sim = self.clone();
        fleet_sim.reset();

        let mut total_demand = 0;
        let mut unmet_demand = 0;
        let mut stall_bins = 0;

        for bin in demand_bins {
            fleet_sim.tick_to(bin.start_cycle);

            let demand = bin.magic_states;
            total_demand += demand;

            let result = fleet_sim.request(demand);
            if result.backpressure {
                unmet_demand += result.pending;
                stall_bins += 1;
            }

            fleet_sim.tick_to(bin.end_cycle);
        }

        let sustained = unmet_demand == 0;
        let utilization = if total_demand > 0 {
            fleet_sim.total_delivered as f64 / total_demand as f64
        } else {
            1.0
        };

        FleetBackpressureReport {
            total_demand,
            total_delivered: fleet_sim.total_delivered,
            unmet_demand,
            total_stall_cycles: fleet_sim.total_stall_cycles,
            stall_bins,
            total_bins: demand_bins.len(),
            sustained,
            utilization,
            factory_count: self.factories.len(),
            aggregate_throughput: self.aggregate_throughput(),
            spacetime_volume: fleet_sim.total_spacetime_volume(),
        }
    }

    /// Reset all factories in the fleet.
    pub fn reset(&mut self) {
        for factory in &mut self.factories {
            factory.reset();
        }
        self.total_delivered = 0;
        self.total_stall_cycles = 0;
        self.next_factory = 0;
    }

    /// Get reference to individual factories.
    pub fn factories(&self) -> &[FactoryPipeline] {
        &self.factories
    }

    /// Get mutable reference to individual factories.
    pub fn factories_mut(&mut self) -> &mut [FactoryPipeline] {
        &mut self.factories
    }

    /// Total states delivered by fleet.
    pub fn total_delivered(&self) -> usize {
        self.total_delivered
    }

    /// Total stall cycles across all requests.
    pub fn total_stall_cycles(&self) -> usize {
        self.total_stall_cycles
    }

    /// Total spacetime volume across all factories.
    ///
    /// Sum of (factory_qubits × current_cycle) for each factory.
    pub fn total_spacetime_volume(&self) -> usize {
        self.factories.iter().map(|f| f.spacetime_volume()).sum()
    }

    /// Average spacetime cost per magic state delivered.
    pub fn spacetime_per_state(&self) -> f64 {
        if self.total_delivered > 0 {
            self.total_spacetime_volume() as f64 / self.total_delivered as f64
        } else {
            0.0
        }
    }
}

impl fmt::Display for FactoryFleet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Factory Fleet:")?;
        writeln!(f, "  Factories:       {}", self.factories.len())?;
        writeln!(f, "  Strategy:        {:?}", self.strategy)?;
        writeln!(
            f,
            "  Throughput:      {:.3} states/cycle (aggregate)",
            self.aggregate_throughput()
        )?;
        writeln!(f, "  Total qubits:    {}", self.total_qubits())?;
        writeln!(
            f,
            "  Warmed up:       {}/{}",
            self.warmed_up_count(),
            self.factories.len()
        )?;
        writeln!(f, "  Available:       {}", self.total_available())?;
        writeln!(f, "  Total delivered: {}", self.total_delivered)?;
        writeln!(
            f,
            "  Spacetime vol:   {} qubit-cycles",
            self.total_spacetime_volume()
        )?;
        writeln!(f, "  Total stalls:    {} cycles", self.total_stall_cycles)?;
        Ok(())
    }
}

/// Result of requesting magic states from a fleet.
#[derive(Clone, Debug)]
pub struct FleetRequest {
    /// Total states delivered immediately
    pub delivered: usize,
    /// Total states still pending
    pub pending: usize,
    /// Maximum wait cycles (across all factories)
    pub wait_cycles: usize,
    /// Was there backpressure?
    pub backpressure: bool,
    /// Individual factory requests (factory_index, result)
    pub factory_requests: Vec<(usize, PipelineRequest)>,
}

impl FleetRequest {
    /// Create an empty request result.
    pub fn empty() -> Self {
        Self {
            delivered: 0,
            pending: 0,
            wait_cycles: 0,
            backpressure: false,
            factory_requests: Vec::new(),
        }
    }
}

/// Backpressure report for a factory fleet.
#[derive(Clone, Debug)]
pub struct FleetBackpressureReport {
    /// Total magic states demanded
    pub total_demand: usize,
    /// Total states actually delivered
    pub total_delivered: usize,
    /// Demand that couldn't be met
    pub unmet_demand: usize,
    /// Total cycles lost to stalls
    pub total_stall_cycles: usize,
    /// Number of bins with stalls
    pub stall_bins: usize,
    /// Total number of bins analyzed
    pub total_bins: usize,
    /// Could fleet sustain demand?
    pub sustained: bool,
    /// Effective utilization (delivered/demand)
    pub utilization: f64,
    /// Number of factories in fleet
    pub factory_count: usize,
    /// Aggregate throughput
    pub aggregate_throughput: f64,
    /// Total spacetime volume across all factories (qubit-cycles)
    pub spacetime_volume: usize,
}

impl fmt::Display for FleetBackpressureReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Fleet Backpressure Report:")?;
        writeln!(f, "  Demand:          {} states", self.total_demand)?;
        writeln!(f, "  Delivered:       {} states", self.total_delivered)?;
        writeln!(f, "  Unmet:           {} states", self.unmet_demand)?;
        writeln!(f, "  Stall cycles:    {}", self.total_stall_cycles)?;
        writeln!(
            f,
            "  Stall bins:      {}/{}",
            self.stall_bins, self.total_bins
        )?;
        writeln!(
            f,
            "  Sustained:       {}",
            if self.sustained { "Yes" } else { "NO" }
        )?;
        writeln!(f, "  Utilization:     {:.1}%", self.utilization * 100.0)?;
        writeln!(
            f,
            "  Spacetime vol:   {} qubit-cycles",
            self.spacetime_volume
        )?;
        writeln!(f, "  Factories:       {}", self.factory_count)?;
        writeln!(
            f,
            "  Agg. throughput: {:.3} states/cycle",
            self.aggregate_throughput
        )?;
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_supply_creation() {
        let supply = MagicSupply::new(1e-3, 1e-10);
        assert!(supply.error_rate() <= 1e-10);
        assert!(supply.throughput() > 0.0);
        assert!(supply.factory_footprint() > 0);
    }

    #[test]
    fn test_consume_from_inventory() {
        let mut supply = MagicSupply::new(1e-3, 1e-10).with_inventory(5);
        assert_eq!(supply.inventory(), 5);

        let result = supply.consume();
        assert!(result.from_inventory);
        assert_eq!(result.delay_cycles, 0);
        assert_eq!(supply.inventory(), 4);
    }

    #[test]
    fn test_consume_with_production() {
        let mut supply = MagicSupply::new(1e-3, 1e-10);
        assert_eq!(supply.inventory(), 0);

        let result = supply.consume();
        assert!(!result.from_inventory);
        assert!(result.delay_cycles > 0);
        assert_eq!(result.delay_cycles, supply.latency());
    }

    #[test]
    fn test_toffoli_consumption() {
        let mut supply = MagicSupply::new(1e-3, 1e-10);
        let results = supply.consume_toffoli();
        assert_eq!(results.len(), 7);
        assert_eq!(supply.total_consumed(), 7);
    }

    #[test]
    fn test_cost_estimation() {
        let supply = MagicSupply::new(1e-3, 1e-10);
        let cost = supply.estimate_cost(100, 10);

        assert_eq!(cost.magic_states, 100);
        assert!(cost.factory_qubits > 0);
        assert!(cost.factory_cycles > 0);
    }

    #[test]
    fn test_throughput_scaling() {
        let supply1 = MagicSupply::new(1e-3, 1e-10);
        let supply2 = MagicSupply::new(1e-3, 1e-10).with_throughput(2.0 * supply1.throughput());

        // Higher throughput needs more qubits
        assert!(supply2.factory_footprint() >= supply1.factory_footprint());
        assert!(supply2.throughput() >= supply1.throughput());
    }

    #[test]
    fn test_factory_config_estimation() {
        let config = estimate_factory_config(
            1000,  // T-count
            50,    // T-depth
            1e-3,  // Physical error
            1e-10, // Target error
            100,   // Max latency
        );

        assert!(config.factory_count > 0);
        assert!(config.total_qubits > 0);
        assert!(config.latency_cycles <= 100 || config.factory_count > 10);
    }

    // =========================================================================
    // Factory Pipeline Tests (Phase 2.1)
    // =========================================================================

    #[test]
    fn test_pipeline_creation() {
        let pipeline = FactoryPipeline::new(0.1, 100, 1000);
        assert_eq!(pipeline.throughput(), 0.1);
        assert_eq!(pipeline.latency(), 100);
        assert_eq!(pipeline.factory_qubits(), 1000);
        assert_eq!(pipeline.current_cycle(), 0);
        assert_eq!(pipeline.available(), 0);
    }

    #[test]
    fn test_pipeline_production() {
        let mut pipeline = FactoryPipeline::new(0.1, 100, 1000);

        // After 10 cycles at 0.1/cycle, should have 1 state
        pipeline.tick_to(10);
        assert_eq!(pipeline.available(), 1);

        // After 20 more cycles (30 total), should have 3 states
        pipeline.tick_to(30);
        assert_eq!(pipeline.available(), 3);
    }

    #[test]
    fn test_pipeline_request_available() {
        let mut pipeline = FactoryPipeline::new(0.1, 100, 1000).with_initial_buffer(5);

        let result = pipeline.request(3);
        assert_eq!(result.delivered, 3);
        assert_eq!(result.pending, 0);
        assert_eq!(result.wait_cycles, 0);
        assert!(!result.backpressure);
        assert_eq!(pipeline.available(), 2);
    }

    #[test]
    fn test_pipeline_request_backpressure() {
        let mut pipeline = FactoryPipeline::new(0.1, 100, 1000);

        // Request more than available
        let result = pipeline.request(5);
        assert_eq!(result.delivered, 0);
        assert_eq!(result.pending, 5);
        assert!(result.wait_cycles >= 50); // 5 / 0.1 = 50 cycles minimum
        assert!(result.backpressure);
    }

    #[test]
    fn test_pipeline_request_blocking() {
        let mut pipeline = FactoryPipeline::new(0.1, 100, 1000);

        // Blocking request advances time
        let result = pipeline.request_blocking(3);
        assert!(result.backpressure);

        // After blocking, we should have advanced enough
        assert!(pipeline.current_cycle() >= 30);
        assert_eq!(pipeline.total_delivered(), 3);
    }

    #[test]
    fn test_pipeline_stall_tracking() {
        let mut pipeline = FactoryPipeline::new(0.1, 10, 1000);

        // Request that causes stall
        pipeline.request(5);

        assert_eq!(pipeline.stall_events().len(), 1);
        assert_eq!(pipeline.stall_events()[0].requested, 5);
        assert!(pipeline.total_stall_cycles() > 0);
    }

    #[test]
    fn test_pipeline_demand_analysis_sustained() {
        use super::super::ft_types::DemandBin;

        // Create a pipeline that can sustain the demand, with warmup buffer
        let pipeline = FactoryPipeline::new(1.0, 10, 1000) // 1 state/cycle
            .with_initial_buffer(3); // Pre-filled buffer for warmup

        // Light demand: 1 state every 10 cycles
        let demand = vec![
            DemandBin {
                start_cycle: 0,
                end_cycle: 10,
                magic_states: 1,
                t_gates: 1,
            },
            DemandBin {
                start_cycle: 10,
                end_cycle: 20,
                magic_states: 1,
                t_gates: 1,
            },
            DemandBin {
                start_cycle: 20,
                end_cycle: 30,
                magic_states: 1,
                t_gates: 1,
            },
        ];

        let report = pipeline.analyze_demand(&demand);
        assert!(report.sustained, "Should sustain light demand");
        assert_eq!(report.unmet_demand, 0);
        assert_eq!(report.stall_bins, 0);
    }

    #[test]
    fn test_pipeline_demand_analysis_unsustained() {
        use super::super::ft_types::DemandBin;

        // Create a slow pipeline
        let pipeline = FactoryPipeline::new(0.1, 10, 1000); // 0.1 states/cycle

        // Heavy demand: 10 states in 10 cycles (needs 1.0/cycle throughput)
        let demand = vec![DemandBin {
            start_cycle: 0,
            end_cycle: 10,
            magic_states: 10,
            t_gates: 10,
        }];

        let report = pipeline.analyze_demand(&demand);
        assert!(!report.sustained, "Should NOT sustain heavy demand");
        assert!(report.unmet_demand > 0);
        assert!(report.recommended_throughput > pipeline.throughput());
    }

    #[test]
    fn test_pipeline_from_supply() {
        let supply = MagicSupply::new(1e-3, 1e-10);
        let pipeline = FactoryPipeline::from_supply(&supply);

        assert_eq!(pipeline.throughput(), supply.throughput());
        assert_eq!(pipeline.latency(), supply.latency());
        assert_eq!(pipeline.factory_qubits(), supply.factory_footprint());
    }

    #[test]
    fn test_pipeline_reset() {
        let mut pipeline = FactoryPipeline::new(0.1, 10, 1000);

        pipeline.tick_to(100);
        pipeline.request(5);

        assert!(pipeline.current_cycle() > 0);
        assert!(pipeline.total_delivered() > 0 || pipeline.pending_requests() > 0);

        pipeline.reset();

        assert_eq!(pipeline.current_cycle(), 0);
        assert_eq!(pipeline.total_delivered(), 0);
        assert_eq!(pipeline.pending_requests(), 0);
        assert_eq!(pipeline.available(), 0);
        assert!(pipeline.stall_events().is_empty());
    }

    // =========================================================================
    // Warm-up Phase Tests (Phase 2.1a)
    // =========================================================================

    #[test]
    fn test_warmup_no_production_during_warmup() {
        let mut pipeline = FactoryPipeline::new(1.0, 10, 1000).with_warmup(50);

        assert!(!pipeline.is_warmed_up());
        assert_eq!(pipeline.warmup_remaining(), 50);

        // Tick to 30 cycles - still in warm-up, no production
        pipeline.tick_to(30);
        assert_eq!(pipeline.available(), 0);
        assert!(!pipeline.is_warmed_up());
        assert_eq!(pipeline.warmup_remaining(), 20);
    }

    #[test]
    fn test_warmup_production_starts_after_warmup() {
        let mut pipeline = FactoryPipeline::new(1.0, 10, 1000).with_warmup(50);

        // Tick past warm-up (50) + some production time
        pipeline.tick_to(60);

        assert!(pipeline.is_warmed_up());
        assert_eq!(pipeline.warmup_remaining(), 0);
        // 10 cycles of production at 1.0/cycle = 10 states
        assert_eq!(pipeline.available(), 10);
    }

    #[test]
    fn test_warmup_request_includes_warmup_wait() {
        let mut pipeline = FactoryPipeline::new(0.5, 10, 1000).with_warmup(100);

        // Request states at cycle 0 - must wait for warmup + production
        let result = pipeline.request(5);

        assert!(result.backpressure);
        // Wait should include warmup (100) + production time for 5 states (10 cycles at 0.5/cycle)
        assert!(result.wait_cycles >= 100);
    }

    #[test]
    fn test_warmup_completed_by_initial_buffer() {
        let pipeline = FactoryPipeline::new(1.0, 10, 1000)
            .with_warmup(50)
            .with_initial_buffer(5);

        // Initial buffer marks warmup as complete
        assert!(pipeline.is_warmed_up());
        assert_eq!(pipeline.warmup_remaining(), 0);
        assert_eq!(pipeline.available(), 5);
    }

    #[test]
    fn test_warmup_reset_restores_warmup_state() {
        let mut pipeline = FactoryPipeline::new(1.0, 10, 1000).with_warmup(50);

        pipeline.tick_to(100);
        assert!(pipeline.is_warmed_up());

        pipeline.reset();
        assert!(!pipeline.is_warmed_up());
        assert_eq!(pipeline.warmup_remaining(), 50);
    }

    #[test]
    fn test_warmup_counters_only_preserves_warmup() {
        let mut pipeline = FactoryPipeline::new(1.0, 10, 1000).with_warmup(50);

        pipeline.tick_to(100);
        pipeline.request(5);

        let was_warmed_up = pipeline.is_warmed_up();
        let cycle = pipeline.current_cycle();
        let available = pipeline.available();

        pipeline.reset_counters_only();

        // Counters reset
        assert_eq!(pipeline.total_delivered(), 0);
        assert_eq!(pipeline.pending_requests(), 0);
        assert!(pipeline.stall_events().is_empty());

        // Warmup state and production preserved
        assert_eq!(pipeline.is_warmed_up(), was_warmed_up);
        assert_eq!(pipeline.current_cycle(), cycle);
        assert_eq!(pipeline.available(), available);
    }

    #[test]
    fn test_warmup_zero_means_instant_start() {
        let pipeline = FactoryPipeline::new(1.0, 10, 1000).with_warmup(0);
        assert!(pipeline.is_warmed_up());
    }

    #[test]
    fn test_from_supply_sets_warmup_to_latency() {
        let supply = MagicSupply::new(1e-3, 1e-10);
        let pipeline = FactoryPipeline::from_supply(&supply);

        // from_supply sets warmup = latency (first state takes latency cycles)
        assert_eq!(pipeline.warmup_cycles(), supply.latency());
    }

    // =========================================================================
    // Factory Fleet Tests (Phase 2.1b/c)
    // =========================================================================

    #[test]
    fn test_fleet_creation_homogeneous() {
        let template = FactoryPipeline::new(0.5, 100, 1000);
        let fleet = FactoryFleet::homogeneous(template, 4);

        assert_eq!(fleet.factory_count(), 4);
        assert!((fleet.aggregate_throughput() - 2.0).abs() < 0.001); // 4 * 0.5
        assert_eq!(fleet.total_qubits(), 4000); // 4 * 1000
        assert_eq!(fleet.max_latency(), 100);
    }

    #[test]
    fn test_fleet_from_supply() {
        let supply = MagicSupply::new(1e-3, 1e-10);
        let fleet = FactoryFleet::from_supply(&supply, 3);

        assert_eq!(fleet.factory_count(), 3);
        assert!((fleet.aggregate_throughput() - 3.0 * supply.throughput()).abs() < 0.001);
    }

    #[test]
    fn test_fleet_tick_advances_all_factories() {
        let template = FactoryPipeline::new(1.0, 10, 1000);
        let mut fleet = FactoryFleet::homogeneous(template, 4);

        fleet.tick_to(50);

        // Each factory should have produced 50 states
        assert_eq!(fleet.total_available(), 200); // 4 * 50
        for factory in fleet.factories() {
            assert_eq!(factory.current_cycle(), 50);
            assert_eq!(factory.available(), 50);
        }
    }

    #[test]
    fn test_fleet_round_robin_distribution() {
        let template = FactoryPipeline::new(1.0, 10, 1000).with_initial_buffer(10);
        let mut fleet =
            FactoryFleet::homogeneous(template, 4).with_strategy(FleetStrategy::RoundRobin);

        // Request 12 states: should be 3,3,3,3 across factories
        let result = fleet.request(12);

        assert_eq!(result.delivered, 12);
        assert!(!result.backpressure);
        assert_eq!(result.factory_requests.len(), 4);

        // Each factory should have delivered 3
        for (_, req) in &result.factory_requests {
            assert_eq!(req.delivered, 3);
        }
    }

    #[test]
    fn test_fleet_least_loaded_distribution() {
        // Create factories with different initial buffers
        let factories = vec![
            FactoryPipeline::new(1.0, 10, 1000).with_initial_buffer(5),
            FactoryPipeline::new(1.0, 10, 1000).with_initial_buffer(10),
            FactoryPipeline::new(1.0, 10, 1000).with_initial_buffer(3),
        ];

        let mut fleet = FactoryFleet::new(factories).with_strategy(FleetStrategy::LeastLoaded);

        // Request 10 states: should take from most available first
        let result = fleet.request(10);

        assert_eq!(result.delivered, 10);
        assert!(!result.backpressure);

        // Factory 1 (10 available) should have been drained first
        // Exact distribution depends on algorithm, but total should be 10
    }

    #[test]
    fn test_fleet_fill_first_distribution() {
        let template = FactoryPipeline::new(1.0, 10, 1000).with_initial_buffer(5);
        let mut fleet =
            FactoryFleet::homogeneous(template, 4).with_strategy(FleetStrategy::FillFirst);

        // Request 12 states: should drain first factories
        let result = fleet.request(12);

        assert_eq!(result.delivered, 12);
        assert!(!result.backpressure);

        // First 3 factories should be drained (5+5+5=15 available, need 12)
        assert_eq!(fleet.factories()[0].available(), 0);
        assert_eq!(fleet.factories()[1].available(), 0);
        assert_eq!(fleet.factories()[2].available(), 3); // 5-2 remaining
        assert_eq!(fleet.factories()[3].available(), 5); // untouched
    }

    #[test]
    fn test_fleet_backpressure() {
        let template = FactoryPipeline::new(0.1, 10, 1000); // No buffer, slow
        let mut fleet = FactoryFleet::homogeneous(template, 2);

        // Request more than available
        let result = fleet.request(10);

        assert_eq!(result.delivered, 0);
        assert_eq!(result.pending, 10);
        assert!(result.backpressure);
        assert!(result.wait_cycles > 0);
    }

    #[test]
    fn test_fleet_warmup_tracking() {
        let template = FactoryPipeline::new(1.0, 10, 1000).with_warmup(50);
        let mut fleet = FactoryFleet::homogeneous(template, 4);

        assert_eq!(fleet.warmed_up_count(), 0);
        assert!(!fleet.all_warmed_up());

        fleet.tick_to(30);
        assert_eq!(fleet.warmed_up_count(), 0); // Still warming up

        fleet.tick_to(60);
        assert_eq!(fleet.warmed_up_count(), 4);
        assert!(fleet.all_warmed_up());
    }

    #[test]
    fn test_fleet_demand_analysis_sustained() {
        use super::super::ft_types::DemandBin;

        // Fleet with high aggregate throughput (no initial buffer needed if demand comes later)
        // 4 factories * 1.0 states/cycle = 4 states/cycle aggregate
        let template = FactoryPipeline::new(1.0, 10, 1000);
        let fleet = FactoryFleet::homogeneous(template, 4);

        // Light demand starting after time for production to accumulate
        // After 50 cycles, each factory has 50 states, total 200
        let demand = vec![
            DemandBin {
                start_cycle: 50,
                end_cycle: 60,
                magic_states: 5,
                t_gates: 5,
            },
            DemandBin {
                start_cycle: 60,
                end_cycle: 70,
                magic_states: 5,
                t_gates: 5,
            },
        ];

        let report = fleet.analyze_demand(&demand);
        assert!(
            report.sustained,
            "Fleet should sustain light demand after warm-up"
        );
        assert_eq!(report.unmet_demand, 0);
        assert_eq!(report.factory_count, 4);
    }

    #[test]
    fn test_fleet_demand_analysis_unsustained() {
        use super::super::ft_types::DemandBin;

        // Single slow factory
        let template = FactoryPipeline::new(0.1, 10, 1000);
        let fleet = FactoryFleet::homogeneous(template, 1);

        // Heavy demand
        let demand = vec![DemandBin {
            start_cycle: 0,
            end_cycle: 5,
            magic_states: 10,
            t_gates: 10,
        }];

        let report = fleet.analyze_demand(&demand);
        assert!(!report.sustained);
        assert!(report.unmet_demand > 0);
    }

    #[test]
    fn test_fleet_reset() {
        let template = FactoryPipeline::new(1.0, 10, 1000).with_warmup(20);
        let mut fleet = FactoryFleet::homogeneous(template, 2);

        fleet.tick_to(50);
        fleet.request(10);

        assert!(fleet.total_delivered() > 0 || fleet.factories()[0].pending_requests() > 0);

        fleet.reset();

        assert_eq!(fleet.total_delivered(), 0);
        assert_eq!(fleet.total_stall_cycles(), 0);
        for factory in fleet.factories() {
            assert_eq!(factory.current_cycle(), 0);
            assert!(!factory.is_warmed_up()); // Warmup reset
        }
    }

    #[test]
    fn test_fleet_heterogeneous() {
        // Create heterogeneous fleet with different configurations
        let fast = FactoryPipeline::new(1.0, 10, 500).with_initial_buffer(10);
        let slow = FactoryPipeline::new(0.1, 100, 1000).with_initial_buffer(5);

        let fleet = FactoryFleet::new(vec![fast, slow]);

        assert_eq!(fleet.factory_count(), 2);
        assert!((fleet.aggregate_throughput() - 1.1).abs() < 0.001);
        assert_eq!(fleet.total_qubits(), 1500);
        assert_eq!(fleet.max_latency(), 100);
    }

    #[test]
    fn test_fleet_empty_request() {
        let template = FactoryPipeline::new(1.0, 10, 1000);
        let mut fleet = FactoryFleet::homogeneous(template, 2);

        let result = fleet.request(0);

        assert_eq!(result.delivered, 0);
        assert_eq!(result.pending, 0);
        assert!(!result.backpressure);
        assert!(result.factory_requests.is_empty());
    }

    // =========================================================================
    // Spacetime Volume Tests (Phase 2.2)
    // =========================================================================

    #[test]
    fn test_pipeline_spacetime_volume() {
        let mut pipeline = FactoryPipeline::new(1.0, 10, 1000); // 1000 qubits

        // At cycle 0, spacetime is 0
        assert_eq!(pipeline.spacetime_volume(), 0);

        // After 50 cycles, spacetime = 1000 * 50 = 50,000
        pipeline.tick_to(50);
        assert_eq!(pipeline.spacetime_volume(), 50_000);
    }

    #[test]
    fn test_pipeline_spacetime_per_state() {
        let mut pipeline = FactoryPipeline::new(1.0, 10, 1000).with_initial_buffer(10);

        // Deliver some states
        pipeline.tick_to(50);
        pipeline.request(20);

        // Spacetime per state = spacetime_volume / total_delivered
        let delivered = pipeline.total_delivered();
        let spacetime = pipeline.spacetime_volume();
        let per_state = pipeline.spacetime_per_state();

        assert!(delivered > 0);
        assert!((per_state - (spacetime as f64 / delivered as f64)).abs() < 0.001);
    }

    #[test]
    fn test_fleet_spacetime_volume() {
        let template = FactoryPipeline::new(1.0, 10, 500); // 500 qubits each
        let mut fleet = FactoryFleet::homogeneous(template, 4); // Total: 2000 qubits

        fleet.tick_to(100);

        // Total spacetime = 4 factories * 500 qubits * 100 cycles = 200,000
        assert_eq!(fleet.total_spacetime_volume(), 200_000);
    }

    #[test]
    fn test_backpressure_report_has_spacetime() {
        use super::super::ft_types::DemandBin;

        let template = FactoryPipeline::new(1.0, 10, 1000);
        let pipeline = template;

        let demand = vec![DemandBin {
            start_cycle: 50,
            end_cycle: 100,
            magic_states: 5,
            t_gates: 5,
        }];

        let report = pipeline.analyze_demand(&demand);

        // Spacetime should be non-zero after simulation
        assert!(report.spacetime_volume > 0);
    }

    #[test]
    fn test_fleet_backpressure_report_has_spacetime() {
        use super::super::ft_types::DemandBin;

        let template = FactoryPipeline::new(1.0, 10, 1000);
        let fleet = FactoryFleet::homogeneous(template, 2);

        let demand = vec![DemandBin {
            start_cycle: 50,
            end_cycle: 100,
            magic_states: 5,
            t_gates: 5,
        }];

        let report = fleet.analyze_demand(&demand);

        // Fleet spacetime should be non-zero after simulation
        assert!(report.spacetime_volume > 0);
    }
}
