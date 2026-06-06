//! EPIC 101.5: Investigating the LSTM Advantage
//!
//! Experiments to determine WHY LSTM outperformed SimpleBrain:
//! - Experiment 1: 50 seeds for statistical power
//! - Experiment 2: DeepSimpleBrain (parameter-matched control)
//! - Experiment 2.5: GatedFeedforward (gating without recurrence)
//! - Experiment 3: StatelessLstm (ablated memory)
//!
//! Run with:
//! ```
//! cargo run --release --example epic101_5_experiment --features burn-compute
//! ```

#[cfg(feature = "burn-compute")]
mod experiment {
    use engine::epic101::{
        DelayedRewardConfig, DelayedRewardTask, Evolution, EvolutionConfig, StationaryConfig,
        StationaryTask, Task,
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // PRNG Utility
    // ═══════════════════════════════════════════════════════════════════════════

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0
        }

        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 33) as f32 / u32::MAX as f32
        }

        fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
            min + self.next_f32() * (max - min)
        }

        fn next_usize(&mut self, max: usize) -> usize {
            (self.next_u64() % max as u64) as usize
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Brain Type 1: SimpleBrain (baseline, ~213 params)
    // ═══════════════════════════════════════════════════════════════════════════

    #[derive(Clone)]
    struct SimpleGenome {
        weights: Vec<f32>,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    }

    impl SimpleGenome {
        fn new_random(
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
            rng: &mut Rng,
        ) -> Self {
            let total =
                input_size * hidden_size + hidden_size + hidden_size * output_size + output_size;
            let weights: Vec<f32> = (0..total).map(|_| rng.next_f32_range(-0.5, 0.5)).collect();
            Self {
                weights,
                input_size,
                hidden_size,
                output_size,
            }
        }

        fn param_count(&self) -> usize {
            self.input_size * self.hidden_size
                + self.hidden_size
                + self.hidden_size * self.output_size
                + self.output_size
        }

        fn forward(&self, input: &[f32]) -> usize {
            let h = self.hidden_size;
            let o = self.output_size;
            let i = self.input_size;

            let w1_end = i * h;
            let b1_end = w1_end + h;
            let w2_end = b1_end + h * o;

            let mut hidden = vec![0.0f32; h];
            for hi in 0..h {
                let mut sum = self.weights[w1_end + hi];
                for ii in 0..i {
                    sum += input[ii] * self.weights[ii * h + hi];
                }
                hidden[hi] = sum.tanh();
            }

            let mut output = vec![0.0f32; o];
            for oi in 0..o {
                let mut sum = self.weights[w2_end + oi];
                for hi in 0..h {
                    sum += hidden[hi] * self.weights[b1_end + hi * o + oi];
                }
                output[oi] = sum;
            }

            output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }

        fn mutate(&mut self, rate: f32, magnitude: f32, rng: &mut Rng) {
            for w in &mut self.weights {
                if rng.next_f32() < rate {
                    *w += rng.next_f32_range(-magnitude, magnitude);
                }
            }
        }

        fn crossover(a: &Self, b: &Self, rng: &mut Rng) -> Self {
            let weights = a
                .weights
                .iter()
                .zip(b.weights.iter())
                .map(|(wa, wb)| if rng.next_f32() < 0.5 { *wa } else { *wb })
                .collect();
            Self {
                weights,
                input_size: a.input_size,
                hidden_size: a.hidden_size,
                output_size: a.output_size,
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Brain Type 2: DeepSimpleBrain (parameter-matched, ~1500 params)
    // 3 layers: input→32→32→output
    // ═══════════════════════════════════════════════════════════════════════════

    #[derive(Clone)]
    struct DeepSimpleGenome {
        weights: Vec<f32>,
        input_size: usize,
        h1_size: usize, // 32
        h2_size: usize, // 32
        output_size: usize,
    }

    impl DeepSimpleGenome {
        fn new_random(input_size: usize, output_size: usize, rng: &mut Rng) -> Self {
            let h1 = 32;
            let h2 = 32;
            // Layer 1: i*h1 + h1, Layer 2: h1*h2 + h2, Layer 3: h2*o + o
            let total = input_size * h1 + h1 + h1 * h2 + h2 + h2 * output_size + output_size;
            let weights: Vec<f32> = (0..total).map(|_| rng.next_f32_range(-0.4, 0.4)).collect();
            Self {
                weights,
                input_size,
                h1_size: h1,
                h2_size: h2,
                output_size,
            }
        }

        fn param_count(&self) -> usize {
            self.input_size * self.h1_size
                + self.h1_size
                + self.h1_size * self.h2_size
                + self.h2_size
                + self.h2_size * self.output_size
                + self.output_size
        }

        fn forward(&self, input: &[f32]) -> usize {
            let i = self.input_size;
            let h1 = self.h1_size;
            let h2 = self.h2_size;
            let o = self.output_size;

            // Weight offsets
            let w1_end = i * h1;
            let b1_end = w1_end + h1;
            let w2_end = b1_end + h1 * h2;
            let b2_end = w2_end + h2;
            let w3_end = b2_end + h2 * o;

            // Layer 1
            let mut hidden1 = vec![0.0f32; h1];
            for hi in 0..h1 {
                let mut sum = self.weights[w1_end + hi];
                for ii in 0..i {
                    sum += input[ii] * self.weights[ii * h1 + hi];
                }
                hidden1[hi] = sum.tanh();
            }

            // Layer 2
            let mut hidden2 = vec![0.0f32; h2];
            for hi in 0..h2 {
                let mut sum = self.weights[b2_end - h2 + hi]; // b2
                for h1i in 0..h1 {
                    sum += hidden1[h1i] * self.weights[b1_end + h1i * h2 + hi];
                }
                hidden2[hi] = sum.tanh();
            }

            // Layer 3
            let mut output = vec![0.0f32; o];
            for oi in 0..o {
                let mut sum = self.weights[w3_end + oi]; // b3
                for h2i in 0..h2 {
                    sum += hidden2[h2i] * self.weights[b2_end + h2i * o + oi];
                }
                output[oi] = sum;
            }

            output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }

        fn mutate(&mut self, rate: f32, magnitude: f32, rng: &mut Rng) {
            for w in &mut self.weights {
                if rng.next_f32() < rate {
                    *w += rng.next_f32_range(-magnitude, magnitude);
                }
            }
        }

        fn crossover(a: &Self, b: &Self, rng: &mut Rng) -> Self {
            let weights = a
                .weights
                .iter()
                .zip(b.weights.iter())
                .map(|(wa, wb)| if rng.next_f32() < 0.5 { *wa } else { *wb })
                .collect();
            Self {
                weights,
                input_size: a.input_size,
                h1_size: a.h1_size,
                h2_size: a.h2_size,
                output_size: a.output_size,
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Brain Type 2.5: GatedFeedforward (gating without recurrence)
    // output = sigmoid(Wg*x) * tanh(Wh*x)
    // ═══════════════════════════════════════════════════════════════════════════

    #[derive(Clone)]
    struct GatedFeedforwardGenome {
        weights: Vec<f32>,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    }

    impl GatedFeedforwardGenome {
        fn new_random(
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
            rng: &mut Rng,
        ) -> Self {
            // For each hidden unit: gate weights (i) + gate bias + value weights (i) + value bias
            // Plus output layer
            let gate_params = input_size * hidden_size + hidden_size; // Wg, bg
            let value_params = input_size * hidden_size + hidden_size; // Wh, bh
            let output_params = hidden_size * output_size + output_size;
            let total = gate_params + value_params + output_params;

            let weights: Vec<f32> = (0..total).map(|_| rng.next_f32_range(-0.4, 0.4)).collect();
            Self {
                weights,
                input_size,
                hidden_size,
                output_size,
            }
        }

        fn param_count(&self) -> usize {
            let gate_params = self.input_size * self.hidden_size + self.hidden_size;
            let value_params = self.input_size * self.hidden_size + self.hidden_size;
            let output_params = self.hidden_size * self.output_size + self.output_size;
            gate_params + value_params + output_params
        }

        fn forward(&self, input: &[f32]) -> usize {
            let i = self.input_size;
            let h = self.hidden_size;
            let o = self.output_size;

            // Weight layout: [Wg: i*h][bg: h][Wh: i*h][bh: h][Wo: h*o][bo: o]
            let wg_end = i * h;
            let bg_end = wg_end + h;
            let wh_end = bg_end + i * h;
            let bh_end = wh_end + h;
            let wo_end = bh_end + h * o;

            // Compute gated hidden layer: gate * value
            let mut hidden = vec![0.0f32; h];
            for hi in 0..h {
                // Gate: sigmoid(Wg*x + bg)
                let mut gate_sum = self.weights[wg_end + hi]; // bg
                for ii in 0..i {
                    gate_sum += input[ii] * self.weights[ii * h + hi];
                }
                let gate = sigmoid(gate_sum);

                // Value: tanh(Wh*x + bh)
                let mut value_sum = self.weights[bh_end - h + hi]; // bh
                for ii in 0..i {
                    value_sum += input[ii] * self.weights[bg_end + ii * h + hi];
                }
                let value = value_sum.tanh();

                hidden[hi] = gate * value;
            }

            // Output layer
            let mut output = vec![0.0f32; o];
            for oi in 0..o {
                let mut sum = self.weights[wo_end + oi]; // bo
                for hi in 0..h {
                    sum += hidden[hi] * self.weights[bh_end + hi * o + oi];
                }
                output[oi] = sum;
            }

            output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }

        fn mutate(&mut self, rate: f32, magnitude: f32, rng: &mut Rng) {
            for w in &mut self.weights {
                if rng.next_f32() < rate {
                    *w += rng.next_f32_range(-magnitude, magnitude);
                }
            }
        }

        fn crossover(a: &Self, b: &Self, rng: &mut Rng) -> Self {
            let weights = a
                .weights
                .iter()
                .zip(b.weights.iter())
                .map(|(wa, wb)| if rng.next_f32() < 0.5 { *wa } else { *wb })
                .collect();
            Self {
                weights,
                input_size: a.input_size,
                hidden_size: a.hidden_size,
                output_size: a.output_size,
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Brain Type 3: StatelessLstm (LSTM architecture, but state reset each tick)
    // ═══════════════════════════════════════════════════════════════════════════

    #[derive(Clone)]
    struct StatelessLstmGenome {
        weights: Vec<f32>,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    }

    impl StatelessLstmGenome {
        fn new_random(
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
            rng: &mut Rng,
        ) -> Self {
            let gate_weights = input_size * hidden_size + hidden_size * hidden_size + hidden_size;
            let total = 4 * gate_weights + hidden_size * output_size + output_size;

            let mut weights: Vec<f32> = (0..total).map(|_| rng.next_f32_range(-0.3, 0.3)).collect();

            // Initialize forget gate bias to 1.0
            let f_bias_start = gate_weights + input_size * hidden_size + hidden_size * hidden_size;
            for idx in 0..hidden_size {
                weights[f_bias_start + idx] = 1.0;
            }

            Self {
                weights,
                input_size,
                hidden_size,
                output_size,
            }
        }

        fn param_count(&self) -> usize {
            let gate_weights = self.input_size * self.hidden_size
                + self.hidden_size * self.hidden_size
                + self.hidden_size;
            4 * gate_weights + self.hidden_size * self.output_size + self.output_size
        }

        fn forward(&self, input: &[f32]) -> usize {
            let i = self.input_size;
            let h = self.hidden_size;
            let o = self.output_size;

            // LSTM with ZERO initial state (no memory carry!)
            let mut hidden = vec![0.0f32; h];
            let mut cell = vec![0.0f32; h];

            let gate_size = i * h + h * h + h;

            // Compute all 4 gates
            let mut gates = [[0.0f32; 32]; 4];

            for (g, gate) in gates.iter_mut().enumerate() {
                let base = g * gate_size;
                let wi_base = base;
                let uh_base = base + i * h;
                let b_base = base + i * h + h * h;

                for hi in 0..h {
                    let mut sum = self.weights[b_base + hi];
                    for ii in 0..i {
                        sum += input[ii] * self.weights[wi_base + ii * h + hi];
                    }
                    // Hidden contribution is zero since hidden = 0
                    for hh in 0..h {
                        sum += hidden[hh] * self.weights[uh_base + hh * h + hi];
                    }
                    gate[hi] = sum;
                }
            }

            // Apply activations and compute output hidden
            for hi in 0..h {
                let i_gate = sigmoid(gates[0][hi]);
                let f_gate = sigmoid(gates[1][hi]);
                let g_gate = gates[2][hi].tanh();
                let o_gate = sigmoid(gates[3][hi]);

                cell[hi] = f_gate * cell[hi] + i_gate * g_gate;
                hidden[hi] = o_gate * cell[hi].tanh();
            }

            // Output layer
            let out_base = 4 * gate_size;
            let out_b_base = out_base + h * o;

            let mut output = vec![0.0f32; o];
            for oi in 0..o {
                let mut sum = self.weights[out_b_base + oi];
                for hi in 0..h {
                    sum += hidden[hi] * self.weights[out_base + hi * o + oi];
                }
                output[oi] = sum;
            }

            output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }

        fn mutate(&mut self, rate: f32, magnitude: f32, rng: &mut Rng) {
            for w in &mut self.weights {
                if rng.next_f32() < rate {
                    *w += rng.next_f32_range(-magnitude, magnitude);
                }
            }
        }

        fn crossover(a: &Self, b: &Self, rng: &mut Rng) -> Self {
            let weights = a
                .weights
                .iter()
                .zip(b.weights.iter())
                .map(|(wa, wb)| if rng.next_f32() < 0.5 { *wa } else { *wb })
                .collect();
            Self {
                weights,
                input_size: a.input_size,
                hidden_size: a.hidden_size,
                output_size: a.output_size,
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Brain Type 4: Full LSTM (with memory carry)
    // ═══════════════════════════════════════════════════════════════════════════

    #[derive(Clone)]
    struct LstmGenome {
        weights: Vec<f32>,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        hidden: Vec<f32>,
        cell: Vec<f32>,
    }

    impl LstmGenome {
        fn new_random(
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
            rng: &mut Rng,
        ) -> Self {
            let gate_weights = input_size * hidden_size + hidden_size * hidden_size + hidden_size;
            let total = 4 * gate_weights + hidden_size * output_size + output_size;

            let mut weights: Vec<f32> = (0..total).map(|_| rng.next_f32_range(-0.3, 0.3)).collect();

            let f_bias_start = gate_weights + input_size * hidden_size + hidden_size * hidden_size;
            for idx in 0..hidden_size {
                weights[f_bias_start + idx] = 1.0;
            }

            Self {
                weights,
                input_size,
                hidden_size,
                output_size,
                hidden: vec![0.0; hidden_size],
                cell: vec![0.0; hidden_size],
            }
        }

        fn param_count(&self) -> usize {
            let gate_weights = self.input_size * self.hidden_size
                + self.hidden_size * self.hidden_size
                + self.hidden_size;
            4 * gate_weights + self.hidden_size * self.output_size + self.output_size
        }

        fn reset_state(&mut self) {
            self.hidden.fill(0.0);
            self.cell.fill(0.0);
        }

        fn forward(&mut self, input: &[f32]) -> usize {
            let i = self.input_size;
            let h = self.hidden_size;
            let o = self.output_size;

            let gate_size = i * h + h * h + h;

            let mut gates = [[0.0f32; 32]; 4];

            for (g, gate) in gates.iter_mut().enumerate() {
                let base = g * gate_size;
                let wi_base = base;
                let uh_base = base + i * h;
                let b_base = base + i * h + h * h;

                for hi in 0..h {
                    let mut sum = self.weights[b_base + hi];
                    for ii in 0..i {
                        sum += input[ii] * self.weights[wi_base + ii * h + hi];
                    }
                    for hh in 0..h {
                        sum += self.hidden[hh] * self.weights[uh_base + hh * h + hi];
                    }
                    gate[hi] = sum;
                }
            }

            for hi in 0..h {
                let i_gate = sigmoid(gates[0][hi]);
                let f_gate = sigmoid(gates[1][hi]);
                let g_gate = gates[2][hi].tanh();
                let o_gate = sigmoid(gates[3][hi]);

                self.cell[hi] = f_gate * self.cell[hi] + i_gate * g_gate;
                self.hidden[hi] = o_gate * self.cell[hi].tanh();
            }

            let out_base = 4 * gate_size;
            let out_b_base = out_base + h * o;

            let mut output = vec![0.0f32; o];
            for oi in 0..o {
                let mut sum = self.weights[out_b_base + oi];
                for hi in 0..h {
                    sum += self.hidden[hi] * self.weights[out_base + hi * o + oi];
                }
                output[oi] = sum;
            }

            output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }

        fn mutate(&mut self, rate: f32, magnitude: f32, rng: &mut Rng) {
            for w in &mut self.weights {
                if rng.next_f32() < rate {
                    *w += rng.next_f32_range(-magnitude, magnitude);
                }
            }
        }

        fn crossover(a: &Self, b: &Self, rng: &mut Rng) -> Self {
            let weights = a
                .weights
                .iter()
                .zip(b.weights.iter())
                .map(|(wa, wb)| if rng.next_f32() < 0.5 { *wa } else { *wb })
                .collect();
            Self {
                weights,
                input_size: a.input_size,
                hidden_size: a.hidden_size,
                output_size: a.output_size,
                hidden: vec![0.0; a.hidden_size],
                cell: vec![0.0; a.hidden_size],
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Evolution Runners (generic over brain type)
    // ═══════════════════════════════════════════════════════════════════════════

    fn select_parents(
        fitnesses: &[f32],
        pop_size: usize,
        tournament_size: usize,
        rng: &mut Rng,
    ) -> (usize, usize) {
        let tournament = |rng: &mut Rng| -> usize {
            let mut best_idx = rng.next_usize(pop_size);
            let mut best_fit = fitnesses[best_idx];
            for _ in 1..tournament_size {
                let idx = rng.next_usize(pop_size);
                if fitnesses[idx] > best_fit {
                    best_idx = idx;
                    best_fit = fitnesses[idx];
                }
            }
            best_idx
        };
        let p1 = tournament(rng);
        let mut p2 = tournament(rng);
        while p2 == p1 && pop_size > 1 {
            p2 = tournament(rng);
        }
        (p1, p2)
    }

    fn run_simple_evolution(
        task: &mut dyn Task,
        pop_size: usize,
        generations: usize,
        ticks: usize,
        grid: usize,
        seed: u64,
    ) -> f32 {
        let mut rng = Rng::new(seed);
        let mut population: Vec<SimpleGenome> = (0..pop_size)
            .map(|_| {
                SimpleGenome::new_random(task.sensor_count(), 16, task.action_count(), &mut rng)
            })
            .collect();

        let evo_config = EvolutionConfig {
            organism_count: pop_size,
            ticks_per_generation: ticks,
            generations: 1,
            mutation_rate: 0.1,
            mutation_strength: 0.3,
            elite_count: pop_size / 10,
            tournament_size: 3,
        };

        for generation in 0..generations {
            let mut fitnesses = vec![0.0f32; pop_size];
            for (idx, genome) in population.iter().enumerate() {
                let mut evo = Evolution::new(
                    evo_config.clone(),
                    grid,
                    seed + generation as u64 * 1000 + idx as u64,
                );
                task.reset_for_generation(&mut evo.world_mut(), seed + generation as u64);
                let genome_clone = genome.clone();
                let stats = evo.run_generation(task, |_, sensors| genome_clone.forward(sensors));
                fitnesses[idx] = stats.mean_fitness;
            }

            let mut indices: Vec<usize> = (0..pop_size).collect();
            indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

            let mut new_pop = Vec::with_capacity(pop_size);
            for &idx in indices.iter().take(pop_size / 10) {
                new_pop.push(population[idx].clone());
            }
            while new_pop.len() < pop_size {
                let (p1, p2) = select_parents(&fitnesses, pop_size, 3, &mut rng);
                let mut child = SimpleGenome::crossover(&population[p1], &population[p2], &mut rng);
                child.mutate(0.1, 0.3, &mut rng);
                new_pop.push(child);
            }
            population = new_pop;
        }

        // Return best final fitness
        let mut fitnesses = vec![0.0f32; pop_size];
        for (idx, genome) in population.iter().enumerate() {
            let mut evo = Evolution::new(
                evo_config.clone(),
                grid,
                seed + generations as u64 * 1000 + idx as u64,
            );
            task.reset_for_generation(&mut evo.world_mut(), seed);
            let genome_clone = genome.clone();
            let stats = evo.run_generation(task, |_, sensors| genome_clone.forward(sensors));
            fitnesses[idx] = stats.mean_fitness;
        }
        fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    fn run_deep_simple_evolution(
        task: &mut dyn Task,
        pop_size: usize,
        generations: usize,
        ticks: usize,
        grid: usize,
        seed: u64,
    ) -> f32 {
        let mut rng = Rng::new(seed);
        let mut population: Vec<DeepSimpleGenome> = (0..pop_size)
            .map(|_| {
                DeepSimpleGenome::new_random(task.sensor_count(), task.action_count(), &mut rng)
            })
            .collect();

        let evo_config = EvolutionConfig {
            organism_count: pop_size,
            ticks_per_generation: ticks,
            generations: 1,
            mutation_rate: 0.1,
            mutation_strength: 0.3,
            elite_count: pop_size / 10,
            tournament_size: 3,
        };

        for generation in 0..generations {
            let mut fitnesses = vec![0.0f32; pop_size];
            for (idx, genome) in population.iter().enumerate() {
                let mut evo = Evolution::new(
                    evo_config.clone(),
                    grid,
                    seed + generation as u64 * 1000 + idx as u64,
                );
                task.reset_for_generation(&mut evo.world_mut(), seed + generation as u64);
                let genome_clone = genome.clone();
                let stats = evo.run_generation(task, |_, sensors| genome_clone.forward(sensors));
                fitnesses[idx] = stats.mean_fitness;
            }

            let mut indices: Vec<usize> = (0..pop_size).collect();
            indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

            let mut new_pop = Vec::with_capacity(pop_size);
            for &idx in indices.iter().take(pop_size / 10) {
                new_pop.push(population[idx].clone());
            }
            while new_pop.len() < pop_size {
                let (p1, p2) = select_parents(&fitnesses, pop_size, 3, &mut rng);
                let mut child =
                    DeepSimpleGenome::crossover(&population[p1], &population[p2], &mut rng);
                child.mutate(0.1, 0.3, &mut rng);
                new_pop.push(child);
            }
            population = new_pop;
        }

        let mut fitnesses = vec![0.0f32; pop_size];
        for (idx, genome) in population.iter().enumerate() {
            let mut evo = Evolution::new(
                evo_config.clone(),
                grid,
                seed + generations as u64 * 1000 + idx as u64,
            );
            task.reset_for_generation(&mut evo.world_mut(), seed);
            let genome_clone = genome.clone();
            let stats = evo.run_generation(task, |_, sensors| genome_clone.forward(sensors));
            fitnesses[idx] = stats.mean_fitness;
        }
        fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    fn run_gated_ff_evolution(
        task: &mut dyn Task,
        pop_size: usize,
        generations: usize,
        ticks: usize,
        grid: usize,
        seed: u64,
    ) -> f32 {
        let mut rng = Rng::new(seed);
        let mut population: Vec<GatedFeedforwardGenome> = (0..pop_size)
            .map(|_| {
                GatedFeedforwardGenome::new_random(
                    task.sensor_count(),
                    24,
                    task.action_count(),
                    &mut rng,
                )
            })
            .collect();

        let evo_config = EvolutionConfig {
            organism_count: pop_size,
            ticks_per_generation: ticks,
            generations: 1,
            mutation_rate: 0.1,
            mutation_strength: 0.3,
            elite_count: pop_size / 10,
            tournament_size: 3,
        };

        for generation in 0..generations {
            let mut fitnesses = vec![0.0f32; pop_size];
            for (idx, genome) in population.iter().enumerate() {
                let mut evo = Evolution::new(
                    evo_config.clone(),
                    grid,
                    seed + generation as u64 * 1000 + idx as u64,
                );
                task.reset_for_generation(&mut evo.world_mut(), seed + generation as u64);
                let genome_clone = genome.clone();
                let stats = evo.run_generation(task, |_, sensors| genome_clone.forward(sensors));
                fitnesses[idx] = stats.mean_fitness;
            }

            let mut indices: Vec<usize> = (0..pop_size).collect();
            indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

            let mut new_pop = Vec::with_capacity(pop_size);
            for &idx in indices.iter().take(pop_size / 10) {
                new_pop.push(population[idx].clone());
            }
            while new_pop.len() < pop_size {
                let (p1, p2) = select_parents(&fitnesses, pop_size, 3, &mut rng);
                let mut child =
                    GatedFeedforwardGenome::crossover(&population[p1], &population[p2], &mut rng);
                child.mutate(0.1, 0.3, &mut rng);
                new_pop.push(child);
            }
            population = new_pop;
        }

        let mut fitnesses = vec![0.0f32; pop_size];
        for (idx, genome) in population.iter().enumerate() {
            let mut evo = Evolution::new(
                evo_config.clone(),
                grid,
                seed + generations as u64 * 1000 + idx as u64,
            );
            task.reset_for_generation(&mut evo.world_mut(), seed);
            let genome_clone = genome.clone();
            let stats = evo.run_generation(task, |_, sensors| genome_clone.forward(sensors));
            fitnesses[idx] = stats.mean_fitness;
        }
        fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    fn run_stateless_lstm_evolution(
        task: &mut dyn Task,
        pop_size: usize,
        generations: usize,
        ticks: usize,
        grid: usize,
        seed: u64,
    ) -> f32 {
        let mut rng = Rng::new(seed);
        let mut population: Vec<StatelessLstmGenome> = (0..pop_size)
            .map(|_| {
                StatelessLstmGenome::new_random(
                    task.sensor_count(),
                    16,
                    task.action_count(),
                    &mut rng,
                )
            })
            .collect();

        let evo_config = EvolutionConfig {
            organism_count: pop_size,
            ticks_per_generation: ticks,
            generations: 1,
            mutation_rate: 0.1,
            mutation_strength: 0.3,
            elite_count: pop_size / 10,
            tournament_size: 3,
        };

        for generation in 0..generations {
            let mut fitnesses = vec![0.0f32; pop_size];
            for (idx, genome) in population.iter().enumerate() {
                let mut evo = Evolution::new(
                    evo_config.clone(),
                    grid,
                    seed + generation as u64 * 1000 + idx as u64,
                );
                task.reset_for_generation(&mut evo.world_mut(), seed + generation as u64);
                let genome_clone = genome.clone();
                let stats = evo.run_generation(task, |_, sensors| genome_clone.forward(sensors));
                fitnesses[idx] = stats.mean_fitness;
            }

            let mut indices: Vec<usize> = (0..pop_size).collect();
            indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

            let mut new_pop = Vec::with_capacity(pop_size);
            for &idx in indices.iter().take(pop_size / 10) {
                new_pop.push(population[idx].clone());
            }
            while new_pop.len() < pop_size {
                let (p1, p2) = select_parents(&fitnesses, pop_size, 3, &mut rng);
                let mut child =
                    StatelessLstmGenome::crossover(&population[p1], &population[p2], &mut rng);
                child.mutate(0.1, 0.3, &mut rng);
                new_pop.push(child);
            }
            population = new_pop;
        }

        let mut fitnesses = vec![0.0f32; pop_size];
        for (idx, genome) in population.iter().enumerate() {
            let mut evo = Evolution::new(
                evo_config.clone(),
                grid,
                seed + generations as u64 * 1000 + idx as u64,
            );
            task.reset_for_generation(&mut evo.world_mut(), seed);
            let genome_clone = genome.clone();
            let stats = evo.run_generation(task, |_, sensors| genome_clone.forward(sensors));
            fitnesses[idx] = stats.mean_fitness;
        }
        fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    fn run_lstm_evolution(
        task: &mut dyn Task,
        pop_size: usize,
        generations: usize,
        ticks: usize,
        grid: usize,
        seed: u64,
    ) -> f32 {
        let mut rng = Rng::new(seed);
        let mut population: Vec<LstmGenome> = (0..pop_size)
            .map(|_| LstmGenome::new_random(task.sensor_count(), 16, task.action_count(), &mut rng))
            .collect();

        let evo_config = EvolutionConfig {
            organism_count: pop_size,
            ticks_per_generation: ticks,
            generations: 1,
            mutation_rate: 0.1,
            mutation_strength: 0.3,
            elite_count: pop_size / 10,
            tournament_size: 3,
        };

        for generation in 0..generations {
            let mut fitnesses = vec![0.0f32; pop_size];
            for (idx, genome) in population.iter_mut().enumerate() {
                genome.reset_state();
                let mut evo = Evolution::new(
                    evo_config.clone(),
                    grid,
                    seed + generation as u64 * 1000 + idx as u64,
                );
                task.reset_for_generation(&mut evo.world_mut(), seed + generation as u64);
                let stats = evo.run_generation(task, |_, sensors| genome.forward(sensors));
                fitnesses[idx] = stats.mean_fitness;
            }

            let mut indices: Vec<usize> = (0..pop_size).collect();
            indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

            let mut new_pop = Vec::with_capacity(pop_size);
            for &idx in indices.iter().take(pop_size / 10) {
                new_pop.push(population[idx].clone());
            }
            while new_pop.len() < pop_size {
                let (p1, p2) = select_parents(&fitnesses, pop_size, 3, &mut rng);
                let mut child = LstmGenome::crossover(&population[p1], &population[p2], &mut rng);
                child.mutate(0.1, 0.3, &mut rng);
                new_pop.push(child);
            }
            population = new_pop;
        }

        let mut fitnesses = vec![0.0f32; pop_size];
        for (idx, genome) in population.iter_mut().enumerate() {
            genome.reset_state();
            let mut evo = Evolution::new(
                evo_config.clone(),
                grid,
                seed + generations as u64 * 1000 + idx as u64,
            );
            task.reset_for_generation(&mut evo.world_mut(), seed);
            let stats = evo.run_generation(task, |_, sensors| genome.forward(sensors));
            fitnesses[idx] = stats.mean_fitness;
        }
        fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Statistics
    // ═══════════════════════════════════════════════════════════════════════════

    fn compute_stats(values: &[f32]) -> (f32, f32) {
        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        (mean, variance.sqrt())
    }

    fn cohens_d(a: &[f32], b: &[f32]) -> f64 {
        let (mean_a, _) = compute_stats(a);
        let (mean_b, _) = compute_stats(b);
        let var_a: f32 = a.iter().map(|v| (v - mean_a).powi(2)).sum::<f32>() / a.len() as f32;
        let var_b: f32 = b.iter().map(|v| (v - mean_b).powi(2)).sum::<f32>() / b.len() as f32;
        let pooled_std = ((var_a + var_b) / 2.0).sqrt();
        if pooled_std < 0.001 {
            0.0
        } else {
            ((mean_b - mean_a) / pooled_std) as f64
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Main Experiment
    // ═══════════════════════════════════════════════════════════════════════════

    pub fn run() {
        println!("╔═══════════════════════════════════════════════════════════════╗");
        println!("║  EPIC 101.5: Investigating the LSTM Advantage                 ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");
        println!();

        // Configuration - optimized for speed
        let pop_size = 20; // Reduced from 30
        let generations = 30; // Reduced from 50
        let ticks = 200; // Reduced from 300
        let grid = 64; // Reduced from 128
        let num_seeds = 10; // Reduced from 20 (still statistically meaningful)

        println!(
            "  Config: pop={}, gens={}, ticks={}, seeds={}",
            pop_size, generations, ticks, num_seeds
        );
        println!();

        // Print parameter counts
        let mut rng = Rng::new(42);
        let simple = SimpleGenome::new_random(8, 16, 5, &mut rng);
        let deep = DeepSimpleGenome::new_random(8, 5, &mut rng);
        let gated = GatedFeedforwardGenome::new_random(8, 24, 5, &mut rng);
        let stateless = StatelessLstmGenome::new_random(8, 16, 5, &mut rng);
        let lstm = LstmGenome::new_random(8, 16, 5, &mut rng);

        println!("  Parameter counts:");
        println!("    SimpleBrain:      {:5} params", simple.param_count());
        println!("    DeepSimple:       {:5} params", deep.param_count());
        println!("    GatedFeedforward: {:5} params", gated.param_count());
        println!("    StatelessLstm:    {:5} params", stateless.param_count());
        println!("    LSTM:             {:5} params", lstm.param_count());
        println!();

        // ═══════════════════════════════════════════════════════════════════
        // Task 1: Stationary Foraging
        // ═══════════════════════════════════════════════════════════════════
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Task 1: Stationary Foraging (Control)                          │");
        println!("└─────────────────────────────────────────────────────────────────┘\n");

        let mut simple_results = Vec::new();
        let mut deep_results = Vec::new();
        let mut gated_results = Vec::new();
        let mut stateless_results = Vec::new();
        let mut lstm_results = Vec::new();

        for seed in 0..num_seeds {
            print!("  Seed {:2}/{}: ", seed + 1, num_seeds);

            let mut task = StationaryTask::new(StationaryConfig {
                grid_size: grid,
                food_patch_count: 15,
                food_radius: 12.0,
                food_value: 10.0,
                sensor_range: 60.0,
            });

            let s = run_simple_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 100,
            );
            print!("Simple={:.0} ", s);
            simple_results.push(s);

            let mut task = StationaryTask::new(StationaryConfig {
                grid_size: grid,
                food_patch_count: 15,
                food_radius: 12.0,
                food_value: 10.0,
                sensor_range: 60.0,
            });
            let d = run_deep_simple_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 100 + 1,
            );
            print!("Deep={:.0} ", d);
            deep_results.push(d);

            let mut task = StationaryTask::new(StationaryConfig {
                grid_size: grid,
                food_patch_count: 15,
                food_radius: 12.0,
                food_value: 10.0,
                sensor_range: 60.0,
            });
            let g = run_gated_ff_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 100 + 2,
            );
            print!("Gated={:.0} ", g);
            gated_results.push(g);

            let mut task = StationaryTask::new(StationaryConfig {
                grid_size: grid,
                food_patch_count: 15,
                food_radius: 12.0,
                food_value: 10.0,
                sensor_range: 60.0,
            });
            let sl = run_stateless_lstm_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 100 + 3,
            );
            print!("Stateless={:.0} ", sl);
            stateless_results.push(sl);

            let mut task = StationaryTask::new(StationaryConfig {
                grid_size: grid,
                food_patch_count: 15,
                food_radius: 12.0,
                food_value: 10.0,
                sensor_range: 60.0,
            });
            let l = run_lstm_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 100 + 4,
            );
            println!("LSTM={:.0}", l);
            lstm_results.push(l);
        }

        // ═══════════════════════════════════════════════════════════════════
        // Results Analysis
        // ═══════════════════════════════════════════════════════════════════
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  Task 1 Results                                                ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let (mean_s, std_s) = compute_stats(&simple_results);
        let (mean_d, std_d) = compute_stats(&deep_results);
        let (mean_g, std_g) = compute_stats(&gated_results);
        let (mean_sl, std_sl) = compute_stats(&stateless_results);
        let (mean_l, std_l) = compute_stats(&lstm_results);

        println!("  Brain Type          | Mean ± Std      | vs Simple | vs LSTM");
        println!("  ────────────────────┼─────────────────┼───────────┼─────────");
        println!(
            "  SimpleBrain         | {:7.1} ± {:5.1} |    --     | d={:.2}",
            mean_s,
            std_s,
            cohens_d(&simple_results, &lstm_results)
        );
        println!(
            "  DeepSimple          | {:7.1} ± {:5.1} | d={:5.2}  | d={:.2}",
            mean_d,
            std_d,
            cohens_d(&simple_results, &deep_results),
            cohens_d(&deep_results, &lstm_results)
        );
        println!(
            "  GatedFeedforward    | {:7.1} ± {:5.1} | d={:5.2}  | d={:.2}",
            mean_g,
            std_g,
            cohens_d(&simple_results, &gated_results),
            cohens_d(&gated_results, &lstm_results)
        );
        println!(
            "  StatelessLstm       | {:7.1} ± {:5.1} | d={:5.2}  | d={:.2}",
            mean_sl,
            std_sl,
            cohens_d(&simple_results, &stateless_results),
            cohens_d(&stateless_results, &lstm_results)
        );
        println!(
            "  LSTM (full)         | {:7.1} ± {:5.1} | d={:5.2}  |    --",
            mean_l,
            std_l,
            cohens_d(&simple_results, &lstm_results)
        );

        // Interpretation
        println!("\n  Interpretation:");
        let d_lstm_vs_simple = cohens_d(&simple_results, &lstm_results);
        let d_stateless_vs_lstm = cohens_d(&stateless_results, &lstm_results);
        let d_deep_vs_lstm = cohens_d(&deep_results, &lstm_results);
        let d_gated_vs_lstm = cohens_d(&gated_results, &lstm_results);

        if d_lstm_vs_simple.abs() < 0.3 {
            println!("    → No significant difference (task too easy / ceiling effect)");
        } else {
            println!(
                "    → LSTM advantage confirmed (d = {:.2})",
                d_lstm_vs_simple
            );

            if d_deep_vs_lstm.abs() < 0.3 {
                println!("    → DeepSimple matches LSTM → CAPACITY explains gap");
            } else if d_gated_vs_lstm.abs() < 0.3 {
                println!("    → GatedFF matches LSTM → GATING explains gap");
            } else if d_stateless_vs_lstm.abs() < 0.3 {
                println!("    → StatelessLstm matches LSTM → ARCHITECTURE (not memory)");
            } else {
                println!("    → Only full LSTM works → MEMORY is valuable!");
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // Task 1 HARD: Stationary Foraging with more challenge
        // ═══════════════════════════════════════════════════════════════════
        println!("\n┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Task 1 HARD: Sparse Food, Limited Sensors                      │");
        println!("└─────────────────────────────────────────────────────────────────┘\n");

        let mut simple_hard = Vec::new();
        let mut deep_hard = Vec::new();
        let mut gated_hard = Vec::new();
        let mut stateless_hard = Vec::new();
        let mut lstm_hard = Vec::new();

        for seed in 0..num_seeds {
            print!("  Seed {:2}/{}: ", seed + 1, num_seeds);

            // HARDER config: smaller food patches, shorter sensor range, larger grid
            let mut task = StationaryTask::new(StationaryConfig {
                grid_size: grid,
                food_patch_count: 8,
                food_radius: 6.0,
                food_value: 10.0,
                sensor_range: 25.0,
            });
            let s = run_simple_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 200,
            );
            print!("S={:.0} ", s);
            simple_hard.push(s);

            let mut task = StationaryTask::new(StationaryConfig {
                grid_size: grid,
                food_patch_count: 8,
                food_radius: 6.0,
                food_value: 10.0,
                sensor_range: 25.0,
            });
            let d = run_deep_simple_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 200 + 1,
            );
            print!("D={:.0} ", d);
            deep_hard.push(d);

            let mut task = StationaryTask::new(StationaryConfig {
                grid_size: grid,
                food_patch_count: 8,
                food_radius: 6.0,
                food_value: 10.0,
                sensor_range: 25.0,
            });
            let g = run_gated_ff_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 200 + 2,
            );
            print!("G={:.0} ", g);
            gated_hard.push(g);

            let mut task = StationaryTask::new(StationaryConfig {
                grid_size: grid,
                food_patch_count: 8,
                food_radius: 6.0,
                food_value: 10.0,
                sensor_range: 25.0,
            });
            let sl = run_stateless_lstm_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 200 + 3,
            );
            print!("SL={:.0} ", sl);
            stateless_hard.push(sl);

            let mut task = StationaryTask::new(StationaryConfig {
                grid_size: grid,
                food_patch_count: 8,
                food_radius: 6.0,
                food_value: 10.0,
                sensor_range: 25.0,
            });
            let l = run_lstm_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 200 + 4,
            );
            println!("L={:.0}", l);
            lstm_hard.push(l);
        }

        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  Task 1 HARD Results                                           ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let (mean_s, std_s) = compute_stats(&simple_hard);
        let (mean_d, std_d) = compute_stats(&deep_hard);
        let (mean_g, std_g) = compute_stats(&gated_hard);
        let (mean_sl, std_sl) = compute_stats(&stateless_hard);
        let (mean_l, std_l) = compute_stats(&lstm_hard);

        println!("  Brain Type          | Mean ± Std      | vs Simple | vs LSTM");
        println!("  ────────────────────┼─────────────────┼───────────┼─────────");
        println!(
            "  SimpleBrain         | {:7.1} ± {:5.1} |    --     | d={:.2}",
            mean_s,
            std_s,
            cohens_d(&simple_hard, &lstm_hard)
        );
        println!(
            "  DeepSimple          | {:7.1} ± {:5.1} | d={:5.2}  | d={:.2}",
            mean_d,
            std_d,
            cohens_d(&simple_hard, &deep_hard),
            cohens_d(&deep_hard, &lstm_hard)
        );
        println!(
            "  GatedFeedforward    | {:7.1} ± {:5.1} | d={:5.2}  | d={:.2}",
            mean_g,
            std_g,
            cohens_d(&simple_hard, &gated_hard),
            cohens_d(&gated_hard, &lstm_hard)
        );
        println!(
            "  StatelessLstm       | {:7.1} ± {:5.1} | d={:5.2}  | d={:.2}",
            mean_sl,
            std_sl,
            cohens_d(&simple_hard, &stateless_hard),
            cohens_d(&stateless_hard, &lstm_hard)
        );
        println!(
            "  LSTM (full)         | {:7.1} ± {:5.1} | d={:5.2}  |    --",
            mean_l,
            std_l,
            cohens_d(&simple_hard, &lstm_hard)
        );

        // ═══════════════════════════════════════════════════════════════════
        // Task 5: Delayed Reward (THE KEY MEMORY TEST)
        // ═══════════════════════════════════════════════════════════════════
        println!("\n┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Task 5: Delayed Reward (KEY MEMORY TEST)                       │");
        println!("└─────────────────────────────────────────────────────────────────┘\n");

        let mut simple_t5 = Vec::new();
        let mut deep_t5 = Vec::new();
        let mut gated_t5 = Vec::new();
        let mut stateless_t5 = Vec::new();
        let mut lstm_t5 = Vec::new();

        for seed in 0..num_seeds {
            print!("  Seed {:2}/{}: ", seed + 1, num_seeds);

            let mut task = DelayedRewardTask::new(DelayedRewardConfig {
                grid_size: grid,
                beacon_x: (grid as f32) / 2.0,
                beacon_y: (grid as f32) / 4.0,
                beacon_radius: 10.0,
                food_x: (grid as f32) / 2.0,
                food_y: (grid as f32) * 3.0 / 4.0,
                food_radius: 12.0,
                food_value: 50.0,
                unlock_duration: 100,
                sensor_range: 40.0,
            });
            let s = run_simple_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 300,
            );
            print!("S={:.0} ", s);
            simple_t5.push(s);

            let mut task = DelayedRewardTask::new(DelayedRewardConfig {
                grid_size: grid,
                beacon_x: (grid as f32) / 2.0,
                beacon_y: (grid as f32) / 4.0,
                beacon_radius: 10.0,
                food_x: (grid as f32) / 2.0,
                food_y: (grid as f32) * 3.0 / 4.0,
                food_radius: 12.0,
                food_value: 50.0,
                unlock_duration: 100,
                sensor_range: 40.0,
            });
            let d = run_deep_simple_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 300 + 1,
            );
            print!("D={:.0} ", d);
            deep_t5.push(d);

            let mut task = DelayedRewardTask::new(DelayedRewardConfig {
                grid_size: grid,
                beacon_x: (grid as f32) / 2.0,
                beacon_y: (grid as f32) / 4.0,
                beacon_radius: 10.0,
                food_x: (grid as f32) / 2.0,
                food_y: (grid as f32) * 3.0 / 4.0,
                food_radius: 12.0,
                food_value: 50.0,
                unlock_duration: 100,
                sensor_range: 40.0,
            });
            let g = run_gated_ff_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 300 + 2,
            );
            print!("G={:.0} ", g);
            gated_t5.push(g);

            let mut task = DelayedRewardTask::new(DelayedRewardConfig {
                grid_size: grid,
                beacon_x: (grid as f32) / 2.0,
                beacon_y: (grid as f32) / 4.0,
                beacon_radius: 10.0,
                food_x: (grid as f32) / 2.0,
                food_y: (grid as f32) * 3.0 / 4.0,
                food_radius: 12.0,
                food_value: 50.0,
                unlock_duration: 100,
                sensor_range: 40.0,
            });
            let sl = run_stateless_lstm_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 300 + 3,
            );
            print!("SL={:.0} ", sl);
            stateless_t5.push(sl);

            let mut task = DelayedRewardTask::new(DelayedRewardConfig {
                grid_size: grid,
                beacon_x: (grid as f32) / 2.0,
                beacon_y: (grid as f32) / 4.0,
                beacon_radius: 10.0,
                food_x: (grid as f32) / 2.0,
                food_y: (grid as f32) * 3.0 / 4.0,
                food_radius: 12.0,
                food_value: 50.0,
                unlock_duration: 100,
                sensor_range: 40.0,
            });
            let l = run_lstm_evolution(
                &mut task,
                pop_size,
                generations,
                ticks,
                grid,
                seed as u64 * 300 + 4,
            );
            println!("L={:.0}", l);
            lstm_t5.push(l);
        }

        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  Task 5: Delayed Reward Results                                ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let (mean_s, std_s) = compute_stats(&simple_t5);
        let (mean_d, std_d) = compute_stats(&deep_t5);
        let (mean_g, std_g) = compute_stats(&gated_t5);
        let (mean_sl, std_sl) = compute_stats(&stateless_t5);
        let (mean_l, std_l) = compute_stats(&lstm_t5);

        println!("  Brain Type          | Mean ± Std      | vs Simple | vs LSTM");
        println!("  ────────────────────┼─────────────────┼───────────┼─────────");
        println!(
            "  SimpleBrain         | {:7.1} ± {:5.1} |    --     | d={:.2}",
            mean_s,
            std_s,
            cohens_d(&simple_t5, &lstm_t5)
        );
        println!(
            "  DeepSimple          | {:7.1} ± {:5.1} | d={:5.2}  | d={:.2}",
            mean_d,
            std_d,
            cohens_d(&simple_t5, &deep_t5),
            cohens_d(&deep_t5, &lstm_t5)
        );
        println!(
            "  GatedFeedforward    | {:7.1} ± {:5.1} | d={:5.2}  | d={:.2}",
            mean_g,
            std_g,
            cohens_d(&simple_t5, &gated_t5),
            cohens_d(&gated_t5, &lstm_t5)
        );
        println!(
            "  StatelessLstm       | {:7.1} ± {:5.1} | d={:5.2}  | d={:.2}",
            mean_sl,
            std_sl,
            cohens_d(&simple_t5, &stateless_t5),
            cohens_d(&stateless_t5, &lstm_t5)
        );
        println!(
            "  LSTM (full)         | {:7.1} ± {:5.1} | d={:5.2}  |    --",
            mean_l,
            std_l,
            cohens_d(&simple_t5, &lstm_t5)
        );

        // Final interpretation
        println!("\n  Task 5 Interpretation:");
        let d5_lstm_vs_simple = cohens_d(&simple_t5, &lstm_t5);
        let d5_stateless_vs_lstm = cohens_d(&stateless_t5, &lstm_t5);

        if d5_lstm_vs_simple > 0.8 {
            println!(
                "    → LSTM DOMINATES on memory task (d = {:.2})",
                d5_lstm_vs_simple
            );
            if d5_stateless_vs_lstm.abs() > 0.5 {
                println!("    → StatelessLstm fails → MEMORY IS THE KEY FACTOR!");
            }
        } else if d5_lstm_vs_simple > 0.3 {
            println!(
                "    → LSTM moderate advantage (d = {:.2})",
                d5_lstm_vs_simple
            );
        } else {
            println!("    → No clear LSTM advantage (task may need tuning)");
        }

        // ═══════════════════════════════════════════════════════════════════
        // Final Summary
        // ═══════════════════════════════════════════════════════════════════
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  FINAL SUMMARY                                                 ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let d_t1_easy = cohens_d(&simple_results, &lstm_results);
        let d_t1_hard = cohens_d(&simple_hard, &lstm_hard);
        let d_t5 = cohens_d(&simple_t5, &lstm_t5);

        println!("  Task             | LSTM vs Simple | Interpretation");
        println!("  ─────────────────┼────────────────┼─────────────────────");
        println!(
            "  Task 1 (Easy)    |   d = {:5.2}    | {}",
            d_t1_easy,
            if d_t1_easy.abs() < 0.3 {
                "Ceiling effect"
            } else {
                "LSTM better"
            }
        );
        println!(
            "  Task 1 (Hard)    |   d = {:5.2}    | {}",
            d_t1_hard,
            if d_t1_hard.abs() < 0.3 {
                "No difference"
            } else if d_t1_hard > 0.3 {
                "LSTM better"
            } else {
                "Simple better"
            }
        );
        println!(
            "  Task 5 (Memory)  |   d = {:5.2}    | {}",
            d_t5,
            if d_t5 > 0.8 {
                "LSTM DOMINATES"
            } else if d_t5 > 0.3 {
                "LSTM better"
            } else {
                "No difference"
            }
        );

        println!("\n  Experiment complete.");
    }
}

fn main() {
    #[cfg(feature = "burn-compute")]
    experiment::run();

    #[cfg(not(feature = "burn-compute"))]
    {
        println!("EPIC 101.5: Investigating the LSTM Advantage");
        println!();
        println!(
            "Run with: cargo run --release --example epic101_5_experiment --features burn-compute"
        );
    }
}
