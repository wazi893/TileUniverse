//! EPIC 101: LSTM vs Simple NN Evolution Experiment
//!
//! Tests whether LSTM memory provides evolutionary advantage over simple feedforward NNs.
//! NOW WITH PROPER NEUROEVOLUTION - each organism has its own evolvable weights.
//!
//! Run with:
//! ```
//! cargo run --release --example epic101_experiment --features burn-compute
//! ```

#[cfg(feature = "burn-compute")]
mod experiment {
    use engine::epic101::{
        DelayedRewardConfig, DelayedRewardTask, Evolution, EvolutionConfig, GenerationStats,
        StationaryConfig, StationaryTask, Task,
    };
    use std::time::Instant;

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

    // ═══════════════════════════════════════════════════════════════════════════
    // Simple Feedforward NN (per-organism weights)
    // ═══════════════════════════════════════════════════════════════════════════

    #[derive(Clone)]
    struct SimpleGenome {
        weights: Vec<f32>, // Flat array of all weights
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        fitness: f32,
    }

    impl SimpleGenome {
        fn new_random(
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
            rng: &mut Rng,
        ) -> Self {
            let total_weights = input_size * hidden_size + hidden_size  // W1 + B1
                              + hidden_size * output_size + output_size; // W2 + B2

            let weights: Vec<f32> = (0..total_weights)
                .map(|_| rng.next_f32_range(-0.5, 0.5))
                .collect();

            Self {
                weights,
                input_size,
                hidden_size,
                output_size,
                fitness: 0.0,
            }
        }

        fn forward(&self, input: &[f32]) -> usize {
            let h = self.hidden_size;
            let o = self.output_size;
            let i = self.input_size;

            // Weight layout: [W1: i*h][B1: h][W2: h*o][B2: o]
            let w1_end = i * h;
            let b1_end = w1_end + h;
            let w2_end = b1_end + h * o;

            // Hidden layer
            let mut hidden = vec![0.0f32; h];
            for hi in 0..h {
                let mut sum = self.weights[w1_end + hi]; // bias
                for ii in 0..i {
                    sum += input[ii] * self.weights[ii * h + hi];
                }
                hidden[hi] = sum.tanh();
            }

            // Output layer
            let mut output = vec![0.0f32; o];
            for oi in 0..o {
                let mut sum = self.weights[w2_end + oi]; // bias
                for hi in 0..h {
                    sum += hidden[hi] * self.weights[b1_end + hi * o + oi];
                }
                output[oi] = sum;
            }

            // Argmax
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

        fn crossover(parent_a: &Self, parent_b: &Self, rng: &mut Rng) -> Self {
            let weights: Vec<f32> = parent_a
                .weights
                .iter()
                .zip(parent_b.weights.iter())
                .map(|(a, b)| if rng.next_f32() < 0.5 { *a } else { *b })
                .collect();

            Self {
                weights,
                input_size: parent_a.input_size,
                hidden_size: parent_a.hidden_size,
                output_size: parent_a.output_size,
                fitness: 0.0,
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // LSTM NN (per-organism weights + state)
    // ═══════════════════════════════════════════════════════════════════════════

    #[derive(Clone)]
    struct LstmGenome {
        weights: Vec<f32>, // All LSTM weights flattened
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        fitness: f32,
        // Runtime state (reset each generation)
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
            // LSTM has 4 gates, each with input weights + hidden weights + bias
            // Plus output layer
            let gate_weights = input_size * hidden_size + hidden_size * hidden_size + hidden_size;
            let total_weights = 4 * gate_weights  // i, f, g, o gates
                              + hidden_size * output_size + output_size; // output layer

            let mut weights: Vec<f32> = (0..total_weights)
                .map(|_| rng.next_f32_range(-0.3, 0.3))
                .collect();

            // Initialize forget gate bias to 1.0 (helps learning)
            let f_bias_start = gate_weights + input_size * hidden_size + hidden_size * hidden_size;
            for i in 0..hidden_size {
                weights[f_bias_start + i] = 1.0;
            }

            Self {
                weights,
                input_size,
                hidden_size,
                output_size,
                fitness: 0.0,
                hidden: vec![0.0; hidden_size],
                cell: vec![0.0; hidden_size],
            }
        }

        fn reset_state(&mut self) {
            self.hidden.fill(0.0);
            self.cell.fill(0.0);
        }

        fn forward(&mut self, input: &[f32]) -> usize {
            let i = self.input_size;
            let h = self.hidden_size;
            let o = self.output_size;

            // Gate weight layout per gate: [Wi: i*h][Uh: h*h][b: h]
            let gate_size = i * h + h * h + h;

            // Compute all 4 gates
            let mut gates = [[0.0f32; 32]; 4]; // Max hidden size 32

            for (g, gate) in gates.iter_mut().enumerate() {
                let base = g * gate_size;
                let wi_base = base;
                let uh_base = base + i * h;
                let b_base = base + i * h + h * h;

                for hi in 0..h {
                    let mut sum = self.weights[b_base + hi];

                    // Input contribution
                    for ii in 0..i {
                        sum += input[ii] * self.weights[wi_base + ii * h + hi];
                    }

                    // Hidden contribution
                    for hh in 0..h {
                        sum += self.hidden[hh] * self.weights[uh_base + hh * h + hi];
                    }

                    gate[hi] = sum;
                }
            }

            // Apply activations and update state
            for hi in 0..h {
                let i_gate = sigmoid(gates[0][hi]);
                let f_gate = sigmoid(gates[1][hi]);
                let g_gate = gates[2][hi].tanh();
                let o_gate = sigmoid(gates[3][hi]);

                self.cell[hi] = f_gate * self.cell[hi] + i_gate * g_gate;
                self.hidden[hi] = o_gate * self.cell[hi].tanh();
            }

            // Output layer
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

            // Argmax
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

        fn crossover(parent_a: &Self, parent_b: &Self, rng: &mut Rng) -> Self {
            let weights: Vec<f32> = parent_a
                .weights
                .iter()
                .zip(parent_b.weights.iter())
                .map(|(a, b)| if rng.next_f32() < 0.5 { *a } else { *b })
                .collect();

            Self {
                weights,
                input_size: parent_a.input_size,
                hidden_size: parent_a.hidden_size,
                output_size: parent_a.output_size,
                fitness: 0.0,
                hidden: vec![0.0; parent_a.hidden_size],
                cell: vec![0.0; parent_a.hidden_size],
            }
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Evolution with Selection + Mutation
    // ═══════════════════════════════════════════════════════════════════════════

    fn select_parents<'a, T>(
        population: &'a [T],
        fitnesses: &[f32],
        tournament_size: usize,
        rng: &mut Rng,
    ) -> (usize, usize) {
        let tournament = |rng: &mut Rng| -> usize {
            let mut best_idx = rng.next_usize(population.len());
            let mut best_fit = fitnesses[best_idx];

            for _ in 1..tournament_size {
                let idx = rng.next_usize(population.len());
                if fitnesses[idx] > best_fit {
                    best_idx = idx;
                    best_fit = fitnesses[idx];
                }
            }
            best_idx
        };

        let p1 = tournament(rng);
        let mut p2 = tournament(rng);
        while p2 == p1 && population.len() > 1 {
            p2 = tournament(rng);
        }

        (p1, p2)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Run Evolution with SimpleBrain
    // ═══════════════════════════════════════════════════════════════════════════

    fn run_simple_evolution(
        task: &mut dyn Task,
        pop_size: usize,
        generations: usize,
        ticks_per_gen: usize,
        grid_size: usize,
        seed: u64,
    ) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let input_size = task.sensor_count();
        let hidden_size = 16;
        let output_size = task.action_count();

        // Initialize population
        let mut population: Vec<SimpleGenome> = (0..pop_size)
            .map(|_| SimpleGenome::new_random(input_size, hidden_size, output_size, &mut rng))
            .collect();

        let mut best_fitness_history = Vec::new();
        let evo_config = EvolutionConfig {
            organism_count: pop_size,
            ticks_per_generation: ticks_per_gen,
            generations: 1,
            mutation_rate: 0.1,
            mutation_strength: 0.3,
            elite_count: pop_size / 10,
            tournament_size: 3,
        };

        for generation in 0..generations {
            // Evaluate each genome
            let mut fitnesses = vec![0.0f32; pop_size];

            for (idx, genome) in population.iter().enumerate() {
                let mut evo = Evolution::new(
                    evo_config.clone(),
                    grid_size,
                    seed + generation as u64 * 1000 + idx as u64,
                );
                task.reset_for_generation(&mut evo.world_mut(), seed + generation as u64);

                let genome_clone = genome.clone();
                let stats =
                    evo.run_generation(task, |_org_idx, sensors| genome_clone.forward(sensors));

                fitnesses[idx] = stats.mean_fitness;
            }

            // Track best
            let best_fit = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean_fit: f32 = fitnesses.iter().sum::<f32>() / pop_size as f32;
            best_fitness_history.push(best_fit);

            if generation % 10 == 0 || generation == generations - 1 {
                println!(
                    "    Gen {:3}: best={:.2}, mean={:.2}",
                    generation, best_fit, mean_fit
                );
            }

            // Selection + Reproduction
            let elite_count = pop_size / 10;
            let mut indices: Vec<usize> = (0..pop_size).collect();
            indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

            let mut new_population = Vec::with_capacity(pop_size);

            // Keep elites
            for &idx in indices.iter().take(elite_count) {
                new_population.push(population[idx].clone());
            }

            // Fill rest with offspring
            while new_population.len() < pop_size {
                let (p1, p2) = select_parents(&population, &fitnesses, 3, &mut rng);
                let mut child = SimpleGenome::crossover(&population[p1], &population[p2], &mut rng);
                child.mutate(0.1, 0.3, &mut rng);
                new_population.push(child);
            }

            population = new_population;
        }

        best_fitness_history
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Run Evolution with LstmBrain
    // ═══════════════════════════════════════════════════════════════════════════

    fn run_lstm_evolution(
        task: &mut dyn Task,
        pop_size: usize,
        generations: usize,
        ticks_per_gen: usize,
        grid_size: usize,
        seed: u64,
    ) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let input_size = task.sensor_count();
        let hidden_size = 16;
        let output_size = task.action_count();

        // Initialize population
        let mut population: Vec<LstmGenome> = (0..pop_size)
            .map(|_| LstmGenome::new_random(input_size, hidden_size, output_size, &mut rng))
            .collect();

        let mut best_fitness_history = Vec::new();
        let evo_config = EvolutionConfig {
            organism_count: pop_size,
            ticks_per_generation: ticks_per_gen,
            generations: 1,
            mutation_rate: 0.1,
            mutation_strength: 0.3,
            elite_count: pop_size / 10,
            tournament_size: 3,
        };

        for generation in 0..generations {
            // Evaluate each genome
            let mut fitnesses = vec![0.0f32; pop_size];

            for (idx, genome) in population.iter_mut().enumerate() {
                genome.reset_state(); // Reset LSTM state for new evaluation

                let mut evo = Evolution::new(
                    evo_config.clone(),
                    grid_size,
                    seed + generation as u64 * 1000 + idx as u64,
                );
                task.reset_for_generation(&mut evo.world_mut(), seed + generation as u64);

                let stats = evo.run_generation(task, |_org_idx, sensors| genome.forward(sensors));

                fitnesses[idx] = stats.mean_fitness;
            }

            // Track best
            let best_fit = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean_fit: f32 = fitnesses.iter().sum::<f32>() / pop_size as f32;
            best_fitness_history.push(best_fit);

            if generation % 10 == 0 || generation == generations - 1 {
                println!(
                    "    Gen {:3}: best={:.2}, mean={:.2}",
                    generation, best_fit, mean_fit
                );
            }

            // Selection + Reproduction
            let elite_count = pop_size / 10;
            let mut indices: Vec<usize> = (0..pop_size).collect();
            indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

            let mut new_population = Vec::with_capacity(pop_size);

            // Keep elites
            for &idx in indices.iter().take(elite_count) {
                new_population.push(population[idx].clone());
            }

            // Fill rest with offspring
            while new_population.len() < pop_size {
                let (p1, p2) = select_parents(&population, &fitnesses, 3, &mut rng);
                let mut child = LstmGenome::crossover(&population[p1], &population[p2], &mut rng);
                child.mutate(0.1, 0.3, &mut rng);
                new_population.push(child);
            }

            population = new_population;
        }

        best_fitness_history
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

    fn cohens_d(group_a: &[f32], group_b: &[f32]) -> f64 {
        let (mean_a, _) = compute_stats(group_a);
        let (mean_b, _) = compute_stats(group_b);

        let var_a: f32 =
            group_a.iter().map(|v| (v - mean_a).powi(2)).sum::<f32>() / group_a.len() as f32;
        let var_b: f32 =
            group_b.iter().map(|v| (v - mean_b).powi(2)).sum::<f32>() / group_b.len() as f32;

        let pooled_std = ((var_a + var_b) / 2.0).sqrt();
        if pooled_std < 0.001 {
            return 0.0;
        }

        ((mean_b - mean_a) / pooled_std) as f64
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Main Experiment
    // ═══════════════════════════════════════════════════════════════════════════

    pub fn run() {
        println!("╔═══════════════════════════════════════════════════════════════╗");
        println!("║  EPIC 101: LSTM vs Simple NN Neuroevolution Experiment        ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");
        println!();
        println!("  Backend: CPU with proper weight evolution");
        println!("  Each organism has its own evolvable neural network");
        println!();

        // Configuration
        let pop_size = 30; // Population size
        let generations = 50; // Generations of evolution
        let ticks_per_gen = 300; // Ticks per fitness evaluation
        let grid_size = 128;
        let num_seeds = 5; // Statistical replicates

        println!(
            "  Config: pop={}, gens={}, ticks={}, seeds={}",
            pop_size, generations, ticks_per_gen, num_seeds
        );
        println!();

        // ═══════════════════════════════════════════════════════════════════
        // Task 1: Stationary Foraging (Control)
        // ═══════════════════════════════════════════════════════════════════
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Task 1: Stationary Foraging (Control - No memory needed)       │");
        println!("└─────────────────────────────────────────────────────────────────┘");

        let mut simple_task1_finals = Vec::new();
        let mut lstm_task1_finals = Vec::new();

        println!("\n  SimpleBrain:");
        for seed in 0..num_seeds {
            let mut task1 = StationaryTask::new(StationaryConfig {
                grid_size,
                food_patch_count: 15,
                food_radius: 12.0,
                food_value: 10.0,
                sensor_range: 60.0,
            });
            let history = run_simple_evolution(
                &mut task1,
                pop_size,
                generations,
                ticks_per_gen,
                grid_size,
                seed as u64 * 1000,
            );
            simple_task1_finals.push(*history.last().unwrap_or(&0.0));
        }

        println!("\n  LstmBrain:");
        for seed in 0..num_seeds {
            let mut task1 = StationaryTask::new(StationaryConfig {
                grid_size,
                food_patch_count: 15,
                food_radius: 12.0,
                food_value: 10.0,
                sensor_range: 60.0,
            });
            let history = run_lstm_evolution(
                &mut task1,
                pop_size,
                generations,
                ticks_per_gen,
                grid_size,
                seed as u64 * 1000 + 500,
            );
            lstm_task1_finals.push(*history.last().unwrap_or(&0.0));
        }

        // ═══════════════════════════════════════════════════════════════════
        // Task 5: Delayed Reward (KEY TEST)
        // ═══════════════════════════════════════════════════════════════════
        println!("\n┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Task 5: Delayed Reward (KEY TEST - Memory required)            │");
        println!("└─────────────────────────────────────────────────────────────────┘");

        let mut simple_task5_finals = Vec::new();
        let mut lstm_task5_finals = Vec::new();

        println!("\n  SimpleBrain:");
        for seed in 0..num_seeds {
            let mut task5 = DelayedRewardTask::new(DelayedRewardConfig {
                grid_size,
                beacon_x: (grid_size as f32) / 2.0,
                beacon_y: (grid_size as f32) / 4.0,
                beacon_radius: 15.0,
                food_x: (grid_size as f32) / 2.0,
                food_y: (grid_size as f32) * 3.0 / 4.0,
                food_radius: 20.0,
                food_value: 50.0,
                unlock_duration: 150,
                sensor_range: 80.0,
            });
            let history = run_simple_evolution(
                &mut task5,
                pop_size,
                generations,
                ticks_per_gen,
                grid_size,
                seed as u64 * 2000,
            );
            simple_task5_finals.push(*history.last().unwrap_or(&0.0));
        }

        println!("\n  LstmBrain:");
        for seed in 0..num_seeds {
            let mut task5 = DelayedRewardTask::new(DelayedRewardConfig {
                grid_size,
                beacon_x: (grid_size as f32) / 2.0,
                beacon_y: (grid_size as f32) / 4.0,
                beacon_radius: 15.0,
                food_x: (grid_size as f32) / 2.0,
                food_y: (grid_size as f32) * 3.0 / 4.0,
                food_radius: 20.0,
                food_value: 50.0,
                unlock_duration: 150,
                sensor_range: 80.0,
            });
            let history = run_lstm_evolution(
                &mut task5,
                pop_size,
                generations,
                ticks_per_gen,
                grid_size,
                seed as u64 * 2000 + 500,
            );
            lstm_task5_finals.push(*history.last().unwrap_or(&0.0));
        }

        // ═══════════════════════════════════════════════════════════════════
        // Results Analysis
        // ═══════════════════════════════════════════════════════════════════
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  Results Analysis                                              ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        // Task 1
        let (mean_s1, std_s1) = compute_stats(&simple_task1_finals);
        let (mean_l1, std_l1) = compute_stats(&lstm_task1_finals);
        let d1 = cohens_d(&simple_task1_finals, &lstm_task1_finals);

        println!("  Task 1: Stationary Foraging (Control)");
        println!("  ────────────────────────────────────────");
        println!("    SimpleBrain: {:.2} ± {:.2}", mean_s1, std_s1);
        println!("    LstmBrain:   {:.2} ± {:.2}", mean_l1, std_l1);
        println!("    Cohen's d:   {:.3}", d1);

        if d1.abs() < 0.3 {
            println!("    → No significant difference (as expected)");
        } else if d1 > 0.3 {
            println!("    → LSTM better (unexpected)");
        } else {
            println!("    → SimpleBrain better");
        }

        // Task 5
        let (mean_s5, std_s5) = compute_stats(&simple_task5_finals);
        let (mean_l5, std_l5) = compute_stats(&lstm_task5_finals);
        let d5 = cohens_d(&simple_task5_finals, &lstm_task5_finals);

        println!("\n  Task 5: Delayed Reward (KEY TEST)");
        println!("  ────────────────────────────────────────");
        println!("    SimpleBrain: {:.2} ± {:.2}", mean_s5, std_s5);
        println!("    LstmBrain:   {:.2} ± {:.2}", mean_l5, std_l5);
        println!("    Cohen's d:   {:.3}", d5);

        if d5 > 0.8 {
            println!("    → LSTM DOMINATES (large effect) ✓");
        } else if d5 > 0.3 {
            println!("    → LSTM moderate advantage");
        } else if d5.abs() < 0.3 {
            println!("    → No significant difference");
        } else {
            println!("    → SimpleBrain better (unexpected)");
        }

        // ═══════════════════════════════════════════════════════════════════
        // Conclusions
        // ═══════════════════════════════════════════════════════════════════
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║  Conclusions                                                   ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let task1_lstm_better = d1 > 0.3;
        let task5_lstm_better = d5 > 0.3;

        if !task1_lstm_better && task5_lstm_better {
            println!("  ✓ HYPOTHESIS CONFIRMED:");
            println!("    - Task 1 (no memory): No LSTM advantage");
            println!("    - Task 5 (memory req): LSTM outperforms");
            println!();
            println!("  → LSTM is valuable for cognitive tasks requiring memory");
            println!("  → Recommended: Hybrid approach (WMMA for speed, LSTM for cognition)");
        } else if !task1_lstm_better && !task5_lstm_better {
            println!("  ✗ LSTM shows no advantage on either task");
            println!();
            if mean_s5 < 20.0 && mean_l5 < 20.0 {
                println!("    Note: Both brains struggled with Task 5");
                println!("    → Task may be too hard or need more generations");
            } else {
                println!("  → SimpleBrain is sufficient; LSTM overhead not justified");
            }
        } else if task1_lstm_better && task5_lstm_better {
            println!("  ⚠ LSTM better on BOTH tasks (unexpected for Task 1)");
            println!("    → LSTM may be learning better representations overall");
            println!("    → Or Task 1 has hidden sequential dependencies");
        } else {
            println!("  ⚠ Mixed results - SimpleBrain better on Task 5?");
            println!("    → Check implementation or increase generations");
        }

        println!("\n  Experiment complete.");
    }
}

fn main() {
    #[cfg(feature = "burn-compute")]
    experiment::run();

    #[cfg(not(feature = "burn-compute"))]
    {
        println!("EPIC 101: LSTM vs Simple NN Neuroevolution");
        println!();
        println!(
            "Run with: cargo run --release --example epic101_experiment --features burn-compute"
        );
    }
}
