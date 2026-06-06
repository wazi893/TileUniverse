//! Ising Problem Encoding for P-bit Networks
//!
//! Provides tools to encode combinatorial optimization problems
//! as Ising models that can be solved by p-bit networks.
//!
//! # Ising Model
//!
//! The Ising Hamiltonian is:
//! H = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ
//!
//! where s ∈ {-1, +1}.
//!
//! # QUBO (Quadratic Unconstrained Binary Optimization)
//!
//! Many problems are naturally expressed as QUBO:
//! minimize x^T Q x, where x ∈ {0, 1}
//!
//! This can be converted to Ising form.

use crate::qec::noise::SimpleRng;

/// An Ising problem specification
#[derive(Debug, Clone)]
pub struct IsingProblem {
    /// Number of spins
    pub n: usize,
    /// Coupling matrix J (flattened, symmetric)
    pub j: Vec<f64>,
    /// Bias vector h
    pub h: Vec<f64>,
    /// Optional: known ground state energy for benchmarking
    pub ground_energy: Option<f64>,
}

impl IsingProblem {
    /// Create an empty Ising problem
    pub fn new(n: usize) -> Self {
        Self {
            n,
            j: vec![0.0; n * n],
            h: vec![0.0; n],
            ground_energy: None,
        }
    }

    /// Set coupling J_ij (automatically symmetric)
    pub fn set_coupling(&mut self, i: usize, j: usize, value: f64) {
        debug_assert!(i < self.n && j < self.n);
        self.j[i * self.n + j] = value;
        self.j[j * self.n + i] = value;
    }

    /// Get coupling J_ij
    pub fn coupling(&self, i: usize, j: usize) -> f64 {
        self.j[i * self.n + j]
    }

    /// Set bias h_i
    pub fn set_bias(&mut self, i: usize, value: f64) {
        self.h[i] = value;
    }

    /// Compute energy for a given spin configuration
    pub fn energy(&self, spins: &[i8]) -> f64 {
        assert_eq!(spins.len(), self.n);
        let mut e = 0.0;

        // Coupling term
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                e -= self.j[i * self.n + j] * (spins[i] as f64) * (spins[j] as f64);
            }
        }

        // Bias term
        for i in 0..self.n {
            e -= self.h[i] * (spins[i] as f64);
        }

        e
    }

    /// Compute energy for a given binary configuration (0/1)
    pub fn energy_binary(&self, state: &[u8]) -> f64 {
        let spins: Vec<i8> = state.iter().map(|&x| 2 * (x as i8) - 1).collect();
        self.energy(&spins)
    }
}

/// Adjacency matrix representation of a graph for MaxCut
#[derive(Debug, Clone)]
pub struct MaxCutGraph {
    /// Number of vertices
    pub n: usize,
    /// Adjacency matrix (flattened, symmetric)
    pub adj: Vec<f64>,
}

impl MaxCutGraph {
    /// Create an empty graph
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![0.0; n * n],
        }
    }

    /// Add an edge with weight
    pub fn add_edge(&mut self, i: usize, j: usize, weight: f64) {
        debug_assert!(i < self.n && j < self.n && i != j);
        self.adj[i * self.n + j] = weight;
        self.adj[j * self.n + i] = weight;
    }

    /// Create a random graph with given edge probability
    pub fn random(n: usize, edge_prob: f64, seed: u64) -> Self {
        let mut graph = Self::new(n);
        let mut rng = SimpleRng::new(seed);

        for i in 0..n {
            for j in (i + 1)..n {
                if rng.next_f64() < edge_prob {
                    graph.add_edge(i, j, 1.0);
                }
            }
        }

        graph
    }

    /// Create a complete graph (all edges)
    pub fn complete(n: usize) -> Self {
        let mut graph = Self::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                graph.add_edge(i, j, 1.0);
            }
        }
        graph
    }

    /// Create a cycle graph
    pub fn cycle(n: usize) -> Self {
        let mut graph = Self::new(n);
        for i in 0..n {
            graph.add_edge(i, (i + 1) % n, 1.0);
        }
        graph
    }

    /// Create a 2D grid graph
    pub fn grid(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut graph = Self::new(n);

        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                // Right neighbor
                if c + 1 < cols {
                    graph.add_edge(i, i + 1, 1.0);
                }
                // Down neighbor
                if r + 1 < rows {
                    graph.add_edge(i, i + cols, 1.0);
                }
            }
        }

        graph
    }

    /// Convert to Ising problem
    ///
    /// MaxCut: maximize Σ_{(i,j)∈E} w_ij * (1 - s_i * s_j) / 2
    /// Ising:  minimize -Σ J_ij s_i s_j
    ///
    /// So J_ij = -w_ij / 2 (negative = antiferromagnetic)
    pub fn to_ising(&self) -> IsingProblem {
        let mut problem = IsingProblem::new(self.n);

        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let w = self.adj[i * self.n + j];
                if w != 0.0 {
                    // Antiferromagnetic coupling for MaxCut
                    problem.set_coupling(i, j, -w / 2.0);
                }
            }
        }

        problem
    }

    /// Compute cut value for a given partition
    pub fn cut_value(&self, partition: &[u8]) -> f64 {
        let mut cut = 0.0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if partition[i] != partition[j] {
                    cut += self.adj[i * self.n + j];
                }
            }
        }
        cut
    }

    /// Number of edges
    pub fn num_edges(&self) -> usize {
        let mut count = 0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if self.adj[i * self.n + j] != 0.0 {
                    count += 1;
                }
            }
        }
        count
    }
}

/// QUBO (Quadratic Unconstrained Binary Optimization) matrix
///
/// Minimize x^T Q x where x ∈ {0, 1}^n
#[derive(Debug, Clone)]
pub struct QuboMatrix {
    /// Size
    pub n: usize,
    /// Q matrix (can be asymmetric, upper triangular is sufficient)
    pub q: Vec<f64>,
}

impl QuboMatrix {
    /// Create empty QUBO
    pub fn new(n: usize) -> Self {
        Self {
            n,
            q: vec![0.0; n * n],
        }
    }

    /// Set Q_ij coefficient
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.q[i * self.n + j] = value;
    }

    /// Get Q_ij
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.q[i * self.n + j]
    }

    /// Evaluate QUBO objective for a binary vector
    pub fn evaluate(&self, x: &[u8]) -> f64 {
        assert_eq!(x.len(), self.n);
        let mut val = 0.0;

        for i in 0..self.n {
            for j in 0..self.n {
                val += self.q[i * self.n + j] * (x[i] as f64) * (x[j] as f64);
            }
        }

        val
    }

    /// Convert QUBO to Ising problem
    ///
    /// Using x = (1 + s) / 2, where s ∈ {-1, +1}
    pub fn to_ising(&self) -> IsingProblem {
        let mut problem = IsingProblem::new(self.n);

        // Convert Q to J and h
        // x_i x_j = (1 + s_i)(1 + s_j)/4 = (1 + s_i + s_j + s_i s_j)/4
        // Q_ij x_i x_j contributes:
        //   Q_ij/4 (constant)
        //   Q_ij/4 s_i (linear in s_i)
        //   Q_ij/4 s_j (linear in s_j)
        //   Q_ij/4 s_i s_j (quadratic)

        for i in 0..self.n {
            for j in 0..self.n {
                let qij = self.q[i * self.n + j];
                if qij != 0.0 {
                    if i == j {
                        // Diagonal: Q_ii x_i = Q_ii (1 + s_i)/2
                        // Contributes Q_ii/2 to constant, Q_ii/2 to h_i
                        problem.h[i] -= qij / 2.0; // Note: Ising has -h_i s_i
                    } else {
                        // Off-diagonal: handled once per pair
                        if i < j {
                            let qij_sym = qij + self.q[j * self.n + i];
                            // Coupling contribution
                            problem.set_coupling(i, j, problem.coupling(i, j) - qij_sym / 4.0);
                            // Linear contributions
                            problem.h[i] -= qij_sym / 4.0;
                            problem.h[j] -= qij_sym / 4.0;
                        }
                    }
                }
            }
        }

        problem
    }
}

/// Pre-built problem instances
pub mod problems {
    use super::*;

    /// Random MaxCut problem
    pub fn random_maxcut(n: usize, edge_prob: f64, seed: u64) -> IsingProblem {
        MaxCutGraph::random(n, edge_prob, seed).to_ising()
    }

    /// SK (Sherrington-Kirkpatrick) model
    ///
    /// Fully connected Ising with random ±1 couplings
    pub fn sk_model(n: usize, seed: u64) -> IsingProblem {
        let mut problem = IsingProblem::new(n);
        let mut rng = SimpleRng::new(seed);

        for i in 0..n {
            for j in (i + 1)..n {
                // Random ±1 coupling
                let j_val = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
                problem.set_coupling(i, j, j_val);
            }
        }

        problem
    }

    /// Random field Ising model (RFIM)
    ///
    /// Ferromagnetic couplings with random biases
    pub fn rfim(n: usize, disorder_strength: f64, seed: u64) -> IsingProblem {
        let mut problem = IsingProblem::new(n);
        let mut rng = SimpleRng::new(seed);

        // Ferromagnetic nearest-neighbor (1D chain for simplicity)
        for i in 0..(n - 1) {
            problem.set_coupling(i, i + 1, 1.0);
        }

        // Random biases
        for i in 0..n {
            let h = (2.0 * rng.next_f64() - 1.0) * disorder_strength;
            problem.set_bias(i, h);
        }

        problem
    }

    /// Number partitioning problem
    ///
    /// Given numbers a_1, ..., a_n, partition into two sets
    /// to minimize |sum(S1) - sum(S2)|
    pub fn number_partition(numbers: &[i32]) -> IsingProblem {
        let n = numbers.len();
        let mut qubo = QuboMatrix::new(n);

        // Minimize (Σ a_i (2x_i - 1))^2 = (Σ a_i s_i)^2
        // = Σ_ij a_i a_j s_i s_j
        for i in 0..n {
            for j in 0..n {
                qubo.set(i, j, (numbers[i] * numbers[j]) as f64);
            }
        }

        qubo.to_ising()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxcut_cycle() {
        // Cycle graph: optimal cut is all edges for even n
        let graph = MaxCutGraph::cycle(4);
        let _ising = graph.to_ising();

        // Alternating partition should cut all edges
        let partition = vec![0, 1, 0, 1];
        let cut = graph.cut_value(&partition);
        assert_eq!(cut, 4.0);
    }

    #[test]
    fn test_qubo_evaluation() {
        let mut qubo = QuboMatrix::new(2);
        qubo.set(0, 0, 1.0); // x0^2 = x0
        qubo.set(1, 1, 2.0); // x1^2 = x1
        qubo.set(0, 1, 3.0); // x0*x1

        assert_eq!(qubo.evaluate(&[0, 0]), 0.0);
        assert_eq!(qubo.evaluate(&[1, 0]), 1.0);
        assert_eq!(qubo.evaluate(&[0, 1]), 2.0);
        assert_eq!(qubo.evaluate(&[1, 1]), 1.0 + 2.0 + 3.0);
    }

    #[test]
    fn test_energy_consistency() {
        let graph = MaxCutGraph::complete(4);
        let ising = graph.to_ising();

        // All in same partition (no cut)
        let e1 = ising.energy_binary(&[0, 0, 0, 0]);
        // Split partition
        let e2 = ising.energy_binary(&[0, 0, 1, 1]);

        // Split should have lower energy (larger cut)
        assert!(e2 < e1, "e1={}, e2={}", e1, e2);
    }
}
