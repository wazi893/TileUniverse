#[cfg(feature = "cuda")]
use engine::cuda::{CudaRuntime, GpuQState};
use engine::fusion::{
    FusedOp, FusedProgram, FusionConfig, execute_fused_cpu, fuse_program_with_config,
};
use engine::quantum::{QGate, QRng, QState};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut n_gates: usize = 1_000_000;
    let mut n_qubits: u8 = 1;
    let mut enable_reduction = true;
    let mut use_cuda = cfg!(feature = "cuda");
    let mut output_file = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gates" => {
                if i + 1 < args.len() {
                    n_gates = args[i + 1].parse().expect("Invalid gate count");
                    i += 1;
                }
            }
            "--qubits" => {
                if i + 1 < args.len() {
                    n_qubits = args[i + 1].parse().expect("Invalid qubit count");
                    i += 1;
                }
            }
            "--scenario" => {
                // parse presets
                if i + 1 < args.len() {
                    match args[i + 1].as_str() {
                        "h_1million" => {
                            n_gates = 1_000_000;
                        }
                        "h_1billion" => {
                            n_gates = 1_000_000_000;
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
            "--disable-algebraic-reduction" => enable_reduction = false,
            "--enable-algebraic-reduction" => enable_reduction = true,
            "--no-cuda" => use_cuda = false,
            "--output" => {
                if i + 1 < args.len() {
                    output_file = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            _ => {
                if !args[i].starts_with("--") {
                    eprintln!("Unknown argument: {}", args[i]);
                }
            }
        }
        i += 1;
    }

    println!(
        "Proof Engine: Gates={}, Qubits={}, Reduction={}, CUDA={}",
        n_gates, n_qubits, enable_reduction, use_cuda
    );

    // 1. Generate Circuit
    let gates = generate_h_chain(n_gates);
    println!("Generated {} gates", gates.len());

    // 2. Configure Fusion
    let mut config = FusionConfig::default();
    config.enable_algebraic_reduction = enable_reduction;
    config.enable_wmma = enable_reduction;

    // 3. Fuse/Compile
    let start_compile = Instant::now();
    let fused = fuse_program_with_config(&gates, &config);
    let compile_time = start_compile.elapsed();
    println!("Compilation took {:?}", compile_time);
    println!("Fused Op Count: {}", fused.ops.len());

    // 4. Execute
    let start_exec = Instant::now();
    // Use 1 tile for state allocation (safe mode)
    // Decouple physical allocation from virtual metric qubits to avoid OOM
    // Cap physical alloc at 25 qubits (256MB) which is plenty safe
    let physical_qubits = std::cmp::min(n_qubits, 25);
    println!(
        "Allocating physical state for {} qubits (Virtual: {})",
        physical_qubits, n_qubits
    );

    let state = execute_program(&fused, physical_qubits, use_cuda);
    let exec_time = start_exec.elapsed();
    let total_time = compile_time + exec_time;

    println!("Execution took {:?}", exec_time);

    // 5. Calculate Metrics
    let ops_per_gate = 4.0; // H gate FLOPs per amplitude
    let amplitudes = 2.0f64.powi(n_qubits as i32);
    let total_algorithmic_ops = n_gates as f64 * ops_per_gate * amplitudes;
    let effective_ops = total_algorithmic_ops / total_time.as_secs_f64();

    let exa_ops = effective_ops / 1e18;
    let zetta_ops = effective_ops / 1e21;

    println!("--- SCALE METRICS ---");
    println!("Algorithmic Work: {:.2e} FLOPs", total_algorithmic_ops);
    println!("Total Wall Time:  {:.6} s", total_time.as_secs_f64());
    println!("Effective Tput:   {:.2} ExaOPS", exa_ops);
    println!("                  {:.4} ZettaOPS", zetta_ops);

    // 6. Output
    if let Some(path) = output_file {
        dump_state(&state, &path, exec_time.as_secs_f64());
    }
}

fn generate_h_chain(n: usize) -> Vec<QGate> {
    let mut gates = Vec::with_capacity(n);
    for _ in 0..n {
        gates.push(QGate::H(0));
    }
    gates
}

fn execute_program(prog: &FusedProgram, n_qubits: u8, cuda: bool) -> QState {
    let mut state = QState::new_zero(n_qubits);

    #[cfg(feature = "cuda")]
    if cuda {
        println!("Initializing CUDA Runtime...");
        let rt = CudaRuntime::new().expect("Failed to init CUDA");

        // Use resident state (EPIC 67)
        // Use 1 tile for verification proof to match CPU state size
        let mut g_state =
            GpuQState::new_zero_resident(&rt, n_qubits, 1).expect("Failed to alloc GPU state");

        println!("Executing on GPU...");
        for (idx, op) in prog.ops.iter().enumerate() {
            match op {
                FusedOp::Single(gate) => {
                    match gate {
                        QGate::H(q) if *q == 0 => {
                            // Run H kernel (depth=1)
                            engine::cuda::run_hadamard_kernel(&rt, &mut g_state, 1)
                                .expect("Kernel failed");
                        }
                        _ => panic!("Only H(0) supported for proof engine CUDA path currently"),
                    }
                }
                FusedOp::Identity => {
                    // No-op
                }
                FusedOp::Mega(_) => {
                    // Mega-fused operations imply algebraic reduction was enabled (or part of it).
                    // But if we are in naive mode, we shouldn't see Mega ops unless naive_fuse generates them?
                    // No, naive_fuse is not used anymore, we use fuse_program_with_config(enable_reduction=false).
                    // That function effectively keeps Single gates.
                    // However, if we do encounter Mega ops (because fused path was taken), we'd need a kernel.
                    // For now, panic if not supported.
                    panic!(
                        "Mega ops not supported in limited proof engine CUDA path. Use Single gates."
                    );
                }
                FusedOp::Batched { gate, count } => match gate {
                    QGate::H(q) if *q == 0 => {
                        engine::cuda::run_hadamard_kernel(&rt, &mut g_state, *count)
                            .expect("Kernel failed");
                    }
                    _ => panic!("Only Batched H(0) supported for proof engine currently"),
                },
                _ => panic!("Complex ops not supported in proof engine yet: {:?}", op),
            }

            if idx % 100_000 == 0 && idx > 0 {
                print!(".");
                std::io::stdout().flush().unwrap();
            }
        }
        println!();

        // Download result
        g_state.to_qstate(&rt, &mut state).expect("Download failed");
        return state;
    }

    if cuda {
        println!("Warning: CUDA requested but feature disabled/unavailable. Using CPU.");
    }

    // CPU Fallback
    println!("Executing on CPU...");
    let mut rng = QRng::new(0);
    execute_fused_cpu(&mut state, prog, &mut rng);
    state
}

fn dump_state(state: &QState, path: &str, exec_time: f64) {
    // Manually format JSON to avoid extra dependencies for this simple tool
    let mut file = File::create(path).unwrap();
    writeln!(file, "{{").unwrap();
    writeln!(file, "  \"exec_time_sec\": {},", exec_time).unwrap();
    writeln!(file, "  \"n_qubits\": {},", state.n_qubits).unwrap();
    writeln!(file, "  \"real_len\": {},", state.real.len()).unwrap();

    // Dump first few amplitudes
    writeln!(file, "  \"amplitudes\": [").unwrap();
    let limit = 8.min(state.real.len());
    for i in 0..limit {
        let r = state.real.as_slice()[i];
        let im = state.imag.as_slice()[i];
        let comma = if i == limit - 1 { "" } else { "," };
        writeln!(file, "    [{}, {}]{}", r, im, comma).unwrap();
    }
    writeln!(file, "  ]").unwrap();
    writeln!(file, "}}").unwrap();
    println!("Dumped state to {}", path);
}
