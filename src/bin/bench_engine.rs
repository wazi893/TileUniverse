use std::env;
use std::time::Instant;

#[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
use engine::quantum_jit;
use engine::simulation::Simulation;
use engine::tilemap::{HEIGHT, WIDTH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchMode {
    FullTick,
    LogicEvalOnly,
    LogicDirtyOnly,
    FieldDiffuseOnly,
    EcosystemOnly,
    FieldReactOnly,
    FieldInteractOnly,
    FieldsAll,
    FieldDecayOnly,
    FieldFeedbackOnly,
    QuantumScalarOnly,
    QuantumBackendOnly,
    CombinedLogicQuantum,
    KernelFarmBench,
    /// EPIC 66: CUDA GPU benchmark mode
    #[cfg(feature = "cuda")]
    CudaGpu,
    /// EPIC 67: CUDA GPU-resident benchmark (state stays in VRAM)
    #[cfg(feature = "cuda")]
    CudaGpuResident,
    /// EPIC 67: CUDA Tensor Core benchmark (FP16)
    #[cfg(feature = "cuda")]
    CudaTensor,
    /// EPIC 67 PERF: Batched GPU execution (single kernel launch)
    #[cfg(feature = "cuda")]
    CudaBatched,
    /// EPIC 67 T2: WMMA Tensor Core benchmark (real mma.sync) - LEGACY
    #[cfg(feature = "cuda")]
    CudaWmma,
    /// EPIC 69-72: Modern GPU FP32 backend
    #[cfg(feature = "cuda")]
    CudaFp32Modern,
    /// EPIC 69-72: WMMA Aligned (run_wmma_hadamard_cached)
    #[cfg(feature = "cuda")]
    CudaWmmaAligned,
    /// EPIC 71: WMMA Packed (run_wmma_packed for misaligned spans)
    #[cfg(feature = "cuda")]
    CudaWmmaPacked,
    /// EPIC 66-72: Full backend comparison summary
    #[cfg(feature = "cuda")]
    CudaEpicSummary,
    /// EPIC 74A: Parallel Logic Simulation (Path B)
    ParallelLogic,
    /// EPIC 74B: GPU Tile Evaluation (Path A)
    #[cfg(feature = "cuda")]
    GpuTileEval,
    /// EPIC 75.1: Multi-World GPU Tile Evaluation
    #[cfg(feature = "cuda")]
    GpuTileMulti,
    /// EPIC 75.3: Depth-Batched GPU Tile Evaluation
    #[cfg(feature = "cuda")]
    GpuTileDepth,
    /// EPIC 75.3B: Shared Memory Stencil GPU Tile Evaluation
    #[cfg(feature = "cuda")]
    GpuTileStencil,
    /// EPIC 75.4: GPU vs CPU Correctness Validation
    #[cfg(feature = "cuda")]
    GpuCorrectness,
    /// EPIC 120: Packed 1-bit Tile Evaluation (64 tiles per u64)
    #[cfg(feature = "cuda")]
    PackedTiles,
    /// EPIC 121: TileAnneal Ising Optimization
    #[cfg(feature = "cuda")]
    Ising,
}

fn parse_mode(args: &[String]) -> BenchMode {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--mode" {
            if let Some(v) = it.next() {
                let s = v.to_ascii_lowercase();
                return match s.as_str() {
                    "logicevalonly" | "logic-eval-only" | "eval" => BenchMode::LogicEvalOnly,
                    "logicdirtyonly" | "logic-dirty-only" | "dirty" => BenchMode::LogicDirtyOnly,
                    "fielddiffuseonly" | "diffuse" => BenchMode::FieldDiffuseOnly,
                    "ecosystemonly" | "ecosystem" => BenchMode::EcosystemOnly,
                    "fieldreactonly" | "react" => BenchMode::FieldReactOnly,
                    "fieldinteractonly" | "interact" => BenchMode::FieldInteractOnly,
                    "fieldsall" | "fields" => BenchMode::FieldsAll,
                    "fielddecayonly" | "decay" => BenchMode::FieldDecayOnly,
                    "fieldfeedbackonly" | "feedback" => BenchMode::FieldFeedbackOnly,
                    "quantumscalaronly" | "quantum" | "qscalar" => BenchMode::QuantumScalarOnly,
                    "quantumbackendonly" | "qbackend" => BenchMode::QuantumBackendOnly,
                    "combinedlogicquantum" | "combined" | "lq" => BenchMode::CombinedLogicQuantum,
                    "kernelfarmbench" | "farm" => BenchMode::KernelFarmBench,
                    #[cfg(feature = "cuda")]
                    "cudagpu" | "cuda" | "gpu" => BenchMode::CudaGpu,
                    #[cfg(feature = "cuda")]
                    "cudagpuresident" | "cuda-resident" | "gpu-resident" => {
                        BenchMode::CudaGpuResident
                    }
                    #[cfg(feature = "cuda")]
                    "cudatensor" | "cuda-tensor" | "tensor" => BenchMode::CudaTensor,
                    #[cfg(feature = "cuda")]
                    "cudabatched" | "cuda-batched" | "batched" => BenchMode::CudaBatched,
                    #[cfg(feature = "cuda")]
                    "cudawmma" | "cuda-wmma" | "wmma" => BenchMode::CudaWmma,
                    #[cfg(feature = "cuda")]
                    "cudafp32modern" | "cuda-fp32-modern" | "fp32-modern" | "fp32modern" => {
                        BenchMode::CudaFp32Modern
                    }
                    #[cfg(feature = "cuda")]
                    "cudawmmaaligned" | "cuda-wmma-aligned" | "wmma-aligned" | "wmmaaligned" => {
                        BenchMode::CudaWmmaAligned
                    }
                    #[cfg(feature = "cuda")]
                    "cudawmmapacked" | "cuda-wmma-packed" | "wmma-packed" | "wmmapacked" => {
                        BenchMode::CudaWmmaPacked
                    }
                    #[cfg(feature = "cuda")]
                    "cudaepicsummary" | "cuda-epic-summary" | "epic-summary" | "epicsummary"
                    | "summary" => BenchMode::CudaEpicSummary,
                    "parallellogic" | "parallel-logic" | "parallel" => BenchMode::ParallelLogic,
                    #[cfg(feature = "cuda")]
                    "gputileeval" | "gpu-tile-eval" | "gpu-tile" | "tile-gpu" => {
                        BenchMode::GpuTileEval
                    }
                    #[cfg(feature = "cuda")]
                    "gputilemulti" | "gpu-tile-multi" | "tile-multi" | "multi-world" => {
                        BenchMode::GpuTileMulti
                    }
                    #[cfg(feature = "cuda")]
                    "gputiledepth" | "gpu-tile-depth" | "tile-depth" | "depth-batch" => {
                        BenchMode::GpuTileDepth
                    }
                    #[cfg(feature = "cuda")]
                    "gputilestencil" | "gpu-tile-stencil" | "tile-stencil" | "stencil" => {
                        BenchMode::GpuTileStencil
                    }
                    #[cfg(feature = "cuda")]
                    "gpucorrectness" | "gpu-correctness" | "correctness" | "validate" => {
                        BenchMode::GpuCorrectness
                    }
                    #[cfg(feature = "cuda")]
                    "packedtiles" | "packed-tiles" | "packed-1bit" | "packed" | "1bit" => {
                        BenchMode::PackedTiles
                    }
                    #[cfg(feature = "cuda")]
                    "ising" | "tileanneal" | "anneal" | "spin" => BenchMode::Ising,
                    _ => BenchMode::FullTick,
                };
            }
        }
    }
    BenchMode::FullTick
}

fn parse_arg_usize(args: &[String], name: &str, default: usize) -> usize {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                if let Ok(n) = v.parse::<usize>() {
                    return n;
                }
            }
        }
    }
    default
}

fn parse_arg_u32(args: &[String], name: &str, default: u32) -> u32 {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                if let Ok(n) = v.parse::<u32>() {
                    return n;
                }
            }
        }
    }
    default
}

fn parse_arg_u8(args: &[String], name: &str, default: u8) -> u8 {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                if let Ok(n) = v.parse::<u8>() {
                    return n;
                }
            }
        }
    }
    default
}

fn parse_arg_backend(args: &[String], name: &str, default: &str) -> String {
    let mut it = args.iter();
    let mut out = default.to_string();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                out = v.to_string();
            }
        }
    }
    out
}

fn parse_arg_string(args: &[String], name: &str, default: &str) -> String {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                return v.to_string();
            }
        }
    }
    default.to_string()
}

fn parse_arg_i32(args: &[String], name: &str, default: i32) -> i32 {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                if let Ok(n) = v.parse::<i32>() {
                    return n;
                }
            }
        }
    }
    default
}

#[cfg(feature = "cuda")]
fn parse_arg_f32(args: &[String], name: &str, default: f32) -> f32 {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                if let Ok(n) = v.parse::<f32>() {
                    return n;
                }
            }
        }
    }
    default
}

fn parse_arg_bool(args: &[String], name: &str, default: bool) -> bool {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                let s = v.to_ascii_lowercase();
                return s == "1" || s == "on" || s == "true" || s == "yes";
            }
        }
    }
    default
}

// EPIC 55: small helpers to construct standard scenarios deterministically
fn build_h_program(nq: u8) -> Vec<engine::quantum::QGate> {
    let mut p = Vec::new();
    for q in 0..nq {
        p.push(engine::quantum::QGate::H(q));
    }
    if nq >= 2 {
        p.push(engine::quantum::QGate::CNot(0, 1));
    }
    if nq >= 4 {
        p.push(engine::quantum::QGate::Phase(2, 0.25));
    }
    p
}

// EPIC 60: Deep kernel scenario - H gates (with depth via JIT_H_DEPTH env) + pairwise CNOT ladder
fn build_deep_h_cnot_program(nq: u8) -> Vec<engine::quantum::QGate> {
    let mut p = Vec::new();
    // Apply H to all qubits (JIT will apply depth times based on JIT_H_DEPTH env var)
    for q in 0..nq {
        p.push(engine::quantum::QGate::H(q));
    }
    // Apply pairwise CNOT ladder (0,1), (2,3), (4,5), ...
    let mut q = 0;
    while q + 1 < nq {
        p.push(engine::quantum::QGate::CNot(q, q + 1));
        q += 2;
    }
    p
}

fn apply_scenario(sim: &mut Simulation, scenario: &str, print_info: bool) {
    match scenario {
        "single_4q" => {
            let nq = 4u8;
            let state = engine::quantum::QState::new_zero(nq);
            let program = build_h_program(nq);
            sim.register_qdemo_tile(12, 12, state, program, 0xA001);
        }
        "single_8q" => {
            let nq = 8u8;
            let state = engine::quantum::QState::new_zero(nq);
            let program = build_h_program(nq);
            sim.register_qdemo_tile(14, 14, state, program, 0xA002);
        }
        "single_12q" => {
            let nq = 12u8;
            let state = engine::quantum::QState::new_zero(nq);
            let program = build_h_program(nq);
            sim.register_qdemo_tile(16, 16, state, program, 0xA003);
        }
        "deep_12q_hcnot" => {
            let nq = 12u8;
            let state = engine::quantum::QState::new_zero(nq);
            let program = build_deep_h_cnot_program(nq);
            let num_gates = program.len();
            sim.register_qdemo_tile(16, 16, state, program, 0xA004);
            if print_info {
                println!(
                    "scenario_info: deep_12q_hcnot qubits={} gates={} (12xH + 6xCNOT pairwise ladder)",
                    nq, num_gates
                );
            }
        }
        "multi_4q" => {
            let nq = 4u8;
            let program = build_h_program(nq);
            let mut seed: u64 = 0xB000;
            let mut placed = 0;
            for y in (8..HEIGHT.saturating_sub(8)).step_by(32) {
                for x in (8..WIDTH.saturating_sub(8)).step_by(32) {
                    let state = engine::quantum::QState::new_zero(nq);
                    sim.register_qdemo_tile(x, y, state, program.clone(), seed);
                    seed = seed.wrapping_add(1);
                    placed += 1;
                    if placed >= 64 {
                        break;
                    }
                }
                if placed >= 64 {
                    break;
                }
            }
            if print_info {
                println!(
                    "scenario_info: multi_4q tiles={} qubits_per_tile={}",
                    placed, nq
                );
            }
        }
        // EPIC 61: Multi-tile SIMD scenarios (tile_count=4, F32X4 SIMD)
        "multitile_4q_h" => {
            let nq = 4u8;
            let tile_count = 4u16;
            let state = engine::quantum::QState::new_zero_multitile(nq, tile_count);
            let program = build_h_program(nq);
            let num_gates = program.len();
            sim.register_qdemo_tile(16, 16, state, program, 0xC001);
            if print_info {
                println!(
                    "scenario_info: multitile_4q_h qubits={} tiles={} gates={} (SIMD F32X4)",
                    nq, tile_count, num_gates
                );
            }
        }
        "multitile_8q_h" => {
            let nq = 8u8;
            let tile_count = 4u16;
            let state = engine::quantum::QState::new_zero_multitile(nq, tile_count);
            let program = build_h_program(nq);
            let num_gates = program.len();
            sim.register_qdemo_tile(16, 16, state, program, 0xC002);
            if print_info {
                println!(
                    "scenario_info: multitile_8q_h qubits={} tiles={} gates={} (SIMD F32X4)",
                    nq, tile_count, num_gates
                );
            }
        }
        "multitile_4q_hcnot" => {
            let nq = 4u8;
            let tile_count = 4u16;
            let state = engine::quantum::QState::new_zero_multitile(nq, tile_count);
            let program = build_deep_h_cnot_program(nq);
            let num_gates = program.len();
            sim.register_qdemo_tile(16, 16, state, program, 0xC003);
            if print_info {
                println!(
                    "scenario_info: multitile_4q_hcnot qubits={} tiles={} gates={} (SIMD F32X4, H+CNOT)",
                    nq, tile_count, num_gates
                );
            }
        }
        // EPIC 61 Phase 3: Single-tile 4q H+CNOT baseline for comparison
        "single_4q_hcnot" => {
            let nq = 4u8;
            let state = engine::quantum::QState::new_zero(nq);
            let program = build_deep_h_cnot_program(nq);
            let num_gates = program.len();
            sim.register_qdemo_tile(16, 16, state, program, 0xC004);
            if print_info {
                println!(
                    "scenario_info: single_4q_hcnot qubits={} gates={} (scalar baseline, H+CNOT)",
                    nq, num_gates
                );
            }
        }
        _ => {}
    }
}

#[cfg(all(feature = "perf-bench", feature = "quantum_jit"))]
fn scenario_qubits(s: &str) -> u8 {
    match s {
        "single_4q" => 4,
        "single_8q" => 8,
        "single_12q" => 12,
        "deep_12q_hcnot" => 12,
        "multi_4q" => 4,
        "multitile_4q_h" => 4,     // EPIC 61
        "multitile_8q_h" => 8,     // EPIC 61
        "multitile_4q_hcnot" => 4, // EPIC 61 Phase 3
        "single_4q_hcnot" => 4,    // EPIC 61 Phase 3
        _ => 0,
    }
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() {
    // Args: --ticks N (default 5000), --warmup W (default 256), --mode <FullTick|LogicEvalOnly|LogicDirtyOnly>, --reps R, --iters I
    let args: Vec<String> = env::args().skip(1).collect();
    let ticks: u32 = parse_arg_u32(&args, "--ticks", 5_000);
    let warmup: u32 = parse_arg_u32(&args, "--warmup", 256);
    let mode = parse_mode(&args);
    let reps: usize = parse_arg_usize(&args, "--reps", 1);
    let iters: usize = parse_arg_usize(&args, "--iters", 1);
    let dispatch = {
        let mut it = args.iter();
        let mut d = "branchless".to_string();
        while let Some(a) = it.next() {
            if a == "--dispatch" {
                if let Some(v) = it.next() {
                    d = v.to_string();
                }
            }
        }
        d
    };
    let unchecked = {
        let mut it = args.iter();
        let mut u = "on".to_string();
        while let Some(a) = it.next() {
            if a == "--unchecked" {
                if let Some(v) = it.next() {
                    u = v.to_string();
                }
            }
        }
        u.to_ascii_lowercase() == "on"
    };
    let fields_dispatch = {
        let mut it = args.iter();
        let mut f = "interior".to_string();
        while let Some(a) = it.next() {
            if a == "--fields-dispatch" {
                if let Some(v) = it.next() {
                    f = v.to_string();
                }
            }
        }
        f
    };
    let _qubits: u8 = parse_arg_u8(&args, "--qubits", 8);
    let qdemo_qubits: u8 = parse_arg_u8(&args, "--qdemo-qubits", 0);
    let backend_str = parse_arg_backend(&args, "--backend", "scalar");
    let scenario = parse_arg_string(&args, "--scenario", "none");
    let jit_unroll: i32 = parse_arg_i32(&args, "--jit-unroll", 2);
    let jit_12q_fastpath: bool = parse_arg_bool(&args, "--jit-12q-fastpath", false);
    #[cfg(feature = "perf-bench")]
    let repeats: usize = parse_arg_usize(&args, "--repeat", 5);
    #[cfg(not(feature = "perf-bench"))]
    let _repeats: usize = parse_arg_usize(&args, "--repeat", 5);

    // Build a standard world sized to constants
    let mut sim = Simulation::new_standard_benchmark_world();
    // Optional: list known kernels/backends for debugging
    if has_flag(&args, "--farm-list") {
        println!("farm_kernels:");
        println!("  kernel_id: logic   kind=Classical backend=scalar");
        println!("  kernel_id: fields  kind=Classical backend=scalar");
        println!("  kernel_id: organism kind=Classical backend=scalar");
        println!("  kernel_id: quantum backend=scalar|avx2|avx512|jit(opt)");
        // Exit early; this helper is informational only
        return;
    }
    if qdemo_qubits > 0 {
        let state = engine::quantum::QState::new_zero(qdemo_qubits);
        let program = vec![engine::quantum::QGate::Measure(0)];
        sim.register_qdemo_tile(10, 10, state, program, 0x1111_2222);
    }
    // EPIC 55: Stress scenarios for quantum throughput
    if scenario != "none" {
        apply_scenario(&mut sim, &scenario, true);
    }
    // JIT: prewarm H kernels for the selected qubit count to ensure fast-path hits.
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    {
        if backend_str.to_ascii_lowercase() == "jit" {
            let nq = if qdemo_qubits > 0 {
                qdemo_qubits
            } else {
                #[cfg(all(feature = "perf-bench", feature = "quantum_jit"))]
                {
                    scenario_qubits(&scenario)
                }
                #[cfg(not(all(feature = "perf-bench", feature = "quantum_jit")))]
                {
                    0
                }
            };
            if nq > 0 {
                engine::quantum_jit::prewarm_h_kernels(nq);
            }
        }
    }
    // Wire --jit-unroll flag to JIT codegen via env var
    unsafe {
        match jit_unroll {
            0 => {
                std::env::set_var("JIT_UNROLL", "0");
            }
            2 => {
                std::env::set_var("JIT_UNROLL", "2");
                println!("jit_unroll: 2");
            }
            other => {
                eprintln!("invalid --jit-unroll={}, using 0", other);
                std::env::set_var("JIT_UNROLL", "0");
            }
        }
    }
    // Optional: allow explicitly enabling 12q unrolled fast-path (default off for stability)
    unsafe {
        if jit_12q_fastpath {
            std::env::set_var("JIT_12Q_FASTPATH", "1");
            println!("jit_12q_fastpath: on");
        } else {
            std::env::set_var("JIT_12Q_FASTPATH", "0");
        }
    }

    // Warm the caches/branch predictors to reduce first-iteration jitter
    #[cfg(feature = "perf-bench")]
    {
        for _ in 0..warmup {
            let _ = sim.tick_bench();
        }
    }
    #[cfg(not(feature = "perf-bench"))]
    {
        for _ in 0..warmup {
            sim.tick();
        }
    }

    // Timed loop
    #[cfg(feature = "perf-bench")]
    let mut total_eval: u64 = 0;
    #[cfg(not(feature = "perf-bench"))]
    let total_eval: u64 = 0;
    #[cfg(feature = "perf-bench")]
    let mut total_change: u64 = 0;
    // Combined mode: measured quantum counters
    #[cfg(feature = "perf-bench")]
    let mut measured_q_ops: u64 = 0;
    #[cfg(not(feature = "perf-bench"))]
    let measured_q_ops: u64 = 0;
    #[cfg(feature = "perf-bench")]
    let mut measured_q_gates: u64 = 0;
    #[cfg(not(feature = "perf-bench"))]
    let measured_q_gates: u64 = 0;
    #[cfg(not(feature = "perf-bench"))]
    let total_change: u64 = 0;
    #[allow(unused_assignments)]
    let mut elapsed: f64 = 0.0;

    #[cfg(feature = "perf-bench")]
    {
        match mode {
            BenchMode::FullTick => {
                let start = Instant::now();
                if dispatch == "match" {
                    for _ in 0..ticks {
                        let (e, c) = sim.tick_bench();
                        total_eval += e as u64;
                        total_change += c as u64;
                    }
                } else {
                    for _ in 0..ticks {
                        let (e, c) = sim.tick_bench_branchless(unchecked);
                        total_eval += e as u64;
                        total_change += c as u64;
                    }
                }
                elapsed = start.elapsed().as_secs_f64();
            }
            BenchMode::CombinedLogicQuantum => {
                let start = Instant::now();
                let mut q_gates: u64 = 0;
                let mut q_ops: u64 = 0;
                if dispatch == "match" {
                    for _ in 0..ticks {
                        let (e, c) = sim.tick_bench();
                        total_eval += e as u64;
                        total_change += c as u64;
                        let (g, qo) = sim.step_quantum_for_bench();
                        q_gates += g;
                        q_ops += qo;
                    }
                } else {
                    for _ in 0..ticks {
                        let (e, c) = sim.tick_bench_branchless(unchecked);
                        total_eval += e as u64;
                        total_change += c as u64;
                        let (g, qo) = sim.step_quantum_for_bench();
                        q_gates += g;
                        q_ops += qo;
                    }
                }
                elapsed = start.elapsed().as_secs_f64();
                measured_q_gates = q_gates;
                measured_q_ops = q_ops;
            }
            BenchMode::KernelFarmBench => {
                if backend_str == "compare" || backend_str == "compare3" {
                    // Repeat and report medians (default 5)
                    let it_ticks = ticks.max(1);
                    let mut s_logic_vals: Vec<f64> = Vec::with_capacity(repeats);
                    let mut s_q_vals: Vec<f64> = Vec::with_capacity(repeats);
                    let mut v_logic_vals: Vec<f64> = Vec::with_capacity(repeats);
                    let mut v_q_vals: Vec<f64> = Vec::with_capacity(repeats);
                    let mut j_logic_vals: Vec<f64> = Vec::with_capacity(repeats);
                    let mut j_q_vals: Vec<f64> = Vec::with_capacity(repeats);

                    for _r in 0..repeats {
                        // Optional JIT prewarm to avoid first-run compilation during timing
                        #[cfg(feature = "quantum_jit")]
                        if backend_str == "compare3" {
                            let nq = scenario_qubits(&scenario);
                            if nq > 0 {
                                engine::quantum_jit::prewarm_h_kernels(nq);
                            }
                        }
                        // Scalar run
                        let mut sim_s = Simulation::new_standard_benchmark_world();
                        if scenario != "none" {
                            apply_scenario(&mut sim_s, &scenario, false);
                        }
                        if qdemo_qubits > 0 {
                            let state = engine::quantum::QState::new_zero(qdemo_qubits);
                            let program = vec![engine::quantum::QGate::Measure(0)];
                            sim_s.register_qdemo_tile(10, 10, state, program, 0x1111_2222);
                        }
                        let mut logic_ops_s: u64 = 0;
                        let mut q_ops_s: u64 = 0;
                        let start_s = Instant::now();
                        if dispatch == "match" {
                            for _ in 0..it_ticks {
                                let (e, _c) = sim_s.tick_bench();
                                logic_ops_s += e as u64;
                                let (_g, qo) = sim_s.step_quantum_for_bench_backend(
                                    engine::quantum::QBackend::Scalar,
                                );
                                q_ops_s += qo;
                            }
                        } else {
                            for _ in 0..it_ticks {
                                let (e, _c) = sim_s.tick_bench_branchless(unchecked);
                                logic_ops_s += e as u64;
                                let (_g, qo) = sim_s.step_quantum_for_bench_backend(
                                    engine::quantum::QBackend::Scalar,
                                );
                                q_ops_s += qo;
                            }
                        }
                        let es = start_s.elapsed().as_secs_f64();
                        s_logic_vals.push(if es > 0.0 {
                            (logic_ops_s as f64) / es
                        } else {
                            0.0
                        });
                        s_q_vals.push(if es > 0.0 { (q_ops_s as f64) / es } else { 0.0 });

                        // AVX2 run
                        let mut sim_v = Simulation::new_standard_benchmark_world();
                        if scenario != "none" {
                            apply_scenario(&mut sim_v, &scenario, false);
                        }
                        if qdemo_qubits > 0 {
                            let state = engine::quantum::QState::new_zero(qdemo_qubits);
                            let program = vec![engine::quantum::QGate::Measure(0)];
                            sim_v.register_qdemo_tile(10, 10, state, program, 0x1111_2222);
                        }
                        let mut logic_ops_v: u64 = 0;
                        let mut q_ops_v: u64 = 0;
                        let start_v = Instant::now();
                        if dispatch == "match" {
                            for _ in 0..it_ticks {
                                let (e, _c) = sim_v.tick_bench();
                                logic_ops_v += e as u64;
                                let (_g, qo) = sim_v.step_quantum_for_bench_backend(
                                    engine::quantum::QBackend::Avx2,
                                );
                                q_ops_v += qo;
                            }
                        } else {
                            for _ in 0..it_ticks {
                                let (e, _c) = sim_v.tick_bench_branchless(unchecked);
                                logic_ops_v += e as u64;
                                let (_g, qo) = sim_v.step_quantum_for_bench_backend(
                                    engine::quantum::QBackend::Avx2,
                                );
                                q_ops_v += qo;
                            }
                        }
                        let ev = start_v.elapsed().as_secs_f64();
                        v_logic_vals.push(if ev > 0.0 {
                            (logic_ops_v as f64) / ev
                        } else {
                            0.0
                        });
                        v_q_vals.push(if ev > 0.0 { (q_ops_v as f64) / ev } else { 0.0 });

                        if backend_str == "compare3" {
                            // JIT run
                            let mut sim_j = Simulation::new_standard_benchmark_world();
                            if scenario != "none" {
                                apply_scenario(&mut sim_j, &scenario, false);
                            }
                            if qdemo_qubits > 0 {
                                let state = engine::quantum::QState::new_zero(qdemo_qubits);
                                let program = vec![engine::quantum::QGate::Measure(0)];
                                sim_j.register_qdemo_tile(10, 10, state, program, 0x1111_2222);
                            }
                            let mut logic_ops_j: u64 = 0;
                            let mut q_ops_j: u64 = 0;
                            let start_j = Instant::now();
                            if dispatch == "match" {
                                for _ in 0..it_ticks {
                                    let (e, _c) = sim_j.tick_bench();
                                    logic_ops_j += e as u64;
                                    let (_g, qo) = sim_j.step_quantum_for_bench_backend(
                                        engine::quantum::QBackend::Jit,
                                    );
                                    q_ops_j += qo;
                                }
                            } else {
                                for _ in 0..it_ticks {
                                    let (e, _c) = sim_j.tick_bench_branchless(unchecked);
                                    logic_ops_j += e as u64;
                                    let (_g, qo) = sim_j.step_quantum_for_bench_backend(
                                        engine::quantum::QBackend::Jit,
                                    );
                                    q_ops_j += qo;
                                }
                            }
                            let ej = start_j.elapsed().as_secs_f64();
                            j_logic_vals.push(if ej > 0.0 {
                                (logic_ops_j as f64) / ej
                            } else {
                                0.0
                            });
                            j_q_vals.push(if ej > 0.0 { (q_ops_j as f64) / ej } else { 0.0 });
                        }
                    }

                    fn median(mut v: Vec<f64>) -> f64 {
                        if v.is_empty() {
                            return 0.0;
                        }
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let m = v.len() / 2;
                        if v.len() % 2 == 1 {
                            v[m]
                        } else {
                            (v[m - 1] + v[m]) * 0.5
                        }
                    }

                    if backend_str == "compare3" {
                        println!(
                            "compare_backend: farm_scalar_vs_avx2_vs_jit (median of {})",
                            repeats
                        );
                        println!("logic_ops_per_sec_scalar: {:.2}", median(s_logic_vals));
                        println!("logic_ops_per_sec_avx2: {:.2}", median(v_logic_vals));
                        println!("logic_ops_per_sec_jit: {:.2}", median(j_logic_vals));
                        let qs = median(s_q_vals);
                        let qv = median(v_q_vals);
                        let qj = median(j_q_vals);
                        println!("q_ops_per_sec_scalar: {:.2}", qs);
                        println!("q_ops_per_sec_avx2: {:.2}", qv);
                        println!("q_ops_per_sec_jit: {:.2}", qj);
                        if qs > 0.0 {
                            println!("speedup_q_ops_avx2: {:.2}x", qv / qs);
                            println!("speedup_q_ops_jit: {:.2}x", qj / qs);
                        }
                    } else {
                        println!(
                            "compare_backend: farm_scalar_vs_avx2 (median of {})",
                            repeats
                        );
                        println!("logic_ops_per_sec_scalar: {:.2}", median(s_logic_vals));
                        println!("logic_ops_per_sec_avx2: {:.2}", median(v_logic_vals));
                        let qs = median(s_q_vals);
                        let qv = median(v_q_vals);
                        println!("q_ops_per_sec_scalar: {:.2}", qs);
                        println!("q_ops_per_sec_avx2: {:.2}", qv);
                        if qs > 0.0 {
                            println!("speedup_q_ops: {:.2}x", qv / qs);
                        }
                    }
                    // Optional JIT prewarm to avoid first-run compilation during timing
                    #[cfg(feature = "quantum_jit")]
                    if backend_str == "compare3" {
                        let nq = scenario_qubits(&scenario);
                        if nq > 0 {
                            engine::quantum_jit::prewarm_h_kernels(nq);
                        }
                    }
                    // Scalar run
                    let mut sim_s = Simulation::new_standard_benchmark_world();
                    if scenario != "none" {
                        apply_scenario(&mut sim_s, &scenario, false);
                    }
                    if qdemo_qubits > 0 {
                        let state = engine::quantum::QState::new_zero(qdemo_qubits);
                        let program = vec![engine::quantum::QGate::Measure(0)];
                        sim_s.register_qdemo_tile(10, 10, state, program, 0x1111_2222);
                    }
                    let mut logic_ops_s: u64 = 0;
                    let mut q_ops_s: u64 = 0;
                    let start_s = Instant::now();
                    if dispatch == "match" {
                        for _ in 0..it_ticks {
                            let (e, _c) = sim_s.tick_bench();
                            logic_ops_s += e as u64;
                            let (_g, qo) = sim_s
                                .step_quantum_for_bench_backend(engine::quantum::QBackend::Scalar);
                            q_ops_s += qo;
                        }
                    } else {
                        for _ in 0..it_ticks {
                            let (e, _c) = sim_s.tick_bench_branchless(unchecked);
                            logic_ops_s += e as u64;
                            let (_g, qo) = sim_s
                                .step_quantum_for_bench_backend(engine::quantum::QBackend::Scalar);
                            q_ops_s += qo;
                        }
                    }
                    let elapsed_s = start_s.elapsed().as_secs_f64();
                    let logic_s = if elapsed_s > 0.0 {
                        (logic_ops_s as f64) / elapsed_s
                    } else {
                        0.0
                    };
                    let q_s = if elapsed_s > 0.0 {
                        (q_ops_s as f64) / elapsed_s
                    } else {
                        0.0
                    };
                    println!("compare_progress: scalar_done elapsed_s={:.3}", elapsed_s);

                    // AVX2 run
                    let mut sim_v = Simulation::new_standard_benchmark_world();
                    if scenario != "none" {
                        apply_scenario(&mut sim_v, &scenario, false);
                    }
                    if qdemo_qubits > 0 {
                        let state = engine::quantum::QState::new_zero(qdemo_qubits);
                        let program = vec![engine::quantum::QGate::Measure(0)];
                        sim_v.register_qdemo_tile(10, 10, state, program, 0x1111_2222);
                    }
                    let mut logic_ops_v: u64 = 0;
                    let mut q_ops_v: u64 = 0;
                    let start_v = Instant::now();
                    if dispatch == "match" {
                        for _ in 0..it_ticks {
                            let (e, _c) = sim_v.tick_bench();
                            logic_ops_v += e as u64;
                            let (_g, qo) = sim_v
                                .step_quantum_for_bench_backend(engine::quantum::QBackend::Avx2);
                            q_ops_v += qo;
                        }
                    } else {
                        for _ in 0..it_ticks {
                            let (e, _c) = sim_v.tick_bench_branchless(unchecked);
                            logic_ops_v += e as u64;
                            let (_g, qo) = sim_v
                                .step_quantum_for_bench_backend(engine::quantum::QBackend::Avx2);
                            q_ops_v += qo;
                        }
                    }
                    let elapsed_v = start_v.elapsed().as_secs_f64();
                    let logic_v = if elapsed_v > 0.0 {
                        (logic_ops_v as f64) / elapsed_v
                    } else {
                        0.0
                    };
                    let q_v = if elapsed_v > 0.0 {
                        (q_ops_v as f64) / elapsed_v
                    } else {
                        0.0
                    };
                    println!("compare_progress: avx2_done elapsed_s={:.3}", elapsed_v);

                    if backend_str == "compare3" {
                        // JIT run (may route to AVX2/scalar depending on features)
                        let mut sim_j = Simulation::new_standard_benchmark_world();
                        if scenario != "none" {
                            apply_scenario(&mut sim_j, &scenario, false);
                        }
                        if qdemo_qubits > 0 {
                            let state = engine::quantum::QState::new_zero(qdemo_qubits);
                            let program = vec![engine::quantum::QGate::Measure(0)];
                            sim_j.register_qdemo_tile(10, 10, state, program, 0x1111_2222);
                        }
                        let mut logic_ops_j: u64 = 0;
                        let mut q_ops_j: u64 = 0;
                        let start_j = Instant::now();
                        if dispatch == "match" {
                            for _ in 0..it_ticks {
                                let (e, _c) = sim_j.tick_bench();
                                logic_ops_j += e as u64;
                                let (_g, qo) = sim_j
                                    .step_quantum_for_bench_backend(engine::quantum::QBackend::Jit);
                                q_ops_j += qo;
                            }
                        } else {
                            for _ in 0..it_ticks {
                                let (e, _c) = sim_j.tick_bench_branchless(unchecked);
                                logic_ops_j += e as u64;
                                let (_g, qo) = sim_j
                                    .step_quantum_for_bench_backend(engine::quantum::QBackend::Jit);
                                q_ops_j += qo;
                            }
                        }
                        let elapsed_j = start_j.elapsed().as_secs_f64();
                        let logic_j = if elapsed_j > 0.0 {
                            (logic_ops_j as f64) / elapsed_j
                        } else {
                            0.0
                        };
                        let q_j = if elapsed_j > 0.0 {
                            (q_ops_j as f64) / elapsed_j
                        } else {
                            0.0
                        };
                        println!("compare_progress: jit_done elapsed_s={:.3}", elapsed_j);

                        println!("compare_backend: farm_scalar_vs_avx2_vs_jit");
                        println!("logic_ops_per_sec_scalar: {:.2}", logic_s);
                        println!("logic_ops_per_sec_avx2: {:.2}", logic_v);
                        println!("logic_ops_per_sec_jit: {:.2}", logic_j);
                        println!("q_ops_per_sec_scalar: {:.2}", q_s);
                        println!("q_ops_per_sec_avx2: {:.2}", q_v);
                        println!("q_ops_per_sec_jit: {:.2}", q_j);
                        if q_s > 0.0 {
                            println!("speedup_q_ops_avx2: {:.2}x", q_v / q_s);
                        }
                        if q_s > 0.0 {
                            println!("speedup_q_ops_jit: {:.2}x", q_j / q_s);
                        }
                        // Use AVX run as the timed result for standard lines
                        elapsed = elapsed_v;
                        total_eval += logic_ops_v;
                        measured_q_ops = q_ops_v;
                    } else {
                        println!("compare_backend: farm_scalar_vs_avx2");
                        println!("logic_ops_per_sec_scalar: {:.2}", logic_s);
                        println!("logic_ops_per_sec_avx2: {:.2}", logic_v);
                        println!("q_ops_per_sec_scalar: {:.2}", q_s);
                        println!("q_ops_per_sec_avx2: {:.2}", q_v);
                        if q_s > 0.0 {
                            println!("speedup_q_ops: {:.2}x", q_v / q_s);
                        }
                        // Use AVX run as the timed result for standard lines
                        elapsed = elapsed_v;
                        total_eval += logic_ops_v;
                        measured_q_ops = q_ops_v;
                    }
                } else {
                    // Bench-time kernel farm: measure logic + fields + organism + quantum in deterministic order
                    let start = Instant::now();
                    let mut logic_ops: u64 = 0;
                    let field_ops: u64 = 0; // reserved
                    let mut q_ops: u64 = 0;
                    if dispatch == "match" {
                        for _ in 0..ticks {
                            let (e, _c) = sim.tick_bench();
                            logic_ops += e as u64;
                            let (_g, qo) =
                                sim.step_quantum_for_bench_backend(match backend_str.as_str() {
                                    "avx2" => engine::quantum::QBackend::Avx2,
                                    "avx512" => engine::quantum::QBackend::Avx512,
                                    "jit" => engine::quantum::QBackend::Jit,
                                    #[cfg(feature = "cuda")]
                                    "cuda" => engine::quantum::QBackend::Cuda,
                                    _ => engine::quantum::QBackend::Scalar,
                                });
                            q_ops += qo;
                        }
                    } else {
                        for _ in 0..ticks {
                            let (e, _c) = sim.tick_bench_branchless(unchecked);
                            logic_ops += e as u64;
                            let (_g, qo) =
                                sim.step_quantum_for_bench_backend(match backend_str.as_str() {
                                    "avx2" => engine::quantum::QBackend::Avx2,
                                    "avx512" => engine::quantum::QBackend::Avx512,
                                    "jit" => engine::quantum::QBackend::Jit,
                                    #[cfg(feature = "cuda")]
                                    "cuda" => engine::quantum::QBackend::Cuda,
                                    _ => engine::quantum::QBackend::Scalar,
                                });
                            q_ops += qo;
                        }
                    }
                    elapsed = start.elapsed().as_secs_f64();
                    let logic_ops_per_sec = if elapsed > 0.0 {
                        (logic_ops as f64) / elapsed
                    } else {
                        0.0
                    };
                    let field_ops_per_sec = if elapsed > 0.0 {
                        (field_ops as f64) / elapsed
                    } else {
                        0.0
                    };
                    let q_ops_per_sec = if elapsed > 0.0 {
                        (q_ops as f64) / elapsed
                    } else {
                        0.0
                    };
                    println!("kernel_id: logic");
                    println!("logic_ops_per_sec: {:.2}", logic_ops_per_sec);
                    println!("kernel_id: fields");
                    println!("field_ops_per_sec: {:.2}", field_ops_per_sec);
                    println!("kernel_id: quantum backend={}", backend_str);
                    println!("q_ops_per_sec_measured: {:.2}", q_ops_per_sec);
                    let total = logic_ops_per_sec + field_ops_per_sec + q_ops_per_sec;
                    println!("farm_total_ops_per_sec: {:.2}", total);
                    total_eval += logic_ops;
                    measured_q_ops = q_ops;
                }
            }
            BenchMode::LogicEvalOnly => {
                let start = Instant::now();
                for _r in 0..reps {
                    for _i in 0..iters {
                        let (e, c) = sim.eval_logic_full_grid_bench();
                        total_eval += e as u64;
                        total_change += c as u64;
                    }
                }
                elapsed = start.elapsed().as_secs_f64();
            }
            BenchMode::LogicDirtyOnly => {
                let mut scratch: Vec<u32> = Vec::with_capacity(((WIDTH * HEIGHT) / 64) as usize);
                for _r in 0..reps {
                    let start = Instant::now();
                    for _i in 0..iters {
                        scratch.clear();
                        for y in (0..HEIGHT).step_by(64) {
                            for x in (0..WIDTH).step_by(64) {
                                scratch.push((y * WIDTH + x) as u32);
                            }
                        }
                        let (e, c) = sim.eval_logic_dirty_batch_bench(&scratch);
                        total_eval += e as u64;
                        total_change += c as u64;
                    }
                    elapsed += start.elapsed().as_secs_f64();
                }
            }
            BenchMode::FieldDiffuseOnly => {
                // Use internal bench API (perf-bench only)
                let start = Instant::now();
                let ops = engine::bench_api::bench_api::diffuse_only(
                    WIDTH,
                    HEIGHT,
                    iters.max(1),
                    &fields_dispatch,
                );
                elapsed = start.elapsed().as_secs_f64();
                total_eval += ops;
            }
            BenchMode::EcosystemOnly => {
                let steps_n = ticks.max(1);
                let start = Instant::now();
                let steps_run = engine::bench_api::bench_api::ecosystem_only(steps_n);
                elapsed = start.elapsed().as_secs_f64();
                total_eval += steps_run as u64;
            }
            BenchMode::FieldReactOnly => {
                let start = Instant::now();
                let ops = engine::bench_api::bench_api::react_only(WIDTH, HEIGHT, iters.max(1));
                elapsed = start.elapsed().as_secs_f64();
                total_eval += ops;
            }
            BenchMode::FieldInteractOnly => {
                let start = Instant::now();
                let ops = engine::bench_api::bench_api::interact_only(WIDTH, HEIGHT, iters.max(1));
                elapsed = start.elapsed().as_secs_f64();
                total_eval += ops;
            }
            BenchMode::FieldsAll => {
                let start = Instant::now();
                let ops = engine::bench_api::bench_api::fields_all(
                    WIDTH,
                    HEIGHT,
                    iters.max(1),
                    &fields_dispatch,
                );
                elapsed = start.elapsed().as_secs_f64();
                total_eval += ops;
            }
            BenchMode::FieldDecayOnly => {
                let start = Instant::now();
                let ops = engine::bench_api::bench_api::decay_only(WIDTH, HEIGHT, iters.max(1));
                elapsed = start.elapsed().as_secs_f64();
                total_eval += ops;
            }
            BenchMode::FieldFeedbackOnly => {
                let start = Instant::now();
                let ops = engine::bench_api::bench_api::feedback_only(WIDTH, HEIGHT, iters.max(1));
                elapsed = start.elapsed().as_secs_f64();
                total_eval += ops;
            }
            BenchMode::QuantumScalarOnly => {
                let start = Instant::now();
                let ops = engine::bench_api::bench_api::quantum_scalar_only(_qubits, iters.max(1));
                elapsed = start.elapsed().as_secs_f64();
                total_eval += ops;
            }
            BenchMode::QuantumBackendOnly => {
                let bsel = backend_str.to_ascii_lowercase();
                if bsel == "compare" {
                    // Run scalar then avx2 and print speedup lines
                    let itn = iters.max(1);
                    let start_s = Instant::now();
                    let (ops_s, _flops_s) = engine::bench_api::bench_api::quantum_backend_only(
                        _qubits,
                        itn,
                        engine::quantum::QBackend::Scalar,
                    );
                    let elapsed_s = start_s.elapsed().as_secs_f64();
                    let qops_s = (ops_s as f64) / elapsed_s;

                    let start_v = Instant::now();
                    let (ops_v, _flops_v) = engine::bench_api::bench_api::quantum_backend_only(
                        _qubits,
                        itn,
                        engine::quantum::QBackend::Avx2,
                    );
                    let elapsed_v = start_v.elapsed().as_secs_f64();
                    let qops_v = (ops_v as f64) / elapsed_v;

                    println!("compare_backend: scalar_vs_avx2");
                    println!("scalar_q_ops_per_sec: {:.2}", qops_s);
                    println!("avx2_q_ops_per_sec: {:.2}", qops_v);
                    if qops_s > 0.0 {
                        println!("speedup: {:.2}x", qops_v / qops_s);
                    }
                    // Set elapsed/total_eval for standard lines
                    elapsed = elapsed_v;
                    total_eval += ops_v;
                } else {
                    let backend = match bsel.as_str() {
                        "avx2" => engine::quantum::QBackend::Avx2,
                        "avx512" => engine::quantum::QBackend::Avx512,
                        "jit" => engine::quantum::QBackend::Jit,
                        _ => engine::quantum::QBackend::Scalar,
                    };
                    let runs = repeats.max(1);
                    let mut samples: Vec<(u64, u64, f64)> = Vec::with_capacity(runs);
                    for _ in 0..runs {
                        let start = Instant::now();
                        let (ops, flops) = engine::bench_api::bench_api::quantum_backend_only(
                            _qubits,
                            iters.max(1),
                            backend,
                        );
                        let e = start.elapsed().as_secs_f64();
                        samples.push((ops, flops, e));
                    }
                    fn median(mut v: Vec<f64>) -> f64 {
                        if v.is_empty() {
                            return 0.0;
                        }
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let m = v.len() / 2;
                        if v.len() % 2 == 1 {
                            v[m]
                        } else {
                            (v[m - 1] + v[m]) * 0.5
                        }
                    }
                    let q_ops_per_sec_vals: Vec<f64> = samples
                        .iter()
                        .map(|(ops, _flops, e)| if *e > 0.0 { (*ops as f64) / *e } else { 0.0 })
                        .collect();
                    let elapsed_vals: Vec<f64> = samples.iter().map(|(_, _, e)| *e).collect();
                    let q_ops_per_sec_median = median(q_ops_per_sec_vals);
                    let elapsed_median = median(elapsed_vals);
                    // Align total_eval with the median-derived throughput so downstream prints stay coherent.
                    total_eval = (q_ops_per_sec_median * elapsed_median) as u64;
                    elapsed = elapsed_median;
                    #[allow(unused_variables)]
                    let _q_flops = samples.last().map(|(_, f, _)| *f).unwrap_or(0);
                    println!("qbackend_repeats: {}", runs);
                    println!("q_ops_per_sec_median: {:.2}", q_ops_per_sec_median);
                    println!("elapsed_median_s: {:.6}", elapsed_median);
                }
            }
            // ================================================================
            // EPIC 66: CUDA GPU Benchmark Mode
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::CudaGpu => {
                use engine::cuda::{
                    CudaContext, GpuQState, is_cuda_available, run_hadamard_kernel,
                };
                use engine::quantum::QState;

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA GPU benchmark requested but no GPU available");
                    eprintln!("Make sure:");
                    eprintln!("  1. NVIDIA GPU is installed");
                    eprintln!("  2. CUDA drivers are installed");
                    eprintln!("  3. CUDA Toolkit is installed (set CUDA_PATH)");
                    std::process::exit(1);
                }

                let ctx = CudaContext::new().expect("Failed to create CUDA context");

                // Benchmark parameters
                let n_qubits = _qubits;
                let tile_count = 16u16; // Multiple tiles for GPU parallelism
                let depth = warmup.max(32) as u32; // Use warmup as depth hint
                let kernel_runs = iters.max(100);

                // Create and upload state
                let qstate = QState::new_zero_multitile(n_qubits, tile_count);

                let mut gpu_state =
                    GpuQState::from_qstate(&ctx, &qstate).expect("Failed to upload state to GPU");

                println!("EPIC 66: CUDA GPU Benchmark");
                println!("============================");
                println!("qubits: {}", n_qubits);
                println!("tile_count: {}", tile_count);
                println!("amplitudes: {}", gpu_state.len());
                println!("depth_per_kernel: {}", depth);
                println!("kernel_runs: {}", kernel_runs);

                // Warmup runs
                for _ in 0..3 {
                    run_hadamard_kernel(&ctx, &mut gpu_state, depth).expect("GPU warmup failed");
                }

                // Timed benchmark
                let start = Instant::now();
                for _ in 0..kernel_runs {
                    run_hadamard_kernel(&ctx, &mut gpu_state, depth).expect("GPU kernel failed");
                }
                elapsed = start.elapsed().as_secs_f64();

                // Calculate metrics
                let total_kernel_ops = kernel_runs as u64 * depth as u64;
                let total_amplitude_updates = total_kernel_ops * gpu_state.len() as u64;
                let kernels_per_sec = (kernel_runs as f64) / elapsed;
                let amplitude_ops_per_sec = (total_amplitude_updates as f64) / elapsed;

                println!("");
                println!("Results:");
                println!("elapsed_s: {:.6}", elapsed);
                println!("kernels_per_sec: {:.2}", kernels_per_sec);
                println!("amplitude_ops_per_sec: {:.2}", amplitude_ops_per_sec);
                println!("total_kernel_ops: {}", total_kernel_ops);

                // Compare with CPU scalar
                println!("");
                println!("CPU Scalar Comparison:");
                let cpu_start = Instant::now();
                let mut cpu_state = qstate.clone();
                let mut rng = engine::quantum::QRng::new(12345);
                for _ in 0..(kernel_runs * depth as usize) {
                    engine::quantum::apply_gate_scalar(
                        &mut cpu_state,
                        &engine::quantum::QGate::H(0),
                        &mut rng,
                    );
                }
                let cpu_elapsed = cpu_start.elapsed().as_secs_f64();
                let cpu_ops_per_sec = (total_kernel_ops as f64) / cpu_elapsed;

                println!("cpu_scalar_elapsed_s: {:.6}", cpu_elapsed);
                println!("cpu_scalar_ops_per_sec: {:.2}", cpu_ops_per_sec);
                if cpu_ops_per_sec > 0.0 {
                    println!(
                        "gpu_speedup: {:.2}x",
                        kernels_per_sec * depth as f64 / cpu_ops_per_sec
                    );
                }

                total_eval = total_kernel_ops;
            }
            // ================================================================
            // EPIC 67: CUDA GPU-Resident Benchmark Mode
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::CudaGpuResident => {
                use engine::cuda::{
                    CudaRuntime, GpuQState, is_cuda_available, run_hadamard_kernel,
                    run_resident_steps,
                };
                use engine::quantum::QState;

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA GPU-resident benchmark requested but no GPU available");
                    std::process::exit(1);
                }

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                // Benchmark parameters
                let n_qubits = _qubits;
                let tile_count = 16u16;
                let depth = warmup.max(32) as u32;
                let kernel_runs = iters.max(100);

                // Create GPU-resident state (no CPU copy)
                let mut gpu_state = GpuQState::new_zero_resident(&rt, n_qubits, tile_count)
                    .expect("Failed to create GPU-resident state");

                println!("EPIC 67: CUDA GPU-Resident Benchmark");
                println!("=====================================");
                println!("qubits: {}", n_qubits);
                println!("tile_count: {}", tile_count);
                println!("amplitudes: {}", gpu_state.len);
                println!("depth_per_step: {}", depth);
                println!("kernel_runs: {}", kernel_runs);
                println!("resident: true (state stays in VRAM)");

                // Warmup runs
                for _ in 0..3 {
                    run_hadamard_kernel(&rt, &mut gpu_state, depth).expect("GPU warmup failed");
                }

                // Timed benchmark - state never leaves GPU
                let start = Instant::now();
                let checksums = run_resident_steps(&rt, &mut gpu_state, kernel_runs, depth)
                    .expect("GPU-resident steps failed");
                elapsed = start.elapsed().as_secs_f64();

                // Calculate metrics
                let total_kernel_ops = kernel_runs as u64 * depth as u64;
                let total_amplitude_updates = total_kernel_ops * gpu_state.len as u64;
                let kernels_per_sec = (kernel_runs as f64) / elapsed;
                let amplitude_ops_per_sec = (total_amplitude_updates as f64) / elapsed;

                println!("");
                println!("Results (GPU-Resident):");
                println!("elapsed_s: {:.6}", elapsed);
                println!("kernels_per_sec: {:.2}", kernels_per_sec);
                println!("amplitude_ops_per_sec: {:.2}", amplitude_ops_per_sec);
                println!("total_kernel_ops: {}", total_kernel_ops);
                println!("checksums_collected: {}", checksums.len());
                println!(
                    "first_checksum: 0x{:08X}",
                    checksums.first().copied().unwrap_or(0)
                );
                println!(
                    "last_checksum: 0x{:08X}",
                    checksums.last().copied().unwrap_or(0)
                );

                // Compare with non-resident CUDA (upload/download each iteration)
                println!("");
                println!("Non-Resident Comparison:");
                let qstate = QState::new_zero_multitile(n_qubits, tile_count);
                let nr_start = Instant::now();
                for _ in 0..kernel_runs {
                    let mut nr_gpu = GpuQState::from_qstate(&rt, &qstate).expect("Upload failed");
                    run_hadamard_kernel(&rt, &mut nr_gpu, depth).expect("Kernel failed");
                    // Download would happen here in real non-resident use
                }
                let nr_elapsed = nr_start.elapsed().as_secs_f64();
                let nr_kernels_per_sec = (kernel_runs as f64) / nr_elapsed;

                println!("non_resident_elapsed_s: {:.6}", nr_elapsed);
                println!("non_resident_kernels_per_sec: {:.2}", nr_kernels_per_sec);
                if nr_kernels_per_sec > 0.0 {
                    println!(
                        "resident_speedup: {:.2}x",
                        kernels_per_sec / nr_kernels_per_sec
                    );
                }

                total_eval = total_kernel_ops;
            }
            // ================================================================
            // EPIC 67: CUDA Tensor Core Benchmark Mode (FP16)
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::CudaTensor => {
                use engine::cuda::{
                    CudaRuntime, GpuQStateF16, is_cuda_available, is_tensor_core_available,
                    run_tensor_hadamard_kernel,
                };

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA Tensor Core benchmark requested but no GPU available");
                    std::process::exit(1);
                }

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                // Benchmark parameters
                let n_qubits = _qubits;
                let tile_count = 16u16;
                let depth = warmup.max(32) as u32;
                let kernel_runs = iters.max(100);

                // Check Tensor Core availability
                let tensor_available = is_tensor_core_available(&rt);
                println!("EPIC 67: CUDA Tensor Core Benchmark (FP16)");
                println!("==========================================");
                println!("tensor_cores_available: {}", tensor_available);
                println!("qubits: {}", n_qubits);
                println!("tile_count: {}", tile_count);
                println!("depth_per_step: {}", depth);
                println!("kernel_runs: {}", kernel_runs);
                println!("precision: FP16 (half)");

                if !tensor_available {
                    println!("");
                    println!("WARNING: Tensor Cores not detected on this GPU");
                    println!("Running FP16 kernel without WMMA acceleration");
                }

                // Create FP16 GPU state from a CPU QState
                use engine::quantum::QState;
                let qstate = QState::new_zero_multitile(n_qubits, tile_count);
                let mut gpu_state = GpuQStateF16::from_qstate(&rt, &qstate)
                    .expect("Failed to create FP16 GPU state");

                println!("amplitudes: {}", gpu_state.len);

                // Warmup runs
                for _ in 0..3 {
                    if let Err(e) = run_tensor_hadamard_kernel(&rt, &mut gpu_state, depth) {
                        eprintln!("Tensor warmup failed: {}", e);
                        eprintln!("Falling back to FP32 benchmark");
                        // Fall through - will report 0 throughput
                    }
                }

                // Timed benchmark
                let start = Instant::now();
                let mut success_count = 0usize;
                for _ in 0..kernel_runs {
                    if run_tensor_hadamard_kernel(&rt, &mut gpu_state, depth).is_ok() {
                        success_count += 1;
                    }
                }
                elapsed = start.elapsed().as_secs_f64();

                // Calculate metrics
                let total_kernel_ops = success_count as u64 * depth as u64;
                let total_amplitude_updates = total_kernel_ops * gpu_state.len as u64;
                let kernels_per_sec = (success_count as f64) / elapsed;
                let amplitude_ops_per_sec = (total_amplitude_updates as f64) / elapsed;

                println!("");
                println!("Results (FP16 Tensor):");
                println!("successful_kernels: {} / {}", success_count, kernel_runs);
                println!("elapsed_s: {:.6}", elapsed);
                println!("kernels_per_sec: {:.2}", kernels_per_sec);
                println!("amplitude_ops_per_sec: {:.2}", amplitude_ops_per_sec);
                println!("total_kernel_ops: {}", total_kernel_ops);

                // Compare with FP32 CUDA
                println!("");
                println!("FP32 CUDA Comparison:");
                use engine::cuda::{GpuQState, run_hadamard_kernel};

                let fp32_qstate = QState::new_zero_multitile(n_qubits, tile_count);
                let mut fp32_gpu = GpuQState::from_qstate(&rt, &fp32_qstate)
                    .expect("Failed to create FP32 GPU state");

                let fp32_start = Instant::now();
                for _ in 0..kernel_runs {
                    run_hadamard_kernel(&rt, &mut fp32_gpu, depth).expect("FP32 kernel failed");
                }
                let fp32_elapsed = fp32_start.elapsed().as_secs_f64();
                let fp32_kernels_per_sec = (kernel_runs as f64) / fp32_elapsed;

                println!("fp32_elapsed_s: {:.6}", fp32_elapsed);
                println!("fp32_kernels_per_sec: {:.2}", fp32_kernels_per_sec);
                if fp32_kernels_per_sec > 0.0 && kernels_per_sec > 0.0 {
                    println!(
                        "fp16_speedup: {:.2}x",
                        kernels_per_sec / fp32_kernels_per_sec
                    );
                }

                // Memory savings
                let fp32_bytes = gpu_state.len * 4 * 2; // real + imag, 4 bytes each
                let fp16_bytes = gpu_state.len * 2 * 2; // real + imag, 2 bytes each
                println!("memory_fp32_bytes: {}", fp32_bytes);
                println!("memory_fp16_bytes: {}", fp16_bytes);
                println!(
                    "memory_savings: {:.1}%",
                    (1.0 - (fp16_bytes as f64 / fp32_bytes as f64)) * 100.0
                );

                total_eval = total_kernel_ops;
            }
            // ================================================================
            // EPIC 67 PERF: Batched GPU Execution (THE REAL BENCHMARK)
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::CudaBatched => {
                use engine::cuda::{
                    CudaRuntime, GpuQState, GpuQStateF16, is_cuda_available,
                    is_tensor_core_available, run_batched_hadamard, run_batched_tensor_hadamard,
                };
                use engine::quantum::QState;

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA batched benchmark requested but no GPU available");
                    std::process::exit(1);
                }

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                // Benchmark parameters
                let n_qubits = _qubits;
                let tile_count = 16u16;
                let depth_per_step = warmup.max(100) as u32;
                let total_steps = iters.max(1000);
                let total_depth = (total_steps as u32) * depth_per_step;

                println!("╔══════════════════════════════════════════════════════════════╗");
                println!("║  EPIC 67: BATCHED GPU EXECUTION - THE REAL NUMBERS           ║");
                println!("╚══════════════════════════════════════════════════════════════╝");
                println!("");
                println!("Configuration:");
                println!("  qubits: {}", n_qubits);
                println!("  tile_count: {}", tile_count);
                println!("  total_steps: {}", total_steps);
                println!("  depth_per_step: {}", depth_per_step);
                println!("  total_depth: {} (all in ONE kernel launch!)", total_depth);
                println!("");

                // Create GPU-resident state
                let gpu_state = GpuQState::new_zero_resident(&rt, n_qubits, tile_count)
                    .expect("Failed to create GPU-resident state");

                let amplitudes = gpu_state.len;
                println!("  amplitudes: {}", amplitudes);
                println!("");

                // ============================================================
                // BENCHMARK 1: Old way (per-step kernel launch) - THE SLOW PATH
                // ============================================================
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!(
                    "OLD WAY: {} kernel launches + {} syncs",
                    total_steps, total_steps
                );
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                // Reset state
                let mut old_state =
                    GpuQState::new_zero_resident(&rt, n_qubits, tile_count).expect("Reset failed");

                // Warmup
                for _ in 0..3 {
                    let _ = engine::cuda::run_hadamard_kernel(&rt, &mut old_state, depth_per_step);
                }

                let old_start = Instant::now();
                for _ in 0..total_steps {
                    engine::cuda::run_hadamard_kernel(&rt, &mut old_state, depth_per_step)
                        .expect("Old kernel failed");
                }
                rt.synchronize().expect("Sync failed");
                let old_elapsed = old_start.elapsed().as_secs_f64();

                let old_ops = (total_steps as u64) * (depth_per_step as u64);
                let old_amplitude_ops = old_ops * (amplitudes as u64);
                let old_ops_per_sec = (old_ops as f64) / old_elapsed;
                let old_amp_ops_per_sec = (old_amplitude_ops as f64) / old_elapsed;

                println!("  elapsed_s: {:.6}", old_elapsed);
                println!("  total_gate_ops: {}", old_ops);
                println!("  gate_ops_per_sec: {:.0}", old_ops_per_sec);
                println!("  amplitude_ops_per_sec: {:.2e}", old_amp_ops_per_sec);
                println!("");

                // ============================================================
                // BENCHMARK 2: New way (SINGLE kernel launch) - THE FAST PATH
                // ============================================================
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!(
                    "NEW WAY: 1 kernel launch, 1 sync (batched depth={})",
                    total_depth
                );
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                // Reset state
                let mut new_state =
                    GpuQState::new_zero_resident(&rt, n_qubits, tile_count).expect("Reset failed");

                // Warmup
                for _ in 0..3 {
                    let _ = run_batched_hadamard(&rt, &mut new_state, depth_per_step);
                }

                // Reset again for clean measurement
                let mut new_state =
                    GpuQState::new_zero_resident(&rt, n_qubits, tile_count).expect("Reset failed");

                let new_start = Instant::now();
                let _checksum = run_batched_hadamard(&rt, &mut new_state, total_depth)
                    .expect("Batched kernel failed");
                let new_elapsed = new_start.elapsed().as_secs_f64();

                let new_ops = total_depth as u64;
                let new_amplitude_ops = new_ops * (amplitudes as u64);
                let new_ops_per_sec = (new_ops as f64) / new_elapsed;
                let new_amp_ops_per_sec = (new_amplitude_ops as f64) / new_elapsed;

                println!("  elapsed_s: {:.6}", new_elapsed);
                println!("  total_gate_ops: {}", new_ops);
                println!("  gate_ops_per_sec: {:.0}", new_ops_per_sec);
                println!("  amplitude_ops_per_sec: {:.2e}", new_amp_ops_per_sec);
                println!("");

                // ============================================================
                // BENCHMARK 3: FP16 batched (if available)
                // ============================================================
                let tensor_available = is_tensor_core_available(&rt);
                if tensor_available {
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    println!("FP16 BATCHED: 1 kernel launch (half precision)");
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                    let qstate = QState::new_zero_multitile(n_qubits, tile_count);
                    let mut fp16_state =
                        GpuQStateF16::from_qstate(&rt, &qstate).expect("FP16 state failed");

                    // Warmup
                    for _ in 0..3 {
                        let _ = run_batched_tensor_hadamard(&rt, &mut fp16_state, depth_per_step);
                    }

                    // Reset
                    let mut fp16_state =
                        GpuQStateF16::from_qstate(&rt, &qstate).expect("FP16 reset failed");

                    let fp16_start = Instant::now();
                    run_batched_tensor_hadamard(&rt, &mut fp16_state, total_depth)
                        .expect("FP16 batched failed");
                    rt.synchronize().expect("Sync failed");
                    let fp16_elapsed = fp16_start.elapsed().as_secs_f64();

                    let fp16_ops_per_sec = (new_ops as f64) / fp16_elapsed;
                    let fp16_amp_ops_per_sec = (new_amplitude_ops as f64) / fp16_elapsed;

                    println!("  elapsed_s: {:.6}", fp16_elapsed);
                    println!("  gate_ops_per_sec: {:.0}", fp16_ops_per_sec);
                    println!("  amplitude_ops_per_sec: {:.2e}", fp16_amp_ops_per_sec);
                    println!("  fp16_vs_fp32_speedup: {:.2}x", new_elapsed / fp16_elapsed);
                    println!("");
                }

                // ============================================================
                // SUMMARY
                // ============================================================
                println!("╔══════════════════════════════════════════════════════════════╗");
                println!("║  RESULTS SUMMARY                                             ║");
                println!("╚══════════════════════════════════════════════════════════════╝");
                println!("");
                let speedup = old_elapsed / new_elapsed;
                println!("  🚀 BATCHING SPEEDUP: {:.1}x", speedup);
                println!("");
                println!("  Old (per-step): {:.0} gate_ops/sec", old_ops_per_sec);
                println!("  New (batched):  {:.0} gate_ops/sec", new_ops_per_sec);
                println!("");
                println!("  Kernel launches: {} → 1", total_steps);
                println!("  Syncs: {} → 1", total_steps);

                elapsed = new_elapsed;
                total_eval = new_ops;
            }
            // ================================================================
            // EPIC 67 T2: REAL WMMA TENSOR CORE BENCHMARK
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::CudaWmma => {
                use engine::cuda::{
                    CudaRuntime, WmmaState, create_hadamard_gate_16x16, create_identity_gate,
                    is_cuda_available, is_wmma_available, run_wmma_transform,
                };

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA WMMA benchmark requested but no GPU available");
                    std::process::exit(1);
                }

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                if !is_wmma_available(&rt) {
                    eprintln!("ERROR: WMMA not available on this GPU (requires SM 7.0+)");
                    std::process::exit(1);
                }

                // Benchmark parameters
                // Each 16x16 tile = 256 elements = 4 qubits worth of amplitudes
                // So n_tiles maps to 4 * n_tiles qubits of state
                let num_tiles_list = vec![1, 4, 16, 64, 256, 1024, 4096];
                let depth = warmup.max(1000) as u32;

                println!("╔══════════════════════════════════════════════════════════════╗");
                println!("║  EPIC 67 T2: REAL WMMA TENSOR CORE BENCHMARK                 ║");
                println!("╠══════════════════════════════════════════════════════════════╣");
                println!("║  Using real wmma::mma_sync intrinsics!                       ║");
                println!("╚══════════════════════════════════════════════════════════════╝");
                println!("");
                println!("Configuration:");
                println!("  depth: {} (transforms per tile)", depth);
                println!("");

                // Create gates once
                let _identity = create_identity_gate(&rt).expect("Identity gate");
                let hadamard = create_hadamard_gate_16x16(&rt).expect("Hadamard gate");

                println!("┌────────────┬──────────────┬────────────────┬────────────────┐");
                println!("│   Tiles    │   Elements   │  Gate ops/sec  │ Amp ops/sec    │");
                println!("├────────────┼──────────────┼────────────────┼────────────────┤");

                for &num_tiles in &num_tiles_list {
                    let len = num_tiles * 256;

                    // Create test data
                    let mut host_data = vec![half::f16::ZERO; len];
                    for i in 0..len {
                        host_data[i] = half::f16::from_f32((i % 256) as f32 * 0.001);
                    }

                    let mut state =
                        WmmaState::from_host(&rt, &host_data).expect("WMMA state creation failed");

                    // Warmup
                    for _ in 0..3 {
                        let _ = run_wmma_transform(&rt, &mut state, &hadamard, 10);
                    }

                    // Reset
                    let mut state =
                        WmmaState::from_host(&rt, &host_data).expect("WMMA state reset failed");

                    // Benchmark
                    let start = Instant::now();
                    run_wmma_transform(&rt, &mut state, &hadamard, depth)
                        .expect("WMMA transform failed");
                    rt.synchronize().expect("Sync failed");
                    let elapsed_bench = start.elapsed().as_secs_f64();

                    // Ops: depth transforms * num_tiles tiles
                    let gate_ops = (depth as u64) * (num_tiles as u64);
                    let amplitude_ops = gate_ops * 256; // Each tile is 16x16=256 matmul
                    let gate_ops_per_sec = (gate_ops as f64) / elapsed_bench;
                    let amp_ops_per_sec = (amplitude_ops as f64) / elapsed_bench;

                    println!(
                        "│{:>11} │{:>13} │{:>15.2e} │{:>15.2e} │",
                        num_tiles, len, gate_ops_per_sec, amp_ops_per_sec
                    );
                }

                println!("└────────────┴──────────────┴────────────────┴────────────────┘");
                println!("");

                // Run a final large benchmark for headline numbers
                let big_tiles = 4096;
                let big_depth = 10000u32;
                let big_len = big_tiles * 256;

                let mut host_data = vec![half::f16::ZERO; big_len];
                for i in 0..big_len {
                    host_data[i] = half::f16::from_f32((i % 256) as f32 * 0.001);
                }

                let mut state =
                    WmmaState::from_host(&rt, &host_data).expect("Big WMMA state failed");

                // Warmup
                for _ in 0..3 {
                    let _ = run_wmma_transform(&rt, &mut state, &hadamard, 100);
                }

                let mut state =
                    WmmaState::from_host(&rt, &host_data).expect("Big WMMA reset failed");

                let start = Instant::now();
                run_wmma_transform(&rt, &mut state, &hadamard, big_depth).expect("Big WMMA failed");
                rt.synchronize().expect("Sync failed");
                let big_elapsed = start.elapsed().as_secs_f64();

                let big_gate_ops = (big_depth as u64) * (big_tiles as u64);
                let big_amp_ops = big_gate_ops * 256;
                let big_gate_ops_sec = (big_gate_ops as f64) / big_elapsed;
                let big_amp_ops_sec = (big_amp_ops as f64) / big_elapsed;

                println!("╔══════════════════════════════════════════════════════════════╗");
                println!(
                    "║  HEADLINE NUMBERS ({} tiles × {} depth)              ║",
                    big_tiles, big_depth
                );
                println!("╚══════════════════════════════════════════════════════════════╝");
                println!("");
                println!("  🚀 WMMA TENSOR CORE OPS/SEC:");
                println!("     Gate ops/sec:      {:.2e}", big_gate_ops_sec);
                println!("     Amplitude ops/sec: {:.2e}", big_amp_ops_sec);
                println!("");
                println!("  Elapsed: {:.6}s", big_elapsed);
                println!("  Total gate ops: {}", big_gate_ops);
                println!("  Total amplitude ops: {}", big_amp_ops);

                elapsed = big_elapsed;
                total_eval = big_gate_ops;
            }
            // ================================================================
            // EPIC 69-72: Modern GPU FP32 Backend Benchmark
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::CudaFp32Modern => {
                use engine::cuda::{
                    CudaRuntime, GpuQState, is_cuda_available, run_batched_hadamard,
                };

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA not available");
                    std::process::exit(1);
                }

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");
                let n_qubits = _qubits.max(8);
                let depth = warmup.max(1000) as u32;
                let iterations = iters.max(10);
                let n = 1usize << n_qubits;

                println!("╔══════════════════════════════════════════════════════════════╗");
                println!("║  EPIC 66-69: GPU FP32 Backend Benchmark                       ║");
                println!("╚══════════════════════════════════════════════════════════════╝");
                println!("");
                println!("Configuration:");
                println!("  qubits: {}", n_qubits);
                println!(
                    "  state_size: {} amplitudes ({:.1} KB)",
                    n,
                    (n * 8) as f64 / 1024.0
                );
                println!("  depth: {} Hadamard applications per iteration", depth);
                println!("  iterations: {}", iterations);
                println!("");

                // Create GPU-resident state
                let mut gpu_state = GpuQState::new_zero_resident(&rt, n_qubits, 1)
                    .expect("Failed to create GPU state");

                // Warmup
                for _ in 0..3 {
                    let _ = run_batched_hadamard(&rt, &mut gpu_state, depth);
                }

                // Timed runs
                let mut times = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    let mut state =
                        GpuQState::new_zero_resident(&rt, n_qubits, 1).expect("Reset failed");
                    let start = Instant::now();
                    run_batched_hadamard(&rt, &mut state, depth).expect("FP32 kernel failed");
                    rt.synchronize().expect("Sync failed");
                    times.push(start.elapsed().as_secs_f64());
                }

                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_time = times[times.len() / 2];
                let total_ops = depth as u64;
                let ops_per_sec = (total_ops as f64) / median_time;

                println!("Results (median of {} runs):", iterations);
                println!("  elapsed_s: {:.6}", median_time);
                println!("  depth: {}", depth);
                println!("  ops_per_sec: {:.2}", ops_per_sec);
                println!(
                    "  latency_per_op_ns: {:.2}",
                    median_time * 1e9 / depth as f64
                );

                elapsed = median_time;
                total_eval = total_ops;
            }
            // ================================================================
            // EPIC 69-72: WMMA Aligned Backend Benchmark
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::CudaWmmaAligned => {
                use engine::cuda::{
                    CudaRuntime, WmmaState, is_cuda_available, is_wmma_available,
                    run_wmma_hadamard_cached,
                };

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA not available");
                    std::process::exit(1);
                }

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                if !is_wmma_available(&rt) {
                    eprintln!("ERROR: WMMA not available (requires SM 7.0+)");
                    std::process::exit(1);
                }

                let n_qubits = _qubits.max(8);
                let depth = warmup.max(1000) as u32;
                let iterations = iters.max(10);
                let n = 1usize << n_qubits;
                let num_tiles = n / 256; // WMMA uses 16x16 tiles

                println!("╔══════════════════════════════════════════════════════════════╗");
                println!("║  EPIC 69: WMMA Aligned Backend Benchmark (Tensor Cores)      ║");
                println!("╚══════════════════════════════════════════════════════════════╝");
                println!("");
                println!("Configuration:");
                println!("  qubits: {}", n_qubits);
                println!(
                    "  state_size: {} amplitudes ({:.1} KB)",
                    n,
                    (n * 4) as f64 / 1024.0
                );
                println!("  tiles: {}", num_tiles);
                println!("  depth: {} Hadamard applications per iteration", depth);
                println!("  iterations: {}", iterations);
                println!("  precision: FP16 (half)");
                println!("");

                // Create WMMA state (FP16) - interleaved format
                let mut host_data: Vec<half::f16> = vec![half::f16::ZERO; n * 2];
                host_data[0] = half::f16::ONE; // real part of |0>

                let mut wmma_state =
                    WmmaState::from_host(&rt, &host_data).expect("Failed to create WMMA state");

                // Warmup
                for _ in 0..3 {
                    let _ = run_wmma_hadamard_cached(&rt, &mut wmma_state, depth);
                }

                // Timed runs
                let mut times = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    let mut state = WmmaState::from_host(&rt, &host_data).expect("Reset failed");
                    let start = Instant::now();
                    run_wmma_hadamard_cached(&rt, &mut state, depth).expect("WMMA kernel failed");
                    rt.synchronize().expect("Sync failed");
                    times.push(start.elapsed().as_secs_f64());
                }

                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_time = times[times.len() / 2];
                let total_ops = depth as u64;
                let ops_per_sec = (total_ops as f64) / median_time;

                println!("Results (median of {} runs):", iterations);
                println!("  elapsed_s: {:.6}", median_time);
                println!("  depth: {}", depth);
                println!("  ops_per_sec: {:.2}", ops_per_sec);
                println!(
                    "  latency_per_op_ns: {:.2}",
                    median_time * 1e9 / depth as f64
                );

                elapsed = median_time;
                total_eval = total_ops;
            }
            // ================================================================
            // EPIC 71: WMMA Packed Backend Benchmark
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::CudaWmmaPacked => {
                use engine::cuda::{
                    CudaRuntime, GpuQState, WmmaGateType, is_cuda_available,
                    is_packed_wmma_available, is_wmma_available, run_wmma_packed,
                };
                use engine::fusion::WmmaPackingMeta;

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA not available");
                    std::process::exit(1);
                }

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                if !is_wmma_available(&rt) {
                    eprintln!("ERROR: WMMA not available (requires SM 7.0+)");
                    std::process::exit(1);
                }

                if !is_packed_wmma_available(&rt) {
                    eprintln!("ERROR: Packed WMMA not available");
                    std::process::exit(1);
                }

                let n_qubits = _qubits.max(8);
                let depth = warmup.max(1000) as u32;
                let iterations = iters.max(10);
                let n = 1usize << n_qubits;

                // Misaligned span: qubits 2..6 (4 qubits, not starting at 0)
                let q_lo = 2u8;
                let q_hi = 6u8.min(n_qubits);
                let span_qubits = q_hi - q_lo;

                println!("╔══════════════════════════════════════════════════════════════╗");
                println!("║  EPIC 71: WMMA Packed Backend Benchmark (Misaligned Spans)   ║");
                println!("╚══════════════════════════════════════════════════════════════╝");
                println!("");
                println!("Configuration:");
                println!("  total_qubits: {}", n_qubits);
                println!(
                    "  target_span: q{}..q{} ({} qubits)",
                    q_lo, q_hi, span_qubits
                );
                println!(
                    "  state_size: {} amplitudes ({:.1} KB)",
                    n,
                    (n * 4) as f64 / 1024.0
                );
                println!("  depth: {} Hadamard applications per iteration", depth);
                println!("  iterations: {}", iterations);
                println!("  precision: FP16 (half) with tile packing");
                println!("");

                // Create GPU state (FP32 resident, will be converted)
                let mut gpu_state = GpuQState::new_zero_resident(&rt, n_qubits, 1)
                    .expect("Failed to create GPU state");

                // Create packing metadata
                let packing_meta = WmmaPackingMeta::new(q_lo, span_qubits, n_qubits);

                // Warmup
                for _ in 0..3 {
                    let _ = run_wmma_packed(
                        &rt,
                        &mut gpu_state,
                        &packing_meta,
                        WmmaGateType::Hadamard,
                        depth,
                    );
                }

                // Timed runs
                let mut times = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    let mut state =
                        GpuQState::new_zero_resident(&rt, n_qubits, 1).expect("Reset failed");
                    let start = Instant::now();
                    run_wmma_packed(
                        &rt,
                        &mut state,
                        &packing_meta,
                        WmmaGateType::Hadamard,
                        depth,
                    )
                    .expect("WMMA packed kernel failed");
                    rt.synchronize().expect("Sync failed");
                    times.push(start.elapsed().as_secs_f64());
                }

                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_time = times[times.len() / 2];
                let total_ops = depth as u64;
                let ops_per_sec = (total_ops as f64) / median_time;

                println!("Results (median of {} runs):", iterations);
                println!("  elapsed_s: {:.6}", median_time);
                println!("  depth: {}", depth);
                println!("  ops_per_sec: {:.2}", ops_per_sec);
                println!(
                    "  latency_per_op_ns: {:.2}",
                    median_time * 1e9 / depth as f64
                );
                println!("  packing_overhead: included (pack + compute + unpack)");

                elapsed = median_time;
                total_eval = total_ops;
            }
            // ================================================================
            // EPIC 66-72: Full Backend Comparison Summary
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::CudaEpicSummary => {
                use engine::cuda::{
                    CudaRuntime, GpuQState, WmmaGateType, WmmaState, is_cuda_available,
                    is_packed_wmma_available, is_wmma_available, run_batched_hadamard,
                    run_wmma_hadamard_cached, run_wmma_packed,
                };
                use engine::fusion::WmmaPackingMeta;
                use engine::quantum::{QGate, QRng, QState, apply_gate_scalar};

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA not available");
                    std::process::exit(1);
                }

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");
                let n_qubits = _qubits.max(10);
                let depth = warmup.max(1000) as u32;
                let iterations = iters.max(10);
                let n = 1usize << n_qubits;

                println!(
                    "╔══════════════════════════════════════════════════════════════════════════╗"
                );
                println!(
                    "║          QUANTUM ENGINE PERFORMANCE BENCHMARK (EPIC 66-72)              ║"
                );
                println!(
                    "╠══════════════════════════════════════════════════════════════════════════╣"
                );
                println!(
                    "║  RTX 4070 · CUDA 13.0 · Tensor Cores · WMMA FP16                        ║"
                );
                println!(
                    "╚══════════════════════════════════════════════════════════════════════════╝"
                );
                println!("");
                println!("Configuration:");
                println!("  Qubits:     {}", n_qubits);
                println!(
                    "  State size: {} amplitudes ({:.1} KB)",
                    n,
                    (n * 8) as f64 / 1024.0
                );
                println!("  Depth:      {} Hadamard applications", depth);
                println!("  Iterations: {} (median taken)", iterations);
                println!("");

                // ============================================================
                // CPU Scalar Baseline
                // ============================================================
                let mut cpu_state = QState::new_zero(n_qubits);
                let h_gate = QGate::H(0);
                let mut rng = QRng::new(12345);

                // Warmup
                for _ in 0..10 {
                    apply_gate_scalar(&mut cpu_state, &h_gate, &mut rng);
                }

                let mut cpu_times = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    let mut state = QState::new_zero(n_qubits);
                    let start = Instant::now();
                    for _ in 0..depth {
                        apply_gate_scalar(&mut state, &h_gate, &mut rng);
                    }
                    cpu_times.push(start.elapsed().as_secs_f64());
                }
                cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let cpu_us = cpu_times[cpu_times.len() / 2] * 1e6;

                // ============================================================
                // GPU FP32
                // ============================================================
                let mut gpu_times = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    let mut state =
                        GpuQState::new_zero_resident(&rt, n_qubits, 1).expect("GPU state failed");
                    let start = Instant::now();
                    run_batched_hadamard(&rt, &mut state, depth).expect("FP32 failed");
                    rt.synchronize().expect("Sync failed");
                    gpu_times.push(start.elapsed().as_secs_f64());
                }
                gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let gpu_fp32_us = gpu_times[gpu_times.len() / 2] * 1e6;

                // ============================================================
                // WMMA Aligned (if available)
                // ============================================================
                let wmma_available = is_wmma_available(&rt);
                let mut wmma_aligned_us = 0.0f64;
                if wmma_available {
                    // Create WMMA state (FP16) - interleaved format
                    let mut host_data: Vec<half::f16> = vec![half::f16::ZERO; n * 2];
                    host_data[0] = half::f16::ONE; // real part of |0>

                    let mut wmma_times = Vec::with_capacity(iterations);
                    for _ in 0..iterations {
                        let mut state =
                            WmmaState::from_host(&rt, &host_data).expect("WMMA state failed");
                        let start = Instant::now();
                        run_wmma_hadamard_cached(&rt, &mut state, depth).expect("WMMA failed");
                        rt.synchronize().expect("Sync failed");
                        wmma_times.push(start.elapsed().as_secs_f64());
                    }
                    wmma_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    wmma_aligned_us = wmma_times[wmma_times.len() / 2] * 1e6;
                }

                // ============================================================
                // WMMA Packed (if available)
                // ============================================================
                let packed_available = is_packed_wmma_available(&rt);
                let mut wmma_packed_us = 0.0f64;
                if wmma_available && packed_available {
                    let q_lo = 2u8;
                    let span_qubits = 4u8.min(n_qubits - q_lo);
                    let packing_meta = WmmaPackingMeta::new(q_lo, span_qubits, n_qubits);

                    let mut packed_times = Vec::with_capacity(iterations);
                    for _ in 0..iterations {
                        let mut state = GpuQState::new_zero_resident(&rt, n_qubits, 1)
                            .expect("GPU state failed");
                        let start = Instant::now();
                        run_wmma_packed(
                            &rt,
                            &mut state,
                            &packing_meta,
                            WmmaGateType::Hadamard,
                            depth,
                        )
                        .expect("WMMA packed failed");
                        rt.synchronize().expect("Sync failed");
                        packed_times.push(start.elapsed().as_secs_f64());
                    }
                    packed_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    wmma_packed_us = packed_times[packed_times.len() / 2] * 1e6;
                }

                // ============================================================
                // Results Table
                // ============================================================
                let cpu_vs_cpu = 1.0f64;
                let fp32_vs_cpu = if gpu_fp32_us > 0.0 {
                    cpu_us / gpu_fp32_us
                } else {
                    0.0
                };
                let wmma_vs_cpu = if wmma_aligned_us > 0.0 {
                    cpu_us / wmma_aligned_us
                } else {
                    0.0
                };
                let wmma_vs_fp32 = if wmma_aligned_us > 0.0 && gpu_fp32_us > 0.0 {
                    gpu_fp32_us / wmma_aligned_us
                } else {
                    0.0
                };
                let packed_vs_cpu = if wmma_packed_us > 0.0 {
                    cpu_us / wmma_packed_us
                } else {
                    0.0
                };
                let packed_vs_fp32 = if wmma_packed_us > 0.0 && gpu_fp32_us > 0.0 {
                    gpu_fp32_us / wmma_packed_us
                } else {
                    0.0
                };

                println!("┌─────────────────────────────┬────────────┬────────────┬────────────┐");
                println!("│ Backend                     │  Time (µs) │   vs CPU   │  vs FP32   │");
                println!("├─────────────────────────────┼────────────┼────────────┼────────────┤");
                println!(
                    "│ CPU Scalar (baseline)       │{:>11.1} │{:>9.1}×  │            │",
                    cpu_us, cpu_vs_cpu
                );
                println!(
                    "│ GPU FP32 (EPIC 66)          │{:>11.1} │{:>9.1}×  │{:>9.1}×  │",
                    gpu_fp32_us, fp32_vs_cpu, 1.0
                );
                if wmma_available {
                    println!(
                        "│ WMMA Aligned (EPIC 69)      │{:>11.1} │{:>9.1}×  │{:>9.1}×  │",
                        wmma_aligned_us, wmma_vs_cpu, wmma_vs_fp32
                    );
                } else {
                    println!(
                        "│ WMMA Aligned (EPIC 69)      │        N/A │        N/A │        N/A │"
                    );
                }
                if wmma_available && packed_available {
                    println!(
                        "│ WMMA Packed (EPIC 71)       │{:>11.1} │{:>9.1}×  │{:>9.1}×  │",
                        wmma_packed_us, packed_vs_cpu, packed_vs_fp32
                    );
                } else {
                    println!(
                        "│ WMMA Packed (EPIC 71)       │        N/A │        N/A │        N/A │"
                    );
                }
                println!("└─────────────────────────────┴────────────┴────────────┴────────────┘");
                println!("");

                // Best result
                let best_us = if wmma_available && wmma_aligned_us > 0.0 {
                    wmma_aligned_us.min(wmma_packed_us.max(0.001))
                } else {
                    gpu_fp32_us
                };
                let best_name = if wmma_available && wmma_aligned_us <= wmma_packed_us.max(0.001) {
                    "WMMA Aligned"
                } else if wmma_available && packed_available {
                    "WMMA Packed"
                } else {
                    "GPU FP32"
                };
                let total_speedup = cpu_us / best_us;

                // Calculate throughput metrics
                let best_ops_per_sec = (depth as f64) / (best_us / 1e6);
                let best_amp_ops_per_sec = best_ops_per_sec * (n as f64);
                let best_kernels_per_sec = 1.0 / (best_us / 1e6); // 1 kernel per run

                // For WMMA, each run is 1 kernel; for comparison, estimate equivalent kernel rate
                let _wmma_kernels_per_sec = if wmma_aligned_us > 0.0 {
                    1e6 / wmma_aligned_us
                } else {
                    0.0
                };
                let _fp32_kernels_per_sec = if gpu_fp32_us > 0.0 {
                    1e6 / gpu_fp32_us
                } else {
                    0.0
                };

                println!("Peak Performance:");
                println!(
                    "  Best backend:  {} ({:.1} µs for {} ops)",
                    best_name, best_us, depth
                );
                println!("");
                println!(
                    "  Gate ops/sec:       {:.2e}  ({:.2} M)",
                    best_ops_per_sec,
                    best_ops_per_sec / 1e6
                );
                println!(
                    "  Amplitude ops/sec:  {:.2e}  ({:.2} G)",
                    best_amp_ops_per_sec,
                    best_amp_ops_per_sec / 1e9
                );
                println!(
                    "  Kernels/sec:        {:.2e}  ({:.0} K)",
                    best_kernels_per_sec,
                    best_kernels_per_sec / 1e3
                );
                println!(
                    "  Latency/op:         {:.2} ns",
                    best_us * 1000.0 / depth as f64
                );
                println!("");

                println!(
                    "┌─────────────────────────────────────────────────────────────────────────┐"
                );
                println!(
                    "│                         SPEEDUP SUMMARY                                 │"
                );
                println!(
                    "├─────────────────────────────────────────────────────────────────────────┤"
                );
                println!(
                    "│  CPU → GPU FP32:            {:>5.1}×  (EPIC 66: CUDA Backend)            │",
                    fp32_vs_cpu
                );
                if wmma_available {
                    println!(
                        "│  FP32 → WMMA Aligned:       {:>5.1}×  (EPIC 69: Tensor Cores)            │",
                        wmma_vs_fp32
                    );
                }
                if wmma_available && packed_available && wmma_packed_us > 0.0 {
                    let packed_vs_aligned = wmma_aligned_us / wmma_packed_us;
                    println!(
                        "│  WMMA Aligned → Packed:     {:>5.1}×  (EPIC 71: Tile Packing)            │",
                        packed_vs_aligned
                    );
                }
                println!(
                    "├─────────────────────────────────────────────────────────────────────────┤"
                );
                println!(
                    "│  TOTAL SPEEDUP (CPU→Best):  {:>5.1}×                                      │",
                    total_speedup
                );
                println!(
                    "└─────────────────────────────────────────────────────────────────────────┘"
                );

                // High-throughput projections (depth scaling)
                let deep_depth = 1_000_000u64;
                let projected_time_us = best_us * (deep_depth as f64 / depth as f64);
                let projected_amp_ops = (deep_depth as f64) * (n as f64);
                let projected_amp_ops_per_sec = projected_amp_ops / (projected_time_us / 1e6);

                println!(
                    "┌─────────────────────────────────────────────────────────────────────────┐"
                );
                println!(
                    "│                    HIGH-THROUGHPUT PROJECTIONS                          │"
                );
                println!(
                    "├─────────────────────────────────────────────────────────────────────────┤"
                );
                println!(
                    "│  At depth 1,000,000 (projected from best backend):                     │"
                );
                println!(
                    "│    Projected time:      {:.2} ms                                        │",
                    projected_time_us / 1000.0
                );
                println!(
                    "│    Amplitude ops/sec:   {:.2e}  ({:.2} T)                       │",
                    projected_amp_ops_per_sec,
                    projected_amp_ops_per_sec / 1e12
                );
                println!(
                    "└─────────────────────────────────────────────────────────────────────────┘"
                );

                // Grep-friendly summary
                println!("");
                println!("cpu_us: {:.1}", cpu_us);
                println!("gpu_fp32_us: {:.1}", gpu_fp32_us);
                println!("wmma_aligned_us: {:.1}", wmma_aligned_us);
                println!("wmma_packed_us: {:.1}", wmma_packed_us);
                println!("speedup_cpu_to_fp32: {:.2}", fp32_vs_cpu);
                println!("speedup_cpu_to_wmma: {:.2}", wmma_vs_cpu);
                println!("speedup_total: {:.2}", total_speedup);
                println!("gate_ops_per_sec: {:.2e}", best_ops_per_sec);
                println!("amplitude_ops_per_sec: {:.2e}", best_amp_ops_per_sec);
                println!("kernels_per_sec: {:.2e}", best_kernels_per_sec);
                println!(
                    "projected_trillion_amp_ops_per_sec: {:.2}",
                    projected_amp_ops_per_sec / 1e12
                );
                // Quantum ops (same as gate/amplitude ops for quantum backends)
                println!("q_gate_ops_per_sec: {:.2e}", best_ops_per_sec);
                println!("q_amplitude_ops_per_sec: {:.2e}", best_amp_ops_per_sec);
                println!("q_ops_per_sec: {:.2e}", best_amp_ops_per_sec); // amplitude ops = quantum state updates

                elapsed = best_us / 1e6;
                total_eval = depth as u64;
            }

            // ================================================================
            // EPIC 74A: Parallel Logic Simulation (Path B)
            // ================================================================
            BenchMode::ParallelLogic => {
                // Get simulation count from --sims (default: available_parallelism)
                let default_sims = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(4);
                let sim_count = parse_arg_usize(&args, "--sims", default_sims);
                let ticks_per_sim = ticks.max(100);
                let full_grid = parse_arg_bool(&args, "--fullgrid", false);

                println!(
                    "╔══════════════════════════════════════════════════════════════════════════╗"
                );
                println!(
                    "║          EPIC 74A: PARALLEL LOGIC SIMULATION (PATH B)                    ║"
                );
                println!(
                    "╠══════════════════════════════════════════════════════════════════════════╣"
                );
                println!(
                    "║  Massive Tile Parallelism: N independent 512×512 simulations             ║"
                );
                println!(
                    "╚══════════════════════════════════════════════════════════════════════════╝"
                );
                println!("");
                println!("Configuration:");
                println!("  Parallel simulations:  {}", sim_count);
                println!("  Ticks per simulation:  {}", ticks_per_sim);
                println!("  Grid size:             512×512 = 262,144 tiles/sim");
                println!(
                    "  Total tiles:           {} ({:.1}M)",
                    sim_count * 262144,
                    (sim_count as f64 * 262144.0) / 1e6
                );
                println!(
                    "  Eval mode:             {}",
                    if full_grid {
                        "full_grid (all tiles)"
                    } else {
                        "dirty_propagation"
                    }
                );
                println!(
                    "  Unchecked bounds:      {}",
                    if unchecked { "on" } else { "off" }
                );
                println!("");

                // Run the parallel simulation benchmark
                let result = if full_grid {
                    engine::bench_api::bench_api::parallel_sim_bench_full_grid(
                        sim_count,
                        ticks_per_sim,
                    )
                } else {
                    engine::bench_api::bench_api::parallel_sim_bench(
                        sim_count,
                        ticks_per_sim,
                        unchecked,
                    )
                };

                // Calculate metrics
                let evals_per_sec = result.evals_per_sec;
                let evals_per_sim_sec = evals_per_sec / sim_count as f64;

                println!(
                    "┌─────────────────────────────────────────────────────────────────────────┐"
                );
                println!(
                    "│                         RESULTS                                         │"
                );
                println!(
                    "├─────────────────────────────────────────────────────────────────────────┤"
                );
                println!(
                    "│  Total Evaluations:     {:>15}                                │",
                    result.total_evals
                );
                println!(
                    "│  Total Changes:         {:>15}                                │",
                    result.total_changes
                );
                println!(
                    "│  Elapsed Time:          {:>12.3} s                                  │",
                    result.elapsed_secs
                );
                println!(
                    "├─────────────────────────────────────────────────────────────────────────┤"
                );
                println!(
                    "│  Throughput (total):    {:>12.2e} logic_evals/sec                   │",
                    evals_per_sec
                );
                println!(
                    "│  Throughput (per sim):  {:>12.2e} logic_evals/sec                   │",
                    evals_per_sim_sec
                );
                println!(
                    "└─────────────────────────────────────────────────────────────────────────┘"
                );
                println!("");

                // Scale analysis
                let target_100b = 100_000_000_000f64; // 100 billion
                let needed_sims = (target_100b / evals_per_sim_sec).ceil() as usize;
                let current_fraction = evals_per_sec / target_100b;

                println!(
                    "┌─────────────────────────────────────────────────────────────────────────┐"
                );
                println!(
                    "│                    SCALE ANALYSIS (100B Target)                        │"
                );
                println!(
                    "├─────────────────────────────────────────────────────────────────────────┤"
                );
                println!(
                    "│  Current throughput:    {:>12.2e} ({:.2}% of 100B)                  │",
                    evals_per_sec,
                    current_fraction * 100.0
                );
                println!(
                    "│  Sims needed for 100B:  {:>12}                                     │",
                    needed_sims
                );
                if needed_sims > sim_count * 10 {
                    println!(
                        "│  Note: CPU scaling limited; consider Path A (GPU tile eval)           │"
                    );
                }
                println!(
                    "└─────────────────────────────────────────────────────────────────────────┘"
                );

                // Grep-friendly summary
                println!("");
                println!("sim_count: {}", sim_count);
                println!("ticks_per_sim: {}", ticks_per_sim);
                println!("total_evals: {}", result.total_evals);
                println!("total_changes: {}", result.total_changes);
                println!("elapsed_s: {:.6}", result.elapsed_secs);
                println!("logic_evals_per_sec: {:.2e}", evals_per_sec);
                println!("logic_evals_per_sim_per_sec: {:.2e}", evals_per_sim_sec);
                println!("target_100b_sims_needed: {}", needed_sims);

                elapsed = result.elapsed_secs;
                total_eval = result.total_evals;
                total_change = result.total_changes;
            }

            // ================================================================
            // EPIC 74B: GPU Tile Evaluation (Path A)
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::GpuTileEval => {
                use engine::cuda::{CudaRuntime, is_cuda_available};
                use engine::cuda_tiles::benchmark_tile_eval_gpu;

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA not available for GPU tile evaluation");
                    std::process::exit(1);
                }

                let steps = ticks.max(100);
                let warmup_steps = warmup.max(10);

                println!(
                    "╔══════════════════════════════════════════════════════════════════════════╗"
                );
                println!(
                    "║          EPIC 74B: GPU TILE EVALUATION (PATH A)                          ║"
                );
                println!(
                    "╠══════════════════════════════════════════════════════════════════════════╣"
                );
                println!(
                    "║  CUDA-accelerated logic tile evaluation                                  ║"
                );
                println!(
                    "╚══════════════════════════════════════════════════════════════════════════╝"
                );
                println!("");
                println!("Configuration:");
                println!("  Evaluation steps:   {}", steps);
                println!("  Warmup steps:       {}", warmup_steps);
                println!("  Grid size:          512×512 = 262,144 tiles");
                println!("  Target:             100B logic_evals/sec");
                println!("");

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                match benchmark_tile_eval_gpu(&rt, steps, warmup_steps) {
                    Ok((total, secs, evals_per_sec)) => {
                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                         RESULTS                                         │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  Total Evaluations:     {:>15}                                │",
                            total
                        );
                        println!(
                            "│  Elapsed Time:          {:>12.6} s                                  │",
                            secs
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  GPU Throughput:        {:>12.2e} logic_evals/sec                   │",
                            evals_per_sec
                        );
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );
                        println!("");

                        // Compare to CPU baseline (1.7B for full grid, 8 sims)
                        let cpu_baseline = 1.7e9;
                        let speedup = evals_per_sec / cpu_baseline;
                        let target_100b = 100_000_000_000f64;
                        let pct_target = evals_per_sec / target_100b * 100.0;

                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                         ANALYSIS                                        │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  CPU Baseline (8 sims): {:>12.2e} evals/sec                         │",
                            cpu_baseline
                        );
                        println!(
                            "│  GPU vs CPU:            {:>12.2}× speedup                            │",
                            speedup
                        );
                        println!(
                            "│  Target 100B progress:  {:>12.2}%                                     │",
                            pct_target
                        );
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );

                        // Grep-friendly summary
                        println!("");
                        println!("gpu_tile_eval_total: {}", total);
                        println!("gpu_tile_eval_elapsed_s: {:.6}", secs);
                        println!("gpu_tile_eval_per_sec: {:.2e}", evals_per_sec);
                        println!("gpu_vs_cpu_speedup: {:.2}", speedup);
                        println!("target_100b_pct: {:.2}", pct_target);

                        elapsed = secs;
                        total_eval = total;
                    }
                    Err(e) => {
                        eprintln!("ERROR: GPU tile evaluation failed: {:?}", e);
                        std::process::exit(1);
                    }
                }
            }

            // ================================================================
            // EPIC 75.1: Multi-World GPU Tile Evaluation
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::GpuTileMulti => {
                use engine::cuda::{CudaRuntime, is_cuda_available};
                use engine::cuda_tiles::benchmark_tile_eval_multi_gpu;

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA not available for multi-world GPU tile evaluation");
                    std::process::exit(1);
                }

                let num_worlds = parse_arg_usize(&args, "--worlds", 4);
                let steps = ticks.max(100);
                let warmup_steps = warmup.max(10);

                println!(
                    "╔══════════════════════════════════════════════════════════════════════════╗"
                );
                println!(
                    "║          EPIC 75.1: MULTI-WORLD GPU TILE EVALUATION                      ║"
                );
                println!(
                    "╠══════════════════════════════════════════════════════════════════════════╣"
                );
                println!(
                    "║  Parallel worlds on GPU for maximum throughput                           ║"
                );
                println!(
                    "╚══════════════════════════════════════════════════════════════════════════╝"
                );
                println!("");
                println!("Configuration:");
                println!("  Parallel worlds:    {}", num_worlds);
                println!("  Evaluation steps:   {}", steps);
                println!("  Warmup steps:       {}", warmup_steps);
                println!("  Grid size/world:    512×512 = 262,144 tiles");
                println!(
                    "  Total tiles:        {} ({:.1}M)",
                    num_worlds * 262144,
                    (num_worlds as f64 * 262144.0) / 1e6
                );
                println!("");

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                match benchmark_tile_eval_multi_gpu(&rt, num_worlds, steps, warmup_steps) {
                    Ok((total, secs, evals_per_sec, memory_mb)) => {
                        let evals_per_world = evals_per_sec / num_worlds as f64;

                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                         RESULTS                                         │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  Total Evaluations:     {:>15}                                │",
                            total
                        );
                        println!(
                            "│  Elapsed Time:          {:>12.6} s                                  │",
                            secs
                        );
                        println!(
                            "│  GPU Memory Used:       {:>12.1} MB                                  │",
                            memory_mb
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  TOTAL Throughput:      {:>12.2e} logic_evals/sec                   │",
                            evals_per_sec
                        );
                        println!(
                            "│  Per-World Throughput:  {:>12.2e} logic_evals/sec                   │",
                            evals_per_world
                        );
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );
                        println!("");

                        // Scaling analysis
                        let single_world_baseline = 22.7e9; // From EPIC 74B
                        let scaling_efficiency =
                            (evals_per_sec / single_world_baseline) / num_worlds as f64 * 100.0;
                        let target_100b = 100_000_000_000f64;
                        let pct_target = evals_per_sec / target_100b * 100.0;

                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                         SCALING ANALYSIS                                │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  Single-world baseline: {:>12.2e} evals/sec                         │",
                            single_world_baseline
                        );
                        println!(
                            "│  Scaling efficiency:    {:>12.1}% (vs linear)                        │",
                            scaling_efficiency
                        );
                        println!(
                            "│  Target 100B progress:  {:>12.1}%                                     │",
                            pct_target
                        );
                        if evals_per_sec >= target_100b {
                            println!(
                                "│  🎯 TARGET 100B ACHIEVED!                                              │"
                            );
                        }
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );

                        // Grep-friendly summary
                        println!("");
                        println!("num_worlds: {}", num_worlds);
                        println!("steps: {}", steps);
                        println!("gpu_memory_mb: {:.1}", memory_mb);
                        println!("total_evals: {}", total);
                        println!("elapsed_s: {:.6}", secs);
                        println!("logic_evals_per_sec: {:.2e}", evals_per_sec);
                        println!("per_world_evals_per_sec: {:.2e}", evals_per_world);
                        println!("scaling_efficiency_pct: {:.1}", scaling_efficiency);
                        println!("target_100b_pct: {:.1}", pct_target);

                        elapsed = secs;
                        total_eval = total;
                    }
                    Err(e) => {
                        eprintln!("ERROR: Multi-world GPU tile evaluation failed: {:?}", e);
                        std::process::exit(1);
                    }
                }
            }

            // ================================================================
            // EPIC 75.3: Depth-Batched GPU Tile Evaluation
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::GpuTileDepth => {
                use engine::cuda::{CudaRuntime, is_cuda_available};
                use engine::cuda_tiles::benchmark_tile_eval_depth_gpu;

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA not available for depth-batched GPU tile evaluation");
                    std::process::exit(1);
                }

                let num_worlds = parse_arg_usize(&args, "--worlds", 5);
                let depth = parse_arg_usize(&args, "--depth", 10) as u32;
                let steps = ticks.max(100);
                let warmup_steps = warmup.max(10);

                println!(
                    "╔══════════════════════════════════════════════════════════════════════════╗"
                );
                println!(
                    "║          EPIC 75.3: DEPTH-BATCHED GPU TILE EVALUATION                    ║"
                );
                println!(
                    "╠══════════════════════════════════════════════════════════════════════════╣"
                );
                println!(
                    "║  Multiple timesteps per kernel launch with ping-pong buffers             ║"
                );
                println!(
                    "╚══════════════════════════════════════════════════════════════════════════╝"
                );
                println!("");
                println!("Configuration:");
                println!("  Parallel worlds:    {}", num_worlds);
                println!("  Depth per launch:   {} steps", depth);
                println!("  Total steps:        {}", steps);
                println!("  Warmup steps:       {}", warmup_steps);
                println!("  Grid size/world:    512×512 = 262,144 tiles");
                println!(
                    "  Total tiles:        {} ({:.1}M)",
                    num_worlds * 262144,
                    (num_worlds as f64 * 262144.0) / 1e6
                );
                println!("");

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                match benchmark_tile_eval_depth_gpu(&rt, num_worlds, steps, depth, warmup_steps) {
                    Ok((total, secs, evals_per_sec, memory_mb)) => {
                        let evals_per_world = evals_per_sec / num_worlds as f64;
                        let kernel_launches = (steps + depth as u32 - 1) / depth as u32;

                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                         RESULTS                                         │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  Total Evaluations:     {:>15}                                │",
                            total
                        );
                        println!(
                            "│  Elapsed Time:          {:>12.6} s                                  │",
                            secs
                        );
                        println!(
                            "│  GPU Memory Used:       {:>12.1} MB                                  │",
                            memory_mb
                        );
                        println!(
                            "│  Kernel Launches:       {:>12}                                      │",
                            kernel_launches
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  TOTAL Throughput:      {:>12.2e} logic_evals/sec                   │",
                            evals_per_sec
                        );
                        println!(
                            "│  Per-World Throughput:  {:>12.2e} logic_evals/sec                   │",
                            evals_per_world
                        );
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );
                        println!("");

                        // Comparison with multi-world baseline (75.1)
                        let multi_world_baseline = 33.6e9; // From EPIC 75.1
                        let single_world_baseline = 22.7e9; // From EPIC 74B
                        let improvement_vs_multi = evals_per_sec / multi_world_baseline * 100.0;
                        let improvement_vs_single = evals_per_sec / single_world_baseline * 100.0;
                        let target_100b = 100_000_000_000f64;
                        let pct_target = evals_per_sec / target_100b * 100.0;

                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                     DEPTH BATCHING ANALYSIS                             │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  Multi-world baseline:  {:>12.2e} evals/sec (75.1)                  │",
                            multi_world_baseline
                        );
                        println!(
                            "│  vs Multi-world:        {:>12.1}%                                    │",
                            improvement_vs_multi
                        );
                        println!(
                            "│  vs Single-world:       {:>12.1}%                                    │",
                            improvement_vs_single
                        );
                        println!(
                            "│  Target 100B progress:  {:>12.1}%                                     │",
                            pct_target
                        );
                        if evals_per_sec >= target_100b {
                            println!(
                                "│  🎯 TARGET 100B ACHIEVED!                                              │"
                            );
                        }
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );

                        // Grep-friendly summary
                        println!("");
                        println!("num_worlds: {}", num_worlds);
                        println!("depth_per_launch: {}", depth);
                        println!("total_steps: {}", steps);
                        println!("kernel_launches: {}", kernel_launches);
                        println!("gpu_memory_mb: {:.1}", memory_mb);
                        println!("total_evals: {}", total);
                        println!("elapsed_s: {:.6}", secs);
                        println!("logic_evals_per_sec: {:.2e}", evals_per_sec);
                        println!("per_world_evals_per_sec: {:.2e}", evals_per_world);
                        println!("vs_multi_world_pct: {:.1}", improvement_vs_multi);
                        println!("vs_single_world_pct: {:.1}", improvement_vs_single);
                        println!("target_100b_pct: {:.1}", pct_target);

                        elapsed = secs;
                        total_eval = total;
                    }
                    Err(e) => {
                        eprintln!("ERROR: Depth-batched GPU tile evaluation failed: {:?}", e);
                        std::process::exit(1);
                    }
                }
            }

            // ================================================================
            // EPIC 75.3B: Shared Memory Stencil GPU Tile Evaluation
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::GpuTileStencil => {
                use engine::cuda::{CudaRuntime, is_cuda_available};
                use engine::cuda_tiles::benchmark_tile_eval_stencil_gpu;

                if !is_cuda_available() {
                    eprintln!(
                        "ERROR: CUDA not available for shared memory stencil GPU tile evaluation"
                    );
                    std::process::exit(1);
                }

                let num_worlds = parse_arg_usize(&args, "--worlds", 5);
                let depth = parse_arg_usize(&args, "--depth", 50) as u32;
                let steps = ticks.max(100);
                let warmup_steps = warmup.max(10);

                println!(
                    "╔══════════════════════════════════════════════════════════════════════════╗"
                );
                println!(
                    "║       EPIC 75.3B: SHARED MEMORY STENCIL GPU TILE EVALUATION              ║"
                );
                println!(
                    "╠══════════════════════════════════════════════════════════════════════════╣"
                );
                println!(
                    "║  16x16 blocks with halo loaded into shared memory                        ║"
                );
                println!(
                    "╚══════════════════════════════════════════════════════════════════════════╝"
                );
                println!("");
                println!("Configuration:");
                println!("  Parallel worlds:    {}", num_worlds);
                println!("  Depth per launch:   {} steps", depth);
                println!("  Total steps:        {}", steps);
                println!("  Warmup steps:       {}", warmup_steps);
                println!("  Grid size/world:    512×512 = 262,144 tiles");
                println!("  Block size:         16×16 (+ 1-wide halo = 18×18 shared)");
                println!("  Shared mem/block:   2,592 bytes (18×18×8)");
                println!(
                    "  Total tiles:        {} ({:.1}M)",
                    num_worlds * 262144,
                    (num_worlds as f64 * 262144.0) / 1e6
                );
                println!("");

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                match benchmark_tile_eval_stencil_gpu(&rt, num_worlds, steps, depth, warmup_steps) {
                    Ok((total, secs, evals_per_sec, memory_mb)) => {
                        let evals_per_world = evals_per_sec / num_worlds as f64;
                        let kernel_launches = (steps + depth as u32 - 1) / depth as u32;

                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                         RESULTS                                         │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  Total Evaluations:     {:>15}                                │",
                            total
                        );
                        println!(
                            "│  Elapsed Time:          {:>12.6} s                                  │",
                            secs
                        );
                        println!(
                            "│  GPU Memory Used:       {:>12.1} MB                                  │",
                            memory_mb
                        );
                        println!(
                            "│  Kernel Launches:       {:>12}                                      │",
                            kernel_launches
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  TOTAL Throughput:      {:>12.2e} logic_evals/sec                   │",
                            evals_per_sec
                        );
                        println!(
                            "│  Per-World Throughput:  {:>12.2e} logic_evals/sec                   │",
                            evals_per_world
                        );
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );
                        println!("");

                        // Comparison with baselines
                        let depth_baseline = 40.0e9; // From EPIC 75.3A
                        let multi_world_baseline = 33.6e9; // From EPIC 75.1
                        let single_world_baseline = 22.7e9; // From EPIC 74B
                        let improvement_vs_depth = evals_per_sec / depth_baseline * 100.0;
                        let improvement_vs_multi = evals_per_sec / multi_world_baseline * 100.0;
                        let improvement_vs_single = evals_per_sec / single_world_baseline * 100.0;
                        let target_100b = 100_000_000_000f64;
                        let pct_target = evals_per_sec / target_100b * 100.0;

                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                  SHARED MEMORY STENCIL ANALYSIS                         │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  Depth-batch baseline:  {:>12.2e} evals/sec (75.3A)                 │",
                            depth_baseline
                        );
                        println!(
                            "│  vs Depth-batch:        {:>12.1}%                                    │",
                            improvement_vs_depth
                        );
                        println!(
                            "│  vs Multi-world:        {:>12.1}%                                    │",
                            improvement_vs_multi
                        );
                        println!(
                            "│  vs Single-world:       {:>12.1}%                                    │",
                            improvement_vs_single
                        );
                        println!(
                            "│  Target 100B progress:  {:>12.1}%                                     │",
                            pct_target
                        );
                        if evals_per_sec >= target_100b {
                            println!(
                                "│  🎯 TARGET 100B ACHIEVED!                                              │"
                            );
                        }
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );

                        // Grep-friendly summary
                        println!("");
                        println!("num_worlds: {}", num_worlds);
                        println!("depth_per_launch: {}", depth);
                        println!("total_steps: {}", steps);
                        println!("kernel_launches: {}", kernel_launches);
                        println!("gpu_memory_mb: {:.1}", memory_mb);
                        println!("total_evals: {}", total);
                        println!("elapsed_s: {:.6}", secs);
                        println!("logic_evals_per_sec: {:.2e}", evals_per_sec);
                        println!("per_world_evals_per_sec: {:.2e}", evals_per_world);
                        println!("vs_depth_batch_pct: {:.1}", improvement_vs_depth);
                        println!("vs_multi_world_pct: {:.1}", improvement_vs_multi);
                        println!("vs_single_world_pct: {:.1}", improvement_vs_single);
                        println!("target_100b_pct: {:.1}", pct_target);

                        elapsed = secs;
                        total_eval = total;
                    }
                    Err(e) => {
                        eprintln!(
                            "ERROR: Shared memory stencil GPU tile evaluation failed: {:?}",
                            e
                        );
                        std::process::exit(1);
                    }
                }
            }

            // ================================================================
            // EPIC 75.4: GPU vs CPU Correctness Validation
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::GpuCorrectness => {
                use engine::cuda::{CudaRuntime, is_cuda_available};
                use engine::cuda_tiles::correctness_report;

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA not available for GPU correctness validation");
                    std::process::exit(1);
                }

                let steps = ticks.max(10) as u32;
                let depth = parse_arg_usize(&args, "--depth", 50) as u32;

                println!("Running GPU vs CPU correctness validation...");
                println!("  Steps:           {}", steps);
                println!("  Depth per launch: {}", depth);
                println!("");

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                match correctness_report(&rt, steps, depth) {
                    Ok(report) => {
                        print!("{}", report);

                        // Grep-friendly
                        println!("mode: correctness");
                        println!("steps: {}", steps);
                        println!("depth: {}", depth);
                    }
                    Err(e) => {
                        eprintln!("ERROR: GPU correctness validation failed: {:?}", e);
                        std::process::exit(1);
                    }
                }
            }

            // ================================================================
            // EPIC 120: Packed 1-Bit Tile Evaluation (64 tiles per u64)
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::PackedTiles => {
                use engine::cuda::{CudaRuntime, is_cuda_available};
                use engine::cuda_tiles::{PackedKernelVariant, benchmark_packed_tiles};

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA not available for packed tile evaluation");
                    std::process::exit(1);
                }

                let width = parse_arg_usize(&args, "--width", 8192);
                let height = parse_arg_usize(&args, "--height", 8192);
                let depth = parse_arg_usize(&args, "--depth", 100) as u32;
                let steps = ticks.max(1000) as u32;
                let warmup_steps = warmup.max(100) as u32;

                // Kernel variant selection
                let variant = if args.iter().any(|a| a == "--register-v3" || a == "-r3") {
                    PackedKernelVariant::RegisterV3
                } else if args.iter().any(|a| a == "--register-v2" || a == "-r2") {
                    PackedKernelVariant::RegisterV2
                } else if args.iter().any(|a| a == "--register" || a == "-r") {
                    PackedKernelVariant::Register
                } else if args.iter().any(|a| a == "--smem" || a == "--shared") {
                    PackedKernelVariant::SharedMemory
                } else if args.iter().any(|a| a == "--shuffle" || a == "-s") {
                    PackedKernelVariant::Shuffle
                } else {
                    PackedKernelVariant::Basic
                };

                let tile_count = width * height;
                let words = ((width + 63) / 64) * height;

                println!(
                    "╔══════════════════════════════════════════════════════════════════════════╗"
                );
                println!(
                    "║          EPIC 120: PACKED 1-BIT TILE EVALUATION                          ║"
                );
                println!(
                    "╠══════════════════════════════════════════════════════════════════════════╣"
                );
                println!(
                    "║  64 spatial tiles packed per u64 word - targeting 10T+ tiles/sec         ║"
                );
                println!(
                    "╚══════════════════════════════════════════════════════════════════════════╝"
                );
                println!("");
                println!("Configuration:");
                println!(
                    "  Grid size:          {}×{} = {} tiles",
                    width, height, tile_count
                );
                println!(
                    "  Packed words:       {} ({:.2} MB)",
                    words,
                    (words * 8) as f64 / 1e6
                );
                println!("  Depth per launch:   {} steps", depth);
                println!("  Total steps:        {}", steps);
                println!("  Warmup steps:       {}", warmup_steps);
                let variant_name = match variant {
                    PackedKernelVariant::Basic => "basic",
                    PackedKernelVariant::Shuffle => "shuffle (warp intrinsics)",
                    PackedKernelVariant::SharedMemory => "shared memory (smem + shuffle)",
                    PackedKernelVariant::Register => "register (16 rows/thread)",
                    PackedKernelVariant::RegisterV2 => "register-v2 (8 rows + coop halo)",
                    PackedKernelVariant::RegisterV3 => "register-v3 (32 rows, 100T+ target)",
                    PackedKernelVariant::Heterogeneous => "heterogeneous (per-word tile types)",
                };
                println!("  Kernel variant:     {}", variant_name);
                println!("");

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                match benchmark_packed_tiles(
                    &rt,
                    width,
                    height,
                    steps,
                    depth,
                    warmup_steps,
                    variant,
                ) {
                    Ok((total, secs, tiles_per_sec, memory_mb)) => {
                        let kernel_launches = (steps + depth - 1) / depth;

                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                         RESULTS                                         │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  Total Tile Evaluations: {:>15}                               │",
                            total
                        );
                        println!(
                            "│  Elapsed Time:           {:>12.6} s                                 │",
                            secs
                        );
                        println!(
                            "│  GPU Memory Used:        {:>12.2} MB                                 │",
                            memory_mb
                        );
                        println!(
                            "│  Kernel Launches:        {:>12}                                     │",
                            kernel_launches
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );

                        // Format throughput nicely
                        let (throughput_val, throughput_unit) = if tiles_per_sec >= 1e12 {
                            (tiles_per_sec / 1e12, "T")
                        } else if tiles_per_sec >= 1e9 {
                            (tiles_per_sec / 1e9, "B")
                        } else {
                            (tiles_per_sec / 1e6, "M")
                        };

                        println!(
                            "│  🚀 THROUGHPUT:          {:>12.2} {} tiles/sec                       │",
                            throughput_val, throughput_unit
                        );
                        println!(
                            "│     (raw):               {:>12.3e} tiles/sec                        │",
                            tiles_per_sec
                        );
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );
                        println!("");

                        // Comparison with baseline
                        let baseline_u64 = 200e9; // 200B from u64-per-tile depth-batch
                        let improvement = tiles_per_sec / baseline_u64;
                        let target_10t = 10e12;
                        let pct_target = tiles_per_sec / target_10t * 100.0;

                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                     PERFORMANCE ANALYSIS                                │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  u64-per-tile baseline:  {:>12.2e} tiles/sec                        │",
                            baseline_u64
                        );
                        println!(
                            "│  Improvement factor:     {:>12.1}×                                   │",
                            improvement
                        );
                        println!(
                            "│  Target 10T progress:    {:>12.2}%                                   │",
                            pct_target
                        );
                        if tiles_per_sec >= target_10t {
                            println!(
                                "│  🎯 TARGET 10 TRILLION ACHIEVED!                                        │"
                            );
                        } else if tiles_per_sec >= 1e12 {
                            println!(
                                "│  ✓ TRILLION-SCALE ACHIEVED!                                             │"
                            );
                        }
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );

                        // Grep-friendly summary
                        println!("");
                        println!("mode: packed-1bit");
                        println!("width: {}", width);
                        println!("height: {}", height);
                        println!("tile_count: {}", tile_count);
                        println!("packed_words: {}", words);
                        println!("depth_per_launch: {}", depth);
                        println!("total_steps: {}", steps);
                        println!("kernel_launches: {}", kernel_launches);
                        println!("gpu_memory_mb: {:.2}", memory_mb);
                        println!("total_evals: {}", total);
                        println!("elapsed_s: {:.6}", secs);
                        println!("tiles_per_sec: {:.3e}", tiles_per_sec);
                        println!("improvement_vs_u64: {:.1}", improvement);
                        println!("target_10t_pct: {:.2}", pct_target);
                        println!("kernel_variant: {}", variant_name);

                        elapsed = secs;
                        total_eval = total;
                    }
                    Err(e) => {
                        eprintln!("ERROR: Packed tile benchmark failed: {:?}", e);
                        std::process::exit(1);
                    }
                }
            }

            // ================================================================
            // EPIC 121: TileAnneal - Packed Ising Optimization
            // ================================================================
            #[cfg(feature = "cuda")]
            BenchMode::Ising => {
                use engine::cuda::{CudaRuntime, is_cuda_available};
                use engine::cuda_tiles::{
                    GpuIsingGrid, IsingConfig, IsingKernelVariant, benchmark_ising,
                    print_energy_log_csv, run_ising_instrumented, run_ising_validation_suite,
                };

                if !is_cuda_available() {
                    eprintln!("ERROR: CUDA not available for Ising simulation");
                    std::process::exit(1);
                }

                let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

                // Check for validation mode
                let validate_mode = args.iter().any(|a| a == "--validate" || a == "--test");

                if validate_mode {
                    println!("\n");
                    match run_ising_validation_suite(&rt) {
                        Ok(results) => {
                            let passed = results.iter().filter(|r| r.passed).count();
                            if passed == results.len() {
                                println!("\nAll validation tests PASSED");
                            } else {
                                println!("\nSome validation tests FAILED");
                                std::process::exit(1);
                            }
                        }
                        Err(e) => {
                            eprintln!("ERROR: Validation failed: {:?}", e);
                            std::process::exit(1);
                        }
                    }
                    return;
                }

                // Parse Ising-specific arguments
                let width = parse_arg_usize(&args, "--width", 8192);
                let height = parse_arg_usize(&args, "--height", width);
                let sweeps = parse_arg_u32(&args, "--sweeps", 1000);
                let warmup_sweeps = parse_arg_u32(&args, "--warmup", 100);
                let beta = parse_arg_f32(&args, "--beta", 0.44); // Near critical temp for 2D Ising

                // Check for T=0 mode
                let variant = if args.iter().any(|a| a == "--t0" || a == "--zero-temp") {
                    IsingKernelVariant::ZeroTemperature
                } else {
                    IsingKernelVariant::Stochastic
                };

                // Check for instrumented mode (logs energy trajectory)
                let instrumented = args
                    .iter()
                    .any(|a| a == "--instrumented" || a == "--log-energy");

                let config = IsingConfig::new(width, height)
                    .with_beta(beta)
                    .with_seed(42);

                let spin_count = config.spin_count();
                let words = (width + 63) / 64 * height;
                let memory_mb = (words * 8 * 2) as f64 / (1024.0 * 1024.0);

                let variant_name = match variant {
                    IsingKernelVariant::Stochastic => "stochastic",
                    IsingKernelVariant::ZeroTemperature => "zero-temperature",
                };

                println!("");
                println!(
                    "┌─────────────────────────────────────────────────────────────────────────┐"
                );
                println!(
                    "│                  EPIC 121: TileAnneal Ising Benchmark                  │"
                );
                println!(
                    "├─────────────────────────────────────────────────────────────────────────┤"
                );
                println!(
                    "│  Grid:         {:>8} × {:<8}  ({:>12} spins)               │",
                    width, height, spin_count
                );
                println!(
                    "│  Packed words: {:>12}                                             │",
                    words
                );
                println!(
                    "│  GPU Memory:   {:>12.2} MB                                        │",
                    memory_mb
                );
                println!(
                    "│  Sweeps:       {:>12} (warmup: {})                              │",
                    sweeps, warmup_sweeps
                );
                println!(
                    "│  Beta (1/T):   {:>12.4}                                           │",
                    beta
                );
                println!(
                    "│  Kernel:       {:>16}                                       │",
                    variant_name
                );
                println!(
                    "└─────────────────────────────────────────────────────────────────────────┘"
                );
                println!("");

                // Instrumented mode: log energy trajectory
                if instrumented {
                    println!(
                        "Running instrumented simulation (logging energy every 10 sweeps)...\n"
                    );

                    let mut grid = GpuIsingGrid::new(&rt, &config).expect("Failed to create grid");
                    let j_coupling = 1.0; // Ferromagnetic
                    let log_interval = 10u32;

                    match run_ising_instrumented(
                        &rt,
                        &mut grid,
                        sweeps,
                        log_interval,
                        variant,
                        j_coupling,
                    ) {
                        Ok(energy_log) => {
                            // Print CSV
                            print_energy_log_csv(&energy_log);

                            // Summary
                            let initial_e = energy_log.first().map(|x| x.1).unwrap_or(0);
                            let final_e = energy_log.last().map(|x| x.1).unwrap_or(0);
                            let final_m = energy_log.last().map(|x| x.2).unwrap_or(0.0);

                            println!(
                                "\n# Summary: E {} -> {}, M={:.4}",
                                initial_e, final_e, final_m
                            );
                            println!("# Energy decreased: {}", final_e < initial_e);
                        }
                        Err(e) => {
                            eprintln!("ERROR: Instrumented run failed: {:?}", e);
                            std::process::exit(1);
                        }
                    }
                    return;
                }

                println!("Running benchmark...");

                match benchmark_ising(&rt, &config, sweeps, warmup_sweeps, variant) {
                    Ok((total_updates, secs, updates_per_sec)) => {
                        let (throughput_val, throughput_unit) = if updates_per_sec >= 1e12 {
                            (updates_per_sec / 1e12, "T")
                        } else if updates_per_sec >= 1e9 {
                            (updates_per_sec / 1e9, "B")
                        } else {
                            (updates_per_sec / 1e6, "M")
                        };

                        println!("");
                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                            ISING RESULTS                                │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  Total spin updates: {:>15}                                  │",
                            total_updates
                        );
                        println!(
                            "│  Elapsed time:       {:>15.6} s                               │",
                            secs
                        );
                        println!(
                            "│                                                                         │"
                        );
                        println!(
                            "│  🚀 THROUGHPUT:      {:>12.2} {} spin-updates/sec                   │",
                            throughput_val, throughput_unit
                        );
                        println!(
                            "│     (raw):           {:>12.3e} updates/sec                        │",
                            updates_per_sec
                        );
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );
                        println!("");

                        // Target analysis
                        let target_10t = 10e12;
                        let pct_target = updates_per_sec / target_10t * 100.0;

                        println!(
                            "┌─────────────────────────────────────────────────────────────────────────┐"
                        );
                        println!(
                            "│                     PERFORMANCE ANALYSIS                                │"
                        );
                        println!(
                            "├─────────────────────────────────────────────────────────────────────────┤"
                        );
                        println!(
                            "│  Target (10T updates/s): {:>12.1}%                                   │",
                            pct_target
                        );
                        if updates_per_sec >= target_10t {
                            println!(
                                "│  🎯 TARGET 10 TRILLION ACHIEVED!                                        │"
                            );
                        } else if updates_per_sec >= 1e12 {
                            println!(
                                "│  ✓ TRILLION-SCALE ACHIEVED!                                             │"
                            );
                        }
                        println!(
                            "│                                                                         │"
                        );
                        println!(
                            "│  For 100M spins at this rate: {:>8.2} sweeps/sec                      │",
                            updates_per_sec / 100_000_000.0
                        );
                        println!(
                            "│  For 1B spins at this rate:   {:>8.2} sweeps/sec                      │",
                            updates_per_sec / 1_000_000_000.0
                        );
                        println!(
                            "└─────────────────────────────────────────────────────────────────────────┘"
                        );

                        // Grep-friendly summary
                        println!("");
                        println!("mode: ising");
                        println!("width: {}", width);
                        println!("height: {}", height);
                        println!("spin_count: {}", spin_count);
                        println!("packed_words: {}", words);
                        println!("sweeps: {}", sweeps);
                        println!("warmup_sweeps: {}", warmup_sweeps);
                        println!("beta: {}", beta);
                        println!("kernel_variant: {}", variant_name);
                        println!("gpu_memory_mb: {:.2}", memory_mb);
                        println!("total_updates: {}", total_updates);
                        println!("elapsed_s: {:.6}", secs);
                        println!("updates_per_sec: {:.3e}", updates_per_sec);
                        println!("target_10t_pct: {:.2}", pct_target);

                        elapsed = secs;
                        total_eval = total_updates;
                    }
                    Err(e) => {
                        eprintln!("ERROR: Ising benchmark failed: {:?}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
    }
    #[cfg(not(feature = "perf-bench"))]
    {
        let start = Instant::now();
        for _ in 0..ticks {
            sim.tick();
        }
        elapsed = start.elapsed().as_secs_f64();
    }
    let tile_count = (WIDTH as u64) * (HEIGHT as u64);
    let ticks_per_sec = (ticks as f64) / elapsed;
    let tiles_per_sec = (tile_count as f64) * ticks_per_sec;
    // Quantum ops accounting (estimate): gates_per_tick * ticks_per_sec, where gates_per_tick = sum(2^Q) over QTiles
    let q_ops_per_tick_est = sim.quantum_ops_per_tick_estimate() as f64;
    let quantum_ops_per_sec_est = q_ops_per_tick_est * ticks_per_sec;

    // Derive ratios and ops/sec when counts are available
    let (dirty_ratio, change_ratio, eval_ops_per_sec, change_ops_per_sec) = if ticks > 0 {
        let avg_eval = (total_eval as f64) / (ticks as f64);
        let avg_change = (total_change as f64) / (ticks as f64);
        let dirty_ratio = if tile_count > 0 {
            avg_eval / (tile_count as f64)
        } else {
            0.0
        };
        let change_ratio = if tile_count > 0 {
            avg_change / (tile_count as f64)
        } else {
            0.0
        };
        let eval_ops_per_sec = (total_eval as f64) / elapsed;
        let change_ops_per_sec = (total_change as f64) / elapsed;
        (
            dirty_ratio,
            change_ratio,
            eval_ops_per_sec,
            change_ops_per_sec,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    // Stable, grep-friendly lines
    println!("mode: {:?}", mode);
    println!("ticks: {}", ticks);
    println!("warmup: {}", warmup);
    println!("reps: {}", reps);
    println!("iters: {}", iters);
    println!("dispatch: {}", dispatch);
    println!("unchecked: {}", if unchecked { "on" } else { "off" });
    println!("fields_dispatch: {}", fields_dispatch);
    println!("elapsed_s: {:.6}", elapsed);
    // Report backend if relevant
    if matches!(
        mode,
        BenchMode::QuantumBackendOnly | BenchMode::QuantumScalarOnly
    ) {
        println!("backend: {}", backend_str);
    }
    println!("ticks_per_sec: {:.2}", ticks_per_sec);
    println!("tiles_per_sec: {:.2}", tiles_per_sec);
    println!("dirty_ratio: {:.6}", dirty_ratio);
    println!("change_ratio: {:.6}", change_ratio);
    println!("logic_eval_ops_per_sec: {:.2}", eval_ops_per_sec);
    println!("logic_change_ops_per_sec: {:.2}", change_ops_per_sec);
    println!("quantum_ops_per_sec_est: {:.2}", quantum_ops_per_sec_est);
    // FullTick: print per-kernel breakdown (estimate-based) for quick reference
    if matches!(mode, BenchMode::FullTick) {
        let logic_ops_per_sec = eval_ops_per_sec;
        let field_ops_per_sec = 0.0f64; // fields not part of FullTick path
        let q_ops_per_sec = quantum_ops_per_sec_est;
        let total_ops_per_sec_est = logic_ops_per_sec + field_ops_per_sec + q_ops_per_sec;
        println!("logic_ops_per_sec: {:.2}", logic_ops_per_sec);
        println!("field_ops_per_sec: {:.2}", field_ops_per_sec);
        println!("q_ops_per_sec_est: {:.2}", q_ops_per_sec);
        println!("total_ops_per_sec_est: {:.2}", total_ops_per_sec_est);
    }
    // Combined mode: report measured per-kernel and total
    if matches!(mode, BenchMode::CombinedLogicQuantum) {
        let logic_ops_per_sec = eval_ops_per_sec;
        let q_ops_per_sec_measured = if elapsed > 0.0 {
            (measured_q_ops as f64) / elapsed
        } else {
            0.0
        };
        let q_gates_per_sec_measured = if elapsed > 0.0 {
            (measured_q_gates as f64) / elapsed
        } else {
            0.0
        };
        let total_ops_per_sec = logic_ops_per_sec + q_ops_per_sec_measured;
        println!("kernel_id: logic");
        println!("logic_ops_per_sec: {:.2}", logic_ops_per_sec);
        println!("kernel_id: quantum");
        println!("q_ops_per_sec_measured: {:.2}", q_ops_per_sec_measured);
        println!("q_gates_per_sec_measured: {:.2}", q_gates_per_sec_measured);
        println!("total_ops_per_sec: {:.2}", total_ops_per_sec);
        // For reference, also print estimate-based total
        let total_ops_per_sec_est = tiles_per_sec * dirty_ratio + quantum_ops_per_sec_est;
        println!("total_ops_per_sec_est: {:.2}", total_ops_per_sec_est);
    }
    if matches!(mode, BenchMode::QuantumBackendOnly) {
        // Recompute q_flops/sec based on the same program contract used in quantum_backend_only
        // Per-iter FLOPs = 6 * len; ops = 4 * len (for our fixed 4-gate program)
        let qlen = (1u64) << (_qubits as u64);
        let per_iter_flops = 6.0 * (qlen as f64);
        let per_iter_ops = 4.0 * (qlen as f64);
        let flops_per_sec = (per_iter_flops * (iters.max(1) as f64)) / elapsed;
        let qops_per_sec = (per_iter_ops * (iters.max(1) as f64)) / elapsed;
        println!("q_flops_per_sec: {:.2}", flops_per_sec);
        println!("q_ops_per_sec: {:.2}", qops_per_sec);
        // Optional: log fast-path vs general hits when JIT_DEBUG=1
        #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
        {
            if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
                let (fast, general) = quantum_jit::h_debug_counts();
                println!(
                    "jit_debug: h_fastpath_hits={} h_general_hits={}",
                    fast, general
                );
            }
        }
    }
    // Explicit contract/version for ops semantics to avoid ambiguity in future changes
    println!("ops_def: eval_ops=tile_evals_v1 change_ops=changed_tiles_v1");

    // World checksum to prove identical initial state across runs/profiles
    let world_checksum: u64 = sim.tilemap.tiles.iter().fold(0u64, |acc, t| {
        acc.wrapping_mul(31).wrapping_add(t.meta.tile_type as u64)
    });
    println!("world_checksum: 0x{:016x}", world_checksum);

    // Phase hint to document cold-start vs steady-state behavior
    let phase = if warmup >= 128 {
        "steady_state"
    } else {
        "cold_start"
    };
    println!("phase: {}", phase);

    // Final logic checksum to prove identical end state across modes/builds
    let logic_checksum: u64 = sim
        .tilemap
        .values_snapshot()
        .iter()
        .fold(0u64, |acc, v| acc.wrapping_mul(31).wrapping_add(*v));
    println!("logic_checksum: 0x{:016x}", logic_checksum);

    // EPIC59A: final summary for JIT backend to ensure counters are visible even if interim logging was suppressed
    #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
    {
        if matches!(mode, BenchMode::QuantumBackendOnly)
            && backend_str.to_ascii_lowercase() == "jit"
        {
            let fast = quantum_jit::H_FASTPATH_HITS.load(std::sync::atomic::Ordering::Relaxed);
            let general = quantum_jit::H_GENERAL_HITS.load(std::sync::atomic::Ordering::Relaxed);
            let fast_addr = &quantum_jit::H_FASTPATH_HITS as *const _ as usize;
            let gen_addr = &quantum_jit::H_GENERAL_HITS as *const _ as usize;
            eprintln!(
                "EPIC59A_ADDR_STDERR: fast_addr=0x{:x} general_addr=0x{:x}",
                fast_addr, gen_addr
            );
            eprintln!(
                "EPIC59A_FINAL_STDERR: h_fastpath_hits={} h_general_hits={}",
                fast, general
            );
            println!(
                "EPIC59A_FINAL: h_fastpath_hits={} h_general_hits={} fast_addr=0x{:x} general_addr=0x{:x}",
                fast, general, fast_addr, gen_addr
            );
        }
    }
}
