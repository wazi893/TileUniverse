// quantum-engine CLI — EPIC 64 Cluster Mode
//
// This binary provides commands for running distributed quantum simulations
// across multiple machines.
//
// Commands:
//   worker     - Start a worker node that accepts jobs from a coordinator
//   run        - Run a coordinator that dispatches jobs to worker nodes

use engine::distributed::NodeTransport;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "worker" => {
            if let Err(e) = run_worker(&args[2..]) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        "run" => {
            if let Err(e) = run_coordinator(&args[2..]) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        "--help" | "-h" => {
            print_usage();
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("quantum-engine — Distributed Quantum Simulation Engine");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  quantum-engine worker --node-id <ID> --bind <ADDR> --tiles <N>");
    eprintln!("  quantum-engine run --cluster <PATH> --qubits <N> --depth <D> --kernels <K>");
    eprintln!();
    eprintln!("COMMANDS:");
    eprintln!("  worker    Start a worker node");
    eprintln!("  run       Run a coordinator");
    eprintln!();
    eprintln!("WORKER OPTIONS:");
    eprintln!("  --node-id <ID>     Node ID (must match cluster.toml)");
    eprintln!("  --bind <ADDR>      Bind address (e.g., 0.0.0.0:5001)");
    eprintln!("  --tiles <N>        Number of tiles (must be divisible by 4)");
    eprintln!();
    eprintln!("RUN OPTIONS:");
    eprintln!("  --cluster <PATH>   Path to cluster.toml");
    eprintln!("  --qubits <N>       Number of qubits");
    eprintln!("  --depth <D>        Kernel depth");
    eprintln!("  --kernels <K>      Number of kernels to run");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("  # Start worker on node 0");
    eprintln!("  quantum-engine worker --node-id 0 --bind 0.0.0.0:5001 --tiles 4");
    eprintln!();
    eprintln!("  # Run coordinator");
    eprintln!("  quantum-engine run --cluster cluster.toml --qubits 4 --depth 100 --kernels 100");
}

fn run_worker(args: &[String]) -> Result<(), String> {
    let mut node_id: Option<u16> = None;
    let mut bind_addr: Option<String> = None;
    let mut tiles: Option<u16> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--node-id" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --node-id".to_string());
                }
                node_id = Some(
                    args[i]
                        .parse()
                        .map_err(|_| format!("Invalid node ID: {}", args[i]))?,
                );
            }
            "--bind" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --bind".to_string());
                }
                bind_addr = Some(args[i].clone());
            }
            "--tiles" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --tiles".to_string());
                }
                tiles = Some(
                    args[i]
                        .parse()
                        .map_err(|_| format!("Invalid tile count: {}", args[i]))?,
                );
            }
            _ => {
                return Err(format!("Unknown worker option: {}", args[i]));
            }
        }
        i += 1;
    }

    let node_id = node_id.ok_or("Missing required option: --node-id")?;
    let bind_addr = bind_addr.ok_or("Missing required option: --bind")?;
    let tiles = tiles.ok_or("Missing required option: --tiles")?;

    // Validate tiles (EPIC 62Q: must be divisible by 4)
    if tiles == 0 || tiles % 4 != 0 {
        return Err(format!(
            "Tiles must be non-zero and divisible by 4 (got {})",
            tiles
        ));
    }

    eprintln!("[quantum-engine] Starting worker node");
    eprintln!("  Node ID: {}", node_id);
    eprintln!("  Bind address: {}", bind_addr);
    eprintln!("  Tiles: {}", tiles);
    eprintln!();

    // Enable strict mode for production
    unsafe {
        std::env::set_var("JIT_DEBUG_STRICT", "1");
        std::env::set_var("JIT_TILE_COUNT", "4");
    }

    // Start TCP worker
    engine::distributed::tcp_worker_main(&bind_addr)?;

    Ok(())
}

fn run_coordinator(args: &[String]) -> Result<(), String> {
    let mut cluster_path: Option<String> = None;
    let mut qubits: Option<u8> = None;
    let mut depth: Option<u32> = None;
    let mut kernels: Option<u32> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--cluster" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --cluster".to_string());
                }
                cluster_path = Some(args[i].clone());
            }
            "--qubits" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --qubits".to_string());
                }
                qubits = Some(
                    args[i]
                        .parse()
                        .map_err(|_| format!("Invalid qubit count: {}", args[i]))?,
                );
            }
            "--depth" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --depth".to_string());
                }
                depth = Some(
                    args[i]
                        .parse()
                        .map_err(|_| format!("Invalid depth: {}", args[i]))?,
                );
            }
            "--kernels" => {
                i += 1;
                if i >= args.len() {
                    return Err("Missing value for --kernels".to_string());
                }
                kernels = Some(
                    args[i]
                        .parse()
                        .map_err(|_| format!("Invalid kernel count: {}", args[i]))?,
                );
            }
            _ => {
                return Err(format!("Unknown run option: {}", args[i]));
            }
        }
        i += 1;
    }

    let cluster_path = cluster_path.ok_or("Missing required option: --cluster")?;
    let qubits = qubits.ok_or("Missing required option: --qubits")?;
    let depth = depth.ok_or("Missing required option: --depth")?;
    let kernels = kernels.ok_or("Missing required option: --kernels")?;

    eprintln!("[quantum-engine] Starting coordinator");
    eprintln!("  Cluster config: {}", cluster_path);
    eprintln!("  Qubits: {}", qubits);
    eprintln!("  Depth: {}", depth);
    eprintln!("  Kernels per node: {}", kernels);
    eprintln!();

    // Load and validate cluster config
    let config = engine::distributed::ClusterConfig::from_file(&cluster_path)?;
    config.validate()?;

    eprintln!("[coordinator] Loaded cluster: {}", config.cluster.name);
    eprintln!("[coordinator] Nodes: {}", config.node.len());
    eprintln!("[coordinator] Total tiles: {}", config.total_tiles());
    eprintln!();

    // Enable strict mode
    unsafe {
        std::env::set_var("JIT_DEBUG_STRICT", "1");
        std::env::set_var("JIT_TILE_COUNT", "4");
        std::env::set_var("JIT_H_DEPTH", &depth.to_string());
    }

    // Compile JIT kernel once
    eprintln!(
        "[coordinator] Compiling JIT kernel (qubits={}, depth={})...",
        qubits, depth
    );
    let _h_kernel = engine::quantum_jit::clif::compile_h_kernel(qubits, 0)
        .ok_or("Failed to compile JIT kernel")?;
    eprintln!("[coordinator] ✓ JIT kernel compiled");
    eprintln!();

    // Connect to all nodes
    let mut transports = Vec::new();
    for node in &config.node {
        let addr = format!("{}:{}", node.host, node.port);
        eprintln!("[coordinator] Connecting to node {} @ {}...", node.id, addr);

        let mut transport = engine::distributed::TcpTransport::connect(node.id, &addr)?;

        // Initialize node
        let (tile_start, tile_end) = config
            .tile_range_for_node(node.id)
            .ok_or(format!("No tile range for node {}", node.id))?;

        transport.send_job(engine::distributed::DistributedJob::InitNode {
            node_id: node.id,
            tile_start,
            tile_end,
            qubits,
        })?;

        eprintln!(
            "[coordinator] ✓ Node {} initialized: tiles [{}..{})",
            node.id, tile_start, tile_end
        );

        transports.push((node.id, transport));
    }

    eprintln!();
    eprintln!("[coordinator] All nodes connected and initialized");
    eprintln!(
        "[coordinator] Running {} kernels on {} nodes...",
        kernels,
        transports.len()
    );
    eprintln!();

    // Run kernels
    let job = engine::distributed::DistributedJob::RunKernel {
        kernel_type: engine::distributed::KernelType::HDeep,
        depth,
        backend: engine::distributed::BackendKind::SimdJit,
        jit_fn_ptr: None, // TCP workers compile their own kernels
    };

    let start = std::time::Instant::now();

    for i in 0..kernels {
        if i % 10 == 0 && i > 0 {
            eprintln!("[coordinator] Progress: {}/{} kernels", i, kernels);
        }

        // Dispatch to all nodes
        for (node_id, transport) in &mut transports {
            transport
                .send_job(job.clone())
                .map_err(|e| format!("Failed to send to node {}: {}", node_id, e))?;
        }

        // Collect results
        for (node_id, transport) in &mut transports {
            let _result = transport
                .recv_result()
                .map_err(|e| format!("Failed to receive from node {}: {}", node_id, e))?;
        }
    }

    let elapsed = start.elapsed();

    eprintln!();
    eprintln!("[coordinator] ✓ All kernels completed");
    eprintln!("[coordinator] Total time: {:.2}s", elapsed.as_secs_f64());
    eprintln!(
        "[coordinator] Throughput: {:.2} kernels/sec (aggregate)",
        (kernels * transports.len() as u32) as f64 / elapsed.as_secs_f64()
    );

    let q_ops_per_kernel = (qubits as u32 * depth) as f64;
    let q_ops_per_sec =
        (kernels * transports.len() as u32) as f64 * q_ops_per_kernel / elapsed.as_secs_f64();
    eprintln!(
        "[coordinator] q_ops throughput: {:.2}M q_ops/sec",
        q_ops_per_sec / 1e6
    );

    // Shutdown nodes
    eprintln!();
    eprintln!("[coordinator] Shutting down nodes...");
    for (node_id, transport) in &mut transports {
        transport
            .send_job(engine::distributed::DistributedJob::Shutdown)
            .map_err(|e| format!("Failed to shutdown node {}: {}", node_id, e))?;
    }

    eprintln!("[coordinator] ✓ Cluster shutdown complete");

    Ok(())
}
