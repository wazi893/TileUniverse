# Distributed Quantum Simulation — Cluster Mode

EPIC 64: Real cluster mode for running quantum simulations across multiple machines.

## Overview

The quantum-engine CLI enables distributed quantum simulation across a cluster of worker nodes. Each worker runs tiles of the quantum state in parallel, with the coordinator dispatching kernels and collecting results.

## Architecture

```
Coordinator (quantum-engine run)
  ├─ Node 0 @ 192.168.1.10:5001  →  TileFarm (4 tiles)
  ├─ Node 1 @ 192.168.1.11:5001  →  TileFarm (4 tiles)
  └─ Node 2 @ 192.168.1.12:5001  →  TileFarm (8 tiles)
```

- **Coordinator**: Dispatches jobs to workers, collects results, reports throughput
- **Workers**: Execute quantum kernels on their assigned tile ranges
- **Tile Sharding**: Global quantum state partitioned across workers (no cross-node entanglement)

## Quick Start

### 1. Build the Binary

```bash
cargo build --release --features "quantum_jit,cranelift_jit,cluster" --bin quantum-engine
```

The binary will be at `./target/release/quantum-engine`.

### 2. Create a Cluster Configuration

Create `cluster.toml`:

```toml
[cluster]
name = "my-cluster"

# Node 0: First worker
[[node]]
id = 0
host = "192.168.1.10"
port = 5001
tiles = 4  # Must be divisible by 4 (EPIC 62Q constraint)

# Node 1: Second worker
[[node]]
id = 1
host = "192.168.1.11"
port = 5001
tiles = 4

# Node 2: Third worker (more powerful machine)
[[node]]
id = 2
host = "192.168.1.12"
port = 5001
tiles = 8
```

**Notes:**
- Each node must have a unique `id`
- `tiles` must be divisible by 4 (EPIC 62Q constraint)
- Nodes with more `tiles` will handle more of the workload

### 3. Start Worker Nodes

On each worker machine, run:

```bash
# On 192.168.1.10 (Node 0)
./target/release/quantum-engine worker \
  --node-id 0 \
  --bind 0.0.0.0:5001 \
  --tiles 4

# On 192.168.1.11 (Node 1)
./target/release/quantum-engine worker \
  --node-id 1 \
  --bind 0.0.0.0:5001 \
  --tiles 4

# On 192.168.1.12 (Node 2)
./target/release/quantum-engine worker \
  --node-id 2 \
  --bind 0.0.0.0:5001 \
  --tiles 8
```

**Worker Logs:**
```
[quantum-engine] Starting worker node
  Node ID: 0
  Bind address: 0.0.0.0:5001
  Tiles: 4

[tcp-worker] Listening on 0.0.0.0:5001
```

### 4. Run the Coordinator

From the coordinator machine:

```bash
./target/release/quantum-engine run \
  --cluster cluster.toml \
  --qubits 4 \
  --depth 100 \
  --kernels 100
```

**Coordinator Logs:**
```
[quantum-engine] Starting coordinator
  Cluster config: cluster.toml
  Qubits: 4
  Depth: 100
  Kernels per node: 100

[coordinator] Loaded cluster: my-cluster
[coordinator] Nodes: 3
[coordinator] Total tiles: 16

[coordinator] Compiling JIT kernel (qubits=4, depth=100)...
[coordinator] ✓ JIT kernel compiled

[coordinator] Connecting to node 0 @ 192.168.1.10:5001...
[coordinator] ✓ Node 0 initialized: tiles [0..4)
[coordinator] Connecting to node 1 @ 192.168.1.11:5001...
[coordinator] ✓ Node 1 initialized: tiles [4..8)
[coordinator] Connecting to node 2 @ 192.168.1.12:5001...
[coordinator] ✓ Node 2 initialized: tiles [8..16)

[coordinator] All nodes connected and initialized
[coordinator] Running 100 kernels on 3 nodes...

[coordinator] ✓ All kernels completed
[coordinator] Total time: 5.23s
[coordinator] Throughput: 57.36 kernels/sec (aggregate)
[coordinator] q_ops throughput: 2.29M q_ops/sec

[coordinator] Shutting down nodes...
[coordinator] ✓ Cluster shutdown complete
```

## Localhost Testing

For development, you can test with multiple workers on localhost:

**cluster-localhost.toml:**
```toml
[cluster]
name = "localhost-test"

[[node]]
id = 0
host = "127.0.0.1"
port = 5001
tiles = 4

[[node]]
id = 1
host = "127.0.0.1"
port = 5002
tiles = 4
```

**Terminal 1:**
```bash
./target/release/quantum-engine worker --node-id 0 --bind 127.0.0.1:5001 --tiles 4
```

**Terminal 2:**
```bash
./target/release/quantum-engine worker --node-id 1 --bind 127.0.0.1:5002 --tiles 4
```

**Terminal 3:**
```bash
./target/release/quantum-engine run \
  --cluster cluster-localhost.toml \
  --qubits 4 \
  --depth 50 \
  --kernels 10
```

## Two-Machine Example

Minimal setup with 2 physical machines:

**Machine A (192.168.1.10):**
```bash
# Start worker
./quantum-engine worker --node-id 0 --bind 0.0.0.0:5001 --tiles 4
```

**Machine B (192.168.1.20):**

`cluster.toml`:
```toml
[cluster]
name = "two-node"

[[node]]
id = 0
host = "192.168.1.10"
port = 5001
tiles = 4

[[node]]
id = 1
host = "192.168.1.20"
port = 5001
tiles = 4
```

```bash
# Start worker locally
./quantum-engine worker --node-id 1 --bind 0.0.0.0:5001 --tiles 4

# Run coordinator (in another terminal on Machine B)
./quantum-engine run --cluster cluster.toml --qubits 4 --depth 100 --kernels 50
```

## Health Monitoring (EPIC 64 Phase 2)

Workers respond to heartbeat pings from the coordinator. If a node becomes unreachable:
- TCP timeout: 30 seconds (configurable via stream settings)
- Coordinator reports clean error: "Failed to receive from node X"

**Health States:**
- **Healthy**: Node responding normally (<10ms latency)
- **Degraded**: Node responding slowly (10-100ms latency)
- **Unreachable**: Node not responding (timeout or connection lost)

## Performance Notes

- **Transport Overhead**: TCP adds ~90% overhead compared to in-process channel
- **Scaling**: Real scaling exists but synchronous dispatch limits efficiency
- **Target**: EPIC 65 will add async/pipelined dispatch for better throughput

## Troubleshooting

**"Failed to bind"**
- Port already in use. Try a different port or kill the existing process.

**"Connection refused"**
- Worker not running or firewall blocking. Check worker logs and network connectivity.

**"Tiles must be divisible by 4"**
- EPIC 62Q constraint. Adjust `tiles` in cluster.toml to be 4, 8, 12, 16, etc.

**"Duplicate node ID"**
- Each node must have a unique `id` in cluster.toml.

## Testing

Run the cluster demo test:

```bash
cargo test --features "quantum_jit,cranelift_jit,cluster" test_p1_cluster_demo_localhost -- --ignored --nocapture
```

This spawns 2 localhost workers and runs a full cluster simulation.

## References

- **EPIC 62Q**: 4-tile-per-worker constraint
- **EPIC 63**: TCP transport with length-prefix framing
- **EPIC 64**: Cluster mode implementation (this document)
- **EPIC 65**: Planned async/batching improvements
