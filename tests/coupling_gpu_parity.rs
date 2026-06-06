//! GPU parity tests for physics-to-logic coupling.
//!
//! These tests verify that the CPU and GPU implementations of physics coupling
//! produce identical results (within epsilon for floating-point, exact for integer).

#![cfg(feature = "cuda")]

use engine::cuda_tiles::{GpuCouplingConfig, GpuTilemapCoupled, run_tile_eval_coupled_gpu};
use engine::physics::{
    HeatDegradationMode, HeatErrorCoupling, PhysicsCouplingConfig, PowerEnableCoupling,
    UnpoweredBehavior,
};
use engine::simulation::Simulation;
use engine::tile_meta::TileType;
use logic_fabric_core::cuda::CudaRuntime;
use std::sync::atomic::Ordering;

/// Helper to collect logic values from simulation
#[allow(dead_code)]
fn collect_logic_values(sim: &Simulation) -> Vec<u64> {
    sim.tilemap
        .tiles
        .iter()
        .map(|t| t.logic.load(Ordering::Relaxed))
        .collect()
}

#[test]
fn coupling_disabled_parity() {
    // When coupling is disabled, GPU should match uncoupled CPU exactly
    let rt = match CudaRuntime::new() {
        Ok(rt) => rt,
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    let mut cpu_sim = Simulation::with_size(32, 32);
    cpu_sim.set_tile(5, 5, TileType::And);
    cpu_sim.set_tile(4, 5, TileType::Wire);
    cpu_sim.set_tile(6, 5, TileType::Wire);
    cpu_sim.set_logic_value(4, 5, 0xFF);
    cpu_sim.set_logic_value(6, 5, 0xFF);

    // CPU simulation (coupling disabled by default)
    for _ in 0..3 {
        cpu_sim.tick();
    }
    let cpu_result = cpu_sim.get_logic_at(5, 5);

    // GPU simulation with coupling disabled
    let mut gpu_sim = Simulation::with_size(32, 32);
    gpu_sim.set_tile(5, 5, TileType::And);
    gpu_sim.set_tile(4, 5, TileType::Wire);
    gpu_sim.set_tile(6, 5, TileType::Wire);
    gpu_sim.set_logic_value(4, 5, 0xFF);
    gpu_sim.set_logic_value(6, 5, 0xFF);

    let mut gpu_tilemap = GpuTilemapCoupled::from_simulation(&rt, &gpu_sim).unwrap();
    let config = GpuCouplingConfig::default(); // Disabled

    // Each tick: toggle clock, then run multiple delta cycles to converge
    // Use MAX_DELTA iterations per tick (matching CPU behavior)
    const DELTA_CYCLES_PER_TICK: u32 = 100;
    let mut global_clock = false;
    let mut prev_clock;
    for _ in 0..3 {
        prev_clock = global_clock;
        global_clock = !global_clock;
        // Run delta cycles within this tick
        for _ in 0..DELTA_CYCLES_PER_TICK {
            run_tile_eval_coupled_gpu(&rt, &mut gpu_tilemap, global_clock, prev_clock, &config)
                .unwrap();
        }
    }
    rt.synchronize().unwrap();
    gpu_tilemap.to_simulation(&rt, &mut gpu_sim).unwrap();

    let gpu_result = gpu_sim.get_logic_at(5, 5);

    assert_eq!(
        cpu_result, gpu_result,
        "CPU and GPU should match when coupling disabled"
    );
}

#[test]
fn heat_coupling_parity() {
    let rt = match CudaRuntime::new() {
        Ok(rt) => rt,
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    // Configure heat coupling
    let physics_config = PhysicsCouplingConfig {
        enabled: true,
        heat_coupling: HeatErrorCoupling {
            enabled: true,
            error_threshold: 50,
            critical_threshold: 200,
            degradation_mask: 0xF,
            mode: HeatDegradationMode::BitFlip,
        },
        ..Default::default()
    };

    // CPU simulation
    let mut cpu_sim = Simulation::with_size(32, 32);
    cpu_sim.set_tile(5, 5, TileType::Wire);
    cpu_sim.set_tile(4, 5, TileType::ClockGlobal);
    cpu_sim.set_heat_field_for_test(5, 5, 100); // Above threshold
    cpu_sim.set_physics_coupling_config(physics_config.clone());

    for _ in 0..3 {
        cpu_sim.tick();
    }
    let cpu_result = cpu_sim.get_logic_at(5, 5);

    // GPU simulation
    let mut gpu_sim = Simulation::with_size(32, 32);
    gpu_sim.set_tile(5, 5, TileType::Wire);
    gpu_sim.set_tile(4, 5, TileType::ClockGlobal);
    gpu_sim.set_heat_field_for_test(5, 5, 100);
    gpu_sim.set_physics_coupling_config(physics_config.clone());

    let mut gpu_tilemap = GpuTilemapCoupled::from_simulation(&rt, &gpu_sim).unwrap();
    let gpu_config = GpuCouplingConfig::from_config(&physics_config);

    // Each tick: toggle clock, then run multiple delta cycles to converge
    const DELTA_CYCLES_PER_TICK: u32 = 100;
    let mut global_clock = false;
    let mut prev_clock;
    for _ in 0..3 {
        prev_clock = global_clock;
        global_clock = !global_clock;
        // Run delta cycles within this tick
        for _ in 0..DELTA_CYCLES_PER_TICK {
            run_tile_eval_coupled_gpu(&rt, &mut gpu_tilemap, global_clock, prev_clock, &gpu_config)
                .unwrap();
        }
    }
    rt.synchronize().unwrap();
    gpu_tilemap.to_simulation(&rt, &mut gpu_sim).unwrap();

    let gpu_result = gpu_sim.get_logic_at(5, 5);

    assert_eq!(
        cpu_result, gpu_result,
        "CPU and GPU heat coupling should produce identical results: CPU={:#x}, GPU={:#x}",
        cpu_result, gpu_result
    );
}

#[test]
fn power_coupling_parity() {
    let rt = match CudaRuntime::new() {
        Ok(rt) => rt,
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    // Configure power coupling
    let physics_config = PhysicsCouplingConfig {
        enabled: true,
        power_coupling: PowerEnableCoupling {
            enabled: true,
            min_power: 10,
            unpowered_behavior: UnpoweredBehavior::Zero,
            exempt_tiles: 1u64 << 7, // ClockGlobal exempt
        },
        ..Default::default()
    };

    // CPU simulation
    let mut cpu_sim = Simulation::with_size(32, 32);
    cpu_sim.set_tile(5, 5, TileType::And);
    cpu_sim.set_tile(4, 5, TileType::ClockGlobal);
    cpu_sim.set_power_field_for_test(5, 5, 5); // Below min_power
    cpu_sim.set_power_field_for_test(4, 5, 255); // Clock has power
    cpu_sim.set_logic_value(5, 5, 0xABCD);
    cpu_sim.set_physics_coupling_config(physics_config.clone());

    cpu_sim.tick();
    let cpu_result = cpu_sim.get_logic_at(5, 5);

    // GPU simulation
    let mut gpu_sim = Simulation::with_size(32, 32);
    gpu_sim.set_tile(5, 5, TileType::And);
    gpu_sim.set_tile(4, 5, TileType::ClockGlobal);
    gpu_sim.set_power_field_for_test(5, 5, 5);
    gpu_sim.set_power_field_for_test(4, 5, 255);
    gpu_sim.set_logic_value(5, 5, 0xABCD);
    gpu_sim.set_physics_coupling_config(physics_config.clone());

    let mut gpu_tilemap = GpuTilemapCoupled::from_simulation(&rt, &gpu_sim).unwrap();
    let gpu_config = GpuCouplingConfig::from_config(&physics_config);

    run_tile_eval_coupled_gpu(&rt, &mut gpu_tilemap, true, false, &gpu_config).unwrap();
    rt.synchronize().unwrap();
    gpu_tilemap.to_simulation(&rt, &mut gpu_sim).unwrap();

    let gpu_result = gpu_sim.get_logic_at(5, 5);

    assert_eq!(
        cpu_result, gpu_result,
        "CPU and GPU power coupling should produce identical results: CPU={:#x}, GPU={:#x}",
        cpu_result, gpu_result
    );
}

#[test]
fn all_coupling_mechanisms_parity() {
    let rt = match CudaRuntime::new() {
        Ok(rt) => rt,
        Err(_) => {
            eprintln!("CUDA not available, skipping test");
            return;
        }
    };

    // Configure all coupling mechanisms
    let physics_config = PhysicsCouplingConfig {
        enabled: true,
        heat_coupling: HeatErrorCoupling {
            enabled: true,
            error_threshold: 100,
            critical_threshold: 200,
            degradation_mask: 0xFF,
            mode: HeatDegradationMode::BitClear,
        },
        power_coupling: PowerEnableCoupling {
            enabled: true,
            min_power: 5,
            unpowered_behavior: UnpoweredBehavior::Hold,
            exempt_tiles: 1u64 << 7,
        },
        ..Default::default()
    };

    // Create a more complex circuit
    let mut cpu_sim = Simulation::with_size(64, 64);

    // Set up various tile types
    cpu_sim.set_tile(10, 10, TileType::ClockGlobal);
    cpu_sim.set_tile(11, 10, TileType::Wire);
    cpu_sim.set_tile(12, 10, TileType::And);
    cpu_sim.set_tile(13, 10, TileType::Or);
    cpu_sim.set_tile(14, 10, TileType::Xor);

    // Set physics fields
    cpu_sim.set_heat_field_for_test(12, 10, 150); // And tile hot
    cpu_sim.set_power_field_for_test(13, 10, 3); // Or tile underpowered
    cpu_sim.set_power_field_for_test(10, 10, 255); // Clock powered
    cpu_sim.set_power_field_for_test(11, 10, 255);
    cpu_sim.set_power_field_for_test(12, 10, 255);
    cpu_sim.set_power_field_for_test(14, 10, 255);

    cpu_sim.set_physics_coupling_config(physics_config.clone());

    // Run CPU simulation
    for _ in 0..5 {
        cpu_sim.tick();
    }

    // Collect CPU results
    let cpu_results: Vec<u64> = [
        cpu_sim.get_logic_at(10, 10),
        cpu_sim.get_logic_at(11, 10),
        cpu_sim.get_logic_at(12, 10),
        cpu_sim.get_logic_at(13, 10),
        cpu_sim.get_logic_at(14, 10),
    ]
    .to_vec();

    // GPU simulation
    let mut gpu_sim = Simulation::with_size(64, 64);
    gpu_sim.set_tile(10, 10, TileType::ClockGlobal);
    gpu_sim.set_tile(11, 10, TileType::Wire);
    gpu_sim.set_tile(12, 10, TileType::And);
    gpu_sim.set_tile(13, 10, TileType::Or);
    gpu_sim.set_tile(14, 10, TileType::Xor);

    gpu_sim.set_heat_field_for_test(12, 10, 150);
    gpu_sim.set_power_field_for_test(13, 10, 3);
    gpu_sim.set_power_field_for_test(10, 10, 255);
    gpu_sim.set_power_field_for_test(11, 10, 255);
    gpu_sim.set_power_field_for_test(12, 10, 255);
    gpu_sim.set_power_field_for_test(14, 10, 255);

    gpu_sim.set_physics_coupling_config(physics_config.clone());

    let mut gpu_tilemap = GpuTilemapCoupled::from_simulation(&rt, &gpu_sim).unwrap();
    let gpu_config = GpuCouplingConfig::from_config(&physics_config);

    // Each tick: toggle clock, then run multiple delta cycles to converge
    const DELTA_CYCLES_PER_TICK: u32 = 100;
    let mut global_clock = false;
    let mut prev_clock;
    for _ in 0..5 {
        prev_clock = global_clock;
        global_clock = !global_clock;
        // Run delta cycles within this tick
        for _ in 0..DELTA_CYCLES_PER_TICK {
            run_tile_eval_coupled_gpu(&rt, &mut gpu_tilemap, global_clock, prev_clock, &gpu_config)
                .unwrap();
        }
    }
    rt.synchronize().unwrap();
    gpu_tilemap.to_simulation(&rt, &mut gpu_sim).unwrap();

    // Collect GPU results
    let gpu_results: Vec<u64> = [
        gpu_sim.get_logic_at(10, 10),
        gpu_sim.get_logic_at(11, 10),
        gpu_sim.get_logic_at(12, 10),
        gpu_sim.get_logic_at(13, 10),
        gpu_sim.get_logic_at(14, 10),
    ]
    .to_vec();

    // Compare
    for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
        assert_eq!(
            cpu, gpu,
            "Mismatch at position {}: CPU={:#x}, GPU={:#x}",
            i, cpu, gpu
        );
    }
}
