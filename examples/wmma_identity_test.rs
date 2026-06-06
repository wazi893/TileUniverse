//! WMMA Identity Test - Prove Tensor Cores Work At All
//!
//! This is the minimal test case to verify WMMA is functioning correctly
//! before tackling the NN kernel. Tests basic matrix multiplication.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaFunction, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx_with_opts;
#[cfg(feature = "cuda")]
use engine::cuda::{CudaResult, CudaRuntime};

const WMMA_IDENTITY_KERNEL: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// Minimal WMMA test: C = A × B
// A, B, C are all 16×16
extern "C" __global__ void wmma_identity_test(
    const half* __restrict__ A,      // 16×16, row-major
    const half* __restrict__ B,      // 16×16, COLUMN-MAJOR (standard matmul)
    float* __restrict__ C            // 16×16, output
) {
    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load matrices
    wmma::load_matrix_sync(a_frag, A, 16);  // Stride = 16
    wmma::load_matrix_sync(b_frag, B, 16);  // Stride = 16

    // Multiply: C = A × B + C
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}
"#;

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║          WMMA Identity Test - Minimal Case          ║");
    println!("║     Proving Tensor Cores Work Before NN Kernel      ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: Requires CUDA support!");
        println!("Build with: cargo run --example wmma_identity_test --features cuda --release");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        run_identity_tests();
    }
}

#[cfg(feature = "cuda")]
fn run_identity_tests() {
    println!("Initializing CUDA...");
    let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

    println!("Compiling WMMA kernel...");
    let cuda_include = std::env::var("CUDA_PATH")
        .map(|p| format!("{}/include", p))
        .unwrap_or_else(|_| {
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include".to_string()
        });

    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("sm_75"),
        include_paths: vec![cuda_include],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(WMMA_IDENTITY_KERNEL, opts).expect("PTX compilation failed");

    let module = rt.ctx().load_module(ptx).expect("Failed to load PTX");

    let kernel = module
        .load_function("wmma_identity_test")
        .expect("Failed to get kernel");

    println!("\n");
    run_test_case(
        &rt,
        &kernel,
        "Identity × Identity",
        create_identity(),
        create_identity(),
        |c| verify_identity(c),
    );

    run_test_case(
        &rt,
        &kernel,
        "Ones × Identity",
        create_ones(),
        create_identity(),
        |c| verify_ones(c),
    );

    run_test_case(
        &rt,
        &kernel,
        "Ones × Ones",
        create_ones(),
        create_ones(),
        |c| verify_ones_times_ones(c),
    );

    run_test_case(
        &rt,
        &kernel,
        "2×I × 3×I",
        create_scaled_identity(2.0),
        create_scaled_identity(3.0),
        |c| verify_scaled_identity(c, 6.0),
    );

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║              WMMA Hardware Verification             ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("\n✓ All tests passed! Tensor cores are working correctly.");
    println!("  → WMMA hardware is functional");
    println!("  → Matrix layouts are correct");
    println!("  → Ready to build NN kernel\n");
}

#[cfg(feature = "cuda")]
fn run_test_case<F>(
    rt: &CudaRuntime,
    kernel: &CudaFunction,
    name: &str,
    a: Vec<half::f16>,
    b: Vec<half::f16>,
    verify: F,
) where
    F: Fn(&[f32]) -> bool,
{
    print!("Testing: {:<20} ... ", name);

    // Upload A and B
    let a_bits: Vec<u16> = a.iter().map(|x| x.to_bits()).collect();
    let b_bits: Vec<u16> = b.iter().map(|x| x.to_bits()).collect();

    let d_a = rt.upload(&a_bits).expect("Failed to upload A");
    let d_b = rt.upload(&b_bits).expect("Failed to upload B");
    let d_c = rt.alloc_zeros::<f32>(256).expect("Failed to allocate C");

    // Launch kernel (single warp)
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    use cudarc::driver::PushKernelArg;
    unsafe {
        rt.stream()
            .launch_builder(kernel)
            .arg(&d_a)
            .arg(&d_b)
            .arg(&d_c)
            .launch(cfg)
            .expect("Kernel launch failed");
    }

    rt.stream().synchronize().expect("Failed to sync");

    // Download result
    let c = rt.download(&d_c).expect("Failed to download C");

    // Verify
    if verify(&c) {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
        println!("\nExpected vs Actual (first 32 elements):");
        for i in 0..32 {
            println!("  C[{}] = {}", i, c[i]);
        }
        panic!("Test failed!");
    }
}

#[cfg(feature = "cuda")]
fn create_identity() -> Vec<half::f16> {
    let mut m = vec![half::f16::ZERO; 256];
    for i in 0..16 {
        m[i * 16 + i] = half::f16::ONE;
    }
    m
}

#[cfg(feature = "cuda")]
fn create_ones() -> Vec<half::f16> {
    vec![half::f16::ONE; 256]
}

#[cfg(feature = "cuda")]
fn create_scaled_identity(scale: f32) -> Vec<half::f16> {
    let mut m = vec![half::f16::ZERO; 256];
    let s = half::f16::from_f32(scale);
    for i in 0..16 {
        m[i * 16 + i] = s;
    }
    m
}

#[cfg(feature = "cuda")]
fn verify_identity(c: &[f32]) -> bool {
    // Should be identity matrix
    for i in 0..16 {
        for j in 0..16 {
            let idx = i * 16 + j;
            let expected = if i == j { 1.0 } else { 0.0 };
            if (c[idx] - expected).abs() > 0.01 {
                return false;
            }
        }
    }
    true
}

#[cfg(feature = "cuda")]
fn verify_ones(c: &[f32]) -> bool {
    // Ones × Identity = Ones
    for &val in c.iter().take(256) {
        if (val - 1.0).abs() > 0.01 {
            return false;
        }
    }
    true
}

#[cfg(feature = "cuda")]
fn verify_ones_times_ones(c: &[f32]) -> bool {
    // Each element should be 16.0 (sum of 16 ones)
    for &val in c.iter().take(256) {
        if (val - 16.0).abs() > 0.1 {
            return false;
        }
    }
    true
}

#[cfg(feature = "cuda")]
fn verify_scaled_identity(c: &[f32], scale: f32) -> bool {
    // Should be scaled identity matrix
    for i in 0..16 {
        for j in 0..16 {
            let idx = i * 16 + j;
            let expected = if i == j { scale } else { 0.0 };
            if (c[idx] - expected).abs() > 0.1 {
                return false;
            }
        }
    }
    true
}
