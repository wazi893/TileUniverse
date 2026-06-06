#[cfg(feature = "burn-compute")]
mod bench {
    use burn::tensor::{Tensor, backend::Backend};
    use burn_cubecl::CubeBackend;
    use cubecl_wgpu::WgpuRuntime;

    pub fn run() {
        // Define the backend with WgpuRuntime, Float, Int, Bool (u8)
        type B = CubeBackend<WgpuRuntime, f32, i32, u8>;

        // Select the default device from the backend trait
        let device = <B as Backend>::Device::default();

        println!(
            "Initializing Burn with CubeCL backend (WGPU) on device: {:?}",
            device
        );

        // Create two tensors
        let tensor_1 = Tensor::<B, 2>::from_data([[2., 3.], [4., 5.]], &device);
        let tensor_2 = Tensor::<B, 2>::from_data([[7., 8.], [9., 10.]], &device);

        println!("Tensor 1:\n{}", tensor_1);
        println!("Tensor 2:\n{}", tensor_2);

        // Perform addition
        let sum = tensor_1 + tensor_2;

        println!("Sum (Tensor 1 + Tensor 2):\n{}", sum);
        println!("Burn + CubeCL integration successful!");
    }
}

fn main() {
    #[cfg(feature = "burn-compute")]
    bench::run();

    #[cfg(not(feature = "burn-compute"))]
    println!("Please run with --features burn-compute to enable this benchmark.");
}
