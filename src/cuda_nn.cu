// EPIC 94: GPU Neural Network Forward Pass (standard CUDA FFI)
//
// Batched NN forward pass: 8 inputs -> 8 hidden (tanh) -> 5 outputs (argmax).
// One thread per organism. Exposed to Rust via the cuda_nn_forward C interface
// (see src/gpu_nn/cuda_nn.rs).
//
// NOTE: The Tensor Core (WMMA) implementation lives in src/gpu_nn/wmma_nn_v2.rs
// (real wmma::mma_sync with the transposed column-major B layout). This file is
// the plain scalar-per-organism reference path only.
//
// Compile with: nvcc -arch=sm_75 -c cuda_nn.cu -o cuda_nn.o

#include <cuda_runtime.h>
#include <cstdio>

// NN architecture
constexpr int INPUT_SIZE = 8;
constexpr int HIDDEN_SIZE = 8;
constexpr int OUTPUT_SIZE = 5;

// Fast tanh approximation (matches CPU implementation)
__device__ __forceinline__ float tanh_approx(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Kernel: Batched NN forward pass using standard CUDA (no WMMA for now)
// Input:  sensors [n_organisms × 8]
// Output: moves [n_organisms] (0-4: Up/Down/Left/Right/Stay)
// Weights: weights_ih [8×8], bias_h [8], weights_ho [8×5], bias_o [5]
__global__ void nn_forward_kernel(
    const float* __restrict__ sensors,      // [n × 8]
    const float* __restrict__ weights_ih,   // [8 × 8] (row-major)
    const float* __restrict__ bias_h,       // [8]
    const float* __restrict__ weights_ho,   // [8 × 5] (row-major)
    const float* __restrict__ bias_o,       // [5]
    int* __restrict__ moves,                // [n] output
    int n_organisms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_organisms) return;

    // Layer 1: Input → Hidden
    float hidden[HIDDEN_SIZE];
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        float sum = bias_h[h];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += sensors[tid * INPUT_SIZE + i] * weights_ih[h * INPUT_SIZE + i];
        }
        hidden[h] = tanh_approx(sum);
    }

    // Layer 2: Hidden → Output
    float output[OUTPUT_SIZE];
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        float sum = bias_o[o];
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            sum += hidden[h] * weights_ho[h * OUTPUT_SIZE + o];
        }
        output[o] = tanh_approx(sum);
    }

    // Argmax to get move index
    int max_idx = 0;
    float max_val = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }

    moves[tid] = max_idx;
}

// C interface for Rust FFI
extern "C" {

// Launch NN forward pass kernel (standard CUDA version)
void cuda_nn_forward(
    const float* sensors,
    const float* weights_ih,
    const float* bias_h,
    const float* weights_ho,
    const float* bias_o,
    int* moves,
    int n_organisms,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n_organisms + threads - 1) / threads;

    nn_forward_kernel<<<blocks, threads, 0, stream>>>(
        sensors, weights_ih, bias_h, weights_ho, bias_o, moves, n_organisms
    );
}

// Check for CUDA errors (callable from Rust)
int cuda_check_last_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

} // extern "C"
