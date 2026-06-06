//! GPU Neural Network implementations
//!
//! CUDA and WMMA-based neural network acceleration.

// cuda_nn requires separate .cu compilation, gated by cuda_nn_ffi feature
#[cfg(feature = "cuda_nn_ffi")]
pub mod cuda_nn;
#[cfg(feature = "cuda")]
pub mod cuda_nn_ptx;
#[cfg(feature = "cuda")]
pub mod gpu_gated_feedforward;
#[cfg(feature = "cuda")]
pub mod gpu_nn;
#[cfg(feature = "cuda")]
pub mod gpu_nn_persistent;
#[cfg(feature = "cuda")]
pub mod wmma_nn;
#[cfg(feature = "cuda")]
pub mod wmma_nn_v2;
