//! Phase 1: GPU Engine Prototype
//! Uses wgpu to run 2D logic simulation via Compute Shaders.

pub mod buffer_types;
pub mod gpu_simulation;
pub mod gpu_slim;
pub mod pipeline;

pub use pipeline::GpuSimulation;
