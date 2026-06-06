use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimulationParams {
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub _padding: u32,
}
