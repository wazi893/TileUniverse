// EPIC 53: Kernel Farm abstraction (bench/instrumentation only)
// Lightweight types to describe kernels and per-kernel stats without changing core tick semantics.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    Logic,
    Fields,
    Organism,
    Quantum,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KernelStats {
    pub logic_ops: u64,
    pub field_ops: u64,
    pub q_ops: u64,
    pub q_ops_flops: u64,
}

pub struct KernelCtx<'a> {
    pub sim: &'a mut crate::simulation::Simulation,
}

pub struct KernelRef {
    pub kind: KernelKind,
    pub name: &'static str,
    pub run: fn(&mut KernelCtx) -> KernelStats,
}

pub struct KernelFarm {
    pub logic: KernelRef,
    pub fields: KernelRef,
    pub organism: KernelRef,
    pub quantum: KernelRef,
}

impl KernelFarm {
    pub fn new(
        logic: KernelRef,
        fields: KernelRef,
        organism: KernelRef,
        quantum: KernelRef,
    ) -> Self {
        Self {
            logic,
            fields,
            organism,
            quantum,
        }
    }
}
