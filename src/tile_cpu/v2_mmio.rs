//! V2 memory-mapped I/O contract.
//!
//! Sprint 139 introduces a reserved MMIO window in the V2 64-cell address
//! space. Reads and writes inside that range are delegated to a device
//! callback instead of normal RAM semantics.

use std::fmt;
use std::rc::Rc;

/// Start of the reserved V2 MMIO address window (inclusive).
/// Sprint 155: expanded from 56 to 48 (16 addresses) for display device.
/// Sprint 157: expanded from 48 to 44 (20 addresses) for quantum bridge.
/// Sprint 159: expanded from 44 to 41 (23 addresses) for dataset device.
pub const V2_MMIO_BASE: u8 = 41;
/// Length of the reserved V2 MMIO window.
/// Sprint 155: expanded from 8 to 16 addresses.
/// Sprint 157: expanded from 16 to 20 addresses.
/// Sprint 159: expanded from 20 to 23 addresses.
pub const V2_MMIO_SIZE: u8 = 23;
/// End of the reserved V2 MMIO address window (inclusive).
pub const V2_MMIO_END: u8 = V2_MMIO_BASE + V2_MMIO_SIZE - 1;

/// MMIO callback interface for TileCpuV2.
pub trait V2MmioDevice {
    /// Read one MMIO address.
    fn read(&self, addr: u8) -> u64;
    /// Write one MMIO address.
    fn write(&self, addr: u8, value: u64);
    /// Optional per-cycle callback.
    fn tick(&self, _cycle: u64) {}
    /// Optional snapshot kind tag for deterministic replay bundles.
    fn snapshot_kind(&self) -> Option<&'static str> {
        None
    }
    /// Optional deterministic snapshot payload.
    fn snapshot_state(&self) -> Option<Vec<u8>> {
        None
    }
    /// Restore a deterministic snapshot payload.
    fn restore_state(&self, _snapshot: &[u8]) -> Result<(), String> {
        Err("snapshot restore unsupported".to_string())
    }
}

/// Cloneable MMIO device handle.
#[derive(Clone)]
pub struct V2MmioHandle {
    inner: Rc<dyn V2MmioDevice>,
}

impl V2MmioHandle {
    pub fn from_rc(inner: Rc<dyn V2MmioDevice>) -> Self {
        Self { inner }
    }

    pub fn new<T: V2MmioDevice + 'static>(device: T) -> Self {
        Self {
            inner: Rc::new(device),
        }
    }

    pub(crate) fn device(&self) -> &dyn V2MmioDevice {
        self.inner.as_ref()
    }

    pub fn snapshot_kind(&self) -> Option<&'static str> {
        self.inner.snapshot_kind()
    }

    pub fn snapshot_state(&self) -> Option<Vec<u8>> {
        self.inner.snapshot_state()
    }

    pub fn restore_state(&self, snapshot: &[u8]) -> Result<(), String> {
        self.inner.restore_state(snapshot)
    }
}

impl fmt::Debug for V2MmioHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("V2MmioHandle(..)")
    }
}

/// Return true when an address belongs to the V2 MMIO window.
///
/// Sprint 362: compare the full `usize` (not a truncated `u8`). With extended
/// main memory, addresses can exceed 255; the old `addr as u8` cast would alias
/// a high address such as 297 onto MMIO addr 41. The MMIO window is 41..=63, so
/// any addr >= 128 is unambiguously non-MMIO (main memory).
pub const fn is_v2_mmio_addr(addr: usize) -> bool {
    addr >= V2_MMIO_BASE as usize && addr <= V2_MMIO_END as usize
}
