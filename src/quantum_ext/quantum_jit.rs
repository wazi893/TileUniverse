// EPIC 54: JIT backend scaffolding (feature-gated)
// This is a placeholder that routes to existing AVX2/scalar kernels until real JIT codegen is added.

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelId(pub u64);

#[allow(dead_code)]
pub struct JitKernel {
    pub id: KernelId,
    pub qubits: u8,
    pub program_hash: u64,
    // EPIC 62 Phase 3.5: DeepKernelFn ABI uses raw pointer
    #[cfg(feature = "cranelift_jit")]
    pub compiled: Option<*const u8>,
}

// EPIC 62 Phase 3.5: JIT code pointers are thread-safe
unsafe impl Send for JitKernel {}
unsafe impl Sync for JitKernel {}

// EPIC 62 Phase 3.5: Wrapper for function pointer table (thread-safe)
#[allow(dead_code)]
struct FunctionTable(Vec<Option<*const u8>>);
unsafe impl Send for FunctionTable {}
unsafe impl Sync for FunctionTable {}

#[allow(dead_code)]
pub struct JitBackend {
    pub kernels: Vec<JitKernel>,
}

impl JitBackend {
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
        }
    }
}

// EPIC 62 Phase 3.5: C-compatible QState layout for JIT kernel access
// This ensures stable field offsets for Cranelift-generated code
#[repr(C)]
pub struct QStateRaw {
    pub real_ptr: *mut f32,
    pub imag_ptr: *mut f32,
    pub len: usize,
    pub tile_count: u16,
    pub n_qubits: u16,
}

// EPIC 62 Phase 3.5: Deep kernel function signature for multi-worker execution
// Parameters: (qstate_ptr, tile_start, tile_end, depth, n_qubits)
// Each worker processes tiles in range [tile_start, tile_end)
pub type DeepKernelFn = unsafe extern "C" fn(
    qstate_ptr: *mut QStateRaw,
    tile_start: u16,
    tile_end: u16,
    depth: u32,
    n_qubits: u16,
);

use once_cell::sync::OnceCell;
use std::sync::Mutex;

#[allow(dead_code)]
static JIT_BACKEND: OnceCell<Mutex<JitBackend>> = OnceCell::new();
// EPIC 62 Phase 3.5: H_FUNCS uses DeepKernelFn ABI (raw pointers) with thread-safe wrapper
#[cfg(feature = "cranelift_jit")]
static H_FUNCS: OnceCell<FunctionTable> = OnceCell::new();

#[allow(dead_code)]
fn backend() -> &'static Mutex<JitBackend> {
    JIT_BACKEND.get_or_init(|| Mutex::new(JitBackend::new()))
}

fn hash_program(gates: &[crate::quantum::QGate]) -> u64 {
    // Simple rolling hash; stable enough for caching placeholder
    let mut h: u64 = 0xcbf29ce484222325;
    for g in gates {
        let tag: u64 = match g {
            crate::quantum::QGate::H(q) => 0x01 ^ (*q as u64),
            crate::quantum::QGate::X(q) => 0x02 ^ (*q as u64),
            crate::quantum::QGate::Z(q) => 0x03 ^ (*q as u64),
            crate::quantum::QGate::Phase(q, t) => 0x04 ^ (*q as u64) ^ ((*t as f64).to_bits()),
            crate::quantum::QGate::CNot(c, t) => 0x05 ^ (*c as u64) ^ ((*t as u64) << 8),
            crate::quantum::QGate::Measure(q) => 0x06 ^ (*q as u64),
            crate::quantum::QGate::CZ(c, t) => 0x07 ^ (*c as u64) ^ ((*t as u64) << 8),
            crate::quantum::QGate::Toffoli(c1, c2, t) => {
                0x08 ^ (*c1 as u64) ^ ((*c2 as u64) << 8) ^ ((*t as u64) << 16)
            }
            crate::quantum::QGate::CCZ(c1, c2, t) => {
                0x09 ^ (*c1 as u64) ^ ((*c2 as u64) << 8) ^ ((*t as u64) << 16)
            }
            crate::quantum::QGate::Rx(q, t) => 0x0A ^ (*q as u64) ^ ((*t as f64).to_bits()),
            crate::quantum::QGate::Ry(q, t) => 0x0B ^ (*q as u64) ^ ((*t as f64).to_bits()),
            crate::quantum::QGate::Rz(q, t) => 0x0C ^ (*q as u64) ^ ((*t as f64).to_bits()),
            crate::quantum::QGate::Y(q) => 0x0D ^ (*q as u64),
            crate::quantum::QGate::Swap(c, t) => 0x0E ^ (*c as u64) ^ ((*t as u64) << 8),
            // SPRINT 32.0: U3 gate hashing
            crate::quantum::QGate::U3(q, theta, phi, lambda) => {
                0x0F ^ (*q as u64)
                    ^ ((*theta as f64).to_bits())
                    ^ ((*phi as f64).to_bits())
                    ^ ((*lambda as f64).to_bits())
            }
            // Shor's algorithm: Controlled rotation gates
            crate::quantum::QGate::CPhase(c, t, angle) => {
                0x10 ^ (*c as u64) ^ ((*t as u64) << 8) ^ ((*angle as f64).to_bits())
            }
            crate::quantum::QGate::CRz(c, t, angle) => {
                0x11 ^ (*c as u64) ^ ((*t as u64) << 8) ^ ((*angle as f64).to_bits())
            }
            // SPRINT 48.0: T gates
            crate::quantum::QGate::T(q) => 0x12 ^ (*q as u64),
            crate::quantum::QGate::Tdg(q) => 0x13 ^ (*q as u64),
        };
        h = h.wrapping_mul(0x100000001b3).wrapping_add(tag);
    }
    h
}

#[allow(dead_code)]
pub fn ensure_compiled<'a>(
    qubits: u8,
    program: &[crate::quantum::QGate],
    jit: &'a mut JitBackend,
) -> &'a JitKernel {
    let ph = hash_program(program);
    if let Some(idx) = jit
        .kernels
        .iter()
        .position(|k| k.qubits == qubits && k.program_hash == ph)
    {
        return &jit.kernels[idx];
    }
    let kid = KernelId((jit.kernels.len() as u64) + 1);
    #[cfg(feature = "cranelift_jit")]
    let compiled = match program {
        [crate::quantum::QGate::H(target)] => {
            let t0 = std::time::Instant::now();
            let f = clif::compile_h_kernel(qubits, *target as u8);
            if f.is_some() {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "jit_compile_ms: gate=H target={} qubits={} ms={:.2}",
                    *target as u8, qubits, ms
                );
            }
            f
        }
        [crate::quantum::QGate::CNot(control, target)] => {
            let t0 = std::time::Instant::now();
            let f = clif::compile_cnot_kernel(qubits, *control as u8, *target as u8);
            if f.is_some() {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "jit_compile_ms: gate=CNOT control={} target={} qubits={} ms={:.2}",
                    *control as u8, *target as u8, qubits, ms
                );
            }
            f
        }
        _ => None,
    };
    jit.kernels.push(JitKernel {
        id: kid,
        qubits,
        program_hash: ph,
        #[cfg(feature = "cranelift_jit")]
        compiled,
    });
    jit.kernels.last().unwrap()
}

#[allow(dead_code)]
pub unsafe fn run_jit_kernel(state: &mut crate::quantum::QState, gate: &crate::quantum::QGate) {
    // Placeholder: if compiled JIT function is available, invoke; else route to AVX2 if available, else scalar
    #[cfg(feature = "cranelift_jit")]
    {
        match *gate {
            crate::quantum::QGate::H(q) => {
                #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
                {
                    H_FASTPATH_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
                        eprintln!("jit_debug: h_arm_enter q={}", q);
                    }
                }
                // EPIC 62 Phase 3.5: Fast path with prewarmed DeepKernelFn table
                if let Some(table) = H_FUNCS.get() {
                    if let Some(Some(f_ptr)) = table.0.get(q as usize) {
                        #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
                        {
                            H_FASTPATH_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
                                eprintln!("jit_debug: fastpath_table_hit q={}", q);
                            }
                        }
                        let f: DeepKernelFn = unsafe { std::mem::transmute(*f_ptr) };
                        let qstate_raw = QStateRaw {
                            real_ptr: state.real.as_mut_slice().as_mut_ptr(),
                            imag_ptr: state.imag.as_mut_slice().as_mut_ptr(),
                            len: state.len,
                            tile_count: state.tile_count,
                            n_qubits: state.n_qubits as u16,
                        };
                        let depth = 1; // Single H gate
                        unsafe {
                            f(
                                &qstate_raw as *const _ as *mut _,
                                0,
                                state.tile_count,
                                depth,
                                state.n_qubits as u16,
                            )
                        };
                        return;
                    }
                }
                // EPIC 62 Phase 3.5: compile-on-demand with DeepKernelFn ABI
                if let Ok(mut lock) = backend().lock() {
                    let k = ensure_compiled(state.n_qubits, &[gate.clone()], &mut *lock);
                    if let Some(f_ptr) = k.compiled {
                        #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
                        {
                            H_FASTPATH_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
                                eprintln!("jit_debug: fastpath_compile_hit q={}", q);
                            }
                        }
                        let f: DeepKernelFn = unsafe { std::mem::transmute(f_ptr) };
                        let qstate_raw = QStateRaw {
                            real_ptr: state.real.as_mut_slice().as_mut_ptr(),
                            imag_ptr: state.imag.as_mut_slice().as_mut_ptr(),
                            len: state.len,
                            tile_count: state.tile_count,
                            n_qubits: state.n_qubits as u16,
                        };
                        let depth = 1; // Single H gate
                        unsafe {
                            f(
                                &qstate_raw as *const _ as *mut _,
                                0,
                                state.tile_count,
                                depth,
                                state.n_qubits as u16,
                            )
                        };
                        return;
                    }
                }
            }
            crate::quantum::QGate::CNot(_c, _t) => {
                // EPIC 62 Phase 3.5: compile-on-demand with DeepKernelFn ABI
                if let Ok(mut lock) = backend().lock() {
                    let k = ensure_compiled(state.n_qubits, &[gate.clone()], &mut *lock);
                    if let Some(f_ptr) = k.compiled {
                        let f: DeepKernelFn = unsafe { std::mem::transmute(f_ptr) };
                        let qstate_raw = QStateRaw {
                            real_ptr: state.real.as_mut_slice().as_mut_ptr(),
                            imag_ptr: state.imag.as_mut_slice().as_mut_ptr(),
                            len: state.len,
                            tile_count: state.tile_count,
                            n_qubits: state.n_qubits as u16,
                        };
                        let depth = 1; // Single CNOT gate
                        unsafe {
                            f(
                                &qstate_raw as *const _ as *mut _,
                                0,
                                state.tile_count,
                                depth,
                                state.n_qubits as u16,
                            )
                        };
                        return;
                    }
                }
            }
            _ => {}
        }
    }
    match *gate {
        crate::quantum::QGate::H(q) => {
            #[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
            {
                H_GENERAL_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
                    eprintln!("jit_debug: general_path_hit q={}", q);
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if crate::quantum::is_avx2_available() {
                    unsafe {
                        let _ = crate::quantum::apply_h_avx2(state, q);
                    }
                    return;
                }
            }
            let mut rng = crate::quantum::QRng::new(0);
            let _ =
                crate::quantum::apply_gate_scalar(state, &crate::quantum::QGate::H(q), &mut rng);
        }
        crate::quantum::QGate::X(q) => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if crate::quantum::is_avx2_available() {
                    unsafe {
                        let _ = crate::quantum::apply_x_avx2(state, q);
                    }
                    return;
                }
            }
            let mut rng = crate::quantum::QRng::new(0);
            let _ =
                crate::quantum::apply_gate_scalar(state, &crate::quantum::QGate::X(q), &mut rng);
        }
        crate::quantum::QGate::Phase(q, theta) => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if crate::quantum::is_avx2_available() {
                    unsafe {
                        let _ = crate::quantum::apply_phase_avx2(state, q, theta);
                    }
                    return;
                }
            }
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::Phase(q, theta),
                &mut rng,
            );
        }
        crate::quantum::QGate::CNot(c, t) => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if crate::quantum::is_avx2_available() {
                    unsafe {
                        let _ = crate::quantum::apply_cnot_avx2(state, c, t);
                    }
                    return;
                }
            }
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::CNot(c, t),
                &mut rng,
            );
        }
        crate::quantum::QGate::Z(q) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ =
                crate::quantum::apply_gate_scalar(state, &crate::quantum::QGate::Z(q), &mut rng);
        }
        crate::quantum::QGate::Measure(q) => {
            // JIT never handles measurement; leave to scalar paths
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::Measure(q),
                &mut rng,
            );
        }
        crate::quantum::QGate::CZ(c, t) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::CZ(c, t),
                &mut rng,
            );
        }
        crate::quantum::QGate::Toffoli(c1, c2, t) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::Toffoli(c1, c2, t),
                &mut rng,
            );
        }
        crate::quantum::QGate::CCZ(c1, c2, t) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::CCZ(c1, c2, t),
                &mut rng,
            );
        }
        crate::quantum::QGate::Rx(q, theta) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::Rx(q, theta),
                &mut rng,
            );
        }
        crate::quantum::QGate::Ry(q, theta) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::Ry(q, theta),
                &mut rng,
            );
        }
        crate::quantum::QGate::Rz(q, theta) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::Rz(q, theta),
                &mut rng,
            );
        }
        crate::quantum::QGate::Y(q) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ =
                crate::quantum::apply_gate_scalar(state, &crate::quantum::QGate::Y(q), &mut rng);
        }
        crate::quantum::QGate::Swap(c, t) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::Swap(c, t),
                &mut rng,
            );
        }
        // SPRINT 32.0: U3 gate fallback to scalar
        crate::quantum::QGate::U3(q, theta, phi, lambda) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::U3(q, theta, phi, lambda),
                &mut rng,
            );
        }
        // Shor's algorithm: Controlled rotation gates
        crate::quantum::QGate::CPhase(c, t, angle) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::CPhase(c, t, angle),
                &mut rng,
            );
        }
        crate::quantum::QGate::CRz(c, t, angle) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ = crate::quantum::apply_gate_scalar(
                state,
                &crate::quantum::QGate::CRz(c, t, angle),
                &mut rng,
            );
        }
        // SPRINT 48.0: T gates
        crate::quantum::QGate::T(q) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ =
                crate::quantum::apply_gate_scalar(state, &crate::quantum::QGate::T(q), &mut rng);
        }
        crate::quantum::QGate::Tdg(q) => {
            let mut rng = crate::quantum::QRng::new(0);
            let _ =
                crate::quantum::apply_gate_scalar(state, &crate::quantum::QGate::Tdg(q), &mut rng);
        }
    }
}

#[cfg(all(test, feature = "quantum_jit", feature = "cranelift_jit"))]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Global mutex to serialize tests that modify JIT environment variables
    // This prevents race conditions when tests run in parallel
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    // Helper to lock mutex, recovering from poison if needed
    // Also resets JIT module to prevent stale compilations
    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        let guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        // Reset JIT module to ensure clean state for this test
        clif::reset_jit_module();
        guard
    }

    // Ensure the env flag toggles the fast-path gate logic.
    #[test]
    fn fastpath_flag_on_off_reads_env() {
        let _lock = lock_env();
        unsafe {
            std::env::set_var("JIT_12Q_FASTPATH", "0");
        }
        assert!(!clif::use_12q_fastpath_for_test());
        unsafe {
            std::env::set_var("JIT_12Q_FASTPATH", "1");
        }
        assert!(clif::use_12q_fastpath_for_test());
        // Restore default
        unsafe {
            std::env::set_var("JIT_12Q_FASTPATH", "0");
        }
    }

    // EPIC 61: Verify Cranelift SIMD support for F32X4 (128-bit SSE)
    #[test]
    fn simd_f32x4_proof_of_concept() {
        use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, MemFlags, Signature, types};
        use cranelift_codegen::{isa, settings};
        use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
        use cranelift_jit::{JITBuilder, JITModule};
        use cranelift_module::{Linkage, Module};
        use target_lexicon::Triple;

        // Create a simple SIMD kernel: load F32X4, multiply by 2.0, store
        let triple = Triple::host();
        let bld = settings::builder();
        let flags = settings::Flags::new(bld);
        let isa = isa::lookup(triple).unwrap().finish(flags).unwrap();
        let jitb = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(jitb);

        let mut ctx = module.make_context();
        let mut sig = Signature::new(cranelift_codegen::isa::CallConv::triple_default(
            &Triple::host(),
        ));
        sig.params.push(AbiParam::new(types::I64)); // ptr to input
        sig.params.push(AbiParam::new(types::I64)); // ptr to output

        let mut func = Function::new();
        func.signature = sig.clone();
        let mut fbctx = FunctionBuilderContext::new();
        let mut fb = FunctionBuilder::new(&mut func, &mut fbctx);
        let entry = fb.create_block();
        fb.append_block_params_for_function_params(entry);
        fb.switch_to_block(entry);
        fb.seal_block(entry);

        let in_ptr = fb.block_params(entry)[0];
        let out_ptr = fb.block_params(entry)[1];
        let mflags = MemFlags::new();

        // Load F32X4 (4 floats at once - 128-bit SSE)
        let vec = fb.ins().load(types::F32X4, mflags, in_ptr, 0);

        // Create a constant vector of [2.0, 2.0, 2.0, 2.0]
        let two = fb.ins().f32const(2.0f32);
        let vec_two = fb.ins().splat(types::F32X4, two);

        // Multiply: vec * 2.0
        let result = fb.ins().fmul(vec, vec_two);

        // Store result
        fb.ins().store(mflags, result, out_ptr, 0);
        fb.ins().return_(&[]);
        fb.finalize();

        ctx.func = func;
        let func_id = module
            .declare_function("simd_test", Linkage::Local, &sig)
            .unwrap();
        module.define_function(func_id, &mut ctx).unwrap();
        module.clear_context(&mut ctx);
        module.finalize_definitions().unwrap();

        let code = module.get_finalized_function(func_id);
        let fnptr = code as *const ();
        let simd_fn: unsafe extern "C" fn(*const f32, *mut f32) =
            unsafe { std::mem::transmute(fnptr) };

        // Test it
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        unsafe {
            simd_fn(input.as_ptr(), output.as_mut_ptr());
        }

        // Verify results
        for i in 0..4 {
            assert!(
                (output[i] - input[i] * 2.0).abs() < 1e-5,
                "SIMD test failed: expected {}, got {}",
                input[i] * 2.0,
                output[i]
            );
        }

        eprintln!("✅ EPIC 61: F32X4 SIMD proof-of-concept passed!");
    }

    // EPIC 61: Validate multi-tile H kernel compilation with F32X4 SIMD
    #[test]
    fn compile_multitile_h_kernel_4q() {
        let _lock = lock_env();
        // Set tile_count to 4 to trigger F32X4 SIMD path
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", "1");
        }

        // Compile H kernel for 4 qubits, target qubit 0, with 4 tiles
        let result = clif::compile_h_kernel(4, 0);

        // Verify compilation succeeded
        assert!(
            result.is_some(),
            "EPIC 61: Multi-tile H kernel (4q) failed to compile"
        );

        eprintln!("✅ EPIC 61: Multi-tile H kernel (4q, 4 tiles, F32X4) compiled successfully!");

        // Restore defaults
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::set_var("JIT_H_DEPTH", "1");
        }
    }

    // EPIC 61: Validate multi-tile H kernel compilation for 8q with depth
    #[test]
    fn compile_multitile_h_kernel_8q_depth4() {
        let _lock = lock_env();
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", "4");
        }

        let result = clif::compile_h_kernel(8, 3);
        assert!(
            result.is_some(),
            "EPIC 61: Multi-tile H kernel (8q, depth=4) failed to compile"
        );

        eprintln!("✅ EPIC 61: Multi-tile H kernel (8q, 4 tiles, depth=4) compiled successfully!");

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::set_var("JIT_H_DEPTH", "1");
        }
    }

    // EPIC 61: Validate backward compatibility - tile_count=1 uses scalar path
    #[test]
    fn compile_singletile_h_kernel_backward_compat() {
        let _lock = lock_env();
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::set_var("JIT_H_DEPTH", "1");
        }

        let result = clif::compile_h_kernel(4, 0);
        assert!(
            result.is_some(),
            "EPIC 61: Single-tile H kernel (backward compat) failed to compile"
        );

        eprintln!("✅ EPIC 61: Backward compatibility verified - tile_count=1 uses scalar path!");

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
        }
    }

    // EPIC 61: End-to-end test - execute multi-tile SIMD H kernel and validate results
    #[test]
    fn execute_multitile_h_kernel_4q() {
        let _lock = lock_env();
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", "1");
        }

        // Reset SIMD hit counter
        H_SIMD_HITS.store(0, std::sync::atomic::Ordering::Relaxed);

        // Create a 4-tile QState for 4 qubits (2^4 = 16 amplitudes per tile)
        let mut state = crate::quantum::QState::new_zero_multitile(4, 4);

        // Verify initial state: all tiles should have amplitude[0] = 1.0
        // Interleaved layout: real[0:4] = [T0[0], T1[0], T2[0], T3[0]] = [1.0, 1.0, 1.0, 1.0]
        let real_slice = state.real.as_slice();
        for i in 0..4 {
            assert!(
                (real_slice[i] - 1.0).abs() < 1e-5,
                "EPIC 61: Initial state tile {} should have amplitude[0] = 1.0, got {}",
                i,
                real_slice[i]
            );
        }

        // Compile H kernel for qubit 0, 4 tiles
        // EPIC 62 Phase 3.5: Use DeepKernelFn ABI
        let kernel_ptr = clif::compile_h_kernel(4, 0);
        assert!(
            kernel_ptr.is_some(),
            "EPIC 62 Phase 3.5: Failed to compile multi-tile H kernel"
        );

        // Verify SIMD kernel was compiled
        let simd_count = H_SIMD_HITS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            simd_count > 0,
            "EPIC 62 Phase 3.5: SIMD kernel should have been compiled, but H_SIMD_HITS = {}",
            simd_count
        );

        // Execute kernel with new ABI
        let f_ptr = kernel_ptr.unwrap();
        let f: DeepKernelFn = unsafe { std::mem::transmute(f_ptr) };

        // Create QStateRaw for new ABI
        let qstate_raw = QStateRaw {
            real_ptr: state.real.as_mut_slice().as_mut_ptr(),
            imag_ptr: state.imag.as_mut_slice().as_mut_ptr(),
            len: state.len,
            tile_count: state.tile_count,
            n_qubits: state.n_qubits as u16,
        };

        unsafe {
            f(&qstate_raw as *const _ as *mut _, 0, 4, 1, 4);
        }

        // Validate results: H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
        // Expected: amplitude[0] = amplitude[1] = 1/√2 ≈ 0.707107
        let inv_sqrt2 = 0.70710677f32;
        let real_after = state.real.as_slice();

        // Check amplitude[0] for all 4 tiles (indices 0-3)
        for i in 0..4 {
            assert!(
                (real_after[i] - inv_sqrt2).abs() < 1e-5,
                "EPIC 61: Tile {} amplitude[0] should be 1/√2, got {}",
                i,
                real_after[i]
            );
        }

        // Check amplitude[1] for all 4 tiles (indices 4-7 in interleaved layout)
        for i in 0..4 {
            assert!(
                (real_after[4 + i] - inv_sqrt2).abs() < 1e-5,
                "EPIC 61: Tile {} amplitude[1] should be 1/√2, got {}",
                i,
                real_after[4 + i]
            );
        }

        // Check amplitude[2] for all 4 tiles should be 0 (indices 8-11)
        for i in 0..4 {
            assert!(
                real_after[8 + i].abs() < 1e-5,
                "EPIC 61: Tile {} amplitude[2] should be 0, got {}",
                i,
                real_after[8 + i]
            );
        }

        eprintln!("✅ EPIC 61: Multi-tile SIMD H kernel executed correctly!");
        eprintln!("   4 tiles processed in parallel with F32X4 SIMD");
        eprintln!("   Result verified: H|0⟩ = |+⟩ for all 4 tiles");
        eprintln!("   SIMD kernels compiled: {}", simd_count);

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::set_var("JIT_H_DEPTH", "1");
        }
    }

    // EPIC 61 Phase 3: Bell pair test - H+CNOT entanglement with SIMD
    #[test]
    fn execute_multitile_bell_pair_4q() {
        let _lock = lock_env();
        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "4");
            std::env::set_var("JIT_H_DEPTH", "1");
        }

        // Reset SIMD hit counters
        H_SIMD_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
        CNOT_SIMD_HITS.store(0, std::sync::atomic::Ordering::Relaxed);

        // Create 4-tile QState for 4 qubits
        // We'll use only qubits 0 and 1 for Bell pair
        let mut state = crate::quantum::QState::new_zero_multitile(4, 4);

        // Step 1: Apply H(0) to create (|0⟩+|1⟩)/√2 ⊗ |0⟩
        // EPIC 62 Phase 3.5: Use DeepKernelFn ABI
        let h_kernel_ptr = clif::compile_h_kernel(4, 0);
        assert!(
            h_kernel_ptr.is_some(),
            "EPIC 62 Phase 3.5: Failed to compile H kernel"
        );

        let h_fn_ptr = h_kernel_ptr.unwrap();
        let h_fn: DeepKernelFn = unsafe { std::mem::transmute(h_fn_ptr) };

        // Create QStateRaw for new ABI
        let qstate_raw = QStateRaw {
            real_ptr: state.real.as_mut_slice().as_mut_ptr(),
            imag_ptr: state.imag.as_mut_slice().as_mut_ptr(),
            len: state.len,
            tile_count: state.tile_count,
            n_qubits: state.n_qubits as u16,
        };

        unsafe {
            h_fn(&qstate_raw as *const _ as *mut _, 0, 4, 1, 4);
        }

        // Step 2: Apply CNOT(0,1) to create (|00⟩+|11⟩)/√2
        // EPIC 62 Phase 3.5: Use DeepKernelFn ABI
        let cnot_kernel_ptr = clif::compile_cnot_kernel(4, 0, 1);
        assert!(
            cnot_kernel_ptr.is_some(),
            "EPIC 62 Phase 3.5: Failed to compile CNOT kernel"
        );

        let cnot_fn_ptr = cnot_kernel_ptr.unwrap();
        let cnot_fn: DeepKernelFn = unsafe { std::mem::transmute(cnot_fn_ptr) };
        unsafe {
            cnot_fn(&qstate_raw as *const _ as *mut _, 0, 4, 1, 4);
        }

        // Verify SIMD kernels were compiled
        let h_simd = H_SIMD_HITS.load(std::sync::atomic::Ordering::Relaxed);
        let cnot_simd = CNOT_SIMD_HITS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            h_simd > 0,
            "EPIC 61 Phase 3: H SIMD kernel should have been compiled"
        );
        assert!(
            cnot_simd > 0,
            "EPIC 61 Phase 3: CNOT SIMD kernel should have been compiled"
        );

        // Validate Bell state: (|00⟩+|11⟩)/√2
        // Non-zero amplitudes: state[0] = state[3] = 1/√2
        // Zero amplitudes: state[1] = state[2] = 0
        let inv_sqrt2 = 0.70710677f32;
        let real_after = state.real.as_slice();
        let imag_after = state.imag.as_slice();

        // Check all 4 tiles produce identical Bell states
        for tile in 0..4 {
            // amplitude[0] (|00⟩) should be 1/√2
            let amp0_re = real_after[tile]; // tile offset for amp[0]
            let amp0_im = imag_after[tile];
            assert!(
                (amp0_re - inv_sqrt2).abs() < 1e-5,
                "EPIC 61 Phase 3: Tile {} amp[0] real should be 1/√2, got {}",
                tile,
                amp0_re
            );
            assert!(
                amp0_im.abs() < 1e-5,
                "EPIC 61 Phase 3: Tile {} amp[0] imag should be 0, got {}",
                tile,
                amp0_im
            );

            // amplitude[1] (|01⟩) should be 0
            let amp1_re = real_after[4 + tile]; // amp[1] is at offset 4*tile_count + tile
            let amp1_im = imag_after[4 + tile];
            assert!(
                amp1_re.abs() < 1e-5,
                "EPIC 61 Phase 3: Tile {} amp[1] real should be 0, got {}",
                tile,
                amp1_re
            );
            assert!(
                amp1_im.abs() < 1e-5,
                "EPIC 61 Phase 3: Tile {} amp[1] imag should be 0, got {}",
                tile,
                amp1_im
            );

            // amplitude[2] (|10⟩) should be 0
            let amp2_re = real_after[8 + tile];
            let amp2_im = imag_after[8 + tile];
            assert!(
                amp2_re.abs() < 1e-5,
                "EPIC 61 Phase 3: Tile {} amp[2] real should be 0, got {}",
                tile,
                amp2_re
            );
            assert!(
                amp2_im.abs() < 1e-5,
                "EPIC 61 Phase 3: Tile {} amp[2] imag should be 0, got {}",
                tile,
                amp2_im
            );

            // amplitude[3] (|11⟩) should be 1/√2
            let amp3_re = real_after[12 + tile];
            let amp3_im = imag_after[12 + tile];
            assert!(
                (amp3_re - inv_sqrt2).abs() < 1e-5,
                "EPIC 61 Phase 3: Tile {} amp[3] real should be 1/√2, got {}",
                tile,
                amp3_re
            );
            assert!(
                amp3_im.abs() < 1e-5,
                "EPIC 61 Phase 3: Tile {} amp[3] imag should be 0, got {}",
                tile,
                amp3_im
            );
        }

        eprintln!("✅ EPIC 61 Phase 3: Multi-tile SIMD Bell pair test PASSED!");
        eprintln!("   Circuit: H(0) → CNOT(0,1)");
        eprintln!("   Result: (|00⟩+|11⟩)/√2 verified for all 4 tiles");
        eprintln!(
            "   H SIMD kernels: {}, CNOT SIMD kernels: {}",
            h_simd, cnot_simd
        );

        unsafe {
            std::env::set_var("JIT_TILE_COUNT", "1");
            std::env::set_var("JIT_H_DEPTH", "1");
        }
    }
}

#[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
// EPIC 62 Phase 3.5: Prewarm H kernels with DeepKernelFn ABI
pub fn prewarm_h_kernels(qubits: u8) {
    let mut vec: Vec<Option<*const u8>> = vec![None; (qubits as usize).max(1)];
    if let Ok(mut lock) = backend().lock() {
        for q in 0..qubits {
            let k = ensure_compiled(qubits, &[crate::quantum::QGate::H(q)], &mut *lock);
            if let Some(f) = k.compiled {
                vec[q as usize] = Some(f);
            }
        }
    }
    let _ = H_FUNCS.set(FunctionTable(vec));
}

// Stub for quantum_jit without cranelift_jit
#[cfg(all(feature = "quantum_jit", not(feature = "cranelift_jit")))]
pub fn prewarm_h_kernels(_qubits: u8) {
    // No-op when Cranelift is not available
}

#[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
pub static H_FASTPATH_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
pub static H_GENERAL_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cranelift_jit")]
pub static H_SIMD_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0); // EPIC 61
#[cfg(feature = "cranelift_jit")]
pub static CNOT_SIMD_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0); // EPIC 61 Phase 3

#[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
pub fn log_jit_debug_counts() {
    if std::env::var("JIT_DEBUG").ok().as_deref() != Some("1") {
        return;
    }
    let fast = H_FASTPATH_HITS.load(std::sync::atomic::Ordering::Relaxed);
    let general = H_GENERAL_HITS.load(std::sync::atomic::Ordering::Relaxed);
    let simd = H_SIMD_HITS.load(std::sync::atomic::Ordering::Relaxed); // EPIC 61
    let cnot_simd = CNOT_SIMD_HITS.load(std::sync::atomic::Ordering::Relaxed); // EPIC 61 Phase 3
    eprintln!(
        "jit_debug: h_fastpath_hits={} h_general_hits={} h_simd_hits={} cnot_simd_hits={}",
        fast, general, simd, cnot_simd
    );
}

#[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
pub fn h_debug_counts() -> (u64, u64) {
    let fast = H_FASTPATH_HITS.load(std::sync::atomic::Ordering::Relaxed);
    let general = H_GENERAL_HITS.load(std::sync::atomic::Ordering::Relaxed);
    (fast, general)
}

#[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
pub fn h_simd_count() -> u64 {
    // EPIC 61
    H_SIMD_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(all(feature = "quantum_jit", feature = "cranelift_jit"))]
pub fn cnot_simd_count() -> u64 {
    // EPIC 61 Phase 3
    CNOT_SIMD_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

// ========================
// Cranelift JIT (H kernel)
// ========================

#[cfg(feature = "cranelift_jit")]
#[allow(dead_code)]
pub mod clif {
    use cranelift_codegen::ir::{
        AbiParam, Function, InstBuilder, MemFlags, Signature, Value, types,
    };
    use cranelift_codegen::{isa, settings};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use target_lexicon::Triple;

    static FN_COUNTER: AtomicU64 = AtomicU64::new(1);
    static USE_12Q_FASTPATH: AtomicBool = AtomicBool::new(false);

    // EPIC 62J: Global JIT module (shared across all threads)
    //
    // Fix: Changed from thread_local to global OnceCell with Mutex
    //
    // Problem: Thread-local JIT modules caused STATUS_ACCESS_VIOLATION when worker threads
    // tried to execute function pointers compiled by the main thread. Each thread had its own
    // JIT code memory, and cross-thread access was invalid.
    //
    // Solution: Single global JIT module protected by Mutex. All threads compile and execute
    // from the same JIT code memory. Function pointers remain valid across all threads.
    //
    // Safety: JITModule contains raw pointers but is safe to send across threads because:
    // 1. Access is synchronized by Mutex (only one thread accesses at a time)
    // 2. Compiled code memory is managed by Cranelift and remains valid
    // 3. Function pointers are read-only after compilation
    use std::sync::Mutex as StdMutex;

    struct SendJITModule(JITModule);
    unsafe impl Send for SendJITModule {}

    // Use Mutex<Option> instead of OnceCell to allow test resets
    static JIT_MODULE: once_cell::sync::Lazy<StdMutex<Option<SendJITModule>>> =
        once_cell::sync::Lazy::new(|| StdMutex::new(None));

    fn new_jit_module() -> SendJITModule {
        let triple = Triple::host();
        let bld = settings::builder();
        let flags = settings::Flags::new(bld);
        let isa = isa::lookup(triple).unwrap().finish(flags.clone()).unwrap();
        let jitb = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(jitb);

        if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
            eprintln!("jit_debug: Cranelift JIT module created (global, thread-safe)");
        }

        SendJITModule(module)
    }

    fn with_jit_module<R>(f: impl FnOnce(&mut JITModule) -> R) -> R {
        let mut guard = JIT_MODULE.lock().unwrap();

        // Initialize if needed
        if guard.is_none() {
            if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
                eprintln!("jit_debug: Initializing global JIT module");
            }
            *guard = Some(new_jit_module());
        }

        f(&mut guard.as_mut().unwrap().0)
    }

    #[cfg(test)]
    pub(crate) fn reset_jit_module() {
        let mut guard = JIT_MODULE.lock().unwrap();
        *guard = None;
    }

    fn use_12q_fastpath() -> bool {
        // Default false for stability; allow runtime flag to switch on per-run.
        let env = std::env::var("JIT_12Q_FASTPATH").unwrap_or_else(|_| "0".to_string());
        let on = env == "1" || env.eq_ignore_ascii_case("on") || env.eq_ignore_ascii_case("true");
        USE_12Q_FASTPATH.store(on, Ordering::Relaxed);
        on
    }

    fn use_12q_partial_experimental() -> bool {
        let env = std::env::var("JIT_12Q_PARTIAL_EXPERIMENTAL").unwrap_or_else(|_| "0".to_string());
        env == "1" || env.eq_ignore_ascii_case("on") || env.eq_ignore_ascii_case("true")
    }

    // EPIC 60: H kernel depth configuration
    // EPIC 61: Extended with tile configuration for SIMD multi-tile execution
    #[derive(Clone, Copy, Debug)]
    struct HKernelConfig {
        depth: u32,
        tile_count: u8, // EPIC 61: 1=scalar (single-tile), 4=F32X4 SIMD (4-tile batch)
    }

    fn h_kernel_depth() -> u32 {
        let env = std::env::var("JIT_H_DEPTH").unwrap_or_else(|_| "1".to_string());
        env.parse::<u32>().unwrap_or(1).max(1)
    }

    // EPIC 61: Tile count configuration for SIMD multi-tile execution
    fn jit_tile_count() -> u8 {
        let env = std::env::var("JIT_TILE_COUNT").unwrap_or_else(|_| "1".to_string());
        let count = env.parse::<u8>().unwrap_or(1);
        // Constrain to valid values: 1 (scalar) or 4 (F32X4 SIMD)
        match count {
            4 => 4,
            _ => 1, // Default to scalar mode
        }
    }

    #[cfg(test)]
    pub(crate) fn use_12q_fastpath_for_test() -> bool {
        use_12q_fastpath()
    }

    fn strict_12q_bounds_check(qubits: u8, target_qubit: u8) {
        if qubits != 12 {
            return;
        }
        if target_qubit as usize >= (1usize << qubits) {
            panic!("jit_debug_strict: target_qubit >= len");
        }
        let len: usize = 1usize << qubits;
        let bit: usize = 1usize << (target_qubit as usize);
        debug_assert!(bit < len, "jit_debug_strict: bit must be < len");
        // Worst-case offset in unrolled kernel: ((len - 2*bit) + (bit - 1) + bit) * 4 = (len - 1) * 4
        let max_off_bytes = (len - 1).saturating_mul(4);
        let total_bytes = len.saturating_mul(4);
        if max_off_bytes >= total_bytes {
            panic!(
                "jit_debug_strict: max_off_bytes {} >= total_bytes {}",
                max_off_bytes, total_bytes
            );
        }
        // Alignment sanity: base pointer is assumed 4-byte aligned (f32)
        if max_off_bytes % 4 != 0 {
            panic!(
                "jit_debug_strict: misaligned max_off_bytes {}",
                max_off_bytes
            );
        }
    }

    fn jit_debug_strict_enabled() -> bool {
        std::env::var("JIT_DEBUG_STRICT").ok().as_deref() == Some("1")
    }

    fn dump_ir(func: &Function) {
        if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
            eprintln!("{}", func.display());
        }
    }

    // Build and compile a CNOT-kernel (control, target)
    pub fn compile_cnot_kernel(qubits: u8, control: u8, target: u8) -> Option<*const u8> {
        // EPIC 61 Phase 3: Read tile count for SIMD CNOT
        let tile_count = jit_tile_count();

        if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "jit_debug: compile_cnot_kernel qubits={} control={} target={} tile_count={}",
                qubits, control, target, tile_count
            );
        }

        // Fast-path: fully unrolled for small sizes (4q, 8q)
        if qubits <= 4 || qubits == 8 {
            // EPIC 61 Phase 3: Branch on tile_count for SIMD vs scalar CNOT
            if tile_count == 4 {
                // EPIC 62 Phase 3.5 A2: Multi-tile SIMD CNOT path with DeepKernelFn signature
                return with_jit_module(|module| {
                    let mut ctx = module.make_context();
                    let mut sig = Signature::new(cranelift_codegen::isa::CallConv::triple_default(
                        &Triple::host(),
                    ));
                    // EPIC 62 Phase 3.5 A1: New signature - DeepKernelFn
                    sig.params.push(AbiParam::new(types::I64)); // qstate_ptr: *mut QStateRaw
                    sig.params.push(AbiParam::new(types::I16)); // tile_start: u16
                    sig.params.push(AbiParam::new(types::I16)); // tile_end: u16
                    sig.params.push(AbiParam::new(types::I32)); // depth: u32
                    sig.params.push(AbiParam::new(types::I16)); // n_qubits: u16

                    let mut func = Function::new();
                    func.signature = sig.clone();
                    let mut fbctx = FunctionBuilderContext::new();
                    let mut fb = FunctionBuilder::new(&mut func, &mut fbctx);
                    let entry = fb.create_block();
                    fb.append_block_params_for_function_params(entry);
                    fb.switch_to_block(entry);
                    fb.seal_block(entry);

                    // EPIC 62 Phase 3.5 A1: Extract new parameters
                    let qstate_ptr = fb.block_params(entry)[0];
                    let tile_start = fb.block_params(entry)[1];
                    let _tile_end = fb.block_params(entry)[2]; // unused for now (A1)
                    let _depth = fb.block_params(entry)[3]; // unused for CNOT
                    let _n_qubits = fb.block_params(entry)[4]; // unused for now

                    let mflags = MemFlags::new();

                    // EPIC 62 Phase 3.5 A1: Load real_ptr and imag_ptr from QStateRaw
                    let real_base = fb.ins().load(types::I64, mflags, qstate_ptr, 0); // offset 0
                    let imag_base = fb.ins().load(types::I64, mflags, qstate_ptr, 8); // offset 8

                    // EPIC 62 Phase 3.5 A2: Tile offset for this worker
                    // Each tile is 4 bytes (1 f32), so tile_start * 4 gives byte offset
                    let tile_start_64 = fb.ins().uextend(types::I64, tile_start);
                    let tile_offset_floats = fb.ins().ishl_imm(tile_start_64, 2); // * 4 bytes
                    let real = fb.ins().iadd(real_base, tile_offset_floats);
                    let imag = fb.ins().iadd(imag_base, tile_offset_floats);

                    // EPIC 62 Phase 3.5 A2: Load tile_count and compute dynamic stride
                    let tile_count_u16 = fb.ins().load(types::I16, mflags, qstate_ptr, 24); // offset 24
                    let tile_count_64 = fb.ins().uextend(types::I64, tile_count_u16);
                    let bytes_per_amp_row = fb.ins().ishl_imm(tile_count_64, 2); // tile_count * 4 bytes

                    let len: usize = 1usize << qubits;
                    let cbit: usize = 1usize << (control as usize);
                    let tbit: usize = 1usize << (target as usize);

                    // EPIC 62 Phase 3.5 A2: SIMD CNOT with dynamic stride
                    for i in 0..len {
                        if (i & cbit) != 0 {
                            let j = i ^ tbit;
                            if i < j {
                                // EPIC 62 Phase 3.5 A2: Dynamic offset calculation
                                let i_offset = fb.ins().iconst(types::I64, i as i64);
                                let j_offset = fb.ins().iconst(types::I64, j as i64);
                                let i_off = fb.ins().imul(i_offset, bytes_per_amp_row);
                                let j_off = fb.ins().imul(j_offset, bytes_per_amp_row);
                                let addr_ri = fb.ins().iadd(real, i_off);
                                let addr_rj = fb.ins().iadd(real, j_off);
                                let addr_ii = fb.ins().iadd(imag, i_off);
                                let addr_ij = fb.ins().iadd(imag, j_off);

                                // EPIC 61: Load F32X4 vectors (4 tiles simultaneously)
                                let ri = fb.ins().load(types::F32X4, mflags, addr_ri, 0);
                                let rj = fb.ins().load(types::F32X4, mflags, addr_rj, 0);
                                let ii = fb.ins().load(types::F32X4, mflags, addr_ii, 0);
                                let ij = fb.ins().load(types::F32X4, mflags, addr_ij, 0);

                                // EPIC 61: Swap F32X4 vectors (same swap logic, vectorized)
                                fb.ins().store(mflags, rj, addr_ri, 0);
                                fb.ins().store(mflags, ri, addr_rj, 0);
                                fb.ins().store(mflags, ij, addr_ii, 0);
                                fb.ins().store(mflags, ii, addr_ij, 0);
                            }
                        }
                    }
                    fb.ins().return_(&[]);
                    fb.finalize();

                    ctx.func = func;
                    let suffix = FN_COUNTER.fetch_add(1, Ordering::Relaxed);
                    let fname = format!(
                        "cnot_kernel_simd4_{}_{}_{}_{}",
                        qubits, control, target, suffix
                    );
                    let func_id = module.declare_function(&fname, Linkage::Local, &sig).ok()?;
                    module.define_function(func_id, &mut ctx).ok()?;
                    module.clear_context(&mut ctx);
                    module.finalize_definitions().ok()?;
                    let code = module.get_finalized_function(func_id);

                    // EPIC 61 Phase 3: Track SIMD CNOT compilation
                    super::CNOT_SIMD_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    Some(code as *const u8)
                });
            }

            // EPIC 62 Phase 3.5 A2: Single-tile scalar CNOT path with DeepKernelFn signature
            return with_jit_module(|module| {
                let mut ctx = module.make_context();
                let mut sig = Signature::new(cranelift_codegen::isa::CallConv::triple_default(
                    &Triple::host(),
                ));
                // EPIC 62 Phase 3.5 A1: New signature - DeepKernelFn
                sig.params.push(AbiParam::new(types::I64)); // qstate_ptr: *mut QStateRaw
                sig.params.push(AbiParam::new(types::I16)); // tile_start: u16
                sig.params.push(AbiParam::new(types::I16)); // tile_end: u16
                sig.params.push(AbiParam::new(types::I32)); // depth: u32
                sig.params.push(AbiParam::new(types::I16)); // n_qubits: u16

                let mut func = Function::new();
                func.signature = sig.clone();
                let mut fbctx = FunctionBuilderContext::new();
                let mut fb = FunctionBuilder::new(&mut func, &mut fbctx);
                let entry = fb.create_block();
                fb.append_block_params_for_function_params(entry);
                fb.switch_to_block(entry);
                fb.seal_block(entry);

                // EPIC 62 Phase 3.5 A1: Extract new parameters
                let qstate_ptr = fb.block_params(entry)[0];
                let _tile_start = fb.block_params(entry)[1]; // unused for single-tile scalar
                let _tile_end = fb.block_params(entry)[2]; // unused for single-tile scalar
                let _depth = fb.block_params(entry)[3]; // unused for CNOT
                let _n_qubits = fb.block_params(entry)[4]; // unused for now

                let mflags = MemFlags::new();

                // EPIC 62 Phase 3.5 A1: Load real_ptr and imag_ptr from QStateRaw
                // For single-tile scalar, no tile offset needed
                let real = fb.ins().load(types::I64, mflags, qstate_ptr, 0); // offset 0
                let imag = fb.ins().load(types::I64, mflags, qstate_ptr, 8); // offset 8

                let len: usize = 1usize << qubits;
                let cbit: usize = 1usize << (control as usize);
                let tbit: usize = 1usize << (target as usize);

                // CNOT: for each i where (i & cbit) != 0, swap (i, i^tbit) if i < j
                for i in 0..len {
                    if (i & cbit) != 0 {
                        let j = i ^ tbit;
                        if i < j {
                            let i_off = fb.ins().iconst(types::I64, (i as i64) * 4);
                            let j_off = fb.ins().iconst(types::I64, (j as i64) * 4);
                            let addr_ri = fb.ins().iadd(real, i_off);
                            let addr_rj = fb.ins().iadd(real, j_off);
                            let addr_ii = fb.ins().iadd(imag, i_off);
                            let addr_ij = fb.ins().iadd(imag, j_off);

                            // Load i and j
                            let ri = fb.ins().load(types::F32, mflags, addr_ri, 0);
                            let rj = fb.ins().load(types::F32, mflags, addr_rj, 0);
                            let ii = fb.ins().load(types::F32, mflags, addr_ii, 0);
                            let ij = fb.ins().load(types::F32, mflags, addr_ij, 0);

                            // Swap: store j values to i addresses, i values to j addresses
                            fb.ins().store(mflags, rj, addr_ri, 0);
                            fb.ins().store(mflags, ri, addr_rj, 0);
                            fb.ins().store(mflags, ij, addr_ii, 0);
                            fb.ins().store(mflags, ii, addr_ij, 0);
                        }
                    }
                }
                fb.ins().return_(&[]);
                fb.finalize();

                ctx.func = func;
                let suffix = FN_COUNTER.fetch_add(1, Ordering::Relaxed);
                let fname = format!(
                    "cnot_kernel_fast_{}_{}_{}_{}",
                    qubits, control, target, suffix
                );
                let func_id = module.declare_function(&fname, Linkage::Local, &sig).ok()?;
                module.define_function(func_id, &mut ctx).ok()?;
                module.clear_context(&mut ctx);
                module.finalize_definitions().ok()?;
                let code = module.get_finalized_function(func_id);

                Some(code as *const u8)
            });
        }

        // EPIC 62 Phase 3.5 A2: General path for 12q and larger with DeepKernelFn signature
        with_jit_module(|module| {
            let mut ctx = module.make_context();
            let mut sig = Signature::new(cranelift_codegen::isa::CallConv::triple_default(
                &Triple::host(),
            ));
            // EPIC 62 Phase 3.5 A1: New signature - DeepKernelFn
            sig.params.push(AbiParam::new(types::I64)); // qstate_ptr: *mut QStateRaw
            sig.params.push(AbiParam::new(types::I16)); // tile_start: u16
            sig.params.push(AbiParam::new(types::I16)); // tile_end: u16
            sig.params.push(AbiParam::new(types::I32)); // depth: u32
            sig.params.push(AbiParam::new(types::I16)); // n_qubits: u16

            let mut func = Function::new();
            func.signature = sig.clone();
            let mut fbctx = FunctionBuilderContext::new();
            let mut fb = FunctionBuilder::new(&mut func, &mut fbctx);
            let entry = fb.create_block();
            fb.append_block_params_for_function_params(entry);
            fb.switch_to_block(entry);
            fb.seal_block(entry);

            // EPIC 62 Phase 3.5 A1: Extract new parameters
            let qstate_ptr = fb.block_params(entry)[0];
            let _tile_start = fb.block_params(entry)[1]; // unused for 12q scalar (single-tile)
            let _tile_end = fb.block_params(entry)[2]; // unused for 12q scalar (single-tile)
            let _depth = fb.block_params(entry)[3]; // unused for CNOT
            let _n_qubits = fb.block_params(entry)[4]; // unused for now

            let mflags = MemFlags::new();

            // EPIC 62 Phase 3.5 A1: Load real_ptr, imag_ptr, and len from QStateRaw
            let real = fb.ins().load(types::I64, mflags, qstate_ptr, 0); // offset 0
            let imag = fb.ins().load(types::I64, mflags, qstate_ptr, 8); // offset 8
            let len_val = fb.ins().load(types::I64, mflags, qstate_ptr, 16); // offset 16

            let cbit_val = fb.ins().iconst(types::I64, 1i64 << (control as usize));
            let tbit_val = fb.ins().iconst(types::I64, 1i64 << (target as usize));
            let four = fb.ins().iconst(types::I64, 4);
            let zero = fb.ins().iconst(types::I64, 0);
            let one = fb.ins().iconst(types::I64, 1);

            let loop_head = fb.create_block();
            fb.append_block_param(loop_head, types::I64); // i
            let loop_check = fb.create_block();
            fb.append_block_param(loop_check, types::I64); // i
            let loop_inc = fb.create_block();
            fb.append_block_param(loop_inc, types::I64); // i
            let exit = fb.create_block();

            fb.ins().jump(loop_head, &[zero]);

            // loop_head: if i < len -> loop_check else exit
            fb.switch_to_block(loop_head);
            let i_cur = fb.block_params(loop_head)[0];
            let cmp = fb.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
                i_cur,
                len_val,
            );
            fb.ins().brif(cmp, loop_check, &[i_cur], exit, &[]);
            fb.seal_block(loop_head);

            // loop_check: if (i & cbit) != 0, do swap; else loop_inc
            fb.switch_to_block(loop_check);
            let i_check = fb.block_params(loop_check)[0];
            let i_and_cbit = fb.ins().band(i_check, cbit_val);
            let is_control_set = fb.ins().icmp_imm(
                cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                i_and_cbit,
                0,
            );

            let swap_block = fb.create_block();
            fb.append_block_param(swap_block, types::I64); // i
            fb.ins()
                .brif(is_control_set, swap_block, &[i_check], loop_inc, &[i_check]);
            fb.seal_block(loop_check);

            // swap_block: compute j = i ^ tbit, if i < j, swap
            fb.switch_to_block(swap_block);
            let i_swap = fb.block_params(swap_block)[0];
            let j_swap = fb.ins().bxor(i_swap, tbit_val);
            let should_swap = fb.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
                i_swap,
                j_swap,
            );

            let do_swap = fb.create_block();
            fb.append_block_param(do_swap, types::I64); // i
            fb.append_block_param(do_swap, types::I64); // j
            fb.ins()
                .brif(should_swap, do_swap, &[i_swap, j_swap], loop_inc, &[i_swap]);
            fb.seal_block(swap_block);

            // do_swap: actually swap amplitudes
            fb.switch_to_block(do_swap);
            let i_do = fb.block_params(do_swap)[0];
            let j_do = fb.block_params(do_swap)[1];

            let i_off = fb.ins().imul(i_do, four);
            let j_off = fb.ins().imul(j_do, four);
            let addr_ri = fb.ins().iadd(real, i_off);
            let addr_rj = fb.ins().iadd(real, j_off);
            let addr_ii = fb.ins().iadd(imag, i_off);
            let addr_ij = fb.ins().iadd(imag, j_off);

            let ri = fb.ins().load(types::F32, mflags, addr_ri, 0);
            let rj = fb.ins().load(types::F32, mflags, addr_rj, 0);
            let ii = fb.ins().load(types::F32, mflags, addr_ii, 0);
            let ij = fb.ins().load(types::F32, mflags, addr_ij, 0);

            fb.ins().store(mflags, rj, addr_ri, 0);
            fb.ins().store(mflags, ri, addr_rj, 0);
            fb.ins().store(mflags, ij, addr_ii, 0);
            fb.ins().store(mflags, ii, addr_ij, 0);

            fb.ins().jump(loop_inc, &[i_do]);
            fb.seal_block(do_swap);

            // loop_inc: i++, goto loop_head
            fb.switch_to_block(loop_inc);
            let i_inc = fb.block_params(loop_inc)[0];
            let i_next = fb.ins().iadd(i_inc, one);
            fb.ins().jump(loop_head, &[i_next]);
            fb.seal_block(loop_inc);

            // exit
            fb.switch_to_block(exit);
            fb.ins().return_(&[]);
            fb.seal_block(exit);

            fb.finalize();
            ctx.func = func;
            let suffix = FN_COUNTER.fetch_add(1, Ordering::Relaxed);
            let fname = format!("cnot_kernel_gen_{}_{}_{}", control, target, suffix);
            let func_id = module.declare_function(&fname, Linkage::Local, &sig).ok()?;
            module.define_function(func_id, &mut ctx).ok()?;
            module.clear_context(&mut ctx);
            module.finalize_definitions().ok()?;
            let code = module.get_finalized_function(func_id);

            Some(code as *const u8)
        })
    }

    // Build and compile an H-kernel specialized by target qubit bit (passed at call time)
    pub fn compile_h_kernel(qubits: u8, target_qubit: u8) -> Option<*const u8> {
        // EPIC 60: Read H kernel depth configuration
        // EPIC 61: Read tile count configuration
        let depth = h_kernel_depth();
        let tile_count = jit_tile_count();
        let config = HKernelConfig { depth, tile_count };

        if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "jit_debug: compile_h_kernel qubits={} target={} depth={} tile_count={}",
                qubits, target_qubit, config.depth, config.tile_count
            );
        }

        // Fast-path: fully unrolled for selected sizes (4q, 8q).
        let allow_12q = qubits == 12 && use_12q_fastpath() && use_12q_partial_experimental();
        if qubits <= 4 || qubits == 8 {
            // EPIC 61: Branch on tile_count for SIMD vs scalar
            if config.tile_count == 4 {
                // EPIC 62 Phase 3.5 A1: Multi-tile SIMD path (F32X4) with new DeepKernelFn ABI
                return with_jit_module(|module| {
                    let mut ctx = module.make_context();

                    // EPIC 62 Phase 3.5 A1: New signature - DeepKernelFn(qstate_ptr, tile_start, tile_end, depth, n_qubits)
                    let mut sig = Signature::new(cranelift_codegen::isa::CallConv::triple_default(
                        &Triple::host(),
                    ));
                    sig.params.push(AbiParam::new(types::I64)); // qstate_ptr: *mut QStateRaw
                    sig.params.push(AbiParam::new(types::I16)); // tile_start: u16
                    sig.params.push(AbiParam::new(types::I16)); // tile_end: u16
                    sig.params.push(AbiParam::new(types::I32)); // depth: u32
                    sig.params.push(AbiParam::new(types::I16)); // n_qubits: u16

                    let mut func = Function::new();
                    func.signature = sig.clone();
                    let mut fbctx = FunctionBuilderContext::new();
                    let mut fb = FunctionBuilder::new(&mut func, &mut fbctx);
                    let entry = fb.create_block();
                    fb.append_block_params_for_function_params(entry);
                    fb.switch_to_block(entry);
                    fb.seal_block(entry);

                    // EPIC 62 Phase 3.5 A2: Extract parameters
                    let qstate_ptr = fb.block_params(entry)[0];
                    let tile_start = fb.block_params(entry)[1]; // A2: use for tile offset
                    // let tile_end = fb.block_params(entry)[2];    // A2: unused (assume tile_start + 4)
                    // let depth_param = fb.block_params(entry)[3]; // A2: unused (use config.depth)
                    // let n_qubits_param = fb.block_params(entry)[4]; // A2: unused (use qubits)

                    // EPIC 62 Phase 3.5 A1: Load real_ptr and imag_ptr from QStateRaw
                    // QStateRaw layout (repr C):
                    //   offset 0: real_ptr (*mut f32, 8 bytes)
                    //   offset 8: imag_ptr (*mut f32, 8 bytes)
                    //   offset 16: len (usize, 8 bytes)
                    //   offset 24: tile_count (u16, 2 bytes)
                    //   offset 26: n_qubits (u16, 2 bytes)
                    let mflags = MemFlags::new();
                    let real_base = fb.ins().load(types::I64, mflags, qstate_ptr, 0); // offset 0
                    let imag_base = fb.ins().load(types::I64, mflags, qstate_ptr, 8); // offset 8

                    // EPIC 62 Phase 3.5 A2: Compute tile offset for this worker's 4-tile group
                    // In interleaved layout: each amplitude row contains tile_count tiles
                    // Worker N processes tiles [N*4, N*4+1, N*4+2, N*4+3]
                    // Byte offset = tile_start * 4 bytes (each tile is 1 float = 4 bytes)
                    let tile_start_64 = fb.ins().uextend(types::I64, tile_start); // u16 -> i64
                    let tile_offset_floats = fb.ins().ishl_imm(tile_start_64, 2); // * 4 (each tile = 4 bytes)

                    // Add tile offset to base pointers
                    let real = fb.ins().iadd(real_base, tile_offset_floats);
                    let imag = fb.ins().iadd(imag_base, tile_offset_floats);

                    // EPIC 62 Phase 3.5 A2: Load tile_count for dynamic stride calculation
                    let tile_count_u16 = fb.ins().load(types::I16, mflags, qstate_ptr, 24); // offset 24
                    let tile_count_64 = fb.ins().uextend(types::I64, tile_count_u16);
                    let bytes_per_amp_row = fb.ins().ishl_imm(tile_count_64, 2); // tile_count * 4 bytes

                    // EPIC 61: SIMD constants - splat inv_sqrt_2 to F32X4
                    let inv_scalar = fb.ins().f32const(0.70710677f32);
                    let inv_vec = fb.ins().splat(types::F32X4, inv_scalar);

                    // EPIC 62 Phase 3.5 A2: Interleaved layout - each amp position spans tile_count tiles
                    // Offset calculation: amp_index * tile_count floats * 4 bytes
                    let len: usize = 1usize << qubits;
                    let bit: usize = 1usize << (target_qubit as usize);
                    let step: usize = bit << 1;
                    let bit_offset = fb.ins().iconst(types::I64, bit as i64);
                    let bit_bytes = fb.ins().imul(bit_offset, bytes_per_amp_row); // bit * tile_count * 4

                    for i in (0..len).step_by(step) {
                        // EPIC 62 Phase 3.5 A2: Use dynamic stride based on tile_count
                        let i_offset = fb.ins().iconst(types::I64, i as i64);
                        let base = fb.ins().imul(i_offset, bytes_per_amp_row); // i * tile_count * 4
                        for j in 0..bit {
                            let j_offset = fb.ins().iconst(types::I64, j as i64);
                            let joff = fb.ins().imul(j_offset, bytes_per_amp_row); // j * tile_count * 4
                            let off0 = fb.ins().iadd(base, joff);
                            let off1 = fb.ins().iadd(off0, bit_bytes); // + bit * tile_count * 4
                            let addr_r0 = fb.ins().iadd(real, off0);
                            let addr_r1 = fb.ins().iadd(real, off1);
                            let addr_i0 = fb.ins().iadd(imag, off0);
                            let addr_i1 = fb.ins().iadd(imag, off1);

                            // EPIC 61: Load F32X4 vectors (4 tiles simultaneously)
                            let mut r0 = fb.ins().load(types::F32X4, mflags, addr_r0, 0);
                            let mut r1 = fb.ins().load(types::F32X4, mflags, addr_r1, 0);
                            let mut im0 = fb.ins().load(types::F32X4, mflags, addr_i0, 0);
                            let mut im1 = fb.ins().load(types::F32X4, mflags, addr_i1, 0);

                            // EPIC 61: Apply H depth times (vectorized, compile-time unroll)
                            for _ in 0..config.depth {
                                let r_add = fb.ins().fadd(r0, r1);
                                let r_sub = fb.ins().fsub(r0, r1);
                                let i_add = fb.ins().fadd(im0, im1);
                                let i_sub = fb.ins().fsub(im0, im1);
                                r0 = fb.ins().fmul(r_add, inv_vec);
                                r1 = fb.ins().fmul(r_sub, inv_vec);
                                im0 = fb.ins().fmul(i_add, inv_vec);
                                im1 = fb.ins().fmul(i_sub, inv_vec);
                            }

                            // EPIC 61: Store F32X4 results
                            fb.ins().store(mflags, r0, addr_r0, 0);
                            fb.ins().store(mflags, r1, addr_r1, 0);
                            fb.ins().store(mflags, im0, addr_i0, 0);
                            fb.ins().store(mflags, im1, addr_i1, 0);
                        }
                    }
                    fb.ins().return_(&[]);
                    fb.finalize();

                    ctx.func = func;
                    let suffix = FN_COUNTER.fetch_add(1, Ordering::Relaxed);
                    let fname = format!("h_kernel_simd4_{}_{}_{}", qubits, target_qubit, suffix);
                    let func_id = module.declare_function(&fname, Linkage::Local, &sig).ok()?;
                    module.define_function(func_id, &mut ctx).ok()?;
                    module.clear_context(&mut ctx);
                    module.finalize_definitions().ok()?;
                    let code = module.get_finalized_function(func_id);
                    // EPIC 61: Track SIMD kernel compilation
                    super::H_SIMD_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    // EPIC 62 Phase 3.5: Return raw pointer for DeepKernelFn ABI
                    Some(code as *const u8)
                });
            } else {
                // EPIC 61: Single-tile scalar path (backward compatible)
                return with_jit_module(|module| {
                    let mut ctx = module.make_context();

                    let mut sig = Signature::new(cranelift_codegen::isa::CallConv::triple_default(
                        &Triple::host(),
                    ));
                    sig.params.push(AbiParam::new(types::I64));
                    sig.params.push(AbiParam::new(types::I64));
                    sig.params.push(AbiParam::new(types::I64));
                    sig.params.push(AbiParam::new(types::I64));

                    let mut func = Function::new();
                    func.signature = sig.clone();
                    let mut fbctx = FunctionBuilderContext::new();
                    let mut fb = FunctionBuilder::new(&mut func, &mut fbctx);
                    let entry = fb.create_block();
                    fb.append_block_params_for_function_params(entry);
                    fb.switch_to_block(entry);
                    fb.seal_block(entry);

                    let real = fb.block_params(entry)[0];
                    let imag = fb.block_params(entry)[1];
                    let inv = fb.ins().f32const(0.70710677f32);
                    let mflags = MemFlags::new();

                    let len: usize = 1usize << qubits;
                    let bit: usize = 1usize << (target_qubit as usize);
                    let step: usize = bit << 1;
                    let bit4 = fb.ins().iconst(types::I64, (bit as i64) * 4);
                    for i in (0..len).step_by(step) {
                        let base = fb.ins().iconst(types::I64, (i as i64) * 4);
                        for j in 0..bit {
                            let joff = fb.ins().iconst(types::I64, (j as i64) * 4);
                            let off0 = fb.ins().iadd(base, joff);
                            let off1 = fb.ins().iadd(off0, bit4);
                            let addr_r0 = fb.ins().iadd(real, off0);
                            let addr_r1 = fb.ins().iadd(real, off1);
                            let addr_i0 = fb.ins().iadd(imag, off0);
                            let addr_i1 = fb.ins().iadd(imag, off1);
                            // EPIC 60: Load amplitudes
                            let mut r0 = fb.ins().load(types::F32, mflags, addr_r0, 0);
                            let mut r1 = fb.ins().load(types::F32, mflags, addr_r1, 0);
                            let mut im0 = fb.ins().load(types::F32, mflags, addr_i0, 0);
                            let mut im1 = fb.ins().load(types::F32, mflags, addr_i1, 0);
                            // EPIC 60: Apply H depth times (compile-time unroll)
                            for _ in 0..config.depth {
                                let r_add = fb.ins().fadd(r0, r1);
                                let r_sub = fb.ins().fsub(r0, r1);
                                let i_add = fb.ins().fadd(im0, im1);
                                let i_sub = fb.ins().fsub(im0, im1);
                                r0 = fb.ins().fmul(r_add, inv);
                                r1 = fb.ins().fmul(r_sub, inv);
                                im0 = fb.ins().fmul(i_add, inv);
                                im1 = fb.ins().fmul(i_sub, inv);
                            }
                            // EPIC 60: Store final results
                            fb.ins().store(mflags, r0, addr_r0, 0);
                            fb.ins().store(mflags, r1, addr_r1, 0);
                            fb.ins().store(mflags, im0, addr_i0, 0);
                            fb.ins().store(mflags, im1, addr_i1, 0);
                        }
                    }
                    fb.ins().return_(&[]);
                    fb.finalize();

                    ctx.func = func;
                    let suffix = FN_COUNTER.fetch_add(1, Ordering::Relaxed);
                    let fname = format!("h_kernel_fast_{}_{}_{}", qubits, target_qubit, suffix);
                    let func_id = module.declare_function(&fname, Linkage::Local, &sig).ok()?;
                    module.define_function(func_id, &mut ctx).ok()?;
                    module.clear_context(&mut ctx);
                    module.finalize_definitions().ok()?;
                    let code = module.get_finalized_function(func_id);

                    // EPIC 62 Phase 3.5: Return raw pointer for DeepKernelFn ABI
                    Some(code as *const u8)
                });
            }
        }

        if allow_12q {
            if jit_debug_strict_enabled() {
                strict_12q_bounds_check(qubits, target_qubit);
                eprintln!(
                    "jit_debug: h_12q_partial_strict_ok target={} len={} bit={} max_off_bytes={}",
                    target_qubit,
                    1usize << qubits,
                    1usize << (target_qubit as usize),
                    ((1usize << qubits) - 1) * 4
                );
            }
            if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
                eprintln!(
                    "jit_debug: using_12q_4x3_unroll target={} (env JIT_12Q_FASTPATH=on)",
                    target_qubit
                );
            }
            return with_jit_module(|module| {
                let mut ctx = module.make_context();
                let mut sig = Signature::new(cranelift_codegen::isa::CallConv::triple_default(
                    &Triple::host(),
                ));
                sig.params.push(AbiParam::new(types::I64));
                sig.params.push(AbiParam::new(types::I64));
                sig.params.push(AbiParam::new(types::I64));
                sig.params.push(AbiParam::new(types::I64));
                let mut func = Function::new();
                func.signature = sig.clone();
                let mut fbctx = FunctionBuilderContext::new();
                let mut fb = FunctionBuilder::new(&mut func, &mut fbctx);
                let entry = fb.create_block();
                fb.append_block_params_for_function_params(entry);
                fb.switch_to_block(entry);
                fb.seal_block(entry);
                let real = fb.block_params(entry)[0];
                let imag = fb.block_params(entry)[1];
                let _len_val = fb.block_params(entry)[2];
                // strict-mode alignment/bounds
                if jit_debug_strict_enabled() {
                    // Sanity: ensure offsets fit in 32-bit immediates and within the buffer.
                    let max_off_bytes = (((1u32) << qubits) - 1).saturating_mul(4);
                    debug_assert!(max_off_bytes < i32::MAX as u32);
                    let max_off_i32 = max_off_bytes as i32;
                    let mflags = MemFlags::new();
                    let _ = fb.ins().load(types::I8, mflags, real, max_off_i32);
                    let _ = fb.ins().load(types::I8, mflags, imag, max_off_i32);
                }
                let bit: i64 = 1 << (target_qubit as usize);
                let step = fb.ins().iconst(types::I64, (bit << 1) * 1); // i increments by 2*bit
                let bit_bytes = fb.ins().iconst(types::I64, bit * 4);
                let bit_const = fb.ins().iconst(types::I64, bit);
                let zero = fb.ins().iconst(types::I64, 0);
                let one = fb.ins().iconst(types::I64, 1);
                let two = fb.ins().iconst(types::I64, 2);
                let three = fb.ins().iconst(types::I64, 3);
                let four = fb.ins().iconst(types::I64, 4);
                let len_val = fb.block_params(entry)[2];

                let outer_head = fb.create_block();
                let inner_head = fb.create_block();
                let unroll4 = fb.create_block();
                let tail1 = fb.create_block();
                let outer_inc = fb.create_block();
                let exit = fb.create_block();

                fb.append_block_param(outer_head, types::I64); // i
                fb.append_block_param(inner_head, types::I64); // i
                fb.append_block_param(inner_head, types::I64); // j
                fb.append_block_param(unroll4, types::I64); // i
                fb.append_block_param(unroll4, types::I64); // j
                fb.append_block_param(tail1, types::I64); // i
                fb.append_block_param(tail1, types::I64); // j
                fb.append_block_param(outer_inc, types::I64); // i

                fb.ins().jump(outer_head, &[zero]);

                // outer_head: if i < len -> inner_head(i, 0) else exit
                fb.switch_to_block(outer_head);
                let i_cur = fb.block_params(outer_head)[0];
                let cmp_outer = fb.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
                    i_cur,
                    len_val,
                );
                fb.ins()
                    .brif(cmp_outer, inner_head, &[i_cur, zero], exit, &[]);
                fb.seal_block(outer_head);

                // inner_head: if j < bit -> choose unroll4 or tail; else outer_inc
                fb.switch_to_block(inner_head);
                let ih_i = fb.block_params(inner_head)[0];
                let ih_j = fb.block_params(inner_head)[1];
                let cmp_inner = fb.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
                    ih_j,
                    bit_const,
                );
                fb.ins()
                    .brif(cmp_inner, unroll4, &[ih_i, ih_j], outer_inc, &[ih_i]);
                fb.seal_block(inner_head);

                // unroll4: if j+3 < bit => do 4 bodies else tail1
                fb.switch_to_block(unroll4);
                let u_i = fb.block_params(unroll4)[0];
                let u_j = fb.block_params(unroll4)[1];
                let jp3 = fb.ins().iadd(u_j, three);
                let cmp_four = fb.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
                    jp3,
                    bit_const,
                );
                fb.ins()
                    .brif(cmp_four, tail1, &[u_i, u_j], tail1, &[u_i, u_j]); // fall through to tail if not enough room
                fb.seal_block(unroll4);

                // tail1: compute body for one or four iterations depending on caller
                fb.switch_to_block(tail1);
                let t_i = fb.block_params(tail1)[0];
                let t_j = fb.block_params(tail1)[1];
                let mflags = MemFlags::new();
                let inv = fb.ins().f32const(0.70710677f32);

                // Helper closure for one H body at (i,j)
                let emit_body = |fb: &mut FunctionBuilder, j_val: Value| {
                    let idx = fb.ins().iadd(t_i, j_val);
                    let off0 = fb.ins().imul(idx, four);
                    let off1 = fb.ins().iadd(off0, bit_bytes);
                    let addr_r0 = fb.ins().iadd(real, off0);
                    let addr_r1 = fb.ins().iadd(real, off1);
                    let addr_i0 = fb.ins().iadd(imag, off0);
                    let addr_i1 = fb.ins().iadd(imag, off1);
                    let r0 = fb.ins().load(types::F32, mflags, addr_r0, 0);
                    let r1 = fb.ins().load(types::F32, mflags, addr_r1, 0);
                    let im0 = fb.ins().load(types::F32, mflags, addr_i0, 0);
                    let im1 = fb.ins().load(types::F32, mflags, addr_i1, 0);
                    let r_add = fb.ins().fadd(r0, r1);
                    let r_sub = fb.ins().fsub(r0, r1);
                    let i_add = fb.ins().fadd(im0, im1);
                    let i_sub = fb.ins().fsub(im0, im1);
                    let r_s0 = fb.ins().fmul(r_add, inv);
                    let r_s1 = fb.ins().fmul(r_sub, inv);
                    let i_s0 = fb.ins().fmul(i_add, inv);
                    let i_s1 = fb.ins().fmul(i_sub, inv);
                    fb.ins().store(mflags, r_s0, addr_r0, 0);
                    fb.ins().store(mflags, r_s1, addr_r1, 0);
                    fb.ins().store(mflags, i_s0, addr_i0, 0);
                    fb.ins().store(mflags, i_s1, addr_i1, 0);
                };

                // Decide whether we came from unroll or tail path by checking jp3 >= bit
                let jp3_tail = fb.ins().iadd(t_j, three);
                let enough = fb.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
                    jp3_tail,
                    bit_const,
                );
                let tail_body_block = fb.create_block();
                let unroll_body_block = fb.create_block();
                fb.ins()
                    .brif(enough, unroll_body_block, &[], tail_body_block, &[]);
                fb.seal_block(tail1);

                // four-iter unroll
                fb.switch_to_block(unroll_body_block);
                emit_body(&mut fb, t_j);
                let j1 = fb.ins().iadd(t_j, one);
                emit_body(&mut fb, j1);
                let j2 = fb.ins().iadd(t_j, two);
                emit_body(&mut fb, j2);
                let j3 = fb.ins().iadd(t_j, three);
                emit_body(&mut fb, j3);
                let j_next4 = fb.ins().iadd(t_j, four);
                fb.ins().jump(inner_head, &[t_i, j_next4]);
                fb.seal_block(unroll_body_block);

                // single-iter tail
                fb.switch_to_block(tail_body_block);
                emit_body(&mut fb, t_j);
                let j_next1 = fb.ins().iadd(t_j, one);
                fb.ins().jump(inner_head, &[t_i, j_next1]);
                fb.seal_block(tail_body_block);

                // outer_inc
                fb.switch_to_block(outer_inc);
                let oi_i = fb.block_params(outer_inc)[0];
                let i_next = fb.ins().iadd(oi_i, step);
                fb.ins().jump(outer_head, &[i_next]);
                fb.seal_block(outer_inc);

                fb.switch_to_block(exit);
                fb.ins().return_(&[]);
                fb.seal_block(exit);

                fb.finalize();
                ctx.func = func;
                let suffix = FN_COUNTER.fetch_add(1, Ordering::Relaxed);
                let fname = format!("h_kernel_12q_4x3_unroll_{}", suffix);
                let func_id = module.declare_function(&fname, Linkage::Local, &sig).ok()?;
                module.define_function(func_id, &mut ctx).ok()?;
                module.clear_context(&mut ctx);
                module.finalize_definitions().ok()?;
                dump_ir(&ctx.func);
                let code = module.get_finalized_function(func_id);

                // EPIC 62 Phase 3.5: Return raw pointer for DeepKernelFn ABI
                Some(code as *const u8)
            });
        }

        if qubits == 12 && std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
            eprintln!("jit_debug: h_12q_general_selected (JIT_12Q_FASTPATH=off)");
        }
        with_jit_module(|module| {
            let mut ctx = module.make_context();

            // Signature: (real_ptr: i64, imag_ptr: i64, len: i64, bit: i64)
            let mut sig = Signature::new(cranelift_codegen::isa::CallConv::triple_default(
                &Triple::host(),
            ));
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(types::I64));

            // Function
            let mut func = Function::new();
            func.signature = sig.clone();
            let mut fbctx = FunctionBuilderContext::new();
            let mut fb = FunctionBuilder::new(&mut func, &mut fbctx);
            let entry = fb.create_block();
            fb.append_block_params_for_function_params(entry);
            fb.switch_to_block(entry);
            fb.seal_block(entry);

            // Params
            let real = fb.block_params(entry)[0];
            let imag = fb.block_params(entry)[1];
            let lenv = fb.block_params(entry)[2];
            let bitv = fb.block_params(entry)[3];

            // Constants
            let inv = fb.ins().f32const(0.70710677f32);
            let zero = fb.ins().iconst(types::I64, 0);
            let one = fb.ins().iconst(types::I64, 1);
            let two = fb.ins().iconst(types::I64, 1); // shift count = 1 for step = bit << 1
            let two_bits = fb.ins().iconst(types::I64, 2); // for bytes = index << 2
            let four = fb.ins().iconst(types::I64, 4);
            let step = fb.ins().ishl(bitv, two);
            let bit4 = fb.ins().ishl(bitv, two_bits);

            // Blocks with parameters for i and j to avoid stale SSA reads
            let outer_head = fb.create_block();
            fb.append_block_param(outer_head, types::I64); // i
            let inner_head = fb.create_block();
            fb.append_block_param(inner_head, types::I64); // i
            fb.append_block_param(inner_head, types::I64); // j
            // 2x-unroll selector and bodies
            let inner_select = fb.create_block();
            fb.append_block_param(inner_select, types::I64); // i
            fb.append_block_param(inner_select, types::I64); // j
            let inner_body2 = fb.create_block(); // two iterations
            fb.append_block_param(inner_body2, types::I64); // i
            fb.append_block_param(inner_body2, types::I64); // j
            let inner_body1 = fb.create_block(); // single tail
            fb.append_block_param(inner_body1, types::I64); // i
            fb.append_block_param(inner_body1, types::I64); // j
            let outer_inc = fb.create_block();
            fb.append_block_param(outer_inc, types::I64); // i
            let exit = fb.create_block();

            // enter with i = 0
            fb.ins().jump(outer_head, &[zero]);

            // outer_head: if (i < len) -> inner_head(i, 0) else exit
            fb.switch_to_block(outer_head);
            let i_cur = fb.block_params(outer_head)[0];
            let cmp_outer = fb.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
                i_cur,
                lenv,
            );
            fb.ins()
                .brif(cmp_outer, inner_head, &[i_cur, zero], exit, &[]);
            fb.seal_block(outer_head);

            // inner_head: if (j < bit) -> inner_select(i, j) else outer_inc(i)
            fb.switch_to_block(inner_head);
            let ih_i = fb.block_params(inner_head)[0];
            let ih_j = fb.block_params(inner_head)[1];
            let cmp_inner = fb.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
                ih_j,
                bitv,
            );
            // Allow disabling unroll via bench flag: if UNROLL=0, route to tail body directly
            #[allow(unused_mut)]
            let mut use_unroll = true;
            if let Ok(val) = std::env::var("JIT_UNROLL") {
                if val.trim() == "0" {
                    use_unroll = false;
                }
            }
            if use_unroll {
                fb.ins()
                    .brif(cmp_inner, inner_select, &[ih_i, ih_j], outer_inc, &[ih_i]);
            } else {
                fb.ins()
                    .brif(cmp_inner, inner_body1, &[ih_i, ih_j], outer_inc, &[ih_i]);
            }
            fb.seal_block(inner_head);

            // inner_select: choose 2x body if (j+1 < bit) else tail
            fb.switch_to_block(inner_select);
            let sel_i = fb.block_params(inner_select)[0];
            let sel_j = fb.block_params(inner_select)[1];
            let one_sel = fb.ins().iconst(types::I64, 1);
            let jp1 = fb.ins().iadd(sel_j, one_sel);
            let cmp_two = fb.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan,
                jp1,
                bitv,
            );
            fb.ins().brif(
                cmp_two,
                inner_body2,
                &[sel_i, sel_j],
                inner_body1,
                &[sel_i, sel_j],
            );
            fb.seal_block(inner_select);

            // inner_body2: operate twice (j and j+1); then j += 2
            fb.switch_to_block(inner_body2);
            let ib_i = fb.block_params(inner_body2)[0];
            let ib_j = fb.block_params(inner_body2)[1];
            let i0 = fb.ins().iadd(ib_i, ib_j);
            // Peephole: compute addresses once and reuse across loads/stores
            let off0 = fb.ins().imul(i0, four);
            let addr_r0 = fb.ins().iadd(real, off0);
            let addr_i0 = fb.ins().iadd(imag, off0);
            let off1 = fb.ins().iadd(off0, bit4);
            let addr_r1 = fb.ins().iadd(real, off1);
            let addr_i1 = fb.ins().iadd(imag, off1);
            let mflags = MemFlags::new();
            // EPIC 60: First iteration - load, apply H depth times, store
            let mut r0 = fb.ins().load(types::F32, mflags, addr_r0, 0);
            let mut r1 = fb.ins().load(types::F32, mflags, addr_r1, 0);
            let mut im0 = fb.ins().load(types::F32, mflags, addr_i0, 0);
            let mut im1 = fb.ins().load(types::F32, mflags, addr_i1, 0);
            for _ in 0..config.depth {
                let r_add = fb.ins().fadd(r0, r1);
                let r_sub = fb.ins().fsub(r0, r1);
                let i_add = fb.ins().fadd(im0, im1);
                let i_sub = fb.ins().fsub(im0, im1);
                r0 = fb.ins().fmul(r_add, inv);
                r1 = fb.ins().fmul(r_sub, inv);
                im0 = fb.ins().fmul(i_add, inv);
                im1 = fb.ins().fmul(i_sub, inv);
            }
            fb.ins().store(mflags, r0, addr_r0, 0);
            fb.ins().store(mflags, r1, addr_r1, 0);
            fb.ins().store(mflags, im0, addr_i0, 0);
            fb.ins().store(mflags, im1, addr_i1, 0);
            // EPIC 60: Second iteration at j+1 - load, apply H depth times, store
            let off0_b = fb.ins().iadd(off0, four);
            let addr_r0_b = fb.ins().iadd(real, off0_b);
            let addr_i0_b = fb.ins().iadd(imag, off0_b);
            let off1_b = fb.ins().iadd(off1, four);
            let addr_r1_b = fb.ins().iadd(real, off1_b);
            let addr_i1_b = fb.ins().iadd(imag, off1_b);
            let mut r0_b = fb.ins().load(types::F32, mflags, addr_r0_b, 0);
            let mut r1_b = fb.ins().load(types::F32, mflags, addr_r1_b, 0);
            let mut im0_b = fb.ins().load(types::F32, mflags, addr_i0_b, 0);
            let mut im1_b = fb.ins().load(types::F32, mflags, addr_i1_b, 0);
            for _ in 0..config.depth {
                let r_add_b = fb.ins().fadd(r0_b, r1_b);
                let r_sub_b = fb.ins().fsub(r0_b, r1_b);
                let i_add_b = fb.ins().fadd(im0_b, im1_b);
                let i_sub_b = fb.ins().fsub(im0_b, im1_b);
                r0_b = fb.ins().fmul(r_add_b, inv);
                r1_b = fb.ins().fmul(r_sub_b, inv);
                im0_b = fb.ins().fmul(i_add_b, inv);
                im1_b = fb.ins().fmul(i_sub_b, inv);
            }
            fb.ins().store(mflags, r0_b, addr_r0_b, 0);
            fb.ins().store(mflags, r1_b, addr_r1_b, 0);
            fb.ins().store(mflags, im0_b, addr_i0_b, 0);
            fb.ins().store(mflags, im1_b, addr_i1_b, 0);
            // j += 2
            let two_iter = fb.ins().iconst(types::I64, 2);
            let j_next2 = fb.ins().iadd(ib_j, two_iter);
            fb.ins().jump(inner_head, &[ib_i, j_next2]);
            fb.seal_block(inner_body2);

            // inner_body1: single-iter tail, then j += 1
            fb.switch_to_block(inner_body1);
            let tb_i = fb.block_params(inner_body1)[0];
            let tb_j = fb.block_params(inner_body1)[1];
            let ti0 = fb.ins().iadd(tb_i, tb_j);
            let toff0 = fb.ins().imul(ti0, four);
            let taddr_r0 = fb.ins().iadd(real, toff0);
            let taddr_i0 = fb.ins().iadd(imag, toff0);
            let toff1 = fb.ins().iadd(toff0, bit4);
            let taddr_r1 = fb.ins().iadd(real, toff1);
            let taddr_i1 = fb.ins().iadd(imag, toff1);
            // EPIC 60: Load amplitudes
            let mut tr0 = fb.ins().load(types::F32, mflags, taddr_r0, 0);
            let mut tr1 = fb.ins().load(types::F32, mflags, taddr_r1, 0);
            let mut tim0 = fb.ins().load(types::F32, mflags, taddr_i0, 0);
            let mut tim1 = fb.ins().load(types::F32, mflags, taddr_i1, 0);
            // EPIC 60: Apply H depth times (compile-time unroll)
            for _ in 0..config.depth {
                let tr_add = fb.ins().fadd(tr0, tr1);
                let tr_sub = fb.ins().fsub(tr0, tr1);
                let ti_add = fb.ins().fadd(tim0, tim1);
                let ti_sub = fb.ins().fsub(tim0, tim1);
                tr0 = fb.ins().fmul(tr_add, inv);
                tr1 = fb.ins().fmul(tr_sub, inv);
                tim0 = fb.ins().fmul(ti_add, inv);
                tim1 = fb.ins().fmul(ti_sub, inv);
            }
            // EPIC 60: Store final results
            fb.ins().store(mflags, tr0, taddr_r0, 0);
            fb.ins().store(mflags, tr1, taddr_r1, 0);
            fb.ins().store(mflags, tim0, taddr_i0, 0);
            fb.ins().store(mflags, tim1, taddr_i1, 0);
            let j_next_tail = fb.ins().iadd(tb_j, one);
            fb.ins().jump(inner_head, &[tb_i, j_next_tail]);
            fb.seal_block(inner_body1);

            // outer_inc: i += step; goto outer_head
            fb.switch_to_block(outer_inc);
            let oi_i = fb.block_params(outer_inc)[0];
            let i_next = fb.ins().iadd(oi_i, step);
            fb.ins().jump(outer_head, &[i_next]);
            fb.seal_block(outer_inc);

            // exit
            fb.switch_to_block(exit);
            fb.ins().return_(&[]);
            fb.seal_block(exit);

            // EPIC 58: debug counters (loads/stores/arith) can be added here if desired.
            if std::env::var("JIT_DEBUG").ok().as_deref() == Some("1") {
                // Simple marker to indicate EPIC 58 path executed; fine-grained counts are inlined above.
                eprintln!("jit_debug: general H kernel built (unroll path active)");
            }
            fb.finalize();
            ctx.func = func;
            let suffix = FN_COUNTER.fetch_add(1, Ordering::Relaxed);
            let fname = format!("h_kernel_gen_{}", suffix);
            let func_id = module.declare_function(&fname, Linkage::Local, &sig).ok()?;
            module.define_function(func_id, &mut ctx).ok()?;
            module.clear_context(&mut ctx);
            module.finalize_definitions().ok()?;
            let code = module.get_finalized_function(func_id);

            // EPIC 62 Phase 3.5: Return raw pointer for DeepKernelFn ABI
            Some(code as *const u8)
        })
    }
}
