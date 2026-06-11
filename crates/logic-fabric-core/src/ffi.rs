use crate::quantum::QState;

// Opaque pointer for C context
pub struct LfcContext {
    state: Option<QState>,
}

#[no_mangle]
pub extern "C" fn lfc_context_create() -> *mut LfcContext {
    Box::into_raw(Box::new(LfcContext { state: None }))
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)] // FFI boundary: caller guarantees ptr validity; null check precedes unsafe block
pub extern "C" fn lfc_context_destroy(ctx: *mut LfcContext) {
    if ctx.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ctx);
    }
}

// Placeholder for future optimization/simulation logic
#[no_mangle]
pub extern "C" fn lfc_say_hello() {
    println!("Logic Fabric Core: C-API initialized.");
}
