// Debug signaling macros for optional verbose simulation output.
// Enabled when built with feature `debug-signals`.

#[cfg(feature = "debug-signals")]
#[macro_export]
macro_rules! dbg_signal {
    ($($arg:tt)*) => {{
        eprintln!($($arg)*);
    }};
}

#[cfg(not(feature = "debug-signals"))]
#[macro_export]
macro_rules! dbg_signal {
    ($($arg:tt)*) => {{
        // no-op in non-debug builds
    }};
}
