// GUI and rendering module

pub mod api;
pub mod backend;
pub mod recorder;
pub mod render;

// Re-export commonly used types from render
pub use render::{
    FieldLayerMode, ImageBuffer, ImageFormat, RenderError, RenderParams, RgbPixel, render_region,
    write_ppm,
};

// Re-export from backend
pub use backend::{
    FrameError, FrameFieldKind, FrameMeta, GuiFrameStream, open_recorded_frames,
    run_organism_frames,
};

// Re-export from api
pub use api::{
    GuiFrame, GuiRunInfo, GuiStream, load_frame, load_frame_range, load_run, preview_organism,
    stream_from_recorded,
};

// Re-export from recorder
pub use recorder::{
    RecordArtifact, RecordManifest, diff_runs, read_manifest, record_organism, record_region_with,
    replay_run,
};
