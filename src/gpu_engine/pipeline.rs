use crate::gpu_engine::buffer_types::SimulationParams;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

// Reverted to original blocking pipeline
pub struct GpuSimulation {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_a: wgpu::BindGroup,
    bind_group_b: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    tex_a: wgpu::Texture,
    tex_b: wgpu::Texture,
    tex_meta: wgpu::Texture,
    tex_color: wgpu::Texture,
    tex_rom: wgpu::Texture,
    // LOD Resources
    tex_satellite: wgpu::Texture,
    downsample_pipeline: wgpu::ComputePipeline,
    bind_group_downsample: wgpu::BindGroup,
    step_count: u64,
    width: u32,
    height: u32,
    // Persistent Staging Buffers
    staging_satellite: wgpu::Buffer,
    staging_roi_color: wgpu::Buffer,
    staging_roi_state: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DownsampleParams {
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
}

impl GpuSimulation {
    pub async fn new(width: u32, height: u32, custom_shader: Option<&str>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an appropriate adapter");

        println!("Selected Adapter: {:?}", adapter.get_info());

        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // 1. Create Textures
        let tex_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Logic Texture A"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let tex_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Logic Texture B"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let tex_meta = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Meta Map Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let tex_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color Output Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let tex_rom = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ROM Texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // LOD
        let sat_width = 4096;
        let sat_height = 4096;
        let tex_satellite = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Satellite Texture"),
            size: wgpu::Extent3d {
                width: sat_width,
                height: sat_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view_a = tex_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = tex_b.create_view(&wgpu::TextureViewDescriptor::default());
        let view_meta = tex_meta.create_view(&wgpu::TextureViewDescriptor::default());
        let view_color = tex_color.create_view(&wgpu::TextureViewDescriptor::default());
        let view_rom = tex_rom.create_view(&wgpu::TextureViewDescriptor::default());
        let view_satellite = tex_satellite.create_view(&wgpu::TextureViewDescriptor::default());

        let params = SimulationParams {
            width,
            height,
            frame_count: 0,
            _padding: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Simulation Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group A"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view_meta),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&view_color),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&view_rom),
                },
            ],
        });

        let bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group B"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view_meta),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&view_color),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&view_rom),
                },
            ],
        });

        let shader_source = if let Some(src) = custom_shader {
            Cow::Borrowed(src)
        } else {
            Cow::Borrowed(include_str!("../shaders/simulation.wgsl"))
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Simulation Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Simulation Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Simulation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // Downsample stuff (LOD)
        let downsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Downsample Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let d_params = DownsampleParams {
            source_width: width,
            source_height: height,
            target_width: sat_width,
            target_height: sat_height,
        };
        let downsample_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Downsample Params"),
                contents: bytemuck::bytes_of(&d_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group_downsample = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Downsample Bind Group"),
            layout: &downsample_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_satellite),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: downsample_params_buffer.as_entire_binding(),
                },
            ],
        });

        let ds_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Downsample Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/downsample.wgsl"
            ))),
        });

        let ds_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Downsample Pipeline Layout"),
            bind_group_layouts: &[&downsample_layout],
            push_constant_ranges: &[],
        });

        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Downsample Pipeline"),
                layout: Some(&ds_pipeline_layout),
                module: &ds_shader,
                entry_point: "main",
                compilation_options: Default::default(),
            });

        let staging_satellite = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent Satellite Staging"),
            size: (4096 * 4096 * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let staging_roi_color = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent ROI Color Staging"),
            size: (4096 * 4096 * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let staging_roi_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent ROI State Staging"),
            size: (4096 * 4096 * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_a,
            bind_group_b,
            params_buffer,
            tex_a,
            tex_b,
            tex_meta,
            tex_color,
            tex_rom,
            tex_satellite,
            downsample_pipeline,
            bind_group_downsample,
            step_count: 0,
            width,
            height,
            staging_satellite,
            staging_roi_color,
            staging_roi_state,
        }
    }

    pub fn step(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let bind_group = if self.step_count % 2 == 0 {
            &self.bind_group_a
        } else {
            &self.bind_group_b
        };

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sim Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups((self.width + 15) / 16, (self.height + 15) / 16, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait); // Reverted to blocking
        self.step_count += 1;

        let params = SimulationParams {
            width: self.width,
            height: self.height,
            frame_count: self.step_count as u32,
            _padding: 0,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    pub fn downsample(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Downsample Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Downsample Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.downsample_pipeline);
            cpass.set_bind_group(0, &self.bind_group_downsample, &[]);
            cpass.dispatch_workgroups(4096 / 16, 4096 / 16, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn update_tile(&self, x: u32, y: u32, tile_type: u8) {
        if x >= self.width || y >= self.height {
            return;
        }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tex_meta,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &[tile_type],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.width),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn upload_meta(&self, data: &[u8]) {
        let width = self.width;
        let height = self.height;
        let unpadded = width as usize;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padding = (align - unpadded % align) % align;
        let padded_row_size = unpadded + padding;
        let mut padded_data = Vec::with_capacity(padded_row_size * height as usize);
        for row in data.chunks(unpadded) {
            padded_data.extend_from_slice(row);
            padded_data.extend(std::iter::repeat(0).take(padding));
        }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tex_meta,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_size as u32),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn update_rom(&self, data: &[u32]) {
        let width = 256;
        let height = 256;
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tex_rom,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn update_state(&self, x: u32, y: u32, value: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let data = [value];
        for tex in [&self.tex_a, &self.tex_b] {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&data),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * 4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    pub async fn capture_satellite(&self) -> Vec<u8> {
        let width = self.tex_satellite.width();
        let height = self.tex_satellite.height();
        let unpadded_bytes_per_row = width * 4;
        let size = (unpadded_bytes_per_row * height) as wgpu::BufferAddress;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.tex_satellite,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.staging_satellite,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(unpadded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let buffer_slice = self.staging_satellite.slice(..size);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait); // Waiting ONLY on readback
        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            self.staging_satellite.unmap();
            result
        } else {
            panic!("Failed to map satellite buffer");
        }
    }

    pub async fn capture_state_roi(&self, x: u32, y: u32, w: u32, h: u32) -> Vec<u32> {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        let w = w.min(self.width - x);
        let h = h.min(self.height - y);
        if w == 0 || h == 0 {
            return Vec::new();
        }
        let unpadded_bytes_per_row = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
        let size = (padded_bytes_per_row * h) as wgpu::BufferAddress;
        let source_tex = if self.step_count % 2 != 0 {
            &self.tex_b
        } else {
            &self.tex_a
        };
        let staging_buffer = &self.staging_roi_state;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: source_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_buffer.slice(..size);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait); // Waiting ONLY on readback
        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let mut result = Vec::with_capacity((w * h) as usize);
            for chunk in data.chunks(padded_bytes_per_row as usize) {
                let row = &chunk[..unpadded_bytes_per_row as usize];
                result.extend_from_slice(bytemuck::cast_slice(row));
            }
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to map State ROI buffer");
        }
    }

    pub fn write_state(&self, x: u32, y: u32, value: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let target_tex = if self.step_count % 2 == 0 {
            &self.tex_a
        } else {
            &self.tex_b
        };
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: target_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::bytes_of(&value),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    pub async fn capture_frame_roi(&self, x: u32, y: u32, w: u32, h: u32) -> Vec<u8> {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        let w = w.min(self.width - x);
        let h = h.min(self.height - y);
        if w == 0 || h == 0 {
            return Vec::new();
        }
        let unpadded_bytes_per_row = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
        let size = (padded_bytes_per_row * h) as wgpu::BufferAddress;
        let staging_buffer = &self.staging_roi_color;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.tex_color,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_buffer.slice(..size);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait); // Waiting ONLY on readback
        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let mut result = Vec::with_capacity((w * h * 4) as usize);
            for chunk in data.chunks(padded_bytes_per_row as usize) {
                result.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
            }
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to map ROI buffer");
        }
    }

    pub async fn capture_frame(&self) -> Vec<u32> {
        vec![]
    }

    pub fn load_rules_only(&mut self, rules: &[u8]) {
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tex_meta,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rules,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.width),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        let target_chunk_bytes = 4 * 1024 * 1024usize;
        let bytes_per_row = (self.width as usize) * 4;
        let rows_per_chunk = (target_chunk_bytes / bytes_per_row).max(1);
        let chunk_bytes = vec![0u8; rows_per_chunk * bytes_per_row];
        for tex in [&self.tex_a, &self.tex_b] {
            let mut y_start = 0u32;
            while y_start < self.height {
                let rows_this_chunk = (self.height - y_start).min(rows_per_chunk as u32);
                let bytes_this_chunk = (rows_this_chunk as usize) * bytes_per_row;
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: y_start,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &chunk_bytes[..bytes_this_chunk],
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(self.width * 4),
                        rows_per_image: Some(rows_this_chunk),
                    },
                    wgpu::Extent3d {
                        width: self.width,
                        height: rows_this_chunk,
                        depth_or_array_layers: 1,
                    },
                );
                y_start += rows_this_chunk;
            }
        }
    }

    pub fn load_state(&mut self, rules: &[u8], state: &[u32]) {
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tex_meta,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rules,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.width),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        let bytes_state = bytemuck::cast_slice(state);
        for tex in [&self.tex_a, &self.tex_b] {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes_state,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * 4),
                    rows_per_image: Some(self.height),
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    pub fn load_rom_compiled(&self, compiled_binary: &[u8]) {
        let num_instructions = compiled_binary.len() / 4;
        let mut data = vec![0u8; 256 * 256 * 4];
        for i in 0..num_instructions {
            let src_offset = i * 4;
            let pixel_idx = i;
            if pixel_idx >= 256 * 256 {
                break;
            }
            let byte_offset = pixel_idx * 4;
            if byte_offset + 4 <= data.len() && src_offset + 4 <= compiled_binary.len() {
                data[byte_offset..byte_offset + 4]
                    .copy_from_slice(&compiled_binary[src_offset..src_offset + 4]);
            }
        }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tex_rom,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(256),
            },
            wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
        );
        println!("Loaded {} pre-compiled GPU instructions", num_instructions);
    }

    pub fn load_rom(&self, tile8_binary: &[u8]) {
        let gpu_instructions = Self::compile_tile8_to_gpu(tile8_binary);
        let mut data = vec![0u8; 256 * 256 * 4];
        for (i, instr) in gpu_instructions.iter().enumerate() {
            let pixel_idx = i * 4;
            if pixel_idx >= 256 * 256 {
                break;
            }
            let byte_offset = pixel_idx * 4;
            if byte_offset + 4 <= data.len() {
                data[byte_offset..byte_offset + 4].copy_from_slice(&instr.to_le_bytes());
            }
        }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tex_rom,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(256),
            },
            wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
        );
        println!(
            "Compiled {} TILE-8 instructions to GPU format",
            tile8_binary.len()
        );
    }

    fn compile_tile8_to_gpu(tile8: &[u8]) -> Vec<u32> {
        let mut gpu_instrs = Vec::with_capacity(tile8.len());
        let mut last_r3_value: Option<u8> = None;
        for (idx, &byte) in tile8.iter().enumerate() {
            let opcode = (byte >> 4) & 0x0F;
            let rd = (byte >> 2) & 0x03;
            let rs_or_imm = byte & 0x03;
            if opcode == 0x0A && rd == 3 {
                last_r3_value = Some(rs_or_imm);
            }
            let gpu_instr = match opcode {
                0x00 => Self::gpu_instr(1, 0, 0, false, 0),
                0x01 => Self::gpu_instr(1, rd, rs_or_imm, false, 0),
                0x02 => Self::gpu_instr(2, rd, rs_or_imm, false, 0),
                0x03 => Self::gpu_instr(3, rd, rs_or_imm, false, 0),
                0x0A => Self::gpu_instr(1, rd, 0, true, rs_or_imm as u16),
                0x0B => Self::gpu_instr(4, 0, 0, false, last_r3_value.unwrap_or(0) as u16 * 4),
                0x0C => Self::gpu_instr(5, 0, 0, false, last_r3_value.unwrap_or(0) as u16 * 4),
                0x0D => Self::gpu_instr(6, 0, 0, false, last_r3_value.unwrap_or(0) as u16 * 4),
                _ => {
                    println!(
                        "Warning: Unsupported TILE-8 opcode 0x{:X} at index {}",
                        opcode, idx
                    );
                    Self::gpu_instr(1, 0, 0, false, 0)
                }
            };
            gpu_instrs.push(gpu_instr);
        }
        gpu_instrs
    }

    fn gpu_instr(op: u8, dest: u8, src: u8, use_imm: bool, imm: u16) -> u32 {
        let flags = if use_imm {
            ((dest & 0x07) << 4) | 0x80
        } else {
            ((dest & 0x07) << 4) | (src & 0x0F)
        };
        let imm_h = ((imm >> 8) & 0xFF) as u8;
        let imm_l = (imm & 0xFF) as u8;
        (op as u32) | ((flags as u32) << 8) | ((imm_h as u32) << 16) | ((imm_l as u32) << 24)
    }
}
