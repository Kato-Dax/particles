use std::{num::NonZeroU64, sync::Arc};

use super::*;
use anyhow::*;
use model::*;
use rand::Rng;
use wgpu::{util::DrawIndexedIndirectArgs, CommandEncoder};
use winit::{
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Uniforms {
    view_proj: cgmath::Matrix4<f32>,
    n_masses: u32,
    n_particles_requested: u32,
    n_particles: u32,
    g: f32,
    delta_time: f32,
    push: f32,
    reset: u32,
    frame: u32,
    cam_pos: cgmath::Vector3<f32>,
    padding: u32,
    random: cgmath::Vector3<f32>,
    padding2: u32,
}

unsafe impl bytemuck::Pod for Uniforms {}
unsafe impl bytemuck::Zeroable for Uniforms {}

impl Uniforms {
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity(),
            n_masses: 0,
            n_particles_requested: 0,
            n_particles: 0,
            g: 0.0,
            delta_time: 0.0,
            reset: 0,
            push: 0.0,
            cam_pos: cgmath::Vector3::zero(),
            frame: 0,
            padding: 0,
            random: cgmath::Vector3::zero(),
            padding2: 0,
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix();
    }
}

#[repr(C)]
#[allow(unused)]
#[derive(Copy, Clone)]
struct Particle {
    position_and_scale: cgmath::Vector4<f32>,
    velocity_and_lifetime: cgmath::Vector4<f32>,
}

unsafe impl bytemuck::Pod for Particle {}
unsafe impl bytemuck::Zeroable for Particle {}

#[repr(C)]
#[derive(Copy, Clone)]
struct Mass {
    position_and_scale: cgmath::Vector4<f32>,
}
unsafe impl bytemuck::Pod for Mass {}
unsafe impl bytemuck::Zeroable for Mass {}

struct Binding {
    buffer: Arc<wgpu::Buffer>,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

struct PipePackage {
    pipeline: wgpu::RenderPipeline,
    model: Model,
    binding: Binding,
}

struct PipePackageMass {
    pipeline: wgpu::RenderPipeline,
    model: Model,
    binding: Binding,
    instances: Vec<Mass>,
}

type Job = Box<dyn FnOnce(&mut CommandEncoder)>;

#[allow(dead_code)]
pub struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    depth_texture: texture::Texture,
    uniforms: Uniforms,
    uniform_binding: Binding,
    particles: PipePackage,
    masses: PipePackageMass,
    indirect_buffer: wgpu::Buffer,
    camera: Camera,
    last_time: std::time::Instant,
    test_angle: f32,
    avg_speed_last_frame: f32,
    g: f32,
    particle_adding_value: i32,
    print: bool,
    growing: bool,
    compute_pipeline: wgpu::ComputePipeline,
    suspend_updates: bool,
    jobs: Vec<Job>,
    frame: u32,
    rng: rand::rngs::ThreadRng,
    random_buffer: wgpu::Buffer,
}
impl State {
    pub async fn new(window: Arc<Window>, particle_number: i32, g: f32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions {
                gl: wgpu::GlBackendOptions {
                    gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
                    fence_behavior: wgpu::GlFenceBehavior::default(),
                    debug_fns: Default::default(),
                },
                dx12: wgpu::Dx12BackendOptions::default(),
                noop: wgpu::NoopBackendOptions { enable: false },
            },
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds {
                for_resource_creation: None,
                for_device_loss: Some(90),
            },
            display: None,
        });
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                    ..Default::default()
                },
                label: None,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .unwrap();
        let window_size = window.inner_size();
        let surface_capabilities = surface.get_capabilities(&adapter);
        let supported_formats = surface_capabilities.formats;
        let supported_alpha_modes = surface_capabilities.alpha_modes;
        let format = if supported_formats.contains(&wgpu::TextureFormat::Bgra8UnormSrgb) {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            supported_formats[0]
        };
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: window_size.width,
            height: window_size.height,
            present_mode: if surface_capabilities
                .present_modes
                .contains(&wgpu::PresentMode::Mailbox)
            {
                wgpu::PresentMode::Mailbox
            } else if surface_capabilities
                .present_modes
                .contains(&wgpu::PresentMode::Immediate)
            {
                wgpu::PresentMode::Immediate
            } else {
                wgpu::PresentMode::Fifo
            },
            alpha_mode: supported_alpha_modes[0],
            view_formats: vec![format],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let camera = Camera::new(surface_config.width as f32 / surface_config.height as f32);

        let mut uniforms = Uniforms::new();
        uniforms.n_particles_requested = particle_number as u32;
        uniforms.update_view_proj(&camera);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                }),
            }],
            label: Some("uniform_bind_group"),
        });
        let uniform_binding = Binding {
            buffer: Arc::new(uniform_buffer),
            bind_group_layout: uniform_bind_group_layout,
            bind_group: uniform_bind_group,
        };
        let depth_texture = texture::Texture::create_depth_texture(
            &device,
            surface_config.width,
            surface_config.height,
            "depth_texture",
        );
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
        let model = model::Model::load(
            &device,
            &queue,
            &texture_bind_group_layout,
            "assets/simpleParticle.obj",
        )?;

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            size: 8 + (1024 * std::mem::size_of::<Particle>()) as u64,
            mapped_at_creation: false,
        });

        let random_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Random Buffer"),
            size: 8 + (1024 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let instance_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            min_binding_size: None,
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("instance_bind_group_layout"),
            });

        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &instance_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &instance_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &random_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
            label: Some("instance_bind_group"),
        });
        let instances_binding = Binding {
            buffer: Arc::new(instance_buffer),
            bind_group_layout: instance_bind_group_layout,
            bind_group: instance_bind_group,
        };

        let shader_module =
            device.create_shader_module(wgpu::include_wgsl!("shaders/particle_shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&texture_bind_group_layout),
                    Some(&uniform_binding.bind_group_layout),
                    Some(&instances_binding.bind_group_layout),
                ],
                immediate_size: 0,
                // push_constant_ranges: &[],
            });

        let vertex_buffer_layouts = &[Vertex::desc()];
        let color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let render_pipeline = create_render_pipeline(
            &device,
            render_pipeline_layout,
            &shader_module,
            &shader_module,
            vertex_buffer_layouts,
            color_targets,
            Some("Particle Pipeline"),
        );

        let mass = model::Model::load(
            &device,
            &queue,
            &texture_bind_group_layout,
            "assets/sphere.obj",
        )?;
        let masses: Vec<Mass> = vec![
            cgmath::Vector4::new(0.0, 0.0, 0.0, -1.0 / 100.0),
            cgmath::Vector4::new(0.0, 0.0, 0.0, 8.0 / 100.0),
            cgmath::Vector4::new(0.0, 0.0, 0.0, -3.0 / 100.0),
            cgmath::Vector4::new(0.0, 0.0, 0.0, 7.0 / 100.0),
        ]
        .into_iter()
        .map(|position_and_scale| Mass { position_and_scale })
        .collect();
        let mass_instance_size = masses.len() * std::mem::size_of::<Mass>();
        let masses_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&masses),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let masses_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                    },
                    count: None,
                }],
                label: Some("instance_bind_group_layout"),
            });
        let masses_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &masses_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &masses_buffer,
                    size: NonZeroU64::new(mass_instance_size as u64),
                    offset: 0,
                }),
            }],
            label: Some("instance_bind_group"),
        });
        let masses_binding = Binding {
            buffer: Arc::new(masses_buffer),
            bind_group_layout: masses_bind_group_layout,
            bind_group: masses_bind_group,
        };
        let mass_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Pipeline"),
            bind_group_layouts: &[
                Some(&texture_bind_group_layout),
                Some(&uniform_binding.bind_group_layout),
                Some(&masses_binding.bind_group_layout),
            ],
            // push_constant_ranges: &[],
            immediate_size: 0,
        });
        let mass_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/mass_shader.wgsl"));
        let vertex_buffer_layouts = &[Vertex::desc()];
        let color_targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let mass_pipeline = create_render_pipeline(
            &device,
            mass_pipe_layout,
            &mass_shader,
            &mass_shader,
            vertex_buffer_layouts,
            color_targets,
            Some("MASS PIPELINE"),
        );
        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect_buffer"),
            size: std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute pipeline layout"),
                bind_group_layouts: &[
                    Some(&uniform_binding.bind_group_layout),
                    Some(&instances_binding.bind_group_layout),
                    Some(&masses_binding.bind_group_layout),
                ],
                // push_constant_ranges: &[],
                immediate_size: 0,
            });
        let compute_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/compute.wgsl"));
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            window,
            rng: rand::thread_rng(),
            frame: 0,
            print: false,
            surface,
            growing: false,
            device,
            queue,
            surface_config,
            particles: PipePackage {
                pipeline: render_pipeline,
                model,
                binding: instances_binding,
            },
            camera,
            uniforms,
            uniform_binding,
            masses: PipePackageMass {
                pipeline: mass_pipeline,
                model: mass,
                binding: masses_binding,
                instances: masses,
            },
            depth_texture,
            last_time: std::time::Instant::now(),
            test_angle: 0.0,
            avg_speed_last_frame: 0.0,
            g,
            particle_adding_value: 0,
            indirect_buffer,
            compute_pipeline,
            suspend_updates: false,
            jobs: Vec::new(),
            random_buffer,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.camera.aspect = (new_size.width as f32) / (new_size.height as f32);
        self.depth_texture = texture::Texture::create_depth_texture(
            &self.device,
            new_size.width,
            new_size.height,
            "depth_texture",
        );
        self.surface.configure(&self.device, &self.surface_config);
    }
    pub fn device_input(&mut self, event: &DeviceEvent) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            self.camera.controller.heading +=
                delta.0 as f32 * -self.camera.controller.mouse_sensitivity;
            self.camera.controller.pitch +=
                delta.1 as f32 * -self.camera.controller.mouse_sensitivity;
        }
    }
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera.process_events(event);
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                match (event.state, event.physical_key) {
                    (ElementState::Pressed, PhysicalKey::Code(KeyCode::KeyC)) => {
                        self.camera.reset();
                    }
                    (ElementState::Pressed, PhysicalKey::Code(KeyCode::KeyU)) => {
                        self.suspend_updates = !self.suspend_updates;
                    }
                    (ElementState::Pressed, PhysicalKey::Code(KeyCode::NumpadAdd)) => {
                        self.g *= 1.05;
                    }
                    (ElementState::Pressed, PhysicalKey::Code(KeyCode::NumpadSubtract)) => {
                        self.g *= 0.95;
                    }
                    (ElementState::Pressed, PhysicalKey::Code(KeyCode::KeyG)) => {
                        self.growing = !self.growing;
                    }
                    (ElementState::Pressed, PhysicalKey::Code(KeyCode::KeyR)) => {
                        self.uniforms.reset = 1;
                    }
                    (ElementState::Pressed, PhysicalKey::Code(KeyCode::KeyB)) => {
                        self.uniforms.push = 1.0;
                    }
                    (state, PhysicalKey::Code(KeyCode::KeyP)) => {
                        self.print = state == ElementState::Pressed;
                    }
                    _ => {}
                };
            }
            _ => return false,
        }
        true
    }
    pub fn update(&mut self) {
        let delta_time = std::time::Instant::now().duration_since(self.last_time);
        let delta_time = delta_time.as_secs_f32();
        self.last_time = std::time::Instant::now();
        self.camera.update(delta_time);
        self.uniforms.update_view_proj(&self.camera);
        self.uniforms.cam_pos = self.camera.eye.to_vec();
        if self.print {
            println!(
                "{}, n_particles: {}, frametime: {}, pos: {:?}",
                if self.growing { '+' } else { '-' },
                self.uniforms.n_particles,
                delta_time,
                self.camera.eye
            );
        }
        if self.growing {
            self.uniforms.n_particles_requested = self.uniforms.n_particles + 1;
        }
        self.grow_particles_buffer();
        let l: f32 = 0.3;
        for i in 0..self.masses.instances.len() {
            self.masses.instances[i].position_and_scale.x =
                (self.test_angle / (i + 1) as f32).cos() * l * i as f32;
            self.masses.instances[i].position_and_scale.z =
                (self.test_angle / (i + 1) as f32).sin() * l * i as f32;
        }
        self.test_angle += 1.0 / std::f32::consts::PI * delta_time;

        let data = bytemuck::cast_slice(&self.masses.instances);
        self.queue
            .write_buffer(&self.masses.binding.buffer, 0, data);

        self.uniforms.g = self.g;
        self.uniforms.random = cgmath::Vector3::new(self.rng.gen(), self.rng.gen(), self.rng.gen());
        self.uniforms.delta_time = delta_time;
        self.uniforms.n_masses = self.masses.instances.len() as u32;
        let data = &[self.uniforms];
        let data = bytemuck::cast_slice(data);
        self.queue
            .write_buffer(&self.uniform_binding.buffer, 0, data);
        self.uniforms.n_particles = self
            .uniforms
            .n_particles_requested
            .min(self.uniforms.n_particles + 256);
        self.uniforms.reset = 0;
        self.uniforms.push = 0.0;
        self.uniforms.frame += 1;
    }
    fn grow_particles_buffer(&mut self) {
        let min_size =
            8 + std::mem::size_of::<Particle>() as u32 * self.uniforms.n_particles_requested;
        if (min_size as u64) < self.particles.binding.buffer.size() {
            return;
        }
        let new_size = min_size as u64 * 3 / 2;
        let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles buffer"),
            size: new_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let new_buffer = Arc::new(new_buffer);
        let old_buffer = Arc::clone(&self.particles.binding.buffer);
        self.particles.binding.buffer = Arc::clone(&new_buffer);
        self.particles.binding.bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particles bind group"),
                layout: &self.particles.binding.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.particles.binding.buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.random_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            });
        self.jobs.push(Box::new(move |encoder| {
            encoder.copy_buffer_to_buffer(
                &old_buffer,
                0,
                &new_buffer,
                0,
                (min_size as u64).min(old_buffer.size()),
            );
        }));
    }
    pub fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            },
            wgpu::CurrentSurfaceTexture::Timeout => return,
            wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => panic!("device lost"),
            wgpu::CurrentSurfaceTexture::Validation => panic!("validation error"),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });
        self.jobs.drain(..).for_each(|f| f(&mut encoder));
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.uniform_binding.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.particles.binding.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.masses.binding.bind_group, &[]);
            compute_pass.dispatch_workgroups(self.uniforms.n_particles / 256 + 1, 1, 1);
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                label: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.particles.pipeline);
            for mesh in &self.particles.model.meshes {
                let material = &self.particles.model.materials[mesh.material];
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.set_bind_group(0, &material.bind_group, &[]);
                render_pass.set_bind_group(1, &self.uniform_binding.bind_group, &[]);
                render_pass.set_bind_group(2, &self.particles.binding.bind_group, &[]);
                // render_pass.draw_indexed_indirect(&self.indirect_buffer, 0);
                render_pass.draw_indexed(0..mesh.num_elements, 0, 0..self.uniforms.n_particles);
            }
            render_pass.set_pipeline(&self.masses.pipeline);
            for mesh in &self.masses.model.meshes {
                let material = &self.masses.model.materials[mesh.material];
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.set_bind_group(0, &material.bind_group, &[]);
                render_pass.set_bind_group(1, &self.uniform_binding.bind_group, &[]);
                render_pass.set_bind_group(2, &self.masses.binding.bind_group, &[]);
                render_pass.draw_indexed(
                    0..mesh.num_elements,
                    0,
                    0..self.masses.instances.len() as u32,
                );
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        frame.present();
    }
}
fn create_render_pipeline(
    device: &wgpu::Device,
    layout: wgpu::PipelineLayout,
    vs_module: &wgpu::ShaderModule,
    fs_module: &wgpu::ShaderModule,
    vertex_buffer_layouts: &[wgpu::VertexBufferLayout],
    color_targets: &[Option<wgpu::ColorTargetState>],
    label: Option<&'static str>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: vs_module,
            entry_point: Some("vs_main"),
            buffers: vertex_buffer_layouts,
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
        },
        fragment: Some(wgpu::FragmentState {
            targets: color_targets,
            module: fs_module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        cache: None,
        multiview_mask: None,
    })
}
