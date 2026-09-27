//! Small OpenCL 1.2 binding loaded directly from the staged Mali userspace library.
//! The device has no system ICD, so linking against a system libOpenCL is unsuitable.
use anyhow::{bail, Context, Result};
use libloading::Library;
use std::{
    collections::{BTreeMap, HashMap},
    ffi::{c_char, c_int, c_uint, c_ulong, c_void, CStr, CString},
    ptr,
    sync::{Arc, Mutex},
    time::Instant,
};

type Handle = *mut c_void;
type GetPlatformIds = unsafe extern "C" fn(c_uint, *mut Handle, *mut c_uint) -> c_int;
type GetDeviceIds =
    unsafe extern "C" fn(Handle, c_ulong, c_uint, *mut Handle, *mut c_uint) -> c_int;
type GetDeviceInfo = unsafe extern "C" fn(Handle, c_uint, usize, *mut c_void, *mut usize) -> c_int;
type CreateContext = unsafe extern "C" fn(
    *const isize,
    c_uint,
    *const Handle,
    Option<unsafe extern "C" fn()>,
    *mut c_void,
    *mut c_int,
) -> Handle;
type CreateCommandQueue = unsafe extern "C" fn(Handle, Handle, c_ulong, *mut c_int) -> Handle;
type CreateBuffer = unsafe extern "C" fn(Handle, c_ulong, usize, *mut c_void, *mut c_int) -> Handle;
type CreateProgramWithSource =
    unsafe extern "C" fn(Handle, c_uint, *const *const c_char, *const usize, *mut c_int) -> Handle;
type BuildProgram = unsafe extern "C" fn(
    Handle,
    c_uint,
    *const Handle,
    *const c_char,
    Option<unsafe extern "C" fn()>,
    *mut c_void,
) -> c_int;
type GetProgramBuildInfo =
    unsafe extern "C" fn(Handle, Handle, c_uint, usize, *mut c_void, *mut usize) -> c_int;
type CreateKernel = unsafe extern "C" fn(Handle, *const c_char, *mut c_int) -> Handle;
type SetKernelArg = unsafe extern "C" fn(Handle, c_uint, usize, *const c_void) -> c_int;
type EnqueueNDRangeKernel = unsafe extern "C" fn(
    Handle,
    Handle,
    c_uint,
    *const usize,
    *const usize,
    *const usize,
    c_uint,
    *const Handle,
    *mut Handle,
) -> c_int;
type EnqueueReadBuffer = unsafe extern "C" fn(
    Handle,
    Handle,
    c_uint,
    usize,
    usize,
    *mut c_void,
    c_uint,
    *const Handle,
    *mut Handle,
) -> c_int;
type EnqueueWriteBuffer = unsafe extern "C" fn(
    Handle,
    Handle,
    c_uint,
    usize,
    usize,
    *const c_void,
    c_uint,
    *const Handle,
    *mut Handle,
) -> c_int;
type Finish = unsafe extern "C" fn(Handle) -> c_int;
type Release = unsafe extern "C" fn(Handle) -> c_int;
type GetEventProfilingInfo =
    unsafe extern "C" fn(Handle, c_uint, usize, *mut c_void, *mut usize) -> c_int;

struct ProfileEvent {
    label: String,
    handle: Handle,
}

pub struct Cl {
    _library: Library,
    context: Handle,
    queue: Handle,
    program: Handle,
    create_buffer: CreateBuffer,
    create_kernel: CreateKernel,
    set_kernel_arg: SetKernelArg,
    enqueue_kernel: EnqueueNDRangeKernel,
    read_buffer: EnqueueReadBuffer,
    write_buffer: EnqueueWriteBuffer,
    finish_fn: Finish,
    get_event_profiling_info: GetEventProfilingInfo,
    release_event: Release,
    profile_enabled: bool,
    profile_events: Mutex<Vec<ProfileEvent>>,
    host_calls: Mutex<BTreeMap<&'static str, (u128, usize)>>,
    kernel_cache: Mutex<HashMap<String, Handle>>,
    release_mem: Release,
    release_kernel: Release,
    release_program: Release,
    release_queue: Release,
    release_context: Release,
}

// The queue and kernel arguments are used serially. The raw handles remain valid
// while Arc<Cl> holds the library, context, queue, and program alive.
unsafe impl Send for Cl {}
unsafe impl Sync for Cl {}

pub struct Buffer {
    handle: Handle,
    len: usize,
    cl: Arc<Cl>,
}

pub struct Kernel {
    handle: Handle,
    cl: Arc<Cl>,
    label: String,
}

fn checked(label: &str, code: c_int) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        bail!("OpenCL {label} failed: {code}")
    }
}

impl Cl {
    pub fn new(path: &str, source: &str) -> Result<Arc<Self>> {
        let library = unsafe { Library::new(path) }.with_context(|| format!("loading {path}"))?;
        macro_rules! load {
            ($symbol:literal, $ty:ty) => {{
                *unsafe { library.get::<$ty>(concat!($symbol, "\0").as_bytes()) }
                    .with_context(|| format!("loading OpenCL symbol {}", $symbol))?
            }};
        }
        let get_platform_ids = load!("clGetPlatformIDs", GetPlatformIds);
        let get_device_ids = load!("clGetDeviceIDs", GetDeviceIds);
        let get_device_info = load!("clGetDeviceInfo", GetDeviceInfo);
        let create_context = load!("clCreateContext", CreateContext);
        let create_command_queue = load!("clCreateCommandQueue", CreateCommandQueue);
        let create_buffer = load!("clCreateBuffer", CreateBuffer);
        let create_program = load!("clCreateProgramWithSource", CreateProgramWithSource);
        let build_program = load!("clBuildProgram", BuildProgram);
        let get_build_info = load!("clGetProgramBuildInfo", GetProgramBuildInfo);
        let create_kernel = load!("clCreateKernel", CreateKernel);
        let set_kernel_arg = load!("clSetKernelArg", SetKernelArg);
        let enqueue_kernel = load!("clEnqueueNDRangeKernel", EnqueueNDRangeKernel);
        let read_buffer = load!("clEnqueueReadBuffer", EnqueueReadBuffer);
        let write_buffer = load!("clEnqueueWriteBuffer", EnqueueWriteBuffer);
        let finish_fn = load!("clFinish", Finish);
        let get_event_profiling_info = load!("clGetEventProfilingInfo", GetEventProfilingInfo);
        let release_event = load!("clReleaseEvent", Release);
        let release_mem = load!("clReleaseMemObject", Release);
        let release_kernel = load!("clReleaseKernel", Release);
        let release_program = load!("clReleaseProgram", Release);
        let release_queue = load!("clReleaseCommandQueue", Release);
        let release_context = load!("clReleaseContext", Release);
        let mut platform = ptr::null_mut();
        checked("GetPlatformIDs", unsafe {
            get_platform_ids(1, &mut platform, ptr::null_mut())
        })?;
        let mut device = ptr::null_mut();
        checked("GetDeviceIDs", unsafe {
            get_device_ids(platform, 4, 1, &mut device, ptr::null_mut())
        })?;
        let mut name = [0u8; 256];
        checked("GetDeviceInfo(NAME)", unsafe {
            get_device_info(
                device,
                0x102B,
                name.len(),
                name.as_mut_ptr().cast(),
                ptr::null_mut(),
            )
        })?;
        let name = unsafe { CStr::from_ptr(name.as_ptr().cast()) }.to_string_lossy();
        if !name.contains("Mali-G610") {
            bail!("expected Mali-G610 GPU, found {name}")
        }
        let mut error = 0;
        let context =
            unsafe { create_context(ptr::null(), 1, &device, None, ptr::null_mut(), &mut error) };
        checked("CreateContext", error)?;
        let profile_enabled = std::env::var("LAYA_MALI_PROFILE").as_deref() == Ok("1");
        let queue = unsafe {
            create_command_queue(
                context,
                device,
                if profile_enabled { 2 } else { 0 },
                &mut error,
            )
        };
        checked("CreateCommandQueue", error)?;
        let source = CString::new(source)?;
        let source_ptr = source.as_ptr();
        let program = unsafe { create_program(context, 1, &source_ptr, ptr::null(), &mut error) };
        checked("CreateProgramWithSource", error)?;
        let options = CString::new("-cl-std=CL1.2")?;
        let build_error =
            unsafe { build_program(program, 1, &device, options.as_ptr(), None, ptr::null_mut()) };
        if build_error != 0 {
            let mut log = vec![0u8; 16384];
            unsafe {
                get_build_info(
                    program,
                    device,
                    0x1183,
                    log.len(),
                    log.as_mut_ptr().cast(),
                    ptr::null_mut(),
                )
            };
            let length = log.iter().position(|b| *b == 0).unwrap_or(log.len());
            bail!(
                "OpenCL BuildProgram failed {build_error}: {}",
                String::from_utf8_lossy(&log[..length])
            );
        }
        Ok(Arc::new(Self {
            _library: library,
            context,
            queue,
            program,
            create_buffer,
            create_kernel,
            set_kernel_arg,
            enqueue_kernel,
            read_buffer,
            write_buffer,
            finish_fn,
            get_event_profiling_info,
            release_event,
            profile_enabled,
            profile_events: Mutex::new(Vec::new()),
            host_calls: Mutex::new(BTreeMap::new()),
            kernel_cache: Mutex::new(HashMap::new()),
            release_mem,
            release_kernel,
            release_program,
            release_queue,
            release_context,
        }))
    }

    pub fn buffer(self: &Arc<Self>, size: usize) -> Result<Buffer> {
        if size == 0 {
            bail!("cannot allocate an empty OpenCL buffer")
        }
        let mut error = 0;
        let started = Instant::now();
        let handle =
            unsafe { (self.create_buffer)(self.context, 1, size, ptr::null_mut(), &mut error) };
        self.record_host("CreateBuffer", started);
        checked("CreateBuffer", error)?;
        Ok(Buffer {
            handle,
            len: size,
            cl: Arc::clone(self),
        })
    }

    pub fn from_bytes(self: &Arc<Self>, bytes: &[u8]) -> Result<Buffer> {
        let buffer = self.buffer(bytes.len())?;
        buffer.write(bytes)?;
        Ok(buffer)
    }

    pub fn kernel(self: &Arc<Self>, name: &str) -> Result<Kernel> {
        let label = name.to_owned();
        let started = Instant::now();
        let mut cache = self.kernel_cache.lock().unwrap();
        let handle = if let Some(&handle) = cache.get(name) {
            handle
        } else {
            let c_name = CString::new(name)?;
            let mut error = 0;
            let create_started = Instant::now();
            let handle = unsafe { (self.create_kernel)(self.program, c_name.as_ptr(), &mut error) };
            self.record_host("CreateKernel", create_started);
            checked("CreateKernel", error)?;
            cache.insert(name.to_owned(), handle);
            handle
        };
        self.record_host("KernelLookup", started);
        Ok(Kernel {
            handle,
            cl: Arc::clone(self),
            label,
        })
    }

    pub fn finish(&self) -> Result<()> {
        let started = Instant::now();
        let status = unsafe { (self.finish_fn)(self.queue) };
        self.record_host("Finish", started);
        checked("Finish", status)
    }

    pub fn warmup(self: &Arc<Self>) -> Result<()> {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = unsafe {
            std::slice::from_raw_parts(input.as_ptr().cast::<u8>(), std::mem::size_of_val(&input))
        };
        let a = self.from_bytes(bytes)?;
        let b = self.from_bytes(bytes)?;
        let output = self.buffer(bytes.len())?;
        self.kernel("vector_add")?
            .buffer(0, &a)?
            .buffer(1, &b)?
            .buffer(2, &output)?
            .i32(3, input.len() as i32)?
            .run(input.len())?;
        self.finish()
    }

    fn record_host(&self, label: &'static str, started: Instant) {
        if self.profile_enabled {
            let mut calls = self.host_calls.lock().unwrap();
            let entry = calls.entry(label).or_default();
            entry.0 += started.elapsed().as_nanos();
            entry.1 += 1;
        }
    }

    pub fn profile_reset(&self) {
        if self.profile_enabled {
            self.host_calls.lock().unwrap().clear();
        }
    }

    pub fn profile_report(&self) -> Result<()> {
        if !self.profile_enabled {
            return Ok(());
        }
        self.finish()?;
        let events = std::mem::take(&mut *self.profile_events.lock().unwrap());
        let mut groups: BTreeMap<String, (u64, usize)> = BTreeMap::new();
        let mut first_start = None;
        let mut previous_end = None;
        let mut last_end = 0u64;
        let mut largest_gap = 0u64;
        for event in events {
            let mut start = 0u64;
            let mut end = 0u64;
            let start_status = unsafe {
                (self.get_event_profiling_info)(
                    event.handle,
                    0x1282,
                    8,
                    (&mut start as *mut u64).cast(),
                    ptr::null_mut(),
                )
            };
            let end_status = unsafe {
                (self.get_event_profiling_info)(
                    event.handle,
                    0x1283,
                    8,
                    (&mut end as *mut u64).cast(),
                    ptr::null_mut(),
                )
            };
            unsafe { (self.release_event)(event.handle) };
            checked("GetEventProfilingInfo(START)", start_status)?;
            checked("GetEventProfilingInfo(END)", end_status)?;
            first_start.get_or_insert(start);
            if let Some(previous) = previous_end {
                largest_gap = largest_gap.max(start.saturating_sub(previous));
            }
            previous_end = Some(end);
            last_end = end;
            let entry = groups.entry(event.label).or_default();
            entry.0 += end.saturating_sub(start);
            entry.1 += 1;
        }
        let total_ns: u64 = groups.values().map(|entry| entry.0).sum();
        eprintln!(
            "GPU_PROFILE total_ms={:.3} kernels={}",
            total_ns as f64 / 1e6,
            groups.values().map(|entry| entry.1).sum::<usize>()
        );
        if let Some(first) = first_start {
            let span = last_end.saturating_sub(first);
            eprintln!(
                "GPU_PROFILE_SPAN span_ms={:.3} gap_ms={:.3} largest_gap_ms={:.3}",
                span as f64 / 1e6,
                span.saturating_sub(total_ns) as f64 / 1e6,
                largest_gap as f64 / 1e6
            );
        }
        for (label, (ns, count)) in groups {
            eprintln!(
                "GPU_PROFILE_KERNEL label={label} count={count} total_ms={:.3} mean_ms={:.3}",
                ns as f64 / 1e6,
                ns as f64 / (count as f64 * 1e6)
            );
        }
        for (label, (ns, count)) in self.host_calls.lock().unwrap().iter() {
            eprintln!(
                "HOST_CALL label={label} count={count} total_ms={:.3} mean_ms={:.3}",
                *ns as f64 / 1e6,
                *ns as f64 / (*count as f64 * 1e6)
            );
        }
        Ok(())
    }
}

impl Buffer {
    pub fn write(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.len {
            bail!("write {} > buffer {}", bytes.len(), self.len)
        }
        let started = Instant::now();
        let status = unsafe {
            (self.cl.write_buffer)(
                self.cl.queue,
                self.handle,
                1,
                0,
                bytes.len(),
                bytes.as_ptr().cast(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        };
        self.cl.record_host("WriteBuffer", started);
        checked("EnqueueWriteBuffer", status)
    }
    pub fn read(&self, bytes: &mut [u8]) -> Result<()> {
        if bytes.len() > self.len {
            bail!("read {} > buffer {}", bytes.len(), self.len)
        }
        let started = Instant::now();
        let status = unsafe {
            (self.cl.read_buffer)(
                self.cl.queue,
                self.handle,
                1,
                0,
                bytes.len(),
                bytes.as_mut_ptr().cast(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        };
        self.cl.record_host("ReadBuffer", started);
        checked("EnqueueReadBuffer", status)
    }
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            (self.cl.release_mem)(self.handle);
        }
    }
}
impl Drop for Cl {
    fn drop(&mut self) {
        unsafe {
            for event in self.profile_events.lock().unwrap().drain(..) {
                (self.release_event)(event.handle);
            }
            for (_, kernel) in self.kernel_cache.lock().unwrap().drain() {
                (self.release_kernel)(kernel);
            }
            (self.release_program)(self.program);
            (self.release_queue)(self.queue);
            (self.release_context)(self.context);
        }
    }
}

impl Kernel {
    pub fn label(mut self, label: &str) -> Self {
        self.label = label.to_owned();
        self
    }
    pub fn buffer(&self, index: u32, value: &Buffer) -> Result<&Self> {
        let started = Instant::now();
        let status = unsafe {
            (self.cl.set_kernel_arg)(
                self.handle,
                index,
                std::mem::size_of::<Handle>(),
                (&value.handle as *const Handle).cast(),
            )
        };
        self.cl.record_host("SetKernelArg", started);
        checked("SetKernelArg(buffer)", status)?;
        Ok(self)
    }
    pub fn i32(&self, index: u32, value: i32) -> Result<&Self> {
        let started = Instant::now();
        let status = unsafe {
            (self.cl.set_kernel_arg)(self.handle, index, 4, (&value as *const i32).cast())
        };
        self.cl.record_host("SetKernelArg", started);
        checked("SetKernelArg(i32)", status)?;
        Ok(self)
    }
    pub fn f32(&self, index: u32, value: f32) -> Result<&Self> {
        let started = Instant::now();
        let status = unsafe {
            (self.cl.set_kernel_arg)(self.handle, index, 4, (&value as *const f32).cast())
        };
        self.cl.record_host("SetKernelArg", started);
        checked("SetKernelArg(f32)", status)?;
        Ok(self)
    }
    pub fn run(&self, global: usize) -> Result<()> {
        if global == 0 {
            return Ok(());
        }
        let mut event = ptr::null_mut();
        let event_ptr = if self.cl.profile_enabled {
            &mut event
        } else {
            ptr::null_mut()
        };
        let started = Instant::now();
        let status = unsafe {
            (self.cl.enqueue_kernel)(
                self.cl.queue,
                self.handle,
                1,
                ptr::null(),
                &global,
                ptr::null(),
                0,
                ptr::null(),
                event_ptr,
            )
        };
        self.cl.record_host("EnqueueKernel", started);
        checked("EnqueueNDRangeKernel", status)?;
        if self.cl.profile_enabled {
            self.cl.profile_events.lock().unwrap().push(ProfileEvent {
                label: self.label.clone(),
                handle: event,
            });
        }
        Ok(())
    }
}
