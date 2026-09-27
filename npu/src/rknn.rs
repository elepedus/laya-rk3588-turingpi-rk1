//! Minimal dynamic RKNN Runtime binding. No Rockchip files are linked into the
//! executable; the staged runtime is loaded from /mnt/warm at execution time.
use anyhow::{bail, Context, Result};
use libloading::Library;
use std::{
    cell::Cell,
    ffi::{c_char, c_void, CStr, CString},
    path::Path,
    ptr,
    time::Instant,
};

type Init = unsafe extern "C" fn(*mut u64, *mut c_void, u32, u32, *mut c_void) -> i32;
type Query = unsafe extern "C" fn(u64, i32, *mut c_void, u32) -> i32;
type InputsSet = unsafe extern "C" fn(u64, u32, *mut RknnInput) -> i32;
type Run = unsafe extern "C" fn(u64, *mut c_void) -> i32;
type SetCoreMask = unsafe extern "C" fn(u64, i32) -> i32;
type OutputsGet = unsafe extern "C" fn(u64, u32, *mut RknnOutput, *mut c_void) -> i32;
type OutputsRelease = unsafe extern "C" fn(u64, u32, *mut RknnOutput) -> i32;
type Destroy = unsafe extern "C" fn(u64) -> i32;
type CreateMem = unsafe extern "C" fn(u64, u32) -> *mut TensorMem;
type DestroyMem = unsafe extern "C" fn(u64, *mut TensorMem) -> i32;
type SetIoMem = unsafe extern "C" fn(u64, *mut TensorMem, *mut TensorAttr) -> i32;
type MemSync = unsafe extern "C" fn(u64, *mut TensorMem, i32) -> i32;

#[repr(C)]
struct RknnInput {
    index: u32,
    buf: *mut c_void,
    size: u32,
    pass_through: u8,
    type_: i32,
    fmt: i32,
}

#[repr(C)]
struct RknnOutput {
    want_float: u8,
    is_prealloc: u8,
    index: u32,
    buf: *mut c_void,
    size: u32,
}

#[repr(C)]
#[derive(Default)]
struct InputOutputNum {
    n_input: u32,
    n_output: u32,
}

#[repr(C)]
struct SdkVersion {
    api: [c_char; 256],
    driver: [c_char; 256],
}

#[repr(C)]
#[derive(Default)]
struct PerfRun {
    run_duration: i64,
}

#[repr(C)]
#[derive(Default)]
struct PerfDetail {
    perf_data: *const c_char,
    data_len: u64,
}

#[repr(C)]
struct TensorAttr {
    index: u32,
    n_dims: u32,
    dims: [u32; 16],
    name: [c_char; 256],
    n_elems: u32,
    size: u32,
    fmt: i32,
    type_: i32,
    qnt_type: i32,
    fl: i8,
    zp: i32,
    scale: f32,
    w_stride: u32,
    size_with_stride: u32,
    pass_through: u8,
    h_stride: u32,
}

#[repr(C)]
struct TensorMem {
    virt_addr: *mut c_void,
    phys_addr: u64,
    fd: i32,
    offset: i32,
    size: u32,
    flags: u32,
    priv_data: *mut c_void,
}

pub struct DeviceMem {
    ptr: *mut TensorMem,
    context: u64,
    destroy: DestroyMem,
}

impl DeviceMem {
    pub fn size(&self) -> u32 {
        unsafe { (*self.ptr).size }
    }

    pub fn write_f16(&mut self, values: &[u16]) -> Result<()> {
        anyhow::ensure!(
            self.size() as usize >= std::mem::size_of_val(values),
            "RKNN memory is too small for FP16 input"
        );
        let address = unsafe { (*self.ptr).virt_addr };
        anyhow::ensure!(!address.is_null(), "RKNN memory has no CPU address");
        let capacity = self.size() as usize / 2;
        let target = unsafe { std::slice::from_raw_parts_mut(address.cast::<u16>(), capacity) };
        target[..values.len()].copy_from_slice(values);
        target[values.len()..].fill(0);
        Ok(())
    }

    pub fn read_f16(&self, elements: usize) -> Result<Vec<u16>> {
        anyhow::ensure!(
            self.size() as usize >= elements * 2,
            "RKNN memory is too small for FP16 output"
        );
        let address = unsafe { (*self.ptr).virt_addr };
        anyhow::ensure!(!address.is_null(), "RKNN memory has no CPU address");
        Ok(unsafe { std::slice::from_raw_parts(address.cast::<u16>(), elements) }.to_vec())
    }

    pub fn read_f32(&self, elements: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            self.size() as usize >= elements * 4,
            "RKNN memory is too small for FP32 output"
        );
        let address = unsafe { (*self.ptr).virt_addr };
        anyhow::ensure!(!address.is_null(), "RKNN memory has no CPU address");
        Ok(unsafe { std::slice::from_raw_parts(address.cast::<f32>(), elements) }.to_vec())
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context, self.ptr);
        }
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u32>,
    pub elements: u32,
    pub size: u32,
    pub format: i32,
    pub type_: i32,
    pub width_stride: u32,
    pub height_stride: u32,
    pub size_with_stride: u32,
    pub pass_through: u8,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct PhaseTiming {
    pub input_ms: f64,
    pub run_ms: f64,
    pub output_ms: f64,
    pub copy_ms: f64,
    pub release_ms: f64,
}

pub struct Rknn {
    _library: Library,
    context: u64,
    query: Query,
    inputs_set: InputsSet,
    run: Run,
    set_core_mask: SetCoreMask,
    outputs_get: OutputsGet,
    outputs_release: OutputsRelease,
    destroy: Destroy,
    create_mem: CreateMem,
    destroy_mem: DestroyMem,
    set_io_mem: SetIoMem,
    mem_sync: MemSync,
    last_phases: Cell<PhaseTiming>,
}

pub enum Input<'a> {
    F32(&'a [f32], i32),
    F16(&'a [u16], i32),
}

fn checked(name: &str, status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        bail!("{name} failed: RKNN status {status}")
    }
}

impl Rknn {
    pub fn load(library_path: &Path, model_path: &Path) -> Result<Self> {
        assert_eq!(std::mem::size_of::<RknnInput>(), 32);
        assert_eq!(std::mem::size_of::<RknnOutput>(), 24);
        assert_eq!(std::mem::size_of::<SdkVersion>(), 512);
        assert_eq!(std::mem::size_of::<TensorAttr>(), 376);
        assert_eq!(std::mem::size_of::<TensorMem>(), 40);
        let library = unsafe { Library::new(library_path) }
            .with_context(|| format!("loading {}", library_path.display()))?;
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                *unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .with_context(|| format!("missing RKNN symbol {}", $name))?
            }};
        }
        let init = symbol!("rknn_init", Init);
        let query = symbol!("rknn_query", Query);
        let inputs_set = symbol!("rknn_inputs_set", InputsSet);
        let run = symbol!("rknn_run", Run);
        let set_core_mask = symbol!("rknn_set_core_mask", SetCoreMask);
        let outputs_get = symbol!("rknn_outputs_get", OutputsGet);
        let outputs_release = symbol!("rknn_outputs_release", OutputsRelease);
        let destroy = symbol!("rknn_destroy", Destroy);
        let create_mem = symbol!("rknn_create_mem", CreateMem);
        let destroy_mem = symbol!("rknn_destroy_mem", DestroyMem);
        let set_io_mem = symbol!("rknn_set_io_mem", SetIoMem);
        let mem_sync = symbol!("rknn_mem_sync", MemSync);

        let path = CString::new(model_path.to_string_lossy().as_bytes())?;
        let mut context = 0u64;
        let flags = if std::env::var_os("LAYA_RKNN_PROFILE").is_some() {
            8
        } else {
            0
        };
        checked("rknn_init", unsafe {
            init(
                &mut context,
                path.as_ptr().cast_mut().cast(),
                0,
                flags,
                ptr::null_mut(),
            )
        })?;
        Ok(Self {
            _library: library,
            context,
            query,
            inputs_set,
            run,
            set_core_mask,
            outputs_get,
            outputs_release,
            destroy,
            create_mem,
            destroy_mem,
            set_io_mem,
            mem_sync,
            last_phases: Cell::new(PhaseTiming::default()),
        })
    }

    pub fn sdk_version(&self) -> Result<(String, String)> {
        let mut value = SdkVersion {
            api: [0; 256],
            driver: [0; 256],
        };
        checked("rknn_query(SDK_VERSION)", unsafe {
            (self.query)(self.context, 5, (&mut value as *mut SdkVersion).cast(), 512)
        })?;
        let api = unsafe { CStr::from_ptr(value.api.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let driver = unsafe { CStr::from_ptr(value.driver.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Ok((api, driver))
    }

    pub fn io_count(&self) -> Result<(u32, u32)> {
        let mut value = InputOutputNum::default();
        checked("rknn_query(IN_OUT_NUM)", unsafe {
            (self.query)(
                self.context,
                0,
                (&mut value as *mut InputOutputNum).cast(),
                std::mem::size_of::<InputOutputNum>() as u32,
            )
        })?;
        Ok((value.n_input, value.n_output))
    }

    pub fn tensor_attr(&self, command: i32, index: u32) -> Result<TensorInfo> {
        let mut attr: TensorAttr = unsafe { std::mem::zeroed() };
        attr.index = index;
        checked("rknn_query(TENSOR_ATTR)", unsafe {
            (self.query)(
                self.context,
                command,
                (&mut attr as *mut TensorAttr).cast(),
                std::mem::size_of::<TensorAttr>() as u32,
            )
        })?;
        anyhow::ensure!(attr.n_dims <= 16, "RKNN returned invalid dimension count");
        Ok(TensorInfo {
            name: unsafe { CStr::from_ptr(attr.name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            dims: attr.dims[..attr.n_dims as usize].to_vec(),
            elements: attr.n_elems,
            size: attr.size,
            format: attr.fmt,
            type_: attr.type_,
            width_stride: attr.w_stride,
            height_stride: attr.h_stride,
            size_with_stride: attr.size_with_stride,
            pass_through: attr.pass_through,
        })
    }

    fn raw_attr(&self, command: i32, index: u32) -> Result<TensorAttr> {
        let mut attr: TensorAttr = unsafe { std::mem::zeroed() };
        attr.index = index;
        checked("rknn_query(TENSOR_ATTR)", unsafe {
            (self.query)(
                self.context,
                command,
                (&mut attr as *mut TensorAttr).cast(),
                std::mem::size_of::<TensorAttr>() as u32,
            )
        })?;
        Ok(attr)
    }

    pub fn create_mem(&self, size: u32) -> Result<DeviceMem> {
        let ptr = unsafe { (self.create_mem)(self.context, size) };
        anyhow::ensure!(!ptr.is_null(), "rknn_create_mem({size}) returned null");
        Ok(DeviceMem {
            ptr,
            context: self.context,
            destroy: self.destroy_mem,
        })
    }

    pub fn sync_to_device(&self, memory: &DeviceMem) -> Result<()> {
        checked("rknn_mem_sync(TO_DEVICE)", unsafe {
            (self.mem_sync)(self.context, memory.ptr, 1)
        })
    }

    pub fn sync_from_device(&self, memory: &DeviceMem) -> Result<()> {
        checked("rknn_mem_sync(FROM_DEVICE)", unsafe {
            (self.mem_sync)(self.context, memory.ptr, 2)
        })
    }

    pub fn bind_input_mem(&self, index: u32, memory: &DeviceMem) -> Result<()> {
        let mut attr = self.raw_attr(1, index)?;
        anyhow::ensure!(
            memory.size() >= attr.size_with_stride,
            "device memory too small for input {index}"
        );
        if std::env::var_os("LAYA_RKNN_PASS_THROUGH").is_some() {
            attr.pass_through = 1;
        }
        checked("rknn_set_io_mem(input)", unsafe {
            (self.set_io_mem)(self.context, memory.ptr, &mut attr)
        })
    }

    pub fn bind_output_mem(&self, index: u32, memory: &DeviceMem) -> Result<()> {
        let mut attr = self.raw_attr(2, index)?;
        if std::env::var_os("LAYA_RKNN_OUTPUT_F32").is_some() {
            attr.type_ = 0;
        }
        let required = if attr.type_ == 0 {
            attr.n_elems * 4
        } else {
            attr.size_with_stride
        };
        anyhow::ensure!(
            memory.size() >= required,
            "device memory too small for output {index}"
        );
        checked("rknn_set_io_mem(output)", unsafe {
            (self.set_io_mem)(self.context, memory.ptr, &mut attr)
        })
    }

    pub fn set_core_mask(&self, mask: i32) -> Result<()> {
        checked("rknn_set_core_mask", unsafe {
            (self.set_core_mask)(self.context, mask)
        })
    }

    fn infer_typed<T: Copy>(
        &self,
        inputs: &[Input<'_>],
        want_float: u8,
        first_index: u32,
    ) -> Result<Vec<T>> {
        let mut descriptors: Vec<RknnInput> = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let (buf, size, type_, fmt) = match input {
                    Input::F32(values, fmt) => (
                        values.as_ptr().cast_mut().cast(),
                        std::mem::size_of_val(*values),
                        0,
                        *fmt,
                    ),
                    Input::F16(values, fmt) => (
                        values.as_ptr().cast_mut().cast(),
                        std::mem::size_of_val(*values),
                        1,
                        *fmt,
                    ),
                };
                Ok(RknnInput {
                    index: first_index + u32::try_from(index)?,
                    buf,
                    size: u32::try_from(size)?,
                    pass_through: 0,
                    type_,
                    fmt,
                })
            })
            .collect::<Result<_>>()?;
        let phase_start = Instant::now();
        checked("rknn_inputs_set", unsafe {
            (self.inputs_set)(
                self.context,
                u32::try_from(descriptors.len())?,
                descriptors.as_mut_ptr(),
            )
        })?;
        let input_ms = phase_start.elapsed().as_secs_f64() * 1000.0;
        let run_start = Instant::now();
        checked("rknn_run", unsafe {
            (self.run)(self.context, ptr::null_mut())
        })?;
        let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
        let mut output = RknnOutput {
            want_float,
            is_prealloc: 0,
            index: 0,
            buf: ptr::null_mut(),
            size: 0,
        };
        let output_start = Instant::now();
        checked("rknn_outputs_get", unsafe {
            (self.outputs_get)(self.context, 1, &mut output, ptr::null_mut())
        })?;
        let output_ms = output_start.elapsed().as_secs_f64() * 1000.0;
        let copy_start = Instant::now();
        let item_size = std::mem::size_of::<T>();
        let data = if !output.buf.is_null() && output.size as usize % item_size == 0 {
            unsafe {
                std::slice::from_raw_parts(output.buf.cast::<T>(), output.size as usize / item_size)
            }
            .to_vec()
        } else {
            unsafe { (self.outputs_release)(self.context, 1, &mut output) };
            bail!("RKNN output pointer or byte length is invalid")
        };
        let copy_ms = copy_start.elapsed().as_secs_f64() * 1000.0;
        let release_start = Instant::now();
        checked("rknn_outputs_release", unsafe {
            (self.outputs_release)(self.context, 1, &mut output)
        })?;
        let release_ms = release_start.elapsed().as_secs_f64() * 1000.0;
        self.last_phases.set(PhaseTiming {
            input_ms,
            run_ms,
            output_ms,
            copy_ms,
            release_ms,
        });
        Ok(data)
    }

    pub fn last_phases(&self) -> PhaseTiming {
        self.last_phases.get()
    }

    pub fn infer_f32(&self, inputs: &[&[f32]], formats: &[i32]) -> Result<Vec<f32>> {
        anyhow::ensure!(inputs.len() == formats.len(), "each input needs a format");
        let buffers: Vec<_> = inputs
            .iter()
            .zip(formats)
            .map(|(data, &fmt)| Input::F32(*data, fmt))
            .collect();
        self.infer_typed(&buffers, 1, 0)
    }

    pub fn infer_mixed_f32(&self, inputs: &[Input<'_>]) -> Result<Vec<f32>> {
        self.infer_typed(inputs, 1, 0)
    }

    pub fn infer_f16(&self, inputs: &[Input<'_>]) -> Result<Vec<u16>> {
        self.infer_typed(inputs, 0, 0)
    }

    pub fn infer_with_bound_input0_f32(&self, mask: &[f32], format: i32) -> Result<Vec<f32>> {
        self.infer_typed(&[Input::F32(mask, format)], 1, 1)
    }

    pub fn run_bound(&self, inputs: &[Input<'_>]) -> Result<()> {
        let mut descriptors: Vec<RknnInput> = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let (buf, size, type_, fmt) = match input {
                    Input::F32(values, fmt) => (
                        values.as_ptr().cast_mut().cast(),
                        std::mem::size_of_val(*values),
                        0,
                        *fmt,
                    ),
                    Input::F16(values, fmt) => (
                        values.as_ptr().cast_mut().cast(),
                        std::mem::size_of_val(*values),
                        1,
                        *fmt,
                    ),
                };
                Ok(RknnInput {
                    index: u32::try_from(index)?,
                    buf,
                    size: u32::try_from(size)?,
                    pass_through: 0,
                    type_,
                    fmt,
                })
            })
            .collect::<Result<_>>()?;
        checked("rknn_inputs_set", unsafe {
            (self.inputs_set)(
                self.context,
                u32::try_from(descriptors.len())?,
                descriptors.as_mut_ptr(),
            )
        })?;
        checked("rknn_run", unsafe {
            (self.run)(self.context, ptr::null_mut())
        })
    }

    pub fn last_duration_us(&self) -> Result<i64> {
        let mut value = PerfRun::default();
        checked("rknn_query(PERF_RUN)", unsafe {
            (self.query)(self.context, 4, (&mut value as *mut PerfRun).cast(), 8)
        })?;
        Ok(value.run_duration)
    }

    pub fn perf_detail(&self) -> Result<String> {
        let mut value = PerfDetail::default();
        checked("rknn_query(PERF_DETAIL)", unsafe {
            (self.query)(self.context, 3, (&mut value as *mut PerfDetail).cast(), 16)
        })?;
        if value.perf_data.is_null() {
            bail!("RKNN performance detail pointer is null")
        }
        let len = usize::try_from(value.data_len)?;
        let bytes = unsafe { std::slice::from_raw_parts(value.perf_data.cast::<u8>(), len) };
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

impl Drop for Rknn {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context);
        }
    }
}
