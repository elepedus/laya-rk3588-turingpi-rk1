#[path = "../../mali/src/api.rs"]
mod api;
#[path = "../../mali/src/cl.rs"]
mod cl;
mod encoder;
mod full;
mod full_service;
#[path = "../../mali/src/model.rs"]
mod model;
#[path = "../../mali/src/packing.rs"]
mod packing;
mod rknn;
#[path = "../../mali/src/route.rs"]
mod route;
mod service;

use anyhow::{Context, Result};
use api::InferenceModel;
use rknn::{Input, Rknn};
use serde_json::Value;
use std::{env, fs, path::Path, rc::Rc, time::Instant};

pub(crate) fn bytes<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

pub(crate) fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values))
    }
}

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    let raw = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut significand = fraction;
            let mut e = -14i32;
            while significand & 0x400 == 0 {
                significand <<= 1;
                e -= 1;
            }
            sign | (((e + 127) as u32) << 23) | ((significand & 0x3ff) << 13)
        }
        31 => sign | 0x7f800000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(raw)
}

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    anyhow::ensure!(bytes.len() % 4 == 0, "{path} is not an FP32 array");
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn main() -> Result<()> {
    let mut arguments = env::args().skip(1);
    let mode = arguments
        .next()
        .context("mode must be serve, probe, probe-binary, or run")?;
    if mode == "serve" {
        return service::run();
    }
    if mode == "serve-full" {
        return full_service::run();
    }
    if mode == "probe-attrs" {
        let path = arguments.next().context("missing RKNN graph path")?;
        let library = env::var("LAYA_RKNNRT")?;
        let graph = Rknn::load(Path::new(&library), Path::new(&path))?;
        let (inputs, outputs) = graph.io_count()?;
        println!("graph={path} inputs={inputs} outputs={outputs}");
        for index in 0..inputs {
            for (command, label) in [(1, "input"), (8, "native_input"), (10, "native_nhwc_input")] {
                println!("{label}[{index}]={:?}", graph.tensor_attr(command, index));
            }
        }
        for index in 0..outputs {
            for (command, label) in [
                (2, "output"),
                (9, "native_output"),
                (11, "native_nhwc_output"),
            ] {
                println!("{label}[{index}]={:?}", graph.tensor_attr(command, index));
            }
        }
        return Ok(());
    }
    if mode == "probe-bound-mask" || mode == "probe-cross-mask" {
        let owner_path = arguments.next().context("missing RKNN graph path")?;
        let target_path = if mode == "probe-cross-mask" {
            arguments.next().context("missing target RKNN graph path")?
        } else {
            owner_path.clone()
        };
        let input = read_f32(&arguments.next().context("missing input FP32 file")?)?;
        let mask = read_f32(&arguments.next().context("missing mask FP32 file")?)?;
        let expected = read_f32(&arguments.next().context("missing expected FP32 file")?)?;
        let library = env::var("LAYA_RKNNRT")?;
        let owner = Rknn::load(Path::new(&library), Path::new(&owner_path))?;
        let target = if mode == "probe-cross-mask" {
            Some(Rknn::load(Path::new(&library), Path::new(&target_path))?)
        } else {
            None
        };
        let graph = target.as_ref().unwrap_or(&owner);
        graph.set_core_mask(7)?;
        anyhow::ensure!(graph.io_count()? == (2, 1), "expected a two-input graph");
        let attr = graph.tensor_attr(1, 1)?;
        let mask_half: Vec<u16> = mask
            .iter()
            .map(|&value| if value < 0.0 { 0xf0e2 } else { 0 })
            .collect();
        anyhow::ensure!(
            mask_half.len() as u32 == attr.elements,
            "mask has {} elements, graph needs {}",
            mask_half.len(),
            attr.elements
        );
        let mut memory = owner.create_mem(attr.size_with_stride)?;
        memory.write_f16(&mask_half)?;
        owner.sync_to_device(&memory)?;
        graph.bind_input_mem(1, &memory)?;
        for iteration in 0..5 {
            let started = Instant::now();
            let output = graph.infer_f32(&[&input], &[3])?;
            anyhow::ensure!(output.len() == expected.len(), "wrong RKNN output length");
            let maximum = output
                .iter()
                .zip(&expected)
                .map(|(actual, reference)| (actual - reference).abs())
                .fold(0f32, f32::max);
            let phases = graph.last_phases();
            println!("run={iteration} wall_ms={:.3} npu_us={} max_abs={maximum:.6} input_ms={:.3} output_ms={:.3} copy_ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0,
                graph.last_duration_us()?, phases.input_ms, phases.output_ms,
                phases.copy_ms);
        }
        return Ok(());
    }
    if mode == "probe-cross-activation" {
        let first_path = arguments.next().context("missing first RKNN layer")?;
        let second_path = arguments.next().context("missing second RKNN layer")?;
        let hidden = read_f32(&arguments.next().context("missing first input")?)?;
        let mask = read_f32(&arguments.next().context("missing second-layer mask")?)?;
        let first_reference = read_f32(&arguments.next().context("missing first oracle")?)?;
        let second_reference = read_f32(&arguments.next().context("missing second oracle")?)?;
        let library = env::var("LAYA_RKNNRT")?;
        let first = Rknn::load(Path::new(&library), Path::new(&first_path))?;
        let second = Rknn::load(Path::new(&library), Path::new(&second_path))?;
        first.set_core_mask(7)?;
        second.set_core_mask(7)?;
        let output_attr = first.tensor_attr(2, 0)?;
        let input_attr = second.tensor_attr(1, 0)?;
        anyhow::ensure!(
            output_attr.elements == input_attr.elements
                && output_attr.type_ == 1
                && input_attr.type_ == 1,
            "adjacent graph activations must match as FP16"
        );
        let output_f32 = env::var_os("LAYA_RKNN_OUTPUT_F32").is_some();
        let output_size = if output_f32 {
            output_attr.elements * 4
        } else {
            output_attr.size_with_stride
        };
        let memory = first.create_mem(output_size.max(input_attr.size_with_stride))?;
        first.bind_output_mem(0, &memory)?;
        second.bind_input_mem(0, &memory)?;
        let first_mask = vec![0f32; mask.len()];
        for iteration in 0..3 {
            let first_started = Instant::now();
            first.run_bound(&[Input::F32(&hidden, 3), Input::F32(&first_mask, 1)])?;
            let first_ms = first_started.elapsed().as_secs_f64() * 1000.0;
            if iteration == 0 {
                first.sync_from_device(&memory)?;
                if output_f32 {
                    let actual = memory.read_f32(output_attr.elements as usize)?;
                    let max = actual
                        .iter()
                        .zip(&first_reference)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let nonzero = actual.iter().filter(|&&value| value != 0.0).count();
                    println!(
                        "first_layer_max_abs={max:.6} nonzero={nonzero} first={:?}",
                        &actual[..actual.len().min(8)]
                    );
                } else {
                    let actual = memory.read_f16(output_attr.elements as usize)?;
                    let max = actual
                        .iter()
                        .zip(&first_reference)
                        .map(|(a, b)| (f16_to_f32(*a) - b).abs())
                        .fold(0f32, f32::max);
                    let nonzero = actual.iter().filter(|&&value| value & 0x7fff != 0).count();
                    println!(
                        "first_layer_max_abs={max:.6} nonzero={nonzero} first={:?}",
                        &actual[..actual.len().min(8)]
                    );
                }
            }
            if output_f32 {
                continue;
            }
            let second_started = Instant::now();
            let output = second.infer_with_bound_input0_f32(&mask, 1)?;
            anyhow::ensure!(
                output.len() == second_reference.len(),
                "second output length mismatch"
            );
            let second_ms = second_started.elapsed().as_secs_f64() * 1000.0;
            let max = output
                .iter()
                .zip(&second_reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("run={iteration} first_ms={first_ms:.3} second_ms={second_ms:.3} second_max_abs={max:.6} second_input_ms={:.3}",
                second.last_phases().input_ms);
        }
        return Ok(());
    }
    if mode == "trace-full" {
        let name = arguments.next().context("missing checkpoint name")?;
        let directory = arguments.next().context("missing checkpoint directory")?;
        let oracle_dir = arguments.next().context("missing oracle directory")?;
        let source: Value =
            serde_json::from_slice(&fs::read(Path::new(&oracle_dir).join("input.json"))?)?;
        let ids: Vec<i32> = source["ids"]
            .as_array()
            .context("missing oracle ids")?
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        let markers: Vec<usize> = source["markers"]
            .as_array()
            .context("missing markers")?
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect();
        let qtype = source["qtype"].as_i64().unwrap_or(0) as i32;
        let runtime = env::var("LAYA_RKNNRT")?;
        let graph_root = env::var("LAYA_RKNN_GRAPH_ROOT")?;
        let model = Rc::new(full::NpuModel::load(
            &name,
            Path::new(&directory),
            Path::new(&graph_root),
            Path::new(&runtime),
        )?);
        let started = Instant::now();
        let hidden = model.run(&ids, qtype)?;
        let reference = read_f32(&Path::new(&oracle_dir).join("head_01.f32").to_string_lossy())?;
        anyhow::ensure!(hidden.len() >= reference.len(), "NPU head output too short");
        let (logits, act) = model.decode(&hidden, &markers)?;
        let reference_logits =
            read_f32(&Path::new(&oracle_dir).join("logits.f32").to_string_lossy())?;
        let reference_action = read_f32(
            &Path::new(&oracle_dir)
                .join("act_logits.f32")
                .to_string_lossy(),
        )?;
        let max_hidden = hidden
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mean_hidden = hidden
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / reference.len() as f64;
        let max_logit = logits
            .iter()
            .zip(&reference_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let max_action = act
            .iter()
            .zip(&reference_action)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("TRACE_FULL name={name} valid={} wall_ms={:.3} hidden_max_abs={max_hidden:.6} hidden_mean_abs={mean_hidden:.6} logits={logits:?} max_logit_abs={max_logit:.6} action={act:?} max_action_abs={max_action:.6}",
            ids.len(), started.elapsed().as_secs_f64() * 1000.0);
        return Ok(());
    }
    if mode == "probe-chain" {
        let library = env::var("LAYA_RKNNRT")?;
        let directory = arguments.next().context("missing NPU graph directory")?;
        let input = read_f32(&arguments.next().context("missing embedding")?)?;
        let expected = read_f32(&arguments.next().context("missing oracle layer output")?)?;
        let count: usize = arguments.next().context("missing layer count")?.parse()?;
        let valid: usize = arguments
            .next()
            .context("missing valid token count")?
            .parse()?;
        let hidden: usize = arguments
            .next()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(1024);
        let encoder = encoder::NpuEncoder::load(
            Path::new(&library),
            Path::new(&directory),
            hidden,
            64,
            count,
        )?;
        let started = Instant::now();
        let actual = encoder.infer(&input, valid)?;
        anyhow::ensure!(
            expected.len() == valid * hidden,
            "oracle output has wrong shape"
        );
        let errors: Vec<f32> = actual[..expected.len()]
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .collect();
        println!(
            "chain_layers={count} wall_ms={:.3} max_abs={:.6} mean_abs={:.6}",
            started.elapsed().as_secs_f64() * 1000.0,
            errors.iter().copied().fold(0f32, f32::max),
            errors.iter().map(|&e| e as f64).sum::<f64>() / errors.len() as f64
        );
        if let Ok(path) = env::var("LAYA_RKNN_CHAIN_OUTPUT") {
            fs::write(path, bytes(&actual))?;
        }
        return Ok(());
    }
    anyhow::ensure!(mode == "probe" || mode == "probe-binary" || mode == "run",
        "usage: laya-rknpu probe MODEL.rknn ORACLE.json | probe-binary MODEL.rknn INPUT.f32 EXPECTED.f32 | run MODEL.rknn INPUT.f32 OUTPUT.f32");
    let model_path = arguments.next().context("missing RKNN model path")?;
    let library = env::var("LAYA_RKNNRT").context("set LAYA_RKNNRT to the staged librknnrt.so")?;
    let mut output_path = None;
    let (input, expected): (Vec<f32>, Vec<f32>) = if mode == "probe" {
        let probe: Value = serde_json::from_slice(&fs::read(
            arguments.next().context("missing oracle JSON path")?,
        )?)?;
        (
            probe["input"]
                .as_array()
                .context("missing input")?
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect(),
            probe["expected"]
                .as_array()
                .context("missing expected")?
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect(),
        )
    } else if mode == "probe-binary" {
        (
            read_f32(&arguments.next().context("missing input binary")?)?,
            read_f32(&arguments.next().context("missing expected binary")?)?,
        )
    } else {
        let input = read_f32(&arguments.next().context("missing input binary")?)?;
        output_path = Some(arguments.next().context("missing output binary")?);
        (input, Vec::new())
    };
    let start = Instant::now();
    let runtime = Rknn::load(Path::new(&library), Path::new(&model_path))?;
    if let Ok(mask) = env::var("LAYA_RKNN_CORE_MASK") {
        runtime.set_core_mask(mask.parse()?)?;
    }
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (api, driver) = runtime.sdk_version()?;
    let (inputs, outputs) = runtime.io_count()?;
    let format: i32 = env::var("LAYA_RKNN_INPUT_FORMAT")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(3);
    println!(
        "RKNN api={api} driver={driver} inputs={inputs} outputs={outputs} load_ms={load_ms:.3}"
    );
    let mask = env::var("LAYA_RKNN_MASK_FILE")
        .ok()
        .map(|path| read_f32(&path))
        .transpose()?;
    let extra = env::var("LAYA_RKNN_EXTRA_FILE")
        .ok()
        .map(|path| read_f32(&path))
        .transpose()?;
    anyhow::ensure!(
        inputs == 1 + u32::from(mask.is_some()) + u32::from(extra.is_some()) && outputs == 1,
        "RKNN graph has {inputs} inputs; set LAYA_RKNN_MASK_FILE and LAYA_RKNN_EXTRA_FILE as needed"
    );
    for index in 0..if mode == "run" { 1 } else { 5 } {
        let started = Instant::now();
        let mut buffers: Vec<&[f32]> = vec![&input];
        let mut formats = vec![format];
        if let Some(mask) = &mask {
            buffers.push(mask);
            formats.push(
                env::var("LAYA_RKNN_MASK_FORMAT")
                    .ok()
                    .map(|s| s.parse())
                    .transpose()?
                    .unwrap_or(1),
            );
        }
        if let Some(extra) = &extra {
            buffers.push(extra);
            formats.push(
                env::var("LAYA_RKNN_EXTRA_FORMAT")
                    .ok()
                    .map(|s| s.parse())
                    .transpose()?
                    .unwrap_or(3),
            );
        }
        let actual = runtime.infer_f32(&buffers, &formats)?;
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        anyhow::ensure!(
            mode == "run" || actual.len() == expected.len(),
            "output length {} != expected {}",
            actual.len(),
            expected.len()
        );
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_abs = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / actual.len() as f64;
        println!("run={index} wall_ms={wall_ms:.3} npu_us={} max_abs={max_abs:.6} mean_abs={mean_abs:.6} first={:?}",
            runtime.last_duration_us()?, &actual[..actual.len().min(4)]);
        if mode == "probe" {
            anyhow::ensure!(max_abs < 0.005, "probe output differs from the CPU oracle");
        }
        if let Some(path) = &output_path {
            let bytes = unsafe {
                std::slice::from_raw_parts(actual.as_ptr().cast::<u8>(), actual.len() * 4)
            };
            fs::write(path, bytes)?;
        }
        if env::var_os("LAYA_RKNN_PROFILE").is_some() {
            println!("{}", runtime.perf_detail()?);
        }
    }
    Ok(())
}
