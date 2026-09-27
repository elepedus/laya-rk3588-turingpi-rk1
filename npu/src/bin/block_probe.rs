#[path = "../rknn.rs"]
mod rknn;

use anyhow::{Context, Result};
use rknn::{Input, Rknn};
use std::{env, fs, path::Path, time::Instant};

fn read_u16(path: &str) -> Result<Vec<u16>> {
    let bytes = fs::read(path)?;
    anyhow::ensure!(bytes.len() % 2 == 0, "invalid FP16 input");
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    anyhow::ensure!(bytes.len() % 4 == 0, "invalid FP32 input");
    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect())
}

fn f16_to_f32(bits: u16) -> f32 {
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

fn compare(actual: &[u16], expected: &[f32]) -> (f32, f64) {
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f16_to_f32(a) - b).abs())
        .fold(0f32, f32::max);
    let mean = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f16_to_f32(a) - b).abs() as f64)
        .sum::<f64>()
        / expected.len() as f64;
    (maximum, mean)
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let fused_path = args.next().context("fused RKNN graph")?;
    let start_layer: usize = args.next().context("start layer")?.parse()?;
    let count: usize = args.next().context("layer count")?.parse()?;
    let separate_root = args.next().context("individual RKNN graph root")?;
    let hidden = read_u16(&args.next().context("input FP16 file")?)?;
    let full_mask = read_f32(&args.next().context("full mask FP32 file")?)?;
    let local_mask = read_f32(&args.next().context("local mask FP32 file")?)?;
    let expected = read_f32(&args.next().context("expected FP32 file")?)?;
    anyhow::ensure!(
        hidden.len() == expected.len(),
        "input and output shapes differ"
    );
    let library = env::var("LAYA_RKNNRT")?;
    let fused = Rknn::load(Path::new(&library), Path::new(&fused_path))?;
    let root = Path::new(&separate_root);
    let grouped = root
        .join(format!(
            "encoder_{start_layer:02}_{:02}.rknn",
            start_layer + 6
        ))
        .is_file();
    let mut separate = Vec::new();
    if grouped {
        let mut start = start_layer;
        while start < start_layer + count {
            let remaining = start_layer + count - start;
            let size = if remaining <= 8 { remaining } else { 7 };
            let end = start + size - 1;
            separate.push(Rknn::load(
                Path::new(&library),
                &root.join(format!("encoder_{start:02}_{end:02}.rknn")),
            )?);
            start = end + 1;
        }
    } else {
        for index in start_layer..start_layer + count {
            let path = root.join(format!("layer{index:02}/encoder_{index:02}.rknn"));
            separate.push(Rknn::load(Path::new(&library), &path)?);
        }
    }
    fused.set_core_mask(7)?;
    for graph in &separate {
        graph.set_core_mask(7)?;
    }
    for round in 0..4 {
        let order = if round % 2 == 0 {
            ["fused", "split"]
        } else {
            ["split", "fused"]
        };
        let mut first_output: Option<(String, Vec<u16>)> = None;
        for variant in order {
            let started = Instant::now();
            let (actual, npu_us) = if variant == "fused" {
                let output = fused.infer_f16(&[
                    Input::F16(&hidden, 3),
                    Input::F32(&full_mask, 1),
                    Input::F32(&local_mask, 1),
                ])?;
                (output, fused.last_duration_us()?)
            } else {
                let mut current = hidden.clone();
                let mut device_us = 0i64;
                for (offset, graph) in separate.iter().enumerate() {
                    current = if grouped {
                        graph.infer_f16(&[
                            Input::F16(&current, 3),
                            Input::F32(&full_mask, 1),
                            Input::F32(&local_mask, 1),
                        ])?
                    } else {
                        let attention = if (start_layer + offset) % 3 == 0 {
                            &full_mask
                        } else {
                            &local_mask
                        };
                        graph.infer_f16(&[Input::F16(&current, 3), Input::F32(attention, 1)])?
                    };
                    device_us += graph.last_duration_us()?;
                }
                (current, device_us)
            };
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            anyhow::ensure!(actual.len() == expected.len(), "RKNN output size mismatch");
            let (max_abs, mean_abs) = compare(&actual, &expected);
            println!("round={round} variant={variant} wall_ms={wall_ms:.3} npu_us={npu_us} max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
            if let Some((first_name, first)) = &first_output {
                let mismatches = first.iter().zip(&actual).filter(|(a, b)| a != b).count();
                let maximum = first
                    .iter()
                    .zip(&actual)
                    .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                    .fold(0f32, f32::max);
                println!("round={round} compare={first_name}_vs_{variant} bit_mismatches={mismatches} max_abs={maximum:.6}");
            } else {
                first_output = Some((variant.to_owned(), actual));
            }
        }
    }
    Ok(())
}
