#[path = "../rknn.rs"]
mod rknn;

use anyhow::{Context, Result};
use rknn::{Input, Rknn};
use std::{env, fs, path::Path, time::Instant};

fn read_f16(path: &str) -> Result<Vec<u16>> {
    let bytes = fs::read(path)?;
    anyhow::ensure!(bytes.len() % 2 == 0, "invalid FP16 file {path}");
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    anyhow::ensure!(bytes.len() % 4 == 0, "invalid FP32 file {path}");
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

fn mask(length: usize, valid: usize, offset: isize) -> Vec<f32> {
    let mut result = vec![0f32; length * length];
    for query in 0..length {
        for key in 0..length {
            let global_key = offset + key as isize;
            if global_key < 0 || global_key >= valid as isize || query.abs_diff(key) > 64 {
                result[query * length + key] = -10000.0;
            }
        }
    }
    result
}

fn run_full(
    graph: &Rknn,
    input: &[u16],
    sequence: usize,
    dim: usize,
) -> Result<(Vec<u16>, f64, i64)> {
    let attention = mask(sequence, sequence, 0);
    let started = Instant::now();
    let output = graph.infer_f16(&[Input::F16(input, 3), Input::F32(&attention, 1)])?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    anyhow::ensure!(
        output.len() == sequence * dim,
        "full model output length mismatch"
    );
    Ok((output, wall_ms, graph.last_duration_us()?))
}

fn run_tiled(
    graph: &Rknn,
    input: &[u16],
    sequence: usize,
    dim: usize,
    window: usize,
) -> Result<(Vec<u16>, f64, i64)> {
    anyhow::ensure!(window > 128 && window <= sequence, "invalid window size");
    let core = window - 128;
    let mut output = vec![0u16; sequence * dim];
    let mut device_us = 0i64;
    let started = Instant::now();
    for tile in 0..sequence.div_ceil(core) {
        let start = (tile * core) as isize - 64;
        let mut buffer = vec![0u16; window * dim];
        for row in 0..window {
            let global = start + row as isize;
            if (0..sequence as isize).contains(&global) {
                let source = global as usize * dim;
                buffer[row * dim..(row + 1) * dim].copy_from_slice(&input[source..source + dim]);
            }
        }
        let attention = mask(window, sequence, start);
        let local = graph.infer_f16(&[Input::F16(&buffer, 3), Input::F32(&attention, 1)])?;
        anyhow::ensure!(local.len() == window * dim, "window output length mismatch");
        device_us += graph.last_duration_us()?;
        for row in 0..core.min(sequence - tile * core) {
            let destination = (tile * core + row) * dim;
            let source = (row + 64) * dim;
            output[destination..destination + dim].copy_from_slice(&local[source..source + dim]);
        }
    }
    Ok((output, started.elapsed().as_secs_f64() * 1000.0, device_us))
}

fn error(actual: &[u16], expected: &[f32]) -> (f32, f64) {
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
    let full_path = args.next().context("full RKNN graph")?;
    let tiled_path = args.next().context("window RKNN graph")?;
    let input = read_f16(&args.next().context("input FP16 file")?)?;
    let expected = read_f32(&args.next().context("expected FP32 file")?)?;
    let sequence: usize = args.next().context("sequence length")?.parse()?;
    let dim: usize = args.next().context("hidden dimension")?.parse()?;
    let window: usize = args.next().context("window length")?.parse()?;
    anyhow::ensure!(
        input.len() == expected.len() && input.len() == sequence * dim,
        "input and oracle shapes differ"
    );
    let library = env::var("LAYA_RKNNRT")?;
    let full = Rknn::load(Path::new(&library), Path::new(&full_path))?;
    let tiled = Rknn::load(Path::new(&library), Path::new(&tiled_path))?;
    full.set_core_mask(7)?;
    tiled.set_core_mask(7)?;
    for round in 0..3 {
        let order = if round % 2 == 0 {
            ["full", "tiled"]
        } else {
            ["tiled", "full"]
        };
        for variant in order {
            let (actual, wall_ms, device_us) = if variant == "full" {
                run_full(&full, &input, sequence, dim)?
            } else {
                run_tiled(&tiled, &input, sequence, dim, window)?
            };
            let (max_abs, mean_abs) = error(&actual, &expected);
            println!("round={round} variant={variant} window={window} wall_ms={wall_ms:.3} npu_us={device_us} max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
        }
    }
    Ok(())
}
