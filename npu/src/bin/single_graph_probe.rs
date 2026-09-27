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
    let errors = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f16_to_f32(a) - b).abs());
    let maximum = errors.clone().fold(0f32, f32::max);
    let mean = errors.map(|value| value as f64).sum::<f64>() / expected.len() as f64;
    (maximum, mean)
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let baseline_path = args.next().context("baseline RKNN graph")?;
    let candidate_path = args.next().context("candidate RKNN graph")?;
    let hidden = read_u16(&args.next().context("hidden FP16 file")?)?;
    let mask = read_f32(&args.next().context("mask FP32 file")?)?;
    let expected = read_f32(&args.next().context("expected FP32 file")?)?;
    anyhow::ensure!(
        hidden.len() == expected.len(),
        "hidden and expected size differ"
    );
    let library = env::var("LAYA_RKNNRT")?;
    let baseline = Rknn::load(Path::new(&library), Path::new(&baseline_path))?;
    let candidate = Rknn::load(Path::new(&library), Path::new(&candidate_path))?;
    baseline.set_core_mask(7)?;
    candidate.set_core_mask(7)?;
    for round in 0..4 {
        let order = if round % 2 == 0 {
            ["baseline", "candidate"]
        } else {
            ["candidate", "baseline"]
        };
        let mut first_output: Option<(String, Vec<u16>)> = None;
        for variant in order {
            let graph = if variant == "baseline" {
                &baseline
            } else {
                &candidate
            };
            let started = Instant::now();
            let actual = graph.infer_f16(&[Input::F16(&hidden, 3), Input::F32(&mask, 1)])?;
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            anyhow::ensure!(actual.len() == expected.len(), "output size mismatch");
            let (max_abs, mean_abs) = compare(&actual, &expected);
            println!("round={round} variant={variant} wall_ms={wall_ms:.3} npu_us={} max_abs={max_abs:.6} mean_abs={mean_abs:.6}",
                graph.last_duration_us()?);
            if let Some((first_name, first)) = &first_output {
                let mismatches = first.iter().zip(&actual).filter(|(a, b)| a != b).count();
                let max_delta = first
                    .iter()
                    .zip(&actual)
                    .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                    .fold(0f32, f32::max);
                println!("round={round} compare={first_name}_vs_{variant} bit_mismatches={mismatches} max_abs={max_delta:.6}");
            } else {
                first_output = Some((variant.to_owned(), actual));
            }
        }
    }
    Ok(())
}
