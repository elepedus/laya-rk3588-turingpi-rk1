mod api;
mod cl;
mod model;
mod packing;
mod route;
mod router;
use anyhow::{Context, Result};
use cl::Cl;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::{
    env,
    fs::{self, File},
    io::{self, Read},
    path::Path,
    sync::Arc,
    time::Instant,
};
use tiny_http::{Header, Method, Response, Server, StatusCode};

fn bytes<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
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

fn linear_test(cl: &std::sync::Arc<Cl>, model_dir: &Path) -> Result<()> {
    let file = File::open(model_dir.join("model.safetensors"))?;
    let mapping = unsafe { Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mapping)?;
    let tensor = tensors.tensor("encoder.layers.0.attn.Wo.weight")?;
    let shape = tensor.shape();
    anyhow::ensure!(shape == [1024, 1024], "unexpected shape: {shape:?}");
    let input: Vec<f32> = (0..1024)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) / 32.0)
        .collect();
    let x = cl.from_bytes(bytes(&input))?;
    let w = cl.from_bytes(tensor.data())?;
    let y = cl.buffer(1024 * 4)?;
    // The no-bias path still needs a valid buffer argument.
    let bias = cl.from_bytes(&[0, 0])?;
    cl.kernel("linear_h")?
        .buffer(0, &x)?
        .buffer(1, &w)?
        .buffer(2, &bias)?
        .buffer(3, &y)?
        .i32(4, 1)?
        .i32(5, 1024)?
        .i32(6, 1024)?
        .i32(7, 0)?
        .run(1024)?;
    let mut output = vec![0f32; 1024];
    y.read(bytes_mut(&mut output))?;
    let mut worst = (0usize, 0f32);
    for row in 0..1024 {
        let mut expected = 0f32;
        for col in 0..1024 {
            let offset = 2 * (row * 1024 + col);
            let bits = u16::from_le_bytes([tensor.data()[offset], tensor.data()[offset + 1]]);
            expected = input[col].mul_add(f16_to_f32(bits), expected);
        }
        let error = (output[row] - expected).abs();
        if error > worst.1 {
            worst = (row, error);
        }
    }
    println!(
        "Mali Rust FP16-weight linear: max absolute error {:.6} at output {}",
        worst.1, worst.0
    );
    anyhow::ensure!(worst.1 < 0.003, "GPU linear differs from CPU reference");
    Ok(())
}

fn bench_linear_layout(cl: &Arc<Cl>, model_dir: &Path) -> Result<()> {
    let file = File::open(model_dir.join("model.safetensors"))?;
    let mapping = unsafe { Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mapping)?;
    let tensor = tensors.tensor("encoder.layers.0.mlp.Wi.weight")?;
    let shape = tensor.shape();
    anyhow::ensure!(shape.len() == 2, "expected a matrix");
    let (out_dim, in_dim, rows) = (shape[0], shape[1], 48usize);
    anyhow::ensure!(
        out_dim % 8 == 0,
        "output dimension must be divisible by eight"
    );
    let input: Vec<f32> = (0..rows * in_dim)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) / 32.0)
        .collect();
    let x = cl.from_bytes(bytes(&input))?;
    let w = cl.from_bytes(tensor.data())?;
    let pack_started = Instant::now();
    let packed = packing::transpose_fp16_row_major(tensor.data(), out_dim, in_dim)?;
    eprintln!(
        "PACK_TIME ms={:.3}",
        pack_started.elapsed().as_secs_f64() * 1e3
    );
    let wt = cl.from_bytes(&packed)?;
    let bias = cl.from_bytes(&[0, 0])?;
    let outputs = [
        cl.buffer(rows * out_dim * 4)?,
        cl.buffer(rows * out_dim * 4)?,
        cl.buffer(rows * out_dim * 4)?,
    ];
    let variants = [
        (
            "original",
            "linear_h_rows4_vec8",
            &w,
            rows.div_ceil(4) * out_dim,
        ),
        (
            "packed4",
            "linear_h_transposed_rows4_cols4",
            &wt,
            rows.div_ceil(4) * (out_dim / 4),
        ),
        (
            "packed8",
            "linear_h_transposed_rows4_cols8",
            &wt,
            rows.div_ceil(4) * (out_dim / 8),
        ),
    ];
    let mut times = Vec::new();
    for iteration in 0..6 {
        let variant = iteration % variants.len();
        let (label, kernel_name, weight, work_items) = variants[variant];
        let started = Instant::now();
        cl.kernel(kernel_name)?
            .buffer(0, &x)?
            .buffer(1, weight)?
            .buffer(2, &bias)?
            .buffer(3, &outputs[variant])?
            .i32(4, rows as i32)?
            .i32(5, in_dim as i32)?
            .i32(6, out_dim as i32)?
            .i32(7, 0)?
            .run(work_items)?;
        cl.finish()?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        times.push((variant, elapsed));
        eprintln!("MATRIX_TIME layout={label} iteration={iteration} ms={elapsed:.3}");
    }
    let mut expected = vec![0f32; rows * out_dim];
    outputs[0].read(bytes_mut(&mut expected))?;
    for variant in 1..variants.len() {
        let mut actual = vec![0f32; rows * out_dim];
        outputs[variant].read(bytes_mut(&mut actual))?;
        let largest = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (*a - *b).abs())
            .fold(0f32, f32::max);
        println!(
            "MATRIX_COMPARE layout={} rows={rows} in={in_dim} out={out_dim} max_abs={largest:.6}",
            variants[variant].0
        );
    }
    for (variant, (label, _, _, _)) in variants.iter().enumerate() {
        let samples: Vec<f64> = times
            .iter()
            .filter(|(index, _)| *index == variant)
            .map(|(_, ms)| *ms)
            .collect();
        println!("MATRIX_TIMES layout={label} ms={samples:?}");
    }
    Ok(())
}

fn respond(request: tiny_http::Request, status: u16, value: &serde_json::Value) -> Result<()> {
    let body = serde_json::to_string(value)?;
    let header = Header::from_bytes(b"Content-Type", b"application/json")
        .map_err(|_| anyhow::anyhow!("invalid HTTP header"))?;
    request.respond(
        Response::from_string(body)
            .with_status_code(StatusCode(status))
            .with_header(header),
    )?;
    Ok(())
}

fn serve(router: &mut router::Router, bind: &str) -> Result<()> {
    let server = Server::http(bind).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    eprintln!("laya-mali listening on {bind}");
    for mut request in server.incoming_requests() {
        let route = (request.method().clone(), request.url().to_owned());
        match route {
            (Method::Get, path) if path == "/health" => respond(
                request,
                200,
                &serde_json::json!({
                "status":"ok","loaded":router.loaded(),"device":"Mali-G610 OpenCL"}),
            )?,
            (Method::Post, path) if path == "/v1/systemone" => {
                let mut body = Vec::new();
                let result = request
                    .as_reader()
                    .take(1_048_577)
                    .read_to_end(&mut body)
                    .map_err(anyhow::Error::from)
                    .and_then(|_| {
                        anyhow::ensure!(body.len() <= 1_048_576, "request body too large");
                        Ok(())
                    })
                    .and_then(|_| serde_json::from_slice(&body).map_err(anyhow::Error::from))
                    .and_then(|value| router.predict(&value));
                match result {
                    Ok(value) => respond(request, 200, &value)?,
                    Err(error) => respond(
                        request,
                        400,
                        &serde_json::json!({"detail":error.to_string()}),
                    )?,
                }
            }
            _ => respond(request, 404, &serde_json::json!({"detail":"not found"}))?,
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let library = env::var("LAYA_MALI_OPENCL")
        .context("set LAYA_MALI_OPENCL to the staged libOpenCL.so.1")?;
    let cl = Cl::new(&library, include_str!("kernels.cl"))?;
    let mode = env::args().nth(1).unwrap_or_else(|| "selftest".into());
    if mode == "bench-linear" {
        let model_dir = env::var("LAYA_MODEL_DIR")?;
        return bench_linear_layout(&cl, Path::new(&model_dir));
    }
    if mode == "trace" {
        let model_dir = env::var("LAYA_MODEL_DIR")?;
        let reference = env::var("LAYA_ORACLE_DIR")?;
        let input: serde_json::Value =
            serde_json::from_slice(&fs::read(Path::new(&reference).join("input.json"))?)?;
        let ids: Vec<i32> = input["ids"]
            .as_array()
            .context("input.json missing ids")?
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let markers: Vec<usize> = input["markers"]
            .as_array()
            .context("input.json missing markers")?
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let start = Instant::now();
        let model = model::GpuModel::load(cl, Path::new(&model_dir), Some(reference.into()))?;
        eprintln!("GPU model load: {:.3}s", start.elapsed().as_secs_f64());
        let started = Instant::now();
        let final_hidden = model.run(&ids, 0)?;
        println!(
            "GPU model forward: {:.3}s, final hidden first 8: {:?}",
            started.elapsed().as_secs_f64(),
            &final_hidden[..8]
        );
        let (logits, act_logits) = model.decode(&final_hidden, &markers)?;
        println!("GPU logits: {logits:?}; action logits: {act_logits:?}");
        return Ok(());
    }
    if mode == "serve" || mode == "predict" {
        let model_dir = env::var("LAYA_MODEL_DIR")?;
        let start = Instant::now();
        let profiler = Arc::clone(&cl);
        let mut router = router::Router::new(cl, model_dir.into());
        let preload =
            env::var("LAYA_MALI_PRELOAD").unwrap_or_else(|_| "english,multilingual".into());
        router.preload(&preload)?;
        eprintln!(
            "laya-mali ready after {:.3}s",
            start.elapsed().as_secs_f64()
        );
        if mode == "serve" {
            if env::var("LAYA_MALI_WARMUP").as_deref() == Ok("1") {
                let started = Instant::now();
                profiler.warmup()?;
                eprintln!(
                    "laya-mali GPU warmup: {:.3}s",
                    started.elapsed().as_secs_f64()
                );
            }
            let bind = env::var("LAYA_MALI_BIND").unwrap_or_else(|_| "127.0.0.1:8002".into());
            return serve(&mut router, &bind);
        }
        let mut input = Vec::new();
        io::stdin().read_to_end(&mut input)?;
        let request: serde_json::Value = serde_json::from_slice(&input)?;
        profiler.profile_reset();
        let started = Instant::now();
        let response = router.predict(&request)?;
        let inference_ms = started.elapsed().as_secs_f64() * 1e3;
        profiler.profile_report()?;
        if env::var("LAYA_MALI_PROFILE").as_deref() == Ok("1")
            || env::var("LAYA_MALI_TIMING").as_deref() == Ok("1")
        {
            eprintln!("HOST_PROFILE inference_ms={inference_ms:.3}");
        }
        println!("{}", serde_json::to_string(&response)?);
        return Ok(());
    }
    anyhow::ensure!(
        mode == "selftest",
        "mode must be selftest, trace, predict, or serve"
    );
    let a: Vec<f32> = (0..1024).map(|n| n as f32).collect();
    let b: Vec<f32> = (0..1024).map(|n| (2 * n) as f32).collect();
    let x = cl.from_bytes(bytes(&a))?;
    let y = cl.from_bytes(bytes(&b))?;
    let z = cl.buffer(a.len() * 4)?;
    cl.kernel("vector_add")?
        .buffer(0, &x)?
        .buffer(1, &y)?
        .buffer(2, &z)?
        .i32(3, a.len() as i32)?
        .run(a.len())?;
    let mut result = vec![0f32; a.len()];
    z.read(bytes_mut(&mut result))?;
    for (i, value) in result.iter().enumerate() {
        anyhow::ensure!(*value == (3 * i) as f32, "GPU mismatch at {i}: {value}");
    }
    println!("Mali Rust vector add: {} values correct", result.len());
    if let Ok(model_dir) = env::var("LAYA_MODEL_DIR") {
        linear_test(&cl, Path::new(&model_dir))?;
    }
    Ok(())
}
