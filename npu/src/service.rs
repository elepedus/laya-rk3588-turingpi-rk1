use crate::{api::Api, cl::Cl, encoder::NpuEncoder, model::GpuModel, route};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    env, fs,
    io::Read,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
    time::Instant,
};
use tiny_http::{Header, Method, Response, Server, StatusCode};

struct Router {
    cl: Arc<Cl>,
    base_dir: PathBuf,
    npu: HashMap<&'static str, Rc<NpuEncoder>>,
    loaded: HashMap<&'static str, Api>,
}

impl Router {
    fn new(cl: Arc<Cl>, base_dir: PathBuf) -> Self {
        Self {
            cl,
            base_dir,
            npu: HashMap::new(),
            loaded: HashMap::new(),
        }
    }

    fn load(&mut self, name: &'static str) -> Result<&Api> {
        if !self.loaded.contains_key(name) {
            let directory = if name == "english" {
                self.base_dir.clone()
            } else {
                self.base_dir.join(name)
            };
            let mut model = GpuModel::load(Arc::clone(&self.cl), &directory, None)
                .with_context(|| format!("loading {name} checkpoint"))?;
            let accelerate = env::var("LAYA_NPU_ACCELERATE")
                .unwrap_or_else(|_| "english,multilingual,typed-decisions".into());
            if accelerate.split(',').any(|item| item.trim() == name) {
                if !self.npu.contains_key(name) {
                    let config: Value =
                        serde_json::from_slice(&fs::read(directory.join("encoder/config.json"))?)?;
                    let hidden = config["hidden_size"]
                        .as_u64()
                        .context("missing hidden_size")? as usize;
                    let layers = config["num_hidden_layers"]
                        .as_u64()
                        .context("missing num_hidden_layers")?
                        as usize;
                    let library = env::var("LAYA_RKNNRT")?;
                    let graph_root = env::var("LAYA_RKNN_GRAPH_ROOT")?;
                    let graph_name = match name {
                        "english" => "english64",
                        "multilingual" => "multilingual64",
                        "typed-decisions" => "typed64",
                        _ => unreachable!(),
                    };
                    let graphs = Path::new(&graph_root).join(graph_name);
                    self.npu.insert(
                        name,
                        Rc::new(NpuEncoder::load(
                            Path::new(&library),
                            &graphs,
                            hidden,
                            64,
                            layers,
                        )?),
                    );
                }
                let npu = Rc::clone(self.npu.get(name).unwrap());
                model.set_encoder_override(17, 64, move |embedding, valid| {
                    npu.infer(embedding, valid)
                });
            }
            self.loaded.insert(name, Api::new(model, &directory)?);
        }
        Ok(self.loaded.get(name).unwrap())
    }

    fn preload(&mut self, names: &str) -> Result<()> {
        for item in names.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let name = match item {
                "english" => "english",
                "multilingual" => "multilingual",
                "typed-decisions" => "typed-decisions",
                _ => anyhow::bail!("unknown checkpoint {item}"),
            };
            self.load(name)?;
        }
        Ok(())
    }

    fn loaded(&self) -> Vec<&'static str> {
        let mut names: Vec<_> = self.loaded.keys().copied().collect();
        names.sort_unstable();
        names
    }

    fn predict(&mut self, request: &Value) -> Result<Value> {
        let questions = request
            .get("questions")
            .and_then(Value::as_object)
            .context("questions must be an object")?;
        let selected = route::decide(request, questions);
        let mut response = self.load(selected.model)?.predict(request)?;
        response["routing"] = selected.metadata;
        Ok(response)
    }
}

fn respond(request: tiny_http::Request, status: u16, value: &Value) -> Result<()> {
    let header = Header::from_bytes(b"Content-Type", b"application/json")
        .map_err(|_| anyhow::anyhow!("invalid content-type header"))?;
    request.respond(
        Response::from_string(serde_json::to_string(value)?)
            .with_status_code(StatusCode(status))
            .with_header(header),
    )?;
    Ok(())
}

pub fn run() -> Result<()> {
    let gpu_library = env::var("LAYA_MALI_OPENCL")?;
    let cl = Cl::new(&gpu_library, include_str!("../../mali/src/kernels.cl"))?;
    let base_dir = PathBuf::from(env::var("LAYA_MODEL_DIR")?);
    let started = Instant::now();
    let mut router = Router::new(Arc::clone(&cl), base_dir);
    router.preload(
        &env::var("LAYA_NPU_PRELOAD")
            .unwrap_or_else(|_| "english,multilingual,typed-decisions".into()),
    )?;
    cl.warmup()?;
    eprintln!(
        "laya-rknpu ready after {:.3}s",
        started.elapsed().as_secs_f64()
    );
    let bind = env::var("LAYA_NPU_BIND").unwrap_or_else(|_| "127.0.0.1:8003".into());
    let server = Server::http(&bind).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    eprintln!("laya-rknpu listening on {bind}");
    for mut request in server.incoming_requests() {
        let route = (request.method().clone(), request.url().to_owned());
        match route {
            (Method::Get, path) if path == "/health" => respond(
                request,
                200,
                &json!({
                    "status": "ok", "loaded": router.loaded(),
                    "device": "RK3588 NPU encoder with Mali decision head",
                    "npu_models": router.npu.keys().copied().collect::<Vec<_>>(),
                    "npu_token_range": [17,64],
                    "npu_inferences": router.npu.iter().map(|(name, engine)| (*name, engine.calls())).collect::<HashMap<_,_>>()
                }),
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
                        serde_json::from_slice(&body).map_err(anyhow::Error::from)
                    })
                    .and_then(|value| router.predict(&value));
                match result {
                    Ok(value) => respond(request, 200, &value)?,
                    Err(error) => respond(request, 400, &json!({"detail": error.to_string()}))?,
                }
            }
            _ => respond(request, 404, &json!({"detail":"not found"}))?,
        }
    }
    Ok(())
}
