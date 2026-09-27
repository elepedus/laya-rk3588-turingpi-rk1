use crate::{api::Api, full::NpuModel, route};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    env,
    io::Read,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};
use tiny_http::{Header, Method, Response, Server, StatusCode};

struct Router {
    base_dir: PathBuf,
    graph_root: PathBuf,
    runtime: PathBuf,
    loaded: HashMap<&'static str, Api>,
    models: HashMap<&'static str, Rc<NpuModel>>,
}

impl Router {
    fn new(base_dir: PathBuf, graph_root: PathBuf, runtime: PathBuf) -> Self {
        Self {
            base_dir,
            graph_root,
            runtime,
            loaded: HashMap::new(),
            models: HashMap::new(),
        }
    }

    fn load(&mut self, name: &'static str) -> Result<&Api> {
        if !self.loaded.contains_key(name) {
            let directory = if name == "english" {
                self.base_dir.clone()
            } else {
                self.base_dir.join(name)
            };
            let model = Rc::new(
                NpuModel::load(name, &directory, &self.graph_root, &self.runtime)
                    .with_context(|| format!("loading {name} NPU model"))?,
            );
            let api = Api::new(Rc::clone(&model), &directory)?;
            self.models.insert(name, model);
            self.loaded.insert(name, api);
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

    fn predict(&mut self, request: &Value) -> Result<Value> {
        let questions = request
            .get("questions")
            .and_then(Value::as_object)
            .context("questions must be an object")?;
        let selected = route::decide(request, questions);
        let keep_caches = env::var_os("LAYA_NPU_KEEP_CACHES").is_some();
        if !keep_caches {
            for (name, model) in &self.models {
                if *name != selected.model {
                    model.clear_cache();
                }
            }
        }
        let attempt = self.load(selected.model)?.predict(request);
        let mut response = match attempt {
            Ok(value) => value,
            Err(error)
                if keep_caches && {
                    let reason = format!("{error:#}");
                    reason.contains("RKNN status -4") || reason.contains("RKNN status -6")
                } =>
            {
                eprintln!("NPU_CACHE_EVICT model={} reason={error:#}", selected.model);
                for (name, model) in &self.models {
                    if *name != selected.model {
                        model.clear_cache();
                    }
                }
                self.load(selected.model)?.predict(request)?
            }
            Err(error) => return Err(error),
        };
        response["routing"] = selected.metadata;
        Ok(response)
    }

    fn health(&self) -> Value {
        let mut loaded: Vec<_> = self.loaded.keys().copied().collect();
        loaded.sort_unstable();
        let ready = self
            .models
            .values()
            .all(|model| model.available_lengths().len() == model.expected_bucket_count());
        json!({
            "status": if ready { "ok" } else { "building" },
            "device": "RK3588 NPU neural operators",
            "loaded": loaded,
            "available_buckets": self.models.iter()
                .map(|(name, model)| (*name, model.available_lengths()))
                .collect::<HashMap<_,_>>(),
            "requests": self.models.iter().map(|(name, model)| (*name, model.calls()))
                .collect::<HashMap<_,_>>(),
            "cached_buckets": self.models.iter()
                .map(|(name, model)| (*name, model.cached_lengths()))
                .collect::<HashMap<_,_>>(),
        })
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
    let base_dir = PathBuf::from(env::var("LAYA_MODEL_DIR")?);
    let graph_root = PathBuf::from(env::var("LAYA_RKNN_GRAPH_ROOT")?);
    let runtime = PathBuf::from(env::var("LAYA_RKNNRT")?);
    anyhow::ensure!(Path::new(&runtime).exists(), "missing RKNN runtime");
    let started = Instant::now();
    let mut router = Router::new(base_dir, graph_root, runtime);
    router.preload(
        &env::var("LAYA_NPU_PRELOAD")
            .unwrap_or_else(|_| "english,multilingual,typed-decisions".into()),
    )?;
    eprintln!(
        "laya-rknpu-full ready after {:.3}s",
        started.elapsed().as_secs_f64()
    );
    let bind = env::var("LAYA_NPU_FULL_BIND").unwrap_or_else(|_| "127.0.0.1:8004".into());
    let server = Server::http(&bind).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    eprintln!("laya-rknpu-full listening on {bind}");
    for mut request in server.incoming_requests() {
        let route = (request.method().clone(), request.url().to_owned());
        match route {
            (Method::Get, path) if path == "/health" => respond(request, 200, &router.health())?,
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
