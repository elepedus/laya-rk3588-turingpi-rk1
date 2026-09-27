use crate::{api::Api, cl::Cl, model::GpuModel, route};
use anyhow::{Context, Result};
use serde_json::Value;
use std::{collections::HashMap, path::PathBuf, sync::Arc};

pub struct Router {
    cl: Arc<Cl>,
    base_dir: PathBuf,
    loaded: HashMap<&'static str, Api>,
}

impl Router {
    pub fn new(cl: Arc<Cl>, base_dir: PathBuf) -> Self {
        Self {
            cl,
            base_dir,
            loaded: HashMap::new(),
        }
    }

    pub fn preload(&mut self, names: &str) -> Result<()> {
        for name in names
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            let key = match name {
                "english" => "english",
                "multilingual" => "multilingual",
                "typed-decisions" => "typed-decisions",
                _ => anyhow::bail!("unknown Laya checkpoint {name}"),
            };
            self.load(key)?;
        }
        Ok(())
    }

    pub fn loaded(&self) -> Vec<&'static str> {
        let mut names: Vec<_> = self.loaded.keys().copied().collect();
        names.sort_unstable();
        names
    }

    fn load(&mut self, name: &'static str) -> Result<&Api> {
        if !self.loaded.contains_key(name) {
            let directory = if name == "english" {
                self.base_dir.clone()
            } else {
                self.base_dir.join(name)
            };
            let model = GpuModel::load(Arc::clone(&self.cl), &directory, None)
                .with_context(|| format!("loading {name}"))?;
            let api = Api::new(model, &directory)?;
            self.loaded.insert(name, api);
        }
        Ok(self.loaded.get(name).unwrap())
    }

    pub fn predict(&mut self, request: &Value) -> Result<Value> {
        let questions = request
            .get("questions")
            .and_then(Value::as_object)
            .context("questions must be an object")?;
        let selected = route::decide(request, questions);
        let mut result = self.load(selected.model)?.predict(request)?;
        result["routing"] = selected.metadata;
        Ok(result)
    }
}
