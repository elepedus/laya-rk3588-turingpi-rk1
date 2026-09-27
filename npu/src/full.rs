//! All neural arithmetic runs through RKNN. Token lookup and scalar answer
//! bookkeeping stay on the host because RKNN executes Gather on the CPU.
use crate::{
    api::InferenceModel,
    rknn::{Input, Rknn},
};
use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, VecDeque},
    fs::{self, File},
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

const LENGTHS: [usize; 7] = [64, 128, 256, 512, 640, 768, 1024];

struct Bucket {
    length: usize,
    embedding_norm: Rknn,
    encoder: Vec<Rknn>,
    blocks: Option<Vec<FusedBlock>>,
    head: Rknn,
}

struct FusedBlock {
    start: usize,
    end: usize,
    graph: Rknn,
}

fn has_block_set(block_directory: &Path, layers: usize, group: usize) -> bool {
    let mut start = 0;
    while start < layers {
        let remaining = layers - start;
        let count = if remaining <= group + 1 {
            remaining
        } else {
            group
        };
        let end = start + count - 1;
        if !block_directory
            .join(format!("encoder_{start:02}_{end:02}.rknn"))
            .is_file()
        {
            return false;
        }
        start = end + 1;
    }
    true
}

fn has_fused_blocks(directory: &Path, layers: usize, group: usize) -> bool {
    has_block_set(&directory.join(format!("blocks{group}")), layers, group)
}

fn load_graph(library: &Path, path: &Path, inputs: u32) -> Result<Rknn> {
    let graph = Rknn::load(library, path)
        .with_context(|| format!("loading RKNN graph {}", path.display()))?;
    let core_mask: i32 = std::env::var("LAYA_RKNN_CORE_MASK")
        .ok()
        .map(|raw| raw.parse())
        .transpose()?
        .unwrap_or(7);
    graph.set_core_mask(core_mask)?;
    anyhow::ensure!(
        graph.io_count()? == (inputs, 1),
        "{} has unexpected input/output count",
        path.display()
    );
    Ok(graph)
}

impl Bucket {
    fn load(library: &Path, directory: &Path, length: usize, layers: usize) -> Result<Self> {
        let started = Instant::now();
        let embedding_norm = load_graph(library, &directory.join("embedding_norm.rknn"), 1)?;
        let requested_group: usize = std::env::var("LAYA_NPU_FUSED_GROUP")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(7);
        anyhow::ensure!(requested_group >= 2, "invalid fused group size");
        let group = if has_fused_blocks(directory, layers, requested_group) {
            requested_group
        } else {
            7
        };
        let use_blocks = std::env::var_os("LAYA_NPU_FUSED_BLOCKS").is_some()
            && has_fused_blocks(directory, layers, group);
        let requested_window: usize = std::env::var("LAYA_NPU_WINDOW_QUERY")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(0);
        let window_directory = directory.join(format!("blocks{group}-window{requested_window}"));
        let use_window =
            use_blocks && requested_window > 0 && has_block_set(&window_directory, layers, group);
        let block_directory = if use_window {
            window_directory
        } else {
            directory.join(format!("blocks{group}"))
        };
        let mut encoder = Vec::new();
        let mut blocks = None;
        if use_blocks {
            let mut fused = Vec::new();
            let mut start = 0;
            while start < layers {
                let remaining = layers - start;
                let count = if remaining <= group + 1 {
                    remaining
                } else {
                    group
                };
                let end = start + count - 1;
                let path = block_directory.join(format!("encoder_{start:02}_{end:02}.rknn"));
                fused.push(FusedBlock {
                    start,
                    end,
                    graph: load_graph(library, &path, 3)?,
                });
                start = end + 1;
            }
            blocks = Some(fused);
        } else {
            encoder.reserve(layers);
            for index in 0..layers {
                let path = directory.join(format!("layer{index:02}/encoder_{index:02}.rknn"));
                encoder.push(load_graph(library, &path, 2)?);
            }
        }
        let head = load_graph(library, &directory.join("head_onehot.rknn"), 3)?;
        eprintln!(
            "NPU_BUCKET_LOAD length={length} layers={layers} fused_group={group} window_query={} fused_blocks={} wall_ms={:.3}",
            if use_window { requested_window } else { 0 },
            blocks.as_ref().map_or(0, Vec::len),
            started.elapsed().as_secs_f64() * 1000.0
        );
        Ok(Self {
            length,
            embedding_norm,
            encoder,
            blocks,
            head,
        })
    }
}

pub struct NpuModel {
    name: String,
    library: PathBuf,
    graph_root: PathBuf,
    weights: Mmap,
    dim: usize,
    layers: usize,
    max_len: usize,
    pad_id: i32,
    half_window: usize,
    buckets: RefCell<HashMap<usize, Rc<Bucket>>>,
    recency: RefCell<VecDeque<usize>>,
    scorer: Rknn,
    action: Rknn,
    calls: Cell<u64>,
}

impl NpuModel {
    pub fn load(name: &str, model_dir: &Path, graph_root: &Path, library: &Path) -> Result<Self> {
        let encoder: Value =
            serde_json::from_slice(&fs::read(model_dir.join("encoder/config.json"))?)?;
        let agent: Value =
            serde_json::from_slice(&fs::read(model_dir.join("rl_agent_config.json"))?)?;
        let number = |key: &str| {
            encoder[key]
                .as_u64()
                .with_context(|| format!("{name} missing encoder config {key}"))
        };
        let dim = number("hidden_size")? as usize;
        let layers = number("num_hidden_layers")? as usize;
        let max_len = agent["max_len"].as_u64().context("missing max_len")? as usize;
        let pad_id = encoder["pad_token_id"].as_i64().unwrap_or(0) as i32;
        let half_window = (encoder["local_attention"].as_u64().unwrap_or(128) / 2) as usize;
        anyhow::ensure!(LENGTHS.contains(&max_len), "unsupported max_len {max_len}");
        let weights = unsafe { Mmap::map(&File::open(model_dir.join("model.safetensors"))?)? };
        let alias = if name == "typed-decisions" {
            "typed"
        } else {
            name
        };
        let decision = graph_root.join(format!("{alias}-decision-ops"));
        let scorer = load_graph(library, &decision.join("scorer.rknn"), 1)?;
        let action = load_graph(library, &decision.join("action.rknn"), 1)?;
        Ok(Self {
            name: alias.to_owned(),
            library: library.to_path_buf(),
            graph_root: graph_root.to_path_buf(),
            weights,
            dim,
            layers,
            max_len,
            pad_id,
            half_window,
            buckets: RefCell::new(HashMap::new()),
            recency: RefCell::new(VecDeque::new()),
            scorer,
            action,
            calls: Cell::new(0),
        })
    }

    pub fn calls(&self) -> u64 {
        self.calls.get()
    }

    pub fn cached_lengths(&self) -> Vec<usize> {
        self.recency.borrow().iter().copied().collect()
    }

    fn bucket_available(&self, length: usize) -> bool {
        if length > self.max_len {
            return false;
        }
        let path = self.graph_root.join(format!("{}{}", self.name, length));
        path.join("embedding_norm.rknn").is_file()
            && path.join("head_onehot.rknn").is_file()
            && (has_fused_blocks(&path, self.layers, 14)
                || path
                    .join(format!(
                        "layer{:02}/encoder_{:02}.rknn",
                        self.layers - 1,
                        self.layers - 1
                    ))
                    .is_file())
    }

    pub fn available_lengths(&self) -> Vec<usize> {
        LENGTHS
            .into_iter()
            .filter(|&length| self.bucket_available(length))
            .collect()
    }

    pub fn expected_bucket_count(&self) -> usize {
        LENGTHS
            .into_iter()
            .filter(|&length| length <= self.max_len && (length != 640 || self.name == "typed"))
            .count()
    }

    pub fn clear_cache(&self) {
        self.buckets.borrow_mut().clear();
        self.recency.borrow_mut().clear();
    }

    fn bucket(&self, valid: usize) -> Result<Rc<Bucket>> {
        anyhow::ensure!(
            valid > 0 && valid <= self.max_len,
            "{} supports 1..{} tokens, got {valid}",
            self.name,
            self.max_len
        );
        let minimum_bucket: usize = std::env::var("LAYA_NPU_MIN_BUCKET")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(0);
        let length = LENGTHS
            .into_iter()
            .find(|&n| valid <= n && minimum_bucket <= n && self.bucket_available(n))
            .context("no RKNN sequence bucket fits the request")?;
        let limit: usize = std::env::var("LAYA_NPU_CACHE_BUCKETS")
            .ok()
            .and_then(|raw| raw.parse().ok())
            .unwrap_or(2)
            .max(1);
        let existing = self.buckets.borrow().get(&length).cloned();
        let bucket = if let Some(bucket) = existing {
            bucket
        } else {
            while self.buckets.borrow().len() >= limit {
                let oldest = self
                    .recency
                    .borrow_mut()
                    .pop_front()
                    .context("RKNN bucket cache lost its recency entry")?;
                self.buckets.borrow_mut().remove(&oldest);
            }
            let directory = self.graph_root.join(format!("{}{}", self.name, length));
            let bucket = Rc::new(Bucket::load(
                &self.library,
                &directory,
                length,
                self.layers,
            )?);
            self.buckets.borrow_mut().insert(length, Rc::clone(&bucket));
            bucket
        };
        let mut order = self.recency.borrow_mut();
        order.retain(|&value| value != length);
        order.push_back(length);
        Ok(bucket)
    }

    fn gather(&self, ids: &[i32], bucket: usize) -> Result<Vec<f32>> {
        let tensors = SafeTensors::deserialize(&self.weights)?;
        let table = tensors.tensor("encoder.embeddings.tok_embeddings.weight")?;
        anyhow::ensure!(
            table.dtype() == Dtype::F16 && table.shape().len() == 2 && table.shape()[1] == self.dim,
            "unexpected embedding table format"
        );
        let vocab = table.shape()[0];
        let bytes = table.data();
        let mut gathered = vec![0f32; bucket * self.dim];
        for row in 0..bucket {
            let id = if row < ids.len() {
                ids[row]
            } else {
                self.pad_id
            };
            let id = usize::try_from(id).context("negative token ID")?;
            anyhow::ensure!(id < vocab, "token ID {id} exceeds vocabulary {vocab}");
            let offset = id * self.dim * 2;
            let target = &mut gathered[row * self.dim..(row + 1) * self.dim];
            for (index, slot) in target.iter_mut().enumerate() {
                let at = offset + index * 2;
                *slot = crate::f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
            }
        }
        Ok(gathered)
    }

    fn masks(&self, valid: usize, length: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut key = vec![0f32; length];
        key[valid..].fill(-10000.0);
        let mut full = vec![0f32; length * length];
        let mut local = vec![0f32; length * length];
        for query in 0..length {
            for position in 0..length {
                let offset = query * length + position;
                full[offset] = key[position];
                local[offset] = if query.abs_diff(position) > self.half_window {
                    -10000.0
                } else {
                    key[position]
                };
            }
        }
        (key, full, local)
    }

    fn compare_layer<I>(&self, index: usize, valid: usize, actual: I) -> Result<()>
    where
        I: IntoIterator<Item = f32>,
    {
        let oracle = match std::env::var("LAYA_ORACLE_DIR") {
            Ok(directory) => PathBuf::from(directory),
            Err(_) => return Ok(()),
        };
        let path = oracle.join(format!("encoder_{index:02}.f32"));
        let expected =
            fs::read(&path).with_context(|| format!("reading oracle layer {}", path.display()))?;
        let count = valid * self.dim;
        anyhow::ensure!(
            expected.len() == count * 4,
            "oracle layer {index} has {} bytes, expected {}",
            expected.len(),
            count * 4
        );
        let mut maximum = 0f32;
        let mut mean = 0f64;
        let mut reference_max = 0f32;
        for (sample, bytes) in actual.into_iter().take(count).zip(expected.chunks_exact(4)) {
            let reference = f32::from_le_bytes(bytes.try_into().unwrap());
            let error = (sample - reference).abs();
            maximum = maximum.max(error);
            mean += error as f64;
            reference_max = reference_max.max(reference.abs());
        }
        eprintln!("ORACLE_LAYER model={} index={index} valid={valid} max_abs={maximum:.6} mean_abs={:.6} reference_abs_max={reference_max:.3}",
            self.name, mean / count as f64);
        Ok(())
    }

    fn timed<R, F>(
        &self,
        stage: &str,
        graph: &Rknn,
        timings: &mut Vec<(String, f64, i64)>,
        run: F,
    ) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        let started = Instant::now();
        let result = run().with_context(|| format!("{} {stage}", self.name))?;
        if std::env::var_os("LAYA_NPU_TIMING").is_some() {
            timings.push((
                stage.to_owned(),
                started.elapsed().as_secs_f64() * 1000.0,
                graph.last_duration_us()?,
            ));
        }
        if std::env::var_os("LAYA_RKNN_PHASES").is_some() {
            let phases = graph.last_phases();
            eprintln!("RKNN_PHASE model={} stage={} input_ms={:.3} run_ms={:.3} output_ms={:.3} copy_ms={:.3} release_ms={:.3}",
                self.name, stage, phases.input_ms, phases.run_ms, phases.output_ms,
                phases.copy_ms, phases.release_ms);
        }
        if std::env::var_os("LAYA_RKNN_PROFILE").is_some() {
            eprintln!(
                "NPU_OPERATOR_PROFILE model={} stage={stage}\n{}",
                self.name,
                graph.perf_detail()?
            );
        }
        Ok(result)
    }

    fn infer_stage(
        &self,
        stage: &str,
        graph: &Rknn,
        buffers: &[&[f32]],
        formats: &[i32],
        timings: &mut Vec<(String, f64, i64)>,
    ) -> Result<Vec<f32>> {
        self.timed(stage, graph, timings, || graph.infer_f32(buffers, formats))
    }

    fn infer_stage_f16(
        &self,
        stage: &str,
        graph: &Rknn,
        inputs: &[Input<'_>],
        timings: &mut Vec<(String, f64, i64)>,
    ) -> Result<Vec<u16>> {
        self.timed(stage, graph, timings, || graph.infer_f16(inputs))
    }

    fn infer_stage_mixed_f32(
        &self,
        stage: &str,
        graph: &Rknn,
        inputs: &[Input<'_>],
        timings: &mut Vec<(String, f64, i64)>,
    ) -> Result<Vec<f32>> {
        self.timed(stage, graph, timings, || graph.infer_mixed_f32(inputs))
    }

    fn report(
        &self,
        bucket: usize,
        valid: usize,
        host_lookup_ms: f64,
        mask_prepare_ms: f64,
        total_ms: f64,
        mut timings: Vec<(String, f64, i64)>,
    ) {
        if std::env::var_os("LAYA_NPU_TIMING").is_none() {
            return;
        }
        let layer_wall: f64 = timings
            .iter()
            .filter(|(name, _, _)| name.starts_with("encoder_"))
            .map(|(_, ms, _)| ms)
            .sum();
        let layer_npu: i64 = timings
            .iter()
            .filter(|(name, _, _)| name.starts_with("encoder_"))
            .map(|(_, _, us)| us)
            .sum();
        eprintln!("NPU_REQUEST model={} valid={} bucket={} lookup_ms={host_lookup_ms:.3} mask_prepare_ms={mask_prepare_ms:.3} encoder_wall_ms={layer_wall:.3} encoder_npu_us={layer_npu} total_ms={total_ms:.3}",
                  self.name, valid, bucket);
        timings.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (stage, wall_ms, npu_us) in timings {
            eprintln!(
                "NPU_STAGE model={} bucket={} stage={} wall_ms={wall_ms:.3} npu_us={npu_us}",
                self.name, bucket, stage
            );
        }
    }
}

impl InferenceModel for Rc<NpuModel> {
    fn run(&self, ids: &[i32], qtype: i32) -> Result<Vec<f32>> {
        let started = Instant::now();
        let bucket = self.bucket(ids.len())?;
        let lookup_started = Instant::now();
        let gathered = self.gather(ids, bucket.length)?;
        let host_lookup_ms = lookup_started.elapsed().as_secs_f64() * 1000.0;
        let mask_started = Instant::now();
        let (key, full, local) = self.masks(ids.len(), bucket.length);
        let mask_prepare_ms = mask_started.elapsed().as_secs_f64() * 1000.0;
        let mut timings = Vec::with_capacity(self.layers + 2);
        anyhow::ensure!((0..3).contains(&qtype), "invalid question type {qtype}");
        let mut qtype_onehot = [0f32; 3];
        qtype_onehot[qtype as usize] = 1.0;
        let mut half = self.infer_stage_f16(
            "embedding_norm",
            &bucket.embedding_norm,
            &[Input::F32(&gathered, 3)],
            &mut timings,
        )?;
        if let Some(blocks) = &bucket.blocks {
            for block in blocks {
                half = self.infer_stage_f16(
                    &format!("encoder_{:02}_{:02}", block.start, block.end),
                    &block.graph,
                    &[
                        Input::F16(&half, 3),
                        Input::F32(&full, 1),
                        Input::F32(&local, 1),
                    ],
                    &mut timings,
                )?;
                anyhow::ensure!(
                    half.len() == bucket.length * self.dim,
                    "NPU fused block {}-{} returned {} values",
                    block.start,
                    block.end,
                    half.len()
                );
                self.compare_layer(
                    block.end,
                    ids.len(),
                    half.iter().copied().map(crate::f16_to_f32),
                )?;
            }
        } else {
            for (index, graph) in bucket.encoder.iter().enumerate() {
                let attention = if bucket.length == 64 {
                    &key
                } else if index % 3 == 0 {
                    &full
                } else {
                    &local
                };
                half = self.infer_stage_f16(
                    &format!("encoder_{index:02}"),
                    graph,
                    &[Input::F16(&half, 3), Input::F32(attention, 1)],
                    &mut timings,
                )?;
                anyhow::ensure!(
                    half.len() == bucket.length * self.dim,
                    "NPU layer {index} returned {} values",
                    half.len()
                );
                self.compare_layer(
                    index,
                    ids.len(),
                    half.iter().copied().map(crate::f16_to_f32),
                )?;
            }
        }
        let hidden = self.infer_stage_mixed_f32(
            "decision_head",
            &bucket.head,
            &[
                Input::F16(&half, 3),
                Input::F32(&full, 3),
                Input::F32(&qtype_onehot, 3),
            ],
            &mut timings,
        )?;
        anyhow::ensure!(
            hidden.len() == bucket.length * self.dim,
            "NPU head returned {} values, expected {}",
            hidden.len(),
            bucket.length * self.dim
        );
        self.calls.set(self.calls.get() + 1);
        self.report(
            bucket.length,
            ids.len(),
            host_lookup_ms,
            mask_prepare_ms,
            started.elapsed().as_secs_f64() * 1000.0,
            timings,
        );
        Ok(hidden)
    }

    fn decode(&self, hidden: &[f32], markers: &[usize]) -> Result<(Vec<f32>, [f32; 2])> {
        anyhow::ensure!(
            !markers.is_empty() && markers.len() <= 32,
            "need 1..32 marker positions"
        );
        let mut rows = vec![0f32; 32 * self.dim];
        for (index, &position) in markers.iter().enumerate() {
            anyhow::ensure!(
                (position + 1) * self.dim <= hidden.len(),
                "marker {position} out of range"
            );
            rows[index * self.dim..(index + 1) * self.dim]
                .copy_from_slice(&hidden[position * self.dim..(position + 1) * self.dim]);
        }
        let mut timings = Vec::new();
        let scores =
            self.infer_stage("marker_scorer", &self.scorer, &[&rows], &[3], &mut timings)?;
        anyhow::ensure!(
            scores.len() == 32,
            "NPU scorer returned {} values",
            scores.len()
        );
        let logits = scores[..markers.len()].to_vec();
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits
            .iter()
            .map(|value| (*value - maximum).exp())
            .collect();
        let total: f32 = exp.iter().sum();
        let probabilities: Vec<f32> = exp.iter().map(|value| *value / total).collect();
        let mut sorted = probabilities.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let p1 = sorted[0];
        let p2 = sorted.get(1).copied().unwrap_or(0.0);
        let k = markers.len().max(2) as f32;
        let entropy = -probabilities
            .iter()
            .map(|p| p * p.max(1e-9).ln())
            .sum::<f32>()
            / k.ln();
        let mut features = Vec::with_capacity(self.dim + 4);
        features.extend_from_slice(&hidden[..self.dim]);
        features.extend_from_slice(&[p1, p1 - p2, entropy, k / 255.0]);
        let output = self.infer_stage(
            "action_head",
            &self.action,
            &[&features],
            &[3],
            &mut timings,
        )?;
        anyhow::ensure!(
            output.len() == 2,
            "NPU action head returned {} values",
            output.len()
        );
        if std::env::var_os("LAYA_NPU_TIMING").is_some() {
            for (stage, wall_ms, npu_us) in timings {
                eprintln!(
                    "NPU_STAGE model={} stage={} wall_ms={wall_ms:.3} npu_us={npu_us}",
                    self.name, stage
                );
            }
        }
        Ok((logits, [output[0], output[1]]))
    }
}
