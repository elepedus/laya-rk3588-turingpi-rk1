use crate::cl::{Buffer, Cl};
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde_json::Value;
use std::{
    collections::HashMap,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct GpuModel {
    cl: Arc<Cl>,
    weights: HashMap<String, Buffer>,
    _mapping: Mmap,
    zero_half: Buffer,
    reference: Option<PathBuf>,
    dim: usize,
    heads: usize,
    layers: usize,
    intermediate: usize,
    head_layers: usize,
    max_len: usize,
    pad_token_id: i32,
    half_window: i32,
    theta_full: f32,
    theta_local: f32,
    linear_kernel: String,
    attention_score_kernel: String,
    norm_kernel: String,
    encoder_override: Option<(usize, usize, Box<dyn Fn(&[f32], usize) -> Result<Vec<f32>>>)>,
}

impl GpuModel {
    pub fn load(cl: Arc<Cl>, model_dir: &Path, reference: Option<PathBuf>) -> Result<Self> {
        let encoder_config: Value =
            serde_json::from_slice(&fs::read(model_dir.join("encoder/config.json"))?)?;
        let agent_config: Value =
            serde_json::from_slice(&fs::read(model_dir.join("rl_agent_config.json"))?)?;
        let number = |key: &str| {
            encoder_config[key]
                .as_u64()
                .with_context(|| format!("encoder config missing {key}"))
        };
        let dim = number("hidden_size")? as usize;
        let heads = number("num_attention_heads")? as usize;
        let layers = number("num_hidden_layers")? as usize;
        let intermediate = number("intermediate_size")? as usize;
        let head_layers = agent_config["head_layers"].as_u64().unwrap_or(2) as usize;
        let max_len = agent_config["max_len"].as_u64().unwrap_or(512) as usize;
        let pad_token_id = encoder_config["pad_token_id"].as_i64().unwrap_or(0) as i32;
        let half_window = (encoder_config["local_attention"].as_i64().unwrap_or(128) / 2) as i32;
        let theta_full = encoder_config["rope_parameters"]["full_attention"]["rope_theta"]
            .as_f64()
            .unwrap_or(160_000.0) as f32;
        let theta_local = encoder_config["rope_parameters"]["sliding_attention"]["rope_theta"]
            .as_f64()
            .unwrap_or(10_000.0) as f32;
        let linear_kernel = std::env::var("LAYA_MALI_LINEAR_KERNEL")
            .unwrap_or_else(|_| "linear_h_transposed_rows4_cols8".to_owned());
        let attention_score_kernel = std::env::var("LAYA_MALI_SCORE_KERNEL")
            .unwrap_or_else(|_| "attention_scores_vec8".to_owned());
        if !matches!(
            attention_score_kernel.as_str(),
            "attention_scores" | "attention_scores_vec8"
        ) {
            bail!("unsupported attention score kernel {attention_score_kernel}")
        }
        let norm_kernel =
            std::env::var("LAYA_MALI_NORM_KERNEL").unwrap_or_else(|_| "norm_rows_vec8".to_owned());
        if !matches!(norm_kernel.as_str(), "norm_rows" | "norm_rows_vec8") {
            bail!("unsupported normalization kernel {norm_kernel}")
        }
        if !matches!(
            linear_kernel.as_str(),
            "linear_h"
                | "linear_h_vec8"
                | "linear_h_rows4_vec8"
                | "linear_h_transposed_rows4_cols4"
                | "linear_h_transposed_rows4_cols8"
        ) {
            bail!("unsupported linear kernel {linear_kernel}")
        }
        if dim / heads != 64 || dim % heads != 0 {
            bail!("unsupported attention shape {dim}/{heads}")
        }
        let file = File::open(model_dir.join("model.safetensors"))?;
        let mapping = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mapping)?;
        let mut weights = HashMap::new();
        let mut packed_count = 0usize;
        let packing_started = std::time::Instant::now();
        for name in tensors.names() {
            let tensor = tensors.tensor(name)?;
            let packed = matches!(
                linear_kernel.as_str(),
                "linear_h_transposed_rows4_cols4" | "linear_h_transposed_rows4_cols8"
            ) && tensor.shape().len() == 2
                && (name.starts_with("encoder.layers.") || name.starts_with("head.layers."));
            let bytes = if packed {
                if tensor.dtype() != safetensors::Dtype::F16 {
                    bail!("packed matrix {name} must be FP16")
                }
                packed_count += 1;
                Some(crate::packing::transpose_fp16_row_major(
                    tensor.data(),
                    tensor.shape()[0],
                    tensor.shape()[1],
                )?)
            } else {
                None
            };
            let half = cl
                .from_bytes(bytes.as_deref().unwrap_or(tensor.data()))
                .with_context(|| format!("uploading {name}"))?;
            weights.insert(name.to_owned(), half);
        }
        cl.finish()?;
        let zero_half = cl.from_bytes(&[0, 0])?;
        eprintln!(
            "uploaded {} checkpoint tensors to Mali OpenCL ({} transposed, {:.3}s)",
            weights.len(),
            packed_count,
            packing_started.elapsed().as_secs_f64()
        );
        Ok(Self {
            cl,
            weights,
            _mapping: mapping,
            zero_half,
            reference,
            dim,
            heads,
            layers,
            intermediate,
            head_layers,
            max_len,
            pad_token_id,
            half_window,
            theta_full,
            theta_local,
            linear_kernel,
            attention_score_kernel,
            norm_kernel,
            encoder_override: None,
        })
    }

    /// Use an external encoder for requests that fit its fixed sequence length.
    /// Embeddings and the decision head still run through this model.
    pub fn set_encoder_override<F>(&mut self, minimum_length: usize, sequence_length: usize, run: F)
    where
        F: Fn(&[f32], usize) -> Result<Vec<f32>> + 'static,
    {
        self.encoder_override = Some((minimum_length, sequence_length, Box::new(run)));
    }

    fn weight(&self, name: &str) -> Result<&Buffer> {
        self.weights
            .get(name)
            .with_context(|| format!("missing weight {name}"))
    }

    fn linear(
        &self,
        input: &Buffer,
        matrix: &str,
        bias: Option<&str>,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Buffer> {
        let output = self.cl.buffer(rows * out_dim * 4)?;
        if self.linear_kernel == "linear_h_transposed_rows4_cols4" && out_dim % 4 != 0 {
            bail!("packed linear output width must be divisible by four")
        }
        if self.linear_kernel == "linear_h_transposed_rows4_cols8" && out_dim % 8 != 0 {
            bail!("packed linear output width must be divisible by eight")
        }
        let b = match bias {
            Some(name) => self.weight(name)?,
            None => &self.zero_half,
        };
        let profile_label = format!("{}:{matrix}", self.linear_kernel);
        self.cl
            .kernel(&self.linear_kernel)?
            .label(&profile_label)
            .buffer(0, input)?
            .buffer(1, self.weight(matrix)?)?
            .buffer(2, b)?
            .buffer(3, &output)?
            .i32(4, rows as i32)?
            .i32(5, in_dim as i32)?
            .i32(6, out_dim as i32)?
            .i32(7, i32::from(bias.is_some()))?
            .run(match self.linear_kernel.as_str() {
                "linear_h_rows4_vec8" => rows.div_ceil(4) * out_dim,
                "linear_h_transposed_rows4_cols4" => rows.div_ceil(4) * (out_dim / 4),
                "linear_h_transposed_rows4_cols8" => rows.div_ceil(4) * (out_dim / 8),
                _ => rows * out_dim,
            })?;
        Ok(output)
    }

    fn norm(
        &self,
        input: &Buffer,
        gamma: &str,
        bias: Option<&str>,
        rows: usize,
        dim: usize,
    ) -> Result<Buffer> {
        let output = self.cl.buffer(rows * dim * 4)?;
        let b = match bias {
            Some(name) => self.weight(name)?,
            None => &self.zero_half,
        };
        self.cl
            .kernel(&self.norm_kernel)?
            .buffer(0, input)?
            .buffer(1, self.weight(gamma)?)?
            .buffer(2, b)?
            .buffer(3, &output)?
            .i32(4, rows as i32)?
            .i32(5, dim as i32)?
            .i32(6, i32::from(bias.is_some()))?
            .run(rows)?;
        Ok(output)
    }

    fn add(&self, x: &Buffer, y: &Buffer, len: usize) -> Result<Buffer> {
        let output = self.cl.buffer(len * 4)?;
        self.cl
            .kernel("vector_add")?
            .buffer(0, x)?
            .buffer(1, y)?
            .buffer(2, &output)?
            .i32(3, len as i32)?
            .run(len)?;
        Ok(output)
    }

    fn attention(
        &self,
        qkv: &Buffer,
        seq: usize,
        valid_len: usize,
        half_window: i32,
    ) -> Result<Buffer> {
        let count = seq * self.heads * seq;
        let scores = self.cl.buffer(count * 4)?;
        let output = self.cl.buffer(seq * self.dim * 4)?;
        let probabilities = self.cl.buffer(count * 4)?;
        self.cl
            .kernel(&self.attention_score_kernel)?
            .buffer(0, qkv)?
            .buffer(1, &scores)?
            .i32(2, seq as i32)?
            .i32(3, self.dim as i32)?
            .i32(4, self.heads as i32)?
            .i32(5, half_window)?
            .i32(6, valid_len as i32)?
            .run(count)?;
        self.cl
            .kernel("softmax_rows")?
            .buffer(0, &scores)?
            .buffer(1, &probabilities)?
            .i32(2, (seq * self.heads) as i32)?
            .i32(3, seq as i32)?
            .run(seq * self.heads)?;
        self.cl
            .kernel("attention_context")?
            .buffer(0, qkv)?
            .buffer(1, &probabilities)?
            .buffer(2, &output)?
            .i32(3, seq as i32)?
            .i32(4, self.dim as i32)?
            .i32(5, self.heads as i32)?
            .run(seq * self.dim)?;
        Ok(output)
    }

    pub fn run(&self, ids: &[i32], qtype: i32) -> Result<Vec<f32>> {
        let valid_len = ids.len();
        if valid_len == 0 || valid_len > self.max_len {
            bail!(
                "checkpoint requires 1 to {} tokens, got {valid_len}",
                self.max_len
            )
        }
        let use_override = self
            .encoder_override
            .as_ref()
            .filter(|(minimum, length, _)| valid_len >= *minimum && valid_len <= *length);
        let seq = use_override
            .map(|(_, length, _)| *length)
            .unwrap_or_else(|| valid_len.max(16));
        let mut padded = ids.to_vec();
        padded.resize(seq, self.pad_token_id);
        let d = self.dim;
        let inter = self.intermediate;
        let token_buffer = self.cl.from_bytes(crate::bytes(&padded))?;
        let mut hidden = self.cl.buffer(seq * d * 4)?;
        self.cl
            .kernel("embedding_norm")?
            .buffer(0, &token_buffer)?
            .buffer(1, self.weight("encoder.embeddings.tok_embeddings.weight")?)?
            .buffer(2, self.weight("encoder.embeddings.norm.weight")?)?
            .buffer(3, &hidden)?
            .i32(4, seq as i32)?
            .i32(5, d as i32)?
            .run(seq)?;
        self.compare("embedding", &hidden)?;

        if let Some((_, _, encoder)) = use_override {
            let mut input = vec![0f32; seq * d];
            hidden.read(crate::bytes_mut(&mut input))?;
            let output = encoder(&input, valid_len)?;
            anyhow::ensure!(
                output.len() == seq * d,
                "external encoder returned {} values; expected {}",
                output.len(),
                seq * d
            );
            hidden = self.cl.from_bytes(crate::bytes(&output))?;
        } else {
            for layer in 0..self.layers {
                let prefix = format!("encoder.layers.{layer}");
                let attn_input = if layer == 0 {
                    None
                } else {
                    Some(self.norm(&hidden, &format!("{prefix}.attn_norm.weight"), None, seq, d)?)
                };
                let qkv = self.linear(
                    attn_input.as_ref().unwrap_or(&hidden),
                    &format!("{prefix}.attn.Wqkv.weight"),
                    None,
                    seq,
                    d,
                    3 * d,
                )?;
                let rotated = self.cl.buffer(seq * 3 * d * 4)?;
                let full = layer % 3 == 0;
                self.cl
                    .kernel("rotary_qkv")?
                    .buffer(0, &qkv)?
                    .buffer(1, &rotated)?
                    .i32(2, seq as i32)?
                    .i32(3, d as i32)?
                    .i32(4, self.heads as i32)?
                    .f32(
                        5,
                        if full {
                            self.theta_full
                        } else {
                            self.theta_local
                        },
                    )?
                    .run(seq * 3 * d)?;
                let context = self.attention(
                    &rotated,
                    seq,
                    valid_len,
                    if full { 0 } else { self.half_window },
                )?;
                let projected = self.linear(
                    &context,
                    &format!("{prefix}.attn.Wo.weight"),
                    None,
                    seq,
                    d,
                    d,
                )?;
                if layer == 0 {
                    self.compare("encoder_00_attention", &projected)?;
                }
                let after_attention = self.add(&hidden, &projected, seq * d)?;
                let normed = self.norm(
                    &after_attention,
                    &format!("{prefix}.mlp_norm.weight"),
                    None,
                    seq,
                    d,
                )?;
                let wi = self.linear(
                    &normed,
                    &format!("{prefix}.mlp.Wi.weight"),
                    None,
                    seq,
                    d,
                    2 * inter,
                )?;
                let gated = self.cl.buffer(seq * inter * 4)?;
                self.cl
                    .kernel("swiglu")?
                    .buffer(0, &wi)?
                    .buffer(1, &gated)?
                    .i32(2, seq as i32)?
                    .i32(3, inter as i32)?
                    .run(seq * inter)?;
                let mlp = self.linear(
                    &gated,
                    &format!("{prefix}.mlp.Wo.weight"),
                    None,
                    seq,
                    inter,
                    d,
                )?;
                if layer == 0 {
                    self.compare("encoder_00_mlp", &mlp)?;
                }
                hidden = self.add(&after_attention, &mlp, seq * d)?;
                self.compare(&format!("encoder_{layer:02}"), &hidden)?;
            }
        }
        hidden = self.norm(&hidden, "encoder.final_norm.weight", None, seq, d)?;
        self.compare("encoder_final", &hidden)?;
        let typed = self.cl.buffer(seq * d * 4)?;
        self.cl
            .kernel("add_type")?
            .buffer(0, &hidden)?
            .buffer(1, self.weight("type_emb.weight")?)?
            .buffer(2, &typed)?
            .i32(3, (seq * d) as i32)?
            .i32(4, qtype)?
            .i32(5, d as i32)?
            .run(seq * d)?;
        hidden = typed;

        for layer in 0..self.head_layers {
            let prefix = format!("head.layers.{layer}");
            let normed = self.norm(
                &hidden,
                &format!("{prefix}.norm1.weight"),
                Some(&format!("{prefix}.norm1.bias")),
                seq,
                d,
            )?;
            let qkv = self.linear(
                &normed,
                &format!("{prefix}.self_attn.in_proj_weight"),
                Some(&format!("{prefix}.self_attn.in_proj_bias")),
                seq,
                d,
                3 * d,
            )?;
            let context = self.attention(&qkv, seq, valid_len, 0)?;
            let projected = self.linear(
                &context,
                &format!("{prefix}.self_attn.out_proj.weight"),
                Some(&format!("{prefix}.self_attn.out_proj.bias")),
                seq,
                d,
                d,
            )?;
            let residual = self.add(&hidden, &projected, seq * d)?;
            let normed = self.norm(
                &residual,
                &format!("{prefix}.norm2.weight"),
                Some(&format!("{prefix}.norm2.bias")),
                seq,
                d,
            )?;
            let ff1 = self.linear(
                &normed,
                &format!("{prefix}.linear1.weight"),
                Some(&format!("{prefix}.linear1.bias")),
                seq,
                d,
                4 * d,
            )?;
            let activated = self.cl.buffer(seq * 4 * d * 4)?;
            self.cl
                .kernel("relu")?
                .buffer(0, &ff1)?
                .buffer(1, &activated)?
                .i32(2, (seq * 4 * d) as i32)?
                .run(seq * 4 * d)?;
            let ff2 = self.linear(
                &activated,
                &format!("{prefix}.linear2.weight"),
                Some(&format!("{prefix}.linear2.bias")),
                seq,
                4 * d,
                d,
            )?;
            hidden = self.add(&residual, &ff2, seq * d)?;
            self.compare(&format!("head_{layer:02}"), &hidden)?;
        }
        let mut output = vec![0f32; seq * d];
        hidden.read(crate::bytes_mut(&mut output))?;
        Ok(output)
    }

    pub fn decode(&self, hidden: &[f32], markers: &[usize]) -> Result<(Vec<f32>, [f32; 2])> {
        let d = self.dim;
        let tensors = SafeTensors::deserialize(&self._mapping)?;
        let load = |name: &str| -> Result<Vec<f32>> {
            let tensor = tensors.tensor(name)?;
            if tensor.dtype() != safetensors::Dtype::F16 {
                bail!("{name} is not FP16")
            }
            Ok(tensor
                .data()
                .chunks_exact(2)
                .map(|chunk| crate::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect())
        };
        let norm_weight = load("scorer.0.weight")?;
        let norm_bias = load("scorer.0.bias")?;
        let linear_weight = load("scorer.1.weight")?;
        let linear_bias = load("scorer.1.bias")?;
        let score_weight = load("scorer.3.weight")?;
        let score_bias = load("scorer.3.bias")?;
        let mut logits = Vec::with_capacity(markers.len());
        for &marker in markers {
            if (marker + 1) * d > hidden.len() {
                bail!("marker {marker} out of range")
            }
            let row = &hidden[marker * d..(marker + 1) * d];
            let mean = row.iter().copied().sum::<f32>() / d as f32;
            let variance = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / d as f32;
            let inv = (variance + 1.0e-5).sqrt().recip();
            let normed: Vec<f32> = (0..d)
                .map(|j| (row[j] - mean) * inv * norm_weight[j] + norm_bias[j])
                .collect();
            let mut activated = vec![0f32; d];
            for out in 0..d {
                let weight = &linear_weight[out * d..(out + 1) * d];
                let mut sum = linear_bias[out];
                for j in 0..d {
                    sum = normed[j].mul_add(weight[j], sum);
                }
                activated[out] =
                    0.5 * sum * (1.0 + libm::erff(sum * std::f32::consts::FRAC_1_SQRT_2));
            }
            let mut score = score_bias[0];
            for j in 0..d {
                score = activated[j].mul_add(score_weight[j], score);
            }
            logits.push(score);
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|x| (x - maximum).exp()).collect();
        let total: f32 = exp.iter().sum();
        let probabilities: Vec<f32> = exp.iter().map(|x| x / total).collect();
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
        let mut features = Vec::with_capacity(d + 4);
        features.extend_from_slice(&hidden[..d]);
        features.extend_from_slice(&[p1, p1 - p2, entropy, k / 255.0]);
        let act_weight = load("act_head.0.weight")?;
        let act_bias = load("act_head.0.bias")?;
        let out_weight = load("act_head.2.weight")?;
        let out_bias = load("act_head.2.bias")?;
        let mut middle = vec![0f32; 256];
        for out in 0..256 {
            let weight = &act_weight[out * (d + 4)..(out + 1) * (d + 4)];
            let mut sum = act_bias[out];
            for j in 0..(d + 4) {
                sum = features[j].mul_add(weight[j], sum);
            }
            middle[out] = 0.5 * sum * (1.0 + libm::erff(sum * std::f32::consts::FRAC_1_SQRT_2));
        }
        let mut act = [0f32; 2];
        for out in 0..2 {
            let mut sum = out_bias[out];
            for j in 0..256 {
                sum = middle[j].mul_add(out_weight[out * 256 + j], sum);
            }
            act[out] = sum;
        }
        if let Some(directory) = &self.reference {
            for (name, values) in [
                ("logits", logits.as_slice()),
                ("act_logits", act.as_slice()),
            ] {
                let expected = fs::read(directory.join(format!("{name}.f32")))?;
                let reference: Vec<f32> = expected
                    .chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                    .collect();
                for (i, &value) in values.iter().enumerate() {
                    eprintln!(
                        "{name}[{i}]: GPU={value:.6} oracle={:.6} abs={:.6}",
                        reference[i],
                        (value - reference[i]).abs()
                    );
                }
            }
        }
        Ok((logits, act))
    }

    fn compare(&self, name: &str, actual: &Buffer) -> Result<()> {
        let Some(directory) = &self.reference else {
            return Ok(());
        };
        let path = directory.join(format!("{name}.f32"));
        if !path.exists() {
            return Ok(());
        }
        let expected = fs::read(&path)?;
        if expected.len() != actual.len() {
            bail!(
                "{name} reference size {} != GPU size {}",
                expected.len(),
                actual.len()
            )
        }
        let mut output = vec![0u8; expected.len()];
        actual.read(&mut output)?;
        let mut max_error = 0f32;
        let mut max_index = 0usize;
        let mut sum_error = 0f64;
        for (i, (a, b)) in output
            .chunks_exact(4)
            .zip(expected.chunks_exact(4))
            .enumerate()
        {
            let actual = f32::from_le_bytes(a.try_into().unwrap());
            let expected = f32::from_le_bytes(b.try_into().unwrap());
            let error = (actual - expected).abs();
            if error > max_error {
                max_error = error;
                max_index = i;
            }
            sum_error += error as f64;
        }
        eprintln!(
            "{name}: max_abs={max_error:.6} mean_abs={:.6} at index {max_index}",
            sum_error / (expected.len() / 4) as f64
        );
        Ok(())
    }
}
