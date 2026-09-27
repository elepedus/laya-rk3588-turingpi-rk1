use crate::rknn::Rknn;
use anyhow::{Context, Result};
use std::{cell::Cell, path::Path, time::Instant};

pub struct NpuEncoder {
    layers: Vec<Rknn>,
    hidden: usize,
    sequence: usize,
    calls: Cell<u64>,
}

impl NpuEncoder {
    pub fn load(
        library: &Path,
        directory: &Path,
        hidden: usize,
        sequence: usize,
        layer_count: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(layer_count);
        let core_mask: i32 = std::env::var("LAYA_RKNN_CORE_MASK")
            .ok()
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(7);
        for index in 0..layer_count {
            let filename = format!("layer{index:02}/encoder_{index:02}.rknn");
            let layer = Rknn::load(library, &directory.join(&filename))
                .with_context(|| format!("loading NPU {filename}"))?;
            layer.set_core_mask(core_mask)?;
            anyhow::ensure!(
                layer.io_count()? == (2, 1),
                "NPU {filename} must have hidden and mask inputs and one output"
            );
            layers.push(layer);
        }
        anyhow::ensure!(!layers.is_empty(), "NPU encoder has no layers");
        let (api, driver) = layers[0].sdk_version()?;
        eprintln!(
            "loaded {} Laya encoder layers on RK3588 NPU; API {api}; driver {driver}",
            layers.len()
        );
        Ok(Self {
            layers,
            hidden,
            sequence,
            calls: Cell::new(0),
        })
    }

    pub fn calls(&self) -> u64 {
        self.calls.get()
    }

    pub fn infer(&self, embedding: &[f32], valid: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            embedding.len() == self.sequence * self.hidden,
            "NPU embedding has {} values; expected {}",
            embedding.len(),
            self.sequence * self.hidden
        );
        anyhow::ensure!(
            valid > 0 && valid <= self.sequence,
            "invalid NPU token length {valid}"
        );
        let mut mask = vec![0f32; self.sequence];
        mask[valid..].fill(-10000.0);
        let mut hidden = embedding.to_vec();
        let started = Instant::now();
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.infer_f32(&[&hidden, &mask], &[3, 1])?;
            anyhow::ensure!(
                hidden.len() == self.sequence * self.hidden,
                "NPU layer {index} returned {} values",
                hidden.len()
            );
            if std::env::var_os("LAYA_NPU_TIMING").is_some() {
                eprintln!("NPU_LAYER index={index} us={}", layer.last_duration_us()?);
            }
        }
        if std::env::var_os("LAYA_NPU_TIMING").is_some() {
            eprintln!(
                "NPU_ENCODER wall_ms={:.3}",
                started.elapsed().as_secs_f64() * 1e3
            );
        }
        self.calls.set(self.calls.get() + 1);
        Ok(hidden)
    }
}
