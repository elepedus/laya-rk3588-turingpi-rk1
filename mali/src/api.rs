use crate::model::GpuModel;
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};
use std::{fs, path::Path, time::Instant};
use tokenizers::Tokenizer;

const MAX_OPTIONS: usize = 32;

pub trait InferenceModel {
    fn run(&self, ids: &[i32], qtype: i32) -> Result<Vec<f32>>;
    fn decode(&self, hidden: &[f32], markers: &[usize]) -> Result<(Vec<f32>, [f32; 2])>;
}

impl InferenceModel for GpuModel {
    fn run(&self, ids: &[i32], qtype: i32) -> Result<Vec<f32>> {
        GpuModel::run(self, ids, qtype)
    }

    fn decode(&self, hidden: &[f32], markers: &[usize]) -> Result<(Vec<f32>, [f32; 2])> {
        GpuModel::decode(self, hidden, markers)
    }
}

pub struct Api {
    model: Box<dyn InferenceModel>,
    tokenizer: Tokenizer,
    mask_text: String,
    cls: u32,
    sep: u32,
    mask: u32,
    config: Value,
    max_len: usize,
    head_max_len: usize,
}

struct Question {
    kind: &'static str,
    qtype: i32,
    labels: Vec<String>,
    criteria: Vec<Value>,
    ids: Vec<i32>,
    markers: Vec<usize>,
}

fn python_json(value: &Value) -> String {
    match value {
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(key).unwrap(),
                        python_json(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(python_json).collect::<Vec<_>>().join(", ")
        ),
        _ => serde_json::to_string(value).unwrap(),
    }
}

fn rendered(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| python_json(value))
}

fn options(q: &Value) -> Result<(&'static str, i32, Vec<String>, Vec<Value>, Vec<String>)> {
    let kind = q
        .get("type")
        .and_then(Value::as_str)
        .context("question.type is required")?;
    let criteria = q.get("criteria");
    match kind {
        "choice" => {
            let mut labels = Vec::new();
            let mut values = Vec::new();
            let mut texts = Vec::new();
            if let Some(map) = criteria.and_then(Value::as_object) {
                for (key, value) in map {
                    labels.push(key.to_owned());
                    values.push(value.clone());
                    texts.push(if value.is_null() || value == "" {
                        key.to_owned()
                    } else {
                        format!("{key}: {}", rendered(value))
                    });
                }
            } else if let Some(array) = criteria.and_then(Value::as_array) {
                for value in array {
                    let label = value
                        .as_str()
                        .context("choice criteria array must contain strings")?;
                    labels.push(label.to_owned());
                    values.push(Value::Null);
                    texts.push(label.to_owned());
                }
            } else {
                bail!("choice criteria must be an object or array")
            }
            if labels.is_empty() || labels.len() > MAX_OPTIONS {
                bail!("choice needs 1 to {MAX_OPTIONS} options")
            }
            Ok(("choice", 0, labels, values, texts))
        }
        "score" => {
            let list = criteria
                .and_then(Value::as_array)
                .context("score criteria must be an array")?;
            if list.is_empty() || list.len() > MAX_OPTIONS {
                bail!("score needs 1 to {MAX_OPTIONS} levels")
            }
            let labels = (0..list.len()).map(|i| i.to_string()).collect();
            let texts = list
                .iter()
                .enumerate()
                .map(|(i, value)| format!("level {i}: {}", rendered(value)))
                .collect();
            Ok(("score", 1, labels, list.clone(), texts))
        }
        "noul" => {
            let label_map = q.get("labels").and_then(Value::as_object);
            let false_label = label_map
                .and_then(|x| x.get("false"))
                .and_then(Value::as_str)
                .unwrap_or("false");
            let true_label = label_map
                .and_then(|x| x.get("true"))
                .and_then(Value::as_str)
                .unwrap_or("true");
            if false_label.trim().is_empty()
                || true_label.trim().is_empty()
                || false_label == true_label
            {
                bail!("invalid noul labels")
            }
            let rubric = criteria.and_then(Value::as_object);
            let false_text = rubric
                .and_then(|x| x.get("false"))
                .filter(|v| !v.is_null() && *v != "")
                .map(rendered)
                .unwrap_or_else(|| "no, the statement does not hold".into());
            let true_text = rubric
                .and_then(|x| x.get("true"))
                .filter(|v| !v.is_null() && *v != "")
                .map(rendered)
                .unwrap_or_else(|| "yes, the statement holds".into());
            Ok((
                "noul",
                2,
                vec!["false".into(), "true".into()],
                vec![],
                vec![
                    format!("{false_label}: {false_text}"),
                    format!("{true_label}: {true_text}"),
                ],
            ))
        }
        _ => bail!("unsupported question type {kind}"),
    }
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponents: Vec<f64> = values.iter().map(|x| (x - maximum).exp()).collect();
    let total: f64 = exponents.iter().sum();
    exponents.into_iter().map(|x| x / total).collect()
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

impl Api {
    pub fn new<M: InferenceModel + 'static>(model: M, model_dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer/tokenizer.json"))
            .map_err(|error| anyhow!(error.to_string()))?;
        let token_config: Value = serde_json::from_slice(&fs::read(
            model_dir.join("tokenizer/tokenizer_config.json"),
        )?)?;
        let config: Value =
            serde_json::from_slice(&fs::read(model_dir.join("rl_agent_config.json"))?)?;
        let special = |name: &str| -> Result<String> {
            Ok(token_config[name]
                .as_str()
                .with_context(|| format!("missing tokenizer {name}"))?
                .to_owned())
        };
        let cls_text = special("cls_token")?;
        let sep_text = special("sep_token")?;
        let mask_text = special("mask_token")?;
        let token = |name: &str| {
            tokenizer
                .token_to_id(name)
                .with_context(|| format!("missing token {name}"))
        };
        let max_len = config["max_len"].as_u64().unwrap_or(512) as usize;
        let head_max_len = config["head_max_len"].as_u64().unwrap_or(192) as usize;
        Ok(Self {
            cls: token(&cls_text)?,
            sep: token(&sep_text)?,
            mask: token(&mask_text)?,
            mask_text,
            tokenizer,
            config,
            max_len,
            head_max_len,
            model: Box::new(model),
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|error| anyhow!(error.to_string()))
    }

    fn prepare(&self, state: &Value, definition: &Value) -> Result<Question> {
        let (kind, qtype, labels, criteria, texts) = options(definition)?;
        let instructions = definition
            .get("instructions")
            .context("question.instructions is required")?;
        let instructions = instructions
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| python_json(instructions))
            .replace(&self.mask_text, " ");
        let mut head = self.encode(&format!("{kind} question: {instructions}"))?;
        let mut option_ids = Vec::new();
        for text in &texts {
            let mut ids = vec![self.mask];
            ids.extend(
                self.encode(&format!(" {}", text.replace(&self.mask_text, " ")))?
                    .into_iter()
                    .take(48),
            );
            option_ids.push(ids);
        }
        let mut budget = self
            .head_max_len
            .saturating_sub(option_ids.iter().map(Vec::len).sum::<usize>());
        if budget < 16 {
            let per = ((self.head_max_len - 16) / option_ids.len()).max(4);
            for ids in &mut option_ids {
                ids.truncate(per);
            }
            budget = self
                .head_max_len
                .saturating_sub(option_ids.iter().map(Vec::len).sum::<usize>());
        }
        head.truncate(budget.max(8));
        let mut ids = vec![self.cls as i32];
        ids.extend(head.into_iter().map(|id| id as i32));
        ids.push(self.sep as i32);
        let mut markers = Vec::new();
        for option in option_ids {
            markers.push(ids.len());
            ids.extend(option.into_iter().map(|id| id as i32));
        }
        ids.push(self.sep as i32);
        let state_text = state
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| python_json(state))
            .replace(&self.mask_text, " ");
        let mut state_ids = self.encode(&state_text)?;
        let room = self.max_len.saturating_sub(ids.len() + 1);
        if state.is_array() && state_ids.len() > room {
            state_ids = state_ids.split_off(state_ids.len() - room);
        }
        state_ids.truncate(room);
        ids.extend(state_ids.into_iter().map(|id| id as i32));
        ids.push(self.sep as i32);
        if ids.len() > self.max_len {
            bail!("question exceeds {} tokens", self.max_len)
        }
        Ok(Question {
            kind,
            qtype,
            labels,
            criteria,
            ids,
            markers,
        })
    }

    fn temperature(&self, question: &Question) -> f64 {
        let k = question.labels.len();
        let bucket = if k <= 2 {
            "2"
        } else if k <= 5 {
            "3-5"
        } else if k <= 10 {
            "6-10"
        } else {
            "11+"
        };
        let key = format!("{}:{bucket}", question.kind);
        self.config
            .get("temperature_by_options")
            .and_then(|v| v.get(&key))
            .and_then(Value::as_f64)
            .or_else(|| {
                self.config
                    .get("temperature")
                    .and_then(|v| v.get(question.qtype as usize))
                    .and_then(Value::as_f64)
            })
            .unwrap_or(1.0)
            .clamp(0.5, 5.0)
    }

    fn answer(&self, question: &Question, logits: &[f32], act: [f32; 2]) -> Result<Value> {
        let k = question.labels.len();
        let temp = self.temperature(question);
        let values: Vec<f64> = logits.iter().map(|value| *value as f64 / temp).collect();
        let probabilities = softmax(&values);
        let best = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let answer_conf = round4(probabilities[best]);
        let entropy = if k < 2 {
            0.0
        } else {
            -probabilities
                .iter()
                .map(|p| p * p.max(1e-12).ln())
                .sum::<f64>()
                / (k as f64).ln()
        };
        let confidence = round4((1.0 - entropy).clamp(0.0, 1.0));
        let action =
            json!({"act_probability": round4(softmax(&[act[0] as f64, act[1] as f64])[0])});
        match question.kind {
            "choice" => {
                let distribution: Map<String, Value> = question
                    .labels
                    .iter()
                    .zip(&probabilities)
                    .map(|(label, value)| (label.clone(), json!(round4(*value))))
                    .collect();
                Ok(
                    json!({"type":"choice","choice":question.labels[best],"probabilities":distribution,
                    "confidence":confidence,"answer_confidence":answer_conf,"action":action}),
                )
            }
            "score" => {
                let distribution: Map<String, Value> = question
                    .labels
                    .iter()
                    .zip(&probabilities)
                    .map(|(label, value)| (label.clone(), json!(round4(*value))))
                    .collect();
                let legend: Map<String, Value> = question
                    .labels
                    .iter()
                    .zip(&question.criteria)
                    .map(|(label, value)| (label.clone(), value.clone()))
                    .collect();
                let score: f64 = probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p)
                    .sum();
                Ok(json!({"type":"score","score":round4(score),"legend":legend,
                    "probabilities":distribution,"confidence":confidence,
                    "answer_confidence":answer_conf,"action":action}))
            }
            _ => Ok(json!({"type":"noul","noul":round4(probabilities[1]),
                "confidence":answer_conf,"answer_confidence":answer_conf,"action":action})),
        }
    }

    pub fn predict(&self, request: &Value) -> Result<Value> {
        let state = request
            .get("state")
            .filter(|v| !v.is_null())
            .context("state is required")?;
        let questions = request
            .get("questions")
            .and_then(Value::as_object)
            .context("questions must be an object")?;
        let text = state
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| python_json(state));
        if text.chars().count() > 4_000 {
            bail!("state exceeds 4000 characters")
        }
        if questions.len() > 16 {
            bail!("too many questions")
        }
        let mut answers = Map::new();
        let mut used = 0;
        for (name, definition) in questions {
            let prepared_at = Instant::now();
            let question = self
                .prepare(state, definition)
                .with_context(|| format!("question {name}"))?;
            let prepare_ms = prepared_at.elapsed().as_secs_f64() * 1000.0;
            if let Ok(reference_dir) = std::env::var("LAYA_ORACLE_DIR") {
                let reference: Value = serde_json::from_slice(&fs::read(
                    Path::new(&reference_dir).join("input.json"),
                )?)?;
                let expected: Vec<i32> = reference["ids"]
                    .as_array()
                    .context("oracle missing ids")?
                    .iter()
                    .map(|value| value.as_i64().unwrap() as i32)
                    .collect();
                if question.ids != expected {
                    let first = question.ids.iter().zip(&expected).position(|(a, b)| a != b);
                    bail!(
                        "token IDs differ from oracle: lengths {} and {}, first mismatch {first:?}",
                        question.ids.len(),
                        expected.len()
                    );
                }
            }
            used += question.ids.len();
            let run_at = Instant::now();
            let hidden = self.model.run(&question.ids, question.qtype)?;
            let run_ms = run_at.elapsed().as_secs_f64() * 1000.0;
            let decode_at = Instant::now();
            let (logits, act) = self.model.decode(&hidden, &question.markers)?;
            let decode_ms = decode_at.elapsed().as_secs_f64() * 1000.0;
            if std::env::var_os("LAYA_NPU_TIMING").is_some() {
                eprintln!("API_STAGE question={name} tokens={} prepare_ms={prepare_ms:.3} run_ms={run_ms:.3} decode_ms={decode_ms:.3}", question.ids.len());
            }
            answers.insert(name.clone(), self.answer(&question, &logits, act)?);
        }
        Ok(json!({"model":"laya-rl-agent","answers":answers,
            "usage":{"input_tokens":used,"output_tokens":0}}))
    }
}
