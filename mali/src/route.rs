use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

pub struct Route {
    pub model: &'static str,
    pub metadata: Value,
}

fn repo(model: &str) -> &'static str {
    match model {
        "multilingual" => "convaiinnovations/laya/multilingual",
        "typed-decisions" => "convaiinnovations/laya/typed-decisions",
        _ => "convaiinnovations/laya",
    }
}

fn normalise_name(name: &str) -> Option<&'static str> {
    match name.trim().to_lowercase().as_str() {
        "english" | "en" | "laya" | "default" => Some("english"),
        "multilingual"
        | "multi"
        | "ml"
        | "laya-multilingual"
        | "convaiinnovations/laya-multilingual" => Some("multilingual"),
        "typed-decisions"
        | "typed_decisions"
        | "typed"
        | "decisions"
        | "laya-typed-decisions"
        | "convaiinnovations/laya-typed-decisions" => Some("typed-decisions"),
        _ => None,
    }
}

fn route(model: &'static str, reason: String, detection: Value, workflow: Value) -> Route {
    Route {
        model,
        metadata: json!({
            "model":model,"repo":repo(model),"reason":reason,
            "detection":detection,"workflow":workflow
        }),
    }
}

fn lang_hint(value: &str) -> Option<bool> {
    let lower = value.trim().to_lowercase();
    let code = lower.split('.').next()?.replace('_', "-");
    let primary = code.split('-').next()?;
    if primary.is_empty() || matches!(primary, "c" | "posix" | "und" | "zxx" | "mul") {
        None
    } else {
        Some(matches!(primary, "en" | "eng" | "english"))
    }
}

fn workflow(questions: &Map<String, Value>) -> Value {
    let actual: HashSet<&str> = questions.keys().map(String::as_str).collect();
    for (name, expected) in [
        (
            "agent_trace_observability",
            ["action", "needs_review", "outcome", "risk", "urgency"],
        ),
        (
            "customer_service",
            ["action", "category", "churn_risk", "needs_human", "urgency"],
        ),
        (
            "invoice_processing",
            [
                "discrepancy_severity",
                "disposition",
                "duplicate",
                "matches_order",
                "urgency",
            ],
        ),
        (
            "security_incidents",
            [
                "credential_compromise",
                "disposition",
                "severity",
                "true_positive",
                "urgency",
            ],
        ),
    ] {
        if actual == expected.into_iter().collect() {
            return json!(name);
        }
    }
    Value::Null
}

fn leaves(value: &Value, depth: usize, result: &mut Vec<String>) {
    if depth > 6 {
        return;
    }
    match value {
        Value::String(s) => result.push(s.clone()),
        Value::Array(items) => {
            for item in items {
                leaves(item, depth + 1, result);
            }
        }
        Value::Object(map) => {
            for item in map.values() {
                leaves(item, depth + 1, result);
            }
        }
        _ => (),
    }
}

fn script(cp: u32) -> &'static str {
    match cp {
        0..=0x02AF | 0x1E00..=0x1EFF | 0xFF21..=0xFF3A | 0xFF41..=0xFF5A => "latin",
        0x0370..=0x03FF | 0x1F00..=0x1FFF => "greek",
        0x0400..=0x052F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => "cyrillic",
        0x0530..=0x058F => "armenian",
        0x0590..=0x05FF => "hebrew",
        0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => {
            "arabic"
        }
        0x0900..=0x097F | 0xA8E0..=0xA8FF => "devanagari",
        0x0980..=0x09FF => "bengali",
        0x0A00..=0x0A7F => "gurmukhi",
        0x0A80..=0x0AFF => "gujarati",
        0x0B00..=0x0B7F => "oriya",
        0x0B80..=0x0BFF => "tamil",
        0x0C00..=0x0C7F => "telugu",
        0x0C80..=0x0CFF => "kannada",
        0x0D00..=0x0D7F => "malayalam",
        0x0D80..=0x0DFF => "sinhala",
        0x0E00..=0x0E7F => "thai",
        0x0E80..=0x0EFF => "lao",
        0x0F00..=0x0FFF => "tibetan",
        0x1000..=0x109F => "myanmar",
        0x10A0..=0x10FF => "georgian",
        0x1200..=0x137F => "ethiopic",
        0x1780..=0x17FF => "khmer",
        0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF => "hangul",
        0x3040..=0x30FF | 0x31F0..=0x31FF => "kana",
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => "han",
        _ => "other",
    }
}

fn language(text: &str) -> (Option<String>, bool, f64) {
    let data: Value = serde_json::from_str(include_str!("../data/lang_stopwords.json"))
        .expect("embedded stopwords");
    let stops = data["stopwords"].as_object().expect("stopwords object");
    let diacritics = data["non_english_diacritics"]
        .as_str()
        .expect("diacritics string");
    let lower = text.to_lowercase();
    let words: Vec<String> = lower
        .split(|c: char| !c.is_alphabetic())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    let diac = lower.chars().filter(|c| diacritics.contains(*c)).count();
    let rate = diac as f64 / lower.chars().count().max(1) as f64;
    let non_english = rate >= 0.02;
    if words.len() < 4 {
        return (None, non_english, rate);
    }
    let sets: BTreeMap<&str, HashSet<&str>> = stops
        .iter()
        .map(|(code, words)| {
            (
                code.as_str(),
                words
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_str().unwrap())
                    .collect(),
            )
        })
        .collect();
    let scores: BTreeMap<&str, usize> = sets
        .iter()
        .map(|(code, set)| {
            (
                *code,
                words.iter().filter(|w| set.contains(w.as_str())).count(),
            )
        })
        .collect();
    let en = *scores.get("en").unwrap_or(&0);
    let mut best = (None, 0usize);
    for (code, set) in &sets {
        if *code == "en" {
            continue;
        }
        let distinctive = words.iter().any(|w| {
            set.contains(w.as_str())
                && sets
                    .iter()
                    .filter(|(_, other)| other.contains(w.as_str()))
                    .count()
                    == 1
        });
        let score = *scores.get(code).unwrap_or(&0);
        if distinctive && score > best.1 {
            best = (Some(*code), score);
        }
    }
    let named = if let (Some(code), score) = best {
        if score >= 2 && (score >= en + 2 || (non_english && score >= en)) {
            Some(code.to_string())
        } else if en > 0 && !non_english {
            Some("en".into())
        } else {
            None
        }
    } else if en > 0 && !non_english {
        Some("en".into())
    } else {
        None
    };
    (named, non_english, rate)
}

fn detect(state: &Value) -> Value {
    let mut parts = Vec::new();
    leaves(state, 0, &mut parts);
    let text: String = parts.join(" ").chars().take(4000).collect();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for ch in text.chars().filter(|c| c.is_alphabetic()) {
        *counts.entry(script(ch as u32)).or_default() += 1;
    }
    let total: usize = counts.values().sum();
    let latin = *counts.get("latin").unwrap_or(&0);
    let dominant = if total == 0 {
        "unknown"
    } else {
        counts
            .iter()
            .max_by(|a, b| {
                a.1.cmp(b.1)
                    .then_with(|| (a.0 != &"latin").cmp(&(b.0 != &"latin")))
            })
            .unwrap()
            .0
    };
    let mut profile = Map::new();
    if latin > 0 {
        profile.insert("latin".into(), json!(latin as f64 / total as f64));
    }
    for (name, count) in &counts {
        if *name != "latin" {
            profile.insert((*name).into(), json!(*count as f64 / total as f64));
        }
    }
    let non_latin = if total > 0 {
        (1.0 - latin as f64 / total as f64).max(0.0)
    } else {
        0.0
    };
    if dominant == "unknown" {
        return json!({"script":"unknown","script_profile":profile,"language":null,
        "is_english":true,"language_undecided":true,"diacritic_rate":0.0,"non_latin_fraction":0.0});
    }
    if dominant != "latin" {
        return json!({"script":dominant,"script_profile":profile,"language":null,
        "is_english":false,"language_undecided":true,"diacritic_rate":0.0,"non_latin_fraction":round4(non_latin)});
    }
    let (lang, looks_non_english, diac_rate) = language(&text);
    let undecided = lang.is_none();
    let english = lang.as_deref() == Some("en") || (undecided && !looks_non_english);
    json!({"script":"latin","script_profile":profile,"language":lang,"is_english":english,
        "language_undecided":undecided,"diacritic_rate":round4(diac_rate),
        "non_latin_fraction":round4(non_latin)})
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

pub fn decide(request: &Value, questions: &Map<String, Value>) -> Route {
    if let Some(raw) = request.get("model").and_then(Value::as_str) {
        if let Some(model) = normalise_name(raw) {
            return route(
                model,
                format!("explicit model='{raw}'"),
                Value::Null,
                Value::Null,
            );
        }
    }
    if let Some(raw) = request.get("task").and_then(Value::as_str) {
        let target = if raw.replace('-', "_") == "typed_decisions" {
            "typed-decisions"
        } else {
            raw
        };
        if let Some(model) = normalise_name(target) {
            return route(
                model,
                format!("explicit task='{raw}'"),
                Value::Null,
                Value::Null,
            );
        }
    }
    let workflow = workflow(questions);
    let auto_task = std::env::var("LAYA_AUTO_TASK").ok().as_deref() == Some("1");
    if auto_task && !workflow.is_null() {
        let reason = format!(
            "question ids match the '{}' typed-decisions workflow",
            workflow.as_str().unwrap()
        );
        return route("typed-decisions", reason, Value::Null, workflow);
    }
    for field in ["lang", "lang_guess"] {
        if let Some(raw) = request.get(field).and_then(Value::as_str) {
            if let Some(english) = lang_hint(raw) {
                let model = if english { "english" } else { "multilingual" };
                let reason = if field == "lang" {
                    format!("explicit lang='{raw}'")
                } else {
                    format!(
                        "lang_guess: the caller identified this as {} text",
                        if english { "English" } else { "non-English" }
                    )
                };
                return route(model, reason, Value::Null, workflow);
            }
        }
    }
    let state = request.get("state").unwrap_or(&Value::Null);
    let detection = detect(state);
    let script = detection["script"].as_str().unwrap_or("unknown");
    let default = if std::env::var("LAYA_DEFAULT_MODEL").as_deref() == Ok("multilingual") {
        "multilingual"
    } else {
        "english"
    };
    let (model, reason) = if script == "unknown" {
        (
            default,
            format!("no letters detected in state; using default ({default})"),
        )
    } else if script != "latin" {
        let fraction = detection["non_latin_fraction"].as_f64().unwrap_or(1.0);
        ("multilingual", format!("non-Latin script ({script}, {:.0}% of letters); the English checkpoint cannot read it", 100.0 * fraction))
    } else if detection["is_english"] == false {
        let lang = detection["language"].as_str();
        let reason = if let Some(lang) = lang {
            format!("Latin script but language looks like '{lang}', not English")
        } else {
            format!("Latin script, language not identified but {:.0}% non-English letters; not safe for the English checkpoint", 100.0 * detection["diacritic_rate"].as_f64().unwrap_or(0.0))
        };
        ("multilingual", reason)
    } else if detection["language_undecided"] == true {
        (default, format!("Latin script, language not identified and no non-English letters; using default ({default})"))
    } else {
        ("english", "English Latin text".into())
    };
    route(model, reason, detection, workflow)
}
