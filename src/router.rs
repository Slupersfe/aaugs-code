use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use ort::value::Tensor;
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer,
    TruncationDirection, TruncationParams, TruncationStrategy,
};

static ROUTER: OnceLock<Router> = OnceLock::new();

pub struct Classification {
    pub target: String,
    pub target_confidence: f32,
    pub intensity: String,
    #[allow(dead_code)]
    pub intensity_confidence: f32,
    pub latency_ms: f64,
}

struct Router {
    session: Mutex<ort::session::Session>,
    tokenizer: Tokenizer,
    id2target: HashMap<i64, String>,
    id2intensity: HashMap<i64, String>,
}

pub fn init(model_dir: &Path) -> Result<()> {
    let onnx_path = model_dir.join("model_quantized.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let config_path = model_dir.join("config.json");

    for required in [&onnx_path, &tokenizer_path, &config_path] {
        if !required.exists() {
            anyhow::bail!(
                "required model file not found: {}",
                required.display()
            );
        }
    }

    // Suppress ONNX Runtime stdout logging (pollutes TUI display)
    if let Ok(env) = ort::environment::current() {
        env.set_log_level(ort::logging::LogLevel::Fatal);
    }

    tracing::info!("loading ONNX router model from {}", model_dir.display());

    let session = ort::session::Session::builder()?
        .commit_from_file(onnx_path)
        .context("failed to load ONNX session")?;

    let mut tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;

    tokenizer.with_truncation(Some(TruncationParams {
        max_length: 128,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
        direction: TruncationDirection::Right,
    })).map_err(|e| anyhow::anyhow!("failed to set truncation: {}", e))?;

    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(128),
        direction: PaddingDirection::Right,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".into(),
        pad_to_multiple_of: None,
    }));

    let config_content = std::fs::read_to_string(&config_path)
        .context("failed to read config.json")?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;

    let id2target = parse_label_map(&config["id2label"]);
    let id2intensity = parse_label_map(&config["id2intensity"]);

    let router = Router {
        session: Mutex::new(session),
        tokenizer,
        id2target,
        id2intensity,
    };
    ROUTER.set(router)
        .map_err(|_| anyhow::anyhow!("router already initialized"))?;

    tracing::info!("ONNX router model loaded successfully");
    Ok(())
}

fn parse_label_map(value: &serde_json::Value) -> HashMap<i64, String> {
    value.as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| {
                    let id = k.parse::<i64>().ok()?;
                    let label = v.as_str()?.to_string();
                    Some((id, label))
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn classify(prompt: &str) -> Option<Classification> {
    let router = ROUTER.get()?;
    let start = std::time::Instant::now();

    let encoding = router.tokenizer.encode(prompt, true).ok()?;

    let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
    let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

    let t_ids = Tensor::from_array(([1i64, 128], ids)).ok()?;
    let t_attn = Tensor::from_array(([1i64, 128], attention_mask)).ok()?;
    let t_type = Tensor::from_array(([1i64, 128], type_ids)).ok()?;

    let mut session = router.session.lock().ok()?;
    let outputs = session.run(
        ort::inputs! {
            "input_ids" => t_ids,
            "attention_mask" => t_attn,
            "token_type_ids" => t_type,
        }
    ).ok()?;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    let target_tensor = outputs.get("target_logits")?;
    let intensity_tensor = outputs.get("intensity_logits")?;

    let (_target_shape, target_slice) = target_tensor.try_extract_tensor::<f32>().ok()?;
    let (_intensity_shape, intensity_slice) = intensity_tensor.try_extract_tensor::<f32>().ok()?;

    let (t_idx, t_conf) = softmax_argmax(target_slice)?;
    let (i_idx, i_conf) = softmax_argmax(intensity_slice)?;

    let target = router.id2target.get(&t_idx)?.clone();
    let intensity = router.id2intensity.get(&i_idx)?.clone();

    Some(Classification {
        target,
        target_confidence: t_conf,
        intensity,
        intensity_confidence: i_conf,
        latency_ms,
    })
}

fn softmax_argmax(logits: &[f32]) -> Option<(i64, f32)> {
    if logits.is_empty() {
        return None;
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return None;
    }
    let idx = probs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i64)?;
    let conf = probs[idx as usize] / sum;
    Some((idx, conf))
}

pub fn is_loaded() -> bool {
    ROUTER.get().is_some()
}
