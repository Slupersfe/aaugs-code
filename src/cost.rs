use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

/// (input per 1M tokens, output per 1M tokens)
type Pricing = (f64, f64);

const CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(3600);

static LIVE_PRICES: LazyLock<Mutex<Option<(HashMap<String, Pricing>, Instant)>>> =
    LazyLock::new(|| Mutex::new(None));
static FETCH_SPAWNED: AtomicBool = AtomicBool::new(false);

/// Spawns a background thread to fetch live pricing from OpenRouter (runs once).
fn ensure_live_prices() {
    if FETCH_SPAWNED.swap(true, Ordering::Relaxed) {
        return;
    }
    std::thread::spawn(|| {
        match fetch_from_openrouter() {
            Ok(prices) => {
                tracing::info!("fetched live pricing for {} models", prices.len());
                let mut cache = LIVE_PRICES.lock().unwrap_or_else(|e| e.into_inner());
                *cache = Some((prices, Instant::now()));
            }
            Err(e) => {
                tracing::warn!("failed to fetch live pricing: {}", e);
            }
        }
    });
}

fn fetch_from_openrouter() -> Result<HashMap<String, Pricing>, reqwest::Error> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()?;
    let resp = client
        .get("https://openrouter.ai/api/v1/models")
        .send()?;
    let body: serde_json::Value = resp.json()?;
    let data = match body.get("data").and_then(|d| d.as_array()) {
        Some(arr) => arr,
        None => return Ok(HashMap::new()),
    };

    let mut prices = HashMap::new();
    for model in data {
        let id = match model.get("id").and_then(|i| i.as_str()) {
            Some(id) => id,
            None => continue,
        };
        let pricing = match model.get("pricing") {
            Some(p) => p,
            None => continue,
        };
        let prompt = pricing
            .get("prompt")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .map(|p| p * 1_000_000.0)
            .unwrap_or(0.0);
        let completion = pricing
            .get("completion")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .map(|p| p * 1_000_000.0)
            .unwrap_or(0.0);
        prices.insert(id.to_string(), (prompt, completion));
    }
    Ok(prices)
}

/// Look up pricing in the live cache (if populated).
fn live_model_cost(model: &str) -> Option<Pricing> {
    let cache = LIVE_PRICES.lock().unwrap_or_else(|e| e.into_inner());
    let (prices, fetched_at) = cache.as_ref()?;
    if fetched_at.elapsed() > CACHE_TTL {
        return None; // stale
    }

    // Exact match
    if let Some(&p) = prices.get(model) {
        return Some(p);
    }
    // Strip provider prefix
    if let Some(short) = model.split('/').next_back() {
        if let Some(&p) = prices.get(short) {
            return Some(p);
        }
    }
    // Prefix/suffix match
    let mut best: Option<(usize, Pricing)> = None;
    for (key, &price) in prices.iter() {
        let mut matched = false;
        if model.starts_with(key) || model.ends_with(key) {
            matched = true;
        }
        if let Some(short) = model.split('/').next_back() {
            if short == key || short.starts_with(key) {
                matched = true;
            }
        }
        if matched {
            match best {
                Some((best_len, _)) if key.len() <= best_len => {}
                _ => best = Some((key.len(), price)),
            }
        }
    }
    best.map(|(_, p)| p)
}

static MODEL_PRICES: LazyLock<HashMap<&'static str, Pricing>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    // OpenAI
    m.insert("gpt-4o", (2.50, 10.00));
    m.insert("gpt-4o-mini", (0.15, 0.60));
    m.insert("gpt-4o-2024-08-06", (2.50, 10.00));
    m.insert("gpt-4o-2024-05-13", (5.00, 15.00));
    m.insert("gpt-4-turbo", (10.00, 30.00));
    m.insert("gpt-4", (30.00, 60.00));
    m.insert("gpt-3.5-turbo", (0.50, 1.50));
    // Anthropic
    m.insert("claude-sonnet-4-20250514", (3.00, 15.00));
    m.insert("claude-sonnet-4", (3.00, 15.00));
    m.insert("claude-sonnet-4.5-20250514", (3.00, 15.00));
    m.insert("claude-opus-4-20250514", (15.00, 75.00));
    m.insert("claude-opus-4", (15.00, 75.00));
    m.insert("claude-haiku-3-5", (0.80, 4.00));
    m.insert("claude-3-5-sonnet", (3.00, 15.00));
    // OpenRouter
    m.insert("anthropic/claude-sonnet-4", (3.00, 15.00));
    m.insert("anthropic/claude-opus-4", (15.00, 75.00));
    m.insert("anthropic/claude-sonnet-4.5", (3.00, 15.00));
    m.insert("openai/gpt-4o", (2.50, 10.00));
    m.insert("openai/gpt-4o-mini", (0.15, 0.60));
    m.insert("openai/gpt-4o-2024-08-06", (2.50, 10.00));
    // Gemini
    m.insert("gemini-2.5-pro", (1.25, 5.00));
    m.insert("gemini-2.5-flash", (0.15, 0.60));
    m.insert("gemini-2.0-flash", (0.10, 0.40));
    // OpenCode Zen
    m.insert("big-pickle", (0.0, 0.0));
    m
});

pub static MODEL_ITER: LazyLock<Vec<(&'static str, (f64, f64))>> = LazyLock::new(|| {
    let mut v: Vec<(&'static str, (f64, f64))> = MODEL_PRICES.iter().map(|(k, v)| (*k, *v)).collect();
    v.sort_by(|a, b| a.0.cmp(b.0));
    v
});

pub fn model_cost(model: &str) -> Pricing {
    // Trigger background fetch of live pricing on first call
    ensure_live_prices();

    // Check live cache first
    if let Some(price) = live_model_cost(model) {
        return price;
    }

    // Try exact match first in hardcoded table
    if let Some(&price) = MODEL_PRICES.get(model) {
        return price;
    }
    // Check prefix/suffix matches — prefer longer (more specific) keys
    let mut best: Option<(usize, Pricing)> = None;
    for (key, &price) in MODEL_PRICES.iter() {
        let mut matched = false;
        if model.starts_with(key) || model.ends_with(key) {
            matched = true;
        }
        // Also check after stripping provider prefix (e.g. "openai/gpt-4o" -> "gpt-4o")
        if let Some(short) = model.split('/').next_back() {
            if short == *key || short.starts_with(key) {
                matched = true;
            }
        }
        if matched {
            match best {
                Some((best_len, _)) if key.len() <= best_len => {}
                _ => best = Some((key.len(), price)),
            }
        }
    }
    if let Some((_, price)) = best {
        return price;
    }
    // Default: assume $2/$10 (rough GPT-4o class)
    (2.0, 10.0)
}

pub fn calculate_cost(model: &str, prompt_tokens: u32, completion_tokens: u32) -> f64 {
    let (input_price, output_price) = model_cost(model);
    let input_cost = (prompt_tokens as f64 / 1_000_000.0) * input_price;
    let output_cost = (completion_tokens as f64 / 1_000_000.0) * output_price;
    input_cost + output_cost
}

pub fn favorite_models(provider: &str) -> Vec<&'static str> {
    match provider {
        "openrouter" => vec![
            "anthropic/claude-sonnet-4",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-chat",
        ],
        "anthropic" => vec![
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-haiku-3-5",
        ],
        "openai" => vec![
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "gemini" => vec![
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        "opencode" => vec!["big-pickle"],
        "custom" => vec![],
        _ => vec![],
    }
}

/// Returns all known models whose name starts with any of the given provider's prefixes.
pub fn models_for_provider(provider: &str) -> Vec<&'static str> {
    let prefixes: &[&str] = match provider {
        "openrouter" => &["anthropic/", "openai/", "google/", "deepseek/", "meta-llama/",
                          "mistralai/", "qwen/", "cohere/"],
        "anthropic" => &["claude-"],
        "openai" => &["gpt-", "o1", "o3"],
        "gemini" => &["gemini-"],
        "opencode" => &["big-pickle"],
        "custom" => &[],
        _ => return Vec::new(),
    };
    MODEL_ITER.iter()
        .filter(|(name, _)| prefixes.iter().any(|p| name.starts_with(p)))
        .map(|(name, _)| *name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_cost_exact() {
        let (inp, outp) = model_cost("gpt-4o");
        assert!((inp - 2.50).abs() < 0.001);
        assert!((outp - 10.00).abs() < 0.001);
    }

    #[test]
    fn test_model_cost_prefix() {
        let (inp, _outp) = model_cost("gpt-4o-mini-unknown");
        assert!((inp - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_model_cost_openrouter_strip() {
        let (inp, _outp) = model_cost("openai/gpt-4o");
        assert!((inp - 2.50).abs() < 0.001);
    }

    #[test]
    fn test_model_cost_unknown_defaults() {
        let (inp, _outp) = model_cost("totally-unknown-model-v42");
        assert!((inp - 2.00).abs() < 0.001);
    }

    #[test]
    fn test_calculate_cost_zero() {
        let cost = calculate_cost("gpt-4o", 0, 0);
        assert!((cost).abs() < 0.0001);
    }

    #[test]
    fn test_calculate_cost_1m_tokens() {
        // 1M input + 1M output tokens at gpt-4o prices = $2.50 + $10.00 = $12.50
        let cost = calculate_cost("gpt-4o", 1_000_000, 1_000_000);
        assert!((cost - 12.50).abs() < 0.001);
    }

    #[test]
    fn test_favorite_models_openrouter() {
        let favs = favorite_models("openrouter");
        assert!(favs.contains(&"anthropic/claude-sonnet-4"));
        assert!(favs.contains(&"openai/gpt-4o"));
    }

    #[test]
    fn test_favorite_models_unknown() {
        let favs = favorite_models("nonexistent");
        assert!(favs.is_empty());
    }

    #[test]
    fn test_models_for_provider_anthropic() {
        let models = models_for_provider("anthropic");
        assert!(models.contains(&"claude-sonnet-4-20250514"));
        assert!(models.contains(&"claude-opus-4"));
        assert!(!models.contains(&"gpt-4o"));
    }

    #[test]
    fn test_models_for_provider_unknown() {
        let models = models_for_provider("nope");
        assert!(models.is_empty());
    }
}
