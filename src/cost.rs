use std::collections::HashMap;

use std::sync::LazyLock;

/// (input per 1M tokens, output per 1M tokens)
type Pricing = (f64, f64);

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
    // Try exact match first, then check if any key is a suffix
    if let Some(&price) = MODEL_PRICES.get(model) {
        return price;
    }
    // Check prefix matches (e.g. "gpt-4o-..." matches "gpt-4o")
    for (key, &price) in MODEL_PRICES.iter() {
        if model.starts_with(key) || model.ends_with(key) {
            return price;
        }
        // Also check after stripping provider prefix (e.g. "openai/gpt-4o" -> "gpt-4o")
        if let Some(short) = model.split('/').last() {
            if short == *key || short.starts_with(key) {
                return price;
            }
        }
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
        _ => vec![],
    }
}

/// Returns all known models whose name contains the given provider prefix.
pub fn models_for_provider(provider: &str) -> Vec<&'static str> {
    let prefixes: &[&str] = match provider {
        "openrouter" => &["anthropic/", "openai/", "google/", "deepseek/", "meta-llama/",
                          "mistralai/", "qwen/", "cohere/"],
        "anthropic" => &["claude-"],
        "openai" => &["gpt-", "o1", "o3"],
        "gemini" => &["gemini-"],
        "opencode" => &["big-pickle"],
        _ => return Vec::new(),
    };
    MODEL_ITER.iter()
        .filter(|(name, _)| prefixes.iter().any(|p| name.starts_with(p)))
        .map(|(name, _)| *name)
        .collect()
}
