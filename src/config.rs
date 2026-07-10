use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCategories {
    pub coding: CategoryModels,
    pub analysis: CategoryModels,
    pub creative: CategoryModels,
}

impl ModelCategories {
    /// Returns a flat list of all non-empty model IDs across all categories/tiers.
    pub fn flatten(&self) -> Vec<String> {
        let mut v = Vec::new();
        for m in [&self.coding, &self.analysis, &self.creative] {
            if let Some(ref id) = m.low { v.push(id.clone()); }
            if let Some(ref id) = m.med { v.push(id.clone()); }
            if let Some(ref id) = m.high { v.push(id.clone()); }
            if let Some(ref id) = m.max { v.push(id.clone()); }
        }
        v
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryModels {
    pub low: Option<String>,
    pub med: Option<String>,
    pub high: Option<String>,
    pub max: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub openrouter: Option<ProviderConfig>,
    #[serde(default)]
    pub anthropic: Option<ProviderConfig>,
    #[serde(default)]
    pub openai: Option<ProviderConfig>,
    #[serde(default)]
    pub gemini: Option<ProviderConfig>,
    #[serde(default)]
    pub opencode: Option<ProviderConfig>,
    #[serde(default)]
    pub custom: Option<ProviderConfig>,
    #[serde(default)]
    pub advanced: AdvancedConfig,
    #[serde(default)]
    pub permissions: PermissionsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub auto_route: Option<bool>,
    #[serde(default)]
    pub router_model_path: Option<String>,
    #[serde(default)]
    pub model_categories: Option<ModelCategories>,
    /// Kept for backward compatibility — newer configs use `model_categories`.
    #[serde(default)]
    pub preferred_models: Vec<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub fallback: Vec<String>,
}

impl ProviderConfig {
    /// Returns the effective model IDs: model_categories first, then preferred_models.
    pub fn effective_models(&self) -> Vec<String> {
        if let Some(ref cats) = self.model_categories {
            let flat = cats.flatten();
            if !flat.is_empty() {
                return flat;
            }
        }
        self.preferred_models.clone()
    }
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: String::new(),
            auto_route: None,
            router_model_path: None,
            preferred_models: Vec::new(),
            model_categories: None,
            base_url: None,
            fallback: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedConfig {
    #[serde(default = "default_api_format")]
    pub api_format: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,
    #[serde(default = "default_max_tool_output_storage")]
    pub max_tool_output_storage: usize,
    #[serde(default = "default_max_cost_per_session")]
    pub max_cost_per_session: f64,
    pub proxy: Option<String>,
    #[serde(default)]
    pub providers: HashMap<String, ProviderOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderOverride {
    #[serde(default)]
    pub api_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionsConfig {
    #[serde(default = "default_permission")]
    pub bash: String,
    #[serde(default = "default_permission")]
    pub write: String,
    #[serde(default = "default_permission")]
    pub read: String,
    #[serde(default = "default_permission")]
    pub glob: String,
    #[serde(default = "default_permission")]
    pub grep: String,
}

fn default_api_format() -> String {
    "auto".to_string()
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_temperature() -> f32 {
    0.0
}

fn default_timeout() -> u64 {
    120
}

fn default_max_context_tokens() -> usize {
    150_000
}

fn default_max_tool_output_storage() -> usize {
    4000
}

fn default_max_cost_per_session() -> f64 {
    0.0
}

fn default_permission() -> String {
    "ask".to_string()
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            api_format: default_api_format(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            timeout_secs: default_timeout(),
            max_context_tokens: default_max_context_tokens(),
            max_tool_output_storage: default_max_tool_output_storage(),
            max_cost_per_session: default_max_cost_per_session(),
            proxy: None,
            providers: HashMap::new(),
        }
    }
}

impl Default for PermissionsConfig {
    fn default() -> Self {
        Self {
            bash: default_permission(),
            write: default_permission(),
            read: "allow".to_string(),
            glob: "allow".to_string(),
            grep: "allow".to_string(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider: "openrouter".to_string(),
            openrouter: Some(ProviderConfig {
                api_key: String::new(),
                model: "anthropic/claude-sonnet-4".to_string(),
                auto_route: None,
                router_model_path: None,
                preferred_models: Vec::new(),
                model_categories: None,
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
                fallback: Vec::new(),
            }),
            anthropic: Some(ProviderConfig {
                api_key: String::new(),
                model: "claude-sonnet-4-20250514".to_string(),
                auto_route: None,
                router_model_path: None,
                preferred_models: Vec::new(),
                model_categories: None,
                base_url: None,
                fallback: Vec::new(),
            }),
            openai: Some(ProviderConfig {
                api_key: String::new(),
                model: "gpt-4o".to_string(),
                auto_route: None,
                router_model_path: None,
                preferred_models: Vec::new(),
                model_categories: None,
                base_url: Some("https://api.openai.com/v1".to_string()),
                fallback: Vec::new(),
            }),
            gemini: Some(ProviderConfig {
                api_key: String::new(),
                model: "gemini-2.5-pro".to_string(),
                auto_route: None,
                router_model_path: None,
                preferred_models: Vec::new(),
                model_categories: None,
                base_url: None,
                fallback: Vec::new(),
            }),
            opencode: Some(ProviderConfig {
                api_key: "public".to_string(),
                model: "big-pickle".to_string(),
                auto_route: None,
                router_model_path: None,
                preferred_models: Vec::new(),
                model_categories: None,
                base_url: Some("https://opencode.ai/zen/v1".to_string()),
                fallback: Vec::new(),
            }),
            custom: None,
            advanced: AdvancedConfig::default(),
            permissions: PermissionsConfig::default(),
        }
    }
}

impl Config {
    pub fn default_path() -> anyhow::Result<PathBuf> {
        let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not find home directory"))?;
        Ok(home.join("vibe").join("config").join("vibe.json"))
    }

    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read config at {}: {}", path.display(), e))?;
        let config: Config = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("invalid config JSON: {}", e))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let provider = self.provider.as_str();
        let valid_formats = ["auto", "openai", "anthropic", "google", "gemini"];
        if !valid_formats.contains(&self.advanced.api_format.as_str()) {
            anyhow::bail!(
                "invalid api_format '{}' — must be one of: auto, openai, anthropic, google",
                self.advanced.api_format
            );
        }
        let has_provider = match provider {
            "openrouter" => self.openrouter.is_some(),
            "anthropic" => self.anthropic.is_some(),
            "openai" => self.openai.is_some(),
            "gemini" => self.gemini.is_some(),
            "opencode" => self.opencode.is_some(),
            "custom" => self.custom.is_some(),
            _ => false,
        };
        if !has_provider {
            anyhow::bail!(
                "provider '{}' is not configured — add an '{}' section to your config",
                provider,
                provider
            );
        }
        Ok(())
    }

    pub fn provider_config(&self) -> Option<&ProviderConfig> {
        match self.provider.as_str() {
            "openrouter" => self.openrouter.as_ref(),
            "anthropic" => self.anthropic.as_ref(),
            "openai" => self.openai.as_ref(),
            "gemini" => self.gemini.as_ref(),
            "opencode" => self.opencode.as_ref(),
            "custom" => self.custom.as_ref(),
            _ => None,
        }
    }

    pub fn resolved_api_format(&self) -> &str {
        if let Some(overrides) = self.advanced.providers.get(&self.provider) {
            if let Some(format) = &overrides.api_format {
                return format;
            }
        }
        &self.advanced.api_format
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> Config {
        Config {
            provider: "openai".to_string(),
            openai: Some(ProviderConfig {
                api_key: "sk-test".to_string(),
                model: "gpt-4o".to_string(),
                auto_route: None,
                router_model_path: None,
                preferred_models: Vec::new(),
                model_categories: None,
                base_url: None,
                fallback: Vec::new(),
            }),
            ..Config::default()
        }
    }

    #[test]
    fn test_validate_valid() {
        let cfg = make_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_api_format() {
        let mut cfg = make_config();
        cfg.advanced.api_format = "bad-format".to_string();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_missing_provider_config() {
        let mut cfg = make_config();
        cfg.openai = None;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_provider_config_returns_some() {
        let cfg = make_config();
        assert!(cfg.provider_config().is_some());
    }

    #[test]
    fn test_provider_config_returns_none_for_unknown() {
        let mut cfg = make_config();
        cfg.provider = "nonexistent".to_string();
        assert!(cfg.provider_config().is_none());
    }

    #[test]
    fn test_resolved_api_format_default() {
        let cfg = make_config();
        assert_eq!(cfg.resolved_api_format(), "auto");
    }

    #[test]
    fn test_default_path_returns_result() {
        let path = Config::default_path();
        assert!(path.is_ok());
        let p = path.unwrap();
        assert!(p.ends_with("vibe/config/vibe.json"));
    }
}
