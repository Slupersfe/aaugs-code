use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
    pub base_url: Option<String>,
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
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
            }),
            anthropic: Some(ProviderConfig {
                api_key: String::new(),
                model: "claude-sonnet-4-20250514".to_string(),
                base_url: None,
            }),
            openai: Some(ProviderConfig {
                api_key: String::new(),
                model: "gpt-4o".to_string(),
                base_url: Some("https://api.openai.com/v1".to_string()),
            }),
            gemini: Some(ProviderConfig {
                api_key: String::new(),
                model: "gemini-2.5-pro".to_string(),
                base_url: None,
            }),
            opencode: Some(ProviderConfig {
                api_key: "public".to_string(),
                model: "big-pickle".to_string(),
                base_url: Some("https://opencode.ai/zen/v1".to_string()),
            }),
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
        let valid_formats = ["auto", "openai", "anthropic", "google"];
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
