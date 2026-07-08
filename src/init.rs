use std::path::PathBuf;

use anyhow::Context;
use dialoguer::{Confirm, Input, Select, theme::ColorfulTheme};
use serde_json::json;

pub fn run(path: &PathBuf) -> anyhow::Result<()> {
    if path.exists() {
        let overwrite = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt(format!(
                "Config already exists at {}. Overwrite?",
                path.display()
            ))
            .default(false)
            .interact()?;
        if !overwrite {
            println!("Init cancelled.");
            return Ok(());
        }
    }

    println!("┌─────────────────────────────────────┐");
    println!("│ aaugs-code config setup              │");
    println!("└─────────────────────────────────────┘");
    println!();

    let provider = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Default provider")
        .default(0)
        .item("OpenRouter (recommended)")
        .item("Anthropic")
        .item("OpenAI")
        .item("Gemini")
        .item("OpenCode")
        .interact()?;

    let provider_map = ["openrouter", "anthropic", "openai", "gemini", "opencode"];
    let provider_key = provider_map[provider];

    let api_key: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("API key")
        .interact_text()?;

    let model: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Model (enter for default)")
        .allow_empty(true)
        .interact_text()?;

    let default_models = [
        "anthropic/claude-sonnet-4",
        "claude-sonnet-4-20250514",
        "gpt-4o",
        "gemini-2.5-pro",
        "big-pickle",
    ];
    let resolved_model = if model.is_empty() {
        default_models[provider].to_string()
    } else {
        model
    };

    println!();
    println!("── Tool Permissions ──");

    let bash_perm = permission_select("bash (run shell commands)")?;
    let write_perm = permission_select("write (create/edit files)")?;

    let config = build_config(provider_key, &api_key, &resolved_model, &bash_perm, &write_perm)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .context("failed to create config directory")?;
    }

    let content = serde_json::to_string_pretty(&config)?;
    std::fs::write(path, &content).context("failed to write config")?;

    println!();
    println!("Config written to {}", path.display());
    println!("Run `aaugs-code` to start coding.");

    Ok(())
}

fn permission_select(label: &str) -> anyhow::Result<String> {
    let idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt(label)
        .default(0)
        .item("ask (prompt me each time)")
        .item("allow (auto-execute)")
        .item("deny (block)")
        .interact()?;
    Ok(["ask", "allow", "deny"][idx].to_string())
}

fn build_config(
    provider: &str,
    api_key: &str,
    model: &str,
    bash_perm: &str,
    write_perm: &str,
) -> anyhow::Result<serde_json::Value> {
    let mut cfg = json!({
        "provider": provider,
        "openrouter": {
            "api_key": "",
            "model": "anthropic/claude-sonnet-4",
            "base_url": "https://openrouter.ai/api/v1"
        },
        "anthropic": {
            "api_key": "",
            "model": "claude-sonnet-4-20250514"
        },
        "openai": {
            "api_key": "",
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1"
        },
        "gemini": {
            "api_key": "",
            "model": "gemini-2.5-pro"
        },
        "opencode": {
            "api_key": "public",
            "model": "big-pickle",
            "base_url": "https://opencode.ai/zen/v1"
        },
        "advanced": {
            "api_format": "auto",
            "max_tokens": 4096,
            "temperature": 0,
            "timeout_secs": 120,
            "proxy": null,
            "providers": {}
        },
        "permissions": {
            "bash": bash_perm,
            "write": write_perm,
            "read": "allow",
            "glob": "allow",
            "grep": "allow"
        }
    });

    // Set the chosen provider's api_key and model
    cfg[provider]["api_key"] = json!(api_key);
    cfg[provider]["model"] = json!(model);

    Ok(cfg)
}
