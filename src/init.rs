use std::io::Write;
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
        .item("Custom")
        .interact()?;

    let provider_map = ["openrouter", "anthropic", "openai", "gemini", "opencode", "custom"];
    let provider_key = provider_map[provider];
    let is_custom = provider_key == "custom";

    let api_key: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("API key")
        .interact_text()?;

    let base_url = if is_custom {
        Some(ask_string("Base URL (e.g. https://api.example.com/v1)")?)
    } else {
        None
    };

    let api_format = if is_custom {
        let fmt = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("API format")
            .default(0)
            .item("OpenAI-compatible")
            .item("Anthropic")
            .item("Gemini")
            .interact()?;
        ["openai", "anthropic", "gemini"][fmt].to_string()
    } else {
        String::new()
    };

    let default_models = [
        "anthropic/claude-sonnet-4",
        "claude-sonnet-4-20250514",
        "gpt-4o",
        "gemini-2.5-pro",
        "big-pickle",
        "",
    ];
    let resolved_model = default_models[provider].to_string();

    println!();
    println!("── Model Categories ──");
    println!("(Optional — fill in any tier you want. Press Enter to skip.)");
    let coding = ask_category("Coding")?;
    let analysis = ask_category("Analysis")?;
    let creative = ask_category("Creative")?;

    println!();
    println!("── Tool Permissions ──");

    let bash_perm = permission_select("bash (run shell commands)")?;
    let write_perm = permission_select("write (create/edit files)")?;

    let config = build_config(
        provider_key, &api_key, &resolved_model, &api_format, &base_url,
        &coding, &analysis, &creative, &bash_perm, &write_perm,
    )?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .context("failed to create config directory")?;
    }

    // ONNX auto-router model download + config update
    println!();
    let model_dir = std::path::Path::new("model");
    let model_exists = model_dir.join("model_quantized.onnx").exists();
    let mut final_config = config;
    if model_exists {
        println!("ONNX auto-router model found at ./model/ — auto-routing enabled.");
        if let Some(pc) = final_config[provider_key].as_object_mut() {
            pc.insert("auto_route".into(), serde_json::Value::Bool(true));
        }
    } else {
        let download = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Download ONNX auto-router model (~158 MB) from code.aaugs.com for auto-routing?")
            .default(true)
            .interact()?;
        if download {
            download_onnx_model(model_dir)?;
            if let Some(pc) = final_config[provider_key].as_object_mut() {
                pc.insert("auto_route".into(), serde_json::Value::Bool(true));
            }
        }
    }

    let content = serde_json::to_string_pretty(&final_config)?;
    std::fs::write(path, &content).context("failed to write config")?;

    println!();
    println!("Config written to {}", path.display());
    println!("Run `aaugs-code` to start coding.");

    Ok(())
}

fn ask_string(prompt: &str) -> anyhow::Result<String> {
    Input::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .interact_text()
        .map_err(Into::into)
}

fn ask_category(label: &str) -> anyhow::Result<[Option<String>; 4]> {
    println!("  {label}:");
    let low = ask_tier("low");
    let med = ask_tier("med");
    let high = ask_tier("high");
    let max = ask_tier("max");
    Ok([low, med, high, max])
}

fn ask_tier(tier: &str) -> Option<String> {
    let input: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt(format!("  └ {tier}"))
        .allow_empty(true)
        .interact_text()
        .unwrap_or_default();
    if input.is_empty() { None } else { Some(input) }
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
    api_format: &str,
    base_url: &Option<String>,
    coding: &[Option<String>; 4],
    analysis: &[Option<String>; 4],
    creative: &[Option<String>; 4],
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

    // Convert category arrays into structured JSON
    let tier_keys = ["low", "med", "high", "max"];
    let cat = |tiers: &[Option<String>; 4]| -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for (i, key) in tier_keys.iter().enumerate() {
            if let Some(ref val) = tiers[i] {
                map.insert(key.to_string(), json!(val));
            }
        }
        json!(map)
    };

    let model_cats = json!({
        "coding": cat(coding),
        "analysis": cat(analysis),
        "creative": cat(creative),
    });

    // Only include model_categories if at least one tier is filled
    let has_any = coding.iter().chain(analysis.iter()).chain(creative.iter()).any(|o| o.is_some());
    if has_any {
        cfg[provider]["model_categories"] = model_cats;
    }

    if let Some(url) = base_url {
        cfg[provider]["base_url"] = json!(url);
    }

    if !api_format.is_empty() && api_format != "auto" {
        cfg["advanced"]["providers"][provider] = json!({
            "api_format": api_format
        });
    }

    Ok(cfg)
}

fn download_onnx_model(model_dir: &std::path::Path) -> anyhow::Result<()> {
    let files = [
        "model_quantized.onnx",
        "config.json",
        "tokenizer.json",
        "spm.model",
        "special_tokens_map.json",
        "vocab.txt",
        "added_tokens.json",
        "tokenizer_config.json",
    ];
    let base_url = "https://code.aaugs.com/onnx-model";

    std::fs::create_dir_all(model_dir)
        .context("failed to create model directory")?;

    for file in &files {
        let url = format!("{}/{}", base_url, file);
        let path = model_dir.join(file);

        if path.exists() {
            println!("  ✓ {} already downloaded", file);
            continue;
        }

        print!("  ⬇ {} ... ", file);
        std::io::stdout().flush().ok();

        let response = reqwest::blocking::get(&url)
            .with_context(|| format!("failed to download {}", url))?;

        let status = response.status();
        if !status.is_success() {
            anyhow::bail!("download of {} returned HTTP {}", url, status);
        }

        let bytes = response.bytes()
            .with_context(|| format!("failed to read response body for {}", url))?;

        std::fs::write(&path, &bytes)
            .with_context(|| format!("failed to write {}", path.display()))?;

        let size_mb = bytes.len() as f64 / (1024.0 * 1024.0);
        if size_mb > 1.0 {
            println!("{:.1} MB", size_mb);
        } else {
            let size_kb = bytes.len() as f64 / 1024.0;
            println!("{:.0} KB", size_kb);
        }
    }

    println!("✅ ONNX auto-router model downloaded to ./model/");
    Ok(())
}
