use std::io::Write;
use std::path::PathBuf;

use anyhow::Context;
use dialoguer::{Confirm, Input, Select, theme::ColorfulTheme};
use serde_json::json;

use crate::cli::InitArgs;

const PROVIDER_NAMES: &[&str] = &["openrouter", "anthropic", "openai", "gemini", "opencode", "custom"];
const PROVIDER_LABELS: &[&str] = &[
    "OpenRouter (recommended)",
    "Anthropic",
    "OpenAI",
    "Gemini",
    "OpenCode",
    "Custom",
];
const DEFAULT_MODELS: &[&str] = &[
    "anthropic/claude-sonnet-4",
    "claude-sonnet-4-20250514",
    "gpt-4o",
    "gemini-2.5-pro",
    "big-pickle",
    "",
];
const VALID_PROVIDERS: &[&str] = PROVIDER_NAMES;

pub fn run(path: &PathBuf, args: &InitArgs) -> anyhow::Result<()> {
    if args.non_interactive {
        run_headless(path, args)
    } else {
        run_interactive(path)
    }
}

// ── Headless path ─────────────────────────────────────────────────

fn run_headless(path: &PathBuf, args: &InitArgs) -> anyhow::Result<()> {
    let provider_key = args.provider.as_deref().unwrap_or("openrouter");
    if !VALID_PROVIDERS.contains(&provider_key) {
        anyhow::bail!("unknown provider '{provider_key}' — must be one of: {}", VALID_PROVIDERS.join(", "));
    }

    let api_key = args.api_key.as_deref().unwrap_or_default();
    if api_key.is_empty() {
        anyhow::bail!("--api-key is required in non-interactive mode");
    }

    if provider_key != "opencode" {
        let result = if provider_key == "custom" {
            let base = args.base_url.as_deref().unwrap_or("");
            verify_custom_api_key(api_key, base, "openai")
        } else {
            verify_api_key(provider_key, api_key)
        };
        if let Err(e) = result {
            anyhow::bail!("API key verification failed: {e}");
        }
    }

    let is_custom = provider_key == "custom";
    let base_url = if is_custom {
        args.base_url.clone()
    } else {
        None
    };

    let model_idx = PROVIDER_NAMES.iter().position(|&p| p == provider_key).unwrap_or(0);
    let resolved_model = args.model.as_deref().unwrap_or(DEFAULT_MODELS[model_idx]);

    let empty_cats = [None, None, None, None];
    let config = build_config(
        provider_key, api_key, resolved_model, "", &base_url,
        &empty_cats, &empty_cats, &empty_cats, "ask", "ask",
    )?;

    let mut final_config = config;

    let model_dir = std::path::Path::new("model");
    let model_exists = model_dir.join("model_quantized.onnx").exists();
    let wants_auto_route = args.auto_route.unwrap_or(false);

    if wants_auto_route && !model_exists {
        download_onnx_model(model_dir)?;
    }
    if wants_auto_route || model_exists {
        if let Some(pc) = final_config[provider_key].as_object_mut() {
            pc.insert("auto_route".into(), serde_json::Value::Bool(true));
        }
    }

    println!();
    show_summary(&final_config, provider_key, api_key, path)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .context("failed to create config directory")?;
    }
    let content = serde_json::to_string_pretty(&final_config)?;
    std::fs::write(path, &content).context("failed to write config")?;
    println!();
    println!("Config written to {}", path.display());
    println!("Run `aaugs-code` to start coding.");

    Ok(())
}

// ── Interactive path ──────────────────────────────────────────────

fn run_interactive(path: &PathBuf) -> anyhow::Result<()> {
    if path.exists() {
        let overwrite = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt(format!("Config already exists at {}. Overwrite?", path.display()))
            .default(false)
            .interact()?;
        if !overwrite {
            println!("Init cancelled.");
            return Ok(());
        }
    }

    println!("┌──────────────────────────────────────────┐");
    println!("│         aaugs-code config setup            │");
    println!("└──────────────────────────────────────────┘");
    println!();

    // ── Provider ──
    let provider = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Default provider")
        .default(0)
        .items(PROVIDER_LABELS)
        .interact()?;

    let provider_key = PROVIDER_NAMES[provider];
    let is_custom = provider_key == "custom";

    // ── API key (with verification) ──
    let api_key: String;
    let base_url: Option<String>;
    let api_format: String;

    if is_custom {
        loop {
            let key: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("API key")
                .interact_text()?;
            let url = Some(ask_string("Base URL (e.g. https://api.example.com/v1)")?);
            let fmt = {
                let idx = Select::with_theme(&ColorfulTheme::default())
                    .with_prompt("API format")
                    .default(0)
                    .item("OpenAI-compatible")
                    .item("Anthropic")
                    .item("Gemini")
                    .interact()?;
                ["openai", "anthropic", "gemini"][idx].to_string()
            };

            match verify_custom_api_key(&key, url.as_deref().unwrap_or(""), &fmt) {
                Ok(()) => {
                    println!("   ✓ API key verified");
                    api_key = key;
                    base_url = url;
                    api_format = fmt;
                    break;
                }
                Err(e) => {
                    println!("   ! {e}");
                    let retry = Confirm::with_theme(&ColorfulTheme::default())
                        .with_prompt("Try again?")
                        .default(true)
                        .interact()?;
                    if !retry {
                        let proceed = Confirm::with_theme(&ColorfulTheme::default())
                            .with_prompt("Continue with this key anyway?")
                            .default(false)
                            .interact()?;
                        if !proceed {
                            println!("Init cancelled.");
                            return Ok(());
                        }
                        api_key = key;
                        base_url = url;
                        api_format = fmt;
                        break;
                    }
                }
            }
        }
    } else {
        loop {
            let key: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("API key")
                .interact_text()?;

            if key.trim().is_empty() {
                println!("   ! API key cannot be empty");
                continue;
            }

            match verify_api_key(provider_key, &key) {
                Ok(()) => {
                    println!("   ✓ API key verified");
                    api_key = key;
                    break;
                }
                Err(e) => {
                    println!("   ! {e}");
                    let retry = Confirm::with_theme(&ColorfulTheme::default())
                        .with_prompt("Try again?")
                        .default(true)
                        .interact()?;
                    if !retry {
                        let proceed = Confirm::with_theme(&ColorfulTheme::default())
                            .with_prompt("Continue with this key anyway?")
                            .default(false)
                            .interact()?;
                        if !proceed {
                            println!("Init cancelled.");
                            return Ok(());
                        }
                        api_key = key;
                        break;
                    }
                }
            }
        }
        base_url = None;
        api_format = String::new();
    }

    let resolved_model = DEFAULT_MODELS[provider].to_string();

    // ── Model categories ──
    println!();
    println!("   Model categories");
    println!("   ─────────────────");
    println!("   (optional — fill in any tier. Press Enter to skip.)");
    let coding = ask_category("Coding");
    let analysis = ask_category("Analysis");
    let creative = ask_category("Creative");

    // ── Tool permissions ──
    println!();
    println!("   Tool permissions");
    println!("   ─────────────────");
    let bash_perm = permission_select("bash (run shell commands)")?;
    let write_perm = permission_select("write (create/edit files)")?;

    // ── Build config ──
    let config = build_config(
        provider_key, &api_key, &resolved_model, &api_format, &base_url,
        &coding, &analysis, &creative, &bash_perm, &write_perm,
    )?;

    // ── ONNX auto-router download ──
    println!();
    let model_dir = std::path::Path::new("model");
    let model_exists = model_dir.join("model_quantized.onnx").exists();
    let mut final_config = config;
    if model_exists {
        println!("   ONNX auto-router model found at ./model/ — auto-routing enabled.");
        if let Some(pc) = final_config[provider_key].as_object_mut() {
            pc.insert("auto_route".into(), serde_json::Value::Bool(true));
        }
    } else {
        let download = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Download ONNX auto-router model (~158 MB) from\n   code.aaugs.com for auto-routing?")
            .default(true)
            .interact()?;
        if download {
            let _ = &download_onnx_model(model_dir)?;
            if let Some(pc) = final_config[provider_key].as_object_mut() {
                pc.insert("auto_route".into(), serde_json::Value::Bool(true));
            }
        }
    }

    // ── Summary + confirm ──
    println!();
    show_summary(&final_config, provider_key, &api_key, path)?;

    let proceed = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Proceed?")
        .default(true)
        .interact()?;
    if !proceed {
        println!("Init cancelled.");
        return Ok(());
    }

    // ── Write ──
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .context("failed to create config directory")?;
    }
    let content = serde_json::to_string_pretty(&final_config)?;
    std::fs::write(path, &content).context("failed to write config")?;

    println!();
    println!("Config written to {}", path.display());
    println!("Run `aaugs-code` to start coding.");

    Ok(())
}

// ── Summary display ────────────────────────────────────────────────

fn show_summary(config: &serde_json::Value, provider_key: &str, api_key: &str, path: &PathBuf) -> anyhow::Result<()> {
    let api_masked = if api_key.len() > 8 {
        format!("{}••••{}", &api_key[..4], &api_key[api_key.len() - 4..])
    } else if !api_key.is_empty() {
        format!("{}••••", &api_key[..api_key.len().min(4)])
    } else {
        "(empty)".to_string()
    };

    let model = config[provider_key]["model"]
        .as_str()
        .unwrap_or("(none)");

    let auto_route = config[provider_key]["auto_route"]
        .as_bool()
        .unwrap_or(false);

    let perm_bash = config["permissions"]["bash"].as_str().unwrap_or("ask");
    let perm_write = config["permissions"]["write"].as_str().unwrap_or("ask");
    let perm_read = config["permissions"]["read"].as_str().unwrap_or("allow");

    let categories = config[provider_key]["model_categories"].as_object();
    let cat_summary = match categories {
        Some(cats) => {
            let parts: Vec<String> = cats.iter().map(|(name, tiers)| {
                let count = tiers.as_object().map(|o| o.len()).unwrap_or(0);
                format!("{}({})", name, count)
            }).collect();
            if parts.is_empty() {
                "(none)".to_string()
            } else {
                parts.join("  ")
            }
        }
        None => "(none)".to_string(),
    };

    println!("   ┌─────────────────────────────────────────────┐");
    println!("   │        Configuration summary                 │");
    println!("   ├─────────────────────────────────────────────┤");
    println!("   │  Provider     {:34} │", provider_key);
    println!("   │  Model        {:34} │", model);
    println!("   │  API key      {:34} │", api_masked);
    println!("   │  Auto-route   {:34} │", if auto_route { "yes" } else { "no" });
    println!("   │  Permissions  {:34} │", format!("bash:{}  write:{}  read:{}", perm_bash, perm_write, perm_read));
    println!("   │  Categories   {:34} │", cat_summary);
    println!("   ├─────────────────────────────────────────────┤");
    println!("   │  Config path  {:34} │", path.display());
    println!("   └─────────────────────────────────────────────┘");

    Ok(())
}

// ── API key verification ────────────────────────────────────────────

fn verify_api_key(provider_key: &str, api_key: &str) -> Result<(), String> {
    if provider_key == "opencode" {
        return Ok(());
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("failed to create HTTP client: {e}"))?;

    let (url, headers): (String, Vec<(&str, String)>) = match provider_key {
        "openrouter" => (
            "https://openrouter.ai/api/v1/models".into(),
            vec![("Authorization", format!("Bearer {api_key}"))],
        ),
        "openai" => (
            "https://api.openai.com/v1/models".into(),
            vec![("Authorization", format!("Bearer {api_key}"))],
        ),
        "anthropic" => (
            "https://api.anthropic.com/v1/models".into(),
            vec![
                ("x-api-key", api_key.to_string()),
                ("anthropic-version", "2023-06-01".to_string()),
            ],
        ),
        "gemini" => (
            format!("https://generativelanguage.googleapis.com/v1/models?key={api_key}"),
            vec![],
        ),
        _ => return Err("unknown provider".into()),
    };

    let mut req = client.get(&url);
    for (name, value) in &headers {
        req = req.header(*name, value.as_str());
    }

    let resp = req.send().map_err(|e| format!("connection failed: {e}"))?;
    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("HTTP {} — key rejected", resp.status()))
    }
}

fn verify_custom_api_key(api_key: &str, base_url: &str, api_format: &str) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("failed to create HTTP client: {e}"))?;

    match api_format {
        "openai" => {
            let url = format!("{}/models", base_url.trim_end_matches('/'));
            let resp = client.get(&url)
                .header("Authorization", format!("Bearer {api_key}"))
                .send()
                .map_err(|e| format!("connection failed: {e}"))?;
            if resp.status().is_success() { Ok(()) }
            else { Err(format!("HTTP {} — key rejected", resp.status())) }
        }
        "anthropic" => {
            let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
            let resp = client.get(&url)
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
                .send()
                .map_err(|e| format!("connection failed: {e}"))?;
            if resp.status().is_success() { Ok(()) }
            else { Err(format!("HTTP {} — key rejected", resp.status())) }
        }
        "gemini" => {
            let url = format!("{}/v1/models?key={api_key}", base_url.trim_end_matches('/'));
            let resp = client.get(&url)
                .send()
                .map_err(|e| format!("connection failed: {e}"))?;
            if resp.status().is_success() { Ok(()) }
            else { Err(format!("HTTP {} — key rejected", resp.status())) }
        }
        _ => Err(format!("unknown API format: {api_format}")),
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn ask_string(prompt: &str) -> anyhow::Result<String> {
    Input::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .interact_text()
        .map_err(Into::into)
}

fn ask_category(label: &str) -> [Option<String>; 4] {
    println!("   {label}:");
    let low = ask_tier("low");
    let med = ask_tier("med");
    let high = ask_tier("high");
    let max = ask_tier("max");
    [low, med, high, max]
}

fn ask_tier(tier: &str) -> Option<String> {
    let input: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt(format!("     └ {tier}"))
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

// ── Config builder ─────────────────────────────────────────────────

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
            "max_tokens": 0,
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

// ── ONNX download ──────────────────────────────────────────────────

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

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .context("failed to build HTTP client")?;

    for file in &files {
        let url = format!("{}/{}", base_url, file);
        let path = model_dir.join(file);

        if path.exists() {
            println!("  ✓ {} already downloaded", file);
            continue;
        }

        let response = client.get(&url)
            .send()
            .with_context(|| format!("failed to download {}", url))?;

        let status = response.status();
        if !status.is_success() {
            anyhow::bail!("download of {} returned HTTP {}", url, status);
        }

        let total = response.content_length().unwrap_or(0);
        let mut buf = Vec::with_capacity(total as usize);
        use std::io::Read;
        let mut reader = response;
        let mut downloaded: u64 = 0;
        let mut chunk = [0u8; 8192];

        loop {
            let n = reader.read(&mut chunk)?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
            downloaded += n as u64;

            if total > 0 {
                let pct = downloaded as f64 / total as f64 * 100.0;
                print!("\r  ⬇ {} ... {:.0}%", file, pct);
            }
            std::io::stdout().flush().ok();
        }

        if total == 0 {
            print!("\r  ⬇ {} ... ", file);
        }

        std::fs::write(&path, &buf)
            .with_context(|| format!("failed to write {}", path.display()))?;

        let size_mb = buf.len() as f64 / (1024.0 * 1024.0);
        if size_mb > 1.0 {
            println!(" ({:.1} MB)", size_mb);
        } else {
            let size_kb = buf.len() as f64 / 1024.0;
            println!(" ({:.0} KB)", size_kb);
        }
    }

    println!("✅ ONNX auto-router model downloaded to ./model/");
    Ok(())
}
