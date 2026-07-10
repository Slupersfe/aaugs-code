mod chat;
mod cli;
mod config;
mod cost;
mod init;
mod router;
mod llm;
mod sandbox;
mod tools;
mod tui;
mod term;
mod update;

use std::sync::Arc;

use clap::Parser;
use cli::{Cli, Commands};
use config::Config;
use tracing_subscriber::EnvFilter;

fn try_init_router(config: &Config) {
    let path = config.provider_config()
        .and_then(|c| c.router_model_path.as_deref())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("model"));

    if path.join("model_quantized.onnx").exists() {
        match router::init(&path) {
            Ok(()) => tracing::info!("ONNX auto-router loaded from {}", path.display()),
            Err(e) => tracing::warn!("failed to load ONNX router (auto-route disabled): {}", e),
        }
    } else {
        tracing::info!("ONNX router model not found at {} — auto-route disabled", path.display());
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();

    let config_path = match &cli.config {
        Some(p) => std::path::PathBuf::from(p),
        None => Config::default_path()?,
    };

    match &cli.command {
        Some(Commands::Init) => {
            init::run(&config_path)?;
            return Ok(());
        }
        Some(Commands::Session { action }) => {
            use cli::SessionCommands;
            match action {
                SessionCommands::Ls => {
                    let sessions = chat::Session::list_sessions()?;
                    if sessions.is_empty() {
                        eprintln!("No saved sessions.");
                    } else {
                        for (id, title, model, created) in &sessions {
                            println!("{:<36}  {:<50}  {:<20}  {}", id, title, model, created);
                        }
                    }
                }
                SessionCommands::Rm { id } => {
                    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not find home directory"))?;
                    let path = home.join("vibe/sessions").join(format!("{}.json", id));
                    if path.exists() {
                        std::fs::remove_file(&path)?;
                        eprintln!("Removed session: {}", id);
                    } else {
                        anyhow::bail!("Session not found: {}", id);
                    }
                }
            }
            return Ok(());
        }
        Some(Commands::Run { prompt }) => {
            let config = Config::load(&config_path)?;
            try_init_router(&config);
            let config = Arc::new(config);
            let provider = llm::resolve_provider(&config)
                .map_err(|e| anyhow::anyhow!("provider error: {}", e))?;
            let model = cli.model.as_deref().unwrap_or_else(|| provider.default_model()).to_string();

            tracing::info!(
                "config={} provider={} model={} auto_route={}",
                config_path.display(), config.provider, model,
                config.provider_config().and_then(|c| c.auto_route).unwrap_or(true) && router::is_loaded(),
            );

            let mut state = chat::ChatState::new(config, provider, model);
            if cli.yes {
                state.set_auto_approve(true);
            }

            if let Ok(Some(latest)) = update::check_update().await {
                let cur = update::current_release().unwrap_or(0);
                eprintln!("Update available: v{} (current: v{}). Run /update to upgrade.", latest, cur);
            }

            chat::run_once(&mut state, prompt).await?;
        }
        None => {
            let config = Config::load(&config_path)?;
            try_init_router(&config);
            let config = Arc::new(config);
            let provider = llm::resolve_provider(&config)
                .map_err(|e| anyhow::anyhow!("provider error: {}", e))?;
            let model = cli.model.as_deref().unwrap_or_else(|| provider.default_model()).to_string();

            tracing::info!(
                "config={} provider={} model={} auto_route={}",
                config_path.display(), config.provider, model,
                config.provider_config().and_then(|c| c.auto_route).unwrap_or(true) && router::is_loaded(),
            );

            let mut state = chat::ChatState::new(config, provider, model);
            if cli.yes {
                state.set_auto_approve(true);
            }

            let latest_release = match update::check_update().await {
                Ok(Some(v)) => {
                    tracing::info!("update available: v{}", v);
                    Some(v)
                }
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!("update check failed: {}", e);
                    None
                }
            };

            chat::run_tui_interactive(state, latest_release).await?;
        }
    }

    Ok(())
}
