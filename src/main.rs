mod chat;
mod cli;
mod config;
mod cost;
mod init;
mod llm;
mod sandbox;
mod tools;
mod tui;
mod tui_app;

use std::sync::Arc;

use clap::Parser;
use cli::{Cli, Commands};
use config::Config;
use tracing_subscriber::EnvFilter;

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
        Some(Commands::Run { prompt }) => {
            let config = Config::load(&config_path)?;
            let config = Arc::new(config);
            let provider = llm::resolve_provider(&config)
                .map_err(|e| anyhow::anyhow!("provider error: {}", e))?;
            let model = cli.model.as_deref().unwrap_or_else(|| provider.default_model()).to_string();

            tracing::info!(
                "config={} provider={} model={}",
                config_path.display(), config.provider, model
            );

            let mut state = chat::ChatState::new(config, provider, model);
            if cli.yes {
                state.set_auto_approve(true);
            }
            chat::run_once(&mut state, prompt).await?;
        }
        None => {
            let config = Config::load(&config_path)?;
            let config = Arc::new(config);
            let provider = llm::resolve_provider(&config)
                .map_err(|e| anyhow::anyhow!("provider error: {}", e))?;
            let model = cli.model.as_deref().unwrap_or_else(|| provider.default_model()).to_string();

            tracing::info!(
                "config={} provider={} model={}",
                config_path.display(), config.provider, model
            );

            let mut state = chat::ChatState::new(config, provider, model);
            if cli.yes {
                state.set_auto_approve(true);
            }
            chat::run_tui_interactive(state).await?;
        }
    }

    Ok(())
}
