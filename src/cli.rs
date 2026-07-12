use clap::Parser;

#[derive(Parser)]
#[command(name = "aaugs-code", about = "AI vibe coding in your terminal")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    #[arg(short, long, help = "Model override (format: provider/model)")]
    pub model: Option<String>,

    #[arg(short, long, help = "Path to config file")]
    pub config: Option<String>,

    #[arg(short = 'y', long, help = "Auto-approve all tool permissions")]
    pub yes: bool,
}

#[derive(Parser)]
pub enum Commands {
    #[command(about = "Run a one-shot prompt")]
    Run {
        prompt: String,
    },
    #[command(about = "Initialize config setup")]
    Init(InitArgs),
    #[command(about = "Manage sessions")]
    Session {
        #[command(subcommand)]
        action: SessionCommands,
    },
}

#[derive(clap::Args, Debug, Clone)]
pub struct InitArgs {
    #[arg(long, help = "Provider name (openrouter, anthropic, openai, gemini, opencode, custom)")]
    pub provider: Option<String>,

    #[arg(long, help = "API key for the provider")]
    pub api_key: Option<String>,

    #[arg(long, help = "Model ID")]
    pub model: Option<String>,

    #[arg(long, help = "Base URL (for custom provider)")]
    pub base_url: Option<String>,

    #[arg(long, help = "Enable auto-routing (requires ONNX model)")]
    pub auto_route: Option<bool>,

    #[arg(short = 'y', long, help = "Skip interactive prompts (requires --provider and --api-key)")]
    pub non_interactive: bool,
}

#[derive(Parser)]
pub enum SessionCommands {
    #[command(about = "List all saved sessions")]
    Ls,
    #[command(about = "Remove a saved session by ID")]
    Rm {
        id: String,
    },
}
