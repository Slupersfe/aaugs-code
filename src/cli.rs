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
    #[command(about = "Initialize interactive config setup")]
    Init,
    #[command(about = "Manage sessions")]
    Session {
        #[command(subcommand)]
        action: SessionCommands,
    },
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
