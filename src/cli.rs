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

    #[arg(short = 'C', long, help = "Resume last session")]
    pub continue_session: bool,

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
}
