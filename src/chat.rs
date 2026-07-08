use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

use crate::config::Config;
use crate::cost;
use crate::llm::{ContentBlock, LLMEvent, Message, Role, Usage};
use crate::llm::LLMProvider;
use crate::sandbox::{PermissionLevel, Sandbox};
use crate::tools::{ToolRegistry, ToolResult};
use crate::tui;

const MAX_TURNS: usize = 50;
const MAX_CONTEXT_TOKENS: usize = 120_000;
const SESSION_DIR: &str = "vibe/sessions";

#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct RequestCost {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cost: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Session {
    pub id: String,
    #[serde(default)]
    pub title: String,
    pub messages: Vec<Message>,
    pub created_at: String,
    pub model: String,
    pub provider_name: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_cost: f64,
    pub request_costs: Vec<RequestCost>,
}

impl Session {
    pub fn new(model: &str, provider_name: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title: String::new(),
            messages: Vec::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
            model: model.to_string(),
            provider_name: provider_name.to_string(),
            input_tokens: 0,
            output_tokens: 0,
            total_cost: 0.0,
            request_costs: Vec::new(),
        }
    }

    fn save_path(&self) -> anyhow::Result<PathBuf> {
        let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not find home directory"))?;
        Ok(home.join(SESSION_DIR).join(format!("{}.json", self.id)))
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = self.save_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, &content)?;
        Ok(())
    }

    pub fn load(id: &str) -> anyhow::Result<Self> {
        let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not find home directory"))?;
        let path = home.join(SESSION_DIR).join(format!("{}.json", id));
        let content = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&content)?)
    }

    pub fn auto_title(&mut self) {
        if !self.title.is_empty() {
            return;
        }
        for msg in &self.messages {
            if msg.role == Role::User {
                for block in &msg.content {
                    if let ContentBlock::Text { text } = block {
                        let t = text.trim();
                        let title = if t.len() > 60 {
                            format!("{}…", &t[..60])
                        } else {
                            t.to_string()
                        };
                        if !title.is_empty() {
                            self.title = title;
                            return;
                        }
                    }
                }
            }
        }
    }

    pub fn list_sessions() -> anyhow::Result<Vec<(String, String, String)>> {
        let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not find home directory"))?;
        let dir = home.join(SESSION_DIR);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(true, |e| e != "json") {
                continue;
            }
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(s) = serde_json::from_str::<Session>(&content) {
                    let display_name = if s.title.is_empty() {
                        s.id.clone()
                    } else {
                        format!("{} ({})", s.title, &s.id[..8])
                    };
                    sessions.push((s.id, display_name, s.model));
                }
            }
        }
        sessions.sort_by(|a, b| a.1.cmp(&b.1));
        Ok(sessions)
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.len() / 4 + text.chars().filter(|&c| c == ' ').count() / 2
}

fn estimate_message_tokens(msg: &Message) -> usize {
    let mut total = 0;
    for block in &msg.content {
        match block {
            ContentBlock::Text { text } => total += estimate_tokens(text),
            ContentBlock::ToolUse { name, input, .. } => {
                total += estimate_tokens(name);
                total += estimate_tokens(&input.to_string());
            }
            ContentBlock::ToolResult { content, .. } => {
                total += estimate_tokens(content);
            }
        }
    }
    total
}

fn truncate_messages(messages: &mut Vec<Message>, max_tokens: usize) {
    let mut total: usize = messages.iter().map(estimate_message_tokens).sum();
    if total <= max_tokens {
        return;
    }

    // Keep system prompt, trim from index 1 onward
    let idx = 1;
    while idx < messages.len() && total > max_tokens {
        let removed = estimate_message_tokens(&messages[idx]);
        messages.remove(idx);
        total = total.saturating_sub(removed);
    }
}

pub struct ChatState {
    pub session: Session,
    pub provider: Box<dyn LLMProvider>,
    pub registry: ToolRegistry,
    pub sandbox: Sandbox,
    pub config: Arc<Config>,
    pub model: String,
}

impl ChatState {
    pub fn new(
        config: Arc<Config>,
        provider: Box<dyn LLMProvider>,
        model: String,
    ) -> Self {
        let provider_name = provider.name().to_string();
        Self {
            session: Session::new(&model, &provider_name),
            provider,
            registry: ToolRegistry::new(),
            sandbox: Sandbox::from_config(&config),
            config,
            model,
        }
    }

    pub fn set_auto_approve(&mut self, val: bool) {
        self.sandbox.set_auto_approve(val);
    }
}

fn build_system_prompt() -> String {
    include_str!("system_prompt.md").to_string()
}

#[async_trait]
trait TurnOutput {
    fn start_turn(&mut self);
    fn on_text(&mut self, text: &str);
    fn on_tool_call(&mut self, name: &str, args: &Value);
    fn finish_streaming(&mut self, assistant_text: &str, has_streamed_text: bool) -> anyhow::Result<()>;
    fn on_tool_exec(&mut self, name: &str, args: &Value);
    fn on_tool_result(&mut self, name: &str, output: &str);
    fn on_tool_denied(&mut self, name: &str);
    fn on_done(&mut self);
    fn on_turn_done(&mut self);
    fn on_info(&mut self, msg: &str);
    fn on_error(&mut self, err: &str);
    fn on_usage(&mut self, _prompt: u32, _completion: u32, _cost: f64) {}
    async fn request_permission(&mut self, sandbox: &Sandbox, name: &str, description: &str) -> bool;
    fn flush_output(&mut self) -> anyhow::Result<()> { Ok(()) }
}

struct StdoutOutput {
    stream_printer: tui::StreamPrinter,
    has_streamed_text: bool,
}

impl StdoutOutput {
    fn new() -> Self {
        Self {
            stream_printer: tui::StreamPrinter::new(),
            has_streamed_text: false,
        }
    }
}

#[async_trait]
impl TurnOutput for StdoutOutput {
    fn start_turn(&mut self) {}
    fn on_text(&mut self, text: &str) {
        self.stream_printer.write(text);
        self.has_streamed_text = true;
    }
    fn on_tool_call(&mut self, _name: &str, _args: &Value) {}
    fn finish_streaming(&mut self, assistant_text: &str, _has_streamed_text: bool) -> anyhow::Result<()> {
        if self.has_streamed_text {
            self.stream_printer.finish(assistant_text);
        } else if !assistant_text.is_empty() {
            tui::render_markdown(assistant_text);
        }
        Ok(())
    }
    fn on_tool_exec(&mut self, name: &str, args: &Value) {
        tui::print_tool_header(name, args);
    }
    fn on_tool_result(&mut self, _name: &str, output: &str) {
        tui::print_tool_result(output);
    }
    fn on_tool_denied(&mut self, name: &str) {
        tui::print_tool_denied(name);
    }
    fn on_done(&mut self) {
        tui::print_success();
    }
    fn on_turn_done(&mut self) {
        println!();
    }
    fn on_info(&mut self, msg: &str) {
        tui::print_info(msg);
    }
    fn on_error(&mut self, _err: &str) {}
    async fn request_permission(&mut self, sandbox: &Sandbox, name: &str, description: &str) -> bool {
        sandbox.request(name, description)
    }
    fn flush_output(&mut self) -> anyhow::Result<()> {
        use std::io::Write;
        std::io::stdout().flush()?;
        Ok(())
    }
}

struct TuiOutput {
    event_tx: std::sync::mpsc::Sender<crate::tui_app::AppEvent>,
    has_streamed_text: bool,
}

impl TuiOutput {
    fn new(event_tx: std::sync::mpsc::Sender<crate::tui_app::AppEvent>) -> Self {
        Self { event_tx, has_streamed_text: false }
    }
}

#[async_trait]
impl TurnOutput for TuiOutput {
    fn start_turn(&mut self) {}
    fn on_text(&mut self, text: &str) {
        let _ = self.event_tx.send(crate::tui_app::AppEvent::Text(text.to_string()));
        self.has_streamed_text = true;
    }
    fn on_tool_call(&mut self, name: &str, args: &Value) {
        let _ = self.event_tx.send(crate::tui_app::AppEvent::ToolCall {
            name: name.to_string(),
            args: args.to_string(),
        });
    }
    fn finish_streaming(&mut self, _assistant_text: &str, _has_streamed_text: bool) -> anyhow::Result<()> {
        Ok(())
    }
    fn on_tool_exec(&mut self, name: &str, args: &Value) {
        let _ = self.event_tx.send(crate::tui_app::AppEvent::ToolCall {
            name: name.to_string(),
            args: args.to_string(),
        });
    }
    fn on_tool_result(&mut self, name: &str, output: &str) {
        let _ = self.event_tx.send(crate::tui_app::AppEvent::ToolResult {
            name: name.to_string(),
            output: output.to_string(),
        });
    }
    fn on_tool_denied(&mut self, name: &str) {
        let _ = self.event_tx.send(crate::tui_app::AppEvent::ToolDenied {
            name: name.to_string(),
        });
    }
    fn on_done(&mut self) {}
    fn on_turn_done(&mut self) {
        let _ = self.event_tx.send(crate::tui_app::AppEvent::TurnDone);
    }
    fn on_info(&mut self, msg: &str) {
        let _ = self.event_tx.send(crate::tui_app::AppEvent::Info(msg.to_string()));
    }
    fn on_error(&mut self, err: &str) {
        let _ = self.event_tx.send(crate::tui_app::AppEvent::Error(err.to_string()));
    }
    fn on_usage(&mut self, prompt_tokens: u32, completion_tokens: u32, cost: f64) {
        let _ = self.event_tx.send(crate::tui_app::AppEvent::Usage {
            prompt_tokens,
            completion_tokens,
            cost,
        });
    }
    async fn request_permission(&mut self, sandbox: &Sandbox, tool_name: &str, description: &str) -> bool {
        match sandbox.check(tool_name) {
            PermissionLevel::Allow => true,
            PermissionLevel::Deny => false,
            PermissionLevel::Ask => {
                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                let _ = self.event_tx.send(crate::tui_app::AppEvent::Question {
                    question: format!("Allow tool '{}': {}?", tool_name, description),
                    options: vec!["y".to_string(), "n".to_string()],
                    tx: resp_tx,
                });
                match resp_rx.await {
                    Ok(answer) => {
                        answer.trim().eq_ignore_ascii_case("y") || answer.trim().eq_ignore_ascii_case("yes")
                    }
                    Err(_) => false,
                }
            }
        }
    }
}

async fn process_turn_inner<O: TurnOutput>(
    state: &mut ChatState,
    output: &mut O,
) -> anyhow::Result<bool> {
    let tools = state.registry.definitions();
    let mut turn_count = 0;

    loop {
        if turn_count >= MAX_TURNS {
            output.on_info("Max turns reached, ending loop");
            return Ok(true);
        }
        turn_count += 1;

        truncate_messages(&mut state.session.messages, MAX_CONTEXT_TOKENS);

        output.start_turn();

        let mut stream = match state.provider.stream_chat(&state.session.messages, &tools).await {
            Ok(s) => s,
            Err(e) => {
                output.on_error(&format!("LLM error: {}", e));
                return Err(anyhow::anyhow!("LLM error: {}", e));
            }
        };

        let mut text_accum = String::new();
        let mut content_blocks = Vec::new();
        let mut has_streamed_text = false;
        let mut last_usage: Option<Usage> = None;

        while let Some(event) = stream.next().await {
            match event? {
                LLMEvent::Text(text) => {
                    output.on_text(&text);
                    text_accum.push_str(&text);
                    has_streamed_text = true;
                }
                LLMEvent::ToolCall { id, name, args } => {
                    if !text_accum.is_empty() {
                        let text = std::mem::take(&mut text_accum);
                        content_blocks.push(ContentBlock::Text { text });
                    }
                    output.on_tool_call(&name, &args);
                    content_blocks.push(ContentBlock::ToolUse { id, name, input: args });
                }
                LLMEvent::Usage(usage) => {
                    last_usage = Some(usage);
                }
                LLMEvent::Stop { finish_reason } => {
                    if !text_accum.is_empty() {
                        let text = std::mem::take(&mut text_accum);
                        content_blocks.push(ContentBlock::Text { text });
                    }
                    tracing::debug!(reason = finish_reason, "stream stopped");
                    break;
                }
            }
        }

        let assistant_text: String = content_blocks.iter()
            .filter_map(|b| {
                if let ContentBlock::Text { text } = b { Some(text.clone()) }
                else { None }
            })
            .collect::<Vec<_>>()
            .join("");

        output.finish_streaming(&assistant_text, has_streamed_text)?;
        output.flush_output()?;

        if content_blocks.is_empty() {
            return Ok(false);
        }

        let assistant_msg = Message::assistant(content_blocks.clone());
        state.session.messages.push(assistant_msg);

        if let Some(usage) = last_usage {
            state.session.input_tokens += usage.prompt_tokens;
            state.session.output_tokens += usage.completion_tokens;
            let cost_val = if usage.cost > 0.0 {
                usage.cost
            } else {
                cost::calculate_cost(&state.session.model, usage.prompt_tokens, usage.completion_tokens)
            };
            state.session.total_cost += cost_val;
            state.session.request_costs.push(RequestCost {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                cost: cost_val,
            });
            output.on_usage(usage.prompt_tokens, usage.completion_tokens, cost_val);
        }

        let tool_calls: Vec<(String, String, Value)> = content_blocks.iter()
            .filter_map(|block| {
                if let ContentBlock::ToolUse { id, name, input } = block {
                    Some((id.clone(), name.clone(), input.clone()))
                } else { None }
            })
            .collect();

        if tool_calls.is_empty() {
            output.on_done();
            if let Err(e) = state.session.save() {
                tracing::warn!("failed to save session: {}", e);
            }
            return Ok(true);
        }

        output.on_turn_done();
        for (id, name, args) in &tool_calls {
            output.on_tool_exec(name, args);

            let description = format!("{}({})", name, args);
            let approved = output.request_permission(&state.sandbox, name, &description).await;

            if !approved {
                output.on_tool_denied(name);
                state.session.messages.push(Message::tool_result(id, format!("tool '{}' was denied", name)));
                continue;
            }

            let result = state.registry.execute(name, args.clone()).await;
            let output_text = format_result(&result);

            output.on_tool_result(name, &output_text);
            state.session.messages.push(Message::tool_result(id, output_text));
        }
    }
}

pub async fn process_turn(state: &mut ChatState) -> anyhow::Result<bool> {
    let spinner = tui::start_spinner();
    let mut output = StdoutOutput::new();
    let result = process_turn_inner(state, &mut output).await;
    spinner.store(false, std::sync::atomic::Ordering::Relaxed);
    tui::clear_line();
    result
}

fn format_result(result: &ToolResult) -> String {
    if result.output.len() > 2000 {
        format!(
            "{} (truncated, {} bytes total)\n{}",
            if result.success { "SUCCESS" } else { "ERROR" },
            result.output.len(),
            &result.output[..2000]
        )
    } else {
        result.output.clone()
    }
}

fn add_system_prompt(state: &mut ChatState) {
    let system_prompt = build_system_prompt();
    state.session.messages.push(Message::system(system_prompt));
}

pub async fn run_once(state: &mut ChatState, prompt: &str) -> anyhow::Result<()> {
    add_system_prompt(state);
    state.session.messages.push(Message::user(prompt));
    state.session.auto_title();
    process_turn(state).await?;
    println!();

    if let Err(e) = state.session.save() {
        tracing::warn!("failed to save session: {}", e);
    }

    Ok(())
}

// --- TUI interactive mode ---

pub async fn process_turn_tui(
    state: &mut ChatState,
    event_tx: &std::sync::mpsc::Sender<crate::tui_app::AppEvent>,
) -> anyhow::Result<bool> {
    let mut output = TuiOutput::new(event_tx.clone());
    process_turn_inner(state, &mut output).await
}

enum SlashResultTui {
    Exit,
    Continue,
}

async fn handle_slash_command_tui(
    state: &mut ChatState,
    input: &str,
    event_tx: &std::sync::mpsc::Sender<crate::tui_app::AppEvent>,
) -> anyhow::Result<SlashResultTui> {
    use crate::tui_app::AppEvent;

    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).copied().unwrap_or("");

    match cmd {
        "/exit" | "/quit" => Ok(SlashResultTui::Exit),
        "/help" => {
            let _ = event_tx.send(AppEvent::Info(
                "Available: /exit, /clear, /model, /tokens, /provider, /browse, /sessions, /resume\nPress Tab/F1 for full help screen.".into()
            ));
            Ok(SlashResultTui::Continue)
        }
        "/clear" => {
            let system = state.session.messages.iter()
                .find(|m| matches!(m.role, Role::System))
                .cloned();
            state.session.messages.clear();
            if let Some(s) = system {
                state.session.messages.push(s);
            }
            let _ = event_tx.send(AppEvent::Info("Conversation cleared".into()));
            Ok(SlashResultTui::Continue)
        }
        "/model" => {
            if arg.is_empty() {
                let _ = event_tx.send(AppEvent::Info(format!("Current model: {}", state.model)));
            } else {
                state.model = arg.to_string();
                state.provider.set_model(&state.model);
                let _ = event_tx.send(AppEvent::ModelChanged(state.model.clone()));
                let _ = event_tx.send(AppEvent::Info(format!("Model changed to: {}", state.model)));
            }
            Ok(SlashResultTui::Continue)
        }
        "/provider" => {
            if arg.is_empty() {
                let _ = event_tx.send(AppEvent::Info(format!("Current provider: {}", state.config.provider)));
            } else {
                let new_name = arg;
                let mut cfg = (*state.config).clone();
                cfg.provider = new_name.to_string();
                if let Err(e) = cfg.validate() {
                    let _ = event_tx.send(AppEvent::Error(format!("Invalid provider: {}", e)));
                    return Ok(SlashResultTui::Continue);
                }
                match crate::llm::resolve_provider(&cfg) {
                    Ok(new_provider) => {
                        state.provider = new_provider;
                        state.config = Arc::new(cfg);
                        state.model = state.provider.default_model().to_string();
                        state.session.provider_name = state.provider.name().to_string();
                        let _ = event_tx.send(AppEvent::ModelChanged(state.model.clone()));
                        let favorite_list = cost::favorite_models(arg);
                        if !favorite_list.is_empty() {
                            let mut msg = format!("Provider changed to: {} (default model: {})", arg, state.model);
                            msg.push_str("\nRecommended models: ");
                            msg.push_str(&favorite_list.join(", "));
                            msg.push_str("\nUse /browse to explore all models.");
                            let _ = event_tx.send(AppEvent::Info(msg));
                        } else {
                            let _ = event_tx.send(AppEvent::Info(
                                format!("Provider changed to: {} (model: {})", arg, state.model)
                            ));
                        }
                    }
                    Err(e) => {
                        let _ = event_tx.send(AppEvent::Error(
                            format!("Failed to switch provider: {}", e)
                        ));
                    }
                }
            }
            Ok(SlashResultTui::Continue)
        }
        "/tokens" => {
            let in_t = state.session.input_tokens;
            let out_t = state.session.output_tokens;
            let cost_val = state.session.total_cost;
            let mut info = format!("Input tokens: {}, Output tokens: {}", in_t, out_t);
            if cost_val > 0.0 {
                info.push_str(&format!("\nTotal cost: ${:.6}", cost_val));
            }
            for (i, rc) in state.session.request_costs.iter().enumerate() {
                info.push_str(&format!("\n  #{}: {} in + {} out = ${:.6}",
                    i + 1, rc.prompt_tokens, rc.completion_tokens, rc.cost));
            }
            let estimated: usize = state.session.messages.iter().map(estimate_message_tokens).sum();
            info.push_str(&format!("\nContext estimate: ~{} tokens", estimated));
            let _ = event_tx.send(AppEvent::Info(info));
            Ok(SlashResultTui::Continue)
        }
        "/browse" => {
            let _ = event_tx.send(AppEvent::OpenBrowse);
            Ok(SlashResultTui::Continue)
        }
        "/sessions" => {
            match Session::list_sessions() {
                Ok(sessions) => {
                    if sessions.is_empty() {
                        let _ = event_tx.send(AppEvent::Info("No saved sessions".into()));
                    } else {
                        let mut info = String::from("Sessions (use /resume <id>):");
                        for (_id, display_name, model) in &sessions {
                            info.push_str(&format!("\n  {} — {}", display_name, model));
                        }
                        let _ = event_tx.send(AppEvent::Info(info));
                    }
                }
                Err(e) => {
                    let _ = event_tx.send(AppEvent::Error(format!("Failed to list sessions: {}", e)));
                }
            }
            Ok(SlashResultTui::Continue)
        }
        "/resume" => {
            if arg.is_empty() {
                let _ = event_tx.send(AppEvent::Info("Usage: /resume <session_id>".into()));
            } else {
                match Session::load(arg) {
                    Ok(loaded) => {
                        // Re-resolve provider from session's provider_name
                        let mut cfg = (*state.config).clone();
                        cfg.provider = loaded.provider_name.clone();
                        match crate::llm::resolve_provider(&cfg) {
                            Ok(new_provider) => {
                                state.provider = new_provider;
                                state.config = Arc::new(cfg);
                            }
                            Err(_) => {
                                // Keep current provider, just set model
                                state.provider.set_model(&loaded.model);
                            }
                        }
                        state.model = loaded.model.clone();
                        state.session = loaded;
                        let _ = event_tx.send(AppEvent::Clear);
                        let _ = event_tx.send(AppEvent::ModelChanged(state.model.clone()));
                        let _ = event_tx.send(AppEvent::Info(format!("Resumed session: {}", state.session.title)));
                    }
                    Err(e) => {
                        let _ = event_tx.send(AppEvent::Error(format!("Failed to load session: {}", e)));
                    }
                }
            }
            Ok(SlashResultTui::Continue)
        }
        _ => {
            let _ = event_tx.send(AppEvent::Info(
                format!("Unknown command: {}. Type /help for commands.", cmd)
            ));
            Ok(SlashResultTui::Continue)
        }
    }
}

pub async fn run_tui_interactive(mut state: ChatState) -> anyhow::Result<()> {
    use crate::tui_app::{self, TuiApp, AppEvent};

    add_system_prompt(&mut state);

    let (event_tx, event_rx) = std::sync::mpsc::channel::<AppEvent>();
    let (input_tx, input_rx) = std::sync::mpsc::channel::<String>();

    let model = state.model.clone();
    let provider_name = state.provider.name().to_string();

    let mut app = TuiApp::new(&model, &provider_name, event_rx);

    // Spawn processing on tokio runtime
    let processing = tokio::spawn(async move {
        while let Ok(input) = input_rx.recv() {

            if input.starts_with('/') {
                match handle_slash_command_tui(&mut state, &input, &event_tx).await? {
                    SlashResultTui::Exit => break,
                    SlashResultTui::Continue => continue,
                }
            }

            state.session.messages.push(Message::user(&input));
            state.session.auto_title();

            if !process_turn_tui(&mut state, &event_tx).await? {
                break;
            }
        }

        if let Err(e) = state.session.save() {
            tracing::warn!("failed to save session: {}", e);
        }

        Ok::<_, anyhow::Error>(())
    });

    // Spawn TUI on blocking thread
    let tui_input_tx = input_tx.clone();
    let tui_handle = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        crossterm::terminal::enable_raw_mode()?;
        let mut terminal = ratatui::Terminal::new(
            ratatui::backend::CrosstermBackend::new(std::io::stdout()),
        )?;
        terminal.clear()?;
        let result = tui_app::run_tui(&mut app, &mut terminal, &tui_input_tx);
        crossterm::terminal::disable_raw_mode()?;
        result
    });

    // Wait for TUI to finish
    let tui_result = tui_handle.await
        .map_err(|e| anyhow::anyhow!("TUI panic: {:?}", e))?;

    // Signal processing to stop
    drop(input_tx);

    // Extract abort handle before awaiting, so we can force-stop if needed
    let abort_handle = processing.abort_handle();
    use tokio::time::{timeout, Duration};
    let proc_result = match timeout(Duration::from_secs(10), processing).await {
        Ok(result) => result.map_err(|e| anyhow::anyhow!("Processing panic: {:?}", e))?,
        Err(_elapsed) => {
            tracing::warn!("processing timed out, aborting");
            abort_handle.abort();
            anyhow::bail!("processing did not finish within 10 seconds of exit signal");
        }
    };

    tui_result?;
    proc_result?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_short() {
        let n = estimate_tokens("hello world");
        assert!(n > 0);
    }

    #[test]
    fn test_estimate_tokens_empty() {
        let n = estimate_tokens("");
        assert_eq!(n, 0);
    }

    #[test]
    fn test_estimate_message_tokens() {
        let msg = Message::user("hello");
        let n = estimate_message_tokens(&msg);
        assert!(n > 0);
    }

    #[test]
    fn test_truncate_messages_under_limit() {
        let mut msgs = vec![Message::system("sys"), Message::user("hi")];
        truncate_messages(&mut msgs, 100_000);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn test_truncate_messages_over_limit() {
        let mut msgs = vec![
            Message::system("sys"),
            Message::user("a".repeat(100_000)),
            Message::user("b".repeat(100_000)),
            Message::user("c".repeat(100_000)),
        ];
        // Set max tokens to roughly 30k chars worth
        truncate_messages(&mut msgs, 30_000);
        // Should have kept system + at most 1 user message
        assert!(msgs.len() < 4, "expected truncation, got {} messages", msgs.len());
        assert_eq!(msgs[0].role, Role::System);
    }

    #[test]
    fn test_auto_title_sets_from_first_user() {
        let mut session = Session::new("gpt-4o", "openai");
        let usr = Message::user("fix the database schema");
        session.messages.push(Message::system("system"));
        session.messages.push(usr);
        session.auto_title();
        assert_eq!(session.title, "fix the database schema");
    }

    #[test]
    fn test_auto_title_does_not_overwrite() {
        let mut session = Session::new("gpt-4o", "openai");
        session.title = "existing title".to_string();
        session.messages.push(Message::user("new message"));
        session.auto_title();
        assert_eq!(session.title, "existing title");
    }

    #[test]
    fn test_auto_title_truncates_long() {
        let mut session = Session::new("gpt-4o", "openai");
        let long = "a".repeat(100);
        session.messages.push(Message::user(long));
        session.auto_title();
        // 60 chars + ellipsis
        assert_eq!(session.title.chars().count(), 61);
        assert!(session.title.ends_with('…'));
    }

    #[test]
    fn test_format_result_short() {
        let r = ToolResult { success: true, output: "ok".to_string() };
        let out = format_result(&r);
        assert_eq!(out, "ok");
    }

    #[test]
    fn test_format_result_long_truncated() {
        let long = "x".repeat(3000);
        let r = ToolResult { success: false, output: long };
        let out = format_result(&r);
        assert!(out.starts_with("ERROR (truncated, 3000 bytes total)"));
        // "ERROR (truncated, 3000 bytes total)\n" + 2000 content chars
        assert_eq!(out.len(), 36 + 2000);
    }

    #[test]
    fn test_format_result_success_prefix() {
        let r = ToolResult { success: true, output: "x".repeat(2500) };
        let out = format_result(&r);
        assert!(out.starts_with("SUCCESS (truncated, 2500 bytes total)"));
    }
}
