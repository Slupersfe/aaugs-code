use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use futures::stream::FuturesUnordered;
use serde_json::Value;

use crate::config::Config;
use crate::cost;
use crate::llm::{ContentBlock, LLMError, LLMEvent, Message, Role, ToolDef, Usage};
use crate::llm::LLMProvider;
use crate::sandbox::{PermissionLevel, Sandbox};
use crate::tools::{ToolRegistry, ToolResult};
use crate::term;

const MAX_TURNS: usize = 50;
const SUMMARY_PREFIX: &str = "[Summary: ";
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
        // Restrict permissions: owner read/write only
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Err(e) = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)) {
                tracing::warn!("failed to set permissions on session file: {}", e);
            }
        }
        Ok(())
    }

    pub fn load(id: &str) -> anyhow::Result<Self> {
        let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not find home directory"))?;
        // Sanitize: strip directory components to prevent path traversal
        let safe_id = std::path::Path::new(id)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("");
        let path = home.join(SESSION_DIR).join(format!("{}.json", safe_id));
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
                    let title = if t.chars().count() > 60 {
                        format!("{}…", t.chars().take(60).collect::<String>())
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

    pub fn list_sessions() -> anyhow::Result<Vec<(String, String, String, String)>> {
        let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not find home directory"))?;
        let dir = home.join(SESSION_DIR);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        let now = chrono::Utc::now();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(s) = serde_json::from_str::<Session>(&content) {
                    // Auto-cleanup sessions older than 30 days
                    if let Ok(created) = chrono::DateTime::parse_from_rfc3339(&s.created_at) {
                        let created_utc = created.with_timezone(&chrono::Utc);
                        let age = now - created_utc;
                        if age.num_days() > 30 {
                            let _ = std::fs::remove_file(&path);
                            continue;
                        }
                    }
                    let display_name = if s.title.is_empty() {
                        s.id.clone()
                    } else {
                        s.title.clone()
                    };
                    // Truncate created_at to just the date portion
                    let created = if s.created_at.len() > 10 {
                        s.created_at[..10].to_string()
                    } else {
                        s.created_at.clone()
                    };
                    sessions.push((s.id, display_name, s.model, created));
                }
            }
        }
        sessions.sort_by(|a, b| a.1.cmp(&b.1));
        Ok(sessions)
    }
}

/// Strips `<think>...</think>` blocks from streaming text chunks.
/// Tracks whether we're inside an unclosed block via `in_block`.
/// Handles the common case where `<think>` and `</think>` arrive in separate chunks.
fn strip_think(chunk: &str, in_block: &mut bool) -> String {
    let mut out = String::new();
    let mut rest = chunk;

    loop {
        if *in_block {
            match rest.find("</think>") {
                Some(end) => {
                    *in_block = false;
                    rest = &rest[end + 8..];
                    continue;
                }
                None => break,
            }
        }

        match rest.find("<think>") {
            Some(start) => {
                out.push_str(&rest[..start]);
                let after = &rest[start + 7..];
                match after.find("</think>") {
                    Some(end) => {
                        rest = &after[end + 8..];
                        continue;
                    }
                    None => {
                        *in_block = true;
                        break;
                    }
                }
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
    }

    out
}

/// Strips any complete `<think>...</think>` blocks from fully assembled text.
/// Used as a final clean-up pass after all streaming chunks are combined.
fn strip_think_final(text: String) -> String {
    let mut result = text;
    loop {
        let before = result.len();
        if let Some(start) = result.find("<think>") {
            if let Some(end) = result[start..].find("</think>") {
                let close = start + end + 8;
                result.replace_range(start..close, "");
            }
        }
        if result.len() == before {
            break;
        }
    }
    result
}

/// Collects assistant text content from a message for summarization.
fn collect_text(message: &Message) -> String {
    let mut out = String::new();
    for block in &message.content {
        if let ContentBlock::Text { text } = block {
            if !text.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(text);
            }
        }
    }
    out
}

/// Summarizes old assistant messages into one sentence when context exceeds the limit.
/// Only summarizes assistant messages (not user). Replaces them with a system summary.
async fn maybe_summarize(
    state: &mut ChatState,
) -> anyhow::Result<()> {
    let total: usize = state.session.messages.iter().map(estimate_message_tokens).sum();
    if total <= state.max_context_tokens {
        return Ok(());
    }

    // Find the range of assistant/tool messages to summarize.
    // Keep the last few user messages and everything after them.
    let cutoff = find_summarize_cutoff(&state.session.messages);
    if cutoff == 0 {
        return Ok(()); // Nothing to summarize
    }

    let to_summarize: Vec<Message> = state.session.messages.drain(..=cutoff).collect();

    // Extract text from assistant messages in the range
    let mut conversation_text = String::new();
    for msg in &to_summarize {
        if msg.role == Role::Assistant {
            let text = collect_text(msg);
            if !text.is_empty() {
                conversation_text.push_str(&text);
                conversation_text.push('\n');
            }
        }
    }

    if conversation_text.is_empty() {
        return Ok(());
    }

    // Build summarization prompt
    let summary_prompt = format!(
        "Summarize the following assistant actions in one sentence:\n\n{}",
        conversation_text.trim()
    );

    let summary_messages = vec![
        Message::system("You summarize assistant actions into one concise sentence. Output only the summary, no preamble."),
        Message::user(&summary_prompt),
    ];

    // Call LLM with no tools, collect the response
    let mut summary = String::new();
    let tools: Vec<ToolDef> = Vec::new();
    if let Ok(mut stream) = state.provider.stream_chat(&summary_messages, &tools).await {
        while let Some(event) = stream.next().await {
            if let Ok(LLMEvent::Text(text)) = event {
                summary.push_str(&text);
            }
        }
    }

    if summary.is_empty() {
        summary = "Assistant actions were summarized.".to_string();
    }

    let summary_msg = Message::system(format!("{}{}]", SUMMARY_PREFIX, summary.trim()));
    state.session.messages.insert(0, summary_msg);
    state.summarized_count += 1;

    Ok(())
}

/// Finds the index of the last message before the cutoff for summarization.
/// Returns the index of the last assistant/tool message to summarize,
/// keeping at least the most recent user message and everything after it.
fn find_summarize_cutoff(messages: &[Message]) -> usize {
    // Find the second-to-last user message from the end
    // We want to keep the last user message and everything after
    let mut user_count = 0;
    for (i, msg) in messages.iter().enumerate().rev() {
        if msg.role == Role::User {
            user_count += 1;
            if user_count >= 2 {
                return i.saturating_sub(1);
            }
        }
    }

    // If there aren't multiple user messages, try to summarize everything
    // except the last few assistant turns
    let mut count = 0;
    for (i, msg) in messages.iter().enumerate().rev() {
        if msg.role == Role::Assistant {
            count += 1;
            if count >= 3 {
                return i.saturating_sub(1);
            }
        }
    }

    0
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
    pub auto_route: bool,
    pub max_context_tokens: usize,
    pub max_tool_output_storage: usize,
    pub max_tool_result_truncation: usize,
    pub summarized_count: usize,
}

impl ChatState {
    pub fn new(
        config: Arc<Config>,
        provider: Box<dyn LLMProvider>,
        model: String,
    ) -> Self {
        let provider_name = provider.name().to_string();
        let max_context_tokens = config.advanced.max_context_tokens;
        let max_tool_output_storage = config.advanced.max_tool_output_storage;
        let max_tool_result_truncation = config.advanced.tool_result_truncation_bytes;
        let auto_route = config.provider_config()
            .and_then(|c| c.auto_route)
            .unwrap_or(true)
            && crate::router::is_loaded();
        Self {
            session: Session::new(&model, &provider_name),
            provider,
            registry: ToolRegistry::new(),
            sandbox: Sandbox::from_config(&config),
            config,
            model,
            auto_route,
            max_context_tokens,
            max_tool_output_storage,
            max_tool_result_truncation,
            summarized_count: 0,
        }
    }

    pub fn set_auto_approve(&mut self, val: bool) {
        self.sandbox.set_auto_approve(val);
    }

    pub fn auto_route_for_prompt(&mut self, prompt: &str) {
        let classification = match crate::router::classify(prompt) {
            Some(c) => c,
            None => return,
        };
        let pc = match self.config.provider_config() {
            Some(pc) => pc,
            None => return,
        };
        let cats = match pc.model_categories.as_ref() {
            Some(c) => c,
            None => return,
        };
        let model_name = match classification.target.as_str() {
            "Coding_API" => match classification.intensity.as_str() {
                "low" => cats.coding.low.clone(),
                "med" => cats.coding.med.clone(),
                "high" => cats.coding.high.clone(),
                "max" => cats.coding.max.clone(),
                _ => None,
            },
            "Analysis_API" => match classification.intensity.as_str() {
                "low" => cats.analysis.low.clone(),
                "med" => cats.analysis.med.clone(),
                "high" => cats.analysis.high.clone(),
                "max" => cats.analysis.max.clone(),
                _ => None,
            },
            "Creative_API" => match classification.intensity.as_str() {
                "low" => cats.creative.low.clone(),
                "med" => cats.creative.med.clone(),
                "high" => cats.creative.high.clone(),
                "max" => cats.creative.max.clone(),
                _ => None,
            },
            _ => None,
        };
        if let Some(chosen) = model_name {
            if chosen != self.model {
                self.model = chosen;
                self.provider.set_model(&self.model);
                tracing::info!(
                    target = classification.target, intensity = classification.intensity,
                    confidence = classification.target_confidence, model = self.model,
                    latency_ms = classification.latency_ms,
                    "auto-routed",
                );
            }
        }
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
    fn is_cancelled(&self) -> bool { false }
}

struct StdoutOutput {
    stream_printer: term::StreamPrinter,
    has_streamed_text: bool,
}

impl StdoutOutput {
    fn new() -> Self {
        Self {
            stream_printer: term::StreamPrinter::new(),
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
            term::render_markdown(assistant_text);
        }
        Ok(())
    }
    fn on_tool_exec(&mut self, name: &str, args: &Value) {
        term::print_tool_header(name, args);
    }
    fn on_tool_result(&mut self, _name: &str, output: &str) {
        term::print_tool_result(output);
    }
    fn on_tool_denied(&mut self, name: &str) {
        term::print_tool_denied(name);
    }
    fn on_done(&mut self) {
        term::print_success();
    }
    fn on_turn_done(&mut self) {
        println!();
    }
    fn on_info(&mut self, msg: &str) {
        term::print_info(msg);
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
    event_tx: std::sync::mpsc::Sender<crate::tui::AppEvent>,
    cancelled: Arc<AtomicBool>,
    has_streamed_text: bool,
}

impl TuiOutput {
    fn new(event_tx: std::sync::mpsc::Sender<crate::tui::AppEvent>, cancelled: Arc<AtomicBool>) -> Self {
        Self { event_tx, cancelled, has_streamed_text: false }
    }
}

#[async_trait]
impl TurnOutput for TuiOutput {
    fn start_turn(&mut self) {}
    fn on_text(&mut self, text: &str) {
        let _ = self.event_tx.send(crate::tui::AppEvent::Text(text.to_string()));
        self.has_streamed_text = true;
    }
    fn on_tool_call(&mut self, name: &str, args: &Value) {
        let _ = self.event_tx.send(crate::tui::AppEvent::ToolCall {
            name: name.to_string(),
            args: args.to_string(),
        });
    }
    fn finish_streaming(&mut self, _assistant_text: &str, _has_streamed_text: bool) -> anyhow::Result<()> {
        Ok(())
    }
    fn on_tool_exec(&mut self, name: &str, args: &Value) {
        let _ = self.event_tx.send(crate::tui::AppEvent::ToolCall {
            name: name.to_string(),
            args: args.to_string(),
        });
    }
    fn on_tool_result(&mut self, name: &str, output: &str) {
        let _ = self.event_tx.send(crate::tui::AppEvent::ToolResult {
            name: name.to_string(),
            output: output.to_string(),
        });
    }
    fn on_tool_denied(&mut self, name: &str) {
        let _ = self.event_tx.send(crate::tui::AppEvent::ToolDenied {
            name: name.to_string(),
        });
    }
    fn on_done(&mut self) {
        let _ = self.event_tx.send(crate::tui::AppEvent::TurnDone);
    }
    fn on_turn_done(&mut self) {
        let _ = self.event_tx.send(crate::tui::AppEvent::TurnDone);
    }
    fn on_info(&mut self, msg: &str) {
        let _ = self.event_tx.send(crate::tui::AppEvent::Info(msg.to_string()));
    }
    fn on_error(&mut self, err: &str) {
        let _ = self.event_tx.send(crate::tui::AppEvent::Error(err.to_string()));
    }
    fn on_usage(&mut self, prompt_tokens: u32, completion_tokens: u32, cost: f64) {
        let _ = self.event_tx.send(crate::tui::AppEvent::Usage {
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
                let _ = self.event_tx.send(crate::tui::AppEvent::Question {
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
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

fn truncate_for_storage(output: &str, max_chars: usize) -> String {
    if output.len() > max_chars {
        let prefix = if output.len() > 7 && (output.starts_with("SUCCESS") || output.starts_with("ERROR")) {
            let space = output[..8].find(' ').unwrap_or(7);
            output[..space].to_string()
        } else {
            "OUTPUT".to_string()
        };
        format!(
            "{} (truncated, {} bytes total)\n{}",
            prefix,
            output.len(),
            &output[..max_chars]
        )
    } else {
        output.to_string()
    }
}

fn strip_tool_call_inputs(messages: &mut Vec<Message>) {
    let last_assistant = messages.iter().rposition(|m| m.role == Role::Assistant);
    for (i, msg) in messages.iter_mut().enumerate() {
        if msg.role != Role::Assistant {
            continue;
        }
        if Some(i) == last_assistant {
            continue;
        }
        for block in &mut msg.content {
            if let ContentBlock::ToolUse { input, .. } = block {
                *input = serde_json::json!({});
            }
        }
    }
}

fn coalesce_text_blocks(messages: &mut Vec<Message>) {
    for msg in messages.iter_mut() {
        let mut i = 0;
        while i + 1 < msg.content.len() {
            let should_merge = matches!(&msg.content[i], ContentBlock::Text { .. })
                && matches!(&msg.content[i + 1], ContentBlock::Text { .. });
            if !should_merge {
                i += 1;
                continue;
            }
            let second = match msg.content.remove(i + 1) {
                ContentBlock::Text { text } => text,
                _ => unreachable!(),
            };
            if let ContentBlock::Text { text } = &mut msg.content[i] {
                text.push_str(&second);
            }
        }
    }
}

fn drop_stale_tool_results(messages: &mut Vec<Message>, keep_assistant_turns: usize) {
    let mut assist_count = 0;
    let cutoff = messages.iter().rposition(|m| {
        if m.role == Role::Assistant {
            assist_count += 1;
        }
        assist_count >= keep_assistant_turns
    });

    let Some(mut cutoff) = cutoff else {
        return;
    };

    let mut i = 0;
    while i < cutoff {
        if messages[i].role == Role::Tool {
            messages.remove(i);
            cutoff = cutoff.saturating_sub(1);
        } else {
            i += 1;
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

        // Check cost budget before each turn
        let budget = state.config.advanced.max_cost_per_session;
        if budget > 0.0 && state.session.total_cost >= budget {
            output.on_info(&format!(
                "Cost budget reached (${:.4} / ${:.4}). Ending session.",
                state.session.total_cost, budget
            ));
            return Ok(true);
        }

        drop_stale_tool_results(&mut state.session.messages, 10);
        strip_tool_call_inputs(&mut state.session.messages);
        coalesce_text_blocks(&mut state.session.messages);
        truncate_messages(&mut state.session.messages, state.max_context_tokens * 3);

        if let Err(e) = maybe_summarize(state).await {
            tracing::warn!("summarization failed: {}", e);
        }

        output.start_turn();

        // Try primary model with retry, then fallbacks
        let mut stream = match try_stream_with_fallback(state, &tools).await {
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
        let mut in_think_block = false;

        while let Some(event) = stream.next().await {
            if output.is_cancelled() {
                let _ = std::mem::take(&mut text_accum);
                let _ = std::mem::take(&mut content_blocks);
                has_streamed_text = false;
                break;
            }
            match event? {
                LLMEvent::Text(text) => {
                    let cleaned = strip_think(&text, &mut in_think_block);
                    if !cleaned.is_empty() {
                        output.on_text(&cleaned);
                        text_accum.push_str(&cleaned);
                        has_streamed_text = true;
                    }
                }
                LLMEvent::ToolCall { id, name, args } => {
                    if !text_accum.is_empty() {
                        let text = strip_think_final(std::mem::take(&mut text_accum));
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
                        let text = strip_think_final(std::mem::take(&mut text_accum));
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

        let max_output_from_args = |args: &Value| -> Option<usize> {
            args.get("max_output").and_then(|v| v.as_u64()).map(|v| v as usize)
        };

        // Parallel tool execution with FuturesUnordered
        if tool_calls.len() == 1 {
            let (id, name, args) = tool_calls.into_iter().next().unwrap();
            let description = format!("{}({})", name, args);
            let approved = output.request_permission(&state.sandbox, &name, &description).await;
            if !approved {
                output.on_tool_denied(&name);
                state.session.messages.push(Message::tool_result(id, format!("tool '{}' was denied", name)));
            } else {
                let custom_max = max_output_from_args(&args);
                let result = state.registry.execute(&name, args).await;
                let max_bytes = custom_max.unwrap_or(state.max_tool_result_truncation);
                let output_text = format_result(&result, max_bytes);
                let stored_text = truncate_for_storage(&result.output, state.max_tool_output_storage);
                output.on_tool_result(&name, &output_text);
                state.session.messages.push(Message::tool_result(id, stored_text));
            }
        } else {
            // Permission checks are sequential (user interaction)
            let mut approved: Vec<(String, String, Value, Option<usize>)> = Vec::new();
            for (id, name, args) in &tool_calls {
                output.on_tool_exec(name, args);
                let description = format!("{}({})", name, args);
                let granted = output.request_permission(&state.sandbox, name, &description).await;
                if granted {
                    let custom_max = max_output_from_args(args);
                    approved.push((id.clone(), name.clone(), args.clone(), custom_max));
                } else {
                    output.on_tool_denied(name);
                    state.session.messages.push(Message::tool_result(id.clone(), format!("tool '{}' was denied", name)));
                }
            }
            // Execute approved tools in parallel
            if !approved.is_empty() {
                let registry = &state.registry;
                let mut tasks = FuturesUnordered::new();
                for (id, name, args, custom_max) in approved {
                    tasks.push(async move {
                        let result = registry.execute(&name, args).await;
                        (id, name, result, custom_max)
                    });
                }
                while let Some((id, name, result, custom_max)) = tasks.next().await {
                    let max_bytes = custom_max.unwrap_or(state.max_tool_result_truncation);
                    let output_text = format_result(&result, max_bytes);
                    let stored_text = truncate_for_storage(&result.output, state.max_tool_output_storage);
                    output.on_tool_result(&name, &output_text);
                    state.session.messages.push(Message::tool_result(id, stored_text));
                }
            }
        }
    }
}

/// Tries the primary provider with retry, then attempts fallback providers on failure.
/// When a fallback succeeds, `state.provider`, `state.model`, and `state.session.provider_name` are updated.
async fn try_stream_with_fallback(
    state: &mut ChatState,
    tools: &[ToolDef],
) -> anyhow::Result<Pin<Box<dyn Stream<Item = Result<LLMEvent, LLMError>> + Send>>> {
    if let Err(e) = crate::llm::validate_request(&state.session.messages, tools) {
        return Err(anyhow::anyhow!("{}", e));
    }

    let primary_result = crate::llm::retry_with_backoff(
        || state.provider.stream_chat(&state.session.messages, tools),
        2,
    ).await;

    match primary_result {
        Ok(stream) => return Ok(stream),
        Err(primary_err) => {
            let fallbacks = state.config.provider_config()
                .map(|c| c.fallback.clone())
                .unwrap_or_default();

            if fallbacks.is_empty() {
                return Err(anyhow::anyhow!("{}", primary_err));
            }

            let mut last_err = primary_err;
            for entry in &fallbacks {
                let (provider_name, model) = match entry.split_once(':') {
                    Some((p, m)) => (p, m),
                    None => continue,
                };
                let mut cfg = (*state.config).clone();
                cfg.provider = provider_name.to_string();
                let mut new_provider = match crate::llm::resolve_provider(&cfg) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!("fallback provider '{}' failed to resolve: {}", provider_name, e);
                        continue;
                    }
                };
                new_provider.set_model(model);
                match crate::llm::retry_with_backoff(
                    || new_provider.stream_chat(&state.session.messages, tools),
                    1,
                ).await {
                    Ok(stream) => {
                        tracing::info!("fell back to {}:{}", provider_name, model);
                        state.provider = new_provider;
                        state.model = model.to_string();
                        state.session.provider_name = provider_name.to_string();
                        return Ok(stream);
                    }
                    Err(e) => {
                        tracing::warn!("fallback {}:{} failed: {}", provider_name, model, e);
                        last_err = e;
                    }
                }
            }
            Err(anyhow::anyhow!("{}", last_err))
        }
    }
}

pub async fn process_turn(state: &mut ChatState) -> anyhow::Result<bool> {
    let spinner = term::start_spinner();
    let mut output = StdoutOutput::new();
    let result = process_turn_inner(state, &mut output).await;
    spinner.store(false, std::sync::atomic::Ordering::Relaxed);
    term::clear_line();
    result
}

fn format_result(result: &ToolResult, max_bytes: usize) -> String {
    if result.output.len() > max_bytes {
        format!(
            "{} (truncated, {} bytes total)\n{}",
            if result.success { "SUCCESS" } else { "ERROR" },
            result.output.len(),
            &result.output[..max_bytes]
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
    if state.auto_route {
        state.auto_route_for_prompt(prompt);
    }
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
    event_tx: &std::sync::mpsc::Sender<crate::tui::AppEvent>,
    cancelled: Arc<AtomicBool>,
) -> anyhow::Result<bool> {
    let mut output = TuiOutput::new(event_tx.clone(), cancelled);
    process_turn_inner(state, &mut output).await
}

enum SlashResultTui {
    Exit,
    Continue,
}

fn messages_to_display_msgs(messages: &[Message]) -> Vec<crate::tui::DisplayMsg> {
    messages.iter().filter_map(|msg| {
        let role = match msg.role {
            Role::System => "System",
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::Tool => "Tool",
        };
        let text = msg.content.iter().map(|block| match block {
            ContentBlock::Text { text } => text.clone(),
            ContentBlock::ToolUse { name, input, .. } => {
                format!("── {} ──\n{}", name, serde_json::to_string_pretty(input).unwrap_or_default())
            }
            ContentBlock::ToolResult { content, .. } => content.clone(),
        }).collect::<Vec<_>>().join("\n");
        if text.trim().is_empty() {
            None
        } else {
            Some(crate::tui::DisplayMsg { role: role.to_string(), text })
        }
    }).collect()
}

async fn handle_slash_command_tui(
    state: &mut ChatState,
    input: &str,
    event_tx: &std::sync::mpsc::Sender<crate::tui::AppEvent>,
) -> anyhow::Result<SlashResultTui> {
    use crate::tui::AppEvent;

    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).copied().unwrap_or("");

    match cmd {
        "/exit" | "/quit" => Ok(SlashResultTui::Exit),
        "/help" => {
            let _ = event_tx.send(AppEvent::Info(
                "Available: /exit, /clear, /model, /tokens, /summarize, /provider, /browse, /sessions, /search, /resume, /update\nPress Tab/F1 for full help screen.".into()
            ));
            Ok(SlashResultTui::Continue)
        }
        "/clear" => {
            let _ = event_tx.send(AppEvent::Clear);
            Ok(SlashResultTui::Continue)
        }
        "/model" => {
            if arg.is_empty() {
                let _ = event_tx.send(AppEvent::OpenBrowse);
            } else if arg == "auto" {
                state.auto_route = true;
                let _ = event_tx.send(AppEvent::AutoRoute(true));
                let _ = event_tx.send(AppEvent::Info("Auto-routing enabled (ONNX router)".into()));
            } else if let Some((cat, tier)) = arg.split_once(':') {
                let resolved = state.config.provider_config().and_then(|c| c.model_categories.as_ref()).and_then(|cats| {
                    let cm = match cat.to_lowercase().as_str() {
                        "coding" => Some(&cats.coding),
                        "analysis" => Some(&cats.analysis),
                        "creative" => Some(&cats.creative),
                        _ => None,
                    }?;
                    match tier.to_lowercase().as_str() {
                        "low" => cm.low.clone(),
                        "med" => cm.med.clone(),
                        "high" => cm.high.clone(),
                        "max" => cm.max.clone(),
                        _ => None,
                    }
                });
                match resolved {
                    Some(m) => {
                        state.auto_route = false;
                        state.model = m.clone();
                        state.provider.set_model(&m);
                        let _ = event_tx.send(AppEvent::ModelChanged(state.model.clone()));
                        let _ = event_tx.send(AppEvent::AutoRoute(false));
                        let _ = event_tx.send(AppEvent::Info(format!("Model changed to: {}", m)));
                    }
                    None => {
                        let _ = event_tx.send(AppEvent::Error(format!("Unknown category:tier '{}'", arg)));
                    }
                }
            } else {
                state.auto_route = false;
                state.model = arg.to_string();
                state.provider.set_model(&state.model);
                let _ = event_tx.send(AppEvent::ModelChanged(state.model.clone()));
                let _ = event_tx.send(AppEvent::AutoRoute(false));
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
            let estimated: usize = state.session.messages.iter().map(estimate_message_tokens).sum();
            let limit = state.max_context_tokens;
            let mut info = format!(
                "Input tokens: {}, Output tokens: {}\nContext: ~{} / ~{} tokens  (summarized {} rounds)",
                in_t, out_t, estimated, limit, state.summarized_count,
            );
            if cost_val > 0.0 {
                info.push_str(&format!("\nTotal cost: ${:.6}", cost_val));
            }
            for (i, rc) in state.session.request_costs.iter().enumerate() {
                info.push_str(&format!("\n  #{}: {} in + {} out = ${:.6}",
                    i + 1, rc.prompt_tokens, rc.completion_tokens, rc.cost));
            }
            let _ = event_tx.send(AppEvent::Info(info));
            Ok(SlashResultTui::Continue)
        }
        "/summarize" => {
            let before = state.session.messages.len();
            match maybe_summarize(state).await {
                Ok(()) => {
                    let removed = before - state.session.messages.len();
                    let msg = if removed > 0 {
                        format!("Summarized {} old messages ({} rounds total)", removed, state.summarized_count)
                    } else {
                        "Nothing to summarize, context is under the limit.".to_string()
                    };
                    let _ = event_tx.send(AppEvent::Info(msg));
                }
                Err(e) => {
                    let _ = event_tx.send(AppEvent::Error(format!("Summarize failed: {}", e)));
                }
            }
            Ok(SlashResultTui::Continue)
        }
        "/update" => {
            let _ = event_tx.send(AppEvent::Info("Pulling changes and building...".into()));
            let event_tx_clone = event_tx.clone();
            tokio::spawn(async move {
                match crate::update::perform_update() {
                    Ok(()) => {
                        let _ = event_tx_clone.send(AppEvent::Info(
                            "Update complete! Restart to use the new version.".into()
                        ));
                    }
                    Err(e) => {
                        let _ = event_tx_clone.send(AppEvent::Error(
                            format!("Update failed: {}", e)
                        ));
                    }
                }
            });
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
                        for (_id, display_name, model, created) in &sessions {
                            info.push_str(&format!("\n  {}  — {}  ({})", display_name, model, created));
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
        "/search" => {
            if arg.is_empty() {
                let _ = event_tx.send(AppEvent::Info("Usage: /search <query> — full-text search across sessions".into()));
            } else {
                let results = search_sessions(arg);
                if results.is_empty() {
                    let _ = event_tx.send(AppEvent::Info(format!("No sessions matched: {}", arg)));
                } else {
                    let mut info = format!("Sessions matching '{}':", arg);
                    for (display_name, snippet) in &results {
                        info.push_str(&format!("\n  {}  — …{}…", display_name, snippet));
                    }
                    let _ = event_tx.send(AppEvent::Info(info));
                }
            }
            Ok(SlashResultTui::Continue)
        }
        "/reload" => {
            let config_path = match Config::default_path() {
                Ok(p) => p,
                Err(_) => {
                    let _ = event_tx.send(AppEvent::Error("Cannot determine config path".into()));
                    return Ok(SlashResultTui::Continue);
                }
            };
            match Config::load(&config_path) {
                Ok(cfg) => {
                    state.config = Arc::new(cfg);
                    if let Ok(new_provider) = crate::llm::resolve_provider(&state.config) {
                        state.provider = new_provider;
                    }
                    let _ = event_tx.send(AppEvent::Info("Config reloaded".into()));
                }
                Err(e) => {
                    let _ = event_tx.send(AppEvent::Error(format!("Failed to reload config: {}", e)));
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
                        let msgs = messages_to_display_msgs(&loaded.messages);
                        state.session = loaded;
                        let _ = event_tx.send(AppEvent::RebuildMessages(msgs));
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

pub async fn run_tui_interactive(mut state: ChatState, latest_release: Option<u32>) -> anyhow::Result<()> {
    use crate::tui::{self, TuiApp, AppEvent};

    add_system_prompt(&mut state);

    let (event_tx, event_rx) = std::sync::mpsc::channel::<AppEvent>();
    let (input_tx, input_rx) = std::sync::mpsc::channel::<String>();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_tui = cancelled.clone();
    let force_exit = Arc::new(AtomicBool::new(false));

    let model = state.model.clone();
    let provider_name = state.provider.name().to_string();
    let preferred_models = state.config.provider_config()
        .map(|c| c.effective_models())
        .unwrap_or_default();

    let mut app = TuiApp::new(&model, &provider_name, state.auto_route, event_rx, latest_release, &preferred_models, state.config.advanced.tool_output_truncation_chars);

    if let Some(ver) = latest_release {
        let _ = event_tx.send(AppEvent::UpdateAvailable(ver));
    }

    // Signal handler for graceful shutdown on SIGTERM/SIGINT
    let fe_tui = force_exit.clone();
    let cancelled_sig = cancelled.clone();
    let sig_input_tx = input_tx.clone();
    tokio::spawn(async move {
        #[cfg(unix)]
        let mut term_signal = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        ).ok();
        #[cfg(not(unix))]
        let mut term_signal: Option<tokio::signal::unix::Signal> = None;

        tokio::select! {
            _ = async {
                if let Some(ref mut sig) = term_signal {
                    sig.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {}
            _ = tokio::signal::ctrl_c() => {
                // First Ctrl+C during TUI is handled by TUI itself.
                // Only act if we're in non-TUI mode or the TUI didn't catch it.
            }
        }

        fe_tui.store(true, Ordering::SeqCst);
        cancelled_sig.store(true, Ordering::SeqCst);
        let _ = sig_input_tx.send("/exit".to_string());
    });

    // Spawn processing on tokio runtime
    let processing = tokio::spawn(async move {
        while let Ok(input) = input_rx.recv() {

            if input.starts_with('/') {
                match handle_slash_command_tui(&mut state, &input, &event_tx).await? {
                    SlashResultTui::Exit => break,
                    SlashResultTui::Continue => continue,
                }
            }

            if state.auto_route {
                state.auto_route_for_prompt(&input);
            }

            state.session.messages.push(Message::user(&input));
            state.session.auto_title();

            cancelled.store(false, Ordering::SeqCst);
            if !process_turn_tui(&mut state, &event_tx, cancelled.clone()).await? {
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
    let fe_tui_ref = force_exit.clone();
    let tui_handle = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let _guard = crate::term::RawModeGuard::new()?;
        let mut terminal = ratatui::Terminal::new(
            ratatui::backend::CrosstermBackend::new(std::io::stdout()),
        )?;
        terminal.clear()?;
        tui::run_tui(&mut app, &mut terminal, &tui_input_tx, &cancelled_tui, &fe_tui_ref)
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

/// Full-text search across all session files. Matches both titles and message content.
fn search_sessions(query: &str) -> Vec<(String, String)> {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return Vec::new(),
    };
    let dir = home.join(SESSION_DIR);
    if !dir.exists() {
        return Vec::new();
    }
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();
    let now = chrono::Utc::now();
    for entry in std::fs::read_dir(dir).ok().into_iter().flatten() {
        let entry = match entry {
            Ok(e) => e,
            _ => continue,
        };
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            _ => continue,
        };
        if !content.to_lowercase().contains(&query_lower) {
            continue;
        }
        let session: Session = match serde_json::from_str(&content) {
            Ok(s) => s,
            _ => continue,
        };
        // Skip expired sessions
        if let Ok(created) = chrono::DateTime::parse_from_rfc3339(&session.created_at) {
            let created_utc = created.with_timezone(&chrono::Utc);
            let age = now - created_utc;
            if age.num_days() > 30 {
                let _ = std::fs::remove_file(&path);
                continue;
            }
        }
        let display_name = if session.title.is_empty() {
            session.id[..8].to_string()
        } else {
            format!("{} ({})", session.title, &session.id[..8])
        };
        // Title match — show without snippet
        let title_lower = session.title.to_lowercase();
        if title_lower.contains(&query_lower) {
            results.push((display_name, String::new()));
            continue;
        }
        // Find a matching snippet in message content
        let snippet = session.messages.iter()
            .filter_map(|m| {
                for block in &m.content {
                    if let ContentBlock::Text { text } = block {
                        if let Some(pos) = text.to_lowercase().find(&query_lower) {
                            let start = pos.saturating_sub(40);
                            let end = (pos + query.len() + 40).min(text.len());
                            let snippet = if start > 0 { "…" } else { "" }.to_string()
                                + &text[start..end]
                                + if end < text.len() { "…" } else { "" };
                            return Some(snippet);
                        }
                    }
                }
                None
            })
            .next()
            .unwrap_or_default();
        results.push((display_name, snippet));
    }
    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
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
        let out = format_result(&r, 2000);
        assert_eq!(out, "ok");
    }

    #[test]
    fn test_format_result_long_truncated() {
        let long = "x".repeat(3000);
        let r = ToolResult { success: false, output: long };
        let out = format_result(&r, 2000);
        assert!(out.starts_with("ERROR (truncated, 3000 bytes total)"));
        // "ERROR (truncated, 3000 bytes total)\n" + 2000 content chars
        assert_eq!(out.len(), 36 + 2000);
    }

    #[test]
    fn test_format_result_success_prefix() {
        let r = ToolResult { success: true, output: "x".repeat(2500) };
        let out = format_result(&r, 2000);
        assert!(out.starts_with("SUCCESS (truncated, 2500 bytes total)"));
    }

    // --- strip_think tests ---

    #[test]
    fn test_strip_think_no_block() {
        let mut in_block = false;
        let result = strip_think("hello world", &mut in_block);
        assert_eq!(result, "hello world");
        assert!(!in_block);
    }

    #[test]
    fn test_strip_think_complete_block() {
        let mut in_block = false;
        let result = strip_think("hello <think>reasoning</think> world", &mut in_block);
        assert_eq!(result, "hello  world");
        assert!(!in_block);
    }

    #[test]
    fn test_strip_think_only_block() {
        let mut in_block = false;
        let result = strip_think("<think>deep reasoning</think>", &mut in_block);
        assert_eq!(result, "");
        assert!(!in_block);
    }

    #[test]
    fn test_strip_think_multi_blocks() {
        let mut in_block = false;
        let result = strip_think("a <think>r1</think> b <think>r2</think> c", &mut in_block);
        assert_eq!(result, "a  b  c");
        assert!(!in_block);
    }

    #[test]
    fn test_strip_think_streaming_open() {
        let mut in_block = false;
        // First chunk opens the block
        let r1 = strip_think("start <think>reasoning", &mut in_block);
        assert_eq!(r1, "start ");
        assert!(in_block);
        // Second chunk closes it
        let r2 = strip_think(" continues</think> end", &mut in_block);
        assert_eq!(r2, " end");
        assert!(!in_block);
    }

    #[test]
    fn test_strip_think_partial_tag_boundary() {
        // When <think> is split across chunks, the streaming strip_think
        // won't catch it. The final clean-up pass handles it.
        let mut in_block = false;
        let _r1 = strip_think("hello <thi", &mut in_block);
        assert!(!in_block);
        let r2 = strip_think("nk>hidden</think> world", &mut in_block);
        assert_eq!(r2, "nk>hidden</think> world");
        assert!(!in_block);
        // Final pass strips the split think block
        let combined = strip_think_final("hello <think>hidden</think> world".into());
        assert_eq!(combined, "hello  world");
    }

    #[test]
    fn test_strip_think_final_cleanup() {
        let result = strip_think_final("a <think>r1</think> b <think>r2</think> c".into());
        assert_eq!(result, "a  b  c");
    }

    #[test]
    fn test_strip_think_final_noop() {
        let result = strip_think_final("no think blocks here".into());
        assert_eq!(result, "no think blocks here");
    }
}
