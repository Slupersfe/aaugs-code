mod palette;
mod markdown;
mod chat;
mod browse;
mod help;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Color, Style};
use ratatui::widgets::Paragraph;
use ratatui::{Frame, Terminal};
use tokio::sync::oneshot;

use crate::cost;

#[derive(Debug)]
pub enum AppEvent {
    Text(String),
    ToolCall { name: String, args: String },
    ToolResult { name: String, output: String },
    ToolDenied { name: String },
    TurnDone,
    #[allow(dead_code)]
    Cancel,
    Error(String),
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
        cost: f64,
    },
    Question {
        question: String,
        options: Vec<String>,
        tx: oneshot::Sender<String>,
    },
    Info(String),
    ModelChanged(String),
    AutoRoute(bool),
    Clear,
    OpenBrowse,
    UpdateAvailable(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppMode {
    Chat,
    Help,
    Browse,
}

struct HelpPage {
    items: Vec<(String, String, Option<Color>)>,
}

pub struct TuiApp {
    pub mode: AppMode,
    pub messages: Vec<DisplayMsg>,
    pub input: String,
    pub cursor: usize,
    pub scroll: usize,
    pub is_loading: bool,
    pub model: String,
    pub auto_route: bool,
    pub provider: String,
    pub total_cost: f64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub event_rx: mpsc::Receiver<AppEvent>,

    help_content: Vec<HelpPage>,
    help_scroll: usize,
    pending_question: Option<(String, Vec<String>, oneshot::Sender<String>)>,

    browse_models: Vec<&'static str>,
    browse_cursor: usize,
    browse_filter: String,
    browse_preferred_count: usize,
    preferred: Vec<String>,
    update_available: Option<u32>,
    in_code_block: bool,
    auto_scroll: bool,
    pub tool_truncation: usize,
    spinner_frame: u8,
    search_query: String,
    search_active: bool,
}

#[derive(Debug, Clone)]
pub struct DisplayMsg {
    pub role: String,
    pub text: String,
}

impl TuiApp {
    pub fn new(
        model: &str,
        provider: &str,
        auto_route: bool,
        event_rx: mpsc::Receiver<AppEvent>,
        latest_release: Option<u32>,
        preferred_models: &[String],
        tool_truncation: usize,
    ) -> Self {
        let help_content = help::build_help_pages();
        Self {
            mode: AppMode::Chat,
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            is_loading: false,
            model: model.to_string(),
            auto_route,
            provider: provider.to_string(),
            total_cost: 0.0,
            input_tokens: 0,
            output_tokens: 0,
            event_rx,
            help_content,
            help_scroll: 0,
            pending_question: None,
            browse_models: Vec::new(),
            browse_cursor: 0,
            browse_filter: String::new(),
            preferred: preferred_models.to_vec(),
            browse_preferred_count: 0,
            update_available: latest_release,
            in_code_block: false,
            auto_scroll: true,
            tool_truncation,
            spinner_frame: 0,
            search_query: String::new(),
            search_active: false,
        }
    }

    fn mark_dirty(&mut self) {}

    pub fn add_user_msg(&mut self, text: &str) {
        self.messages.push(DisplayMsg {
            role: "User".to_string(),
            text: text.to_string(),
        });
        self.mark_dirty();
    }

    pub fn add_assistant_text(&mut self, text: &str) {
        if let Some(last) = self.messages.last_mut() {
            if last.role == "Assistant" {
                last.text.push_str(text);
                self.mark_dirty();
                return;
            }
        }
        self.messages.push(DisplayMsg {
            role: "Assistant".to_string(),
            text: text.to_string(),
        });
        self.mark_dirty();
    }

    pub fn add_tool_msg(&mut self, name: &str, output: &str) {
        let limit = self.tool_truncation;
        let display = if output.len() > limit {
            format!("── {} ──\n{}... (truncated)", name, &output[..limit])
        } else {
            format!("── {} ──\n{}", name, output)
        };
        self.mark_dirty();
        self.messages.push(DisplayMsg {
            role: "Tool".to_string(),
            text: display,
        });
    }

    pub fn add_tool_denied(&mut self, name: &str) {
        let msg = format!("tool '{}' denied", name);
        self.mark_dirty();
        self.messages.push(DisplayMsg {
            role: "Tool".to_string(),
            text: msg,
        });
    }

    pub fn add_error(&mut self, err: &str) {
        self.messages.push(DisplayMsg {
            role: "Error".to_string(),
            text: err.to_string(),
        });
        self.mark_dirty();
    }

    pub fn scroll_up(&mut self) {
        self.auto_scroll = false;
        self.scroll = self.scroll.saturating_sub(1);
    }

    pub fn scroll_down(&mut self) {
        let total: usize = self.messages.iter().map(|m| m.text.lines().count()).sum();
        let max_scroll = total.saturating_sub(1);
        self.scroll = self.scroll.saturating_add(1).min(max_scroll);
        self.auto_scroll = self.scroll >= max_scroll;
    }

    pub fn open_browse(&mut self) {
        let all_models = cost::models_for_provider(&self.provider);
        let all_models: Vec<&'static str> = if all_models.is_empty() {
            cost::MODEL_ITER.iter().map(|(n, _)| *n).collect()
        } else {
            all_models
        };

        let mut seen = std::collections::HashSet::new();
        let mut merged = Vec::new();

        for pref in &self.preferred {
            if let Some(m) = all_models.iter().find(|m| **m == pref.as_str()) {
                if seen.insert(m) {
                    merged.push(*m);
                }
            }
        }
        self.browse_preferred_count = merged.len();

        for m in &all_models {
            if seen.insert(m) {
                merged.push(*m);
            }
        }

        self.browse_models = merged;
        self.browse_cursor = 0;
        self.browse_filter = self.model.clone();
        if let Some(pos) = self
            .browse_models
            .iter()
            .position(|m| *m == self.model.as_str())
        {
            self.browse_cursor = pos;
        }
        self.mode = AppMode::Browse;
    }

    pub fn help_scroll_up(&mut self) {
        self.help_scroll = self.help_scroll.saturating_sub(1);
    }

    pub fn help_scroll_down(&mut self) {
        self.help_scroll = self.help_scroll.saturating_add(1);
    }

    /// Move cursor to start of previous word (or to position 0).
    fn word_left(&mut self) {
        let before = &self.input[..byte_idx(&self.input, self.cursor)];
        let pos = before
            .char_indices()
            .rev()
            .skip_while(|(_, c)| c.is_whitespace())
            .skip_while(|(_, c)| !c.is_whitespace())
            .next()
            .map(|(i, _)| self.input[..i].chars().count())
            .unwrap_or(0);
        self.cursor = pos;
    }

    /// Move cursor to start of next word (or to end of input).
    fn word_right(&mut self) {
        let after = &self.input[byte_idx(&self.input, self.cursor)..];
        let pos = self.cursor
            + after
                .chars()
                .skip_while(|c| c.is_whitespace())
                .skip_while(|c| !c.is_whitespace())
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
        self.cursor = pos.min(self.input.len());
        // If we didn't move at all, jump to end
        if self.cursor < after.len() {
            let remaining = &self.input[byte_idx(&self.input, self.cursor)..];
            let next_word = remaining
                .chars()
                .skip_while(|c| c.is_whitespace());
            let skip: usize = next_word.map(|c| c.len_utf8()).sum();
            if skip > 0 {
                self.cursor = (self.cursor + skip).min(self.input.len());
            }
        }
    }

    /// Delete the word behind the cursor (Ctrl+Backspace).
    fn delete_word_left(&mut self) {
        let before = &self.input[..byte_idx(&self.input, self.cursor)];
        let pos = before
            .char_indices()
            .rev()
            .skip_while(|(_, c)| c.is_whitespace())
            .skip_while(|(_, c)| !c.is_whitespace())
            .next()
            .map(|(i, _)| i)
            .unwrap_or(0);
        let old_len = self.input.len();
        self.input.drain(pos..byte_idx(&self.input, self.cursor));
        let removed = self.input.len().abs_diff(old_len);
        self.cursor = self.cursor.saturating_sub(removed);
    }
}

// ── Render dispatcher ─────────────────────────────────────────────────

pub fn draw(frame: &mut Frame, app: &mut TuiApp) {
    let bg = Paragraph::new("").style(Style::default().bg(palette::SURFACE));
    frame.render_widget(bg, frame.area());
    match app.mode {
        AppMode::Chat => chat::draw_chat(frame, app),
        AppMode::Help => help::draw_help(frame, app),
        AppMode::Browse => browse::draw_browse(frame, app),
    }
}

// ── TUI runner ────────────────────────────────────────────────────────

pub fn run_tui(
    app: &mut TuiApp,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    input_tx: &mpsc::Sender<String>,
    cancelled: &Arc<AtomicBool>,
    force_exit: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    loop {
        if force_exit.load(Ordering::Relaxed) {
            return Ok(());
        }

        terminal.draw(|f| draw(f, app))?;
        app.spinner_frame = app.spinner_frame.wrapping_add(1);

        loop {
            match app.event_rx.try_recv() {
                Ok(event) => handle_app_event(app, event),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            }
        }

        if event::poll(Duration::from_millis(10))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    let mut should_exit = false;
                    match app.mode {
                        AppMode::Chat => {
                            handle_chat_key(app, key, input_tx, &mut should_exit, cancelled);
                            if should_exit {
                                return Ok(());
                            }
                        }
                        AppMode::Help => {
                            help::handle_help_key(app, key);
                        }
                        AppMode::Browse => {
                            browse::handle_browse_key(app, key, input_tx);
                        }
                    }
                }
                Event::Resize(_, _) => {
                    let _ = terminal.clear();
                }
                _ => {}
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

fn byte_idx(s: &str, cursor: usize) -> usize {
    s.char_indices()
        .nth(cursor)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

fn handle_chat_key(
    app: &mut TuiApp,
    key: crossterm::event::KeyEvent,
    input_tx: &mpsc::Sender<String>,
    should_exit: &mut bool,
    cancelled: &Arc<AtomicBool>,
) {
    if app.pending_question.is_some() {
        handle_question_input(app, key, should_exit);
        return;
    }

    match key.code {
        // ── Search mode ──────────────────────────────────
        KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) && !app.search_active => {
            app.search_active = true;
            app.search_query.clear();
        }
        KeyCode::Esc if app.search_active => {
            app.search_active = false;
            app.search_query.clear();
        }
        KeyCode::Enter if app.search_active => {
            app.search_active = false;
        }
        KeyCode::Backspace if app.search_active => {
            app.search_query.pop();
        }
        KeyCode::Char(c) if app.search_active => {
            app.search_query.push(c);
        }

        // ── Normal mode ──────────────────────────────────
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.is_loading {
                cancelled.store(true, Ordering::SeqCst);
                app.is_loading = false;
            } else {
                *should_exit = true;
            }
        }
        KeyCode::Right if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.word_right();
        }
        KeyCode::Left if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.word_left();
        }
        KeyCode::Backspace if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.delete_word_left();
        }
        KeyCode::Tab | KeyCode::F(1) => {
            app.mode = AppMode::Help;
            app.help_scroll = 0;
        }
        KeyCode::Up => {
            app.scroll_up();
        }
        KeyCode::Down => {
            app.scroll_down();
        }
        KeyCode::PageUp => {
            app.auto_scroll = false;
            app.scroll = app.scroll.saturating_sub(10);
        }
        KeyCode::PageDown => {
            app.scroll = app.scroll.saturating_add(10);
            let total: usize = app.messages.iter().map(|m| m.text.lines().count()).sum();
            let max_scroll = total.saturating_sub(1);
            app.auto_scroll = app.scroll >= max_scroll;
        }
        KeyCode::Left => {
            app.cursor = app.cursor.saturating_sub(1);
        }
        KeyCode::Right => {
            app.cursor = app
                .cursor
                .min(app.input.len().saturating_sub(1))
                .saturating_add(1);
            app.cursor = app.cursor.min(app.input.len());
        }
        KeyCode::Home => {
            app.cursor = 0;
        }
        KeyCode::End => {
            app.cursor = app.input.len();
        }
        KeyCode::Enter => {
            let input = app.input.trim().to_string();
            if !input.is_empty() && !app.is_loading {
                app.add_user_msg(&input);
                app.is_loading = true;
                app.auto_scroll = true;
                let _ = input_tx.send(input);
            }
            app.input.clear();
            app.cursor = 0;
        }
        KeyCode::Backspace => {
            if app.cursor > 0 {
                let idx = byte_idx(&app.input, app.cursor.saturating_sub(1));
                app.input.drain(idx..byte_idx(&app.input, app.cursor));
                app.cursor = app.cursor.saturating_sub(1);
            }
        }
        KeyCode::Delete => {
            if app.cursor < app.input.len() {
                let start = byte_idx(&app.input, app.cursor);
                let end = byte_idx(&app.input, app.cursor.saturating_add(1));
                app.input.drain(start..end);
            }
        }
        KeyCode::Char(c) if !app.is_loading => {
            let pos = byte_idx(&app.input, app.cursor);
            app.input.insert(pos, c);
            app.cursor = app.cursor.saturating_add(1).min(app.input.len());
        }
        _ => {}
    }
}

fn handle_question_input(
    app: &mut TuiApp,
    key: crossterm::event::KeyEvent,
    should_exit: &mut bool,
) {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            *should_exit = true;
        }
        KeyCode::Enter => {
            if let Some((_, _, tx)) = app.pending_question.take() {
                let answer = app.input.trim().to_string();
                let _ = tx.send(answer);
                app.input.clear();
            }
        }
        KeyCode::Backspace => {
            app.input.pop();
        }
        KeyCode::Esc => {
            if let Some((_, _, tx)) = app.pending_question.take() {
                let _ = tx.send(String::new());
            }
            app.input.clear();
        }
        KeyCode::Tab | KeyCode::F(1) => {
            app.mode = AppMode::Help;
            app.help_scroll = 0;
        }
        KeyCode::Char(c) => {
            app.input.push(c);
        }
        _ => {}
    }
}

fn handle_app_event(app: &mut TuiApp, event: AppEvent) {
    match event {
        AppEvent::Text(text) => {
            app.add_assistant_text(&text);
        }
        AppEvent::ToolCall { name, args } => {
            app.add_assistant_text(&format!("\n-- tool: {} --", name));
            if !args.is_empty() && args != "null" {
                app.add_assistant_text(&format!(" {}", args));
            }
        }
        AppEvent::ToolResult { name, output } => {
            app.add_tool_msg(&name, &output);
        }
        AppEvent::ToolDenied { name } => {
            app.add_tool_denied(&name);
        }
        AppEvent::TurnDone => {
            app.is_loading = false;
        }
        AppEvent::Cancel => {
            app.is_loading = false;
        }
        AppEvent::Error(err) => {
            app.add_error(&err);
            app.is_loading = false;
        }
        AppEvent::Usage {
            prompt_tokens,
            completion_tokens,
            cost,
        } => {
            app.input_tokens += prompt_tokens;
            app.output_tokens += completion_tokens;
            app.total_cost += cost;
        }
        AppEvent::Question {
            question,
            options,
            tx,
        } => {
            app.pending_question = Some((question, options, tx));
        }
        AppEvent::Info(text) => {
            app.messages.push(DisplayMsg {
                role: "System".to_string(),
                text,
            });
            app.is_loading = false;
        }
        AppEvent::ModelChanged(model) => {
            app.model = model;
        }
        AppEvent::AutoRoute(enabled) => {
            app.auto_route = enabled;
        }
        AppEvent::Clear => {
            app.messages.clear();
            app.scroll = 0;
        }
        AppEvent::OpenBrowse => {
            app.open_browse();
        }
        AppEvent::UpdateAvailable(ver) => {
            app.update_available = Some(ver);
        }
    }
}
