use std::sync::mpsc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
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
    Error(String),
    Usage { prompt_tokens: u32, completion_tokens: u32, cost: f64 },
    Question {
        question: String,
        options: Vec<String>,
        tx: oneshot::Sender<String>,
    },
    Info(String),
    ModelChanged(String),
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
    pub scroll: usize,
    pub is_loading: bool,
    pub model: String,
    pub provider: String,
    pub total_cost: f64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub event_rx: mpsc::Receiver<AppEvent>,
    estimated_lines: usize,

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
        event_rx: mpsc::Receiver<AppEvent>,
        latest_release: Option<u32>,
        preferred_models: &[String],
    ) -> Self {
        let help_content = build_help_pages();
        Self {
            mode: AppMode::Chat,
            messages: Vec::new(),
            input: String::new(),
            scroll: 0,
            is_loading: false,
            model: model.to_string(),
            provider: provider.to_string(),
            total_cost: 0.0,
            input_tokens: 0,
            output_tokens: 0,
            event_rx,
            estimated_lines: 0,
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
        }
    }

    fn track_lines(&mut self, text: &str) {
        self.estimated_lines += text.lines().count().max(1);
        self.auto_scroll = true;
    }

    pub fn add_user_msg(&mut self, text: &str) {
        self.messages.push(DisplayMsg {
            role: "User".to_string(),
            text: text.to_string(),
        });
        self.track_lines(text);
    }

    pub fn add_assistant_text(&mut self, text: &str) {
        if let Some(last) = self.messages.last_mut() {
            if last.role == "Assistant" {
                last.text.push_str(text);
                self.track_lines(text);
                return;
            }
        }
        self.messages.push(DisplayMsg {
            role: "Assistant".to_string(),
            text: text.to_string(),
        });
        self.track_lines(text);
    }

    pub fn add_tool_msg(&mut self, name: &str, output: &str) {
        let display = if output.len() > 200 {
            format!("── {} ──\n{}... (truncated)", name, &output[..200])
        } else {
            format!("── {} ──\n{}", name, output)
        };
        self.track_lines(&display);
        self.messages.push(DisplayMsg {
            role: "Tool".to_string(),
            text: display,
        });
    }

    pub fn add_tool_denied(&mut self, name: &str) {
        let msg = format!("tool '{}' denied", name);
        self.track_lines(&msg);
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
        self.track_lines(err);
    }

    pub fn add_done(&mut self) {
        self.messages.push(DisplayMsg {
            role: "System".to_string(),
            text: "── done ──".to_string(),
        });
        self.track_lines("done");
    }

    pub fn scroll_up(&mut self) {
        self.auto_scroll = false;
        self.scroll = self.scroll.saturating_sub(1);
    }

    pub fn scroll_down(&mut self) {
        let max_scroll = self.estimated_lines.saturating_sub(1);
        self.auto_scroll = false;
        self.scroll = self.scroll.saturating_add(1).min(max_scroll);
    }

    pub fn open_browse(&mut self) {
        let all_models = cost::models_for_provider(&self.provider);
        let all_models: Vec<&'static str> = if all_models.is_empty() {
            cost::MODEL_ITER.iter().map(|(n, _)| *n).collect()
        } else {
            all_models
        };

        // Build list: preferred models first, then all others, no duplicates
        let mut seen = std::collections::HashSet::new();
        let mut merged = Vec::new();

        // We don't store preferred_models as &'static str, so match against all_models
        // to find the static versions
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
        if let Some(pos) = self.browse_models.iter().position(|m| *m == self.model.as_str()) {
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
}

// --- Help content builder ---

fn build_help_pages() -> Vec<HelpPage> {
    let mut pages = Vec::new();

    // Commands
    pages.push(HelpPage {
        items: vec![
            ("Commands".to_string(), String::new(), Some(Color::Cyan)),
            (String::new(), String::new(), None),
            ("  /help".to_string(), "Show this help".to_string(), None),
            ("  /exit  /quit".to_string(), "Exit the program".to_string(), None),
            ("  /clear".to_string(), "Clear conversation".to_string(), None),
            ("  /model [name]".to_string(), "Show or change model".to_string(), None),
            ("  /browse".to_string(), "Browse & select models".to_string(), None),
            ("  /update".to_string(), "Pull latest & rebuild".to_string(), None),
            ("  /provider [name]".to_string(), "Switch provider in-session".to_string(), None),
            ("  /tokens".to_string(), "Show token usage, context & cost".to_string(), None),
            ("  /summarize".to_string(), "Manually summarize context".to_string(), None),
            ("  /sessions".to_string(), "List saved sessions".to_string(), None),
            ("  /resume <id>".to_string(), "Resume a saved session".to_string(), None),
        ],
    });

    // Providers and pricing
    let mut provider_items = vec![
        ("Providers & Pricing".to_string(), String::new(), Some(Color::Cyan)),
        (String::new(), String::new(), None),
        ("Each row: model  —  input price / output price per 1M tokens".to_string(), String::new(), Some(Color::DarkGray)),
        (String::new(), String::new(), None),
    ];

    let known = [
        ("anthropic/claude-", "OpenRouter (Claude)"),
        ("openai/gpt-", "OpenRouter (OpenAI)"),
        ("openai/o", "OpenRouter (OpenAI Reasoning)"),
        ("google/gemini-", "OpenRouter (Gemini)"),
        ("deepseek/", "OpenRouter (DeepSeek)"),
        ("meta-llama/", "OpenRouter (Llama)"),
        ("mistralai/", "OpenRouter (Mistral)"),
        ("qwen/", "OpenRouter (Qwen)"),
        ("cohere/", "OpenRouter (Cohere)"),
    ];

    for (prefix, section_name) in &known {
        let mut section_items: Vec<(String, String, Option<Color>)> = Vec::new();
        section_items.push((format!("  {} (recommended)", section_name), String::new(), Some(Color::Green)));
        for (model_name, (inp, outp)) in cost::MODEL_ITER.iter() {
            if model_name.starts_with(prefix) {
                let price = format!("    {}   ${:.2}/{}k / ${:.2}/{}k", model_name, inp, (inp * 4.0) as u32, outp, (outp * 4.0) as u32);
                section_items.push((price, String::new(), None));
            }
        }
        if section_items.len() > 1 {
            provider_items.extend(section_items);
            provider_items.push((String::new(), String::new(), None));
        }
    }

    // Dedicated providers
    let dedicated = [
        ("Anthropic".to_string(), vec![
            ("claude-sonnet-4-20250514", "$3.00 / $15.00 per 1M"),
            ("claude-sonnet-4.5-20250514", "$3.00 / $15.00 per 1M"),
            ("claude-opus-4-20250514", "$15.00 / $75.00 per 1M"),
            ("claude-haiku-3-5", "$0.80 / $4.00 per 1M"),
        ]),
        ("OpenAI".to_string(), vec![
            ("gpt-4o", "$2.50 / $10.00 per 1M"),
            ("gpt-4o-mini", "$0.15 / $0.60 per 1M"),
            ("gpt-4-turbo", "$10.00 / $30.00 per 1M"),
        ]),
        ("Gemini".to_string(), vec![
            ("gemini-2.5-pro", "$1.25 / $5.00 per 1M"),
            ("gemini-2.5-flash", "$0.15 / $0.60 per 1M"),
        ]),
        ("OpenCode Zen".to_string(), vec![
            ("big-pickle", "Free"),
        ]),
    ];

    provider_items.push(("".to_string(), String::new(), Some(Color::Cyan)));
    provider_items.push(("Dedicated Providers".to_string(), String::new(), Some(Color::Cyan)));
    provider_items.push((String::new(), String::new(), None));
    for (provider_name, models) in &dedicated {
        provider_items.push((format!("  {}", provider_name), String::new(), Some(Color::Green)));
        for (model_name, price_str) in models {
            provider_items.push((format!("    {}   {}", model_name, price_str), String::new(), None));
        }
        provider_items.push((String::new(), String::new(), None));
    }

    pages.push(HelpPage { items: provider_items });

    // Settings
    pages.push(HelpPage {
        items: vec![
            ("Settings".to_string(), String::new(), Some(Color::Cyan)),
            (String::new(), String::new(), None),
            ("  Config file".to_string(), "~/.config/vibe/vibe.json or --config".to_string(), None),
            ("  Max tokens".to_string(), "4096 (configurable)".to_string(), None),
            ("  Temperature".to_string(), "0.0 (configurable)".to_string(), None),
            ("  Timeout".to_string(), "120s (configurable)".to_string(), None),
            ("  Permissions".to_string(), "ask/allow/deny per tool".to_string(), None),
            (String::new(), String::new(), None),
            ("Keybinds".to_string(), String::new(), Some(Color::Cyan)),
            (String::new(), String::new(), None),
            ("  Enter".to_string(), "Submit message".to_string(), None),
            ("  Ctrl+C".to_string(), "Exit".to_string(), None),
            ("  Tab / F1".to_string(), "Toggle help screen".to_string(), None),
            ("  PgUp / PgDn".to_string(), "Scroll chat / help".to_string(), None),
            ("  Up / Down".to_string(), "Scroll messages".to_string(), None),
        ],
    });

    pages
}

// --- Markdown rendering ---

fn push_md_line(text: &mut Text, line: &str, base_style: Style, in_code: &mut bool) {
    if *in_code {
        if line.trim_end() == "```" {
            *in_code = false;
            return;
        }
        text.push_line(Line::from(Span::styled(line.to_string(), base_style.bg(Color::Rgb(40, 40, 40)))));
        return;
    }
    if line.trim_start().starts_with("```") {
        *in_code = true;
        return;
    }

    let trimmed = line.trim_start();
    // Headers
    if let Some(rest) = trimmed.strip_prefix("### ") {
        text.push_line(Line::from(Span::styled(rest.to_string(), base_style.fg(Color::Cyan).add_modifier(Modifier::BOLD))));
        return;
    }
    if let Some(rest) = trimmed.strip_prefix("## ") {
        text.push_line(Line::from(Span::styled(rest.to_string(), base_style.fg(Color::Cyan).add_modifier(Modifier::BOLD))));
        return;
    }
    if let Some(rest) = trimmed.strip_prefix("# ") {
        text.push_line(Line::from(Span::styled(rest.to_string(), base_style.fg(Color::Yellow).add_modifier(Modifier::BOLD))));
        return;
    }

    // Inline markdown: parse **bold** and `code`
    let spans = parse_inline_md(line, base_style);
    text.push_line(Line::from(spans));
}

fn parse_inline_md(line: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut buf = String::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Bold **...**
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), base));
            }
            i += 2;
            let mut bold = String::new();
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '*') {
                bold.push(chars[i]);
                i += 1;
            }
            if i + 1 < chars.len() {
                i += 2; // skip closing **
            }
            spans.push(Span::styled(bold, base.add_modifier(Modifier::BOLD)));
            continue;
        }
        // Inline code `...`
        if chars[i] == '`' {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), base));
            }
            i += 1;
            let mut code = String::new();
            while i < chars.len() && chars[i] != '`' {
                code.push(chars[i]);
                i += 1;
            }
            if i < chars.len() {
                i += 1; // skip closing `
            }
            spans.push(Span::styled(code, base.fg(Color::Cyan)));
            continue;
        }
        buf.push(chars[i]);
        i += 1;
    }
    if !buf.is_empty() {
        spans.push(Span::styled(buf, base));
    }
    spans
}

pub fn draw(frame: &mut Frame, app: &mut TuiApp) {
    match app.mode {
        AppMode::Chat => draw_chat(frame, app),
        AppMode::Help => draw_help(frame, app),
        AppMode::Browse => draw_browse(frame, app),
    }
}

fn draw_chat(frame: &mut Frame, app: &mut TuiApp) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

    // Status bar
    let update_indicator = if let Some(ver) = app.update_available {
        format!(" UPDATE v{} available! Run /update ", ver)
    } else {
        String::new()
    };
    let status = format!(
        " aaugs-code v{}  |  {}  |  {}  |  tokens: {} in / {} out  |  ${:.6}{}",
        env!("CARGO_PKG_VERSION"),
        app.provider,
        app.model,
        app.input_tokens,
        app.output_tokens,
        app.total_cost,
        update_indicator,
    );
    let status_bar = Paragraph::new(Span::styled(
        status,
        Style::default().fg(Color::White).bg(Color::Blue),
    ));
    frame.render_widget(status_bar, chunks[0]);

    // Messages with markdown rendering
    let mut text = Text::default();
    let mut in_code = app.in_code_block;
    for msg in &app.messages {
        let base_style = match msg.role.as_str() {
            "User" => Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            "Assistant" => Style::default().fg(Color::White),
            "Tool" => Style::default().fg(Color::Yellow),
            "Error" => Style::default().fg(Color::Red),
            "System" => Style::default().fg(Color::DarkGray),
            _ => Style::default(),
        };
        let prefix = if msg.role == "User" { "> " } else if msg.role == "Tool" { "  " } else { "  " };
        for (i, line) in msg.text.lines().enumerate() {
            if i == 0 && !prefix.trim().is_empty() {
                text.push_line(Line::from(vec![Span::styled(prefix, base_style)]));
            }
            push_md_line(&mut text, line, base_style, &mut in_code);
        }
    }
    app.in_code_block = in_code;

    let msg_area = chunks[1];
    let msg_area_inner = Rect {
        x: msg_area.x,
        y: msg_area.y,
        width: msg_area.width,
        height: msg_area.height.saturating_sub(1),
    };

    // Auto-scroll to bottom when new content arrives
    if app.auto_scroll {
        let h = msg_area_inner.height as usize;
        app.scroll = app.estimated_lines.saturating_sub(h);
    }

    let msg_widget = Paragraph::new(text)
        .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(Color::DarkGray)))
        .wrap(Wrap { trim: false })
        .scroll((app.scroll as u16, 0));
    frame.render_widget(msg_widget, msg_area_inner);

    // Input
    let input_style = if app.is_loading {
        Style::default().fg(Color::DarkGray)
    } else if app.pending_question.is_some() {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::White)
    };
    let input_text = if app.is_loading {
        " waiting for response...".to_string()
    } else if app.pending_question.is_some() {
        if app.input.is_empty() {
            " answer y/n...".to_string()
        } else {
            format!(" {}", app.input)
        }
    } else if app.input.is_empty() {
        " type a message...".to_string()
    } else {
        format!(" {}", app.input)
    };
    let input_widget = Paragraph::new(Span::styled(input_text, input_style))
        .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(Color::DarkGray)));
    frame.render_widget(input_widget, chunks[2]);

    // Keybinds bar
    let keys = Paragraph::new(Line::from(vec![
        Span::styled(" Tab/F1:Help ", Style::default().fg(Color::Cyan)),
        Span::styled(" | Down:Scroll ", Style::default().fg(Color::Cyan)),
        Span::styled(" Enter:Send ", Style::default().fg(Color::Cyan)),
        Span::styled(" Ctrl+C:Quit", Style::default().fg(Color::Cyan)),
    ]));
    frame.render_widget(keys, chunks[3]);

    // Question prompt overlay
    if let Some((question, options, _)) = &app.pending_question {
        let question_area = Rect {
            x: area.width / 6,
            y: area.height / 3,
            width: area.width * 2 / 3,
            height: 5,
        };
        frame.render_widget(ratatui::widgets::Clear, question_area);

        let options_str = if options.is_empty() {
            String::new()
        } else {
            format!(" [{}]", options.join(", "))
        };
        let input_display = if app.input.is_empty() {
            "type answer...".to_string()
        } else {
            app.input.clone()
        };
        let content = format!(
            "{}{}\n\n> {}",
            question, options_str, input_display
        );
        let prompt = Paragraph::new(Span::styled(content, Style::default().fg(Color::Yellow)))
            .block(
                Block::default()
                    .title(" Permission Required ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow)),
            );
        frame.render_widget(prompt, question_area);
    }
}

fn draw_help(frame: &mut Frame, app: &TuiApp) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(area);

    let mut text = Text::default();
    for page in &app.help_content {
        for (line_text, desc, color) in &page.items {
            let style = match color {
                Some(c) => Style::default().fg(*c).add_modifier(Modifier::BOLD),
                None => Style::default().fg(Color::White),
            };
            let content = if desc.is_empty() {
                line_text.clone()
            } else {
                format!("  {:<30} {}", line_text, desc)
            };
            text.push_line(Line::from(Span::styled(content, style)));
        }
    }

    let block = Block::default()
        .title(" Help & Reference ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let help = Paragraph::new(text)
        .block(block)
        .scroll((app.help_scroll as u16, 0));
    frame.render_widget(help, chunks[0]);

    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" | /:Scroll ", Style::default().fg(Color::Cyan)),
        Span::styled(" Tab/F1/Esc:Back ", Style::default().fg(Color::Cyan)),
    ]));
    frame.render_widget(footer, chunks[1]);
}

fn draw_browse(frame: &mut Frame, app: &TuiApp) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(area);

    let mut text = Text::default();
    text.push_line(Line::from(Span::styled(
        format!(" Provider: {}  |  Current model: {}", app.provider, app.model),
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));
    text.push_line(Line::from(""));

    let preferred_count = app.browse_preferred_count;
    let all_count = app.browse_models.len();

    for (i, model_name) in app.browse_models.iter().enumerate() {
        // Section headers
        if i == 0 && preferred_count > 0 {
            text.push_line(Line::from(Span::styled(
                " ★ Preferred Models",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            )));
            text.push_line(Line::from(""));
        }
        if i == preferred_count && preferred_count > 0 && i < all_count {
            text.push_line(Line::from(Span::styled(
                " ── All Models ──",
                Style::default().fg(Color::DarkGray),
            )));
            text.push_line(Line::from(""));
        }

        let prefix = if i == app.browse_cursor { " > " } else { "   " };
        let (inp, outp) = cost::model_cost(model_name);
        let line = format!("{}{}  (${:.2}/{}k in, ${:.2}/{}k out)",
            prefix, model_name, inp, (inp * 4.0) as u32, outp, (outp * 4.0) as u32);
        let style = if i == app.browse_cursor {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else if i < preferred_count {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::White)
        };
        text.push_line(Line::from(Span::styled(line, style)));
    }

    let block = Block::default()
        .title(" Model Browser ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));
    let browse = Paragraph::new(text)
        .block(block)
        .scroll((app.browse_cursor.saturating_sub(5) as u16, 0));
    frame.render_widget(browse, chunks[0]);

    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" | /:Navigate ", Style::default().fg(Color::Cyan)),
        Span::styled(" Enter:Select ", Style::default().fg(Color::Cyan)),
        Span::styled(" Esc/Tab:Back ", Style::default().fg(Color::Cyan)),
    ]));
    frame.render_widget(footer, chunks[1]);
}

// --- TUI runner ---

pub fn run_tui(
    app: &mut TuiApp,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    input_tx: &mpsc::Sender<String>,
) -> anyhow::Result<()> {
    loop {
        terminal.draw(|f| draw(f, app))?;

        // Check for app events from processing task
        match app.event_rx.try_recv() {
            Ok(event) => {
                handle_app_event(app, event);
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                while let Ok(e) = app.event_rx.try_recv() {
                    handle_app_event(app, e);
                }
                return Ok(());
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }

        // Poll keyboard
        if event::poll(Duration::from_millis(10))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    let mut should_exit = false;
                    match app.mode {
                        AppMode::Chat => {
                            handle_chat_key(app, key, input_tx, &mut should_exit);
                            if should_exit {
                                return Ok(());
                            }
                        }
                        AppMode::Help => {
                            handle_help_key(app, key);
                        }
                        AppMode::Browse => {
                            handle_browse_key(app, key, input_tx);
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

fn handle_chat_key(
    app: &mut TuiApp,
    key: crossterm::event::KeyEvent,
    input_tx: &mpsc::Sender<String>,
    should_exit: &mut bool,
) {
    // Handle question prompts first
    if app.pending_question.is_some() {
        handle_question_input(app, key, should_exit);
        return;
    }

    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            *should_exit = true;
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
            app.scroll = app.scroll.saturating_sub(10);
        }
        KeyCode::PageDown => {
            app.scroll = app.scroll.saturating_add(10);
        }
        KeyCode::Enter => {
            let input = app.input.trim().to_string();
            if !input.is_empty() && !app.is_loading {
                app.add_user_msg(&input);
                app.is_loading = true;
                let _ = input_tx.send(input);
            }
            app.input.clear();
        }
        KeyCode::Backspace => {
            app.input.pop();
        }
        KeyCode::Char(c) if !app.is_loading => {
            app.input.push(c);
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

fn handle_browse_key(
    app: &mut TuiApp,
    key: crossterm::event::KeyEvent,
    input_tx: &mpsc::Sender<String>,
) {
    match key.code {
        KeyCode::Esc | KeyCode::Tab | KeyCode::F(1) => {
            app.mode = AppMode::Chat;
        }
        KeyCode::Up => {
            if app.browse_cursor > 0 {
                app.browse_cursor -= 1;
            }
        }
        KeyCode::Down => {
            if app.browse_cursor + 1 < app.browse_models.len() {
                app.browse_cursor += 1;
            }
        }
        KeyCode::PageUp => {
            app.browse_cursor = app.browse_cursor.saturating_sub(10);
        }
        KeyCode::PageDown => {
            app.browse_cursor = app.browse_cursor.saturating_add(10)
                .min(app.browse_models.len().saturating_sub(1));
        }
        KeyCode::Enter => {
            if let Some(model) = app.browse_models.get(app.browse_cursor) {
                let cmd = format!("/model {}", model);
                let _ = input_tx.send(cmd);
            }
            app.mode = AppMode::Chat;
        }
        KeyCode::Char(c) => {
            app.browse_filter.push(c);
            // Filter models by typed prefix
            let filter = app.browse_filter.to_lowercase();
            if let Some(pos) = app.browse_models.iter().position(|m| m.to_lowercase().contains(&filter)) {
                app.browse_cursor = pos;
            }
        }
        KeyCode::Backspace => {
            app.browse_filter.pop();
        }
        _ => {}
    }
}

fn handle_help_key(app: &mut TuiApp, key: crossterm::event::KeyEvent) {
    match key.code {
        KeyCode::Esc | KeyCode::Tab | KeyCode::F(1) => {
            app.mode = AppMode::Chat;
        }
        KeyCode::Up => {
            app.help_scroll_up();
        }
        KeyCode::Down => {
            app.help_scroll_down();
        }
        KeyCode::PageUp => {
            app.help_scroll = app.help_scroll.saturating_sub(10);
        }
        KeyCode::PageDown => {
            app.help_scroll = app.help_scroll.saturating_add(10);
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
            app.add_done();
            app.is_loading = false;
        }
        AppEvent::Error(err) => {
            app.add_error(&err);
            app.is_loading = false;
        }
        AppEvent::Usage { prompt_tokens, completion_tokens, cost } => {
            app.input_tokens += prompt_tokens;
            app.output_tokens += completion_tokens;
            app.total_cost += cost;
        }
        AppEvent::Question { question, options, tx } => {
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
        AppEvent::Clear => {
            app.messages.clear();
            app.scroll = 0;
            app.estimated_lines = 0;
        }
        AppEvent::OpenBrowse => {
            app.open_browse();
        }
        AppEvent::UpdateAvailable(ver) => {
            app.update_available = Some(ver);
        }
    }
}
