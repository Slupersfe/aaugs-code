use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Frame;

use super::palette::{BLUE, MAUVE};
use super::HelpPage;
use super::TuiApp;

pub(super) fn build_help_pages() -> Vec<HelpPage> {
    let mut pages = Vec::new();

    pages.push(HelpPage {
        items: vec![
            ("Commands".to_string(), String::new(), Some(MAUVE)),
            (String::new(), String::new(), None),
            ("  /help".to_string(), "Show this help".to_string(), None),
            ("  /exit  /quit".to_string(), "Exit the program".to_string(), None),
            ("  /clear".to_string(), "Clear screen (visual only)".to_string(), None),
            ("  /model [name]".to_string(), "Show or change model".to_string(), None),
            ("  /browse".to_string(), "Browse & select models".to_string(), None),
            ("  /update".to_string(), "Pull latest & rebuild".to_string(), None),
            ("  /provider [name]".to_string(), "Switch provider".to_string(), None),
            ("  /tokens".to_string(), "Show token usage & cost".to_string(), None),
            ("  /summarize".to_string(), "Manually summarize".to_string(), None),
            ("  /sessions".to_string(), "List saved sessions".to_string(), None),
            ("  /resume <id>".to_string(), "Resume a session".to_string(), None),
            ("  /reload".to_string(), "Reload config from disk".to_string(), None),
        ],
    });

    pages.push(HelpPage {
        items: vec![
            ("Settings".to_string(), String::new(), Some(MAUVE)),
            (String::new(), String::new(), None),
            ("  Config file".to_string(), "~/vibe/config/vibe.json".to_string(), None),
            ("  Max tokens".to_string(), "4096 (configurable)".to_string(), None),
            ("  Temperature".to_string(), "0.0 (configurable)".to_string(), None),
            ("  Timeout".to_string(), "120s (configurable)".to_string(), None),
            ("  Permissions".to_string(), "ask/allow/deny per tool".to_string(), None),
            (String::new(), String::new(), None),
            ("Keybinds".to_string(), String::new(), Some(MAUVE)),
            (String::new(), String::new(), None),
            ("  Enter".to_string(), "Submit message".to_string(), None),
            ("  Ctrl+C".to_string(), "Exit / abort stream".to_string(), None),
            ("  Tab / F1".to_string(), "Toggle help".to_string(), None),
            ("  PgUp / PgDn".to_string(), "Scroll chat / help".to_string(), None),
            ("  Up / Down".to_string(), "Scroll messages".to_string(), None),
            ("  Left / Right".to_string(), "Move cursor".to_string(), None),
            ("  Home / End".to_string(), "Jump to start/end".to_string(), None),
            ("  Delete".to_string(), "Delete char forward".to_string(), None),
        ],
    });

    pages
}

pub(super) fn draw_help(frame: &mut Frame, app: &TuiApp) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(area);

    let mut text = Text::default();
    for page in &app.help_content {
        for (line_text, desc, color) in &page.items {
            let style = match color {
                Some(c) => Style::default().fg(*c).add_modifier(Modifier::BOLD),
                None => Style::default().fg(ratatui::style::Color::Reset),
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
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BLUE));
    let help = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((app.help_scroll as u16, 0));
    frame.render_widget(help, chunks[0]);

    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" ↑↓:Scroll ", Style::default().fg(BLUE)),
        Span::styled(" Tab/F1/Esc:Back ", Style::default().fg(BLUE)),
    ]));
    frame.render_widget(footer, chunks[1]);
}

pub(super) fn handle_help_key(app: &mut TuiApp, key: crossterm::event::KeyEvent) {
    match key.code {
        crossterm::event::KeyCode::Esc
        | crossterm::event::KeyCode::Tab
        | crossterm::event::KeyCode::F(1) => {
            app.mode = super::AppMode::Chat;
        }
        crossterm::event::KeyCode::Up => {
            app.help_scroll_up();
        }
        crossterm::event::KeyCode::Down => {
            app.help_scroll_down();
        }
        crossterm::event::KeyCode::PageUp => {
            app.help_scroll = app.help_scroll.saturating_sub(10);
        }
        crossterm::event::KeyCode::PageDown => {
            app.help_scroll = app.help_scroll.saturating_add(10);
        }
        _ => {}
    }
}
