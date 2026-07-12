use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use ratatui::Frame;

use super::palette::{BLUE, BORDER, GREEN, SUBTLE, YELLOW};
use super::TuiApp;
use crate::cost;

pub(super) fn draw_browse(frame: &mut Frame, app: &TuiApp) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(area);

    let mut text = Text::default();
    text.push_line(Line::from(Span::styled(
        format!(" Provider: {}  |  Current model: {}", app.provider, app.model),
        Style::default().fg(BLUE).add_modifier(Modifier::BOLD),
    )));
    text.push_line(Line::from(""));

    let preferred_count = app.browse_preferred_count;
    let all_count = app.browse_models.len();

    for (i, model_name) in app.browse_models.iter().enumerate() {
        if i == 0 && preferred_count > 0 {
            text.push_line(Line::from(Span::styled(
                " ★ Preferred",
                Style::default().fg(GREEN).add_modifier(Modifier::BOLD),
            )));
            text.push_line(Line::from(""));
        }
        if i == preferred_count && preferred_count > 0 && i < all_count {
            text.push_line(Line::from(Span::styled(
                " ── All Models ──",
                Style::default().fg(SUBTLE),
            )));
            text.push_line(Line::from(""));
        }

        let cursor = i == app.browse_cursor;
        let prefix = if cursor { " ▌" } else { "  " };
        let (inp, outp) = cost::model_cost(model_name);
        let line = format!(
            "{}{}  ${:.2}/{}k in, ${:.2}/{}k out",
            prefix,
            model_name,
            inp,
            (inp * 4.0) as u32,
            outp,
            (outp * 4.0) as u32
        );
        let mut style = if cursor {
            Style::default().fg(YELLOW).add_modifier(Modifier::BOLD)
        } else if i < preferred_count {
            Style::default().fg(GREEN)
        } else {
            Style::default().fg(ratatui::style::Color::Reset)
        };
        if cursor {
            style = style.bg(BORDER);
        }
        text.push_line(Line::from(Span::styled(line, style)));
    }

    let block = Block::default()
        .title(" Model Browser ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(YELLOW));
    let browse = Paragraph::new(text)
        .block(block)
        .scroll((app.browse_cursor.saturating_sub(5) as u16, 0));
    frame.render_widget(browse, chunks[0]);

    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" ↑↓:Navigate ", Style::default().fg(BLUE)),
        Span::styled(" Enter:Select ", Style::default().fg(BLUE)),
        Span::styled(" Esc:Back ", Style::default().fg(BLUE)),
    ]));
    frame.render_widget(footer, chunks[1]);
}

pub(super) fn handle_browse_key(
    app: &mut TuiApp,
    key: crossterm::event::KeyEvent,
    input_tx: &std::sync::mpsc::Sender<String>,
) {
    match key.code {
        crossterm::event::KeyCode::Esc | crossterm::event::KeyCode::Tab | crossterm::event::KeyCode::F(1) => {
            app.mode = super::AppMode::Chat;
        }
        crossterm::event::KeyCode::Up if app.browse_cursor > 0 => {
            app.browse_cursor -= 1;
        }
        crossterm::event::KeyCode::Down if app.browse_cursor + 1 < app.browse_models.len() => {
            app.browse_cursor += 1;
        }
        crossterm::event::KeyCode::PageUp => {
            app.browse_cursor = app.browse_cursor.saturating_sub(10);
        }
        crossterm::event::KeyCode::PageDown => {
            app.browse_cursor = app.browse_cursor
                .saturating_add(10)
                .min(app.browse_models.len().saturating_sub(1));
        }
        crossterm::event::KeyCode::Enter => {
            if let Some(model) = app.browse_models.get(app.browse_cursor) {
                let cmd = format!("/model {}", model);
                let _ = input_tx.send(cmd);
            }
            app.mode = super::AppMode::Chat;
        }
        crossterm::event::KeyCode::Char(c) => {
            app.browse_filter.push(c);
            let filter = app.browse_filter.to_lowercase();
            if let Some(pos) = app
                .browse_models
                .iter()
                .position(|m| m.to_lowercase().contains(&filter))
            {
                app.browse_cursor = pos;
            }
        }
        crossterm::event::KeyCode::Backspace => {
            app.browse_filter.pop();
        }
        _ => {}
    }
}
