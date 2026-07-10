use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use super::palette::{
    BLUE, BORDER, GREEN, MAUVE, OVERLAY, RED, SUBTLE, SURFACE, TEXT, YELLOW,
};
use super::TuiApp;

pub(super) fn draw_chat(frame: &mut Frame, app: &mut TuiApp) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

    // ── Status bar ─────────────────────────────────────────────────────
    draw_status_bar(frame, app, chunks[0]);

    // ── Messages ───────────────────────────────────────────────────────
    draw_messages(frame, app, chunks[1]);

    // ── Input area ─────────────────────────────────────────────────────
    draw_input(frame, app, chunks[2]);

    // ── Keybinds bar ──────────────────────────────────────────────────
    draw_keybinds(frame, chunks[3]);

    // ── Question overlay ──────────────────────────────────────────────
    if app.pending_question.is_some() {
        draw_question_overlay(frame, app, area);
    }
}

fn draw_status_bar(frame: &mut Frame, app: &TuiApp, area: Rect) {
    let update = app
        .update_available
        .map(|v| format!(" UP v{}! /update ", v))
        .unwrap_or_default();
    let model = if app.auto_route && crate::router::is_loaded() {
        format!("{} [Auto]", app.model)
    } else {
        app.model.clone()
    };
    let cost = if app.total_cost > 0.0001 {
        format!("${:.4}", app.total_cost)
    } else if app.total_cost > 0.0 {
        "<$0.0001".to_string()
    } else {
        String::new()
    };

    let mut spans = vec![
        Span::styled(
            " aaugs-code ",
            Style::default().fg(TEXT).bg(MAUVE).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" ▌", Style::default().fg(BORDER)),
        Span::styled(
            format!(" {} ", app.provider),
            Style::default().fg(TEXT).bg(SURFACE),
        ),
    ];
    if !update.is_empty() {
        spans.push(Span::styled(" ▌", Style::default().fg(BORDER)));
        spans.push(Span::styled(
            format!(" {} ", update.trim()),
            Style::default().fg(YELLOW).bg(OVERLAY).add_modifier(Modifier::BOLD),
        ));
    }
    spans.push(Span::styled(" ▌", Style::default().fg(BORDER)));
    spans.push(Span::styled(
        format!(" {} ", model),
        Style::default().fg(BLUE).bg(SURFACE),
    ));
    spans.push(Span::styled(" ▌", Style::default().fg(BORDER)));
    spans.push(Span::styled(
        format!(" {} in / {} out ", app.input_tokens, app.output_tokens),
        Style::default().fg(SUBTLE).bg(SURFACE),
    ));
    if !cost.is_empty() {
        spans.push(Span::styled(" ▌", Style::default().fg(BORDER)));
        spans.push(Span::styled(
            format!(" {} ", cost),
            Style::default().fg(GREEN).bg(SURFACE),
        ));
    }
    spans.push(Span::styled(
        " ".repeat(area.width.saturating_sub(
            spans.iter().map(|s| s.content.len() as u16).sum::<u16>(),
        ) as usize),
        Style::default().bg(SURFACE),
    ));

    let bar = Paragraph::new(Line::from(spans));
    frame.render_widget(bar, area);
}

fn draw_messages(frame: &mut Frame, app: &mut TuiApp, area: Rect) {
    let mut text = Text::default();
    let mut in_code = app.in_code_block;

    // Build message lines
    for msg in &app.messages {
        let (gutter_color, role_fg) = match msg.role.as_str() {
            "User" => (GREEN, GREEN),
            "Assistant" => (BLUE, TEXT),
            "Tool" => (YELLOW, SUBTLE),
            "Error" => (RED, RED),
            "System" => (SUBTLE, SUBTLE),
            _ => (SUBTLE, TEXT),
        };
        let role_style = Style::default().fg(gutter_color);
        let gutter = Span::styled("▐", role_style);

        // Add the role line
        let role_line = Line::from(vec![
            gutter,
            Span::styled(
                format!(" {} ", msg.role),
                role_style.add_modifier(Modifier::BOLD),
            ),
        ]);
        text.push_line(role_line);

        // Message content
        let base = Style::default().fg(role_fg);
        for line in msg.text.lines() {
            let indent = Span::styled("  ", base);
            // We need to incorporate the indent into the line
            // Build the line with indent + parsed content
            // Use a temporary text for parsing, then prepend indent
            let mut line_text = Text::default();
            super::markdown::push_md_line(&mut line_text, line, base, &mut in_code);
            for text_line in line_text.lines.iter() {
                let mut spans = vec![indent.clone()];
                spans.extend(text_line.spans.iter().cloned());
                text.push_line(Line::from(spans));
            }
        }
    }
    app.in_code_block = in_code;

    let msg_area_inner = Rect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: area.height.saturating_sub(1),
    };

    if app.auto_scroll {
        let h = msg_area_inner.height as usize;
        let total: usize = app.messages.iter().map(|m| m.text.lines().count()).sum();
        // Add role lines
        let total = total + app.messages.len();
        app.scroll = total.saturating_sub(h);
    }

    let msg_widget = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(BORDER)),
        )
        .wrap(Wrap { trim: false })
        .scroll((app.scroll as u16, 0));
    frame.render_widget(msg_widget, msg_area_inner);
}

fn draw_input(frame: &mut Frame, app: &TuiApp, area: Rect) {
    let (fg, prompt) = if app.is_loading {
        (SUBTLE, " ◌ ".to_string())
    } else if app.pending_question.is_some() {
        (YELLOW, " ❓ ".to_string())
    } else if app.input.is_empty() {
        (SUBTLE, " ❯ ".to_string())
    } else {
        (TEXT, " ❯ ".to_string())
    };
    let display = if app.is_loading {
        "waiting for response...".to_string()
    } else if app.pending_question.is_some() {
        if app.input.is_empty() {
            "type answer...".to_string()
        } else {
            app.input.clone()
        }
    } else if app.input.is_empty() {
        "type a message...".to_string()
    } else {
        app.input.clone()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(BORDER));
    let inner = block.inner(area);
    frame.render_widget(Clear, area);
    frame.render_widget(block, area);

    let spans = vec![
        Span::styled(prompt, Style::default().fg(MAUVE).add_modifier(Modifier::BOLD)),
        Span::styled(display, Style::default().fg(fg)),
    ];
    let input_widget = Paragraph::new(Line::from(spans));
    frame.render_widget(input_widget, inner);
}

fn draw_keybinds(frame: &mut Frame, area: Rect) {
    let keys = Paragraph::new(Line::from(vec![
        Span::styled(" Tab/F1 Help ", Style::default().fg(BLUE)),
        Span::styled("│", Style::default().fg(BORDER)),
        Span::styled(" ↑/↓ Scroll ", Style::default().fg(BLUE)),
        Span::styled("│", Style::default().fg(BORDER)),
        Span::styled(" Enter Send ", Style::default().fg(BLUE)),
        Span::styled("│", Style::default().fg(BORDER)),
        Span::styled(" Ctrl+C Quit", Style::default().fg(BLUE)),
    ]));
    frame.render_widget(keys, area);
}

fn draw_question_overlay(frame: &mut Frame, app: &TuiApp, area: Rect) {
    if let Some((question, options, _)) = &app.pending_question {
        let q_area = Rect {
            x: area.width / 6,
            y: area.height / 3,
            width: area.width * 2 / 3,
            height: 6,
        };
        frame.render_widget(Clear, q_area);

        let opts = if options.is_empty() {
            String::new()
        } else {
            format!(" [{}]", options.join(", "))
        };
        let answer = if app.input.is_empty() {
            "type answer...".to_string()
        } else {
            app.input.clone()
        };

        let mut spans = vec![Span::styled(
            " ❓ ",
            Style::default().fg(YELLOW).add_modifier(Modifier::BOLD),
        )];
        spans.push(Span::styled(
            format!("{}{}", question, opts),
            Style::default().fg(TEXT),
        ));
        let content = Line::from(spans);

        let mut input_spans = vec![Span::styled(
            "  ❯ ",
            Style::default().fg(MAUVE).add_modifier(Modifier::BOLD),
        )];
        input_spans.push(Span::styled(answer, Style::default().fg(YELLOW)));
        let input_line = Line::from(input_spans);

        let text = Text::from(vec![content, Line::from(""), input_line]);

        let block = Block::default()
            .title(" Permission Required ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(YELLOW))
            .border_type(BorderType::Plain);
        let prompt = Paragraph::new(text).block(block);
        frame.render_widget(prompt, q_area);
    }
}
