use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};

use super::palette::{BLUE, CODE_BG, PEACH, SUBTLE, YELLOW};

/// Parse a line of text with markdown formatting and add it to `text`.
/// Handles code blocks, headers, blockquotes, lists, bold, italic, inline code.
pub fn push_md_line(text: &mut Text, line: &str, base_style: Style, in_code: &mut bool) {
    if *in_code {
        if line.trim_end() == "```" {
            *in_code = false;
            return;
        }
        let code_style = base_style.bg(CODE_BG);
        let header = line.trim_start().starts_with("//") || line.trim_start().starts_with('#')
            || line.trim_start().starts_with("/*");
        let style = if header {
            code_style.fg(SUBTLE)
        } else {
            code_style
        };
        text.push_line(Line::from(Span::styled(line.to_string(), style)));
        return;
    }
    if line.trim_start().starts_with("```") {
        *in_code = true;
        let lang = line.trim_start().trim_start_matches("```").trim();
        if !lang.is_empty() {
            text.push_line(Line::from(vec![
                Span::styled(" ▌", base_style.fg(SUBTLE)),
                Span::styled(format!(" {} ", lang), base_style.fg(SUBTLE).bg(CODE_BG)),
            ]));
        }
        return;
    }

    let trimmed = line.trim_start();
    let indent = line.len() - trimmed.len();

    // Headers
    if let Some(rest) = trimmed.strip_prefix("### ") {
        text.push_line(Line::from(Span::styled(
            " ".repeat(indent) + rest,
            base_style.fg(BLUE).add_modifier(Modifier::BOLD),
        )));
        return;
    }
    if let Some(rest) = trimmed.strip_prefix("## ") {
        text.push_line(Line::from(Span::styled(
            " ".repeat(indent) + rest,
            base_style.fg(BLUE).add_modifier(Modifier::BOLD),
        )));
        return;
    }
    if let Some(rest) = trimmed.strip_prefix("# ") {
        text.push_line(Line::from(Span::styled(
            " ".repeat(indent) + rest,
            base_style.fg(YELLOW).add_modifier(Modifier::BOLD),
        )));
        return;
    }

    // Blockquotes
    if trimmed.starts_with('>') {
        let content = trimmed.trim_start_matches('>').trim();
        text.push_line(Line::from(Span::styled(
            " ".repeat(indent) + "▎ " + content,
            base_style.fg(SUBTLE).add_modifier(Modifier::ITALIC),
        )));
        return;
    }

    // Lists
    if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
        let inner = &trimmed[2..];
        let spans = parse_inline_md(inner, base_style);
        let mut result = vec![Span::styled(" ".repeat(indent) + "• ", base_style.fg(PEACH))];
        result.extend(spans);
        text.push_line(Line::from(result));
        return;
    }
    if let Some(rest) = trimmed.strip_prefix(|c: char| c.is_ascii_digit()) {
        if rest.starts_with(". ") {
            let inner = &rest[2..];
            let spans = parse_inline_md(inner, base_style);
            let num = &trimmed[..trimmed.len() - rest.len()];
            let mut result = vec![Span::styled(
                " ".repeat(indent) + num + ".",
                base_style.fg(PEACH),
            )];
            result.push(Span::styled(" ", base_style));
            result.extend(spans);
            text.push_line(Line::from(result));
            return;
        }
    }

    // Inline markdown
    let spans = parse_inline_md(line, base_style);
    text.push_line(Line::from(spans));
}

/// Parse bold, italic, and inline code spans.
pub fn parse_inline_md(line: &str, base: Style) -> Vec<Span<'static>> {
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
                i += 2;
            }
            spans.push(Span::styled(bold, base.add_modifier(Modifier::BOLD)));
            continue;
        }
        // Italic *text* (only when not followed by another *)
        if chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] != '*' {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), base));
            }
            i += 1;
            let mut italic = String::new();
            while i < chars.len() && chars[i] != '*' {
                italic.push(chars[i]);
                i += 1;
            }
            if i < chars.len() {
                i += 1;
            }
            spans.push(Span::styled(italic, base.add_modifier(Modifier::ITALIC)));
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
                i += 1;
            }
            spans.push(Span::styled(code, base.fg(YELLOW).bg(CODE_BG)));
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
