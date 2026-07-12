use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use termimad::MadSkin;

/// RAII guard that restores terminal raw mode on drop or panic.
pub struct RawModeGuard;

impl RawModeGuard {
    pub fn new() -> std::io::Result<Self> {
        crossterm::terminal::enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        if let Err(e) = crossterm::terminal::disable_raw_mode() {
            tracing::warn!("failed to disable raw mode: {e}");
        }
        if let Err(e) = crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen) {
            tracing::warn!("failed to leave alternate screen: {e}");
        }
    }
}

static SKIN: std::sync::LazyLock<MadSkin> = std::sync::LazyLock::new(|| {
    use termimad::crossterm::style::Color;
    let mut skin = MadSkin::default();
    skin.set_headers_fg(Color::Rgb { r: 230, g: 120, b: 120 });
    skin.bold.set_fg(Color::Rgb { r: 240, g: 200, b: 140 });
    skin.italic.set_fg(Color::Rgb { r: 200, g: 130, b: 150 });
    skin.code_block.set_fg(Color::Rgb { r: 210, g: 160, b: 120 });
    skin.code_block.set_bg(Color::Rgb { r: 16, g: 8, b: 10 });
    skin
});

pub fn render_markdown(text: &str) {
    SKIN.print_text(text);
}

pub fn print_tool_header(name: &str, args: &serde_json::Value) {
    let pretty = serde_json::to_string_pretty(args).unwrap_or_default();
    let dim_fg = termimad::crossterm::style::Color::Rgb { r: 235, g: 160, b: 120 };
    use termimad::crossterm::style::Stylize;
    eprintln!("{}", format!("── tool: {} ──", name).with(dim_fg));
    if !pretty.is_empty() && pretty != "null" {
        eprintln!("{}", pretty.as_str().with(dim_fg));
    }
}

pub fn print_tool_result(output: &str) {
    if output.is_empty() {
        return;
    }
    let dim_fg = termimad::crossterm::style::Color::Rgb { r: 210, g: 160, b: 120 };
    use termimad::crossterm::style::Stylize;
    let lines: Vec<&str> = output.lines().collect();
    let display = if lines.len() > 20 {
        let truncated: Vec<&str> = lines[..20].to_vec();
        truncated.join("\n") + &format!("\n... ({} more lines)", lines.len() - 20).with(dim_fg).to_string()
    } else {
        output.to_string()
    };
    eprintln!("{}", display.as_str().with(dim_fg));
}

pub fn print_tool_denied(name: &str) {
    let warn_fg = termimad::crossterm::style::Color::Red;
    use termimad::crossterm::style::Stylize;
    eprintln!("{}", format!("⛔ tool '{}' denied", name).with(warn_fg));
}

pub fn print_success() {
    let dim_fg = termimad::crossterm::style::Color::Rgb { r: 210, g: 160, b: 120 };
    use termimad::crossterm::style::Stylize;
    eprintln!("{}", "── done ──".with(dim_fg));
}

pub fn print_info(text: &str) {
    let dim_fg = termimad::crossterm::style::Color::Rgb { r: 155, g: 125, b: 130 };
    use termimad::crossterm::style::Stylize;
    eprintln!("{}", text.with(dim_fg));
}

pub fn clear_line() {
    use std::io::Write;
    use termimad::crossterm::style::Stylize;
    eprint!("\r{}", " ".repeat(12).as_str().with(termimad::crossterm::style::Color::Reset));
    let _ = std::io::stderr().flush();
}

// --- Spinner ---

pub fn start_spinner() -> Arc<AtomicBool> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    thread::spawn(move || {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let mut i = 0;
        while r.load(Ordering::Relaxed) {
            use std::io::Write;
            eprint!("\r{} thinking...", frames[i % frames.len()]);
            let _ = std::io::stderr().flush();
            thread::sleep(Duration::from_millis(80));
            i += 1;
        }
        // Clear the spinner line
        use std::io::Write;
        eprint!("\r                      \r");
        let _ = std::io::stderr().flush();
    });
    running
}

// --- Streaming output ---

pub struct StreamPrinter {
    line_count: usize,
}

impl StreamPrinter {
    pub fn new() -> Self {
        Self { line_count: 0 }
    }

    pub fn write(&mut self, text: &str) {
        // Count lines printed
        self.line_count += text.chars().filter(|&c| c == '\n').count();
        if text.contains('\n') || !self.text_ends_with_newline(text) {
            // If there's a trailing partial line, we've advanced past it
        }
        use std::io::Write;
        print!("{}", text);
        let _ = std::io::stdout().flush();
    }

    fn text_ends_with_newline(&self, text: &str) -> bool {
        text.ends_with('\n')
    }

    pub fn finish(&mut self, rendered_text: &str) {
        use std::io::Write;
        std::io::stdout().flush().ok();

        if self.line_count == 0 {
            // No newlines — go back to start of line, clear it, render markdown
            print!("\r\x1b[K");
            std::io::stdout().flush().ok();
            render_markdown(rendered_text);
            return;
        }

        // Move cursor up by line_count, clear from cursor to end of screen, then render markdown
        print!("\x1b[{}A\x1b[J", self.line_count);
        std::io::stdout().flush().ok();

        render_markdown(rendered_text);
    }
}
