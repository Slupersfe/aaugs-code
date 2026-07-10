pub mod message;
mod anthropic;
mod openai;
mod gemini;

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::Value;

pub use message::*;

#[derive(Debug)]
pub enum LLMError {
    Http { status: reqwest::StatusCode, body: String },
    Network(reqwest::Error),
    Serde(serde_json::Error),
    Stream(String),
    Config(String),
}

impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LLMError::Http { status, body } => {
                let msg = extract_error_message(body);
                write!(f, "HTTP {}: {}", status, msg)
            }
            LLMError::Network(e) => write!(f, "Network error: {}", e),
            LLMError::Serde(e) => write!(f, "Serialization error: {}", e),
            LLMError::Stream(e) => write!(f, "Stream error: {}", e),
            LLMError::Config(e) => write!(f, "Configuration error: {}", e),
        }
    }
}

impl std::error::Error for LLMError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LLMError::Network(e) => Some(e),
            LLMError::Serde(e) => Some(e),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for LLMError {
    fn from(e: reqwest::Error) -> Self {
        LLMError::Network(e)
    }
}

impl From<serde_json::Error> for LLMError {
    fn from(e: serde_json::Error) -> Self {
        LLMError::Serde(e)
    }
}



fn extract_error_message(body: &str) -> String {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(body) {
        // Common error formats: { "error": { "message": "..." } }
        if let Some(msg) = val.get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return msg.to_string();
        }
        // { "error": "..." }
        if let Some(msg) = val.get("error").and_then(|e| e.as_str()) {
            return msg.to_string();
        }
    }
    // If it looks like HTML, return a generic message
    if body.trim_start().starts_with('<') {
        return "server returned HTML (expected JSON API response)".to_string();
    }
    // Truncate very long responses
    let max_len = 200;
    if body.len() > max_len {
        return body[..max_len].to_string();
    }
    body.to_string()
}

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[allow(dead_code)]
    pub total_tokens: u32,
    pub cost: f64,
}

#[derive(Debug, Clone)]
pub enum LLMEvent {
    Text(String),
    ToolCall { id: String, name: String, args: Value },
    Stop { finish_reason: String },
    Usage(Usage),
}

#[async_trait]
pub trait LLMProvider: Send + Sync {
    fn name(&self) -> &str;
    fn default_model(&self) -> &str;
    fn set_model(&mut self, model: &str);
    async fn stream_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMEvent, LLMError>> + Send>>, LLMError>;
}

pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

pub async fn read_sse_stream(
    response: reqwest::Response,
) -> Result<Pin<Box<dyn Stream<Item = Result<SseEvent, LLMError>> + Send>>, LLMError> {
    use tokio::sync::mpsc;

    let (tx, rx) = mpsc::unbounded_channel::<Result<SseEvent, LLMError>>();
    let mut byte_stream = response.bytes_stream();

    tokio::spawn(async move {
        let mut current_event: Option<String> = None;
        let mut current_data = String::new();
        let mut buf = String::new();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Err(LLMError::Network(e)));
                    return;
                }
            };

            buf.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines from the buffer
            loop {
                let newline_pos = match buf.find('\n') {
                    Some(pos) => pos,
                    None => break,
                };

                let line = buf[..newline_pos].to_string();
                buf = buf[newline_pos + 1..].to_string();
                let trimmed = line.trim_end();

                if trimmed.is_empty() {
                    if !current_data.is_empty() {
                        let event = SseEvent {
                            event: current_event.take(),
                            data: std::mem::take(&mut current_data),
                        };
                        let _ = tx.send(Ok(event));
                    }
                    continue;
                }

                if let Some(data) = trimmed.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        let _ = tx.send(Err(LLMError::Stream("done".to_string())));
                        return;
                    }
                    current_data.push_str(data);
                } else if let Some(event) = trimmed.strip_prefix("event: ") {
                    current_event = Some(event.to_string());
                }
            }
        }

        // Flush remaining data
        if !current_data.is_empty() {
            let event = SseEvent {
                event: current_event.take(),
                data: std::mem::take(&mut current_data),
            };
            let _ = tx.send(Ok(event));
        }
    });

    let rx_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

    // Filter out the Stream error we used for DONE signal
    let filtered = rx_stream.filter_map(|result| {
        futures::future::ready(match result {
            Ok(event) => Some(Ok(event)),
            Err(LLMError::Stream(ref msg)) if msg == "done" => None,
            Err(e) => Some(Err(e)),
        })
    });

    Ok(Box::pin(filtered))
}

/// Retries an async fallible operation with exponential backoff.
/// Retries on HTTP 429 (rate limit) and 5xx (server errors) up to `max_retries` times.
/// Non-retryable errors are returned immediately.
pub async fn retry_with_backoff<F, Fut, T>(f: F, max_retries: u32) -> Result<T, LLMError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, LLMError>>,
{
    for attempt in 0..=max_retries {
        match f().await {
            Ok(val) => return Ok(val),
            Err(LLMError::Http { status, body }) if status.as_u16() == 429 || status.as_u16() >= 500 => {
                if attempt == max_retries {
                    return Err(LLMError::Http { status: status.clone(), body: body.clone() });
                }
                let wait_ms = 1000u64 * 2u64.pow(attempt);
                tokio::time::sleep(Duration::from_millis(wait_ms.min(30_000))).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}

pub fn resolve_provider(config: &crate::config::Config) -> Result<Box<dyn LLMProvider>, LLMError> {
    let provider_cfg = config.provider_config()
        .ok_or_else(|| LLMError::Config(format!("provider '{}' is not configured", config.provider)))?;

    let api_format = config.resolved_api_format();
    let max_tokens = config.advanced.max_tokens;
    let temperature = config.advanced.temperature;

    let pname = &config.provider;

    let new_anthropic = || anthropic::AnthropicProvider::new(provider_cfg, max_tokens, temperature, pname);
    let new_openai = || openai::OpenAIProvider::new(provider_cfg, max_tokens, temperature, pname);
    let new_gemini = || gemini::GeminiProvider::new(provider_cfg, max_tokens, temperature, pname);

    match pname.as_str() {
        "anthropic" => Ok(Box::new(new_anthropic())),
        "openai" => Ok(Box::new(new_openai())),
        "gemini" => Ok(Box::new(new_gemini())),
        "opencode" => {
            let dev_path = dirs::home_dir()
                .map(|h| h.join("vibe").join("dev.config"));
            let dev_enabled = dev_path
                .as_ref()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .is_some_and(|s| s.trim() == "TRUE");
            if !dev_enabled {
                return Err(LLMError::Config(
                    "opencode provider requires ~/vibe/dev.config with content 'TRUE'".into()
                ));
            }
            Ok(Box::new(new_openai()))
        }
        "openrouter" => match api_format {
            "anthropic" => Ok(Box::new(new_anthropic())),
            _           => Ok(Box::new(new_openai())),
        },
        "custom" => match api_format {
            "anthropic" => Ok(Box::new(new_anthropic())),
            "gemini" | "google" => Ok(Box::new(new_gemini())),
            _ => Ok(Box::new(new_openai())),
        },
        _ => Err(LLMError::Config(format!("unknown provider '{}'", pname))),
    }
}
