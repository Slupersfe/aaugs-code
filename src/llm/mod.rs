pub mod message;
mod anthropic;
mod openai;
mod gemini;

use std::pin::Pin;

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

pub fn resolve_provider(config: &crate::config::Config) -> Result<Box<dyn LLMProvider>, LLMError> {
    let provider_cfg = config.provider_config()
        .ok_or_else(|| LLMError::Config(format!("provider '{}' is not configured", config.provider)))?;

    let api_format = config.resolved_api_format();

    match config.provider.as_str() {
        "anthropic" => Ok(Box::new(anthropic::AnthropicProvider::new(provider_cfg))),
        "openai" => Ok(Box::new(openai::OpenAIProvider::new(provider_cfg))),
        "gemini" => Ok(Box::new(gemini::GeminiProvider::new(provider_cfg))),
        "opencode" => Ok(Box::new(openai::OpenAIProvider::new(provider_cfg))),
        "openrouter" => match api_format {
            "anthropic" => Ok(Box::new(anthropic::AnthropicProvider::new(provider_cfg))),
            _ => Ok(Box::new(openai::OpenAIProvider::new(provider_cfg))),
        },
        _ => Err(LLMError::Config(format!("unknown provider '{}'", config.provider))),
    }
}
