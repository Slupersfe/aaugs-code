use std::collections::HashMap;
use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tracing;

use super::{
    LLMError, LLMEvent, LLMProvider, Message, ToolDef, read_sse_stream,
    ContentBlock, Role,
};
use crate::config::ProviderConfig;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool>,
    stream: bool,
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunction,
}

#[derive(Serialize)]
struct OpenAIFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIToolFunction,
}

#[derive(Serialize)]
struct OpenAIToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

impl OpenAIProvider {
    pub fn new(cfg: &ProviderConfig) -> Self {
        Self {
            client: Client::new(),
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            base_url: cfg.base_url.clone().unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
        }
    }

    fn base_url(&self) -> String {
        self.base_url.trim_end_matches('/').to_string()
    }

    fn convert_messages(&self, messages: &[Message]) -> Vec<OpenAIMessage> {
        let mut api_messages = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    let text = msg.content.iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text { text } = b { Some(text.clone()) }
                            else { None }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    api_messages.push(OpenAIMessage {
                        role: "system".to_string(),
                        content: Some(json!(text)),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                Role::User => {
                    let text = msg.content.iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text { text } = b { Some(text.clone()) }
                            else { None }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    api_messages.push(OpenAIMessage {
                        role: "user".to_string(),
                        content: Some(json!(text)),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                Role::Assistant => {
                    let text = msg.content.iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text { text } = b { Some(text.clone()) }
                            else { None }
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    let tool_calls: Vec<OpenAIToolCall> = msg.content.iter()
                        .filter_map(|b| {
                            if let ContentBlock::ToolUse { id, name, input } = b {
                                Some(OpenAIToolCall {
                                    id: id.clone(),
                                    call_type: "function".to_string(),
                                    function: OpenAIFunction {
                                        name: name.clone(),
                                        arguments: input.to_string(),
                                    },
                                })
                            } else { None }
                        })
                        .collect();

                    let content = if text.is_empty() { None }
                        else { Some(json!(text)) };

                    api_messages.push(OpenAIMessage {
                        role: "assistant".to_string(),
                        content,
                        tool_calls: if tool_calls.is_empty() { None }
                            else { Some(tool_calls) },
                        tool_call_id: None,
                    });
                }
                Role::Tool => {
                    for block in &msg.content {
                        if let ContentBlock::ToolResult { tool_use_id, content } = block {
                            api_messages.push(OpenAIMessage {
                                role: "tool".to_string(),
                                content: Some(json!(content)),
                                tool_calls: None,
                                tool_call_id: Some(tool_use_id.clone()),
                            });
                        }
                    }
                }
            }
        }

        api_messages
    }

    fn convert_tools(&self, tools: &[ToolDef]) -> Vec<OpenAITool> {
        tools.iter().map(|t| OpenAITool {
            tool_type: "function".to_string(),
            function: OpenAIToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        }).collect()
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn default_model(&self) -> &str {
        &self.model
    }

    fn set_model(&mut self, model: &str) {
        self.model = model.to_string();
    }

    async fn stream_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMEvent, LLMError>> + Send>>, LLMError> {
        let api_messages = self.convert_messages(messages);
        let api_tools = self.convert_tools(tools);

        let body = OpenAIRequest {
            model: self.model.clone(),
            messages: api_messages,
            tools: api_tools,
            stream: true,
        };

        let url = format!("{}/chat/completions", self.base_url());
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LLMError::Http { status, body });
        }

        use tokio::sync::mpsc;

        let (tx, rx) = mpsc::unbounded_channel::<Result<LLMEvent, LLMError>>();
        let mut sse_stream = read_sse_stream(response).await?;

        tokio::spawn(async move {
            struct PendingToolCall {
                id: Option<String>,
                name: Option<String>,
                args: String,
            }

            let mut text_buf = String::new();
            let mut tool_calls: HashMap<usize, PendingToolCall> = HashMap::new();

            while let Some(event_result) = sse_stream.next().await {
                match event_result {
                    Ok(sse) => {
                        let parsed: Value = match serde_json::from_str(&sse.data) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };

                        // Extract usage from top-level field (common in final chunk)
                        let mut usage_event: Option<LLMEvent> = None;
                        if let Some(usage) = parsed.get("usage") {
                            let prompt = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            let completion = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            let total = usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            let cost = parsed.get("cost")
                                .and_then(|v| v.as_str())
                                .and_then(|s| s.parse::<f64>().ok())
                                .or_else(|| parsed.get("cost").and_then(|v| v.as_f64()))
                                .unwrap_or(0.0);
                            if total > 0 {
                                usage_event = Some(LLMEvent::Usage(crate::llm::Usage {
                                    prompt_tokens: prompt,
                                    completion_tokens: completion,
                                    total_tokens: total,
                                    cost,
                                }));
                            }
                        }

                        let choices = match parsed.get("choices").and_then(|c| c.as_array()) {
                            Some(c) => c,
                            None => continue,
                        };

                        for choice in choices {
                            let delta = match choice.get("delta") {
                                Some(d) => d,
                                None => continue,
                            };

                            // Text content
                            if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
                                text_buf.push_str(text);
                            }

                            // Tool calls
                            if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                                for tc in tcs {
                                    let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                                    let pending = tool_calls.entry(index).or_insert_with(|| PendingToolCall {
                                        id: None,
                                        name: None,
                                        args: String::new(),
                                    });

                                    if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                                        pending.id = Some(id.to_string());
                                    }

                                    if let Some(func) = tc.get("function") {
                                        if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                                            pending.name = Some(name.to_string());
                                        }
                                        if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                                            pending.args.push_str(args);
                                        }
                                    }
                                }
                            }

                            // Finish reason
                            if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
                                if !reason.is_empty() && reason != "null" {
                                    // Flush text
                                    if !text_buf.is_empty() {
                                        let _ = tx.send(Ok(LLMEvent::Text(
                                            std::mem::take(&mut text_buf),
                                        )));
                                    }

                                    // Emit tool calls
                                    let mut indices: Vec<_> = tool_calls.keys().copied().collect();
                                    indices.sort();
                                    for idx in indices {
                                        if let Some(pending) = tool_calls.remove(&idx) {
                                            if let (Some(id), Some(name)) = (pending.id, pending.name) {
                                                let args: Value = serde_json::from_str(&pending.args)
                                                    .unwrap_or_else(|e| {
                                                        tracing::warn!("failed to parse tool call args: {}", e);
                                                        json!({})
                                                    });
                                                let _ = tx.send(Ok(LLMEvent::ToolCall {
                                                    id,
                                                    name,
                                                    args,
                                                }));
                                            }
                                        }
                                    }

                                    let _ = tx.send(Ok(LLMEvent::Stop {
                                        finish_reason: reason.to_string(),
                                    }));
                                    if let Some(usage) = usage_event.take() {
                                        let _ = tx.send(Ok(usage));
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }

            // Flush any remaining text
            if !text_buf.is_empty() {
                let _ = tx.send(Ok(LLMEvent::Text(text_buf)));
            }
        });

        let rx_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        Ok(Box::pin(rx_stream))
    }
}
