use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};

use super::{
    LLMError, LLMEvent, LLMProvider, Message, ToolDef, read_sse_stream,
    ContentBlock, Role,
};
use crate::config::ProviderConfig;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
    temperature: f32,
    provider_name: String,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<serde_json::Value>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: Value,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

impl AnthropicProvider {
    pub fn new(cfg: &ProviderConfig, max_tokens: u32, temperature: f32, provider_name: &str) -> Self {
        Self {
            client: Client::new(),
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            base_url: cfg.base_url.clone().unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            max_tokens,
            temperature,
            provider_name: provider_name.to_string(),
        }
    }

    fn base_url(&self) -> String {
        self.base_url.trim_end_matches('/').to_string()
    }

    fn convert_messages(&self, messages: &[Message]) -> (Option<serde_json::Value>, Vec<AnthropicMessage>) {
        let mut system: Option<serde_json::Value> = None;
        let mut api_messages = Vec::new();
        let total_msgs = messages.len();

        for (idx, msg) in messages.iter().enumerate() {
            match msg.role {
                Role::System => {
                    let text = msg.content.iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text { text } = b { Some(text.clone()) }
                            else { None }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text.is_empty() {
                        system = Some(json!([{
                            "type": "text",
                            "text": text,
                            "cache_control": { "type": "ephemeral" }
                        }]));
                    }
                }
                Role::User => {
                    let mut content = if msg.content.len() == 1 {
                        if let Some(ContentBlock::Text { text }) = msg.content.first() {
                            json!([{ "type": "text", "text": text }])
                        } else {
                            self.convert_content_blocks(&msg.content)
                        }
                    } else {
                        self.convert_content_blocks(&msg.content)
                    };
                    // Cache the first user message for prompt caching on subsequent turns
                    if idx == 0 && total_msgs > 1 {
                        if let Value::Array(ref mut blocks) = content {
                            if let Some(first) = blocks.first_mut() {
                                if let Value::Object(ref mut map) = first {
                                    map.insert("cache_control".into(), json!({ "type": "ephemeral" }));
                                }
                            }
                        }
                    }
                    api_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content,
                    });
                }
                Role::Assistant => {
                    api_messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: self.convert_content_blocks(&msg.content),
                    });
                }
                Role::Tool => {
                    let blocks: Vec<Value> = msg.content.iter().map(|block| {
                        match block {
                            ContentBlock::ToolResult { tool_use_id, content } => {
                                json!({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": content
                                })
                            }
                            _ => json!({}),
                        }
                    }).collect();
                    api_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: if blocks.len() == 1 { blocks[0].clone() }
                                 else { Value::Array(blocks) },
                    });
                }
            }
        }

        (system, api_messages)
    }

    fn convert_content_blocks(&self, blocks: &[ContentBlock]) -> Value {
        let items: Vec<Value> = blocks.iter().map(|block| match block {
            ContentBlock::Text { text } => {
                json!({"type": "text", "text": text})
            }
            ContentBlock::ToolUse { id, name, input } => {
                json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input
                })
            }
            ContentBlock::ToolResult { tool_use_id, content } => {
                json!({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content
                })
            }
        }).collect();
        Value::Array(items)
    }

    fn convert_tools(&self, tools: &[ToolDef]) -> Vec<AnthropicTool> {
        tools.iter().map(|t| AnthropicTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        }).collect()
    }

}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.provider_name
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
        let (system, api_messages) = self.convert_messages(messages);
        let api_tools = self.convert_tools(tools);

        let body = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            system,
            messages: api_messages,
            tools: api_tools,
            temperature: Some(self.temperature),
            stream: true,
        };

        let url = format!("{}/messages", self.base_url());
        let response = self.client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LLMError::Http { status, body });
        }

        // For streaming, we need a more sophisticated approach due to
        // Anthropic's content_block events requiring state tracking.
        // We'll use a simple stateful channel-based approach.
        use tokio::sync::mpsc;

        let (tx, rx) = mpsc::unbounded_channel::<Result<LLMEvent, LLMError>>();
        let mut sse_stream = read_sse_stream(response).await?;

        tokio::spawn(async move {
            let mut text_buf = String::new();
            let mut tool_id: Option<String> = None;
            let mut tool_name: Option<String> = None;
            let mut tool_json_buf = String::new();
            let mut in_tool_block = false;

            while let Some(event_result) = sse_stream.next().await {
                match event_result {
                    Ok(sse) => {
                        let data = &sse.data;
                        let parsed: Value = match serde_json::from_str(data) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };

                        let ev_type = sse.event.as_deref()
                            .or_else(|| parsed.get("type").and_then(|t| t.as_str()))
                            .unwrap_or("");

                        match ev_type {
                            "content_block_start" => {
                                // Flush any accumulated text
                                if !text_buf.is_empty() {
                                    let _ = tx.send(Ok(LLMEvent::Text(
                                        std::mem::take(&mut text_buf)
                                    )));
                                }

                                if let Some(block) = parsed.get("content_block") {
                                    match block.get("type").and_then(|t| t.as_str()) {
                                        Some("text") => {}
                                        Some("tool_use") => {
                                            in_tool_block = true;
                                            tool_id = block.get("id").and_then(|v| v.as_str()).map(String::from);
                                            tool_name = block.get("name").and_then(|v| v.as_str()).map(String::from);
                                            tool_json_buf.clear();
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_delta" => {
                                if let Some(delta) = parsed.get("delta") {
                                    match delta.get("type").and_then(|t| t.as_str()) {
                                        Some("text_delta") => {
                                            if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                                text_buf.push_str(text);
                                            }
                                        }
                                        Some("input_json_delta")
                                            if in_tool_block => {
                                                if let Some(partial) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                                    tool_json_buf.push_str(partial);
                                                }
                                            }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_stop"
                                if in_tool_block => {
                                    in_tool_block = false;
                                    let args: Value = serde_json::from_str(&tool_json_buf)
                                        .unwrap_or_else(|e| {
                                            tracing::warn!("failed to parse tool call args: {}", e);
                                            json!({})
                                        });
                                    if let (Some(id), Some(name)) = (tool_id.take(), tool_name.take()) {
                                        let _ = tx.send(Ok(LLMEvent::ToolCall {
                                            id,
                                            name,
                                            args,
                                        }));
                                    }
                                }
                            "message_delta" => {
                                // Flush remaining text
                                if !text_buf.is_empty() {
                                    let _ = tx.send(Ok(LLMEvent::Text(
                                        std::mem::take(&mut text_buf)
                                    )));
                                }
                                if let Some(delta) = parsed.get("delta") {
                                    if let Some(reason) = delta.get("stop_reason").and_then(|r| r.as_str()) {
                                        let _ = tx.send(Ok(LLMEvent::Stop {
                                            finish_reason: reason.to_string(),
                                        }));
                                    }
                                }
                                // Extract usage
                                if let Some(usage) = parsed.get("usage") {
                                    let prompt = usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                    let completion = usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                    let total = prompt + completion;
                                    if total > 0 {
                                        let _ = tx.send(Ok(LLMEvent::Usage(crate::llm::Usage {
                                            prompt_tokens: prompt,
                                            completion_tokens: completion,
                                            total_tokens: total,
                                            cost: 0.0,
                                        })));
                                    }
                                }
                            }
                            "error" => {
                                let err_val = parsed.get("error");
                                let msg = err_val
                                    .and_then(|e| e.get("message").and_then(|v| v.as_str()))
                                    .or_else(|| err_val.and_then(|e| e.as_str().map(|s| s)));
                                if let Some(msg) = msg {
                                    let _ = tx.send(Err(LLMError::Http {
                                        status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                                        body: msg.to_string(),
                                    }));
                                }
                            }
                            _ => {}
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
        });

        let rx_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        Ok(Box::pin(rx_stream))
    }
}
