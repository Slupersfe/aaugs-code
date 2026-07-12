use std::pin::Pin;
use std::time::Duration;

use anthropic_sdk::{
    types::{
        ContentBlockDelta, AnthropicError, ContentBlock as AnthropicContentBlock,
        ContentBlockParam, MessageContent, MessageParam, MessageStreamEvent, Role as AnthropicRole,
        StopReason, Tool, ToolInputSchema,
    },
    Anthropic, ClientConfig, MessageCreateBuilder,
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::{json, Map, Value};

use super::{ContentBlock, LLMError, LLMEvent, LLMProvider, Message, Role, ToolDef};
use crate::config::ProviderConfig;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

pub struct AnthropicProvider {
    client: Anthropic,
    model: String,
    max_tokens: u32,
    temperature: f32,
    provider_name: String,
}

fn convert_error(e: AnthropicError) -> LLMError {
    match e {
        AnthropicError::BadRequest { message, status }
        | AnthropicError::Authentication { message, status }
        | AnthropicError::PermissionDenied { message, status }
        | AnthropicError::NotFound { message, status }
        | AnthropicError::UnprocessableEntity { message, status }
        | AnthropicError::RateLimit { message, status }
        | AnthropicError::InternalServer { message, status } => {
            let code = reqwest::StatusCode::from_u16(status)
                .unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR);
            LLMError::Http { status: code, body: message }
        }
        AnthropicError::Connection { message }
        | AnthropicError::NetworkError(message)
        | AnthropicError::HttpError { message, .. }
        | AnthropicError::ServiceUnavailable { message }
        | AnthropicError::Other(message) => LLMError::Stream(message),
        AnthropicError::ConnectionTimeout => LLMError::Stream("connection timeout".into()),
        AnthropicError::UserAbort => LLMError::Stream("user aborted".into()),
        AnthropicError::StreamError(msg) => LLMError::Stream(msg),
        AnthropicError::Configuration { message } => LLMError::Config(message),
        AnthropicError::InvalidApiKey => LLMError::Config("invalid API key".into()),
        AnthropicError::Timeout => LLMError::Stream("timeout".into()),
    }
}

fn convert_messages(messages: &[Message]) -> (Option<String>, Vec<MessageParam>) {
    let mut system: Option<String> = None;
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
                    system = Some(text);
                }
            }
            Role::User => {
                let blocks: Vec<ContentBlockParam> = msg.content.iter().map(|b| {
                    match b {
                        ContentBlock::Text { text } => {
                            ContentBlockParam::Text { text: text.clone() }
                        }
                        _ => ContentBlockParam::Text { text: String::new() },
                    }
                }).collect();

                if idx == 0 && total_msgs > 1 {
                    // Cache the first user message for prompt caching
                    // (the SDK doesn't expose cache_control directly, so skip for now)
                }

                api_messages.push(MessageParam {
                    role: AnthropicRole::User,
                    content: if blocks.len() == 1 {
                        match blocks.into_iter().next().unwrap() {
                            ContentBlockParam::Text { text } => MessageContent::Text(text),
                            other => MessageContent::Blocks(vec![other]),
                        }
                    } else {
                        MessageContent::Blocks(blocks)
                    },
                });
            }
            Role::Assistant => {
                let mut blocks: Vec<ContentBlockParam> = msg.content.iter().filter_map(|b| {
                    match b {
                        ContentBlock::Text { text } => {
                            if text.is_empty() { None }
                            else { Some(ContentBlockParam::Text { text: text.clone() }) }
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            Some(ContentBlockParam::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: input.clone(),
                            })
                        }
                        _ => None,
                    }
                }).collect();

                if blocks.is_empty() {
                    blocks.push(ContentBlockParam::Text { text: String::new() });
                }

                api_messages.push(MessageParam {
                    role: AnthropicRole::Assistant,
                    content: MessageContent::Blocks(blocks),
                });
            }
            Role::Tool => {
                let blocks: Vec<ContentBlockParam> = msg.content.iter().map(|b| {
                    match b {
                        ContentBlock::ToolResult { tool_use_id, content } => {
                            ContentBlockParam::ToolResult {
                                tool_use_id: tool_use_id.clone(),
                                content: Some(content.clone()),
                                is_error: None,
                            }
                        }
                        _ => ContentBlockParam::Text { text: String::new() },
                    }
                }).collect();

                api_messages.push(MessageParam {
                    role: AnthropicRole::User,
                    content: MessageContent::Blocks(blocks),
                });
            }
        }
    }

    (system, api_messages)
}

fn convert_tools(tools: &[ToolDef]) -> Vec<Tool> {
    tools.iter().map(|t| {
        let properties: Map<String, Value> = t.input_schema.get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let required: Vec<String> = t.input_schema.get("required")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        Tool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties,
                required,
                additional: Map::new(),
            },
        }
    }).collect()
}

impl AnthropicProvider {
    pub fn new(cfg: &ProviderConfig, max_tokens: u32, temperature: f32, provider_name: &str, timeout_secs: u64) -> Result<Self, LLMError> {
        let api_key = cfg.api_key.clone();
        let base_url = cfg.base_url.clone().unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let config = ClientConfig::new(api_key)
            .with_base_url(base_url)
            .with_timeout(Duration::from_secs(timeout_secs));
        let client = Anthropic::with_config(config)
            .map_err(|e| LLMError::Config(e.to_string()))?;
        // Anthropic requires max_tokens >= 1; 0 means no limit, so use a large default
        let effective_max = if max_tokens == 0 { 8192 } else { max_tokens };
        Ok(Self {
            client,
            model: cfg.model.clone(),
            max_tokens: effective_max,
            temperature,
            provider_name: provider_name.to_string(),
        })
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

    #[tracing::instrument(skip(self, messages, tools))]
    async fn stream_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMEvent, LLMError>> + Send>>, LLMError> {
        let (system, api_messages) = convert_messages(messages);
        let api_tools = convert_tools(tools);

        let mut builder = MessageCreateBuilder::new(&self.model, self.max_tokens)
            .temperature(self.temperature)
            .stream(true);

        if let Some(system_text) = system {
            builder = builder.system(system_text);
        }

        for msg in api_messages {
            builder = builder.message(msg.role, msg.content);
        }

        if !api_tools.is_empty() {
            builder = builder.tools(api_tools);
        }

        let params = builder.build();
        let sdk_stream = self.client.messages().create_stream(params).await
            .map_err(convert_error)?;

        use tokio::sync::mpsc;

        let (tx, rx) = mpsc::unbounded_channel::<Result<LLMEvent, LLMError>>();

        tokio::spawn(async move {
            let mut text_buf = String::new();
            let mut tool_id: Option<String> = None;
            let mut tool_name: Option<String> = None;
            let mut tool_json_buf = String::new();
            let mut in_tool_block = false;

            let mut stream = sdk_stream;

            while let Some(event_result) = stream.next().await {
                let event = match event_result {
                    Ok(e) => e,
                    Err(e) => {
                        let _ = tx.send(Err(convert_error(e)));
                        break;
                    }
                };

                match event {
                    MessageStreamEvent::ContentBlockStart { content_block, .. } => {
                        if !text_buf.is_empty() {
                            let _ = tx.send(Ok(LLMEvent::Text(
                                std::mem::take(&mut text_buf),
                            )));
                        }

                        match content_block {
                            AnthropicContentBlock::Text { .. } => {}
                            AnthropicContentBlock::ToolUse { id, name, .. } => {
                                in_tool_block = true;
                                tool_id = Some(id);
                                tool_name = Some(name);
                                tool_json_buf.clear();
                            }
                            _ => {}
                        }
                    }
                    MessageStreamEvent::ContentBlockDelta { delta, .. } => {
                        match delta {
                            ContentBlockDelta::TextDelta { text } => {
                                text_buf.push_str(&text);
                            }
                            ContentBlockDelta::InputJsonDelta { partial_json }
                                if in_tool_block => {
                                    tool_json_buf.push_str(&partial_json);
                                }
                            _ => {}
                        }
                    }
                    MessageStreamEvent::ContentBlockStop { .. } if in_tool_block => {
                        in_tool_block = false;
                        let args: Value = serde_json::from_str(&tool_json_buf)
                            .unwrap_or_else(|e| {
                                tracing::warn!("failed to parse tool call args: {}", e);
                                json!({})
                            });
                        if let (Some(id), Some(name)) = (tool_id.take(), tool_name.take()) {
                            let _ = tx.send(Ok(LLMEvent::ToolCall { id, name, args }));
                        }
                    }
                    MessageStreamEvent::MessageDelta { delta, usage } => {
                        if !text_buf.is_empty() {
                            let _ = tx.send(Ok(LLMEvent::Text(
                                std::mem::take(&mut text_buf),
                            )));
                        }

                        if let Some(reason) = delta.stop_reason {
                            let reason_str = match reason {
                                StopReason::EndTurn => "end_turn",
                                StopReason::MaxTokens => "max_tokens",
                                StopReason::StopSequence => "stop_sequence",
                                StopReason::ToolUse => "tool_use",
                            };
                            let _ = tx.send(Ok(LLMEvent::Stop {
                                finish_reason: reason_str.to_string(),
                            }));
                        }

                        if usage.output_tokens > 0 {
                            let _ = tx.send(Ok(LLMEvent::Usage(crate::llm::Usage {
                                prompt_tokens: usage.input_tokens.unwrap_or(0),
                                completion_tokens: usage.output_tokens,
                                total_tokens: usage.input_tokens.unwrap_or(0) + usage.output_tokens,
                                cost: 0.0,
                            })));
                        }
                    }
                    MessageStreamEvent::MessageStop => break,
                    _ => {}
                }
            }

            let _ = std::mem::take(&mut text_buf);
        });

        let rx_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        Ok(Box::pin(rx_stream))
    }
}
