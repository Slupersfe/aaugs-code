use std::collections::HashMap;
use std::pin::Pin;

use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCall,
        ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        ChatCompletionStreamOptions, ChatCompletionTool, ChatCompletionTools,
        CreateChatCompletionRequestArgs, FinishReason, FunctionCall, FunctionObject,
    },
    Client,
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};

use super::{ContentBlock, LLMError, LLMEvent, LLMProvider, Message, Role, ToolDef};
use crate::config::ProviderConfig;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAIProvider {
    client: Client<OpenAIConfig>,
    model: String,
    max_tokens: u32,
    temperature: f32,
    provider_name: String,
}

fn convert_error(e: async_openai::error::OpenAIError) -> LLMError {
    match e {
        async_openai::error::OpenAIError::Reqwest(e) => {
            if let Some(status) = e.status() {
                let body = e.to_string();
                LLMError::Http { status, body }
            } else {
                LLMError::Stream(e.to_string())
            }
        }
        async_openai::error::OpenAIError::ApiError(e) => {
            LLMError::Http { status: e.status_code, body: e.api_error.message }
        }
        async_openai::error::OpenAIError::JSONDeserialize(e, _) => LLMError::Serde(e),
        async_openai::error::OpenAIError::StreamError(e) => LLMError::Stream(e.to_string()),
        async_openai::error::OpenAIError::InvalidArgument(e) => LLMError::Config(e),
        _ => LLMError::Stream(e.to_string()),
    }
}

fn convert_message(msg: &Message) -> ChatCompletionRequestMessage {
    match msg.role {
        Role::System => {
            let text = msg.content.iter()
                .filter_map(|b| {
                    if let ContentBlock::Text { text } = b { Some(text.clone()) }
                    else { None }
                })
                .collect::<Vec<_>>()
                .join("\n");
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(text),
                name: None,
            })
        }
        Role::User => {
            let text = msg.content.iter()
                .filter_map(|b| {
                    if let ContentBlock::Text { text } = b { Some(text.clone()) }
                    else { None }
                })
                .collect::<Vec<_>>()
                .join("\n");
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(text),
                name: None,
            })
        }
        Role::Assistant => {
            let text = msg.content.iter()
                .filter_map(|b| {
                    if let ContentBlock::Text { text } = b { Some(text.clone()) }
                    else { None }
                })
                .collect::<Vec<_>>()
                .join("");

            let tool_calls: Vec<ChatCompletionMessageToolCalls> = msg.content.iter()
                .filter_map(|b| {
                    if let ContentBlock::ToolUse { id, name, input } = b {
                        Some(ChatCompletionMessageToolCalls::Function(
                            ChatCompletionMessageToolCall {
                                id: id.clone(),
                                function: FunctionCall {
                                    name: name.clone(),
                                    arguments: input.to_string(),
                                },
                            },
                        ))
                    } else { None }
                })
                .collect();

            let content = if text.is_empty() { None }
                else { Some(ChatCompletionRequestAssistantMessageContent::Text(text)) };

            ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                content,
                tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                ..Default::default()
            })
        }
        Role::Tool => {
            for block in &msg.content {
                if let ContentBlock::ToolResult { tool_use_id, content } = block {
                    return ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
                        content: ChatCompletionRequestToolMessageContent::Text(content.clone()),
                        tool_call_id: tool_use_id.clone(),
                    });
                }
            }
            // fallback — shouldn't happen
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(String::new()),
                name: None,
            })
        }
    }
}

fn convert_tools(tools: &[ToolDef]) -> Vec<ChatCompletionTools> {
    tools.iter().map(|t| {
        ChatCompletionTools::Function(ChatCompletionTool {
            function: FunctionObject {
                name: t.name.clone(),
                description: Some(t.description.clone()),
                parameters: Some(t.input_schema.clone()),
                strict: None,
            },
        })
    }).collect()
}

impl OpenAIProvider {
    /// Note: `timeout_secs` is accepted for API consistency but not applied to the SDK,
    /// because async-openai 0.41 depends on reqwest 0.13 (project uses 0.12),
    /// preventing us from passing a custom reqwest client with a timeout.
    pub fn new(cfg: &ProviderConfig, max_tokens: u32, temperature: f32, provider_name: &str, _timeout_secs: u64) -> Result<Self, LLMError> {
        let api_key = cfg.api_key.clone();
        let base_url = cfg.base_url.clone().unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(base_url);
        Ok(Self {
            client: Client::with_config(config),
            model: cfg.model.clone(),
            max_tokens,
            temperature,
            provider_name: provider_name.to_string(),
        })
    }

}

#[async_trait]
impl LLMProvider for OpenAIProvider {
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
        let api_messages: Vec<ChatCompletionRequestMessage> = messages.iter()
            .map(convert_message)
            .collect();

        let api_tools = convert_tools(tools);

        let mut builder = CreateChatCompletionRequestArgs::default();
        builder.model(&self.model);
        builder.messages(api_messages);
        builder.stream(true);
        builder.max_tokens(self.max_tokens);
        builder.temperature(self.temperature);
        builder.stream_options(ChatCompletionStreamOptions {
            include_usage: Some(true),
            include_obfuscation: None,
        });

        if !api_tools.is_empty() {
            builder.tools(api_tools);
        }

        let request = builder.build().map_err(|e| LLMError::Config(e.to_string()))?;

        let sdk_stream = self.client.chat().create_stream(request).await
            .map_err(convert_error)?;

        use tokio::sync::mpsc;

        let (tx, rx) = mpsc::unbounded_channel::<Result<LLMEvent, LLMError>>();

        tokio::spawn(async move {
            struct PendingToolCall {
                id: Option<String>,
                name: Option<String>,
                args: String,
            }

            let mut tool_calls: HashMap<usize, PendingToolCall> = HashMap::new();
            let mut text_buf = String::new();

            let mut stream = sdk_stream;

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(Err(convert_error(e)));
                        break;
                    }
                };

                let mut usage_event: Option<LLMEvent> = None;
                if let Some(usage) = chunk.usage {
                    if usage.total_tokens > 0 {
                        usage_event = Some(LLMEvent::Usage(crate::llm::Usage {
                            prompt_tokens: usage.prompt_tokens,
                            completion_tokens: usage.completion_tokens,
                            total_tokens: usage.total_tokens,
                            cost: 0.0,
                        }));
                    }
                }

                for choice in chunk.choices {
                    let delta = choice.delta;

                    if let Some(text) = delta.content {
                        if !text.is_empty() {
                            text_buf.push_str(&text);
                            let _ = tx.send(Ok(LLMEvent::Text(text)));
                        }
                    }

                    if let Some(tcs) = delta.tool_calls {
                        for tc in tcs {
                            let index = tc.index as usize;
                            let pending = tool_calls.entry(index).or_insert_with(|| PendingToolCall {
                                id: None,
                                name: None,
                                args: String::new(),
                            });

                            if let Some(id) = tc.id {
                                pending.id = Some(id);
                            }

                            if let Some(func) = tc.function {
                                if let Some(name) = func.name {
                                    pending.name = Some(name);
                                }
                                if let Some(args) = func.arguments {
                                    pending.args.push_str(&args);
                                }
                            }
                        }
                    }

                    if let Some(reason) = choice.finish_reason {
                        let reason_str = match reason {
                            FinishReason::Stop => "stop",
                            FinishReason::Length => "length",
                            FinishReason::ToolCalls => "tool_calls",
                            FinishReason::ContentFilter => "content_filter",
                            FinishReason::FunctionCall => "function_call",
                        };

                        let _ = std::mem::take(&mut text_buf);

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
                            finish_reason: reason_str.to_string(),
                        }));
                        if let Some(usage) = usage_event.take() {
                            let _ = tx.send(Ok(usage));
                        }
                    }
                }
            }

            let _ = std::mem::take(&mut text_buf);
        });

        let rx_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        Ok(Box::pin(rx_stream))
    }
}
