use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::Value;

use super::{
    LLMError, LLMEvent, LLMProvider, Message, ToolDef,
    ContentBlock, Role,
};
use crate::config::ProviderConfig;

use gemini_rust::{
    ClientError, Content, FunctionCall, FunctionDeclaration,
    FunctionResponse, Gemini, GeminiBuilder, Part, Tool, FinishReason,
};

fn convert_error(e: ClientError) -> LLMError {
    match e {
        ClientError::BadResponse { code, description } => {
            let status = reqwest::StatusCode::from_u16(code)
                .unwrap_or(reqwest::StatusCode::BAD_REQUEST);
            LLMError::Http { status, body: description.unwrap_or_default() }
        }
        ClientError::PerformRequest { source, .. }
        | ClientError::PerformRequestNew { source }
        | ClientError::DecodeResponse { source } => LLMError::Network(source),
        ClientError::Deserialize { source } => LLMError::Serde(source),
        ClientError::InvalidApiKey { .. }
        | ClientError::ConstructUrl { .. }
        | ClientError::UrlParse { .. }
        | ClientError::InvalidResourceName { .. } => LLMError::Config(e.to_string()),
        ClientError::OperationTimeout { name } => LLMError::Stream(format!("request timed out: {name}")),
        ClientError::MissingResponseHeader { header } => LLMError::Stream(format!("missing response header: {header}")),
        ClientError::BadPart { source } => LLMError::Stream(format!("SSE parse error: {source}")),
        ClientError::Io { source } => LLMError::Stream(format!("I/O error: {source}")),
        ClientError::OperationFailed { name, code, message } => {
            let status = reqwest::StatusCode::from_u16(code as u16)
                .unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR);
            LLMError::Http { status, body: format!("{name}: {message}") }
        }
    }
}

pub struct GeminiProvider {
    client: Gemini,
    model: String,
    max_tokens: u32,
    temperature: f32,
    provider_name: String,
}

impl GeminiProvider {
    pub fn new(cfg: &ProviderConfig, max_tokens: u32, temperature: f32, provider_name: &str, timeout_secs: u64) -> Result<Self, LLMError> {
        let model = if cfg.model.starts_with("models/") {
            cfg.model.clone()
        } else {
            format!("models/{}", cfg.model)
        };

        let client_builder = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs));
        let mut builder = GeminiBuilder::new(&cfg.api_key)
            .with_model(model.clone())
            .with_http_client(client_builder);

        if let Some(base_url_str) = &cfg.base_url {
            let base_url = url::Url::parse(base_url_str)
                .map_err(|e| LLMError::Config(format!("invalid Gemini base URL: {e}")))?;
            builder = builder.with_base_url(base_url);
        }

        let client = builder.build()
            .map_err(|e| LLMError::Config(format!("failed to create Gemini client: {e}")))?;

        Ok(Self {
            client,
            model,
            max_tokens,
            temperature,
            provider_name: provider_name.to_string(),
        })
    }

    fn convert_messages(&self, messages: &[Message]) -> (Option<String>, Vec<Content>) {
        let mut system: Option<String> = None;
        let mut contents: Vec<Content> = Vec::new();
        let mut tool_name_for_id: std::collections::HashMap<String, String> = std::collections::HashMap::new();

        for msg in messages {
            if msg.role == Role::Assistant {
                for block in &msg.content {
                    if let ContentBlock::ToolUse { id, name, .. } = block {
                        tool_name_for_id.entry(id.clone()).or_insert_with(|| name.clone());
                    }
                }
            }
        }

        for msg in messages {
            match msg.role {
                Role::System => {
                    let text = msg.content.iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text { text } = b { Some(text.clone()) } else { None }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    system = Some(text);
                }
                Role::User => {
                    let parts: Vec<Part> = msg.content.iter().map(|block| {
                        match block {
                            ContentBlock::Text { text } => Part::Text {
                                text: text.clone(),
                                thought: None,
                                thought_signature: None,
                            },
                            _ => Part::Text {
                                text: String::new(),
                                thought: None,
                                thought_signature: None,
                            },
                        }
                    }).collect();
                    contents.push(
                        Content { parts: Some(parts), role: None }
                            .with_role(gemini_rust::Role::User)
                    );
                }
                Role::Assistant => {
                    let mut parts: Vec<Part> = Vec::new();
                    for block in &msg.content {
                        match block {
                            ContentBlock::Text { text } if !text.is_empty() => {
                                parts.push(Part::Text {
                                    text: text.clone(),
                                    thought: None,
                                    thought_signature: None,
                                });
                            }
                            ContentBlock::ToolUse { id: _, name, input } => {
                                parts.push(Part::FunctionCall {
                                    function_call: FunctionCall::new(name.clone(), input.clone()),
                                    thought_signature: None,
                                });
                            }
                            _ => {}
                        }
                    }
                    if !parts.is_empty() {
                        contents.push(
                            Content { parts: Some(parts), role: None }
                                .with_role(gemini_rust::Role::Model)
                        );
                    }
                }
                Role::Tool => {
                    let mut parts: Vec<Part> = Vec::new();
                    for block in &msg.content {
                        if let ContentBlock::ToolResult { tool_use_id, content } = block {
                            let name = tool_name_for_id.get(tool_use_id)
                                .cloned()
                                .unwrap_or_else(|| "unknown".to_string());
                            parts.push(Part::FunctionResponse {
                                function_response: FunctionResponse::new(name, serde_json::json!({"result": content})),
                            });
                        }
                    }
                    if !parts.is_empty() {
                        contents.push(
                            Content { parts: Some(parts), role: None }
                                .with_role(gemini_rust::Role::User)
                        );
                    }
                }
            }
        }

        (system, contents)
    }

    fn convert_tools(&self, tools: &[ToolDef]) -> Vec<Tool> {
        if tools.is_empty() {
            return Vec::new();
        }

        let declarations: Vec<FunctionDeclaration> = tools.iter().map(|t| {
            let mut decl = FunctionDeclaration::new(&t.name, &t.description, None);
            if t.input_schema != Value::Null {
                decl = decl.with_parameters_value(t.input_schema.clone());
            }
            decl
        }).collect();

        vec![Tool::with_functions(declarations)]
    }
}

#[async_trait]
impl LLMProvider for GeminiProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn default_model(&self) -> &str {
        &self.model
    }

    fn set_model(&mut self, model: &str) {
        self.model = model.to_string();
    }

    #[allow(deprecated)]
    #[tracing::instrument(skip(self, messages, tools))]
    async fn stream_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMEvent, LLMError>> + Send>>, LLMError> {
        let (system_text, contents) = self.convert_messages(messages);
        let api_tools = self.convert_tools(tools);

        let mut builder = self.client.generate_content()
            .with_temperature(self.temperature)
            .with_max_output_tokens(self.max_tokens as i32);

        if let Some(text) = system_text {
            builder = builder.with_system_instruction(text);
        }

        for tool in api_tools {
            builder = builder.with_tool(tool);
        }

        builder.contents = contents;

        let stream = builder.execute_stream().await.map_err(convert_error)?;

        use tokio::sync::mpsc;
        let (tx, rx) = mpsc::unbounded_channel::<Result<LLMEvent, LLMError>>();

        tokio::spawn(async move {
            let mut stream = stream;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(response) => {
                        if let Some(usage) = &response.usage_metadata {
                            let prompt = usage.prompt_token_count.unwrap_or(0) as u32;
                            let completion = usage.candidates_token_count.unwrap_or(0) as u32;
                            let total = usage.total_token_count.unwrap_or(0) as u32;
                            if total > 0 {
                                let _ = tx.send(Ok(LLMEvent::Usage(super::Usage {
                                    prompt_tokens: prompt,
                                    completion_tokens: completion,
                                    total_tokens: total,
                                    cost: 0.0,
                                })));
                            }
                        }

                        for candidate in &response.candidates {
                            if let Some(parts) = &candidate.content.parts {
                                for part in parts {
                                    match part {
                                        Part::Text { text, .. } if !text.is_empty() => {
                                            let _ = tx.send(Ok(LLMEvent::Text(text.clone())));
                                        }
                                        Part::FunctionCall { function_call, .. } => {
                                            let id = format!("fc_{}", uuid::Uuid::new_v4());
                                            let _ = tx.send(Ok(LLMEvent::ToolCall {
                                                id,
                                                name: function_call.name.clone(),
                                                args: function_call.args.clone(),
                                            }));
                                        }
                                        _ => {}
                                    }
                                }
                            }

                            if let Some(reason) = &candidate.finish_reason {
                                let mapped = match reason {
                                    FinishReason::Stop => "end_turn",
                                    FinishReason::MaxTokens => "max_tokens",
                                    FinishReason::Safety => "safety",
                                    _ => "other",
                                };
                                let _ = tx.send(Ok(LLMEvent::Stop {
                                    finish_reason: mapped.to_string(),
                                }));
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(convert_error(e)));
                        break;
                    }
                }
            }
        });

        let rx_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        Ok(Box::pin(rx_stream))
    }
}
