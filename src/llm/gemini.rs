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

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

pub struct GeminiProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
    temperature: f32,
    provider_name: String,
}

#[derive(Serialize)]
struct GeminiRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<SystemInstruction>,
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GeminiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Serialize)]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<Part>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Part {
    Text { text: String },
    FunctionCall { function_call: FunctionCall },
    FunctionResponse { function_response: FunctionResponse },
}

#[derive(Serialize)]
struct FunctionCall {
    name: String,
    args: Value,
}

#[derive(Serialize)]
struct FunctionResponse {
    name: String,
    response: Value,
}

#[derive(Serialize)]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: Value,
}

impl GeminiProvider {
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

    fn model_path(&self) -> String {
        let model = if self.model.contains('/') {
            self.model.split('/').next_back().unwrap_or(&self.model)
        } else {
            &self.model
        };
        format!("models/{}", model)
    }

    fn base_url(&self) -> String {
        self.base_url.trim_end_matches('/').to_string()
    }

    fn convert_messages(&self, messages: &[Message]) -> (Option<SystemInstruction>, Vec<GeminiContent>) {
        let mut system: Option<SystemInstruction> = None;
        let mut contents = Vec::new();
        let mut tool_name_for_id: std::collections::HashMap<String, String> = std::collections::HashMap::new();

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
                    system = Some(SystemInstruction {
                        parts: vec![Part::Text { text }],
                    });
                }
                Role::User => {
                    let parts: Vec<Part> = msg.content.iter().map(|block| {
                        match block {
                            ContentBlock::Text { text } => Part::Text { text: text.clone() },
                            _ => Part::Text { text: String::new() },
                        }
                    }).collect();
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts,
                    });
                }
                Role::Assistant => {
                    let parts: Vec<Part> = msg.content.iter().filter_map(|block| {
                        match block {
                            ContentBlock::Text { text } => {
                                if text.is_empty() { None }
                                else { Some(Part::Text { text: text.clone() }) }
                            }
                            ContentBlock::ToolUse { id: _, name, input } => {
                                Some(Part::FunctionCall {
                                    function_call: FunctionCall {
                                        name: name.clone(),
                                        args: input.clone(),
                                    },
                                })
                            }
                            _ => None,
                        }
                    }).collect();

                    if !parts.is_empty() {
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts,
                        });
                    }
                }
                Role::Tool => {
                    // Build mapping from tool_use_id -> name by scanning assistant messages first
                    for m in messages {
                        if m.role == Role::Assistant {
                            for block in &m.content {
                                if let ContentBlock::ToolUse { id, name, .. } = block {
                                    tool_name_for_id.entry(id.clone()).or_insert_with(|| name.clone());
                                }
                            }
                        }
                    }
                    let parts: Vec<Part> = msg.content.iter().filter_map(|block| {
                        if let ContentBlock::ToolResult { tool_use_id, content } = block {
                            let name = tool_name_for_id.get(tool_use_id)
                                .cloned()
                                .unwrap_or_else(|| "unknown".to_string());
                            Some(Part::FunctionResponse {
                                function_response: FunctionResponse {
                                    name,
                                    response: json!({"result": content}),
                                },
                            })
                        } else { None }
                    }).collect();

                    if !parts.is_empty() {
                        contents.push(GeminiContent {
                            role: "function".to_string(),
                            parts,
                        });
                    }
                }
            }
        }

        (system, contents)
    }

    fn convert_tools(&self, tools: &[ToolDef]) -> Vec<GeminiTool> {
        if tools.is_empty() {
            return Vec::new();
        }
        vec![GeminiTool {
            function_declarations: tools.iter().map(|t| GeminiFunctionDeclaration {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            }).collect(),
        }]
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

    async fn stream_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMEvent, LLMError>> + Send>>, LLMError> {
        let (system, contents) = self.convert_messages(messages);
        let api_tools = self.convert_tools(tools);

        let body = GeminiRequest {
            system_instruction: system,
            contents,
            tools: api_tools,
            generation_config: Some(GenerationConfig {
                max_output_tokens: Some(self.max_tokens),
                temperature: Some(self.temperature),
            }),
        };

        let url = format!(
            "{}/{}:streamGenerateContent?key={}&alt=sse",
            self.base_url(),
            self.model_path(),
            self.api_key
        );

        let response = self.client
            .post(&url)
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
            while let Some(event_result) = sse_stream.next().await {
                match event_result {
                    Ok(sse) => {
                        let parsed: Value = match serde_json::from_str(&sse.data) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };

                        // Extract usage metadata (top-level in Gemini)
                        if let Some(meta) = parsed.get("usageMetadata") {
                            let prompt = meta.get("promptTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            let completion = meta.get("candidatesTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            let total = meta.get("totalTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            if total > 0 {
                                let _ = tx.send(Ok(LLMEvent::Usage(crate::llm::Usage {
                                    prompt_tokens: prompt,
                                    completion_tokens: completion,
                                    total_tokens: total,
                                    cost: 0.0,
                                })));
                            }
                        }

                        let candidates = match parsed.get("candidates").and_then(|c| c.as_array()) {
                            Some(c) => c,
                            None => continue,
                        };

                        for candidate in candidates {
                            let content = match candidate.get("content") {
                                Some(c) => c,
                                None => continue,
                            };

                            let parts = match content.get("parts").and_then(|p| p.as_array()) {
                                Some(p) => p,
                                None => continue,
                            };

                            for part in parts {
                                // Text part
                                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                    if !text.is_empty() {
                                        let _ = tx.send(Ok(LLMEvent::Text(text.to_string())));
                                    }
                                }

                                // Function call part
                                if let Some(fc) = part.get("functionCall") {
                                    if let Some(name) = fc.get("name").and_then(|n| n.as_str()) {
                                        let args = fc.get("args").cloned().unwrap_or_else(|| {
                                            tracing::warn!("missing 'args' in tool call response");
                                            json!({})
                                        });
                                        let id = format!("fc_{}", uuid::Uuid::new_v4());
                                        let _ = tx.send(Ok(LLMEvent::ToolCall {
                                            id,
                                            name: name.to_string(),
                                            args,
                                        }));
                                    }
                                }
                            }

                            // Finish reason
                            if let Some(reason) = candidate.get("finishReason").and_then(|r| r.as_str()) {
                                let mapped = match reason {
                                    "STOP" => "end_turn",
                                    "MAX_TOKENS" => "max_tokens",
                                    "SAFETY" => "safety",
                                    _ => reason,
                                };
                                let _ = tx.send(Ok(LLMEvent::Stop {
                                    finish_reason: mapped.to_string(),
                                }));
                            }
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
