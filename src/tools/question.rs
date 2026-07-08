use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolResult};

pub struct QuestionTool;

#[async_trait]
impl Tool for QuestionTool {
    fn name(&self) -> &str {
        "question"
    }

    fn description(&self) -> &str {
        "Ask the user a clarifying question and get their response. Use this when you need additional information or confirmation."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user"
                },
                "options": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional list of predefined options for the user to choose from"
                }
            },
            "required": ["question"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let question = match args.get("question").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return ToolResult {
                success: false,
                output: "missing required argument: question".to_string(),
            },
        };

        let options = args.get("options").and_then(|v| v.as_array());

        let answer = match options {
            Some(opts) => {
                let items: Vec<&str> = opts.iter()
                    .filter_map(|v| v.as_str())
                    .collect();

                if items.is_empty() {
                    ask_text(question).await
                } else {
                    ask_select(question, &items).await
                }
            }
            None => ask_text(question).await,
        };

        ToolResult {
            success: true,
            output: answer,
        }
    }
}

async fn ask_text(question: &str) -> String {
    dialoguer::Input::<String>::new()
        .with_prompt(question)
        .interact_text()
        .unwrap_or_else(|_| String::new())
}

async fn ask_select(question: &str, options: &[&str]) -> String {
    let idx = dialoguer::Select::new()
        .with_prompt(question)
        .items(options)
        .default(0)
        .interact()
        .unwrap_or(0);
    options[idx].to_string()
}
