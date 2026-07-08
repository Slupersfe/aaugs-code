use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolResult};

pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Supports optional offset and limit for reading specific line ranges."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed, default: 1)",
                    "minimum": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                    "minimum": 1
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult {
                success: false,
                output: "missing required argument: path".to_string(),
            },
        };

        let content = match tokio::fs::read_to_string(path).await {
            Ok(c) => c,
            Err(e) => return ToolResult {
                success: false,
                output: format!("failed to read {}: {}", path, e),
            },
        };

        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let limit = args.get("limit").and_then(|v| v.as_u64());

        if offset == 1 && limit.is_none() {
            return ToolResult {
                success: true,
                output: content,
            };
        }

        let lines: Vec<&str> = content.lines().collect();
        let start = offset.saturating_sub(1);
        let selected: Vec<&str> = match limit {
            Some(l) => lines.iter().copied().skip(start).take(l as usize).collect(),
            None => lines.iter().copied().skip(start).collect(),
        };

        ToolResult {
            success: true,
            output: selected.join("\n"),
        }
    }
}
