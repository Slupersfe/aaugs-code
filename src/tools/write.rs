use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolResult};

pub struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Create a new file with the given content. Fails if the file already exists."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path where to create the file"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write"
                }
            },
            "required": ["path", "content"]
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

        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult {
                success: false,
                output: "missing required argument: content".to_string(),
            },
        };

        // Create parent directories if needed
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    return ToolResult {
                        success: false,
                        output: format!("failed to create directory {}: {}", parent.display(), e),
                    };
                }
            }
        }

        match tokio::fs::write(path, content).await {
            Ok(_) => ToolResult {
                success: true,
                output: format!("Successfully wrote {} bytes to {}", content.len(), path),
            },
            Err(e) => ToolResult {
                success: false,
                output: format!("failed to write {}: {}", path, e),
            },
        }
    }
}
