use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolResult};

pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Supports patterns like **/*.rs, src/**/*.ts, etc."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to search for"
                },
                "max_output": {
                    "type": "integer",
                    "description": "Maximum bytes in the result sent to the model. Use a high value (e.g. 100000) to get full output without truncation.",
                    "minimum": 1
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult {
                success: false,
                output: "missing required argument: pattern".to_string(),
            },
        };

        let entries: Vec<String> = match glob::glob(pattern) {
            Ok(entries) => {
                let mut paths: Vec<String> = entries
                    .filter_map(|e| e.ok().map(|p| p.display().to_string()))
                    .collect();
                paths.sort();
                paths
            }
            Err(e) => return ToolResult {
                success: false,
                output: format!("invalid glob pattern '{}': {}", pattern, e),
            },
        };

        ToolResult {
            success: true,
            output: if entries.is_empty() {
                format!("no matches for '{}'", pattern)
            } else {
                entries.join("\n")
            },
        }
    }
}
