use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolResult};

pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Replace occurrences of old_string with new_string in a file. By default replaces the first occurrence; use replace_all: true to replace all."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to search for"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace all occurrences; otherwise only the first",
                    "default": false
                }
            },
            "required": ["path", "old_string", "new_string"]
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

        let old = match args.get("old_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult {
                success: false,
                output: "missing required argument: old_string".to_string(),
            },
        };

        let new = match args.get("new_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult {
                success: false,
                output: "missing required argument: new_string".to_string(),
            },
        };

        let replace_all = args.get("replace_all").and_then(|v| v.as_bool()).unwrap_or(false);

        let content = match tokio::fs::read_to_string(path).await {
            Ok(c) => c,
            Err(e) => return ToolResult {
                success: false,
                output: format!("failed to read {}: {}", path, e),
            },
        };

        let (new_content, count) = if replace_all {
            let count = content.matches(old).count();
            (content.replace(old, new), count)
        } else {
            match content.find(old) {
                Some(pos) => {
                    let new_content = format!("{}{}{}", &content[..pos], new, &content[pos + old.len()..]);
                    (new_content, 1)
                }
                None => {
                    return ToolResult {
                        success: false,
                        output: format!("could not find '{}' in {}", old, path),
                    };
                }
            }
        };

        match tokio::fs::write(path, &new_content).await {
            Ok(_) => ToolResult {
                success: true,
                output: format!(
                    "Successfully replaced {} occurrence{} in {}",
                    count,
                    if count == 1 { "" } else { "s" },
                    path
                ),
            },
            Err(e) => ToolResult {
                success: false,
                output: format!("failed to write {}: {}", path, e),
            },
        }
    }
}
