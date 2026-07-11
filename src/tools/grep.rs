use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolResult};

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents using a regex pattern. Supports filtering by file pattern."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "include": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g. '*.rs', 'src/**/*.ts')"
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
        let pattern_str = match args.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult {
                success: false,
                output: "missing required argument: pattern".to_string(),
            },
        };

        let re = match regex::Regex::new(pattern_str) {
            Ok(r) => r,
            Err(e) => return ToolResult {
                success: false,
                output: format!("invalid regex '{}': {}", pattern_str, e),
            },
        };

        let include = args.get("include").and_then(|v| v.as_str());

        // Walk files matching the include pattern, or all files recursively
        let search_root = std::env::current_dir().unwrap_or_default();
        let mut results = Vec::new();

        let walker = walk_dir(&search_root, include);
        for entry in walker {
            let path = match entry {
                Ok(p) => p,
                Err(_) => continue,
            };

            if path.is_dir() {
                continue;
            }

            let content = match tokio::fs::read_to_string(&path).await {
                Ok(c) => c,
                Err(_) => continue,
            };

            for (line_num, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    results.push(format!("{}:{}:{}", path.display(), line_num + 1, line.trim()));
                }
            }
        }

        ToolResult {
            success: true,
            output: if results.is_empty() {
                format!("no matches for '{}'", pattern_str)
            } else {
                results.join("\n")
            },
        }
    }
}

fn walk_dir(root: &std::path::Path, include: Option<&str>) -> Vec<std::io::Result<std::path::PathBuf>> {
    let mut entries = Vec::new();

    if let Ok(read_dir) = std::fs::read_dir(root) {
        for entry in read_dir {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();

            // Skip hidden directories and node_modules/.git
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with('.') || name == "node_modules" || name == "target" {
                    continue;
                }
            }

            if path.is_dir() {
                entries.extend(walk_dir(&path, include));
            } else if path.is_file() {
                if let Some(pat) = include {
                    let path_str = path.display().to_string();
                    if glob::Pattern::new(pat).map(|p| p.matches(&path_str)).unwrap_or(false) {
                        entries.push(Ok(path));
                    }
                } else {
                    entries.push(Ok(path));
                }
            }
        }
    }

    entries
}
