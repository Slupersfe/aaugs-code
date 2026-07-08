use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;

use super::{Tool, ToolResult};

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command in the project directory. Returns stdout and stderr output."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Maximum execution time in milliseconds (default: 30000)",
                    "minimum": 1000
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command (default: current directory)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let command = match args.get("command").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult {
                success: false,
                output: "missing required argument: command".to_string(),
            },
        };

        let timeout_ms = args.get("timeout_ms").and_then(|v| v.as_u64()).unwrap_or(30000);
        let workdir = args.get("workdir").and_then(|v| v.as_str());

        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = Command::new("cmd");
            c.arg("/C");
            c.arg(command);
            c
        } else {
            let mut c = Command::new("sh");
            c.arg("-c");
            c.arg(command);
            c
        };

        if let Some(dir) = workdir {
            cmd.current_dir(dir);
        }

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            cmd.output(),
        ).await;

        match result {
            Ok(Ok(output)) => {
                let mut result = String::new();

                if !output.stdout.is_empty() {
                    if !result.is_empty() { result.push_str("\n"); }
                    result.push_str(&String::from_utf8_lossy(&output.stdout));
                }
                if !output.stderr.is_empty() {
                    if !result.is_empty() { result.push_str("\n"); }
                    result.push_str(&String::from_utf8_lossy(&output.stderr));
                }

                let success = output.status.success();
                let exit_code = output.status.code().unwrap_or(-1);

                if !success {
                    result = format!("exit code {}\n{}", exit_code, result);
                } else {
                    result = result.trim().to_string();
                }

                ToolResult {
                    success,
                    output: result,
                }
            }
            Ok(Err(e)) => ToolResult {
                success: false,
                output: format!("failed to execute command: {}", e),
            },
            Err(_) => ToolResult {
                success: false,
                output: format!("command timed out after {}ms", timeout_ms),
            },
        }
    }
}
