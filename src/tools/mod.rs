mod read;
mod write;
mod edit;
mod bash;
mod glob;
mod grep;
mod question;

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::llm::ToolDef;

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn execute(&self, args: Value) -> ToolResult;
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };
        registry.register(Box::new(read::ReadTool));
        registry.register(Box::new(write::WriteTool));
        registry.register(Box::new(edit::EditTool));
        registry.register(Box::new(bash::BashTool));
        registry.register(Box::new(glob::GlobTool));
        registry.register(Box::new(grep::GrepTool));
        registry.register(Box::new(question::QuestionTool));
        registry
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn definitions(&self) -> Vec<ToolDef> {
        self.tools.values().map(|tool| ToolDef {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            input_schema: tool.parameters(),
        }).collect()
    }

    pub async fn execute(&self, name: &str, args: Value) -> ToolResult {
        match self.get(name) {
            Some(tool) => tool.execute(args).await,
            None => ToolResult {
                success: false,
                output: format!("unknown tool: {}", name),
            },
        }
    }
}
