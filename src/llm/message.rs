use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    #[allow(dead_code)]
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content,
        }
    }

    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
            }],
        }
    }

    #[allow(dead_code)]
    pub fn text_content(&self) -> Option<&str> {
        for block in &self.content {
            if let ContentBlock::Text { text } = block {
                return Some(text);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_system() {
        let m = Message::system("prompt");
        assert_eq!(m.role, Role::System);
        assert_eq!(m.text_content(), Some("prompt"));
    }

    #[test]
    fn test_message_user() {
        let m = Message::user("hello");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.text_content(), Some("hello"));
    }

    #[test]
    fn test_message_assistant() {
        let m = Message::assistant(vec![ContentBlock::Text { text: "hi".into() }]);
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.text_content(), Some("hi"));
    }

    #[test]
    fn test_message_tool_result() {
        let m = Message::tool_result("call_1", "output");
        assert_eq!(m.role, Role::Tool);
        assert!(m.text_content().is_none()); // ToolResult is not a Text block
    }

    #[test]
    fn test_serialize_roundtrip() {
        let m = Message::user("hello");
        let json = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.text_content(), Some("hello"));
    }

    #[test]
    fn test_content_block_tool_use_serialize() {
        let input = serde_json::json!({"path": "/tmp"});
        let block = ContentBlock::ToolUse {
            id: "tu_1".into(),
            name: "read".into(),
            input: input.clone(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("tool_use"));
        assert!(json.contains("read"));
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        match back {
            ContentBlock::ToolUse { name, input: inp, .. } => {
                assert_eq!(name, "read");
                assert_eq!(inp, input);
            }
            _ => panic!("expected ToolUse"),
        }
    }
}
