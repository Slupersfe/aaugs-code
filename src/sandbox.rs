use std::collections::HashMap;

use crate::config::Config;

pub enum PermissionLevel {
    Allow,
    Ask,
    Deny,
}

pub struct Sandbox {
    rules: HashMap<String, String>,
    auto_approve: bool,
}

impl Sandbox {
    pub fn from_config(config: &Config) -> Self {
        let mut rules = HashMap::new();
        rules.insert("bash".to_string(), config.permissions.bash.clone());
        rules.insert("write".to_string(), config.permissions.write.clone());
        rules.insert("read".to_string(), config.permissions.read.clone());
        rules.insert("glob".to_string(), config.permissions.glob.clone());
        rules.insert("grep".to_string(), config.permissions.grep.clone());
        rules.insert("question".to_string(), "allow".to_string());
        Self { rules, auto_approve: false }
    }

    pub fn set_auto_approve(&mut self, val: bool) {
        self.auto_approve = val;
    }

    pub fn check(&self, tool_name: &str) -> PermissionLevel {
        if self.auto_approve {
            return PermissionLevel::Allow;
        }
        match self.rules.get(tool_name).map(|s| s.as_str()) {
            Some("allow") => PermissionLevel::Allow,
            Some("deny") => PermissionLevel::Deny,
            _ => PermissionLevel::Ask,
        }
    }

    pub fn request(&self, tool_name: &str, description: &str) -> bool {
        match self.check(tool_name) {
            PermissionLevel::Allow => true,
            PermissionLevel::Deny => false,
            PermissionLevel::Ask => {
                if !atty::is(atty::Stream::Stdin) {
                    // Non-TTY: auto-deny to avoid hanging
                    return false;
                }
                let approve = dialoguer::Confirm::new()
                    .with_prompt(format!("Allow tool '{}': {}", tool_name, description))
                    .default(true)
                    .interact()
                    .unwrap_or(false);
                approve
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, PermissionsConfig};

    fn sandbox_with_permissions(bash: &str, write: &str, read: &str) -> Sandbox {
        let mut cfg = Config::default();
        cfg.permissions = PermissionsConfig {
            bash: bash.to_string(),
            write: write.to_string(),
            read: read.to_string(),
            ..PermissionsConfig::default()
        };
        Sandbox::from_config(&cfg)
    }

    #[test]
    fn test_check_allow() {
        let s = sandbox_with_permissions("allow", "allow", "allow");
        assert!(matches!(s.check("bash"), PermissionLevel::Allow));
    }

    #[test]
    fn test_check_deny() {
        let s = sandbox_with_permissions("deny", "deny", "deny");
        assert!(matches!(s.check("bash"), PermissionLevel::Deny));
    }

    #[test]
    fn test_check_ask() {
        let s = sandbox_with_permissions("ask", "ask", "ask");
        assert!(matches!(s.check("bash"), PermissionLevel::Ask));
    }

    #[test]
    fn test_check_auto_approve() {
        let mut s = sandbox_with_permissions("ask", "ask", "ask");
        s.set_auto_approve(true);
        assert!(matches!(s.check("bash"), PermissionLevel::Allow));
    }

    #[test]
    fn test_check_question_always_allowed() {
        let s = sandbox_with_permissions("deny", "deny", "deny");
        assert!(matches!(s.check("question"), PermissionLevel::Allow));
    }

    #[test]
    fn test_check_default_is_ask() {
        let s = sandbox_with_permissions("unknown_value", "unknown_value", "unknown_value");
        assert!(matches!(s.check("bash"), PermissionLevel::Ask));
    }
}
