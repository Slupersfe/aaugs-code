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
