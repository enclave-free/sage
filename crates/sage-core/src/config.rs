use anyhow::{Context, Result};

use crate::marmot::MarmotConfig;

#[derive(Debug, Clone, PartialEq)]
pub enum MessengerType {
    Signal,
    Marmot,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Config {
    pub tinfoil_api_url: String,
    pub tinfoil_api_key: Option<String>,
    pub tinfoil_model: String,
    pub tinfoil_embedding_model: String,
    pub tinfoil_vision_model: String,

    pub database_url: String,

    /// Which messaging provider to use
    pub messenger_type: MessengerType,

    // Signal-specific config
    pub signal_phone_number: Option<String>,
    pub signal_allowed_users: Vec<String>,
    /// If set, connect to signal-cli daemon via TCP instead of spawning subprocess
    pub signal_cli_host: Option<String>,
    pub signal_cli_port: u16,

    // Marmot-specific config
    pub marmot_binary: String,
    pub marmot_relays: Vec<String>,
    pub marmot_state_dir: String,
    pub marmot_allowed_pubkeys: Vec<String>,
    pub marmot_auto_accept_welcomes: bool,

    pub brave_api_key: Option<String>,

    /// Workspace directory for shell commands and file operations
    pub workspace_path: String,

    pub http_port: u16,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            tinfoil_api_url: std::env::var("TINFOIL_API_URL")
                .unwrap_or_else(|_| "http://localhost:8089/v1".to_string()),
            tinfoil_api_key: std::env::var("TINFOIL_API_KEY").ok(),
            tinfoil_model: std::env::var("TINFOIL_MODEL")
                .unwrap_or_else(|_| "kimi-k2-6".to_string()),
            tinfoil_embedding_model: std::env::var("TINFOIL_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "nomic-embed-text".to_string()),
            tinfoil_vision_model: std::env::var("TINFOIL_VISION_MODEL").unwrap_or_else(|_| {
                std::env::var("TINFOIL_MODEL").unwrap_or_else(|_| "kimi-k2-6".to_string())
            }),

            database_url: std::env::var("DATABASE_URL").context("DATABASE_URL must be set")?,

            messenger_type: match std::env::var("MESSENGER")
                .unwrap_or_else(|_| "signal".to_string())
                .to_lowercase()
                .as_str()
            {
                "marmot" => MessengerType::Marmot,
                _ => MessengerType::Signal,
            },

            signal_phone_number: std::env::var("SIGNAL_PHONE_NUMBER").ok(),
            signal_allowed_users: std::env::var("SIGNAL_ALLOWED_USERS")
                .map(|s| s.split(',').map(|u| u.trim().to_string()).collect())
                .unwrap_or_default(),
            signal_cli_host: std::env::var("SIGNAL_CLI_HOST").ok(),
            signal_cli_port: std::env::var("SIGNAL_CLI_PORT")
                .unwrap_or_else(|_| "7583".to_string())
                .parse()
                .unwrap_or(7583),

            marmot_binary: std::env::var("MARMOT_BINARY").unwrap_or_else(|_| "marmotd".to_string()),
            marmot_relays: std::env::var("MARMOT_RELAYS")
                .map(|s| {
                    s.split(',')
                        .map(|r| r.trim().to_string())
                        .filter(|r| !r.is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            marmot_state_dir: std::env::var("MARMOT_STATE_DIR")
                .unwrap_or_else(|_| "/data/marmot-state".to_string()),
            marmot_allowed_pubkeys: std::env::var("MARMOT_ALLOWED_PUBKEYS")
                .map(|s| {
                    s.split(',')
                        .map(|p| p.trim().to_string())
                        .filter(|p| !p.is_empty())
                        .map(|p| {
                            if p == "*" {
                                p
                            } else {
                                crate::marmot::normalize_pubkey(&p).unwrap_or(p)
                            }
                        })
                        .collect()
                })
                .unwrap_or_default(),
            marmot_auto_accept_welcomes: std::env::var("MARMOT_AUTO_ACCEPT_WELCOMES")
                .map(|s| s != "false" && s != "0")
                .unwrap_or(true),

            brave_api_key: std::env::var("BRAVE_API_KEY").ok(),

            workspace_path: std::env::var("SAGE_WORKSPACE")
                .unwrap_or_else(|_| "/workspace".to_string()),

            http_port: std::env::var("HEALTH_PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .context("HEALTH_PORT must be a valid port number")?,
        })
    }

    pub fn marmot_config(&self) -> MarmotConfig {
        MarmotConfig {
            binary_path: self.marmot_binary.clone(),
            relays: self.marmot_relays.clone(),
            state_dir: self.marmot_state_dir.clone(),
            allowed_pubkeys: self.marmot_allowed_pubkeys.clone(),
            auto_accept_welcomes: self.marmot_auto_accept_welcomes,
        }
    }

    pub fn allowed_users(&self) -> &[String] {
        match self.messenger_type {
            MessengerType::Signal => &self.signal_allowed_users,
            MessengerType::Marmot => &self.marmot_allowed_pubkeys,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn config_reads_tinfoil_env_only() {
        let _guard = env_lock().lock().unwrap();

        let previous_database = std::env::var("DATABASE_URL").ok();
        let previous_tinfoil_api_url = std::env::var("TINFOIL_API_URL").ok();
        let previous_tinfoil_api_key = std::env::var("TINFOIL_API_KEY").ok();
        let previous_tinfoil_model = std::env::var("TINFOIL_MODEL").ok();
        let previous_tinfoil_embedding_model = std::env::var("TINFOIL_EMBEDDING_MODEL").ok();
        let previous_tinfoil_vision_model = std::env::var("TINFOIL_VISION_MODEL").ok();
        let previous_maple_api_url = std::env::var("MAPLE_API_URL").ok();
        let previous_maple_api_key = std::env::var("MAPLE_API_KEY").ok();
        let previous_maple_model = std::env::var("MAPLE_MODEL").ok();
        let previous_maple_embedding_model = std::env::var("MAPLE_EMBEDDING_MODEL").ok();
        let previous_maple_vision_model = std::env::var("MAPLE_VISION_MODEL").ok();

        std::env::set_var("DATABASE_URL", "postgres://sage:sage@localhost:5434/sage");
        std::env::set_var("TINFOIL_API_URL", "http://localhost:8089/v1");
        std::env::set_var("TINFOIL_API_KEY", "test-key");
        std::env::set_var("TINFOIL_MODEL", "kimi-k2-6");
        std::env::set_var("TINFOIL_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("TINFOIL_VISION_MODEL", "qwen3-vl-30b");

        std::env::set_var("MAPLE_API_URL", "http://legacy.invalid/v1");
        std::env::set_var("MAPLE_API_KEY", "legacy-key");
        std::env::set_var("MAPLE_MODEL", "legacy-model");
        std::env::set_var("MAPLE_EMBEDDING_MODEL", "legacy-embed");
        std::env::set_var("MAPLE_VISION_MODEL", "legacy-vision");

        let config = Config::from_env().unwrap();

        assert_eq!(config.tinfoil_api_url, "http://localhost:8089/v1");
        assert_eq!(config.tinfoil_api_key.as_deref(), Some("test-key"));
        assert_eq!(config.tinfoil_model, "kimi-k2-6");
        assert_eq!(config.tinfoil_embedding_model, "nomic-embed-text");
        assert_eq!(config.tinfoil_vision_model, "qwen3-vl-30b");

        restore_env("DATABASE_URL", previous_database);
        restore_env("TINFOIL_API_URL", previous_tinfoil_api_url);
        restore_env("TINFOIL_API_KEY", previous_tinfoil_api_key);
        restore_env("TINFOIL_MODEL", previous_tinfoil_model);
        restore_env("TINFOIL_EMBEDDING_MODEL", previous_tinfoil_embedding_model);
        restore_env("TINFOIL_VISION_MODEL", previous_tinfoil_vision_model);
        restore_env("MAPLE_API_URL", previous_maple_api_url);
        restore_env("MAPLE_API_KEY", previous_maple_api_key);
        restore_env("MAPLE_MODEL", previous_maple_model);
        restore_env("MAPLE_EMBEDDING_MODEL", previous_maple_embedding_model);
        restore_env("MAPLE_VISION_MODEL", previous_maple_vision_model);
    }

    fn restore_env(key: &str, previous: Option<String>) {
        if let Some(value) = previous {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }
}
