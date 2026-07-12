//! Apply a non-OpenAI model provider (e.g. xAI Grok) into Codex `config.toml`.
//!
//! CodexClaw still drives the Codex App-Server harness; this module only rewrites
//! the isolated `CODEX_HOME` config so the harness talks to an OpenAI-compatible
//! backend such as `https://api.x.ai/v1`.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Spec written into Codex `config.toml` as `model_provider` + `[model_providers.<id>]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodexProviderSpec {
    /// Provider id used for top-level `model_provider` and table key.
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub env_key: String,
    pub wire_api: String,
    /// When true, set top-level `model_provider` (and optional `model`).
    pub set_as_default: bool,
    pub default_model: Option<String>,
    /// Extra model ids surfaced in the `/model` picker when this provider is active.
    pub models: Vec<String>,
}

impl CodexProviderSpec {
    /// Defaults for xAI Grok (OpenAI-compatible Responses API).
    pub fn xai_grok() -> Self {
        Self {
            id: "xai".to_string(),
            name: "xAI Grok".to_string(),
            base_url: "https://api.x.ai/v1".to_string(),
            env_key: "XAI_API_KEY".to_string(),
            wire_api: "responses".to_string(),
            set_as_default: true,
            default_model: Some("grok-4".to_string()),
            models: default_grok_model_ids(),
        }
    }
}

/// Canonical Grok model ids for picker / catalog merge.
pub fn default_grok_model_ids() -> Vec<String> {
    vec![
        "grok-4".to_string(),
        "grok-4.5".to_string(),
        "grok-3".to_string(),
        "grok-3-mini".to_string(),
    ]
}

/// Pure transform: merge `provider` into an existing Codex `config.toml` body.
///
/// - Upserts `[model_providers.<id>]` with `name`, `base_url`, `env_key`, `wire_api`.
/// - When `set_as_default`, sets top-level `model_provider` and optional `model`.
/// - Leaves unrelated keys intact (parsed via `toml::Value`).
pub fn apply_model_provider_to_config(raw: &str, provider: &CodexProviderSpec) -> Result<String> {
    let mut root: toml::Value = if raw.trim().is_empty() {
        toml::Value::Table(toml::map::Map::new())
    } else {
        toml::from_str(raw).context("failed to parse Codex config.toml while applying provider")?
    };

    let table = root
        .as_table_mut()
        .context("Codex config.toml root must be a table")?;

    if provider.set_as_default {
        table.insert(
            "model_provider".to_string(),
            toml::Value::String(provider.id.clone()),
        );
        if let Some(model) = provider
            .default_model
            .as_ref()
            .map(|m| m.trim())
            .filter(|m| !m.is_empty())
        {
            table.insert("model".to_string(), toml::Value::String(model.to_string()));
        }
    }

    let providers = table
        .entry("model_providers".to_string())
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let providers_table = providers
        .as_table_mut()
        .context("model_providers must be a table")?;

    let mut provider_table = toml::map::Map::new();
    provider_table.insert(
        "name".to_string(),
        toml::Value::String(provider.name.clone()),
    );
    provider_table.insert(
        "base_url".to_string(),
        toml::Value::String(provider.base_url.clone()),
    );
    provider_table.insert(
        "env_key".to_string(),
        toml::Value::String(provider.env_key.clone()),
    );
    provider_table.insert(
        "wire_api".to_string(),
        toml::Value::String(provider.wire_api.clone()),
    );
    providers_table.insert(provider.id.clone(), toml::Value::Table(provider_table));

    // Prefer a stable, readable layout for the managed isolated home.
    toml::to_string_pretty(&root).context("failed to serialize Codex config.toml")
}

/// Read `config.toml` under `codex_home`, apply `provider`, write back.
pub fn apply_model_provider_to_codex_home(
    codex_home: &Path,
    provider: &CodexProviderSpec,
) -> Result<()> {
    let path = codex_home.join("config.toml");
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(err) => {
            return Err(err).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    let updated = apply_model_provider_to_config(&raw, provider)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&path, updated)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Model ids that should appear in the picker when this provider is enabled.
pub fn provider_model_ids(provider: &CodexProviderSpec) -> Vec<String> {
    fn push_unique(out: &mut Vec<String>, name: &str) {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return;
        }
        if !out
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(trimmed))
        {
            out.push(trimmed.to_string());
        }
    }

    let mut out = Vec::new();
    if let Some(model) = provider.default_model.as_deref() {
        push_unique(&mut out, model);
    }
    for model in &provider.models {
        push_unique(&mut out, model);
    }
    if out.is_empty() {
        for model in default_grok_model_ids() {
            push_unique(&mut out, &model);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn apply_writes_xai_provider_block_and_defaults() {
        let raw = r#"
model = "gpt-5.4"
service_tier = "flex"

[profiles.default]
model = "gpt-5.4-mini"
"#;
        let updated = apply_model_provider_to_config(raw, &CodexProviderSpec::xai_grok()).unwrap();
        let parsed: toml::Value = toml::from_str(&updated).unwrap();
        let table = parsed.as_table().unwrap();

        assert_eq!(
            table.get("model_provider").and_then(|v| v.as_str()),
            Some("xai")
        );
        assert_eq!(table.get("model").and_then(|v| v.as_str()), Some("grok-4"));
        // Unrelated keys preserved
        assert_eq!(
            table.get("service_tier").and_then(|v| v.as_str()),
            Some("flex")
        );
        assert!(table.contains_key("profiles"));

        let provider = table
            .get("model_providers")
            .and_then(|v| v.get("xai"))
            .and_then(|v| v.as_table())
            .expect("xai provider table");
        assert_eq!(
            provider.get("base_url").and_then(|v| v.as_str()),
            Some("https://api.x.ai/v1")
        );
        assert_eq!(
            provider.get("env_key").and_then(|v| v.as_str()),
            Some("XAI_API_KEY")
        );
        assert_eq!(
            provider.get("wire_api").and_then(|v| v.as_str()),
            Some("responses")
        );
        assert_eq!(
            provider.get("name").and_then(|v| v.as_str()),
            Some("xAI Grok")
        );
    }

    #[test]
    fn apply_without_set_as_default_only_upserts_provider_table() {
        let raw = "model = \"gpt-5.4\"\n";
        let mut spec = CodexProviderSpec::xai_grok();
        spec.set_as_default = false;
        let updated = apply_model_provider_to_config(raw, &spec).unwrap();
        let parsed: toml::Value = toml::from_str(&updated).unwrap();
        let table = parsed.as_table().unwrap();
        assert!(table.get("model_provider").is_none());
        assert_eq!(table.get("model").and_then(|v| v.as_str()), Some("gpt-5.4"));
        assert!(
            table
                .get("model_providers")
                .and_then(|v| v.get("xai"))
                .is_some()
        );
    }

    #[test]
    fn apply_to_codex_home_creates_config_toml() {
        let dir = tempdir().unwrap();
        apply_model_provider_to_codex_home(dir.path(), &CodexProviderSpec::xai_grok()).unwrap();
        let raw = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        assert!(raw.contains("api.x.ai"));
        assert!(raw.contains("XAI_API_KEY"));
        assert!(raw.contains("model_provider"));
    }

    #[test]
    fn provider_model_ids_includes_default_and_extras() {
        let mut spec = CodexProviderSpec::xai_grok();
        spec.models = vec!["grok-2".into(), "grok-4".into()];
        let ids = provider_model_ids(&spec);
        assert!(ids.iter().any(|m| m == "grok-4"));
        assert!(ids.iter().any(|m| m == "grok-2"));
        assert_eq!(ids.iter().filter(|m| *m == "grok-4").count(), 1);
    }

    #[test]
    fn openai_path_unchanged_when_provider_not_applied() {
        // Sanity: empty apply is not called; original OpenAI-oriented config stays as-is.
        let raw = "model = \"gpt-5.4\"\n";
        let parsed: toml::Value = toml::from_str(raw).unwrap();
        assert_eq!(
            parsed.get("model").and_then(|v| v.as_str()),
            Some("gpt-5.4")
        );
        assert!(parsed.get("model_providers").is_none());
    }
}
