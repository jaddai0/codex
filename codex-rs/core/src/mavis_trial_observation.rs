//! Host-readable observation of the effective Mavis E1 trial configuration.

use codex_protocol::openai_models::ModelsResponse;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

#[derive(Serialize)]
struct Observation<'a> {
    schema_version: &'static str,
    trial_id: &'a str,
    session_id: String,
    thread_id: String,
    model: &'a str,
    model_provider: &'a str,
    catalog_sha256: String,
    base_instructions_sha256: String,
    rollout_path: Option<&'a str>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn verified_catalog_digest(
    catalog_path: &Path,
    loaded_catalog: Option<&ModelsResponse>,
) -> anyhow::Result<String> {
    let catalog_bytes = std::fs::read(catalog_path)?;
    let source_catalog: ModelsResponse = serde_json::from_slice(&catalog_bytes)?;
    if loaded_catalog != Some(&source_catalog) {
        anyhow::bail!("Mavis E1 loaded model catalog differs from host catalog");
    }
    Ok(sha256(&catalog_bytes))
}

pub(crate) fn emit_if_requested(
    codex_home: &Path,
    loaded_catalog: Option<&ModelsResponse>,
    base_instructions: &str,
    session_id: impl ToString,
    thread_id: impl ToString,
    model: &str,
    model_provider: &str,
    rollout_path: Option<&str>,
) -> anyhow::Result<()> {
    let Some(event_path) = std::env::var_os("MAVIS_E1_EFFECTIVE_CONFIG_EVENT") else {
        return Ok(());
    };
    let trial_id = std::env::var("MAVIS_E1_TRIAL_ID")?;
    let catalog_path = std::env::var_os("MAVIS_E1_CATALOG_PATH")
        .ok_or_else(|| anyhow::anyhow!("Mavis E1 catalog path is missing"))?;
    let event_path = std::path::PathBuf::from(event_path);
    let event_parent = event_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Mavis E1 event path has no parent"))?;
    if trial_id.is_empty()
        || !trial_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        || event_parent.canonicalize()? != codex_home.canonicalize()?
    {
        anyhow::bail!("Mavis E1 observation identity or path is invalid");
    }
    let catalog_sha256 = verified_catalog_digest(Path::new(&catalog_path), loaded_catalog)?;
    let observation = Observation {
        schema_version: "mavis.e1-effective-config/v1",
        trial_id: &trial_id,
        session_id: session_id.to_string(),
        thread_id: thread_id.to_string(),
        model,
        model_provider,
        catalog_sha256,
        base_instructions_sha256: sha256(base_instructions.as_bytes()),
        rollout_path,
    };
    let encoded = serde_json::to_vec(&observation)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&event_path)?;
    file.write_all(&encoded)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatched_loaded_catalog_is_rejected_before_event_creation() {
        let home = tempfile::tempdir().unwrap();
        let catalog_path = home.path().join("catalog.json");
        std::fs::write(&catalog_path, r#"{"models":[]}"#).unwrap();
        let error = verified_catalog_digest(&catalog_path, None).unwrap_err();
        assert!(error.to_string().contains("loaded model catalog differs"));
        let source: ModelsResponse =
            serde_json::from_slice(&std::fs::read(&catalog_path).unwrap()).unwrap();
        assert_eq!(
            verified_catalog_digest(&catalog_path, Some(&source)).unwrap(),
            sha256(&std::fs::read(&catalog_path).unwrap())
        );
    }
}
