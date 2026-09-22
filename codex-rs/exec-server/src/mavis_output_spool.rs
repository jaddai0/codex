use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::PathBuf;

use uuid::Uuid;

pub(crate) struct MavisOutputSpool {
    path: PathBuf,
    file: File,
}

impl MavisOutputSpool {
    pub(crate) fn maybe_open(required: bool, process_id: &str) -> io::Result<Option<Self>> {
        if !required {
            return Ok(None);
        }
        #[cfg(windows)]
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "private Mavis raw output storage is unsupported on Windows",
        ));
        #[cfg(not(windows))]
        {
            let home = std::env::var_os("MAVIS_HOME").ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "MAVIS_HOME is required")
            })?;
            let home = PathBuf::from(home);
            if !home.is_absolute() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "MAVIS_HOME must be absolute",
                ));
            }
            Self::open_in(home.join("tool-output").join("exec-server"), process_id).map(Some)
        }
    }

    pub(crate) fn open_in(dir: PathBuf, process_id: &str) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
        let dir = fs::canonicalize(dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Some(parent) = dir.parent() {
                fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
            }
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
        }
        let id = Uuid::new_v4();
        let path = dir.join(format!("{id}.raw"));
        let manifest_path = dir.join(format!("{id}.json"));
        let file = private_file(&path)?;
        let mut manifest = private_file(&manifest_path)?;
        let body = serde_json::json!({
            "process_id": process_id,
            "raw_output_path": path,
        });
        serde_json::to_writer(&mut manifest, &body).map_err(io::Error::other)?;
        manifest.sync_all()?;
        Ok(Self { path, file })
    }

    pub(crate) fn append(&mut self, chunk: &[u8]) -> io::Result<()> {
        self.file.write_all(chunk)?;
        self.file.sync_data()
    }

    pub(crate) fn path(&self) -> &PathBuf {
        &self.path
    }
}

fn private_file(path: &PathBuf) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn rejects_relative_home() {
        let prior_home = std::env::var_os("MAVIS_HOME");
        unsafe { std::env::set_var("MAVIS_HOME", "relative-mavis-home") };
        let result = MavisOutputSpool::maybe_open(true, "test-process");
        match prior_home {
            Some(value) => unsafe { std::env::set_var("MAVIS_HOME", value) },
            None => unsafe { std::env::remove_var("MAVIS_HOME") },
        }
        assert_eq!(result.err().unwrap().kind(), io::ErrorKind::InvalidInput);
    }
}
