use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::PathBuf;
#[cfg(not(windows))]
use std::process::Command;

use serde::Serialize;
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct RawOutputReference {
    pub path: PathBuf,
    pub bytes: u64,
    pub complete: bool,
}

pub(crate) struct RawOutputSpool {
    path: PathBuf,
    file: File,
    bytes: u64,
}

impl RawOutputSpool {
    pub(crate) fn maybe_open() -> io::Result<Option<Self>> {
        if std::env::var("MAVIS_RAW_OUTPUT_REQUIRED").as_deref() != Ok("1") {
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
            let dir = match std::env::var_os("MAVIS_PROJECT_ROOT") {
                Some(root) => {
                    let user_home = std::env::var_os("HOME").ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "HOME is required")
                    })?;
                    project_spool_dir(PathBuf::from(root), PathBuf::from(user_home))?
                }
                None => home.join("tool-output"),
            };
            Self::open_in(dir).map(Some)
        }
    }

    pub(crate) fn open_in(dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
        let dir = fs::canonicalize(dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
        }
        let path = dir.join(format!("{}.raw", Uuid::new_v4()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&path)?;
        Ok(Self {
            path,
            file,
            bytes: 0,
        })
    }

    pub(crate) fn append(&mut self, chunk: &[u8]) -> io::Result<()> {
        self.file.write_all(chunk)?;
        self.bytes = self.bytes.saturating_add(chunk.len() as u64);
        self.file.sync_data()
    }

    pub(crate) fn reference(&mut self, complete: bool) -> io::Result<RawOutputReference> {
        self.file.sync_all()?;
        Ok(RawOutputReference {
            path: self.path.clone(),
            bytes: self.bytes,
            complete,
        })
    }
}

#[cfg(not(windows))]
fn project_spool_dir(root: PathBuf, user_home: PathBuf) -> io::Result<PathBuf> {
    if !root.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "MAVIS_PROJECT_ROOT must be absolute",
        ));
    }
    let root = fs::canonicalize(root)?;
    if root == fs::canonicalize(user_home)? {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the home-directory Git root cannot be a Mavis project",
        ));
    }
    let output = project_git(&root)
        .args(["rev-parse", "--show-toplevel"])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "MAVIS_PROJECT_ROOT must be the Git checkout root",
        ));
    }
    let top = String::from_utf8(output.stdout)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    if fs::canonicalize(top.trim())? != root {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "MAVIS_PROJECT_ROOT must be the Git checkout root",
        ));
    }

    let state = root.join(".mavis");
    ensure_private_directory(&state)?;
    let ignore = state.join(".gitignore");
    match fs::symlink_metadata(&ignore) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "project .mavis/.gitignore must be a regular file",
            ));
        }
        Ok(_) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            options.open(&ignore)?.write_all(b"*\n")?;
        }
        Err(err) => return Err(err),
    }
    let dir = state.join("tool-output");
    ensure_private_directory(&dir)?;
    let ignored = project_git(&root)
        .args(["check-ignore", "-q", "--", ".mavis/tool-output/probe.raw"])
        .status()?;
    if !ignored.success() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "project .mavis/tool-output is not Git-ignored",
        ));
    }
    Ok(dir)
}

#[cfg(not(windows))]
fn project_git(root: &std::path::Path) -> Command {
    let mut command = Command::new("git");
    command.current_dir(root);
    for name in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_PREFIX",
    ] {
        command.env_remove(name);
    }
    command
}

#[cfg(not(windows))]
fn ensure_private_directory(path: &std::path::Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_dir() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} must be a real directory", path.display()),
            ));
        }
        Ok(_) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => fs::create_dir(path)?,
        Err(err) => return Err(err),
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_exec::head_tail_buffer::HeadTailBuffer;
    use pretty_assertions::assert_eq;

    #[cfg(unix)]
    #[test]
    #[serial_test::serial]
    fn non_project_session_keeps_service_home_spool() {
        let temp = tempfile::tempdir().unwrap();
        let prior_required = std::env::var_os("MAVIS_RAW_OUTPUT_REQUIRED");
        let prior_home = std::env::var_os("MAVIS_HOME");
        let prior_project = std::env::var_os("MAVIS_PROJECT_ROOT");
        unsafe {
            std::env::set_var("MAVIS_RAW_OUTPUT_REQUIRED", "1");
            std::env::set_var("MAVIS_HOME", temp.path());
            std::env::remove_var("MAVIS_PROJECT_ROOT");
        }
        let result = RawOutputSpool::maybe_open()
            .unwrap()
            .unwrap()
            .reference(true);
        match prior_required {
            Some(value) => unsafe { std::env::set_var("MAVIS_RAW_OUTPUT_REQUIRED", value) },
            None => unsafe { std::env::remove_var("MAVIS_RAW_OUTPUT_REQUIRED") },
        }
        match prior_home {
            Some(value) => unsafe { std::env::set_var("MAVIS_HOME", value) },
            None => unsafe { std::env::remove_var("MAVIS_HOME") },
        }
        match prior_project {
            Some(value) => unsafe { std::env::set_var("MAVIS_PROJECT_ROOT", value) },
            None => unsafe { std::env::remove_var("MAVIS_PROJECT_ROOT") },
        }
        let reference = result.unwrap();
        assert_eq!(
            reference.path.parent(),
            Some(
                temp.path()
                    .canonicalize()
                    .unwrap()
                    .join("tool-output")
                    .as_path()
            )
        );
    }

    #[cfg(unix)]
    #[test]
    fn project_spool_is_private_ignored_and_rejects_the_home_git_root() {
        let temp = tempfile::tempdir().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();
        assert!(
            Command::new("git")
                .args(["init", "-q"])
                .current_dir(&project)
                .status()
                .unwrap()
                .success()
        );
        assert_eq!(
            project_spool_dir(project.clone(), project.clone())
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert!(!project.join(".mavis").exists());

        let dir = project_spool_dir(project.clone(), temp.path().to_path_buf()).unwrap();
        assert_eq!(
            dir,
            project.canonicalize().unwrap().join(".mavis/tool-output")
        );
        let mut spool = RawOutputSpool::open_in(dir).unwrap();
        spool.append(b"private failure\n").unwrap();
        let reference = spool.reference(true).unwrap();
        assert_eq!(fs::read(&reference.path).unwrap(), b"private failure\n");
        assert_eq!(
            fs::read_to_string(project.join(".mavis/.gitignore")).unwrap(),
            "*\n"
        );
        assert!(
            Command::new("git")
                .args(["check-ignore", "-q", "--", ".mavis/tool-output/probe.raw"])
                .current_dir(&project)
                .status()
                .unwrap()
                .success()
        );
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(project.join(".mavis"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(project.join(".mavis/tool-output"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
    }

    #[cfg(unix)]
    #[test]
    fn project_spool_rejects_unignored_and_symlinked_storage() {
        let temp = tempfile::tempdir().unwrap();
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();
        assert!(
            Command::new("git")
                .args(["init", "-q"])
                .current_dir(&project)
                .status()
                .unwrap()
                .success()
        );
        let state = project.join(".mavis");
        fs::create_dir(&state).unwrap();
        fs::write(state.join(".gitignore"), b"").unwrap();
        assert_eq!(
            project_spool_dir(project.clone(), temp.path().to_path_buf())
                .unwrap_err()
                .kind(),
            io::ErrorKind::PermissionDenied
        );
        fs::remove_dir_all(&state).unwrap();
        std::os::unix::fs::symlink(temp.path(), &state).unwrap();
        assert_eq!(
            project_spool_dir(project, temp.path().to_path_buf())
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[cfg(unix)]
    #[test]
    #[serial_test::serial]
    fn rejects_relative_home_before_opening_capture() {
        let prior_required = std::env::var_os("MAVIS_RAW_OUTPUT_REQUIRED");
        let prior_home = std::env::var_os("MAVIS_HOME");
        unsafe {
            std::env::set_var("MAVIS_RAW_OUTPUT_REQUIRED", "1");
            std::env::set_var("MAVIS_HOME", "relative-mavis-home");
        }
        let result = RawOutputSpool::maybe_open();
        match prior_required {
            Some(value) => unsafe { std::env::set_var("MAVIS_RAW_OUTPUT_REQUIRED", value) },
            None => unsafe { std::env::remove_var("MAVIS_RAW_OUTPUT_REQUIRED") },
        }
        match prior_home {
            Some(value) => unsafe { std::env::set_var("MAVIS_HOME", value) },
            None => unsafe { std::env::remove_var("MAVIS_HOME") },
        }
        assert_eq!(result.err().unwrap().kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn preserves_buried_failure_beyond_head_tail_cap() {
        let temp = tempfile::tempdir().unwrap();
        let mut spool = RawOutputSpool::open_in(temp.path().join("tool-output")).unwrap();
        let mut capped = HeadTailBuffer::<1_048_576>::default();
        let chunks = [
            vec![b'a'; 700_000],
            b"\nFAILURE: buried diagnostic\n".to_vec(),
            vec![b'z'; 700_000],
        ];
        for chunk in &chunks {
            spool.append(chunk).unwrap();
            capped.push_chunk(chunk);
        }
        let reference = spool.reference(true).unwrap();
        assert!(reference.path.is_absolute());
        let raw = fs::read(&reference.path).unwrap();
        assert!(
            raw.windows(b"FAILURE: buried diagnostic".len())
                .any(|window| window == b"FAILURE: buried diagnostic")
        );
        assert!(
            !capped
                .to_bytes()
                .windows(b"FAILURE: buried diagnostic".len())
                .any(|window| window == b"FAILURE: buried diagnostic")
        );
        assert_eq!(reference.bytes, raw.len() as u64);
        assert!(reference.complete);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&reference.path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(reference.path.parent().unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
    }
}
