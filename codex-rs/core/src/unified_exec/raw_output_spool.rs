use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::PathBuf;

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
        let home = std::env::var_os("MAVIS_HOME")
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "MAVIS_HOME is required"))?;
        Self::open_in(PathBuf::from(home).join("tool-output")).map(Some)
    }

    pub(crate) fn open_in(dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
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
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_exec::head_tail_buffer::HeadTailBuffer;

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
