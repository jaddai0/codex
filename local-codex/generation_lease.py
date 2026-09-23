"""One host-owned local generation at a time across Mavis launch paths."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


@contextmanager
def generation_lease(mavis_home: Path, *, purpose: str):
    home = Path(mavis_home).resolve()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = home / "generation.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another Mavis local generation owns the host lease"
            ) from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "purpose": purpose,
                        "acquired_at_epoch": time.time(),
                    }
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
