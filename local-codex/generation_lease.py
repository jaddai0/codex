"""One host-owned local generation at a time across Mavis launch paths."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat
import time


@contextmanager
def generation_lease(mavis_home: Path, *, purpose: str):
    home = Path(mavis_home).resolve()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = home / "generation.lock"
    inherited = os.environ.pop("MAVIS_GENERATION_LEASE_FD", None)
    if inherited is not None:
        try:
            descriptor = int(inherited)
            if descriptor < 3:
                raise ValueError
            source = os.fstat(descriptor)
            target = path.stat()
        except (OSError, ValueError) as error:
            raise RuntimeError("inherited Mavis host lease descriptor is invalid") from error
        if (
            not stat.S_ISREG(source.st_mode)
            or (source.st_dev, source.st_ino) != (target.st_dev, target.st_ino)
        ):
            raise RuntimeError("inherited Mavis host lease is not generation.lock")
        try:
            # The parent's open file description already owns this flock. A
            # newly opened descriptor loses this check while that owner exists.
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("inherited Mavis host lease is not owned") from error
        # Do not unlock a shared open file description: the parent keeps it
        # locked while it waits for this child to exit.
        yield descriptor
        return
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
            yield handle.fileno()
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
