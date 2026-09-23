"""Content fingerprint for an installed Mavis Python package."""

import hashlib
from pathlib import Path


def package_tree_sha256(root: Path) -> str:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("installed Mavis package is unavailable")
    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"))
    if not entries or any(path.is_symlink() for path in entries):
        raise ValueError("installed Mavis package has no files or contains a symlink")
    files = [path for path in entries if "__pycache__" not in path.parts and path.suffix != ".pyc"]
    if not any(path.is_file() for path in files):
        raise ValueError("installed Mavis package has no source files")
    for path in files:
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("installed Mavis package contains an unsupported entry")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
