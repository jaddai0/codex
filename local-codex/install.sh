#!/bin/zsh
set -euo pipefail

repo_root=${0:A:h:h}
install_bin=${LOCAL_CODEX_INSTALL_BIN:-${HOME}/.local/bin}
install_share=${LOCAL_CODEX_INSTALL_SHARE:-${HOME}/.local/share/local-codex}

python3 - "$repo_root" <<'PY'
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys

try:
    mcp_version = version("mcp")
except PackageNotFoundError as error:
    raise SystemExit("Mavis install requires Python mcp>=1.16,<2") from error
parts = mcp_version.split(".")
if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit() or \
        int(parts[0]) != 1 or int(parts[1]) < 16:
    raise SystemExit(f"Mavis install requires Python mcp>=1.16,<2; found {mcp_version}")

source = Path(sys.argv[1]).resolve(strict=True)
revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
dirty = subprocess.check_output(
    ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
    text=True,
)
if dirty:
    raise SystemExit("Mavis install requires a clean committed source checkout")
base = "local-codex/mavis/mavis"
committed = set(subprocess.check_output(
    ["git", "-C", str(source), "ls-tree", "-r", "--name-only", revision, base],
    text=True,
).splitlines())
actual = {path.relative_to(source).as_posix() for path in (source / base).rglob("*")
          if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
if not committed or actual != committed:
    raise SystemExit("Mavis install requires an exact committed source package")
for relative in sorted(committed):
    source_bytes = (source / relative).read_bytes()
    committed_bytes = subprocess.check_output(["git", "-C", str(source), "show", f"{revision}:{relative}"])
    if source_bytes != committed_bytes:
        raise SystemExit(f"Mavis install source differs from Git revision: {relative}")
PY

cd "$repo_root/codex-rs"
if [[ ${MAVIS_REUSE_CORE:-0} == 1 ]]; then
    python3 - "$repo_root" "$install_share" <<'PY'
from pathlib import Path
import hashlib
import json
import subprocess
import sys

source = Path(sys.argv[1]).resolve(strict=True)
share = Path(sys.argv[2]).resolve(strict=True)
manifest = json.loads((share / "install-manifest.json").read_text())
previous = manifest.get("source_revision")
installed = share / "local-codex-core"
built = source / "codex-rs/target/release/codex"
if (manifest.get("schema_version") != "mavis.installed-core/v1"
        or Path(manifest.get("source_repository", "")).resolve() != source
        or manifest.get("core_binary") != str(installed)
        or not isinstance(previous, str)
        or not installed.is_file() or not built.is_file()):
    raise SystemExit("Mavis cannot reuse an unbound core binary")
for binary in (installed, built):
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != manifest.get("core_sha256"):
        raise SystemExit("Mavis reusable core differs from the installed candidate")
if subprocess.run(
    ["git", "-C", str(source), "diff", "--quiet", previous, "HEAD", "--", "codex-rs"],
    check=False,
).returncode != 0:
    raise SystemExit("Mavis Rust source changed since the verified core build")
PY
else
    cargo build --release --bin codex
fi

mkdir -p "$install_bin" "$install_share"
install -m 0755 target/release/codex "$install_share/local-codex-core"
install -m 0755 "$repo_root/local-codex/bin/local-codex" "$install_bin/mavis"
install -m 0755 "$repo_root/local-codex/bin/local-codex" "$install_bin/local-codex"
install -m 0755 "$repo_root/local-codex/prepare_runtime.py" "$install_share/prepare_runtime.py"
install -m 0755 "$repo_root/local-codex/launch_core.py" "$install_share/launch_core.py"
install -m 0644 "$repo_root/local-codex/generation_lease.py" "$install_share/generation_lease.py"
install -m 0755 "$repo_root/local-codex/trial_runtime.py" "$install_share/trial_runtime.py"
install -m 0644 "$repo_root/local-codex/persona.toml" "$install_share/persona.toml"
install -m 0644 "$repo_root/codex-rs/models-manager/prompt.md" "$install_share/base-instructions.md"
rm -rf "$install_share/mavis"
cp -R "$repo_root/local-codex/mavis/mavis" "$install_share/mavis"
find "$install_share/mavis" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$install_share/mavis" -type f -name '*.pyc' -delete
find "$install_share/mavis" -type d -exec chmod 0755 {} +
find "$install_share/mavis" -type f -exec chmod 0644 {} +

desktop_launcher=${MAVIS_DESKTOP_LAUNCHER:-${HOME}/Desktop/Mavis.command}
install -m 0755 "$repo_root/local-codex/mavis/Mavis.command" "$desktop_launcher"

PYTHONPATH="$install_share${PYTHONPATH:+:$PYTHONPATH}" python3 - "$install_share" "$install_bin/mavis" "$repo_root" "$desktop_launcher" <<'PY'
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from mavis.package_provenance import package_tree_sha256

share = Path(sys.argv[1]).resolve()
launcher = Path(sys.argv[2]).resolve()
source = Path(sys.argv[3]).resolve(strict=True)
desktop_launcher = Path(sys.argv[4]).resolve(strict=True)
revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
dirty = subprocess.check_output(
    ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
    text=True,
)
if dirty:
    raise SystemExit("Mavis install source changed while installing")
committed_package = subprocess.check_output(
    ["git", "-C", str(source), "ls-tree", "-r", "--name-only", revision,
     "local-codex/mavis/mavis"], text=True,
).splitlines()
if not committed_package:
    raise SystemExit("install source revision has no committed Mavis package")
for relative in committed_package:
    installed = share / relative.removeprefix("local-codex/mavis/")
    committed = subprocess.check_output(["git", "-C", str(source), "show", f"{revision}:{relative}"])
    if not installed.is_file() or installed.read_bytes() != committed:
        raise SystemExit(f"installed Mavis source differs from Git revision: {relative}")
installed_sources = {
    "local-codex/bin/local-codex": [launcher, launcher.with_name("local-codex")],
    "local-codex/prepare_runtime.py": [share / "prepare_runtime.py"],
    "local-codex/launch_core.py": [share / "launch_core.py"],
    "local-codex/generation_lease.py": [share / "generation_lease.py"],
    "local-codex/trial_runtime.py": [share / "trial_runtime.py"],
    "local-codex/persona.toml": [share / "persona.toml"],
    "codex-rs/models-manager/prompt.md": [share / "base-instructions.md"],
    "local-codex/mavis/Mavis.command": [desktop_launcher],
}
for relative, destinations in installed_sources.items():
    committed = subprocess.check_output(["git", "-C", str(source), "show", f"{revision}:{relative}"])
    for installed in destinations:
        if not installed.is_file() or installed.read_bytes() != committed:
            raise SystemExit(f"installed Mavis source differs from Git revision: {relative}")
core = share / "local-codex-core"
manifest = {
    "schema_version": "mavis.installed-core/v1",
    "core_binary": str(core),
    "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
    "launcher": str(launcher),
    "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
    "trial_runtime_sha256": hashlib.sha256((share / "trial_runtime.py").read_bytes()).hexdigest(),
    "launch_core_sha256": hashlib.sha256((share / "launch_core.py").read_bytes()).hexdigest(),
    "prepare_runtime_sha256": hashlib.sha256((share / "prepare_runtime.py").read_bytes()).hexdigest(),
    "generation_lease_sha256": hashlib.sha256((share / "generation_lease.py").read_bytes()).hexdigest(),
    "base_instructions_sha256": hashlib.sha256((share / "base-instructions.md").read_bytes()).hexdigest(),
    "persona_sha256": hashlib.sha256((share / "persona.toml").read_bytes()).hexdigest(),
    "mavis_package_sha256": package_tree_sha256(share / "mavis"),
    "python_executable": sys.executable,
    "mcp_version": version("mcp"),
    "source_repository": str(source),
    "source_revision": revision,
}
path = share / "install-manifest.json"
temporary = share / ".install-manifest.json.tmp"
temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
temporary.replace(path)
PY

print "Installed $install_bin/mavis and $desktop_launcher (with local-codex compatibility command)"
