#!/bin/zsh
set -euo pipefail

repo_root=${0:A:h:h}
install_bin=${LOCAL_CODEX_INSTALL_BIN:-${HOME}/.local/bin}
install_share=${LOCAL_CODEX_INSTALL_SHARE:-${HOME}/.local/share/local-codex}

cd "$repo_root/codex-rs"
cargo build --release --bin codex

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
find "$install_share/mavis" -type d -exec chmod 0755 {} +
find "$install_share/mavis" -type f -exec chmod 0644 {} +

PYTHONPATH="$install_share${PYTHONPATH:+:$PYTHONPATH}" python3 - "$install_share" "$install_bin/mavis" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from mavis.package_provenance import package_tree_sha256

share = Path(sys.argv[1]).resolve()
launcher = Path(sys.argv[2]).resolve()
core = share / "local-codex-core"
manifest = {
    "schema_version": "mavis.installed-core/v1",
    "core_binary": str(core),
    "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
    "launcher": str(launcher),
    "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
    "trial_runtime_sha256": hashlib.sha256((share / "trial_runtime.py").read_bytes()).hexdigest(),
    "launch_core_sha256": hashlib.sha256((share / "launch_core.py").read_bytes()).hexdigest(),
    "mavis_package_sha256": package_tree_sha256(share / "mavis"),
}
path = share / "install-manifest.json"
temporary = share / ".install-manifest.json.tmp"
temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
temporary.replace(path)
PY

desktop_launcher=${MAVIS_DESKTOP_LAUNCHER:-${HOME}/Desktop/Mavis.command}
install -m 0755 "$repo_root/local-codex/mavis/Mavis.command" "$desktop_launcher"

print "Installed $install_bin/mavis and $desktop_launcher (with local-codex compatibility command)"
