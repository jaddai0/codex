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
install -m 0644 "$repo_root/local-codex/persona.toml" "$install_share/persona.toml"
install -m 0644 "$repo_root/codex-rs/models-manager/prompt.md" "$install_share/base-instructions.md"

print "Installed $install_bin/mavis (with local-codex compatibility command)"
