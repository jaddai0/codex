#!/bin/zsh
set -euo pipefail

launcher=${MAVIS_BIN:-${HOME}/.local/bin/mavis}
if [[ ! -x "$launcher" ]]; then
  print -u2 "Mavis is not installed at $launcher"
  exit 127
fi

cd "${MAVIS_PROJECT_DIR:-${HOME}/Dev-Projects}"
exec "$launcher"
