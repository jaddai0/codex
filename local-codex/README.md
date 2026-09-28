# Mavis

This fork keeps Codex's local coding harness and sends model traffic only to a loopback oMLX server. It uses a separate home at `~/.local-codex`, does not reuse OpenAI OAuth state, disables analytics, telemetry, feedback, and update checks, and blocks OpenAI login/cloud commands in the installed launcher.

The disabled update setting also suppresses `doctor` network probes for GitHub and the installed ChatGPT desktop update feed. `doctor` still tests the active oMLX endpoint.

The launcher reads the live oMLX inventory every time it starts and writes an isolated runtime config with the served context limit. MiMo-V2.6-Flash-RL-UNCENSORED-MLX-oQ4 is the default Mavis model. Its isolated oMLX settings default to the model's 1,048,576-token context and 131,072-token maximum output; explicit per-model settings remain in control. Set `LOCAL_CODEX_MODEL` for a one-session alternative or use Codex's `--model` option. The full limits describe the configured capacity, not a completed million-token local inference test.

The normal Codex shell, patch, planning, skills, MCP, and multi-agent surfaces
remain available. OpenAI's remote app and curated-plugin catalogs are disabled
because they violate the local-only boundary. External MCP servers still need
their own setup; the launcher does not inherit servers or credentials from
`~/.codex`.

Run `./local-codex/install.sh`, then open any project directory in Terminal and start it with `mavis`. The older `local-codex` command remains as a compatibility alias.

Mavis's persona is stored separately at `~/.local-codex/persona.toml`. Change only the `name` value to rename the coding persona; the launcher regenerates `~/.local-codex/AGENTS.md` on the next start. Mavis remains explicitly separate from Iris.

Environment variables:

- `OMLX_BASE_URL` defaults to Mavis's isolated `http://127.0.0.1:8001/v1` and must remain loopback-only. IRIS remains on port 8000.
- `LOCAL_CODEX_MODEL` selects any coding-capable model reported by oMLX.
- `LOCAL_CODEX_HOME` changes the isolated harness home.
- `MAVIS_HOME` changes Mavis service state (default `~/.local-codex/mavis-service`; `~/.mavis` is intentionally not used because it is an existing MiniMax alias on this machine).
- `MAVIS_OMLX_BIN` and `MAVIS_MODEL_DIR` override the isolated oMLX executable and shared read-only weight directory.
