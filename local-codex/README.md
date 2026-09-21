# Mavis

This fork keeps Codex's local coding harness and sends model traffic only to a loopback oMLX server. It uses a separate home at `~/.local-codex`, does not reuse OpenAI OAuth state, disables analytics, telemetry, feedback, and update checks, and blocks OpenAI login/cloud commands in the installed launcher.

The disabled update setting also suppresses `doctor` network probes for GitHub and the installed ChatGPT desktop update feed. `doctor` still tests the active oMLX endpoint.

The launcher reads the live oMLX inventory every time it starts. It selects the loaded default model, writes an isolated runtime config and accurate context metadata, and exposes the model picker without hardcoding a model name. Set `LOCAL_CODEX_MODEL` for a one-session default or use Codex's `--model` option.

The normal Codex shell, patch, planning, skills, MCP, and multi-agent surfaces
remain available. OpenAI's remote app and curated-plugin catalogs are disabled
because they violate the local-only boundary. External MCP servers still need
their own setup; the launcher does not inherit servers or credentials from
`~/.codex`.

Run `./local-codex/install.sh`, then open any project directory in Terminal and start it with `mavis`. The older `local-codex` command remains as a compatibility alias.

Mavis's persona is stored separately at `~/.local-codex/persona.toml`. Change only the `name` value to rename the coding persona; the launcher regenerates `~/.local-codex/AGENTS.md` on the next start. Mavis remains explicitly separate from Iris.

Environment variables:

- `OMLX_BASE_URL` defaults to `http://127.0.0.1:8000/v1` and must remain loopback-only.
- `LOCAL_CODEX_MODEL` selects any coding-capable model reported by oMLX.
- `LOCAL_CODEX_HOME` changes the isolated harness home.
