# Component ownership

| Component | Source owner | Runtime owner | Acceptance owner |
|---|---|---|---|
| Codex extension and launcher | `local-codex` fork | Mavis isolated home | Terra verifier |
| Mavis service and contracts | `local-codex/mavis` | Mavis service | Terra verifier |
| Provider and subscription routing | canonical model-gateway plugin | Model gateway | Terra verifier |
| Native harness processes | model-gateway adapter | tmux session per job | Mavis supervisor |
| Local inference | oMLX upstream/runtime | isolated Mavis oMLX process | E0 isolation gate |
| Model profiles | Mavis service | profile store | promotion verifier |
| Project evidence and retrieval | project `.mavis/` | Mavis service | retrieval gates |
| Global coding knowledge | Mavis home | Mavis service | promotion verifier |
| Evaluation fixtures | Mavis evaluator | evaluator only | independent verifier |

The coordinator owns architecture-sensitive seams and exceptions. MiniMax owns
routine launcher, schema, test, and hygiene work once its OpenCode lane passes a
canary. GLM owns routing, quota, diagnostics, and failure-testing work through
ZCode. Local Mavis becomes the ordinary implementation owner only after E0.
