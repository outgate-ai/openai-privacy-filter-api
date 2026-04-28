# openai-privacy-filter-api

Ollama-compatible HTTP server for the [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) (OPF) model. Drop-in replacement for the `/api/chat` endpoint when you want PII detection instead of text generation.

Because OPF is a token classifier (not a generative LLM), the assistant reply returned from `/api/chat` is a **JSON array of detected PII spans**, not generated text.

[![ci](https://github.com/outgate-ai/openai-privacy-filter-api/actions/workflows/ci.yml/badge.svg)](https://github.com/outgate-ai/openai-privacy-filter-api/actions/workflows/ci.yml)
[![docker](https://github.com/outgate-ai/openai-privacy-filter-api/actions/workflows/docker.yml/badge.svg)](https://github.com/outgate-ai/openai-privacy-filter-api/actions/workflows/docker.yml)

## Quick start

### Docker (recommended)

CPU:
```bash
docker run --rm -p 11435:11435 \
  -v "$HOME/.opf:/home/opf/.opf" \
  ghcr.io/outgate-ai/openai-privacy-filter-api:latest
```

GPU (NVIDIA):
```bash
docker run --rm --gpus all -p 11435:11435 \
  -v "$HOME/.opf:/home/opf/.opf" \
  ghcr.io/outgate-ai/openai-privacy-filter-api:latest-cuda
```

The `-v "$HOME/.opf:/home/opf/.opf"` mount persists the ~3GB model checkpoint across container restarts. Without it, the model re-downloads on every run.

### Python (pip)

```bash
pip install git+https://github.com/outgate-ai/openai-privacy-filter-api.git
opf-api --device cpu      # or --device cuda
```

## Call it

```bash
curl -s http://127.0.0.1:11435/api/chat -H 'content-type: application/json' -d '{
  "model": "openai-privacy-filter",
  "messages": [{"role": "user", "content": "Contact i@izs.me"}],
  "stream": false
}' | jq
```

Response:
```json
{
  "model": "openai-privacy-filter",
  "created_at": "2026-04-24T09:46:46.944230Z",
  "message": {
    "role": "assistant",
    "content": "{\"detections\": [{\"text\": \"i@izs.me\", \"category\": \"personal_information\", \"source_category\": \"private_email\"}]}"
  },
  "done": true,
  "done_reason": "stop",
  "total_duration": 22953715,
  "prompt_eval_count": 17,
  "eval_count": 0
}
```

## Contract

### `POST /api/chat`

- Ollama-compatible request shape: `{ model, messages, stream, options? }`
- `stream: true` → **HTTP 400** (streaming not supported)
- Only the **last `user` message** is analyzed
- `options.*` (e.g. `temperature`, `num_predict`) are silently ignored
- `model` must equal the configured server name (default `openai-privacy-filter`, set via `OPF_API_MODEL_NAME`). Any other value — including an empty string — returns **HTTP 404** with an Ollama-style "model not found" message. This prevents clients from sending requests intended for `llama3` / `glm-4` / etc. and getting silently answered by a PII filter.
- `message.content` is a **JSON-encoded string** (not fenced) containing a wrapped detections object suitable for the [outgate-ai guardrail](https://github.com/outgate-ai/regional-stack/tree/main/services/guardrail) service:

  ```json
  {
    "detections": [
      {"text": "Alice", "category": "personal_information", "source_category": "private_person"},
      {"text": "i@izs.me", "category": "personal_information", "source_category": "private_email"}
    ]
  }
  ```

- Empty case: `{"detections": []}`.
- `category` is one of the guardrail risk categories: `personal_information` or `credentials`.
- `source_category` preserves OPF's native label so downstream consumers can filter or report at finer granularity.

#### OPF → guardrail category mapping

| OPF native `source_category` | Guardrail `category` |
|---|---|
| `private_person`, `private_email`, `private_phone`, `private_address`, `private_date`, `account_number` | `personal_information` |
| `private_url`, `secret` | `credentials` |

Unknown OPF labels (forward-compat, in case the upstream model adds new categories) fall back to `personal_information` and emit a warning log.

### Other endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/tags` | Lists the one advertised model — for Ollama client compatibility. |
| `GET /api/version` | Server version. |
| `GET /health` | Returns engine state: `loading` / `ready` / `error`. |
| `POST /api/generate` | **400** — use `/api/chat`. |
| `POST /api/embeddings` | **400** — not supported. |

### Headers

- `x-request-id` — echoed back if set by the caller, otherwise a new UUID is generated and returned.
- `authorization: Bearer <token>` or `x-api-key: <token>` — **required on all `/api/*` endpoints when `OPF_API_AUTH_TOKEN` is set.** Unauthenticated requests return HTTP 401 with `WWW-Authenticate: Bearer`. `/health` is exempt so orchestrators can still probe it. Leave the env var unset to run open (default).

Example with auth:

```bash
curl -s http://127.0.0.1:11435/api/chat \
  -H "authorization: Bearer $OPF_API_AUTH_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"model":"openai-privacy-filter","messages":[{"role":"user","content":"Contact i@izs.me"}],"stream":false}'
```

## Configuration

All settings are available as CLI flags and environment variables. **CLI > env > default.** See [ENVIRONMENT.md](ENVIRONMENT.md) for the full table.

| Flag | Env var | Default |
|------|---------|---------|
| `--host` | `OPF_API_HOST` | `127.0.0.1` |
| `--port` | `OPF_API_PORT` | `11435` |
| `--device` | `OPF_API_DEVICE` | `cuda` |
| `--model-path` | `OPF_API_MODEL_PATH` | `~/.opf/privacy_filter` |
| `--model-name` | `OPF_API_MODEL_NAME` | `openai-privacy-filter` |
| `--log-level` | `OPF_API_LOG_LEVEL` | `info` |
| `--context-window-length` | `OPF_API_CONTEXT_WINDOW_LENGTH` | `131072` (128k, model max) |
| `--output-mode` | `OPF_API_OUTPUT_MODE` | `typed` |
| `--decode-mode` | `OPF_API_DECODE_MODE` | `viterbi` |
| `--viterbi-calibration-path` | `OPF_API_VITERBI_CALIBRATION_PATH` | unset |
| `--auth-token` | `OPF_API_AUTH_TOKEN` | unset (open) |
| `--normalize-whitespace` / `--no-normalize-whitespace` | `OPF_API_NORMALIZE_WHITESPACE` | `true` |

OPF's recall on multi-line input (emails, OCR'd PDFs, address blocks) improves substantially when newlines/tabs are flattened to single spaces, so the server does this by default. The flattening also defensively decodes literal `\n` / `\r` / `\t` escape sequences that arrive when clients double-encode their JSON body. Disable with `OPF_API_NORMALIZE_WHITESPACE=false` if you need raw input passthrough.

We default `--context-window-length` to the model's 128k maximum. OPF is small (50M active params), so this fits on most hardware, but the buffer is allocated at load time — if you run on a memory-constrained host, lower it. See [ENVIRONMENT.md](ENVIRONMENT.md) for details.

Upstream `OPF_*` tuning variables (`OPF_ALLOW_TF32`, `OPF_MOE_TRITON`, etc.) are passed through untouched — see [ENVIRONMENT.md](ENVIRONMENT.md).

## Development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

The test suite uses a fake engine — it runs in under a second and does not download the model.

## Why port 11435?

Ollama itself defaults to `11434`. We default to `11435` so both can run on the same host. Override with `--port 11434` (and bring down the real Ollama) for a true drop-in replacement.

## License

Apache 2.0 — matches the upstream OpenAI Privacy Filter license.
