# openai-privacy-filter-api

Ollama-compatible HTTP server for the [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) (OPF) model. Drop-in replacement for the `/api/chat` endpoint when you want PII detection instead of text generation.

Because OPF is a token classifier, the assistant reply returned from `/api/chat` is a **JSON array of detected PII spans**, not generated text.

## Install

```bash
pip install git+https://github.com/outgate-ai/openai-privacy-filter-api.git
```

This also installs the upstream `opf` package from GitHub.

## Run

```bash
opf-api --host 127.0.0.1 --port 11435 --device cuda
```

Flags:

- `--host` (default `127.0.0.1`)
- `--port` (default `11435` — Ollama itself uses `11434`)
- `--device` `cuda` (default) or `cpu`
- `--model-path` override OPF checkpoint dir (default: `$OPF_CHECKPOINT` or `~/.opf/privacy_filter`)

On first run, if the checkpoint is missing the server logs a clear "downloading from Hugging Face" message and pulls it automatically (via the upstream `opf` package).

## Endpoints

### `POST /api/chat`

Request (Ollama-compatible):

```json
{
  "model": "openai-privacy-filter",
  "messages": [
    { "role": "user", "content": "My name is Alice and my email is alice@example.com" }
  ],
  "stream": false
}
```

- `stream: true` is **rejected with HTTP 400**.
- Only the **last `user` message** is analyzed.
- `options.*` (e.g. `temperature`, `num_predict`) are silently ignored.
- `model` is echoed back in the response; any value is accepted.

Response:

```json
{
  "model": "openai-privacy-filter",
  "created_at": "2026-04-24T09:46:46.944230Z",
  "message": {
    "role": "assistant",
    "content": "[{\"text\": \"Alice\", \"category\": \"private_person\"}, {\"text\": \"alice@example.com\", \"category\": \"private_email\"}]"
  },
  "done": true,
  "done_reason": "stop",
  "total_duration": 22953715,
  "prompt_eval_count": 54,
  "eval_count": 0
}
```

The `content` field is a JSON-encoded string containing an array of detected spans. Categories are passed through from OPF unchanged:

- `private_person`
- `private_email`
- `private_phone`
- `private_address`
- `private_url`
- `private_date`
- `account_number`
- `secret`

When no PII is detected, `content` is `"[]"`.

### `GET /api/tags`

Lists the one available model (`openai-privacy-filter`). Implemented so clients that probe the Ollama server don't break.

### `GET /api/version`

Returns this server's version.

### `GET /health`

Returns `{"status": "ready" | "loading" | "error", "error": ...}`.

### Unsupported

- `POST /api/generate` → 400
- `POST /api/embeddings` → 400

## Why port `11435`?

Ollama itself defaults to `11434`. We use `11435` so both can run side by side. Change with `--port 11434` if you want a true drop-in replacement.

## License

Apache 2.0 — matches the upstream OpenAI Privacy Filter license.
