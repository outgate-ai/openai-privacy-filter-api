# Environment variables

`opf-api` is configured via CLI flags and environment variables. **CLI flags take precedence** over env vars, which take precedence over built-in defaults.

## Server-level variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPF_API_HOST` | `127.0.0.1` (bare) / `0.0.0.0` (Docker) | Host interface to bind. |
| `OPF_API_PORT` | `11435` | TCP port to listen on. Ollama itself uses `11434`; the default here avoids a collision. |
| `OPF_API_DEVICE` | `cuda` | Inference device. One of `cpu` or `cuda`. Must be `cpu` if no NVIDIA GPU is available. |
| `OPF_API_MODEL_PATH` | unset | Override the OPF checkpoint directory. If unset, falls back to `OPF_CHECKPOINT`, then `~/.opf/privacy_filter`. Missing checkpoints are auto-downloaded from Hugging Face on first load. |
| `OPF_API_MODEL_NAME` | `openai-privacy-filter` | The model name advertised on `GET /api/tags` **and required** as the `model` field on `POST /api/chat`. Requests with any other `model` value return HTTP 404. Change this only if you want to hide behind a different identifier (e.g. to mimic an existing Ollama tag your client already calls). |
| `OPF_API_LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warning`, or `error`. |

## Input preprocessing

| Variable | Default | Description |
|----------|---------|-------------|
| `OPF_API_NORMALIZE_WHITESPACE` | `true` | When `true` (default), the server collapses every run of whitespace in the user content to a single space — including real newlines/tabs/NBSP **and** literal two-character escape sequences (`\\n`, `\\r`, `\\t`) that arrive when clients double-encode the JSON body. Empirically improves recall on multi-line input (emails with signature blocks, OCR'd PDFs, German address blocks, etc.) where line breaks would otherwise split spans the model would catch on a single line. Set to `false` if you need to pass content through unchanged. Accepts `true`/`false`/`1`/`0`/`yes`/`no`/`on`/`off`. |

## Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `OPF_API_AUTH_TOKEN` | unset (auth disabled) | Shared secret required by all `/api/*` endpoints when set. Clients must send `Authorization: Bearer <token>` or `X-API-Key: <token>`. Requests without a valid token return HTTP 401 with `WWW-Authenticate: Bearer`. `/health` is always open so orchestrators can probe readiness without credentials. Token comparison uses `hmac.compare_digest` (constant-time). Failed attempts are logged at `warning` with the client IP and request ID. |

## Presidio pass (optional second detector)

OPF finds whole spans — a connection string with the password inside it, a signature block with the name and the phone — where pattern matchers return only the fragment they recognise. Presidio adds deterministic, checksum-backed recognizers (IBAN mod-97, card Luhn, SSN) that do not vary with the model's mood. With the pass enabled both run on every request and their detections are merged: when two spans describe the same value, the longer one is kept and the shorter folds into it, contributing only its provenance (`source` becomes `opf+presidio`). Nothing is ever dropped because the other detector missed it.

The pass needs a reachable [Presidio analyzer](https://microsoft.github.io/presidio/) (`ghcr.io/data-privacy-stack/presidio-analyzer`).

| Variable | Default | Description |
|----------|---------|-------------|
| `OPF_API_PRESIDIO_ENABLED` | `false` | Master switch. When `false` the response is byte-for-byte what it was before this feature existed (no `source` field, OPF order). |
| `OPF_API_PRESIDIO_URL` | `http://presidio:3000` | Base URL of the analyzer; the server posts to `<url>/analyze`. |
| `OPF_API_PRESIDIO_LANGUAGE` | `en` | Language code passed to the analyzer. The analyzer must have a model for it. |
| `OPF_API_PRESIDIO_SCORE_THRESHOLD` | `0.5` | Results below this confidence are discarded by the analyzer. |
| `OPF_API_PRESIDIO_ENTITIES` | see below | Comma-separated entity allowlist, or `*` for every entity the analyzer supports. |
| `OPF_API_PRESIDIO_TIMEOUT_MS` | `3000` | Per-request timeout for the analyzer call. |
| `OPF_API_PRESIDIO_FAIL_OPEN` | `true` | `true`: an analyzer error logs a warning and the scan degrades to OPF-only. `false`: the request fails with HTTP 503. |
| `OPF_API_PRECISION_FILTERS` | `true` | Drops detections whose shape alone does not support the claim. Two rules: a `secret` that is a single all-alphabetic token of 12 characters or fewer is treated as a word, not a credential (a company or product name in a path, a hostname, an image reference); and a hit under a noisy category is dropped when a second detector ran and did not corroborate it. Set to `false` to emit every detection the model produces. |
| `OPF_API_NOISY_CATEGORIES` | `private_date,account_number` | Comma-separated OPF native categories subject to the corroboration rule above. These fire on machine output — ISO timestamps, invoice numbers, MAC addresses, any long digit run — while the values that matter under them (card, IBAN, SSN) are also matched by Presidio's checksum-backed recognizers. The rule is inactive when the Presidio pass is off, so a single-detector deployment keeps its recall unchanged. |

The default allowlist is the set of recognizers that hold up on ordinary engineering prose:

```
CREDIT_CARD, CRYPTO, EMAIL_ADDRESS, IBAN_CODE, IP_ADDRESS, MAC_ADDRESS,
MEDICAL_LICENSE, PHONE_NUMBER, UK_NHS, US_BANK_NUMBER, US_DRIVER_LICENSE,
US_ITIN, US_PASSPORT, US_SSN
```

`PERSON`, `LOCATION`, `NRP`, `DATE_TIME` and `URL` are deliberately left out: OPF already covers people and addresses, and on normal text those recognizers fire on "Thursday", "Hamburg" and every `https://` link. Add them explicitly if you want them.

Each detection carries `source` (`opf`, `presidio`, or `opf+presidio`) and `source_category` — the native OPF label or the Presidio entity type — so you can tell the detectors apart downstream.

## Model-behavior variables

These control what the model outputs and how long an input it can process.
Every setting is fixed at **server startup** — there are no per-request overrides.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPF_API_CONTEXT_WINDOW_LENGTH` | `131072` | Maximum input length in tokens. **We default to the model's advertised maximum (128k).** See the note below on memory cost. Valid range: `1` to `131072`. |
| `OPF_API_OUTPUT_MODE` | `typed` | `typed` returns one of the 8 OPF categories per span (`private_email`, `private_person`, …). `redacted` collapses every span to a generic `redacted` label — use when you don't care about category, just "is this sensitive." |
| `OPF_API_DECODE_MODE` | `viterbi` | How per-token logits become spans. `viterbi` enforces BIOES transition constraints and gives coherent span boundaries (recommended). `argmax` is a per-token greedy decode — faster on very long inputs but produces fragmented boundaries. |
| `OPF_API_VITERBI_CALIBRATION_PATH` | unset | Path to a Viterbi calibration JSON artifact that shifts the precision/recall operating point. Favoring span-entry/continuation → higher recall (redacts more, more false positives). Favoring background-persistence → higher precision (redacts less, more false negatives). Only used when `OPF_API_DECODE_MODE=viterbi`. |

### A note on `OPF_API_CONTEXT_WINDOW_LENGTH`

OPF is an **encoder-style token classifier** with a 128k-token ceiling. Because it's a small model (1.5B total / 50M active), running at 128k is feasible even on modest hardware — our default value is `131072`.

That said, longer contexts cost memory proportional to the sequence length:
- **CPU**: a ~128k-token request may use several GB of extra RAM for activations.
- **GPU**: fits comfortably on a 16 GB+ card; may be tight on smaller GPUs at fp32.
- **Memory is allocated at model load**, sized for the worst case. A smaller `OPF_API_CONTEXT_WINDOW_LENGTH` reduces the server's resident footprint even if you never send long inputs.

If you run on a memory-constrained host, set it lower — `16384` or `32768` is still much larger than typical text, and startup plus per-request memory use will drop accordingly. You will still get correct behavior on inputs up to whatever value you set; inputs longer than the configured window are truncated by OPF.

## Upstream OPF variables (passed through)

The underlying [`opf`](https://github.com/openai/privacy-filter) package honors these. Set them if you need to tune inference behavior; defaults are fine for most users.

| Variable | Purpose |
|----------|---------|
| `OPF_CHECKPOINT` | Explicit checkpoint directory. Lower precedence than `OPF_API_MODEL_PATH`. |
| `OPF_MOE_TRITON` | `1` to force Triton MoE kernels, `0` to disable. Default is on for CUDA, off for CPU. |
| `OPF_MOE_FUSED_SWIGLU_W2` | `1` to fuse SwiGLU with MLP2 (default), `0` to disable. |
| `OPF_ALLOW_TF32` | `1` to allow TF32 matmul on Ampere+ GPUs (faster, slightly lower precision). |
| `OPF_ATTN_LOW_PRECISION` | `1` to run attention scoring in bf16 instead of fp32. |
| `OPF_EXPERTS_PER_TOKEN` | Override the number of MoE experts routed per token (default from checkpoint, typically 4). |

## Precedence summary

```
CLI flag  >  OPF_API_* env var  >  built-in default
```

For upstream `OPF_*` variables (not `OPF_API_*`), there is no CLI equivalent — they're read directly by the `opf` library.

## Examples

Bare:

```bash
OPF_API_DEVICE=cpu OPF_API_PORT=8080 opf-api
```

Docker (CPU):

```bash
docker run --rm -p 11435:11435 \
  -v "$HOME/.opf:/home/opf/.opf" \
  ghcr.io/outgate-ai/openai-privacy-filter-api:latest
```

With the Presidio pass (compose sketch — analyzer on the same network):

```bash
docker run --rm --gpus all -p 11435:11435 \
  -e OPF_API_PRESIDIO_ENABLED=true \
  -e OPF_API_PRESIDIO_URL=http://presidio:3000 \
  ghcr.io/outgate-ai/openai-privacy-filter-api:latest-cuda
```

Docker (CUDA):

```bash
docker run --rm --gpus all -p 11435:11435 \
  -v "$HOME/.opf:/home/opf/.opf" \
  -e OPF_ALLOW_TF32=1 \
  ghcr.io/outgate-ai/openai-privacy-filter-api:latest-cuda
```

The `~/.opf` mount is strongly recommended in Docker — otherwise the ~3GB checkpoint re-downloads every time the container is recreated.
