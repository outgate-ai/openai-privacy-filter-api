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

Docker (CUDA):

```bash
docker run --rm --gpus all -p 11435:11435 \
  -v "$HOME/.opf:/home/opf/.opf" \
  -e OPF_ALLOW_TF32=1 \
  ghcr.io/outgate-ai/openai-privacy-filter-api:latest-cuda
```

The `~/.opf` mount is strongly recommended in Docker — otherwise the ~3GB checkpoint re-downloads every time the container is recreated.
