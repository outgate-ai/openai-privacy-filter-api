# Environment variables

`opf-api` is configured via CLI flags and environment variables. **CLI flags take precedence** over env vars, which take precedence over built-in defaults.

## Server-level variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPF_API_HOST` | `127.0.0.1` (bare) / `0.0.0.0` (Docker) | Host interface to bind. |
| `OPF_API_PORT` | `11435` | TCP port to listen on. Ollama itself uses `11434`; the default here avoids a collision. |
| `OPF_API_DEVICE` | `cuda` | Inference device. One of `cpu` or `cuda`. Must be `cpu` if no NVIDIA GPU is available. |
| `OPF_API_MODEL_PATH` | unset | Override the OPF checkpoint directory. If unset, falls back to `OPF_CHECKPOINT`, then `~/.opf/privacy_filter`. Missing checkpoints are auto-downloaded from Hugging Face on first load. |
| `OPF_API_MODEL_NAME` | `openai-privacy-filter` | The model name advertised on `GET /api/tags`. Clients can still send any `model` field in `/api/chat` and it will be echoed back. |
| `OPF_API_LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warning`, or `error`. |

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
