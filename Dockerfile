# CPU image — small, python-slim base, CPU-only torch.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPF_API_HOST=0.0.0.0 \
    OPF_API_PORT=11435 \
    OPF_API_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install CPU-only torch first from the official index to avoid pulling CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
    && pip install .

RUN useradd --create-home --uid 1000 opf \
    && mkdir -p /home/opf/.opf \
    && chown -R opf:opf /home/opf

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /home/opf

EXPOSE 11435

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${OPF_API_PORT}/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
