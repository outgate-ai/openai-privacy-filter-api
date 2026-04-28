# CPU image — small, python-slim base, CPU-only torch.
#
# Layer ordering matches Dockerfile.cuda: heavy, rarely-changing layers
# (torch, OS packages, user setup) come first; the application code is
# last. A typical code-change release re-pushes only the small app layer.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPF_API_HOST=0.0.0.0 \
    OPF_API_PORT=11435 \
    OPF_API_DEVICE=cpu

# --- Layer 1: OS packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl gosu \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 2: torch + CPU wheels (~250 MB).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"

# --- Layer 3: unprivileged user + entrypoint.
RUN useradd --create-home --uid 1000 opf \
    && mkdir -p /home/opf/.opf \
    && chown -R opf:opf /home/opf
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# --- Layer 4: project metadata.
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./

# --- Layer 5: application source + project pip install. Only this
# layer re-pushes on a typical code-change release.
COPY src ./src
RUN pip install .

WORKDIR /home/opf

EXPOSE 11435

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${OPF_API_PORT}/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
