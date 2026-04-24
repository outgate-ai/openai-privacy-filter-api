#!/usr/bin/env bash
set -euo pipefail

# If we are root (e.g. because a freshly created named volume is mounted at
# /home/opf/.opf with root ownership), fix ownership of the model cache and
# drop privileges to the unprivileged `opf` user. Otherwise run as-is.

if [[ "$(id -u)" -eq 0 ]]; then
    chown -R opf:opf /home/opf/.opf || true
    exec gosu opf opf-api "$@"
fi

exec opf-api "$@"
