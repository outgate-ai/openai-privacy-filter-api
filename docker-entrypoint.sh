#!/usr/bin/env bash
set -euo pipefail

# A freshly created named volume mounts root-owned, so fix the model cache
# before dropping privileges.
if [[ "$(id -u)" -eq 0 ]]; then
    chown -R opf:opf /home/opf/.opf || true
    exec gosu opf opf-api "$@"
fi

exec opf-api "$@"
