#!/usr/bin/env bash
set -euo pipefail

# A freshly created named volume mounts root-owned, so fix the model cache
# before dropping privileges.
if [[ "$(id -u)" -eq 0 ]]; then
    chown -R opf:opf /home/opf/.opf || true
    exec gosu opf "$0" "$@"
fi

# A released model is selected with OPF_API_MODEL_ID or OPF_API_MODEL_CHANNEL; the
# fetcher downloads and verifies it into ~/.opf/models/<id>/ and exports
# OPF_API_MODEL_PATH (+ its Viterbi calibration). With neither set it prints
# nothing and the upstream checkpoint is used exactly as before. An explicit
# OPF_API_MODEL_PATH always wins.
if [[ -z "${OPF_API_MODEL_PATH:-}" ]] && [[ -n "${OPF_API_MODEL_ID:-}${OPF_API_MODEL_CHANNEL:-}" ]]; then
    eval "$(python3 /usr/local/bin/opf_models.py fetch --exports)"
fi

# Optional shadow model for live comparison (counts only, see ENVIRONMENT.md).
if [[ -z "${OPF_API_SHADOW_MODEL_PATH:-}" ]] && [[ -n "${OPF_API_SHADOW_MODEL_ID:-}${OPF_API_SHADOW_MODEL_CHANNEL:-}" ]]; then
    eval "$(python3 /usr/local/bin/opf_models.py fetch --shadow --exports)"
fi
exec opf-api "$@"
