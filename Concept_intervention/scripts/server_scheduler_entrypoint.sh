#!/usr/bin/env bash
# Stable foreground entrypoint for a reviewed ServerScheduler registration.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly CODE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly PYTHON="/scr/del6500/J-lens/envs/three-method/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "error: registered J-lens Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m jlens_workspace.server_scheduler_adapter "$@"
