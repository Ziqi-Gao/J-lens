#!/usr/bin/env bash
# Download and verify every non-generated input needed by a clean-server run.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(git rev-parse --show-toplevel)}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to a persistent experiment workspace}"
PYTHON="${JLENS_PYTHON:-${CODE_ROOT}/.venv/bin/python}"
MODEL_ID="Qwen/Qwen3.5-4B"
MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATASET_ID="google-research-datasets/go_emotions"
DATASET_REVISION="add492243ff905527e67aeb8b80c082af02207c3"
DATA_SOURCE_ROOT="${RUN_ROOT}/artifacts/source_data/go_emotions_simplified_${DATASET_REVISION}"
DATA_OUTPUT_ROOT="${RUN_ROOT}/artifacts/data/go_emotions_7concept_full_ovr_v2"
RAPTOR_ROOT="${RUN_ROOT}/third_party_external/RAPTOR"
RAPTOR_COMMIT="cf7405899174af39f3970e093e4b86bf0972ff87"

if [[ -n "$(git -C "${CODE_ROOT}" status --short)" ]]; then
  echo "error: bootstrap requires a clean immutable code checkout" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "error: run 'uv sync --extra dev --extra llm' or set JLENS_PYTHON" >&2
  exit 2
fi

export CODE_ROOT RUN_ROOT JLENS_PYTHON="${PYTHON}"
export HF_HOME="${HF_HOME:-${RUN_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${HF_HUB_CACHE}" "${DATA_SOURCE_ROOT}" "${RUN_ROOT}/third_party_external"

"${PYTHON}" - "${HF_HUB_CACHE}" "${DATA_SOURCE_ROOT}" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

cache = Path(sys.argv[1])
destination = Path(sys.argv[2])
model_id = "Qwen/Qwen3.5-4B"
model_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
dataset_id = "google-research-datasets/go_emotions"
dataset_revision = "add492243ff905527e67aeb8b80c082af02207c3"

snapshot_download(
    repo_id=model_id,
    revision=model_revision,
    cache_dir=cache,
)
for split in ("train", "validation", "test"):
    source = hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=dataset_revision,
        filename=f"simplified/{split}-00000-of-00001.parquet",
        cache_dir=cache,
    )
    shutil.copyfile(source, destination / f"{split}.parquet")
PY

if [[ -e "${DATA_OUTPUT_ROOT}" && ! -f "${DATA_OUTPUT_ROOT}/manifest.json" ]]; then
  echo "error: incomplete prepared dataset exists: ${DATA_OUTPUT_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${DATA_OUTPUT_ROOT}/manifest.json" ]]; then
  "${PYTHON}" "${CODE_ROOT}/scripts/prepare_go_emotions.py" \
    --allowlist "${CODE_ROOT}/Concept_intervention/data/go_emotions_7concept_full_allowlist.json" \
    --train "${DATA_SOURCE_ROOT}/train.parquet" \
    --validation "${DATA_SOURCE_ROOT}/validation.parquet" \
    --test "${DATA_SOURCE_ROOT}/test.parquet" \
    --output "${DATA_OUTPUT_ROOT}" \
    --seed 42
fi

if [[ ! -d "${RAPTOR_ROOT}/.git" ]]; then
  git clone https://github.com/Ziqi-Gao/RAPTOR.git "${RAPTOR_ROOT}"
fi
git -C "${RAPTOR_ROOT}" fetch origin "${RAPTOR_COMMIT}"
git -C "${RAPTOR_ROOT}" switch --detach "${RAPTOR_COMMIT}"

JLENS_GIT_COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)"
export JLENS_GIT_COMMIT
source "${CODE_ROOT}/Concept_intervention/scripts/three_method_env.sh"

jlens doctor --require-llm
for config in "${SHARED_CONFIG}" "${J_CONFIG}" "${ITI_CONFIG}" "${RAPTOR_CONFIG}"; do
  jlens config validate "${config}"
done
jlens data validate --config "${SHARED_CONFIG}"
"${PYTHON}" -c "from jlens_workspace.concept_intervention.raptor import verify_raptor_adaptive_epsilon_parity, verify_raptor_checkout; root=verify_raptor_checkout('third_party_external/RAPTOR'); verify_raptor_adaptive_epsilon_parity(root)"

printf '%s\n' \
  "bootstrap_complete=true" \
  "code_commit=${JLENS_GIT_COMMIT}" \
  "run_root=${RUN_ROOT}" \
  "model=${MODEL_ID}@${MODEL_REVISION}" \
  "dataset=${DATASET_ID}@${DATASET_REVISION}" \
  "raptor_commit=${RAPTOR_COMMIT}"
