#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_all_checkpoints.sh"

FULL_ROOT="${FULL_ROOT:-/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird_singlenode_2b_best_ckpt_segment_0_plan}"
NO_OPENMM_ROOT="${NO_OPENMM_ROOT:-/project/siyuh/common/chaoyi/workspace/code/SWIMBIRD/swimbird_singlenode_2b_best_ckpt_segment_0_plan_no_openmmreasoner}"

EXPECTED_STEP="${EXPECTED_STEP:-5774}"
POLL_SECONDS="${POLL_SECONDS:-300}"

FULL_EXP_NAME="${FULL_EXP_NAME:-seg0_plan_full_eval}"
NO_OPENMM_EXP_NAME="${NO_OPENMM_EXP_NAME:-seg0_plan_no_openmm_eval}"

FULL_DONE_FILE="${FULL_DONE_FILE:-${FULL_ROOT}/.eval_submitted_step_${EXPECTED_STEP}}"
NO_OPENMM_DONE_FILE="${NO_OPENMM_DONE_FILE:-${NO_OPENMM_ROOT}/.eval_submitted_step_${EXPECTED_STEP}}"

TARGETS=("$@")

usage() {
    cat <<'EOF'
Usage:
  bash VLMEvalKit/slurm_scripts/auto_submit_seg0_plan_eval.sh [benchmark1 benchmark2 ...]

Behavior:
  - Polls the two seg0 plan training roots until both contain checkpoint-${EXPECTED_STEP}
  - Submits eval for both models via submit_all_checkpoints.sh
  - Writes a sentinel file in each root to avoid duplicate submission

Environment overrides:
  EXPECTED_STEP=5774
  POLL_SECONDS=300
  FULL_ROOT=/project/.../swimbird_singlenode_2b_best_ckpt_segment_0_plan
  NO_OPENMM_ROOT=/project/.../swimbird_singlenode_2b_best_ckpt_segment_0_plan_no_openmmreasoner
  FULL_EXP_NAME=seg0_plan_full_eval
  NO_OPENMM_EXP_NAME=seg0_plan_no_openmm_eval

Examples:
  bash VLMEvalKit/slurm_scripts/auto_submit_seg0_plan_eval.sh
  EXPECTED_STEP=3000 bash VLMEvalKit/slurm_scripts/auto_submit_seg0_plan_eval.sh HRBench4K VStarBench
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

submit_eval_once() {
    local exp_name="$1"
    local checkpoint_root="$2"
    local done_file="$3"
    shift 3
    local benchmarks=("$@")

    if [[ -f "${done_file}" ]]; then
        echo "[auto-eval] already submitted: ${checkpoint_root}"
        return 0
    fi

    echo "[auto-eval] submitting eval for ${checkpoint_root}"
    if (( ${#benchmarks[@]} == 0 )); then
        bash "${SUBMIT_SCRIPT}" "${exp_name}" "${checkpoint_root}"
    else
        bash "${SUBMIT_SCRIPT}" "${exp_name}" "${checkpoint_root}" "${benchmarks[@]}"
    fi

    touch "${done_file}"
    echo "[auto-eval] wrote sentinel ${done_file}"
}

checkpoint_ready() {
    local root="$1"
    [[ -d "${root}/checkpoint-${EXPECTED_STEP}" ]]
}

echo "[auto-eval] repo_root=${REPO_ROOT}"
echo "[auto-eval] full_root=${FULL_ROOT}"
echo "[auto-eval] no_openmm_root=${NO_OPENMM_ROOT}"
echo "[auto-eval] expected_step=${EXPECTED_STEP}"
echo "[auto-eval] poll_seconds=${POLL_SECONDS}"
if (( ${#TARGETS[@]} > 0 )); then
    echo "[auto-eval] benchmarks=${TARGETS[*]}"
else
    echo "[auto-eval] benchmarks=ALL_DEFAULTS"
fi

while true; do
    full_ready=0
    no_openmm_ready=0

    if checkpoint_ready "${FULL_ROOT}"; then
        full_ready=1
    fi
    if checkpoint_ready "${NO_OPENMM_ROOT}"; then
        no_openmm_ready=1
    fi

    echo "[auto-eval] $(date '+%Y-%m-%d %H:%M:%S') full_ready=${full_ready} no_openmm_ready=${no_openmm_ready}"

    if [[ "${full_ready}" == "1" && "${no_openmm_ready}" == "1" ]]; then
        submit_eval_once "${FULL_EXP_NAME}" "${FULL_ROOT}" "${FULL_DONE_FILE}" "${TARGETS[@]}"
        submit_eval_once "${NO_OPENMM_EXP_NAME}" "${NO_OPENMM_ROOT}" "${NO_OPENMM_DONE_FILE}" "${TARGETS[@]}"
        echo "[auto-eval] both eval submissions completed"
        exit 0
    fi

    sleep "${POLL_SECONDS}"
done
