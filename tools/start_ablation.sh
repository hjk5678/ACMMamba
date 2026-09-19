#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/BUAS/HJK/ACMMamba"
CONFIG_PATH="${1:-}"
if [[ -z "${CONFIG_PATH}" ]]; then
    echo "Usage: bash tools/start_ablation.sh <config.yaml>"
    exit 2
fi
if [[ "${CONFIG_PATH}" != /* ]]; then
    CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config does not exist: ${CONFIG_PATH}"
    exit 2
fi

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PROCESS_COUNT="${NPROC_PER_NODE:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="$("${PYTHON_BIN}" -c \
    'import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["log_dir"])' \
    "${CONFIG_PATH}")"
CHECKPOINT_DIR="$("${PYTHON_BIN}" -c \
    'import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["checkpoint_dir"])' \
    "${CONFIG_PATH}")"
RUN_NAME="$(basename "${CONFIG_PATH}" .yaml)"
PID_FILE="${LOG_DIR}/train.pid"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"
if [[ -f "${PID_FILE}" ]]; then
    read -r existing_pid < "${PID_FILE}" || true
    if [[ "${existing_pid:-}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "${RUN_NAME} is already running with PID ${existing_pid}."
        exit 1
    fi
fi

timestamp="$(date '+%Y%m%d_%H%M%S')"
console_log="${LOG_DIR}/console_${timestamp}.log"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

setsid "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${PROCESS_COUNT}" \
    train.py \
    --config "${CONFIG_PATH}" \
    > "${console_log}" 2>&1 < /dev/null &

training_pid=$!
echo "${training_pid}" > "${PID_FILE}"
echo "Ablation started: ${RUN_NAME}"
echo "PID: ${training_pid}"
echo "Console log: ${console_log}"
echo "Structured log: ${LOG_DIR}/train.log"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
