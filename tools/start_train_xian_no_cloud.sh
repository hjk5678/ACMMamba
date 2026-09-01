#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/BUAS/HJK/ACMMamba"
LOG_DIR="${PROJECT_ROOT}/logs/xian_no_cloud"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/xian_no_cloud"
PID_FILE="${LOG_DIR}/train.pid"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PROCESS_COUNT="${NPROC_PER_NODE:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

if [[ -f "${PID_FILE}" ]]; then
    read -r existing_pid < "${PID_FILE}" || true
    if [[ "${existing_pid:-}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Xi'an-no-cloud training is already running with PID ${existing_pid}."
        exit 1
    fi
fi

timestamp="$(date '+%Y%m%d_%H%M%S')"
console_log="${LOG_DIR}/console_${timestamp}.log"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONNOUSERSITE=1

setsid "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${PROCESS_COUNT}" \
    train.py \
    --config configs/train_xian_no_cloud.yaml \
    > "${console_log}" 2>&1 < /dev/null &

training_pid=$!
echo "${training_pid}" > "${PID_FILE}"

echo "Xi'an-no-cloud training started."
echo "PID: ${training_pid}"
echo "Console log: ${console_log}"
echo "Structured log: ${LOG_DIR}/train.log"
echo "Best checkpoint: ${CHECKPOINT_DIR}/xian_no_cloud_best_miou.pt"
echo "Last checkpoint: ${CHECKPOINT_DIR}/xian_no_cloud_last.pt"
