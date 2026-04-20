#!/usr/bin/env bash
# run_multiseed_2gpu.sh [env_name] [seeds...]
#
# Trains exp10_multiseed.py's SAC ensemble across TWO GPUs in parallel,
# one process per seed, then runs eval and saves the aggregated results pkl.
#
# Usage (default: ant with all 5 seeds):
#   bash run_multiseed_2gpu.sh
#
# Custom env / seed list:
#   bash run_multiseed_2gpu.sh ant 42 123 7 2024 31
#   bash run_multiseed_2gpu.sh halfcheetah 42 123 7
#
# Seed-to-GPU assignment: round-robin (even index -> GPU 0, odd -> GPU 1).
# Each A100 comfortably hosts 2-3 concurrent SAC processes (each ~3 GB VRAM).
#
# Resumable: cached zips are skipped, so re-running after a partial failure
# is safe.

set -eu

# ------------------------------------------------------------
# Environment install (one-time).
# Host: CUDA 13.0 driver, cuDNN 9.19.0, 2x A100.
# PyTorch cu128 wheels are forward-compatible with CUDA 13.0 driver.
# Set SKIP_INSTALL=1 to skip; set FORCE_INSTALL=1 to always run.
# By default, installs only if torch is missing.
# ------------------------------------------------------------
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  NEED_INSTALL=0
  if [ "${FORCE_INSTALL:-0}" = "1" ]; then
    NEED_INSTALL=1
  else
    python -c "import torch, stable_baselines3, gymnasium, mujoco" 2>/dev/null \
      || NEED_INSTALL=1
  fi
  if [ "${NEED_INSTALL}" = "1" ]; then
    echo ">>> Installing Python deps (CUDA 13.0 / cuDNN 9.19 / 2x A100)..."
    python -m pip install --upgrade pip
    python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
      torch
    python -m pip install \
      stable-baselines3==2.8.0 \
      gymnasium==1.2.3 \
      mujoco==3.1.6 \
      numpy \
      matplotlib
    echo ">>> Install done."
  else
    echo ">>> Deps already present; skipping install. (Use FORCE_INSTALL=1 to reinstall.)"
  fi
fi

ENV_NAME="${1:-ant}"
shift || true
if [ "$#" -eq 0 ]; then
  SEEDS=(42 123 7 2024 31)
else
  SEEDS=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="../cache_exp10/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  run_multiseed_2gpu"
echo "  env:   ${ENV_NAME}"
echo "  seeds: ${SEEDS[*]}"
echo "  GPUs:  0 and 1 (round-robin)"
echo "  logs:  ${LOG_DIR}/"
echo "============================================================"

PIDS=()
i=0
for seed in "${SEEDS[@]}"; do
  gpu=$(( i % 2 ))
  log_path="${LOG_DIR}/train_${ENV_NAME}_seed${seed}_gpu${gpu}.log"
  echo ">>> Launching: seed=${seed} on GPU ${gpu} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" python run_seed.py "${ENV_NAME}" "${seed}" \
      > "${log_path}" 2>&1 &
  PIDS+=($!)
  i=$((i+1))
  # Brief stagger so CUDA init on each GPU doesn't collide.
  sleep 3
done

echo ""
echo "Waiting for ${#PIDS[@]} training processes to finish..."
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "!!! Process ${pid} exited with non-zero status."
    FAILED=$((FAILED+1))
  fi
done

if [ "${FAILED}" -gt 0 ]; then
  echo ""
  echo "!!! ${FAILED} process(es) failed. Check ${LOG_DIR}/ for tracebacks."
  echo "!!! Skipping eval. Re-run this script to resume after fixing."
  exit 1
fi

echo ""
echo "============================================================"
echo "  Training complete. Running eval..."
echo "============================================================"
CUDA_VISIBLE_DEVICES=0 python exp10_multiseed.py eval "${ENV_NAME}" \
    2>&1 | tee "${LOG_DIR}/eval_${ENV_NAME}.log"

PKL_PATH="../cache_exp10/results_multiseed_${ENV_NAME}.pkl"
echo ""
echo "============================================================"
echo "  DONE."
echo "  Result pickle (numbers only, no model weights):"
echo "    ${PKL_PATH}"
echo "  Copy this file back to the main workstation for table/slope update."
echo "============================================================"
