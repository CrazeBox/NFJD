#!/usr/bin/env bash
set -euo pipefail

# CIFAR-10 b=10 tuning script.
# Runs p=0.1 first, then p=0.2.
# Run from the project root on Linux:
#   bash run_cifar_b10_p01_p02_tuning.sh
#
# Optional environment variables:
#   PYTHON_BIN=python3 bash run_cifar_b10_p01_p02_tuning.sh
#   OUT_ROOT=results/cifar10_b10_tune bash run_cifar_b10_p01_p02_tuning.sh
#   FORCE=1 bash run_cifar_b10_p01_p02_tuning.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/cifar10_b10_tune}"
FORCE="${FORCE:-0}"

SEEDS=(7 13 42)

COMMON_ARGS=(
  --scenarios cifar10_fedmgda_paper
  --cifar-model paper_fedmgda_plus
  --cifar-clients 100
  --min-samples-per-client 1
  --eval-interval 0
  --fedmgda-paper-cifar10-preset
  --cifar-paper-batch small
)

run_job() {
  local out_dir="$1"
  shift

  if [[ "$FORCE" != "1" && -f "$out_dir/summary.csv" ]]; then
    echo "[skip] $out_dir already has summary.csv"
    return 0
  fi

  mkdir -p "$out_dir"
  echo "[run] $out_dir"
  "$PYTHON_BIN" -m fedjd.experiments.federated_vision.run_femnist_cifar10 \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --output-dir "$out_dir"
}

name_float() {
  local value="$1"
  echo "${value//./p}"
}

run_stage() {
  local tag="$1"
  local clients_per_round="$2"
  local stage_root="$OUT_ROOT/$tag"

  echo "=== CIFAR-10 b=10 ${tag} ==="
  echo "Python: $PYTHON_BIN"
  echo "Output root: $stage_root"
  echo "Seeds: ${SEEDS[*]}"
  echo "Clients per round: $clients_per_round"
  echo "Force rerun: $FORCE"

  local stage_args=(
    --cifar-paper-clients-per-round "$clients_per_round"
  )

  echo "=== FedAvg baseline ==="
  for seed in "${SEEDS[@]}"; do
    run_job "$stage_root/fedavg_seed${seed}" \
      "${stage_args[@]}" \
      --methods fedavg \
      --seed "$seed"
  done

  echo "=== qFedAvg tuning ==="
  QFEDAVG_CONFIGS=(
    "0.5 1.0 0.1"
    "0.1 0.1 0.01"
    "0.5 0.1 0.01"
    "1.0 0.1 0.01"
    "0.1 1.0 0.01"
    "0.5 1.0 0.01"
    "1.0 1.0 0.01"
  )

  for cfg in "${QFEDAVG_CONFIGS[@]}"; do
    read -r q lipschitz update_scale <<< "$cfg"
    q_name=$(name_float "$q")
    l_name=$(name_float "$lipschitz")
    scale_name=$(name_float "$update_scale")

    for seed in "${SEEDS[@]}"; do
      run_job "$stage_root/qfedavg_q${q_name}_L${l_name}_scale${scale_name}_seed${seed}" \
        "${stage_args[@]}" \
        --methods qfedavg \
        --seed "$seed" \
        --qfedavg-q "$q" \
        --qfedavg-lipschitz "$lipschitz" \
        --qfedavg-update-scale "$update_scale" \
        --qfedavg-mode official_delta
    done
  done

  echo "=== FedMGDA+ tuning ==="
  FEDMGDA_CONFIGS=(
    "1.0 0 true"
    "1.0 0.1 true"
    "1.5 0.1 true"
    "1.0 0.025 true"
    "1.5 0.025 true"
  )

  for cfg in "${FEDMGDA_CONFIGS[@]}"; do
    read -r update_scale decay normalize_updates <<< "$cfg"
    scale_name=$(name_float "$update_scale")
    decay_name=$(name_float "$decay")

    for seed in "${SEEDS[@]}"; do
      run_job "$stage_root/fedmgda_plus_scale${scale_name}_decay${decay_name}_seed${seed}" \
        "${stage_args[@]}" \
        --methods fedmgda_plus \
        --seed "$seed" \
        --fedmgda-plus-update-scale "$update_scale" \
        --fedmgda-plus-update-decay "$decay" \
        --fedmgda-plus-normalize-updates
    done
  done

  echo "=== FedClient-UPGrad tuning ==="
  UPGRAD_CONFIGS=(
    "1.0 0.1 batched_pgd 250 0.1"
    "1.5 0.1 batched_pgd 250 0.1"
    "2.0 0.1 batched_pgd 250 0.1"
    "1.0 0.025 batched_pgd 250 0.1"
    "1.5 0.025 batched_pgd 250 0.1"
    "2.0 0.025 batched_pgd 250 0.1"
  )

  for cfg in "${UPGRAD_CONFIGS[@]}"; do
    read -r update_scale decay solver max_iters upgrad_lr <<< "$cfg"
    scale_name=$(name_float "$update_scale")
    decay_name=$(name_float "$decay")
    lr_name=$(name_float "$upgrad_lr")

    for seed in "${SEEDS[@]}"; do
      run_job "$stage_root/fedclient_upgrad_scale${scale_name}_decay${decay_name}_${solver}_iters${max_iters}_uplr${lr_name}_seed${seed}" \
        "${stage_args[@]}" \
        --methods fedclient_upgrad \
        --seed "$seed" \
        --fedclient-update-scale "$update_scale" \
        --fedclient-update-decay "$decay" \
        --fedclient-normalize-updates \
        --fedclient-upgrad-solver "$solver" \
        --fedclient-upgrad-max-iters "$max_iters" \
        --fedclient-upgrad-lr "$upgrad_lr"
    done
  done
}

run_stage "p01" 10
run_stage "p02" 20

echo "=== Done ==="
echo "Expected completed runs: 54 (p=0.1) + 54 (p=0.2) = 108"
echo "Summaries are under: $OUT_ROOT"
