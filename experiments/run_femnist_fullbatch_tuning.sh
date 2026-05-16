#!/usr/bin/env bash
set -euo pipefail

# FEMNIST full-batch tuning script.
# Run from the project root on Linux:
#   bash run_femnist_fullbatch_tuning.sh
#
# Optional environment variables:
#   PYTHON_BIN=python3 bash run_femnist_fullbatch_tuning.sh
#   OUT_ROOT=results/femnist_fullbatch_tune bash run_femnist_fullbatch_tuning.sh
#   FORCE=1 bash run_femnist_fullbatch_tuning.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/femnist_fullbatch_tune}"
FORCE="${FORCE:-0}"

SEEDS=(7 13 42)

COMMON_ARGS=(
  --scenarios femnist
  --femnist-model paper_fedmgda_plus
  --femnist-use-leaf-train-test-split
  --femnist-use-client-test-union-global
  --femnist-leaf-preprocess-kind full
  --femnist-leaf-root data/leaf/data/femnist
  --torchvision-root data/torchvision
  --femnist-clients 3406
  --min-samples-per-client 1
  --num-rounds 1500
  --local-epochs 1
  --local-batch-size 0
  --learning-rate 0.1
  --participation-rate 0.002935995302407516
  --eval-interval 0
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

echo "=== FEMNIST full-batch tuning ==="
echo "Python: $PYTHON_BIN"
echo "Output root: $OUT_ROOT"
echo "Seeds: ${SEEDS[*]}"
echo "Force rerun: $FORCE"

echo "=== FedAvg baseline ==="
for seed in "${SEEDS[@]}"; do
  run_job "$OUT_ROOT/fedavg_seed${seed}" \
    --methods fedavg \
    --seed "$seed"
done

echo "=== qFedAvg tuning ==="
QFEDAVG_CONFIGS=(
  "0.1 0.1 0.1"
  "0.1 1.0 0.1"
  "0.5 0.1 0.1"
  "0.5 1.0 0.1"
  "1.0 0.1 0.1"
  "1.0 1.0 0.1"
)

for cfg in "${QFEDAVG_CONFIGS[@]}"; do
  read -r q lipschitz update_scale <<< "$cfg"
  q_name=$(name_float "$q")
  l_name=$(name_float "$lipschitz")
  scale_name=$(name_float "$update_scale")

  for seed in "${SEEDS[@]}"; do
    run_job "$OUT_ROOT/qfedavg_q${q_name}_L${l_name}_scale${scale_name}_seed${seed}" \
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
  "1.0 none"
  "1.0 0.5"
  "2.0 0.5"
  "1.0 0.2"
  "2.0 0.2"
)

for cfg in "${FEDMGDA_CONFIGS[@]}"; do
  read -r update_scale decay <<< "$cfg"
  scale_name=$(name_float "$update_scale")
  decay_name=$(name_float "$decay")

  for seed in "${SEEDS[@]}"; do
    out_dir="$OUT_ROOT/fedmgda_plus_scale${scale_name}_decay${decay_name}_seed${seed}"
    if [[ "$decay" == "none" ]]; then
      run_job "$out_dir" \
        --methods fedmgda_plus \
        --seed "$seed" \
        --fedmgda-plus-update-scale "$update_scale" \
        --fedmgda-plus-normalize-updates
    else
      run_job "$out_dir" \
        --methods fedmgda_plus \
        --seed "$seed" \
        --fedmgda-plus-update-scale "$update_scale" \
        --fedmgda-plus-update-decay "$decay" \
        --fedmgda-plus-normalize-updates
    fi
  done
done

echo "=== FedClient-UPGrad tuning ==="
UPGRAD_CONFIGS=(
  "1.0 0.2 batched_pgd 250 0.1"
  "1.5 0.2 batched_pgd 250 0.1"
  "2.0 0.2 batched_pgd 250 0.1"
  "1.0 0.5 batched_pgd 250 0.1"
  "1.5 0.5 batched_pgd 250 0.1"
  "2.0 0.5 batched_pgd 100 0.1"
)

for cfg in "${UPGRAD_CONFIGS[@]}"; do
  read -r update_scale decay solver max_iters upgrad_lr <<< "$cfg"
  scale_name=$(name_float "$update_scale")
  decay_name=$(name_float "$decay")
  lr_name=$(name_float "$upgrad_lr")

  for seed in "${SEEDS[@]}"; do
    run_job "$OUT_ROOT/fedclient_upgrad_scale${scale_name}_decay${decay_name}_${solver}_iters${max_iters}_uplr${lr_name}_seed${seed}" \
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

echo "=== Done ==="
echo "Expected completed runs: 54"
echo "Summaries are under: $OUT_ROOT"
