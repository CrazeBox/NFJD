# Sent140 Client-Objective Experiment

This experiment adds a natural user-level federated benchmark for the client-level multi-objective setting.

## Dataset

Sent140 follows the LEAF federated benchmark convention:

1. Each client is a Twitter user.
2. Each example is a tweet.
3. The task is binary sentiment classification.
4. Client heterogeneity is natural user-level heterogeneity, not artificial Dirichlet label skew.

The loader expects LEAF-style JSON files with `users` and `user_data` fields under one of these layouts:

```text
data/sent140/
  data/
    train/*.json
    test/*.json
```

or:

```text
data/sent140/
  train/*.json
  test/*.json
```

If only one directory of JSON files is present, the loader falls back to a per-client train/test split controlled by `--client-test-fraction`.

## LEAF Preparation Notes

LEAF includes Sent140 preprocessing scripts, but the raw Sentiment140 CSV is not always downloaded automatically in every environment. If automatic setup fails, prepare Sent140 with LEAF separately and copy the generated JSON files into `data/sent140`.

Automatic setup is enabled by default:

```bash
python fedjd/experiments/nfjd_phases/run_sent140_client_objectives.py \
  --methods fedavg fedclient_upgrad \
  --num-clients 20 \
  --num-rounds 5 \
  --local-epochs 1 \
  --participation-rate 0.5 \
  --learning-rate 0.01 \
  --min-samples-per-client 10 \
  --sent140-root data/sent140 \
  --output-dir results/smoke_sent140
```

Disable automatic setup if JSON files are already prepared:

```bash
python fedjd/experiments/nfjd_phases/run_sent140_client_objectives.py \
  --methods fedavg fedclient_upgrad \
  --num-clients 100 \
  --num-rounds 100 \
  --local-epochs 1 \
  --participation-rate 0.5 \
  --learning-rate 0.01 \
  --min-samples-per-client 20 \
  --sent140-root data/sent140 \
  --no-auto-prepare-sent140 \
  --output-dir results/sent140_pilot_seed7
```

## Model

The default model is `MeanPooledTextClassifier`:

1. Token embedding.
2. Masked mean pooling over tweet tokens.
3. Small MLP binary classifier.

This model is intentionally lightweight so that Sent140 can be used as a pilot client-level benchmark without introducing a heavy language-model training pipeline.

## Recommended Pilot

Run a small pilot before committing to full three-seed experiments:

```bash
for method in fedavg fedclient_upgrad; do
  python fedjd/experiments/nfjd_phases/run_sent140_client_objectives.py \
    --methods "$method" \
    --seed 7 \
    --num-clients 100 \
    --num-rounds 100 \
    --local-epochs 1 \
    --participation-rate 0.5 \
    --learning-rate 0.01 \
    --min-samples-per-client 20 \
    --sent140-root data/sent140 \
    --no-auto-prepare-sent140 \
    --eval-interval 0 \
    --output-dir "results/sent140_pilot_seed7_${method}"
done
```

Inspect mean accuracy, worst-10% accuracy, client accuracy standard deviation, and loss. Expand to four methods and three seeds only if FedClient-UPGrad shows a useful tail-client or dispersion signal.
