# Shakespeare Client-Objective Experiment

This experiment tests the client-level multi-objective setting: each selected Shakespeare speaker/client defines one objective through that client's next-character prediction loss.

## Dataset

The loader uses the LEAF Shakespeare format. By default, the experiment resolves `data/shakespeare` relative to the repository root and tries automatic preparation if the LEAF JSON files are missing.

Automatic preparation requires `git` and `bash` on the server. The script clones LEAF next to the data directory and runs LEAF's Shakespeare preprocessing:

```bash
python fedjd/experiments/nfjd_phases/run_shakespeare_client_objectives.py \
  --methods nfjd fedavg qfedavg fedmgda_plus fedclient_upgrad \
  --num-clients 20 \
  --num-rounds 100 \
  --local-epochs 1 \
  --participation-rate 0.5 \
  --learning-rate 0.01 \
  --shakespeare-root data/shakespeare \
  --output-dir results/shakespeare_client_objectives
```

## Manual Preparation

Use this if automatic setup fails because the server cannot access GitHub or does not have `bash`.

1. Clone LEAF manually on a machine that has network access:

```bash
git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/shakespeare
bash preprocess.sh -s niid --sf 1.0 -k 0 -t sample
```

2. Copy the generated Shakespeare directory to the server, or copy the JSON files into the project data directory.

Accepted layouts include:

```text
data/shakespeare/
  data/
    train/
      *.json
    test/
      *.json
```

or:

```text
data/shakespeare/
  train/
    *.json
  test/
    *.json
```

or directly:

```text
data/shakespeare/
  *.json
```

3. Run with automatic preparation disabled:

```bash
python fedjd/experiments/nfjd_phases/run_shakespeare_client_objectives.py \
  --methods nfjd fedavg qfedavg fedmgda_plus fedclient_upgrad \
  --num-clients 20 \
  --num-rounds 100 \
  --local-epochs 1 \
  --participation-rate 0.5 \
  --learning-rate 0.01 \
  --shakespeare-root data/shakespeare \
  --no-auto-prepare-shakespeare \
  --output-dir results/shakespeare_client_objectives
```

## Objective Definition

For `K` selected clients, the objective vector has length `K`. Each batch carries a `client_id`; the objective for that client is its cross-entropy next-character loss, while objectives for clients absent from the batch are differentiable zero values. This keeps the existing multi-objective trainer interface while making the target dimension equal to the number of clients.

Final reporting is client-level: mean client accuracy/loss, worst-10% client accuracy, worst-10% client loss, and client-level standard deviations.
