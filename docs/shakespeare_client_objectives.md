# Shakespeare Client-Objective Experiment

This experiment tests the client-level multi-objective setting: each selected Shakespeare speaker/client defines one objective through that client's next-character prediction loss.

## Dataset

The loader supports LEAF Shakespeare JSON, but the recommended path for this project is the custom LEAF-style construction from the Project Gutenberg raw text. The custom path keeps the same client semantics: each Shakespeare speaker/character is one federated client. It does not depend on LEAF's outdated Gutenberg preprocessing scripts.

By default, the experiment resolves `data/shakespeare` relative to the repository root. If LEAF JSON files are empty or missing, `--shakespeare-source auto` falls back to custom raw-text construction.

Custom construction:

1. Read `raw_data.txt` from `data/shakespeare/data/raw_data/raw_data.txt`, or the same layout under the path passed with `--shakespeare-root`.
2. Parse uppercase speaker headings and collect each speaker's dialogue.
3. Generate next-character samples with a sliding window.
4. Select the clients with the most samples by default.
5. Split each selected client into local train/test samples.

The important controls are:

```text
--shakespeare-source custom    Use our raw-text construction directly.
--sequence-length 80           Input context length.
--sequence-stride 5            Sliding-window stride; smaller values create more samples.
--min-samples-per-client 100   Minimum generated samples required before train/test split.
--random-clients               Randomly select eligible clients instead of using the largest clients.
```

Automatic preparation requires `git` and `bash` on the server. The script clones LEAF next to the data directory and runs LEAF's Shakespeare preprocessing:

```bash
python fedjd/experiments/nfjd_phases/run_shakespeare_client_objectives.py \
  --methods fedavg qfedavg fedmgda_plus fedclient_upgrad \
  --num-clients 50 \
  --num-rounds 100 \
  --local-epochs 1 \
  --participation-rate 0.5 \
  --learning-rate 0.01 \
  --shakespeare-source custom \
  --sequence-length 80 \
  --sequence-stride 5 \
  --min-samples-per-client 100 \
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

For the custom construction, you only need the raw text:

```text
data/shakespeare/
  data/
    raw_data/
      raw_data.txt
```

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
  --methods fedavg qfedavg fedmgda_plus fedclient_upgrad \
  --num-clients 50 \
  --num-rounds 100 \
  --local-epochs 1 \
  --participation-rate 0.5 \
  --learning-rate 0.01 \
  --shakespeare-source custom \
  --sequence-stride 5 \
  --min-samples-per-client 100 \
  --shakespeare-root data/shakespeare \
  --no-auto-prepare-shakespeare \
  --output-dir results/shakespeare_client_objectives
```

## Objective Definition

For `K` selected clients, the objective vector has length `K`. Each batch carries a `client_id`; the objective for that client is its cross-entropy next-character loss, while objectives for clients absent from the batch are differentiable zero values. This keeps the existing multi-objective trainer interface while making the target dimension equal to the number of clients.

Final reporting is client-level: mean client accuracy/loss, worst-10% client accuracy, worst-10% client loss, and client-level standard deviations.
