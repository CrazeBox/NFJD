# Shakespeare Client-Objective Experiment

This experiment tests the client-level multi-objective setting: each selected Shakespeare speaker/client defines one objective through that client's next-character prediction loss.

## Dataset

The loader supports LEAF Shakespeare JSON, but the recommended path for this project is the custom LEAF-style construction from the Project Gutenberg raw text. The custom path keeps the same client semantics: each Shakespeare speaker/character is one federated client. It does not depend on LEAF's outdated Gutenberg preprocessing scripts.

By default, the experiment resolves `data/shakespeare` relative to the repository root. If LEAF JSON files are empty or missing, `--shakespeare-source auto` falls back to custom raw-text construction.

Custom construction:

1. Read `raw_data.txt` from `data/shakespeare/data/raw_data/raw_data.txt`, or the same layout under the path passed with `--shakespeare-root`.
2. Parse plays and character lines with the same indentation-based regular expressions used by LEAF's Shakespeare preprocessor, including the `THE COMEDY OF ERRORS` special case.
3. Use LEAF's `play_character` user key convention so same-named speakers in different plays are not merged.
4. Generate next-character samples from each user's text with LEAF's default 80-character context and stride 1.
5. Apply LEAF-style non-IID sampling with `--sample-fraction`; `1.0` keeps the full raw user pool.
6. Remove users below `--min-samples-per-client`.
7. Split each selected client by sample with the FedAvg Shakespeare convention: 80% train and 20% test by default. For Shakespeare, the test segment starts after a `sequence_length - 1` gap to avoid overlapping train/test windows, matching LEAF's `split_data.py` behavior.

The important controls are:

```text
--shakespeare-source custom    Use our raw-text construction directly.
--sequence-length 80           Input context length.
--sequence-stride 1            Sliding-window stride; 1 matches LEAF's next-character construction.
--min-samples-per-client 100   Minimum generated samples required before train/test split.
--client-test-fraction 0.2     Per-client held-out test fraction; 0.2 gives 80/20 train/test.
--max-samples-per-client 0     Keep all generated samples instead of capping each client.
--sample-fraction 1.0          LEAF-style non-IID sample fraction over users/samples.
--client-selection leaf        Choose top, random, or LEAF sampled order from eligible clients.
--vocab-scope all              Build character vocabulary from all parsed users, not only selected clients.
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
  --client-selection leaf \
  --vocab-scope all \
  --shakespeare-root data/shakespeare \
  --output-dir results/shakespeare_client_objectives
```

## Manual Preparation

Use this if automatic setup fails because the server cannot access GitHub or does not have `bash`.

1. Clone LEAF manually on a machine that has network access:

```bash
git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/shakespeare
bash preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8
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
  --client-selection leaf \
  --vocab-scope all \
  --shakespeare-root data/shakespeare \
  --no-auto-prepare-shakespeare \
  --output-dir results/shakespeare_client_objectives
```

## Objective Definition

For `K` selected clients, the objective vector has length `K`. Each batch carries a `client_id`; the objective for that client is its cross-entropy next-character loss, while objectives for clients absent from the batch are differentiable zero values. This keeps the existing multi-objective trainer interface while making the target dimension equal to the number of clients.

Final reporting is client-level: mean client accuracy/loss, worst-10% client accuracy, worst-10% client loss, and client-level standard deviations.
