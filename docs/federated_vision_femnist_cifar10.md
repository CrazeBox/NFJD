# Federated Vision FEMNIST+CIFAR10+CelebA Experiment Plan

This experiment family replaces the previous phase-style exploratory runs as the current vision benchmark.

## Datasets

The benchmark uses standard federated vision datasets. All dataset roots are configurable through command-line flags and default to project-root-relative paths, so the experiment can be moved to a server without editing source code. Relative paths are resolved from the repository root, not from the shell's current working directory.

1. FEMNIST uses the LEAF/FEMNIST writer partition. Each writer is one federated client. Experiments sample 50 to 100 writers from the available writer pool. A writer's samples are never assigned to another client.
2. CIFAR10 uses the torchvision CIFAR10 training split. Client partitions are generated with label-skew Dirichlet sampling at alpha=0.1 and alpha=0.5.
3. CelebA uses the aligned CelebA images and binary attribute labels. The runner supports IID and non-IID client splits and treats selected attributes as multi-label binary tasks.

After client assignment, every client is split into local training and local test subsets using the same fixed test fraction. Client data never crosses client boundaries. Final accuracy is reported on client-held-out data to avoid mixing strict writer-level FEMNIST training with a different global EMNIST/byclass evaluation protocol.

## Models

FEMNIST uses a compact convolutional network for 28x28 grayscale character recognition. CIFAR10 currently uses a compact CNN for faster federated sweeps. CelebA uses a shared CNN trunk with per-attribute binary heads.

## Compared Methods

The default comparison set is pure FedAvg, qFedAvg, FedMGDA+, and FedClient-UPGrad. The `fedavg` method uses a dedicated plain FedAvg server that averages local model deltas by sampled-client example counts.

## Metrics

Global performance metrics:

1. Mean client test accuracy: arithmetic mean of local test accuracy over all clients.

Fairness and optimality metrics:

1. Worst-10% client accuracy: average accuracy of the bottom 10% clients.
2. Client accuracy standard deviation: standard deviation over client test accuracies.

Method-level visualization:

1. Method-level mean-vs-worst10 plots compare average client performance and tail-client performance.
2. Method-level Pareto plots mark methods that are not dominated in both mean client accuracy and worst-10% client accuracy.
3. Sorted client accuracy curves show the full client performance distribution.
4. Efficiency trade-off plots compare average round time against mean and worst-10% client accuracy.

Efficiency and convergence metrics:

1. Average round time.
2. Average upload bytes.
3. Single-round aggregation compute overhead, reported as the mean and maximum aggregation/direction-solve time recorded by the server.

## Full-Run Runtime Optimizations

The full FEMNIST+CIFAR10 multi-seed runs use implementation-level optimizations that preserve the reported paper metrics. These changes affect runtime bookkeeping, not the federated optimization objective or final client evaluation protocol.

1. `--eval-interval 0` disables intermediate curve evaluation during training. Final metrics are still computed after training by evaluating the final global model on every client's held-out test split. This preserves `mean_client_test_accuracy`, `worst10_client_accuracy`, `client_accuracy_std`, and `mean_client_test_loss`; it only omits dense convergence curves.
2. `initial_loss` is skipped only for methods that do not use it in their update rule: `fedavg` and `fedclient_upgrad`. qFedAvg keeps `initial_loss` because q-FFL weighting depends on the pre-update client loss. FedMGDA+ also keeps `initial_loss` for conservative bookkeeping of sampled-client objective values. Therefore the final model updates for qFedAvg and FedMGDA+ are unchanged, and FedAvg/FedClient-UPGrad avoid an unnecessary pre-training forward pass.
3. Local model materialization reuses a local model object within each communication round. Before every sampled client update, the local model is reset with a strict `load_state_dict` from a cloned snapshot of the server model at the start of the round. This preserves the federated semantics that every sampled client trains from the same broadcast model `theta_t`; clients do not train sequentially from each other's updated models.

The reusable-local-model path is equivalent to repeatedly deep-copying the server model at the algorithm level:

```text
round t:
  global_state = clone(theta_t)
  local_model = reusable model container
  for each sampled client i:
    local_model.load_state_dict(global_state, strict=True)
    client i trains local_model for E local epochs
    server records delta_i
  server aggregates all delta_i and updates theta_t -> theta_{t+1}
```

These optimizations should not be mixed with algorithmic changes such as AMP, larger batch sizes, or different participation rates unless all methods are rerun under the same setting.

## Output Layout

New results are written under `results/federated_vision/`.

Each run writes:

1. `summary.csv`: one row per dataset/method/seed run.
2. `clients_<exp_id>.csv`: per-client accuracy, loss, sample counts, and Pareto-front flags.
3. `method_mean_vs_worst10_<group>.png`: method-level performance-fairness scatter plot.
4. `method_pareto_mean_worst10_<group>.png`: method-level Pareto plot over mean and worst-10% client accuracy.
5. `sorted_client_accuracy_<group>.png`: sorted client accuracy curves.
6. `method_efficiency_tradeoff_<group>.png`: time/accuracy trade-off plot.

## FEMNIST Data Requirement

For strict writer-level FEMNIST, provide LEAF/FEMNIST JSON data via `--femnist-leaf-root`. The loader expects LEAF-style files containing `users` and `user_data` fields under `data/train`, `data/test`, `train`, `test`, or the given root. If those JSON files are missing, automatic preparation is enabled by default: the script attempts to clone `https://github.com/TalwalkarLab/leaf.git` next to `--femnist-leaf-root` and runs LEAF's FEMNIST non-IID preprocessing script. Disable this with `--no-auto-prepare-femnist`.

The script does not silently substitute EMNIST class shards for FEMNIST clients because that would violate the one-writer-one-client rule. EMNIST/byclass is used only as the auxiliary standard global test set unless `--femnist-use-client-test-union-global` is set.

LEAF FEMNIST labels are normalized into torchvision EMNIST/byclass label order `0-9, A-Z, a-z`; both integer labels in `[0, 61]` and ASCII labels are supported. LEAF image tensors and torchvision EMNIST tensors use the same rotate/flip orientation correction by default. Disable the LEAF-side orientation correction only for debugging with `--no-femnist-orientation-fix`.

## CelebA Data Requirement

CelebA defaults to the project-root-relative root `data/celeba` and can be moved with `--celeba-root <path>`. Do not hard-code machine-specific paths in source files; pass the root path on the command line or set up the same relative directory on the server.

Automatic preparation is enabled by default. The loader first checks whether the required CelebA files already exist under `--celeba-root`; if not, it tries `torchvision.datasets.CelebA(..., download=True)`. This automatic download may fail because CelebA is hosted through Google Drive and can hit quota or confirmation limits. In that case, prepare the dataset manually.

Manual preparation from the official site:

1. Open the official CelebA page: `https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html`.
2. On the page, find the download table and collect these three official parts:
3. In the `Img` section, download `img_align_celeba.zip` from `Align&Cropped Images`.
4. In the `Anno` section, download `list_attr_celeba.txt` from `Attributes Annotations`.
5. In the `Eval` section, download `list_eval_partition.txt` from `Train/Val/Test Partitions`.
6. Extract `img_align_celeba.zip` and place the extracted images plus the two text files under the chosen CelebA root.

Recommended portable layout:

```text
data/celeba/
  list_attr_celeba.txt
  list_eval_partition.txt
  img_align_celeba/
    000001.jpg
    000002.jpg
    ...
```

The loader also accepts the common nested image layout produced by some extractors:

```text
data/celeba/
  list_attr_celeba.txt
  list_eval_partition.txt
  img_align_celeba/
    img_align_celeba/
      000001.jpg
      000002.jpg
      ...
```

Run a smoke test after preparing the files:

```powershell
python fedjd/experiments/federated_vision/run_femnist_cifar10.py --scenarios celeba --methods fedavg --seed 7 --num-rounds 2 --local-epochs 1 --participation-rate 0.2 --learning-rate 0.0001 --celeba-clients 5 --celeba-tasks 4 --celeba-noniid --eval-interval 0 --celeba-root data/celeba --output-dir results/smoke_celeba
```
