# Federated Vision FEMNIST+CIFAR10 Experiment Plan

This experiment family replaces the previous phase-style exploratory runs as the current vision benchmark.

## Datasets

The benchmark uses two standard image classification datasets.

1. FEMNIST uses the LEAF/FEMNIST writer partition. Each writer is one federated client. Experiments sample 50 to 100 writers from the available writer pool. A writer's samples are never assigned to another client.
2. CIFAR10 uses the torchvision CIFAR10 training split. Client partitions are generated with label-skew Dirichlet sampling at alpha=0.1 and alpha=0.5.

After client assignment, every client is split into local training and local test subsets using the same fixed test fraction. Client data never crosses client boundaries. The standard IID CIFAR10 test split and the standard EMNIST/byclass test split for FEMNIST-style 62-way evaluation are used only for global-model evaluation.

## Models

FEMNIST uses a compact convolutional network for 28x28 grayscale character recognition. CIFAR10 uses a ResNet-18 variant adapted to 32x32 images, with a 3x3 stride-1 first convolution and no ImageNet max-pool.

## Compared Methods

The default comparison set is pure FedAvg, qFedAvg, FedMGDA+, and FedClient-UPGrad. The `fedavg` method uses a dedicated plain FedAvg server that averages local model deltas by sampled-client example counts.

## Metrics

Global performance metrics:

1. Mean client test accuracy: arithmetic mean of local test accuracy over all clients.
2. Global IID test accuracy: accuracy of the final global model on the standard IID test split.

Fairness and optimality metrics:

1. Worst-10% client accuracy: average accuracy of the bottom 10% clients.
2. Client accuracy standard deviation: standard deviation over client test accuracies.

Pareto visualization:

1. Two-dimensional projection: client test accuracy versus negative client test loss.
2. Three-dimensional projection: client test accuracy, negative client test loss, and normalized client train-set size.
3. Non-dominated clients are exported and plotted as an approximate Pareto frontier.

Efficiency and convergence metrics:

1. Average round time.
2. Average upload bytes.
3. Single-round aggregation compute overhead, reported as the mean and maximum aggregation/direction-solve time recorded by the server.

## Output Layout

New results are written under `results/federated_vision/`.

Each run writes:

1. `summary.csv`: one row per dataset/method/seed run.
2. `clients_<exp_id>.csv`: per-client accuracy, loss, sample counts, and Pareto-front flags.
3. `pareto2d_<exp_id>.png`: 2D client Pareto projection.
4. `pareto3d_<exp_id>.png`: 3D client Pareto projection.

## FEMNIST Data Requirement

For strict writer-level FEMNIST, provide LEAF/FEMNIST JSON data via `--femnist-leaf-root`. The loader expects LEAF-style files containing `users` and `user_data` fields under `data/train`, `data/test`, `train`, `test`, or the given root. If those JSON files are missing, automatic preparation is enabled by default: the script attempts to clone `https://github.com/TalwalkarLab/leaf.git` next to `--femnist-leaf-root` and runs LEAF's FEMNIST non-IID preprocessing script. Disable this with `--no-auto-prepare-femnist`.

The script does not silently substitute EMNIST class shards for FEMNIST clients because that would violate the one-writer-one-client rule. EMNIST/byclass is used only as the auxiliary standard global test set unless `--femnist-use-client-test-union-global` is set.
