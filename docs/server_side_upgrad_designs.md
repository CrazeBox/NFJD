# Server-Side UPGrad Designs

This note records the three server-side baselines added after the RiverFlow NFJD investigation.

## Motivation

The RiverFlow runs showed that adding `nfjd_common_safe` only weakly improves exact local NFJD. A more direct way to use UPGrad is to move conflict handling to the server, where client updates are combined into one global model update.

## `fedavg_upgrad`

Workflow:

1. The server samples clients and sends the current global model.
2. Each client computes a local per-task gradient matrix on its local data.
3. The server sample-weights and averages those client matrices.
4. The server applies UPGrad to the aggregated task-gradient matrix.
5. The server updates and broadcasts the global model next round.

Interpretation:

This is the closest server-side analogue of FedJD with UPGrad replacing MGDA/min-norm. The objectives are still task losses, not clients.

Expected cost:

Each client uploads `m x d` values, so communication is higher than FedAvg-style single-vector methods.

## `fedclient_upgrad`

Workflow:

1. The server samples clients and sends the current global model.
2. Each client performs ordinary local weighted-sum training.
3. Each client uploads one model delta.
4. The server treats sampled clients as objectives and applies UPGrad to `-delta_theta` rows.
5. The server applies the resulting common client-descent update.

Interpretation:

This changes the MOO objective set from task losses to client local losses. It directly asks for a global update that is compatible across sampled clients.

Expected cost:

Each client uploads one `d`-dimensional vector, matching FedAvg-style communication. This is the more scalable of the two proposed designs.

## `fedmgda_plus`

Workflow:

1. Each client computes local per-objective updates.
2. Each client solves a local MGDA/min-norm problem.
3. Each client uploads one local common direction.
4. The server sample-weights and averages those directions.

Interpretation:

This is the communication-saving FedMGDA+ style baseline: conflict handling happens locally, while the server performs federated averaging of local common descent directions.

## Recommendation

For RiverFlow, prioritize this order:

1. `fedclient_upgrad`: best match to the user's client-loss common descent idea and low communication.
2. `fedavg_upgrad`: cleanest server-side UPGrad over task objectives, but higher upload cost.
3. `fedmgda_plus`: important official-style baseline for interpreting results.

If `fedclient_upgrad` is competitive with `fedavg_pcgrad/cagrad`, the next step is to tune its local epochs and server UPGrad normalization. If it is not competitive, the evidence suggests UPGrad itself is not the right conflict handler for RiverFlow.
