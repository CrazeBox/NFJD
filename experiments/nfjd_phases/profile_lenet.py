from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedjd.aggregators import MinNormAggregator
from fedjd.core.scaling import (
    AdaptiveRescaling, LocalMomentum, StochasticGramianSolver, compute_avg_cosine_sim
)
from fedjd.models.lenet_mtl import LeNetMTL

OUTPUT_FILE = Path("results/nfjd_tools/profile_lenet_result.txt")

def log(msg):
    print(msg)
    with open(OUTPUT_FILE, "a") as f:
        f.write(msg + "\n")

def profile_lenet():
    """鐢?LeNetMTL 妯″瀷鍒嗘瀽鎬ц兘鐡堕"""
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    
    log("=" * 80)
    log("NFJD 鎬ц兘鍒嗘瀽 - LeNetMTL 妯″瀷 (Multimnist 瀹為檯浣跨敤)")
    log("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\n璁惧: {device}")

    torch.manual_seed(7)
    # Multimnist: 60000 samples, 10 clients 鈫?6000 per client
    # batch_size=64 鈫?~94 batches per epoch
    # 浣嗕负浜嗗揩閫熸祴璇曪紝鎴戜滑鍙敤 1000 samples
    n_samples = 1000
    input_dim = (1, 36, 36)
    m = 2  # MultiMNIST has 2 tasks
    batch_size = 64
    n_batches_per_epoch = n_samples // batch_size  # ~15 batches
    
    X = torch.randn(n_samples, *input_dim)
    Y = torch.randint(0, 10, (n_samples, m))
    dataset = TensorDataset(X, Y)

    model = LeNetMTL(input_channels=1, num_tasks=2, num_classes=10).to(device)
    m_param = sum(p.numel() for p in model.parameters())
    log(f"妯″瀷鍙傛暟閲? {m_param:,}")

    def objective_fn(pred, target, _):
        losses = []
        for i in range(m):
            losses.append(nn.functional.cross_entropy(pred[:, i], target[:, i].long()))
        return losses

    log(f"\n{'姝ラ':<30s} {'鑰楁椂(绉?':<12s} {'鍗犳瘮':<10s}")
    log("-" * 60)

    times = {
        'data_load': 0.0,
        'forward': 0.0,
        'loss_compute': 0.0,
        'jacobian_backward': 0.0,
        'minnorm_qp': 0.0,
        'rescaling': 0.0,
        'cosine_sim': 0.0,
        'momentum': 0.0,
        'param_update': 0.0,
    }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = 0

    for epoch in range(1):  # 鍙祴1涓猠poch
        for batch_inputs, batch_targets in loader:
            batch_t0 = time.time()
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            times['data_load'] += time.time() - batch_t0

            model.zero_grad(set_to_none=True)
            forward_t0 = time.time()
            predictions = model(batch_inputs)
            times['forward'] += time.time() - forward_t0

            loss_t0 = time.time()
            losses = objective_fn(predictions, batch_targets, batch_inputs)
            times['loss_compute'] += time.time() - loss_t0

            # Jacobian: m娆＄嫭绔嬪弽鍚戜紶鎾?
            jac_t0 = time.time()
            independent_grads = []
            for i in range(m):
                model.zero_grad(set_to_none=True)
                retain = i < m - 1
                losses[i].backward(retain_graph=retain)
                chunks = []
                for p in model.parameters():
                    if p.grad is None:
                        chunks.append(torch.zeros_like(p).reshape(-1))
                    else:
                        chunks.append(p.grad.detach().reshape(-1).clone())
                independent_grads.append(torch.cat(chunks))
            jacobian = torch.stack(independent_grads, dim=0)
            model.zero_grad(set_to_none=True)
            times['jacobian_backward'] += time.time() - jac_t0

            # MinNorm QP姹傝В
            qp_t0 = time.time()
            aggregator = MinNormAggregator(max_iters=250, lr=0.1)
            direction = aggregator(jacobian)
            times['minnorm_qp'] += time.time() - qp_t0

            # AdaptiveRescaling
            rescale_t0 = time.time()
            n_raw = torch.norm(jacobian.mean(dim=0), p=2)
            n_d = torch.norm(direction, p=2)
            scale = min(n_raw / (n_d + 1e-8), 10.0)
            direction = direction * scale
            times['rescaling'] += time.time() - rescale_t0

            # Cosine similarity (CAM)
            cos_t0 = time.time()
            avg_cosine_sim = compute_avg_cosine_sim(jacobian)
            times['cosine_sim'] += time.time() - cos_t0

            # Momentum
            mom_t0 = time.time()
            momentum = LocalMomentum(beta=0.9)
            momentum_dir = momentum.update(direction)
            times['momentum'] += time.time() - mom_t0

            # Parameter update
            upd_t0 = time.time()
            current_flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
            current_flat = current_flat - 0.01 * momentum_dir
            offset = 0
            for p in model.parameters():
                size = p.numel()
                p.data.copy_(current_flat[offset:offset + size].view_as(p))
                offset += size
            times['param_update'] += time.time() - upd_t0

            n_batches += 1

    total_time = sum(times.values())
    log(f"\n鎬昏 {n_batches} 涓猙atch, 1涓猠poch:")
    log("-" * 60)
    for name, t in times.items():
        pct = t / total_time * 100 if total_time > 0 else 0
        log(f"{name:<30s} {t:<12.4f} {pct:>5.1f}%")
    log(f"{'TOTAL':<30s} {total_time:<12.4f} 100.0%")

    per_step = total_time / n_batches if n_batches > 0 else 0
    log(f"\nAvg per-step time: {per_step:.4f}s")
    log(f"Per-round (10 batches x 3 epochs x 5 clients) parallel estimate: {10 * 3 * 5 * per_step:.2f}s")
    log(f"Per-round (10 batches x 3 epochs x 5 clients) serial estimate: {10 * 3 * 5 * per_step:.2f}s")

    return times

if __name__ == "__main__":
    profile_lenet()
    log("\n\n" + "=" * 80)
    log("鎬ц兘鍒嗘瀽瀹屾垚")
    log("=" * 80)
    log(f"\n缁撴灉鏂囦欢: {OUTPUT_FILE}")

