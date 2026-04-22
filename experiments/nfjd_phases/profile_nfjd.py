from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedjd.aggregators import MinNormAggregator
from fedjd.core import NFJDClient, FedJDClient
from fedjd.core.scaling import (
    AdaptiveRescaling, LocalMomentum, StochasticGramianSolver, compute_avg_cosine_sim
)
from fedjd.data.synthetic import make_synthetic_federated_regression
from fedjd.problems import multi_objective_regression
from fedjd.models.small_regressor import SmallRegressor

OUTPUT_FILE = Path("results/nfjd_tools/profile_result.txt")

def log(msg):
    print(msg)
    with open(OUTPUT_FILE, "a") as f:
        f.write(msg + "\n")

def profile_nfjd_client():
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    
    log("=" * 80)
    log("NFJD 客户端耗时分析")
    log("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\n设备: {device}")

    torch.manual_seed(7)
    input_dim, m, n_samples = 8, 2, 500
    X = torch.randn(n_samples, input_dim)
    Y = torch.randn(n_samples, m)
    dataset = TensorDataset(X, Y)
    batch_size = 64

    model = SmallRegressor(input_dim=input_dim, output_dim=m).to(device)
    m_param = sum(p.numel() for p in model.parameters())
    log(f"模型参数数量: {m_param:,}")

    def objective_fn(pred, target, _):
        losses = []
        for i in range(m):
            diff = pred[:, i] - target[:, i]
            losses.append((diff ** 2).mean())
        return losses

    log(f"\n{'步骤':<30s} {'耗时(秒)':<12s} {'占比':<10s}")
    log("-" * 60)

    theta_init = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()

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

    for epoch in range(3):
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

            lam = torch.ones(m, device=device) / m
            L_total = sum(lam[i] * losses[i] for i in range(m))
            model.zero_grad(set_to_none=True)
            L_total.backward(retain_graph=True)

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

            qp_t0 = time.time()
            aggregator = MinNormAggregator(max_iters=250, lr=0.1)
            direction = aggregator(jacobian)
            times['minnorm_qp'] += time.time() - qp_t0

            rescale_t0 = time.time()
            n_raw = torch.norm(jacobian.mean(dim=0), p=2)
            n_d = torch.norm(direction, p=2)
            scale = min(n_raw / (n_d + 1e-8), 10.0)
            direction = direction * scale
            times['rescaling'] += time.time() - rescale_t0

            cos_t0 = time.time()
            avg_cosine_sim = compute_avg_cosine_sim(jacobian)
            times['cosine_sim'] += time.time() - cos_t0

            mom_t0 = time.time()
            momentum = LocalMomentum(beta=0.9)
            momentum_dir = momentum.update(direction)
            times['momentum'] += time.time() - mom_t0

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
    log(f"\n总计 {n_batches} 个batch, 3个epochs:")
    log("-" * 60)
    for name, t in times.items():
        pct = t / total_time * 100 if total_time > 0 else 0
        log(f"{name:<30s} {t:<12.4f} {pct:>5.1f}%")
    log(f"{'TOTAL':<30s} {total_time:<12.4f} 100.0%")

    per_step = total_time / n_batches if n_batches > 0 else 0
    log(f"\n平均每步时间: {per_step:.4f}秒")
    log(f"每轮(20步)预计时间: {20 * per_step:.2f}秒")

    return times

def check_sync_points():
    log("\n\n" + "=" * 80)
    log("异步同步点检查")
    log("=" * 80)

    sync_patterns = ['.item()', '.cpu()', '.numpy()', 'print(']
    base_dir = Path(__file__).resolve().parent.parent.parent

    for pattern in sync_patterns:
        log(f"\n搜索 '{pattern}':")
        count = 0
        for py_file in base_dir.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue
            try:
                content = py_file.read_text()
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if pattern in line and not line.strip().startswith('#'):
                        rel_path = py_file.relative_to(base_dir)
                        log(f"  {rel_path}:{i} {line.strip()[:80]}")
                        count += 1
            except Exception:
                pass
        if count == 0:
            log("  无匹配")
        else:
            log(f"  共{count} 处")

if __name__ == "__main__":
    profile_nfjd_client()
    check_sync_points()
    log("\n\n" + "=" * 80)
    log("性能分析完成")
    log("=" * 80)
    log(f"\n结果文件: {OUTPUT_FILE}")

