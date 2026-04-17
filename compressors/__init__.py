from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class JacobianCompressor(ABC):
    @abstractmethod
    def compress(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @abstractmethod
    def decompress(self, compressed: torch.Tensor, meta: dict) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class NoCompressor(JacobianCompressor):
    @property
    def name(self) -> str:
        return "none"

    def compress(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return jacobian, {"shape": tuple(jacobian.shape)}

    def decompress(self, compressed: torch.Tensor, meta: dict) -> torch.Tensor:
        return compressed


class Float16Compressor(JacobianCompressor):
    @property
    def name(self) -> str:
        return "float16"

    def compress(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, dict]:
        compressed = jacobian.to(torch.float16)
        meta = {"shape": tuple(jacobian.shape), "original_dtype": str(jacobian.dtype)}
        return compressed, meta

    def decompress(self, compressed: torch.Tensor, meta: dict) -> torch.Tensor:
        return compressed.to(torch.float32)


class TopKCompressor(JacobianCompressor):
    def __init__(self, k_ratio: float = 0.1) -> None:
        self.k_ratio = k_ratio

    @property
    def name(self) -> str:
        return f"topk_{self.k_ratio}"

    def compress(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, dict]:
        flat = jacobian.reshape(-1)
        k = max(1, int(flat.numel() * self.k_ratio))
        topk_vals, topk_indices = torch.topk(flat.abs(), k)
        compressed_vals = flat[topk_indices]
        meta = {
            "shape": tuple(jacobian.shape),
            "indices": topk_indices,
            "total_elements": flat.numel(),
        }
        return compressed_vals, meta

    def decompress(self, compressed: torch.Tensor, meta: dict) -> torch.Tensor:
        shape = meta["shape"]
        indices = meta["indices"]
        result = torch.zeros(meta["total_elements"], dtype=compressed.dtype, device=compressed.device)
        result.scatter_(0, indices, compressed)
        return result.reshape(shape)


class RowTopKCompressor(JacobianCompressor):
    def __init__(self, k_ratio: float = 0.1) -> None:
        self.k_ratio = k_ratio

    @property
    def name(self) -> str:
        return f"rowtopk_{self.k_ratio}"

    def compress(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, dict]:
        m, d = jacobian.shape
        k = max(1, int(d * self.k_ratio))
        all_vals = []
        all_indices = []
        for row_idx in range(m):
            row = jacobian[row_idx]
            _, topk_indices = torch.topk(row.abs(), k)
            all_vals.append(row[topk_indices])
            all_indices.append(topk_indices)
        compressed = torch.stack(all_vals)
        meta = {
            "shape": tuple(jacobian.shape),
            "indices": torch.stack(all_indices),
            "k": k,
        }
        return compressed, meta

    def decompress(self, compressed: torch.Tensor, meta: dict) -> torch.Tensor:
        shape = meta["shape"]
        indices = meta["indices"]
        m, d = shape
        result = torch.zeros(m, d, dtype=compressed.dtype, device=compressed.device)
        for row_idx in range(m):
            result[row_idx].scatter_(0, indices[row_idx], compressed[row_idx])
        return result


class LowRankCompressor(JacobianCompressor):
    def __init__(self, rank: int = 2) -> None:
        self.rank = rank

    @property
    def name(self) -> str:
        return f"lowrank_r{self.rank}"

    def compress(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, dict]:
        U, S, Vh = torch.linalg.svd(jacobian, full_matrices=False)
        r = min(self.rank, U.shape[1])
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        compressed = torch.cat([U_r.reshape(-1), S_r, Vh_r.reshape(-1)])
        meta = {
            "shape": tuple(jacobian.shape),
            "m": jacobian.shape[0],
            "d": jacobian.shape[1],
            "rank": r,
        }
        return compressed, meta

    def decompress(self, compressed: torch.Tensor, meta: dict) -> torch.Tensor:
        m = meta["m"]
        d = meta["d"]
        r = meta["rank"]
        u_size = m * r
        s_size = r
        vh_size = r * d
        U_r = compressed[:u_size].reshape(m, r)
        S_r = compressed[u_size:u_size + s_size]
        Vh_r = compressed[u_size + s_size:u_size + s_size + vh_size].reshape(r, d)
        return U_r @ torch.diag(S_r) @ Vh_r


class RandomSketchCompressor(JacobianCompressor):
    def __init__(self, sketch_dim: int = 4, seed: int = 0) -> None:
        self.sketch_dim = sketch_dim
        self.seed = seed

    @property
    def name(self) -> str:
        return f"sketch_s{self.sketch_dim}"

    def compress(self, jacobian: torch.Tensor) -> tuple[torch.Tensor, dict]:
        m, d = jacobian.shape
        gen = torch.Generator(device=jacobian.device).manual_seed(self.seed)
        sketch_matrix = torch.randn(d, self.sketch_dim, generator=gen, device=jacobian.device, dtype=jacobian.dtype)
        sketch_matrix, _ = torch.linalg.qr(sketch_matrix)
        sketched = jacobian @ sketch_matrix
        compressed = torch.cat([sketched.reshape(-1), sketch_matrix.reshape(-1)])
        meta = {
            "shape": tuple(jacobian.shape),
            "m": m,
            "d": d,
            "sketch_dim": self.sketch_dim,
        }
        return compressed, meta

    def decompress(self, compressed: torch.Tensor, meta: dict) -> torch.Tensor:
        m = meta["m"]
        d = meta["d"]
        s = meta["sketch_dim"]
        sketched_size = m * s
        sketched = compressed[:sketched_size].reshape(m, s)
        sketch_matrix = compressed[sketched_size:sketched_size + d * s].reshape(d, s)
        return sketched @ sketch_matrix.T


COMPRESSOR_REGISTRY: dict[str, type[JacobianCompressor]] = {
    "none": NoCompressor,
    "float16": Float16Compressor,
    "topk_0.1": lambda: TopKCompressor(k_ratio=0.1),
    "topk_0.3": lambda: TopKCompressor(k_ratio=0.3),
    "rowtopk_0.1": lambda: RowTopKCompressor(k_ratio=0.1),
    "rowtopk_0.3": lambda: RowTopKCompressor(k_ratio=0.3),
    "lowrank_r2": lambda: LowRankCompressor(rank=2),
    "lowrank_r4": lambda: LowRankCompressor(rank=4),
    "sketch_s2": lambda: RandomSketchCompressor(sketch_dim=2),
    "sketch_s4": lambda: RandomSketchCompressor(sketch_dim=4),
}


def make_compressor(name: str) -> JacobianCompressor:
    if name not in COMPRESSOR_REGISTRY:
        raise ValueError(f"Unknown compressor '{name}'. Available: {list(COMPRESSOR_REGISTRY.keys())}")
    factory = COMPRESSOR_REGISTRY[name]
    if callable(factory) and not isinstance(factory, type):
        return factory()
    return factory()


__all__ = [
    "JacobianCompressor",
    "NoCompressor",
    "Float16Compressor",
    "TopKCompressor",
    "RowTopKCompressor",
    "LowRankCompressor",
    "RandomSketchCompressor",
    "COMPRESSOR_REGISTRY",
    "make_compressor",
]
