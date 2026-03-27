# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant 3-bit KV cache compression state and configuration.

TurboQuant compresses KV cache from fp16 (512 bytes/token/head) to a packed
format (86 bytes/token/head) — ~6x compression. It uses:
1. Random rotation (orthogonal matrix) + Lloyd-Max 2-bit scalar quantization
   for MSE reconstruction
2. QJL 1-bit sign projection for unbiased inner product preservation
3. A fused Triton decode kernel that reads the packed format directly

Target: MiniMax-M2.5 (62 layers, 8 KV heads GQA, head_dim=128) on MI355X.
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression."""
    bit_width: int = 3        # Total bits: 2 MSE + 1 QJL
    seed: int = 42             # Base seed for deterministic matrices


class TurboQuantState:
    """Per-layer state for TurboQuant compression.

    Holds the rotation matrix Pi, QJL projection matrix S, precomputed
    products (PiT, PiST), Lloyd-Max codebook, and boundaries.
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        head_size: int,
        layer_idx: int,
        device: torch.device,
    ):
        self.config = config
        self.head_size = head_size
        self.device = device
        self.layer_idx = layer_idx

        # Deterministic rotation matrix per layer via QR decomposition
        gen = torch.Generator(device="cpu")
        gen.manual_seed(config.seed + layer_idx)
        G = torch.randn(
            head_size, head_size,
            generator=gen, device="cpu", dtype=torch.float32,
        )
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        Pi = (Q * diag_sign.unsqueeze(0)).to(device)
        self.Pi = Pi.contiguous()
        self.PiT = Pi.T.contiguous()

        # QJL projection matrix (different seed)
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(config.seed + layer_idx + 10000)
        S = torch.randn(
            head_size, head_size,
            generator=gen2, device="cpu", dtype=torch.float32,
        ).to(device)
        self.S = S
        self.ST = S.T.contiguous()
        self.PiST = (Pi @ S.T).contiguous()

        # Lloyd-Max codebook (2-bit = 4 centroids for MSE component)
        mse_bits = 2  # 3-bit total: 2 MSE + 1 QJL
        self.centroids = self._solve_codebook(head_size, mse_bits).to(device)
        self.boundaries = (
            (self.centroids[:-1] + self.centroids[1:]) / 2.0
        ).contiguous()

        # Correction scale for QJL estimator
        self.corr_scale = math.sqrt(math.pi / 2) / math.sqrt(head_size)

    @staticmethod
    def _solve_codebook(d: int, bits: int) -> torch.Tensor:
        """Solve Lloyd-Max optimal codebook for Gaussian(0, 1/sqrt(d)).

        Args:
            d: Dimensionality (determines sigma = 1/sqrt(d))
            bits: Number of quantization bits (produces 2^bits centroids)

        Returns:
            Tensor of shape [2^bits] with optimal centroid values.
        """
        from scipy import integrate

        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma**2)) * math.exp(
                -x * x / (2 * sigma**2)
            )

        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [
            lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)
        ]

        for _ in range(200):
            boundaries = [
                (centroids[i] + centroids[i + 1]) / 2.0
                for i in range(n_levels - 1)
            ]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_c = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_c.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_c[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_c

        return torch.tensor(centroids, dtype=torch.float32)
