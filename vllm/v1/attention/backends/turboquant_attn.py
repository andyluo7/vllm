# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant packed KV cache attention backend.

Compresses KV cache from fp16 to ~6x smaller packed format using:
- 2-bit Lloyd-Max scalar quantization (MSE component)
- 1-bit QJL sign projection (inner product preservation)
- Fused Triton decode kernel reading packed format directly

Target: MiniMax-M2.5 on AMD MI355X GPUs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Constants for D=128
_D = 128
_DQ = _D // 4   # 32 — packed 2-bit indices per byte
_DE = _D // 8   # 16 — packed 1-bit signs per byte
_K_SLOT_BYTES = _DQ + _DE + 2 + 2  # 32 + 16 + 2(rnorm) + 2(norm) = 52
_V_SLOT_BYTES = _DQ + 2            # 32 + 2(norm) = 34
_SLOT_BYTES = max(_K_SLOT_BYTES, _V_SLOT_BYTES)  # 52


@dataclass
class TurboQuantAttentionMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class TurboQuantAttentionBackend(AttentionBackend):
    """TurboQuant packed KV cache attention backend."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["turboquant"]
    accept_output_buffer = False

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT_ATTN"

    @staticmethod
    def get_impl_cls() -> type[TurboQuantAttentionImpl]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TurboQuantAttentionMetadataBuilder]:
        return TurboQuantAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Packed KV cache layout.

        K per token per head: DQ(32) + DE(16) + 2(rnorm) + 2(norm) = 52 bytes
        V per token per head: DQ(32) + 2(norm) = 34 bytes
        Total: 86 bytes per token per head

        Shape: [num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES]
        where 2 = {K, V}, SLOT_BYTES = 52 (V wastes 18 bytes for uniformity)
        """
        DQ = head_size // 4
        DE = head_size // 8
        k_slot = DQ + DE + 4  # indices + signs + rnorm(2) + norm(2)
        v_slot = DQ + 2       # indices + norm(2)
        slot_bytes = max(k_slot, v_slot)
        return (num_blocks, 2, block_size, num_kv_heads, slot_bytes)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype == "turboquant"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


class TurboQuantAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builds metadata for TurboQuant attention."""

    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TurboQuantAttentionMetadata:
        return TurboQuantAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


class TurboQuantAttentionImpl(AttentionImpl[TurboQuantAttentionMetadata]):
    """TurboQuant attention implementation.

    On first forward call per layer, initializes TurboQuantState.
    During prefill: compresses K,V using Triton kernels, stores packed in cache.
    During decode: uses packed_attention_v7 to compute attention directly.
    """

    # Per-layer state cache (shared across instances)
    _layer_states: dict[int, object] = {}
    _initialized: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        # Lazily initialized per-layer
        self._state = None
        self._layer_idx_counter = 0

    def _ensure_state(self, device: torch.device) -> None:
        """Initialize TurboQuantState for this layer if not yet done."""
        if self._state is not None:
            return

        from vllm.model_executor.layers.quantization.turboquant import (
            TurboQuantConfig,
            TurboQuantState,
        )

        # Use class-level counter for layer indexing
        layer_idx = TurboQuantAttentionImpl._layer_idx_counter
        TurboQuantAttentionImpl._layer_idx_counter += 1

        config = TurboQuantConfig()
        if layer_idx not in TurboQuantAttentionImpl._layer_states:
            state = TurboQuantState(config, self.head_size, layer_idx, device)
            TurboQuantAttentionImpl._layer_states[layer_idx] = state
        self._state = TurboQuantAttentionImpl._layer_states[layer_idx]

    def _write_k_to_cache(
        self,
        key: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Compress K and write packed data to cache slots.

        Args:
            key: [num_tokens, num_kv_heads, head_size] fp16/bf16
            kv_cache: [num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES] uint8
            slot_mapping: [num_tokens] int — maps tokens to block slots
        """
        from vllm.v1.attention.ops.turboquant_kernels import (
            compress_k_packed,
            permute_signs_for_chunked,
        )

        state = self._state
        num_tokens, num_kv_heads, D = key.shape
        DQ = D // 4
        DE = D // 8
        block_size = kv_cache.shape[2]

        for h in range(num_kv_heads):
            k_head = key[:, h, :].float().contiguous()  # [N, D]
            k_idx, k_signs, k_rnorm, k_norm = compress_k_packed(
                k_head, state.PiT, state.PiST,
                state.centroids, state.boundaries,
            )
            # Permute signs to stride-4 order
            k_signs_perm = permute_signs_for_chunked(k_signs, D)

            # Write to cache slot by slot
            for t in range(num_tokens):
                slot = slot_mapping[t].item()
                if slot < 0:
                    continue
                block_idx = slot // block_size
                block_off = slot % block_size

                # K layout in cache[block, 0, offset, head, :]:
                # [0:DQ] = k_idx, [DQ:DQ+DE] = k_signs_perm,
                # [DQ+DE:DQ+DE+2] = k_rnorm (fp16), [DQ+DE+2:DQ+DE+4] = k_norm (fp16)
                cache_slot = kv_cache[block_idx, 0, block_off, h]
                cache_slot[:DQ] = k_idx[t]
                cache_slot[DQ:DQ + DE] = k_signs_perm[t]
                cache_slot[DQ + DE:DQ + DE + 2] = k_rnorm[t].view(torch.uint8)
                cache_slot[DQ + DE + 2:DQ + DE + 4] = k_norm[t].view(torch.uint8)

    def _write_v_to_cache(
        self,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Compress V and write packed data to cache slots.

        Args:
            value: [num_tokens, num_kv_heads, head_size] fp16/bf16
            kv_cache: [num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES] uint8
            slot_mapping: [num_tokens] int
        """
        from vllm.v1.attention.ops.turboquant_kernels import compress_v_packed

        state = self._state
        num_tokens, num_kv_heads, D = value.shape
        DQ = D // 4
        block_size = kv_cache.shape[2]

        for h in range(num_kv_heads):
            v_head = value[:, h, :].float().contiguous()  # [N, D]
            v_idx, v_norm = compress_v_packed(
                v_head, state.PiT, state.centroids, state.boundaries,
            )

            for t in range(num_tokens):
                slot = slot_mapping[t].item()
                if slot < 0:
                    continue
                block_idx = slot // block_size
                block_off = slot % block_size

                # V layout in cache[block, 1, offset, head, :]:
                # [0:DQ] = v_idx, [DQ:DQ+2] = v_norm (fp16)
                cache_slot = kv_cache[block_idx, 1, block_off, h]
                cache_slot[:DQ] = v_idx[t]
                cache_slot[DQ:DQ + 2] = v_norm[t].view(torch.uint8)

    def _gather_kv_from_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Gather packed KV data from paged cache into contiguous tensors.

        For decode, each request has exactly 1 query token, so we gather
        all cached K/V for each head into contiguous [BH, Sk, ...] tensors.

        Returns:
            ki: [BH, max_seq, DQ] uint8
            ks_perm: [BH, max_seq, DE] uint8
            kr: [BH, max_seq] fp16
            kn: [BH, max_seq] fp16
            vi: [BH, max_seq, DQ] uint8
            vn: [BH, max_seq] fp16
        """
        D = head_size
        DQ = D // 4
        DE = D // 8
        block_size = kv_cache.shape[2]
        batch_size = seq_lens.shape[0]
        max_seq = seq_lens.max().item()
        device = kv_cache.device
        BH = batch_size * num_kv_heads

        ki = torch.zeros(BH, max_seq, DQ, dtype=torch.uint8, device=device)
        ks_perm = torch.zeros(BH, max_seq, DE, dtype=torch.uint8, device=device)
        kr = torch.zeros(BH, max_seq, dtype=torch.float16, device=device)
        kn = torch.zeros(BH, max_seq, dtype=torch.float16, device=device)
        vi = torch.zeros(BH, max_seq, DQ, dtype=torch.uint8, device=device)
        vn = torch.zeros(BH, max_seq, dtype=torch.float16, device=device)

        for b in range(batch_size):
            seq_len = seq_lens[b].item()
            for pos in range(seq_len):
                block_idx = block_table[b, pos // block_size].item()
                block_off = pos % block_size

                for h in range(num_kv_heads):
                    bh = b * num_kv_heads + h
                    k_slot = kv_cache[block_idx, 0, block_off, h]
                    ki[bh, pos] = k_slot[:DQ]
                    ks_perm[bh, pos] = k_slot[DQ:DQ + DE]
                    kr[bh, pos] = k_slot[DQ + DE:DQ + DE + 2].view(torch.float16)
                    kn[bh, pos] = k_slot[DQ + DE + 2:DQ + DE + 4].view(torch.float16)

                    v_slot = kv_cache[block_idx, 1, block_off, h]
                    vi[bh, pos] = v_slot[:DQ]
                    vn[bh, pos] = v_slot[DQ:DQ + 2].view(torch.float16)

        return ki, ks_perm, kr, kn, vi, vn

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for TurboQuant attention.

        During prefill (max_query_len > 1):
          - Compress K,V and store in packed cache
          - Use standard attention for prefill output (fallback to PyTorch)
        During decode (max_query_len == 1):
          - Compress new K,V token and store in cache
          - Gather all cached KV into contiguous tensors
          - Run packed_attention_v7 Triton kernel
        """
        from vllm.v1.attention.ops.turboquant_kernels import packed_attention_v7

        self._ensure_state(query.device)
        state = self._state
        num_tokens = query.shape[0]

        # Reshape for per-head processing
        # query: [num_tokens, num_heads * head_size] -> [num_tokens, num_heads, head_size]
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_size = self.head_size

        q = query.view(num_tokens, num_heads, head_size)
        k = key.view(num_tokens, num_kv_heads, head_size)
        v = value.view(num_tokens, num_kv_heads, head_size)

        # Write compressed K,V to cache
        if kv_cache.numel() > 0:
            self._write_k_to_cache(k, kv_cache, attn_metadata.slot_mapping)
            self._write_v_to_cache(v, kv_cache, attn_metadata.slot_mapping)

        is_decode = attn_metadata.max_query_len == 1

        if is_decode and kv_cache.numel() > 0:
            # Decode: use packed attention kernel
            batch_size = attn_metadata.seq_lens.shape[0]
            BH = batch_size * num_kv_heads

            # Gather KV from paged cache
            ki, ks_perm, kr, kn, vi, vn = self._gather_kv_from_cache(
                kv_cache, attn_metadata.block_table,
                attn_metadata.seq_lens, num_kv_heads, head_size,
            )

            # Prepare Q: need to expand for GQA (repeat Q heads per KV head)
            heads_per_kv = num_heads // num_kv_heads
            # q shape: [batch_size, num_heads, head_size]
            # We need [BH, head_size] where BH = batch * num_kv_heads
            # For GQA, we average/sum the Q heads mapping to each KV head
            # Actually for attention, each Q head independently attends to its KV head
            # So we need to run attention per Q head group

            # Simple approach: run packed attention for each KV head group
            # Q_group: [batch_size * heads_per_kv, head_size] per KV head
            # But the kernel expects [BH, D] Q with one Q per head
            # For GQA decode: each of the num_heads Q vectors maps to one of num_kv_heads
            # So output is [batch_size, num_heads, head_size]

            out_all = torch.empty(
                batch_size, num_heads, head_size,
                dtype=query.dtype, device=query.device,
            )

            for kv_h in range(num_kv_heads):
                q_head_start = kv_h * heads_per_kv
                q_head_end = q_head_start + heads_per_kv

                for q_h_offset in range(heads_per_kv):
                    q_h = q_head_start + q_h_offset
                    # Q for this head across batch: [batch_size, head_size]
                    Q_batch = q[:, q_h, :].contiguous().to(torch.float16)

                    # Q_proj = Q @ S.T
                    Q_proj = (Q_batch.float() @ state.ST.float()).to(torch.float16)

                    # KV for this kv_head: select from gathered data
                    # ki is [BH, max_seq, DQ] where BH = batch * num_kv_heads
                    # We need indices for kv_h across all batches
                    kv_indices = torch.arange(
                        batch_size, device=query.device
                    ) * num_kv_heads + kv_h

                    ki_h = ki[kv_indices]       # [batch, max_seq, DQ]
                    ks_h = ks_perm[kv_indices]  # [batch, max_seq, DE]
                    kr_h = kr[kv_indices]       # [batch, max_seq]
                    kn_h = kn[kv_indices]       # [batch, max_seq]
                    vi_h = vi[kv_indices]       # [batch, max_seq, DQ]
                    vn_h = vn[kv_indices]       # [batch, max_seq]

                    # Run packed attention
                    attn_out = packed_attention_v7(
                        Q_batch, Q_proj,
                        ki_h, ks_h, kr_h, kn_h,
                        vi_h, vn_h,
                        state.centroids, state.corr_scale,
                    )  # [batch_size, head_size] fp16

                    out_all[:, q_h, :] = attn_out

            # Reshape output
            result = out_all.reshape(num_tokens, num_heads * head_size)
            if output is not None:
                output.copy_(result)
                return output
            return result

        else:
            # Prefill: fallback to standard scaled dot-product attention
            # This is simpler and only runs once per sequence
            import torch.nn.functional as F

            # GQA: expand KV heads
            if num_kv_heads != num_heads:
                heads_per_kv = num_heads // num_kv_heads
                k = k.repeat_interleave(heads_per_kv, dim=1)
                v = v.repeat_interleave(heads_per_kv, dim=1)

            # For prefill, compute attention per-request using seq_lens
            out_list = []
            token_offset = 0
            query_start_loc = attn_metadata.query_start_loc

            for i in range(attn_metadata.seq_lens.shape[0]):
                q_start = query_start_loc[i].item()
                q_end = query_start_loc[i + 1].item()
                q_len = q_end - q_start

                q_i = q[q_start:q_end]  # [q_len, num_heads, head_size]
                k_i = k[q_start:q_end]  # [q_len, num_heads, head_size]
                v_i = v[q_start:q_end]  # [q_len, num_heads, head_size]

                # [num_heads, q_len, head_size]
                q_t = q_i.transpose(0, 1)
                k_t = k_i.transpose(0, 1)
                v_t = v_i.transpose(0, 1)

                # Causal attention
                attn_out = F.scaled_dot_product_attention(
                    q_t.float(), k_t.float(), v_t.float(),
                    is_causal=True,
                    scale=self.scale,
                )
                # [num_heads, q_len, head_size] -> [q_len, num_heads, head_size]
                out_list.append(attn_out.transpose(0, 1).to(query.dtype))

            result = torch.cat(out_list, dim=0).reshape(
                num_tokens, num_heads * head_size
            )
            if output is not None:
                output.copy_(result)
                return output
            return result
