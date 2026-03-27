# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant Triton kernels for KV cache compression and packed attention.

Contains:
  - K compression kernel (_fully_packed_k): normalize, rotate, quantize, pack
  - V compression kernel (_fully_packed_v): normalize, rotate, quantize, pack
  - Sign permutation (permute_signs_for_chunked): reorder for stride-4 chunks
  - v7 attention kernel (_packed_attn_v7): split-K decode from packed cache
  - Split-K reduce kernel (_splitk_reduce)
  - Python wrappers: compress_k_packed, compress_v_packed, packed_attention_v7

Tuned on AMD Instinct MI355X (gfx950, 256 CUs), March 2026.
"""

import math

import torch
import triton
import triton.language as tl


# ===========================================================================
# Compression kernels
# ===========================================================================

@triton.jit
def _fully_packed_k(
    K_ptr, PiT_ptr, PiST_ptr,
    cb0, cb1, cb2, cb3, bd0, bd1, bd2,
    Scratch_ptr,
    Kidx_ptr, Ksigns_ptr, Krnorm_ptr, Knorm_ptr,
    N_total,
    D: tl.constexpr, DQ: tl.constexpr, DE: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    n_mask = rn < N_total

    k = tl.load(K_ptr + rn[:, None] * D + rd[None, :],
                mask=n_mask[:, None], other=0.0).to(tl.float32)
    k_sq = tl.sum(k * k, axis=1)
    k_norm_val = tl.sqrt(k_sq)
    k_normalized = k * (1.0 / (k_norm_val + 1e-8))[:, None]
    tl.store(Knorm_ptr + rn, k_norm_val.to(tl.float16), mask=n_mask)

    pit = tl.load(PiT_ptr + rd[:, None] * D + rd[None, :]).to(tl.float32)
    rotated = tl.dot(k_normalized.to(tl.float16), pit.to(tl.float16)).to(tl.float32)

    idx = (rotated >= bd0).to(tl.int32) + (rotated >= bd1).to(tl.int32) + (rotated >= bd2).to(tl.int32)
    recon = tl.where(idx == 0, cb0, tl.where(idx == 1, cb1, tl.where(idx == 2, cb2, cb3)))

    rot_resid = (rotated - recon) * k_norm_val[:, None]
    r_norm_sq = tl.sum(rot_resid * rot_resid, axis=1)
    tl.store(Krnorm_ptr + rn, tl.sqrt(r_norm_sq).to(tl.float16), mask=n_mask)

    pist = tl.load(PiST_ptr + rd[:, None] * D + rd[None, :]).to(tl.float32)
    proj = tl.dot(rot_resid.to(tl.float16), pist.to(tl.float16)).to(tl.float32)
    sign_bits = (proj >= 0).to(tl.int32)

    # Pack idx: 4x2-bit per byte
    shift_2bit = ((rd % 4) * 2).to(tl.int32)
    shifted_idx = idx << shift_2bit[None, :]
    tl.store(Scratch_ptr + rn[:, None] * D + rd[None, :], shifted_idx, mask=n_mask[:, None])

    rq = tl.arange(0, DQ)
    p0 = tl.load(Scratch_ptr + rn[:, None] * D + (rq*4+0)[None, :], mask=n_mask[:, None], other=0)
    p1 = tl.load(Scratch_ptr + rn[:, None] * D + (rq*4+1)[None, :], mask=n_mask[:, None], other=0)
    p2 = tl.load(Scratch_ptr + rn[:, None] * D + (rq*4+2)[None, :], mask=n_mask[:, None], other=0)
    p3 = tl.load(Scratch_ptr + rn[:, None] * D + (rq*4+3)[None, :], mask=n_mask[:, None], other=0)
    tl.store(Kidx_ptr + rn[:, None] * DQ + rq[None, :],
             (p0 | p1 | p2 | p3).to(tl.int8), mask=n_mask[:, None])

    # Pack signs: 8x1-bit per byte
    shift_1bit = (rd % 8).to(tl.int32)
    shifted_signs = sign_bits << shift_1bit[None, :]
    tl.store(Scratch_ptr + rn[:, None] * D + rd[None, :], shifted_signs, mask=n_mask[:, None])

    re = tl.arange(0, DE)
    ps = tl.load(Scratch_ptr + rn[:, None]*D + (re*8+0)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+1)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+2)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+3)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+4)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+5)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+6)[None, :], mask=n_mask[:, None], other=0)
    ps = ps | tl.load(Scratch_ptr + rn[:, None]*D + (re*8+7)[None, :], mask=n_mask[:, None], other=0)
    tl.store(Ksigns_ptr + rn[:, None] * DE + re[None, :], ps.to(tl.int8), mask=n_mask[:, None])


@triton.jit
def _fully_packed_v(
    V_ptr, PiT_ptr, bd0, bd1, bd2,
    Scratch_ptr, Vidx_ptr, Vnorm_ptr,
    N_total,
    D: tl.constexpr, DQ: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    n_mask = rn < N_total

    v = tl.load(V_ptr + rn[:, None] * D + rd[None, :],
                mask=n_mask[:, None], other=0.0).to(tl.float32)
    v_sq = tl.sum(v * v, axis=1)
    v_norm_val = tl.sqrt(v_sq)
    v_normalized = v * (1.0 / (v_norm_val + 1e-8))[:, None]
    tl.store(Vnorm_ptr + rn, v_norm_val.to(tl.float16), mask=n_mask)

    pit = tl.load(PiT_ptr + rd[:, None] * D + rd[None, :]).to(tl.float32)
    rotated = tl.dot(v_normalized.to(tl.float16), pit.to(tl.float16)).to(tl.float32)
    idx = (rotated >= bd0).to(tl.int32) + (rotated >= bd1).to(tl.int32) + (rotated >= bd2).to(tl.int32)

    shift_2bit = ((rd % 4) * 2).to(tl.int32)
    shifted_idx = idx << shift_2bit[None, :]
    tl.store(Scratch_ptr + rn[:, None] * D + rd[None, :], shifted_idx, mask=n_mask[:, None])

    rq = tl.arange(0, DQ)
    p0 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+0)[None, :], mask=n_mask[:, None], other=0)
    p1 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+1)[None, :], mask=n_mask[:, None], other=0)
    p2 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+2)[None, :], mask=n_mask[:, None], other=0)
    p3 = tl.load(Scratch_ptr + rn[:, None]*D + (rq*4+3)[None, :], mask=n_mask[:, None], other=0)
    tl.store(Vidx_ptr + rn[:, None] * DQ + rq[None, :],
             (p0 | p1 | p2 | p3).to(tl.int8), mask=n_mask[:, None])


# ===========================================================================
# Sign permutation utility
# ===========================================================================

def permute_signs_for_chunked(signs_packed: torch.Tensor, D: int = 128) -> torch.Tensor:
    """Permute sign bytes from sequential to stride-4 order for v7 attention.

    Input: [*, DE] uint8 where DE=D//8, byte j has bits for dims 8j..8j+7
    Output: [*, DE] uint8, reordered so chunk c's bits are in bytes
            [c*SDE..(c+1)*SDE-1] where SDE = DQ//8 = 4

    This allows the attention kernel to process QJL signs in 4x[BS, DQ]
    chunks matching the stride-4 K index layout.
    """
    DE = D // 8
    DQ = D // 4
    shape = signs_packed.shape
    flat = signs_packed.reshape(-1, DE)
    N = flat.shape[0]
    device = flat.device

    # Unpack all D bits
    p = flat.to(torch.int32).unsqueeze(-1)  # [N, DE, 1]
    bits_idx = torch.arange(8, device=device)  # [8]
    all_bits = ((p >> bits_idx) & 1).reshape(N, D)  # [N, D]

    # Permute: chunk c gets dims c, c+4, c+8, ..., c+4*(DQ-1)
    perm_bits = torch.zeros_like(all_bits)
    for c in range(4):
        chunk_dims = torch.arange(c, D, 4, device=device)  # [DQ]
        perm_bits[:, c * DQ:(c + 1) * DQ] = all_bits[:, chunk_dims]

    # Repack into bytes
    perm_bytes = torch.zeros(N, DE, dtype=torch.uint8, device=device)
    for b in range(DE):
        val = torch.zeros(N, dtype=torch.int32, device=device)
        for bit in range(8):
            val |= perm_bits[:, b * 8 + bit].to(torch.int32) << bit
        perm_bytes[:, b] = val.to(torch.uint8)

    return perm_bytes.reshape(shape)


# ===========================================================================
# v7 Attention kernel (production decode)
# ===========================================================================

@triton.jit
def _packed_attn_v7(
    Q_ptr,             # [BH, D] fp16
    Q_proj_cs_ptr,     # [BH, D] fp16 — Q_proj * corr_scale (precomputed)
    Kidx_ptr,          # [BH, Sk, DQ] uint8 — packed 2-bit K indices
    Ksigns_perm_ptr,   # [BH, Sk, DE] uint8 — packed 1-bit K signs (PERMUTED)
    Krnorm_ptr,        # [BH, Sk] fp16
    Knorm_ptr,         # [BH, Sk] fp16
    Vidx_ptr,          # [BH, Sk, DQ] uint8 — packed 2-bit V indices
    Vnorm_ptr,         # [BH, Sk] fp16
    Cb_ptr,            # [4] fp32 — codebook centroids
    Out_ptr,           # [BH, num_splits, D] fp32
    Lse_ptr,           # [BH, num_splits] fp32
    Sk,
    num_splits,
    D: tl.constexpr,
    DQ: tl.constexpr,     # D // 4 = 32
    DE: tl.constexpr,     # D // 8 = 16
    SDE: tl.constexpr,    # DQ // 8 = 4 (sign bytes per chunk)
    BLOCK_SK: tl.constexpr,
):
    head_id = tl.program_id(0)
    split_id = tl.program_id(1)
    tokens_per_split = (Sk + num_splits - 1) // num_splits
    sk_start = split_id * tokens_per_split
    sk_end = tl.minimum(sk_start + tokens_per_split, Sk)

    rdq = tl.arange(0, DQ)
    rd = tl.arange(0, D)

    # Stride-4 Q loading to match packed byte interleaving
    q0 = tl.load(Q_ptr + head_id*D + rdq*4 + 0).to(tl.float32)
    q1 = tl.load(Q_ptr + head_id*D + rdq*4 + 1).to(tl.float32)
    q2 = tl.load(Q_ptr + head_id*D + rdq*4 + 2).to(tl.float32)
    q3 = tl.load(Q_ptr + head_id*D + rdq*4 + 3).to(tl.float32)

    # Stride-4 Q_proj*corr_scale loading (corr_scale precomputed)
    qp0 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 0).to(tl.float32)
    qp1 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 1).to(tl.float32)
    qp2 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 2).to(tl.float32)
    qp3 = tl.load(Q_proj_cs_ptr + head_id*D + rdq*4 + 3).to(tl.float32)

    # Sign unpack indices for permuted layout
    sign_byte_in_chunk = (rdq // 8).to(tl.int32)   # [DQ]
    sign_bit_in_byte = (rdq % 8).to(tl.int32)       # [DQ]

    # V unpack maps (full D, unchanged)
    packed_col_full = (rd // 4).to(tl.int32)
    shift_2bit_full = ((rd % 4) * 2).to(tl.int32)

    # Online softmax state
    m_prev = float('-inf')
    l_prev = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for tile_start in range(sk_start, sk_end, BLOCK_SK):
        tile_end = tl.minimum(tile_start + BLOCK_SK, sk_end)
        rsk = tile_start + tl.arange(0, BLOCK_SK)
        sk_mask = rsk < tile_end

        # === PHASE 1: Batch all loads (K idx + 4 sign chunks + norms) ===
        kidx_packed = tl.load(Kidx_ptr + head_id*Sk*DQ + rsk[:, None]*DQ + rdq[None, :],
                              mask=sk_mask[:, None], other=0).to(tl.int32)
        s0_raw = tl.load(Ksigns_perm_ptr + head_id*Sk*DE + rsk[:, None]*DE + sign_byte_in_chunk[None, :],
                         mask=sk_mask[:, None], other=0).to(tl.int32)
        s1_raw = tl.load(Ksigns_perm_ptr + head_id*Sk*DE + rsk[:, None]*DE + SDE + sign_byte_in_chunk[None, :],
                         mask=sk_mask[:, None], other=0).to(tl.int32)
        s2_raw = tl.load(Ksigns_perm_ptr + head_id*Sk*DE + rsk[:, None]*DE + 2*SDE + sign_byte_in_chunk[None, :],
                         mask=sk_mask[:, None], other=0).to(tl.int32)
        s3_raw = tl.load(Ksigns_perm_ptr + head_id*Sk*DE + rsk[:, None]*DE + 3*SDE + sign_byte_in_chunk[None, :],
                         mask=sk_mask[:, None], other=0).to(tl.int32)
        k_norms = tl.load(Knorm_ptr + head_id*Sk + rsk, mask=sk_mask, other=0.0).to(tl.float32)
        k_rnorms = tl.load(Krnorm_ptr + head_id*Sk + rsk, mask=sk_mask, other=0.0).to(tl.float32)

        # === PHASE 2: K MSE score (codebook gather) ===
        kr0 = tl.load(Cb_ptr + (kidx_packed & 3))
        kr1 = tl.load(Cb_ptr + ((kidx_packed >> 2) & 3))
        kr2 = tl.load(Cb_ptr + ((kidx_packed >> 4) & 3))
        kr3 = tl.load(Cb_ptr + ((kidx_packed >> 6) & 3))

        score_mse = (tl.sum(kr0 * q0[None, :], axis=1) +
                     tl.sum(kr1 * q1[None, :], axis=1) +
                     tl.sum(kr2 * q2[None, :], axis=1) +
                     tl.sum(kr3 * q3[None, :], axis=1)) * k_norms

        # === PHASE 3: Chunked QJL score (corr_scale already in qp) ===
        sf0 = ((s0_raw >> sign_bit_in_byte[None, :]) & 1).to(tl.float32) * 2.0 - 1.0
        sqjl0 = tl.sum(qp0[None, :] * sf0 * k_rnorms[:, None], axis=1)

        sf1 = ((s1_raw >> sign_bit_in_byte[None, :]) & 1).to(tl.float32) * 2.0 - 1.0
        sqjl1 = tl.sum(qp1[None, :] * sf1 * k_rnorms[:, None], axis=1)

        sf2 = ((s2_raw >> sign_bit_in_byte[None, :]) & 1).to(tl.float32) * 2.0 - 1.0
        sqjl2 = tl.sum(qp2[None, :] * sf2 * k_rnorms[:, None], axis=1)

        sf3 = ((s3_raw >> sign_bit_in_byte[None, :]) & 1).to(tl.float32) * 2.0 - 1.0
        sqjl3 = tl.sum(qp3[None, :] * sf3 * k_rnorms[:, None], axis=1)

        score = tl.where(sk_mask, score_mse + sqjl0 + sqjl1 + sqjl2 + sqjl3, float('-inf'))

        # === Online softmax ===
        m_new = tl.maximum(m_prev, tl.max(score, axis=0))
        alpha = tl.exp(m_prev - m_new)
        p = tl.exp(score - m_new)
        l_new = alpha * l_prev + tl.sum(p, axis=0)

        # === V: full [BS, D] reconstruction via indirect gather ===
        vidx_raw = tl.load(Vidx_ptr + head_id*Sk*DQ + rsk[:, None]*DQ + packed_col_full[None, :],
                           mask=sk_mask[:, None], other=0).to(tl.int32)
        v_idx_full = (vidx_raw >> shift_2bit_full[None, :]) & 3
        v_recon_full = tl.load(Cb_ptr + v_idx_full)
        v_norms = tl.load(Vnorm_ptr + head_id*Sk + rsk, mask=sk_mask, other=0.0).to(tl.float32)
        v_mse = v_recon_full * v_norms[:, None]

        acc = alpha * acc + tl.sum(p[:, None] * v_mse, axis=0)
        m_prev = m_new
        l_prev = l_new

    out_offset = head_id * num_splits * D + split_id * D
    tl.store(Out_ptr + out_offset + rd, acc)
    tl.store(Lse_ptr + head_id*num_splits + split_id, m_prev + tl.log(l_prev))


@triton.jit
def _splitk_reduce(
    Out_ptr,    # [BH, num_splits, D] fp32
    Lse_ptr,    # [BH, num_splits] fp32
    Final_ptr,  # [BH, D] fp16
    num_splits,
    D: tl.constexpr,
    NS: tl.constexpr,
):
    head_id = tl.program_id(0)
    rd = tl.arange(0, D)
    rs = tl.arange(0, NS)

    lses = tl.load(Lse_ptr + head_id * num_splits + rs,
                   mask=rs < num_splits, other=float('-inf'))
    max_lse = tl.max(lses, axis=0)

    acc = tl.zeros([D], dtype=tl.float32)
    total_w = 0.0
    for s in range(NS):
        if s < num_splits:
            lse_s = tl.load(Lse_ptr + head_id * num_splits + s)
            w = tl.exp(lse_s - max_lse)
            out_s = tl.load(Out_ptr + head_id * num_splits * D + s * D + rd)
            acc += w * out_s
            total_w += w

    acc = acc / total_w
    tl.store(Final_ptr + head_id * D + rd, acc.to(tl.float16))


# ===========================================================================
# Tuning tables
# ===========================================================================

# MiniMax-M2.5 specific tuning (8 KV heads, head_dim=128)
_TUNING_TABLE_MINIMAX = {
    # (BH, Sk_threshold): (BLOCK_SK, num_splits, num_warps, num_stages)
    (4, 32768): (128, 64, 8, 2),
    (4, 65536): (128, 64, 8, 2),
    (4, 131072): (128, 64, 8, 1),
    (4, 196608): (128, 64, 8, 2),
    (8, 32768): (128, 64, 2, 1),
    (8, 65536): (128, 64, 2, 1),
    (8, 131072): (128, 64, 2, 2),
    (8, 196608): (32, 128, 4, 1),
}

# General tuning table for other models
_TUNING_TABLE_GENERAL = [
    # (max_BH, max_Sk, BLOCK_SK, num_splits, num_warps, num_stages)
    (16,  1024,  32,  16,  4,  2),
    (16,  4096, 128,  16,  8,  1),
    (16, 16384,  32,  64,  8,  2),
    (16, 99999,  64,  64,  8,  2),
    (48,  1024,  16,  16,  4,  2),
    (48,  4096,  16,  64,  4,  1),
    (48, 16384,  16,  64,  2,  2),
    (48, 99999,  16,  64,  2,  1),
    (99999,  1024,  16,  16,  2,  1),
    (99999,  4096,  16,  32,  2,  2),
    (99999, 16384,  16,  32,  2,  2),
    (99999, 99999,  16, 128,  2,  1),
]


def _get_config(BH: int, Sk: int) -> tuple[int, int, int, int]:
    """Look up tuned config from table.

    Returns (BLOCK_SK, num_splits, num_warps, num_stages).
    """
    # Try MiniMax-specific table first (exact BH match)
    for (bh, sk_thresh), (bs, sp, w, s) in _TUNING_TABLE_MINIMAX.items():
        if BH == bh and Sk <= sk_thresh:
            sp = min(sp, max(1, Sk // bs))
            return bs, sp, w, s

    # Fall back to general table
    for max_bh, max_sk, bs, sp, w, s in _TUNING_TABLE_GENERAL:
        if BH <= max_bh and Sk <= max_sk:
            sp = min(sp, max(1, Sk // bs))
            return bs, sp, w, s

    # Fallback
    return 16, 64, 2, 1


# ===========================================================================
# Python wrapper functions
# ===========================================================================

def compress_k_packed(
    K: torch.Tensor,
    PiT: torch.Tensor,
    PiST: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress K vectors to packed format.

    Args:
        K: [N, D] float32 key vectors
        PiT: [D, D] rotation matrix transpose
        PiST: [D, D] Pi @ S.T product
        centroids: [4] codebook centroids
        boundaries: [3] decision boundaries

    Returns:
        k_idx: [N, DQ] uint8 — packed 2-bit indices (4 per byte)
        k_signs: [N, DE] uint8 — packed 1-bit signs (8 per byte)
        k_rnorm: [N] fp16 — residual norms
        k_norm: [N] fp16 — vector norms
    """
    N, D = K.shape
    cb = centroids.tolist()
    bd = boundaries.tolist()
    scratch = torch.empty(N, D, dtype=torch.int32, device=K.device)
    k_idx = torch.empty(N, D // 4, dtype=torch.int8, device=K.device)
    k_signs = torch.empty(N, D // 8, dtype=torch.int8, device=K.device)
    k_rnorm = torch.empty(N, dtype=torch.float16, device=K.device)
    k_norm = torch.empty(N, dtype=torch.float16, device=K.device)

    # Tuned on MI355X
    if N <= 2048:
        BN, nw, ns = 32, 4, 1
    elif N <= 32768:
        BN, nw, ns = 128, 4, 1
    else:
        BN, nw, ns = 64, 2, 2

    grid = ((N + BN - 1) // BN,)
    _fully_packed_k[grid](
        K, PiT, PiST,
        cb[0], cb[1], cb[2], cb[3],
        bd[0], bd[1], bd[2],
        scratch, k_idx, k_signs, k_rnorm, k_norm,
        N, D=D, DQ=D // 4, DE=D // 8, BN=BN,
        num_warps=nw, num_stages=ns,
    )
    return k_idx.to(torch.uint8), k_signs.to(torch.uint8), k_rnorm, k_norm


def compress_v_packed(
    V: torch.Tensor,
    PiT: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress V vectors to packed format.

    Args:
        V: [N, D] float32 value vectors
        PiT: [D, D] rotation matrix transpose
        centroids: [4] codebook centroids (used only for boundaries)
        boundaries: [3] decision boundaries

    Returns:
        v_idx: [N, DQ] uint8 — packed 2-bit indices (4 per byte)
        v_norm: [N] fp16 — vector norms
    """
    N, D = V.shape
    bd = boundaries.tolist()
    scratch = torch.empty(N, D, dtype=torch.int32, device=V.device)
    v_idx = torch.empty(N, D // 4, dtype=torch.int8, device=V.device)
    v_norm = torch.empty(N, dtype=torch.float16, device=V.device)

    # Tuned on MI355X
    if N <= 32768:
        BN, nw, ns = 64, 8, 2
    else:
        BN, nw, ns = 128, 8, 1

    grid = ((N + BN - 1) // BN,)
    _fully_packed_v[grid](
        V, PiT,
        bd[0], bd[1], bd[2],
        scratch, v_idx, v_norm,
        N, D=D, DQ=D // 4, BN=BN,
        num_warps=nw, num_stages=ns,
    )
    return v_idx.to(torch.uint8), v_norm


def packed_attention_v7(
    Q: torch.Tensor,
    Q_proj: torch.Tensor,
    ki: torch.Tensor,
    ks_perm: torch.Tensor,
    kr: torch.Tensor,
    kn: torch.Tensor,
    vi: torch.Tensor,
    vn: torch.Tensor,
    centroids: torch.Tensor,
    corr_scale: float,
    num_splits: int | None = None,
    block_sk: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """v7 packed attention with split-K and auto-dispatch.

    Signs must be in permuted (stride-4) order — use permute_signs_for_chunked().
    Auto-selects (BLOCK_SK, num_splits, warps, stages) from tuning table
    unless overridden by explicit arguments.

    Args:
        Q: [BH, D] fp16 query vectors
        Q_proj: [BH, D] fp16 projected query vectors (Q @ S.T)
        ki: [BH, Sk, DQ] uint8 packed K indices
        ks_perm: [BH, Sk, DE] uint8 packed K signs (PERMUTED)
        kr: [BH, Sk] fp16 K residual norms
        kn: [BH, Sk] fp16 K vector norms
        vi: [BH, Sk, DQ] uint8 packed V indices
        vn: [BH, Sk] fp16 V vector norms
        centroids: [4] fp32 codebook
        corr_scale: sqrt(pi/2) / sqrt(D)

    Returns:
        output: [BH, D] fp16
    """
    BH, D = Q.shape
    Sk = kr.shape[1]
    DQ = D // 4
    DE = D // 8
    SDE = DQ // 8  # = 4

    # Auto-dispatch from tuning table
    bs_auto, sp_auto, w_auto, s_auto = _get_config(BH, Sk)
    BS = block_sk if block_sk is not None else bs_auto
    nsplits = num_splits if num_splits is not None else sp_auto
    nw = num_warps if num_warps is not None else w_auto
    ns = num_stages if num_stages is not None else s_auto

    cb_ptr = centroids.contiguous()

    # Precompute Q_proj * corr_scale (saves 4 muls/tile in kernel)
    Q_proj_cs = (Q_proj.float() * corr_scale).to(Q_proj.dtype)

    out_splits = torch.empty(BH, nsplits, D, dtype=torch.float32, device=Q.device)
    lse_splits = torch.empty(BH, nsplits, dtype=torch.float32, device=Q.device)

    _packed_attn_v7[(BH, nsplits)](
        Q, Q_proj_cs,
        ki, ks_perm, kr, kn,
        vi, vn,
        cb_ptr,
        out_splits, lse_splits,
        Sk, nsplits,
        D=D, DQ=DQ, DE=DE, SDE=SDE, BLOCK_SK=BS,
        num_warps=nw, num_stages=ns,
    )

    # Reduce splits
    NS_pad = 1
    while NS_pad < nsplits:
        NS_pad *= 2

    final = torch.empty(BH, D, dtype=torch.float16, device=Q.device)
    _splitk_reduce[(BH,)](
        out_splits, lse_splits, final,
        nsplits, D=D, NS=NS_pad,
        num_warps=4,
    )

    return final
