# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-independent tests for MiniMax M3 indexer selection."""

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm.models.minimax_m3.common.indexer as indexer_mod
from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerTritonImpl
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseMetadata,
    minimax_m3_rebase_block_table_to_page16,
    minimax_m3_rebase_slots_to_page16,
)


def _selector_kwargs(**overrides):
    kwargs = {
        "topk_blocks": 16,
        "sparse_block_size": 128,
        "num_index_heads": 2,
        "index_head_dim": 128,
        "score_type": "max",
        "indexer_kv_dtype": "fp8",
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def aiter_indexer_module(monkeypatch: pytest.MonkeyPatch):
    import vllm.models.minimax_m3.amd.indexer_aiter as aiter_indexer
    import vllm.platforms.rocm as rocm_platform

    monkeypatch.setattr(aiter_indexer.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: True)
    monkeypatch.setattr(
        aiter_indexer, "aiter_msa_kernels_unavailable_reason", lambda: None
    )
    return aiter_indexer


@pytest.mark.parametrize(("num_index_heads", "max_decode_query_len"), [(1, 16), (2, 8)])
def test_aiter_indexer_accepts_compiled_shape_limits(
    aiter_indexer_module, num_index_heads: int, max_decode_query_len: int
) -> None:
    reason = aiter_indexer_module.aiter_indexer_unsupported_reason(
        **_selector_kwargs(num_index_heads=num_index_heads),
        max_model_len=1_048_576,
        max_decode_query_len=max_decode_query_len,
    )
    assert reason is None


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"topk_blocks": 8}, "topk_blocks=8 is not built"),
        ({"num_index_heads": 4}, "num_index_heads=4 is not built"),
        ({"index_head_dim": 64}, "index_head_dim=64 is not built"),
        ({"sparse_block_size": 64}, "sparse_block_size=64 is not built"),
        ({"score_type": "sum"}, "score_type='sum' is not built"),
        ({"indexer_kv_dtype": "bf16"}, "needs an fp8 e4m3 index cache"),
    ],
)
def test_aiter_indexer_rejects_uncompiled_shapes(
    aiter_indexer_module, overrides: dict, expected_reason: str
) -> None:
    reason = aiter_indexer_module.aiter_indexer_unsupported_reason(
        **_selector_kwargs(**overrides),
        max_model_len=1_048_576,
        max_decode_query_len=1,
    )
    assert reason is not None
    assert expected_reason in reason


@pytest.mark.parametrize(
    ("num_index_heads", "max_decode_query_len", "max_model_len", "expected_reason"),
    [
        (1, 17, 1_048_576, "exceeds the 16 MFMA columns"),
        (2, 9, 1_048_576, "exceeds the 16 MFMA columns"),
        (2, 1, 1_048_577, "more than the top-k's 8192"),
    ],
)
def test_aiter_indexer_rejects_shapes_past_compiled_limits(
    aiter_indexer_module,
    num_index_heads: int,
    max_decode_query_len: int,
    max_model_len: int,
    expected_reason: str,
) -> None:
    reason = aiter_indexer_module.aiter_indexer_unsupported_reason(
        **_selector_kwargs(num_index_heads=num_index_heads),
        max_model_len=max_model_len,
        max_decode_query_len=max_decode_query_len,
    )
    assert reason is not None
    assert expected_reason in reason


def test_aiter_indexer_rejects_unsupported_runtime(
    aiter_indexer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vllm.platforms.rocm as rocm_platform

    monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: False)
    reason = aiter_indexer_module.aiter_indexer_unsupported_reason(
        **_selector_kwargs(), max_model_len=1_048_576
    )
    assert reason is not None
    assert "needs gfx950" in reason

    monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: True)
    monkeypatch.setattr(
        aiter_indexer_module,
        "aiter_msa_kernels_unavailable_reason",
        lambda: "AITER cannot supply the MSA score/top-k kernels",
    )
    reason = aiter_indexer_module.aiter_indexer_unsupported_reason(
        **_selector_kwargs(), max_model_len=1_048_576
    )
    assert reason == "AITER cannot supply the MSA score/top-k kernels"


def test_rocm_fp8_selector_preserves_aiter_rejection_reason(
    aiter_indexer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1_048_576),
        speculative_config=None,
    )
    monkeypatch.setattr(indexer_mod, "get_current_vllm_config", lambda: config)

    with pytest.raises(NotImplementedError, match="num_index_heads=4 is not built"):
        indexer_mod.select_indexer_impl_cls(**_selector_kwargs(num_index_heads=4))

    selected = indexer_mod.select_indexer_impl_cls(
        **_selector_kwargs(indexer_kv_dtype="bf16", num_index_heads=4)
    )
    assert selected is MiniMaxM3IndexerTritonImpl


@pytest.mark.parametrize(
    ("is_cuda", "capability", "dtype", "topk", "expected"),
    [
        (True, 100, "fp8", 16, "msa"),
        (True, 100, "bf16", 16, "msa"),
        (True, 90, "bf16", 16, "triton"),
        (False, 0, "bf16", 16, "triton"),
    ],
)
def test_non_rocm_selector_never_imports_amd_or_aiter(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda: bool,
    capability: int,
    dtype: str,
    topk: int,
    expected: str,
) -> None:
    class FakeMSAImpl:
        pass

    fake_msa = ModuleType("vllm.models.minimax_m3.nvidia.indexer_msa")
    fake_msa.MiniMaxM3IndexerMSAImpl = FakeMSAImpl  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "vllm.models.minimax_m3.nvidia.indexer_msa", fake_msa
    )
    monkeypatch.setattr(indexer_mod.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(indexer_mod.current_platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(
        indexer_mod.current_platform,
        "is_device_capability_family",
        lambda family: is_cuda and family == capability,
    )

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("vllm.models.minimax_m3.amd")
        assert not name.startswith("aiter")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    selected = indexer_mod.select_indexer_impl_cls(
        **_selector_kwargs(topk_blocks=topk, indexer_kv_dtype=dtype)
    )
    if expected == "msa":
        assert selected is FakeMSAImpl
    else:
        assert selected is MiniMaxM3IndexerTritonImpl


def test_page16_rebases_preserve_negative_padding() -> None:
    slots = torch.tensor([-1, 0, 127, 128, 129, 255, 256], dtype=torch.int64)
    expected_slots = torch.tensor([-1, 0, 127, 256, 257, 383, 512], dtype=torch.int64)
    slots_out = torch.empty_like(slots)
    actual_slots = minimax_m3_rebase_slots_to_page16(slots, 128, out=slots_out)
    assert actual_slots.data_ptr() == slots_out.data_ptr()
    assert torch.equal(actual_slots, expected_slots)

    block_table = torch.tensor([[-1, 0, 3]], dtype=torch.int32)
    expected_table = torch.tensor([[-2, 0, 6]], dtype=torch.int32)
    table_out = torch.empty_like(block_table)
    actual_table = minimax_m3_rebase_block_table_to_page16(block_table, out=table_out)
    assert actual_table.data_ptr() == table_out.data_ptr()
    assert torch.equal(actual_table, expected_table)


def test_aiter_topk_uses_current_keyword_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.models.minimax_m3.amd.indexer_aiter as aiter_indexer
    from vllm.models.minimax_m3.common.indexer import (
        MiniMaxM3IndexerDecodeMetadata,
        MiniMaxM3IndexerPrefillMetadata,
    )

    calls = []

    def score_decode(*args, **kwargs):
        return None

    def score_prefill(*args, **kwargs):
        return None

    def topk(
        score,
        topk_idx,
        block_table,
        seq_lens,
        sparse_bt,
        sparse_ctx,
        max_seq_len=0,
        block_size=0,
        query_len=1,
        num_waves=0,
        num_valid_pages=None,
        row_req_id=None,
        kv_lens=None,
        num_kv_heads=1,
        pages_per_block=8,
    ):
        calls.append(
            SimpleNamespace(
                topk_idx=topk_idx,
                sparse_bt=sparse_bt,
                sparse_ctx=sparse_ctx,
                num_valid_pages=num_valid_pages,
            )
        )

    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_msa = ModuleType("aiter.ops.msa_attention")
    fake_msa.pa_sparse_block_score_decode = score_decode  # type: ignore[attr-defined]
    fake_msa.pa_sparse_block_score_prefill = score_prefill  # type: ignore[attr-defined]
    fake_msa.pa_sparse_block_topk = topk  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.msa_attention", fake_msa)

    block_table = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.int32)
    decode = MiniMaxM3IndexerDecodeMetadata(
        seq_lens=torch.tensor([257], dtype=torch.int32),
        block_table=block_table[:1],
        max_seq_len=258,
        decode_query_len=1,
        max_decode_query_len=1,
    )
    prefill = MiniMaxM3IndexerPrefillMetadata(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([258], dtype=torch.int32),
        context_lens=torch.tensor([256], dtype=torch.int32),
        block_table=block_table[1:],
        max_query_len=2,
        max_seq_len=258,
    )
    index_md = aiter_indexer.MiniMaxM3IndexerAiterMetadata(
        seq_lens=torch.tensor([257, 258], dtype=torch.int32),
        max_seq_len=258,
        slot_mapping=torch.arange(3, dtype=torch.int64),
        num_actual_tokens=3,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=1,
        num_prefill_tokens=2,
        prefill=prefill,
        decode=decode,
        prefill_num_valid_pages=torch.tensor([3, 3], dtype=torch.int32),
        prefill_row_req_id=torch.tensor([0, 0], dtype=torch.int32),
        prefill_kv_lens=torch.tensor([257, 258], dtype=torch.int32),
    )
    attend_md = MiniMaxM3SparseMetadata(
        seq_lens=index_md.seq_lens,
        max_seq_len=258,
        slot_mapping=index_md.slot_mapping,
        num_actual_tokens=3,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=1,
        num_prefill_tokens=2,
        decode=SimpleNamespace(page16_block_table=block_table[:1] * 2),
        prefill=SimpleNamespace(page16_block_table=block_table[1:] * 2),
    )

    impl = object.__new__(aiter_indexer.MiniMaxM3IndexerAiterImpl)
    torch.nn.Module.__init__(impl)
    impl.num_index_heads = 1
    impl.index_head_dim = 128
    impl.num_kv_heads = 1
    impl.block_size = 128
    impl.topk_blocks = 16
    impl.init_blocks = 0
    impl.local_blocks = 0
    impl.attend_layer_name = "attend"
    impl.index_cache = SimpleNamespace(
        prefix="index", kv_cache=torch.empty(1, 128, 128)
    )
    impl.topk_indices_buffer = torch.empty(1, 3, 16, dtype=torch.int32)
    impl.sparse_bt_buffer = torch.empty(3, 128, dtype=torch.int32)
    impl.sparse_ctx_buffer = torch.empty(3, dtype=torch.int32)
    monkeypatch.setattr(
        impl,
        "_new_score",
        lambda rows, max_seq_len: torch.empty(1, rows, 64),
    )
    monkeypatch.setattr(
        aiter_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"index": index_md, "attend": attend_md}),
    )

    decode_topk, prefill_topk = impl(torch.empty(3, 128))

    assert decode_topk is not None and prefill_topk is not None
    assert len(calls) == 2
    assert calls[0].sparse_bt.data_ptr() == impl.sparse_bt_buffer[:1].data_ptr()
    assert calls[0].sparse_ctx.data_ptr() == impl.sparse_ctx_buffer[:1].data_ptr()
    assert calls[0].num_valid_pages is None
    assert calls[1].sparse_bt.data_ptr() == impl.sparse_bt_buffer[1:].data_ptr()
    assert calls[1].sparse_ctx.data_ptr() == impl.sparse_ctx_buffer[1:].data_ptr()
    assert calls[1].num_valid_pages is index_md.prefill_num_valid_pages
