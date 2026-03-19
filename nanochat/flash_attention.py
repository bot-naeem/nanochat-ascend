"""
Unified Flash Attention interface with automatic NPU/FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly.

Priority:
1. NPU Flash Attention (torch_npu.npu_fusion_attention) on Ascend NPU
2. Flash Attention 3 on Hopper GPUs (sm90)
3. PyTorch SDPA fallback on other devices

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F
import math


# =============================================================================
# Detection: Check device and available backends
# =============================================================================

def _is_npu_available():
    """Check if running on Huawei Ascend NPU."""
    try:
        import torch_npu
        return torch_npu.npu.is_available()
    except ImportError:
        return False

def _has_npu_flash_attention():
    """Check if NPU Flash Attention API is available."""
    try:
        import torch_npu
        return hasattr(torch_npu, 'npu_fusion_attention')
    except ImportError:
        return False

IS_NPU = _is_npu_available()
HAS_NPU_FA = _has_npu_flash_attention()


def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

_override_impl = None


def _resolve_backend():
    """
    Decide which backend to use.
    Returns: 'npu', 'fa3', or 'sdpa'
    """
    if _override_impl:
        return _override_impl
    
    if IS_NPU and HAS_NPU_FA:
        return 'npu'
    
    if HAS_FA3:
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return 'fa3'
    
    return 'sdpa'

BACKEND = _resolve_backend()
USE_NPU_FA = BACKEND == 'npu'
USE_FA3 = BACKEND == 'fa3'


# =============================================================================
# NPU Flash Attention helpers
# =============================================================================
def _npu_flash_attention(q, k, v, causal=True, window_size=(-1, -1)):
    """
    NPU Flash Attention using torch_npu.npu_fusion_attention with BSH layout.
    
    Uses BSH (Batch, Sequence, Hidden) layout to avoid costly transpose operations.
    reshape/view is zero-copy (just changes strides), while transpose triggers contiguous().
    
    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.
    
    Returns:
        Output tensor of shape (B, T, H, D)
    """
    import torch_npu
    
    B, T, H, D = q.shape
    scale = 1.0 / math.sqrt(D)
    
    # BSH layout: reshape (B, T, H, D) -> (B, T, H*D), zero-copy operation
    q_bsh = q.reshape(B, T, H * D)
    k_bsh = k.reshape(B, T, H * D)
    v_bsh = v.reshape(B, T, H * D)
    
    left_window = window_size[0] if window_size else -1
    
    if left_window > 0 and left_window < T:
        sparse_mode = 4
        pre_tockens = left_window
        next_tockens = 0
    elif causal:
        sparse_mode = 2
        pre_tockens = 2147483647
        next_tockens = 0
    else:
        sparse_mode = 0
        pre_tockens = 2147483647
        next_tockens = 2147483647
    
    out = torch_npu.npu_fusion_attention(
        q_bsh, k_bsh, v_bsh,
        head_num=H,
        input_layout="BSH",
        scale=scale,
        keep_prob=1.0,
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        sparse_mode=sparse_mode,
    )
    
    if isinstance(out, tuple):
        out = out[0]
    
    # BSH output (B, T, H*D) -> (B, T, H, D), reshape handles non-contiguous layouts safely
    return out.reshape(B, T, H, D)


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_NPU_FA:
        return _npu_flash_attention(q, k, v, causal=causal, window_size=window_size)
    
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k.to(k_cache.dtype)
        v_cache[:, pos:pos+T_new, :, :] = v.to(v_cache.dtype)

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
