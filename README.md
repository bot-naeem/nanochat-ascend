# nanochat-ascend

nanochat-ascend is a Huawei Ascend NPU port of [nanochat](https://github.com/karpathy/nanochat).

## Difference from nanochat

| Component | nanochat | nanochat-ascend |
|-----------|----------|-----------------|
| Device | CUDA/GPU | **NPU (Huawei Ascend)** |
| Attention | Flash Attention 3  | **NPU Flash Attention** |
| Dtype | auto-detect | **NPU + bf16** auto-detect |


## Key Changes

- `flash_attention.py`: Added NPU Flash Attention (`torch_npu.npu_fusion_attention`)
- `common.py`: Added NPU detection and bf16 selection
- All modules: Added `torch_npu` import with try-except fallback

## Optimization History

The project optimization work so far can be summarized in five stages:

1. Basic Ascend enablement: added NPU device detection, initialization, bf16 selection, and training/runtime support.
2. Attention backend migration: moved from `torch_npu`-optimized SDPA to `torch_npu.npu_fusion_attention`.
3. Training configuration tuning: switched to NPU-friendly settings such as `window-pattern=L` and larger device batch sizes.
4. Kernel-path optimization:
   - switched NPU Flash Attention from `BNSD` layout to `BSH` layout to reduce transpose/reorder overhead
   - replaced decomposed `F.rms_norm` with fused `torch.ops.npu.npu_rms_norm`
5. Optimizer and engineering fixes:
   - conditionally integrated `NpuFusedAdamW`
   - implemented `state_dict()` / `load_state_dict()` bridging for checkpoint-resume correctness
   - moved cache/checkpoint base dir to `NANOCHAT_BASE_DIR` so training can use a larger disk

## Training Comparison

Single-card Ascend 910B benchmark highlights collected during this port:

| Stage | Main Change | Typical Config | Avg tok/sec | Avg dt | Notes |
|------|-------------|----------------|------------:|-------:|-------|
| Baseline | SDPA (`torch_npu` optimized) | `seq=1024, bs=8, total=32768` | ~58,000 | ~560 ms | Initial working NPU baseline |
| Attention upgrade | `npu_fusion_attention` | `seq=1024, bs=8, total=32768` | ~59,500 | ~550 ms | ~3% faster than SDPA |
| Batch tuning | Larger micro-batch | `seq=1024, bs=48, total=49152` | ~65,200 | ~753 ms | ~12% faster than the `bs=8` baseline |
| Kernel optimization | `BSH` Flash Attention + fused NPU RMSNorm | `seq=1024, bs=48, total=49152` | ~84,600 | ~581 ms | ~30% faster than the previous `bs=48` baseline |

Additional `seq=2048` batch-size ladder results:

| Config | Status | Avg tok/sec | Avg dt | Peak Memory |
|--------|--------|------------:|-------:|------------:|
| `seq=2048, bs=16` | OK | 78,935 | 415.12 ms | 30,139.92 MiB |
| `seq=2048, bs=20` | OK | 80,058 | 511.63 ms | 37,120.49 MiB |
| `seq=2048, bs=24` | OK | 80,294 | 612.15 ms | 44,118.03 MiB |
| `seq=2048, bs=28` | OK | 80,225 | 714.79 ms | 51,104.09 MiB |
| `seq=2048, bs=30` | OK | 80,311 | 765.02 ms | 54,588.63 MiB |
| `seq=2048, bs=32` | OOM in full training | - | - | - |

Practical takeaways:

- For pure throughput on current code, `seq=1024, bs=48` is still the best measured point.
- For native `seq=2048` training, `bs=24` is the best balance between speed, memory headroom, and stability.
- Increasing `bs` beyond `24` at `seq=2048` provides almost no extra throughput, only higher memory pressure.

## Quick Start

```bash
# Create conda environment
conda create -n npu python=3.10
conda activate npu

# Install PyTorch with NPU support
pip install torch-npu

# Install dependencies
pip install -r requirements.txt

# Quick Experimentation
sh runs/speedrun.sh
```

## Requirements

- Python 3.10+
- Huawei Ascend NPU
- torch-npu

See [README_CN.md](README_CN.md) for Chinese documentation.
