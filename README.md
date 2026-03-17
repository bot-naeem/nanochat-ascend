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
