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

## Quick Start

```bash
# Create conda environment
conda create -n npu python=3.10
conda activate npu

# Install PyTorch with NPU support
pip install torch-npu

# Install dependencies
pip install -r requirements.txt

# Try
sh runs/speedrun.sh
```

## Requirements

- Python 3.10+
- Huawei Ascend NPU
- torch-npu

See [README_CN.md](README_CN.md) for Chinese documentation.
