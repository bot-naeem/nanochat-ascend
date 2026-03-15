# nanochat-ascend

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat-ascend is a Huawei Ascend NPU adapter for [nanochat](https://github.com/karpathy/nanochat), designed to run LLMs on Huawei Ascend NPU devices. This project maintains full compatibility with the original nanochat while adding native support for Ascend NPU accelerators.

## Project Goal

The primary goal of this project is to port nanochat to Huawei Ascend NPU platforms, enabling LLM training and inference on Chinese domestic AI accelerators. The adaptation follows a **minimal intrusion** approach, ensuring backward compatibility with the original nanochat while adding NPU support.

## Key Features

- **NPU Flash Attention**: Native implementation using `torch_npu.npu_fusion_attention`
- **Sliding Window Support**: NPU Flash Attention with sliding window attention mechanism
- **Automatic Device Detection**: Automatically detects NPU and selects optimal dtype (bf16)
- **Transparent Fallback**: Falls back to PyTorch SDPA when NPU is not available
- **Full Compatibility**: Maintains compatibility with original nanochat codebase

## Quick Start

### Environment Setup

This project requires a conda environment with Huawei Ascend NPU support.

```bash
# Create conda environment
conda create -n npu python=3.10

# Activate environment
conda activate npu

# Install PyTorch with NPU support (choose one):

# Option 1: Install from official source
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Option 2: Install torch-npu (Huawei Ascend)
pip install torch-npu

# Install project dependencies
pip install -r requirements.txt
```

### Running on NPU

```bash
# Training
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12

# Inference
python -m scripts.chat_web
```

## NPU Adaptation Details

### Core Changes

The adaptation adds NPU support across the codebase with minimal changes:

| Component | Change |
|-----------|--------|
| `flash_attention.py` | Added NPU Flash Attention using `torch_npu.npu_fusion_attention`, sliding window support |
| `common.py` | Added NPU detection and bf16 dtype selection |
| `gpt.py` | Added `torch_npu` import with fallback |
| `optim.py` | Added NPU support with `backend="eager"` for torch.compile |
| Training scripts | Added NPU device detection and logging |

### NPU Flash Attention Implementation

```python
def _is_npu_available():
    try:
        import torch_npu
        return torch_npu.npu.is_available()
    except ImportError:
        return False

def _npu_flash_attention(q, k, v, causal=True, window_size=(-1, -1)):
    import torch_npu
    out = torch_npu.npu_fusion_attention(
        q_npu, k_npu, v_npu,
        head_num=H,
        input_layout="BNSD",
        scale=scale,
        keep_prob=1.0,
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        sparse_mode=sparse_mode,
    )
```

### Automatic Device Detection

```python
def autodetect_device_type():
    if HAS_TORCH_NPU and hasattr(torch, 'npu') and torch.npu.is_available():
        device_type = "npu"
    elif torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
```

### Device Detection Priority

The system automatically detects hardware in this priority order:
1. **NPU** (Huawei Ascend) - if `torch_npu` is available
2. **CUDA** (NVIDIA) - if CUDA is available
3. **MPS** (Apple Silicon)
4. **CPU**

### Precision

| Hardware | Default dtype |
|----------|---------------|
| NPU | bfloat16 |
| CUDA SM 80+ (A100, H100, ...) | bfloat16 |
| CUDA SM < 80 (V100, T4, ...) | float32 |
| CPU / MPS | float32 |

You can override with `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"
```

## Adaptation Strategy

nanochat-ascend uses **minimal intrusion** modifications:

1. **Unified Entry**: All scripts and library files add `torch_npu` import (with try-except protection)
2. **Automatic Detection**: NPU is automatically detected in `common.py` and bf16 is selected
3. **Core Breakthrough**: `flash_attention.py` adds NPU Flash Attention backend
4. **Transparent Fallback**: Falls back to SDPA when NPU is not available

This design maintains compatibility with the original project while adding NPU support.

## Benchmark Results

Performance benchmarking on NPU hardware is tracked in [dev/LEADERBOARD.md](dev/LEADERBOARD.md).

## File Structure

```
.
├── LICENSE
├── README.md
├── README_CN.md              # Chinese documentation
├── dev/
│   ├── gen_synthetic_data.py
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py
├── nanochat/
│   ├── __init__.py
│   ├── checkpoint_manager.py
│   ├── common.py             # NPU detection added
│   ├── core_eval.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── engine.py
│   ├── execution.py
│   ├── flash_attention.py    # NPU Flash Attention added
│   ├── fp8.py
│   ├── gpt.py               # NPU support added
│   ├── logo.svg
│   ├── loss_eval.py
│   ├── optim.py             # NPU support added
│   ├── report.py
│   └── tokenizer.py
├── pyproject.toml
├── runs/
│   ├── miniseries.sh
│   ├── runcpu.sh
│   ├── scaling_laws.sh
│   └── speedrun.sh
├── scripts/
│   ├── base_eval.py
│   ├── base_train.py         # NPU support added
│   ├── chat_cli.py
│   ├── chat_eval.py
│   ├── chat_rl.py
│   ├── chat_sft.py
│   ├── chat_web.py
│   ├── tok_eval.py
│   └── tok_train.py
├── tasks/
│   ├── arc.py
│   ├── common.py
│   ├── customjson.py
│   ├── gsm8k.py
│   ├── humaneval.py
│   ├── mmlu.py
│   ├── smoltalk.py
│   └── spellingbee.py
├── tests/
│   ├── test_attention_fallback.py
│   └── test_engine.py
└── uv.lock
```

## Acknowledgements

- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- Huawei Ascend NPU team for torch_npu support
- Original nanochat contributors

## License

MIT
