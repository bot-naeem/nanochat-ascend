# nanochat vs nanochat-ascend 代码差异报告

## 概述

本文档对比 `nanochat`（原始版本）与 `nanochat-ascend`（华为昇腾NPU适配版本）之间的代码差异。

---

## 1. 核心差异：NPU Flash Attention

### flash_attention.py

**原始版本** (`nanochat/nanochat/flash_attention.py`):
- 仅支持 Flash Attention 3 (Hopper GPU) 和 PyTorch SDPA
- 无 NPU 支持

**适配版本** (`nanochat-ascend/nanochat/flash_attention.py`):
- 新增 NPU 检测函数
- 新增 `_npu_flash_attention()` 函数，使用 `torch_npu.npu_fusion_attention`
- 优先级：NPU FA > FA3 > SDPA

```python
# 新增代码
def _is_npu_available():
    try:
        import torch_npu
        return torch_npu.npu.is_available()
    except ImportError:
        return False

def _npu_flash_attention(q, k, v, causal=True, window_size=(-1, -1)):
    import torch_npu
    # 使用 torch_npu.npu_fusion_attention
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

---

## 2. common.py 差异

### 设备检测和 dtype 选择

**原始版本**:
```python
def _detect_compute_dtype():
    if torch.cuda.is_available():
        # CUDA 检测逻辑
        ...
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
```

**适配版本**:
```python
def _detect_compute_dtype():
    # 新增 NPU 检测
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.bfloat16, "auto-detected: NPU (bf16 supported)"
    if torch.cuda.is_available():
        # CUDA 检测逻辑
        ...
```

### 导入差异

**原始版本**:
```python
import torch
import torch.distributed as dist
```

**适配版本**:
```python
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    HAS_TORCH_NPU = False
import torch.distributed as dist
```

### autodetect_device_type() 差异

**原始版本**:
```python
def autodetect_device_type():
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
```

**适配版本**:
```python
def autodetect_device_type():
    # 新增 NPU 优先检测
    if HAS_TORCH_NPU and hasattr(torch, 'npu') and torch.npu.is_available():
        device_type = "npu"
    elif torch.cuda.is_available():
        device_type = "cuda"
    ...
```

---

## 3. gpt.py 差异

**原始版本**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

**适配版本**:
```python
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False
import torch.nn as nn
import torch.nn.functional as F
```

---

## 4. optim.py 差异

**原始版本**:
```python
import torch
import torch.distributed as dist
```

**适配版本**:
```python
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass
import torch.distributed as dist
```

同时 `@torch.compile` 装饰器增加了 `backend="eager"` 参数。

---

## 5. 训练脚本差异 (base_train.py)

### 导入差异

**原始版本**:
```python
import wandb
import torch
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3
```

**适配版本**:
```python
import wandb
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    HAS_TORCH_NPU = False
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3, HAS_NPU_FA, USE_NPU_FA
```

### 日志输出差异

**原始版本**:
```python
if using_fa3:
    print0("✓ Using Flash Attention 3 ...")
else:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
```

**适配版本**:
```python
if using_fa3:
    print0("✓ Using Flash Attention 3 ...")
elif using_npu_fa:
    print0("✓ Using NPU Flash Attention (torch_npu.npu_fusion_attention) ...")
else:
    is_npu = hasattr(torch, 'npu') and torch.npu.is_available()
    if is_npu:
        print0("INFO: Running on NPU, using torch_npu optimized SDPA")
```

---

## 6. 其他库文件差异

以下文件在适配版本中都添加了 `torch_npu` 导入：

| 文件 | 变更 |
|------|------|
| `checkpoint_manager.py` | 新增 torch_npu 导入 |
| `dataloader.py` | 新增 torch_npu 导入 |
| `fp8.py` | 新增 torch_npu 导入 |
| `core_eval.py` | 新增 torch_npu 导入 |
| `loss_eval.py` | 新增 torch_npu 导入 |
| `report.py` | 新增 torch_npu 导入 |
| `engine.py` | 新增 torch_npu 导入 |

---

## 7. 脚本文件差异

| 脚本文件 | 变更 |
|---------|------|
| `scripts/base_train.py` | 新增 torch_npu 导入 + NPU FA 日志 |
| `scripts/base_eval.py` | 新增 torch_npu 导入 |
| `scripts/chat_sft.py` | 新增 torch_npu 导入 + NPU FA 检测 |
| `scripts/chat_rl.py` | 新增 torch_npu 导入 |
| `scripts/chat_eval.py` | 新增 torch_npu 导入 |
| `scripts/chat_cli.py` | 新增 torch_npu 导入 + npu 设备选项 |
| `scripts/chat_web.py` | 新增 torch_npu 导入 |
| `scripts/tok_train.py` | 新增 torch_npu 导入 |

---

## 8. 测试文件差异

| 测试文件 | 变更 |
|---------|------|
| `tests/test_engine.py` | 新增 torch_npu 导入 |
| `tests/test_attention_fallback.py` | 新增 torch_npu 导入 + HAS_NPU_FA |

---

## 9. NPU 适配策略总结

nanochat-ascend 采用**最小侵入式**修改：

1. **统一入口**：所有脚本和库文件添加 `torch_npu` 导入（带 try-except 保护）
2. **自动检测**：在 `common.py` 中自动检测 NPU 并选择 bf16
3. **核心突破**：`flash_attention.py` 新增 NPU Flash Attention 后端
4. **透明回退**：无 NPU 时自动回退到 SDPA

这种设计保持了与原始项目的兼容性，同时实现了 NPU 支持。
