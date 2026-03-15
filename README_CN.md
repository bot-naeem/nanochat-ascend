# nanochat-ascend

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat-ascend 是 [nanochat](https://github.com/karpathy/nanochat) 的华为昇腾 NPU 适配版本，旨在支持在华为昇腾 NPU 设备上运行大语言模型。本项目在保持与原始 nanochat 完全兼容的同时，新增了对昇腾 NPU 加速器的原生支持。

## 项目目标

本项目的主要目标是将 nanochat 移植到华为昇腾 NPU 平台，实现基于国产 AI 加速器的大语言模型训练与推理。适配工作采用**最小侵入式**修改策略，确保与原始 nanochat 的向后兼容性，同时添加 NPU 支持。

## 核心特性

- **NPU Flash Attention**：使用 `torch_npu.npu_fusion_attention` 实现原生 NPU 闪光注意力
- **滑动窗口支持**：NPU Flash Attention 支持滑动窗口注意力机制
- **自动设备检测**：自动检测 NPU 并选择最优数据类型 (bf16)
- **透明回退**：无 NPU 时自动回退到 PyTorch SDPA
- **完全兼容**：保持与原始 nanochat 代码库的兼容性

## 快速开始

### 环境要求

```bash
# 安装依赖
pip install -r requirement.txt

# 安装 torch_npu (华为昇腾 PyTorch 插件)
pip install torch_npu
```

### 在 NPU 上运行

```bash
# 训练
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12

# 推理
python -m scripts.chat_web
```

## NPU 适配详情

### 核心修改

适配工作以最小改动添加了 NPU 支持：

| 组件 | 修改内容 |
|------|----------|
| `flash_attention.py` | 使用 `torch_npu.npu_fusion_attention` 添加 NPU Flash Attention，支持滑动窗口 |
| `common.py` | 添加 NPU 检测和 bf16 数据类型选择 |
| `gpt.py` | 添加带回退保护的 `torch_npu` 导入 |
| `optim.py` | 添加 NPU 支持，`torch.compile` 使用 `backend="eager"` |
| 训练脚本 | 添加 NPU 设备检测和日志输出 |

### NPU Flash Attention 实现

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

### 自动设备检测

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

### 设备检测优先级

系统按以下优先级自动检测硬件：
1. **NPU** (华为昇腾) - 如果 `torch_npu` 可用
2. **CUDA** (英伟达) - 如果 CUDA 可用
3. **MPS** (苹果硅芯片)
4. **CPU**

### 数据精度

| 硬件 | 默认数据类型 |
|------|-------------|
| NPU | bfloat16 |
| CUDA SM 80+ (A100, H100, ...) | bfloat16 |
| CUDA SM < 80 (V100, T4, ...) | float32 |
| CPU / MPS | float32 |

可使用 `NANOCHAT_DTYPE` 环境变量覆盖默认设置：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"
```

## 适配策略

nanochat-ascend 采用**最小侵入式**修改：

1. **统一入口**：所有脚本和库文件添加 `torch_npu` 导入（带 try-except 保护）
2. **自动检测**：在 `common.py` 中自动检测 NPU 并选择 bf16
3. **核心突破**：`flash_attention.py` 新增 NPU Flash Attention 后端
4. **透明回退**：无 NPU 时自动回退到 SDPA

这种设计保持了与原始项目的兼容性，同时实现了 NPU 支持。

## 性能基准

在 NPU 硬件上的训练基准测试请参阅 [dev/LEADERBOARD.md](dev/LEADERBOARD.md)。

## 目录结构

```
.
├── LICENSE
├── README.md
├── README_CN.md
├── dev/
│   ├── gen_synthetic_data.py
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py
├── nanochat/
│   ├── __init__.py
│   ├── checkpoint_manager.py
│   ├── common.py             # 已添加 NPU 检测
│   ├── core_eval.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── engine.py
│   ├── execution.py
│   ├── flash_attention.py    # 已添加 NPU Flash Attention
│   ├── fp8.py
│   ├── gpt.py               # 已添加 NPU 支持
│   ├── logo.svg
│   ├── loss_eval.py
│   ├── optim.py             # 已添加 NPU 支持
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
│   ├── base_train.py        # 已添加 NPU 支持
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

## 致谢

- [nanochat](https://github.com/karpathy/nanochat) - Andrej Karpathy
- 华为昇腾 NPU 团队 - torch_npu 支持
- 原始 nanochat 贡献者

## 许可证

MIT
