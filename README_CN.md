# nanochat-ascend

nanochat-ascend 是 [nanochat](https://github.com/karpathy/nanochat) 的华为昇腾 NPU 适配版本。

## 与 nanochat 的区别

| 组件 | nanochat | nanochat-ascend |
|------|----------|-----------------|
| 设备 | CUDA/GPU | **NPU (华为昇腾)** |
| Attention | Flash Attention 3 / SDPA | **NPU Flash Attention** + SDPA 回退 |
| 数据类型 | 自动检测 | **NPU + bf16** 自动检测 |
| 兼容性 | 仅 NVIDIA | **完全向后兼容** |

## 主要修改

- `flash_attention.py`: 添加 NPU Flash Attention (`torch_npu.npu_fusion_attention`)
- `common.py`: 添加 NPU 检测和 bf16 选型
- 所有模块: 添加 `torch_npu` 导入（带 try-except 回退）

## 快速开始

```bash
# 创建 conda 环境
conda create -n npu python=3.10
conda activate npu

# 安装支持 NPU 的 PyTorch
pip install torch-npu

# 安装依赖
pip install -r requirements.txt

# 训练
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12

# 推理
python -m scripts.chat_web
```

## 环境要求

- Python 3.10+
- 华为昇腾 NPU
- torch-npu
