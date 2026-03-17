# nanochat-ascend

nanochat-ascend 是 [nanochat](https://github.com/karpathy/nanochat) 的华为昇腾 NPU 适配版本。

## 与 nanochat 的区别

| 组件 | nanochat | nanochat-ascend |
|------|----------|-----------------|
| 设备 | CUDA/GPU | **NPU (华为昇腾)** |
| Attention | Flash Attention 3  | **NPU Flash Attention** |
| 数据类型 | 自动检测 | **NPU + bf16** 自动检测 |

## 主要修改

- `flash_attention.py`: 添加 NPU Flash Attention (`torch_npu.npu_fusion_attention`)
- `common.py`: 添加 NPU 检测和 bf16 选型
- 所有模块: 添加 `torch_npu` 导入（带 try-except 回退）

## 优化历程

到目前为止，这个项目的优化工作大致分成五个阶段：

1. 昇腾基础适配：补齐 NPU 设备检测、初始化、bf16 选型，以及训练主流程支持。
2. Attention 后端切换：从 `torch_npu` 自动优化的 SDPA 切到 `torch_npu.npu_fusion_attention`。
3. 训练配置调优：在 NPU 上切到更合适的配置，例如 `window-pattern=L` 和更大的 `device-batch-size`。
4. Kernel 路径优化：
   - 将 NPU Flash Attention 从 `BNSD` 布局改为 `BSH` 布局，减少 `transpose` / 内存重排开销
   - 将拆分的 `F.rms_norm` 改为融合的 `torch.ops.npu.npu_rms_norm`
5. Optimizer 和工程完善：
   - 条件接入 `NpuFusedAdamW`
   - 为 delegate optimizer 补齐 `state_dict()` / `load_state_dict()`，保证 checkpoint 恢复语义正确


## 快速开始

```bash
# 创建 conda 环境
conda create -n npu python=3.10
conda activate npu

# 安装支持 NPU 的 PyTorch
pip install torch-npu

# 安装依赖
pip install -r requirements.txt

# 全流程尝试
sh runs/speedrun.sh
```

## 环境要求

- Python 3.10+
- torch-npu
