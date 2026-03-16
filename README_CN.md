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
   - 引入 `NANOCHAT_BASE_DIR`，让 cache / checkpoint 可以写到更大的磁盘目录

## 训练对比数据

以下是单卡 Ascend 910B 上，这次移植过程中比较关键的训练数据：

| 阶段 | 主要改动 | 典型配置 | 平均 tok/sec | 平均 dt | 说明 |
|------|----------|----------|-------------:|--------:|------|
| 初始基线 | SDPA（`torch_npu` 自动优化） | `seq=1024, bs=8, total=32768` | ~58,000 | ~560 ms | 最早跑通的 NPU 基线 |
| Attention 优化 | `npu_fusion_attention` | `seq=1024, bs=8, total=32768` | ~59,500 | ~550 ms | 比 SDPA 快约 3% |
| Batch 调优 | 提升 micro-batch | `seq=1024, bs=48, total=49152` | ~65,200 | ~753 ms | 比 `bs=8` 基线快约 12% |
| Kernel 优化 | `BSH` Flash Attention + 融合 NPU RMSNorm | `seq=1024, bs=48, total=49152` | ~84,600 | ~581 ms | 比旧 `bs=48` 基线快约 30% |

额外的 `seq=2048` batch 梯度探测结果如下：

| 配置 | 状态 | 平均 tok/sec | 平均 dt | 峰值显存 |
|------|------|-------------:|--------:|---------:|
| `seq=2048, bs=16` | OK | 78,935 | 415.12 ms | 30,139.92 MiB |
| `seq=2048, bs=20` | OK | 80,058 | 511.63 ms | 37,120.49 MiB |
| `seq=2048, bs=24` | OK | 80,294 | 612.15 ms | 44,118.03 MiB |
| `seq=2048, bs=28` | OK | 80,225 | 714.79 ms | 51,104.09 MiB |
| `seq=2048, bs=30` | OK | 80,311 | 765.02 ms | 54,588.63 MiB |
| `seq=2048, bs=32` | 完整训练 OOM | - | - | - |

实际建议：

- 如果当前目标是纯吞吐，`seq=1024, bs=48` 仍然是当前测到的最佳点。
- 如果目标是保持 nanochat 原生 `2048` 上下文，`seq=2048, bs=24` 是速度、显存余量和稳定性之间最均衡的选择。
- 在 `seq=2048` 下把 `bs` 从 `24` 继续往上加，吞吐几乎不再提升，只会明显增加显存压力。

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
- 华为昇腾 NPU
- torch-npu
