# Training Baseline

## Baseline Definition

The default training-efficiency baseline for this repository is:

- Hardware: 1x Ascend910B2
- Model: `d12`
- Attention backend: `torch_npu.npu_fusion_attention`
- Window pattern: `L`
- Precision: `bf16`
- `device-batch-size=16`
- `max-seq-len=2048`
- Effective total batch size: `32768`
- Gradient accumulation steps: `1`

This baseline should be used for future performance comparisons unless a document explicitly states a different benchmark target.

## Baseline Metrics

Use the following values as the reference baseline:

| Metric | Baseline |
|--------|----------|
| Step time (`dt`) | `~414-415 ms/step` |
| Throughput (`tok/sec`) | `~79k tok/sec` |
| `bf16_mfu` | `~21.9` |
| Peak memory | `~30.1 GiB` |
