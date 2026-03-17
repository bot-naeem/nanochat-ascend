"""
Test for torch_npu.npu_fusion_attention interface.

This test verifies that the NPU fusion attention API works correctly
on Huawei Ascend NPU devices.

Usage:
    conda activate npu
    python test_npu_fa.py

Or with specific device:
    conda run -n npu python test_npu_fa.py --device 0
"""
import argparse
import sys

import torch


def test_npu_fusion_attention():
    """Test npu_fusion_attention with various configurations."""
    print("=" * 60)
    print("Testing torch_npu.npu_fusion_attention")
    print("=" * 60)

    # Check if torch_npu is available
    try:
        import torch_npu
        print("[OK] torch_npu imported successfully")
    except ImportError as e:
        print(f"[FAIL] Cannot import torch_npu: {e}")
        return False

    # Check if NPU is available
    if not hasattr(torch_npu, 'npu'):
        print("[FAIL] torch_npu.npu not available")
        return False

    # Set device
    device = torch_npu.npu.device_count() > 0
    if not device:
        print("[FAIL] No NPU device found")
        return False

    print(f"[OK] NPU device found")

    # Test configurations
    configs = [
        # (batch, seq_len, head_num, head_dim, input_layout)
        (1, 32, 8, 64, "BSND"),  # Batch, Seq, Num_heads, Head_dim
        (1, 32, 8, 64, "BNSD"),  # Batch, Num_heads, Seq, Head_dim
        (2, 16, 4, 32, "BSND"),
        (1, 8, 2, 16, "BSND"),
    ]

    success_count = 0

    for i, (B, S, H, D, layout) in enumerate(configs):
        print(f"\n--- Test config {i+1}: B={B}, S={S}, H={H}, D={D}, layout={layout} ---")

        try:
            # Create input tensors based on layout
            if layout == "BSND":
                # (Batch, Seq, Num_heads, Head_dim)
                q = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
                k = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
                v = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
            elif layout == "BNSD":
                # (Batch, Num_heads, Seq, Head_dim)
                q = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
                k = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
                v = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
            else:
                print(f"[SKIP] Unsupported layout: {layout}")
                continue

            # Call npu_fusion_attention
            # Note: The API signature may vary, trying different parameter styles
            try:
                # Try with common parameters
                output = torch_npu.npu_fusion_attention(
                    q, k, v,
                    head_num=H,
                    input_layout=layout,
                    scale=1.0,
                    keep_prob=1.0,
                )
            except TypeError as e:
                # Try alternative parameter names
                print(f"  [WARN] First attempt failed: {e}")
                try:
                    output = torch_npu.npu_fusion_attention(
                        q, k, v,
                        head_num=H,
                        input_layout=layout,
                    )
                except Exception as e2:
                    print(f"  [WARN] Second attempt failed: {e2}")
                    # Try minimal call
                    output = torch_npu.npu_fusion_attention(q, k, v)

            # Handle tuple output (some APIs return tuple with output and workspace)
            if isinstance(output, tuple):
                output_tensor = output[0] if len(output) > 0 else output
                print(f"  [INFO] Output is tuple with {len(output)} elements")
            else:
                output_tensor = output

            # Verify output shape
            if layout == "BSND":
                expected_shape = (B, S, H, D)
            else:  # BNSD
                expected_shape = (B, H, S, D)

            if output_tensor.shape == expected_shape:
                print(f"[OK] Output shape: {output_tensor.shape}")
                success_count += 1
            else:
                print(f"[FAIL] Output shape {output_tensor.shape} != expected {expected_shape}")

        except Exception as e:
            print(f"[FAIL] Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{len(configs)} tests passed")
    print("=" * 60)

    return success_count > 0


def test_npu_fusion_attention_v2():
    """Test npu_fusion_attention_v2 which may have different API."""
    print("\n" + "=" * 60)
    print("Testing torch_npu.npu_fusion_attention_v2")
    print("=" * 60)

    try:
        import torch_npu
    except ImportError:
        print("[SKIP] torch_npu not available")
        return False

    # Check if v2 is available
    if not hasattr(torch_npu, 'npu_fusion_attention_v2'):
        print("[SKIP] npu_fusion_attention_v2 not available")
        return False

    try:
        B, S, H, D = 1, 16, 4, 32
        q = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
        k = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
        v = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")

        output = torch_npu.npu_fusion_attention_v2(
            q, k, v,
            head_num=H,
            input_layout="BSND",
        )

        # Handle tuple output
        if isinstance(output, tuple):
            output_tensor = output[0]
            print(f"[OK] npu_fusion_attention_v2 works, output shape: {output_tensor.shape}")
        else:
            print(f"[OK] npu_fusion_attention_v2 works, output shape: {output.shape}")
        return True

    except Exception as e:
        print(f"[FAIL] npu_fusion_attention_v2 error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_npu_fusion_attention_grad():
    """Test npu_fusion_attention_grad (backward pass) using PyTorch autograd."""
    print("\n" + "=" * 60)
    print("Testing torch_npu.npu_fusion_attention_grad (via autograd)")
    print("=" * 60)

    try:
        import torch_npu
    except ImportError:
        print("[SKIP] torch_npu not available")
        return False

    # Check if grad is available
    if not hasattr(torch_npu, 'npu_fusion_attention_grad'):
        print("[SKIP] npu_fusion_attention_grad not available")
        return False

    success_count = 0
    configs = [
        (1, 32, 8, 64, "BSND"),
        (1, 32, 8, 64, "BNSD"),
        (2, 16, 4, 32, "BSND"),
    ]

    for i, (B, S, H, D, layout) in enumerate(configs):
        print(f"\n--- Test config {i+1}: B={B}, S={S}, H={H}, D={D}, layout={layout} ---")

        try:
            # Create input tensors with gradients enabled
            if layout == "BSND":
                q = torch.randn(B, S, H, D, dtype=torch.float16, device="npu").requires_grad_(True)
                k = torch.randn(B, S, H, D, dtype=torch.float16, device="npu").requires_grad_(True)
                v = torch.randn(B, S, H, D, dtype=torch.float16, device="npu").requires_grad_(True)
                expected_shape = (B, S, H, D)
            else:  # BNSD
                q = torch.randn(B, H, S, D, dtype=torch.float16, device="npu").requires_grad_(True)
                k = torch.randn(B, H, S, D, dtype=torch.float16, device="npu").requires_grad_(True)
                v = torch.randn(B, H, S, D, dtype=torch.float16, device="npu").requires_grad_(True)
                expected_shape = (B, H, S, D)

            # Forward pass
            output = torch_npu.npu_fusion_attention(
                q, k, v,
                head_num=H,
                input_layout=layout,
                scale=1.0,
                keep_prob=1.0,
            )

            # Handle tuple output
            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output

            # Create loss and compute gradients using PyTorch autograd
            loss = output_tensor.sum()
            loss.backward()

            # Check gradients
            if q.grad is not None and k.grad is not None and v.grad is not None:
                if (q.grad.abs().sum() > 0 and k.grad.abs().sum() > 0 and v.grad.abs().sum() > 0):
                    print(f"[OK] Autograd gradients computed: q_grad={q.grad.shape}, k_grad={k.grad.shape}, v_grad={v.grad.shape}")
                    success_count += 1
                else:
                    print(f"[FAIL] Some gradients are all zeros")
            else:
                print(f"[FAIL] Gradients not computed (None)")

        except Exception as e:
            print(f"[FAIL] Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{len(configs)} tests passed")
    print("=" * 60)

    return success_count > 0


def test_npu_fusion_attention_grad_v2():
    """Test npu_fusion_attention_grad_v2 (backward pass v2) using autograd."""
    print("\n" + "=" * 60)
    print("Testing torch_npu.npu_fusion_attention_grad_v2 (via autograd)")
    print("=" * 60)

    try:
        import torch_npu
    except ImportError:
        print("[SKIP] torch_npu not available")
        return False

    # Check if grad v2 is available
    if not hasattr(torch_npu, 'npu_fusion_attention_grad_v2'):
        print("[SKIP] npu_fusion_attention_grad_v2 not available")
        return False

    try:
        B, S, H, D = 1, 16, 4, 32
        layout = "BSND"

        q = torch.randn(B, S, H, D, dtype=torch.float16, device="npu").requires_grad_(True)
        k = torch.randn(B, S, H, D, dtype=torch.float16, device="npu").requires_grad_(True)
        v = torch.randn(B, S, H, D, dtype=torch.float16, device="npu").requires_grad_(True)

        # Forward pass
        output = torch_npu.npu_fusion_attention_v2(
            q, k, v,
            head_num=H,
            input_layout=layout,
        )

        # Handle tuple output
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        # Use PyTorch autograd to compute gradients
        loss = output_tensor.sum()
        loss.backward()

        # Check gradients
        if q.grad is not None and k.grad is not None and v.grad is not None:
            if (q.grad.abs().sum() > 0 and k.grad.abs().sum() > 0 and v.grad.abs().sum() > 0):
                print(f"[OK] Autograd gradients computed: q_grad={q.grad.shape}, k_grad={k.grad.shape}, v_grad={v.grad.shape}")
                return True

        print(f"[FAIL] Gradients not computed correctly")
        return False

    except Exception as e:
        print(f"[FAIL] npu_fusion_attention_grad_v2 error: {e}")
        import traceback
        traceback.print_exc()
        return False


def explore_npu_attention_api():
    """Explore available attention-related APIs in torch_npu."""
    print("\n" + "=" * 60)
    print("Exploring torch_npu attention APIs")
    print("=" * 60)

    import torch_npu

    # Find all attention-related functions
    attention_funcs = [
        name for name in dir(torch_npu)
        if 'attention' in name.lower() or 'atten' in name.lower()
    ]

    print(f"Found attention-related functions: {attention_funcs}")

    # Also check npu module
    if hasattr(torch_npu, 'npu'):
        npu_attention = [
            name for name in dir(torch_npu.npu)
            if 'attention' in name.lower() or 'atten' in name.lower()
        ]
        print(f"Found in torch_npu.npu: {npu_attention}")


def main():
    import torch_npu
    parser = argparse.ArgumentParser(description="Test NPU fusion attention")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
    parser.add_argument("--explore", action="store_true", help="Explore available APIs")
    args = parser.parse_args()

    if args.device is not None:
        torch_npu.npu.set_device(args.device)

    # First explore available APIs
    explore_npu_attention_api()

    # Run tests
    test_npu_fusion_attention()
    test_npu_fusion_attention_v2()
    test_npu_fusion_attention_grad()
    test_npu_fusion_attention_grad_v2()


if __name__ == "__main__":
    main()
