#!/usr/bin/env python3
"""
Very simple test to verify basic functionality.
"""

import torch

print("Testing basic torch functionality...")
print(f"PyTorch version: {torch.__version__}")

# Test basic tensor operations
x = torch.randn(2, 1, 28, 28)
print(f"Created tensor with shape: {x.shape}")

# Test importing our modules
try:
    from src.data.medmnist_data import MedMNISTDataset

    print("✓ Successfully imported MedMNISTDataset")
except Exception as e:
    print(f"✗ Failed to import MedMNISTDataset: {e}")

try:
    from src.models.disentangled_conditional_vae import DisentangledConditionalVAE

    print("✓ Successfully imported DisentangledConditionalVAE")
except Exception as e:
    print(f"✗ Failed to import DisentangledConditionalVAE: {e}")

print("Basic test completed!")
