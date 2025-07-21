#!/usr/bin/env python3
"""
Quick test script to verify modality-specific channel handling.
"""

import torch
import torch.nn as nn
from src.data.medmnist_data import MedMNISTDataset
from src.models.disentangled_conditional_vae import DisentangledConditionalVAE


def test_single_dataset():
    """Test a single dataset first."""
    print("Testing single dataset...")

    # Test chestmnist (should be grayscale)
    dataset = MedMNISTDataset(
        dataset_name="chestmnist",
        split="train",
        size=28,  # Use smaller size for faster download
        root="./data",
        download=True,
    )

    # Get a sample
    image, label, modality, modality_idx = dataset[0]
    expected_channels = dataset.target_channels
    actual_channels = image.shape[0]

    print(f"chestmnist: Expected {expected_channels} channels, got {actual_channels}")
    print(f"  Modality index: {modality_idx.item()}")
    print(f"  Image shape: {image.shape}")

    assert actual_channels == expected_channels, f"Channel mismatch for chestmnist"
    print("✓ Single dataset test passed!")


def test_model_quick():
    """Test model with smaller setup."""
    print("\nTesting model with small setup...")

    # Create smaller model
    model = DisentangledConditionalVAE(
        num_modalities=5,
        shared_latent_dim=4,
        modality_latent_dim=4,
        resolution=28,
        hidden_channels=16,
        ch_mult=(1, 2),
        num_res_blocks=1,
    )

    print(f"Model max channels: {model.input_channels}")
    print(f"Modality channels: {model.modality_channels}")

    # Test with grayscale input
    batch_size = 2
    height, width = 28, 28
    x_gray = torch.randn(batch_size, 1, height, width)
    modality_indices = torch.tensor([0, 3])  # chestmnist, pneumoniamnist

    print(f"Input shape: {x_gray.shape}")
    outputs = model(x_gray, modality_indices)
    print(f"Output shape: {outputs['reconstruction'].shape}")

    # Verify output has same channels as input
    assert (
        outputs["reconstruction"].shape[1] == 1
    ), "Output should be 1-channel for grayscale"
    print("✓ Model test passed!")


if __name__ == "__main__":
    print("Quick test of modality-specific channel handling...\n")

    try:
        test_single_dataset()
        test_model_quick()

        print("\n🎉 Quick tests passed!")
        print("Key findings:")
        print("- chestmnist is correctly processed as 1-channel grayscale")
        print("- Model properly handles channel projection")
        print("- Output maintains input channel count")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
