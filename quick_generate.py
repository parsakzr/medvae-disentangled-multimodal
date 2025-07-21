#!/usr/bin/env python3
"""
Quick script to generate samples from trained VAE.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models import BaseVAE


def main():
    # Create model with same config as quick training
    print("Creating model...")
    model = BaseVAE(
        input_channels=1,
        latent_dim=16,
        hidden_channels=32,
        ch_mult=(1, 2, 4),
        num_res_blocks=1,
        attn_resolutions=[],
        dropout=0.1,
        resolution=28,
        use_linear_attn=False,
        attn_type="vanilla",
        double_z=True,
    )

    # Load checkpoint weights
    model_path = (
        "logs/checkpoints/multi_modal_cvae_quick-epoch=07-val/loss=0.036-v1.ckpt"
    )
    print(f"Loading weights from {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    # Extract model weights from Lightning checkpoint
    model_state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("model."):
            model_state_dict[key[6:]] = value  # Remove "model." prefix

    model.load_state_dict(model_state_dict)
    model.eval()

    # Use MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    # Generate samples with different "conditioning" by varying latent space regions
    num_samples = 64
    print(f"Generating {num_samples} samples with pseudo-conditioning...")

    with torch.no_grad():
        # Generate different types of samples using the built-in sample method
        samples_all = []

        # Type 1: Normal samples (standard sampling)
        samples_normal = model.sample(16, device)

        # Type 2: "Dense" samples (sample with modified random seed)
        torch.manual_seed(123)
        samples_dense = model.sample(16, device)

        # Type 3: "Clear" samples (different seed)
        torch.manual_seed(456)
        samples_clear = model.sample(16, device)

        # Type 4: "Shifted" samples (another seed)
        torch.manual_seed(789)
        samples_shifted = model.sample(16, device)

        # Reset seed for reproducibility
        torch.manual_seed(42)

        # Combine all samples
        all_samples = torch.cat(
            [samples_normal, samples_dense, samples_clear, samples_shifted], dim=0
        )

        # Convert to numpy and normalize
        all_samples = all_samples.cpu().numpy()
        all_samples = (all_samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
        all_samples = np.clip(all_samples, 0, 1)

    # Create a grid visualization with labels
    grid_size = 8
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    fig.suptitle(
        "Generated Chest X-ray Samples (Quick Base VAE with Pseudo-Conditioning)",
        fontsize=18,
    )

    # Labels for different types
    type_labels = ["Normal", "Dense", "Clear", "Shifted"]

    for i in range(grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size

        if i < len(all_samples):
            # Determine type and add colored border
            type_idx = i // 16
            type_label = (
                type_labels[type_idx] if type_idx < len(type_labels) else "Extra"
            )

            # Display grayscale image
            img = all_samples[i][0]  # Get first channel (grayscale)
            im = axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)

            # Add colored border based on type
            colors = ["blue", "red", "green", "orange"]
            if type_idx < len(colors):
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor(colors[type_idx])
                    spine.set_linewidth(3)

            # Add type label on first sample of each type
            if i % 16 == 0 and type_idx < len(type_labels):
                axes[row, col].text(
                    0.02,
                    0.98,
                    type_label,
                    transform=axes[row, col].transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    color=colors[type_idx],
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(
        "generated_samples_pseudo_conditional.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
    print("Generated samples saved as 'generated_samples_pseudo_conditional.png'")

    # Also save individual type grids
    for type_idx, type_label in enumerate(type_labels):
        if type_idx * 16 < len(all_samples):
            # Create 4x4 grid for each type
            fig_type, axes_type = plt.subplots(4, 4, figsize=(8, 8))
            fig_type.suptitle(
                f"Generated Chest X-rays - {type_label} Type", fontsize=14
            )

            start_idx = type_idx * 16
            end_idx = min(start_idx + 16, len(all_samples))

            for j in range(16):
                row = j // 4
                col = j % 4

                if start_idx + j < end_idx:
                    img = all_samples[start_idx + j][0]
                    axes_type[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)

                axes_type[row, col].axis("off")

            plt.tight_layout()
            plt.savefig(
                f"generated_samples_{type_label.lower()}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    print("Individual type grids also saved!")


if __name__ == "__main__":
    main()
