#!/usr/bin/env python3
"""
Quick script to generate samples from trained multi-modal conditional VAE.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from src.models import ConditionalVAE, DisentangledConditionalVAE


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from trained multi-modal VAE"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["conditional", "disentangled"],
        default="disentangled",
        help="Type of model to load",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples to generate per modality",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Create model based on type
    print(f"Creating {args.model_type} VAE model...")
    if args.model_type == "disentangled":
        model = DisentangledConditionalVAE(
            input_channels=3,  # RGB inputs
            latent_dim=16,
            shared_latent_dim=8,
            modality_latent_dim=8,
            hidden_channels=32,
            ch_mult=(1, 2, 4),
            num_res_blocks=1,
            attn_resolutions=[],
            dropout=0.1,
            resolution=28,
            use_linear_attn=False,
            attn_type="vanilla",
            double_z=True,
            num_modalities=5,  # chest, oct, path, pneumonia, derm
            modality_separation_weight=0.1,
            contrastive_weight=0.05,
        )
    else:
        model = ConditionalVAE(
            input_channels=3,  # RGB inputs
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
            num_modalities=5,  # chest, oct, path, pneumonia, derm
        )

    # Load checkpoint weights
    print(f"Loading weights from {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Extract model weights from Lightning checkpoint
    model_state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("model."):
            model_state_dict[key[6:]] = value  # Remove "model." prefix

    # Handle the dynamic conv_in layer modification for DisentangledConditionalVAE
    if args.model_type == "disentangled":
        # Check if the checkpoint has a modified conv_in layer
        conv_in_key = "encoder.conv_in.weight"
        if conv_in_key in model_state_dict:
            checkpoint_conv_shape = model_state_dict[conv_in_key].shape
            model_conv_shape = model.encoder.conv_in.weight.shape

            if (
                checkpoint_conv_shape[1] != model_conv_shape[1]
            ):  # Different input channels
                print(
                    f"Adjusting conv_in layer: checkpoint has {checkpoint_conv_shape[1]} input channels, "
                    f"model expects {model_conv_shape[1]} channels"
                )

                # Create a new conv_in layer with the checkpoint's input channels
                original_conv = model.encoder.conv_in
                model.encoder.conv_in = nn.Conv2d(
                    checkpoint_conv_shape[1],  # Use checkpoint's input channels
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                )

    # Load the state dict
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError as e:
        print(f"Warning: Some keys couldn't be loaded: {e}")
        # Try loading with strict=False
        missing_keys, unexpected_keys = model.load_state_dict(
            model_state_dict, strict=False
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    model.eval()

    # Use MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    # Define modalities (updated to match the 5-modality dataset)
    modality_names = [
        "ChestMNIST",
        "OCTMNIST",
        "PathMNIST",
        "PneumoniaMNIST",
        "DermaMNIST",
    ]
    modality_descriptions = [
        "Chest X-Ray",
        "Retinal OCT",
        "Colon Pathology",
        "Pneumonia X-Ray",
        "Dermatoscope",
    ]

    # Generate samples for each modality
    num_samples_per_modality = args.num_samples
    print(f"Generating {num_samples_per_modality} samples per modality...")

    with torch.no_grad():
        all_samples = []
        all_modalities = []

        for modality_idx in range(5):  # Updated to 5 modalities
            print(f"Generating samples for {modality_names[modality_idx]}...")

            if args.model_type == "disentangled":
                # Use conditional sampling for DisentangledConditionalVAE
                modality_indices = torch.full(
                    (num_samples_per_modality,), modality_idx, device=device
                )
                samples = model.sample_conditional(
                    num_samples_per_modality, modality_indices, device
                )
            else:
                # For regular ConditionalVAE, create one-hot encoding
                modality_one_hot = torch.zeros(
                    num_samples_per_modality, 5, device=device
                )
                modality_one_hot[:, modality_idx] = 1.0

                # Generate samples using the inherited sample method with conditioning
                # Note: This might need to be adjusted based on the actual ConditionalVAE interface
                if hasattr(model, "sample_conditional"):
                    samples = model.sample_conditional(
                        num_samples_per_modality, modality_one_hot, device
                    )
                else:
                    # Fallback to unconditional sampling
                    torch.manual_seed(42 + modality_idx * 100)
                    samples = model.sample(num_samples_per_modality, device)

            all_samples.append(samples)
            all_modalities.extend([modality_idx] * num_samples_per_modality)

        # Combine all samples
        all_samples = torch.cat(all_samples, dim=0)

        # Convert to numpy and normalize
        all_samples = all_samples.cpu().numpy()
        all_samples = (all_samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
        all_samples = np.clip(all_samples, 0, 1)

    # Create a grid visualization with modality labels
    total_samples = len(all_samples)
    grid_cols = 5  # One column per modality
    grid_rows = num_samples_per_modality
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 3 * grid_rows))
    fig.suptitle(
        f"Generated Multi-Modal Medical Images ({args.model_type.title()} VAE)",
        fontsize=16,
    )

    # Colors for different modalities
    colors = ["blue", "red", "green", "purple", "orange"]

    for modality_idx in range(5):
        for sample_idx in range(num_samples_per_modality):
            row = sample_idx
            col = modality_idx

            # Calculate the index in the flattened all_samples array
            sample_array_idx = modality_idx * num_samples_per_modality + sample_idx

            if sample_array_idx < len(all_samples):
                img = all_samples[sample_array_idx]

                # Display image (convert RGB to displayable format)
                if img.shape[0] == 3:  # RGB
                    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                    axes[row, col].imshow(img)
                else:  # Grayscale
                    axes[row, col].imshow(img[0], cmap="gray", vmin=0, vmax=1)

                # Add colored border based on modality
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor(colors[modality_idx])
                    spine.set_linewidth(3)

                # Add modality label on first sample of each modality
                if sample_idx == 0:
                    axes[row, col].set_title(
                        modality_names[modality_idx],
                        color=colors[modality_idx],
                        fontweight="bold",
                        fontsize=12,
                    )
            else:
                axes[row, col].set_visible(False)

            axes[row, col].axis("off")

    plt.tight_layout()

    # Save with checkpoint info in filename
    checkpoint_name = os.path.basename(args.checkpoint).replace(".ckpt", "")
    output_filename = f"generated_samples_{args.model_type}_{checkpoint_name}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Generated samples saved as '{output_filename}'")

    # Also save individual modality grids
    for modality_idx, (modality_name, description) in enumerate(
        zip(modality_names, modality_descriptions)
    ):
        if modality_idx >= 5:  # Safety check
            break

        # Create grid for each modality
        grid_size = int(np.ceil(np.sqrt(num_samples_per_modality)))
        fig_mod, axes_mod = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig_mod.suptitle(
            f"Generated {description} Images ({modality_name}) - {args.model_type.title()} VAE",
            fontsize=14,
        )

        # Flatten axes for easier indexing
        if grid_size == 1:
            axes_mod = [axes_mod]
        else:
            axes_mod = axes_mod.flatten()

        start_idx = modality_idx * num_samples_per_modality

        for j in range(grid_size * grid_size):
            sample_idx = start_idx + j

            if j < num_samples_per_modality and sample_idx < len(all_samples):
                img = all_samples[sample_idx]
                if img.shape[0] == 3:  # RGB
                    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                    axes_mod[j].imshow(img)
                else:  # Grayscale
                    axes_mod[j].imshow(img[0], cmap="gray", vmin=0, vmax=1)
            else:
                axes_mod[j].set_visible(False)

            axes_mod[j].axis("off")

        plt.tight_layout()
        modality_filename = (
            f"generated_{modality_name.lower()}_{args.model_type}_{checkpoint_name}.png"
        )
        plt.savefig(modality_filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {modality_name} samples as '{modality_filename}'")

    print("Generation complete! All sample grids saved.")


if __name__ == "__main__":
    main()
