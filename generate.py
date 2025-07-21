"""
Generate samples from trained VAE model.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from src.lightning_module import VAELightningModule
from src.models import ConditionalVAE


def generate_samples(
    model_path: str,
    num_samples: int = 64,
    output_dir: str = "generated_samples",
    device: str = "cuda",
    modality: str = None,
    grid_size: int = 8,
):
    """
    Generate samples from trained VAE.

    Args:
        model_path: Path to trained model checkpoint
        num_samples: Number of samples to generate
        output_dir: Directory to save generated samples
        device: Device to run on
        modality: Modality for conditional generation
        grid_size: Grid size for visualization
    """
    # Load model
    print(f"Loading model from {model_path}")
    lightning_module = VAELightningModule.load_from_checkpoint(model_path)
    model = lightning_module.model
    model.eval()
    model.to(device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        if isinstance(model, ConditionalVAE) and modality is not None:
            # Conditional generation
            condition = model.get_modality_condition(modality)
            condition = condition.unsqueeze(0).repeat(num_samples, 1).to(device)
            samples = model.conditional_sample(num_samples, condition, device)
        else:
            # Unconditional generation
            samples = model.sample(num_samples, device)

        # Convert to numpy and rescale
        samples = samples.cpu().numpy()
        samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
        samples = np.clip(samples, 0, 1)

        # Save individual samples
        for i, sample in enumerate(samples):
            if sample.shape[0] == 1:
                # Grayscale
                img = (sample[0] * 255).astype(np.uint8)
                img = Image.fromarray(img, mode="L")
            else:
                # RGB
                img = (sample.transpose(1, 2, 0) * 255).astype(np.uint8)
                img = Image.fromarray(img, mode="RGB")

            img.save(os.path.join(output_dir, f"sample_{i:04d}.png"))

        # Create grid visualization
        grid_samples = samples[: grid_size * grid_size]
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
        )

        for i, sample in enumerate(grid_samples):
            row = i // grid_size
            col = i % grid_size

            if sample.shape[0] == 1:
                axes[row, col].imshow(sample[0], cmap="gray")
            else:
                axes[row, col].imshow(sample.transpose(1, 2, 0))

            axes[row, col].axis("off")

        # Hide empty subplots
        for i in range(len(grid_samples), grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "sample_grid.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    print(f"Generated {num_samples} samples saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained VAE")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--num_samples", type=int, default=64, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_samples",
        help="Output directory for samples",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--modality", type=str, default=None, help="Modality for conditional generation"
    )
    parser.add_argument(
        "--grid_size", type=int, default=8, help="Grid size for visualization"
    )

    args = parser.parse_args()

    generate_samples(
        model_path=args.model_path,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device,
        modality=args.modality,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
