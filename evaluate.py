"""
Evaluation script for trained VAE models.
"""

import os
import argparse
import torch
import numpy as np
from typing import Dict, Any
import json

from src.lightning_module import VAELightningModule
from src.data import MedMNISTDataModule
from src.utils.metrics import compute_reconstruction_metrics, compute_latent_metrics
from src.utils.visualization import (
    plot_reconstructions,
    plot_samples,
    plot_latent_space,
)


def evaluate_model(
    model_path: str,
    data_config: Dict[str, Any],
    output_dir: str = "evaluation_results",
    device: str = "cuda",
    num_samples: int = 1000,
):
    """
    Evaluate trained VAE model.

    Args:
        model_path: Path to trained model checkpoint
        data_config: Data configuration
        output_dir: Directory to save results
        device: Device to run on
        num_samples: Number of samples for evaluation
    """
    # Load model
    print(f"Loading model from {model_path}")
    lightning_module = VAELightningModule.load_from_checkpoint(model_path)
    model = lightning_module.model
    model.eval()
    model.to(device)

    # Load data
    print("Setting up data...")
    datamodule = MedMNISTDataModule(**data_config)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Evaluation metrics
    all_metrics = {
        "reconstruction": [],
        "latent": [],
    }

    all_latents = []
    all_labels = []
    all_modalities = []
    sample_images = []
    sample_reconstructions = []

    print("Running evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples // data_config["batch_size"]:
                break

            if len(batch) == 3:
                x, labels, modality = batch
                x, labels, modality = (
                    x.to(device),
                    labels.to(device),
                    modality.to(device),
                )
            else:
                x, labels = batch
                x, labels = x.to(device), labels.to(device)
                modality = None

            # Forward pass
            if modality is not None:
                outputs = model(x, modality)
            else:
                outputs = model(x)

            # Compute metrics
            recon_metrics = compute_reconstruction_metrics(x, outputs["reconstruction"])
            latent_metrics = compute_latent_metrics(outputs["z"])

            all_metrics["reconstruction"].append(recon_metrics)
            all_metrics["latent"].append(latent_metrics)

            # Collect data for analysis
            all_latents.append(outputs["z"].cpu())
            all_labels.append(labels.cpu())
            if modality is not None:
                all_modalities.append(modality.cpu())

            # Collect sample images (first batch only)
            if i == 0:
                sample_images = x[:16].cpu()
                sample_reconstructions = outputs["reconstruction"][:16].cpu()

    # Aggregate metrics
    print("Computing final metrics...")
    final_metrics = {}

    for metric_type in all_metrics:
        final_metrics[metric_type] = {}
        for key in all_metrics[metric_type][0].keys():
            values = [m[key] for m in all_metrics[metric_type]]
            final_metrics[metric_type][key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

    # Concatenate collected data
    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if all_modalities:
        all_modalities = torch.cat(all_modalities, dim=0)
    else:
        all_modalities = None

    # Save metrics
    print("Saving results...")
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Generate visualizations
    print("Creating visualizations...")

    # Reconstruction comparison
    recon_fig = plot_reconstructions(
        sample_images,
        sample_reconstructions,
        num_samples=8,
        save_path=os.path.join(output_dir, "reconstructions.png"),
        title="Sample Reconstructions",
    )

    # Generated samples
    generated_samples = model.sample(16, device)
    sample_fig = plot_samples(
        generated_samples.cpu(),
        num_samples=16,
        nrow=4,
        save_path=os.path.join(output_dir, "generated_samples.png"),
        title="Generated Samples",
    )

    # Latent space visualization
    if all_latents.shape[0] > 100:  # Only if we have enough samples
        latent_fig = plot_latent_space(
            all_latents[:1000],
            all_labels[:1000],
            all_modalities[:1000] if all_modalities is not None else None,
            method="tsne",
            save_path=os.path.join(output_dir, "latent_space.png"),
            title="Latent Space Visualization",
        )

    print(f"Evaluation completed! Results saved to {output_dir}")
    print("\nFinal Metrics Summary:")
    print("-" * 50)

    for metric_type, metrics in final_metrics.items():
        print(f"\n{metric_type.upper()} METRICS:")
        for key, stats in metrics.items():
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained VAE model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to data configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples for evaluation"
    )

    args = parser.parse_args()

    # Load data config (simplified - in practice would use Hydra)
    data_config = {
        "dataset_names": ["chestmnist"],
        "batch_size": 32,
        "num_workers": 4,
        "size": 224,
        "as_rgb": False,
        "root": "./data",
        "normalize": True,
        "augment_train": False,
    }

    evaluate_model(
        model_path=args.model_path,
        data_config=data_config,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
