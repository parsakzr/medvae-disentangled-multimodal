"""
Visualization utilities for VAE outputs.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_reconstructions(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    title: str = "Reconstructions",
) -> plt.Figure:
    """
    Plot original vs reconstructed images.

    Args:
        original: Original images [B, C, H, W]
        reconstructed: Reconstructed images [B, C, H, W]
        num_samples: Number of samples to show
        save_path: Path to save figure
        title: Plot title
    """
    num_samples = min(num_samples, original.shape[0])

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_samples):
        # Original
        orig_img = original[i].cpu()
        if orig_img.shape[0] == 1:
            orig_img = orig_img.squeeze(0)
        elif orig_img.shape[0] == 3:
            orig_img = orig_img.permute(1, 2, 0)

        axes[0, i].imshow(orig_img, cmap="gray" if len(orig_img.shape) == 2 else None)
        axes[0, i].set_title("Original" if i == 0 else "")
        axes[0, i].axis("off")

        # Reconstructed
        recon_img = reconstructed[i].cpu()
        if recon_img.shape[0] == 1:
            recon_img = recon_img.squeeze(0)
        elif recon_img.shape[0] == 3:
            recon_img = recon_img.permute(1, 2, 0)

        axes[1, i].imshow(recon_img, cmap="gray" if len(recon_img.shape) == 2 else None)
        axes[1, i].set_title("Reconstructed" if i == 0 else "")
        axes[1, i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_samples(
    samples: torch.Tensor,
    num_samples: int = 16,
    nrow: int = 4,
    save_path: Optional[str] = None,
    title: str = "Generated Samples",
) -> plt.Figure:
    """
    Plot generated samples in a grid.

    Args:
        samples: Generated samples [B, C, H, W]
        num_samples: Number of samples to show
        nrow: Number of samples per row
        save_path: Path to save figure
        title: Plot title
    """
    num_samples = min(num_samples, samples.shape[0])
    ncol = (num_samples + nrow - 1) // nrow

    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    if ncol == 1:
        axes = axes.reshape(1, -1)
    elif nrow == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_samples):
        row = i // nrow
        col = i % nrow

        sample_img = samples[i].cpu()
        if sample_img.shape[0] == 1:
            sample_img = sample_img.squeeze(0)
        elif sample_img.shape[0] == 3:
            sample_img = sample_img.permute(1, 2, 0)

        axes[row, col].imshow(
            sample_img, cmap="gray" if len(sample_img.shape) == 2 else None
        )
        axes[row, col].axis("off")

    # Hide empty subplots
    for i in range(num_samples, ncol * nrow):
        row = i // nrow
        col = i % nrow
        axes[row, col].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_latent_space(
    latents: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    modalities: Optional[torch.Tensor] = None,
    method: str = "tsne",
    save_path: Optional[str] = None,
    title: str = "Latent Space",
) -> plt.Figure:
    """
    Plot 2D projection of latent space.

    Args:
        latents: Latent vectors [B, D] or [B, D, H, W]
        labels: Class labels [B]
        modalities: Modality labels [B]
        method: Dimensionality reduction method ("tsne", "pca")
        save_path: Path to save figure
        title: Plot title
    """
    # Flatten latents if needed
    if latents.dim() > 2:
        latents = latents.flatten(1)

    latents_np = latents.cpu().numpy()

    # Reduce dimensionality
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    latents_2d = reducer.fit_transform(latents_np)

    fig, axes = plt.subplots(
        1,
        2 if modalities is not None else 1,
        figsize=(12 if modalities is not None else 6, 5),
    )
    if modalities is None:
        axes = [axes]

    # Plot by class labels
    if labels is not None:
        labels_np = labels.cpu().numpy()
        scatter = axes[0].scatter(
            latents_2d[:, 0], latents_2d[:, 1], c=labels_np, cmap="tab10", alpha=0.7
        )
        axes[0].set_title("Colored by Class")
        plt.colorbar(scatter, ax=axes[0])
    else:
        axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.7)
        axes[0].set_title("Latent Space")

    axes[0].set_xlabel(f"{method.upper()} 1")
    axes[0].set_ylabel(f"{method.upper()} 2")

    # Plot by modality
    if modalities is not None:
        modalities_np = modalities.cpu().numpy()
        if modalities_np.dim() > 1:  # One-hot encoded
            modalities_np = modalities_np.argmax(axis=1)
        scatter = axes[1].scatter(
            latents_2d[:, 0], latents_2d[:, 1], c=modalities_np, cmap="Set1", alpha=0.7
        )
        axes[1].set_title("Colored by Modality")
        axes[1].set_xlabel(f"{method.upper()} 1")
        axes[1].set_ylabel(f"{method.upper()} 2")
        plt.colorbar(scatter, ax=axes[1])

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    loss_names: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs
        loss_names: Names of different loss components
        save_path: Path to save figure
        title: Plot title
    """
    if loss_names is None:
        loss_names = ["Total Loss"]

    fig, axes = plt.subplots(1, len(loss_names), figsize=(6 * len(loss_names), 5))
    if len(loss_names) == 1:
        axes = [axes]

    epochs = range(1, len(train_losses[0]) + 1)

    for i, loss_name in enumerate(loss_names):
        axes[i].plot(epochs, train_losses[i], label=f"Train {loss_name}", marker="o")
        axes[i].plot(epochs, val_losses[i], label=f"Val {loss_name}", marker="s")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].set_title(loss_name)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
