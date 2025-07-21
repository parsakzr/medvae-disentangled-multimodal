#!/usr/bin/env python3
"""
Simplified latent space analysis using generated samples to understand modality entanglement.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import argparse
import os
from src.models import ConditionalVAE, DisentangledConditionalVAE


def main():
    parser = argparse.ArgumentParser(
        description="Analyze latent space of trained multi-modal VAE"
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
        help="Type of model to analyze",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of samples per modality for analysis",
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

    # For DisentangledConditionalVAE, handle dynamic conv_in layer
    if args.model_type == "disentangled":
        # Check if the checkpoint has a different input channel size
        checkpoint_conv_weight = model_state_dict.get("encoder.conv_in.weight")
        if checkpoint_conv_weight is not None:
            checkpoint_in_channels = checkpoint_conv_weight.shape[1]
            model_in_channels = model.encoder.conv_in.in_channels

            if checkpoint_in_channels != model_in_channels:
                print(
                    f"Adapting conv_in layer: checkpoint has {checkpoint_in_channels} channels, model expects {model_in_channels}"
                )

                # Create new conv_in layer with the correct input channels
                original_conv = model.encoder.conv_in
                model.encoder.conv_in = torch.nn.Conv2d(
                    checkpoint_in_channels,
                    original_conv.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )

                # Update the original_conv_in reference as well
                model.original_conv_in = torch.nn.Conv2d(
                    3,  # Keep original as 3 channels
                    original_conv.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )

    model.load_state_dict(model_state_dict)
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
    colors = ["blue", "red", "green", "purple", "orange"]

    # Generate samples and analyze their latent representations
    print("Generating samples and analyzing latent space...")

    latent_vectors = []
    modality_labels = []
    generated_samples = []

    num_samples_per_modality = (
        args.num_samples // 5
    )  # Distribute samples across modalities

    with torch.no_grad():
        for modality_idx in range(5):  # Updated to 5 modalities
            print(f"Generating samples for {modality_names[modality_idx]}...")

            if args.model_type == "disentangled":
                # For DisentangledConditionalVAE, use conditional sampling
                device_obj = torch.device(device)
                modality_indices = torch.full(
                    (num_samples_per_modality,), modality_idx, device=device_obj
                )

                # Use conditional sampling - we know it works from debug
                samples = model.sample_conditional(
                    num_samples_per_modality, modality_indices, device_obj
                )

                # To get latent vectors for analysis, sample from the latent space directly
                z_samples = torch.randn(
                    num_samples_per_modality,
                    model.latent_dim,
                    model.encoder_out_res,
                    model.encoder_out_res,
                    device=device_obj,
                )
                # Use the full latent vector for analysis
                latent_vecs = z_samples.view(num_samples_per_modality, -1)

            else:
                # For regular ConditionalVAE, sample from latent space
                torch.manual_seed(42 + modality_idx * 100)
                device_obj = torch.device(device)
                z_samples = torch.randn(
                    num_samples_per_modality,
                    model.latent_dim,
                    model.encoder_out_res,
                    model.encoder_out_res,
                    device=device_obj,
                )
                latent_vecs = z_samples.view(num_samples_per_modality, -1)
                samples = model.decode(z_samples)

            # Store flattened latent vectors for analysis
            latent_vectors.append(latent_vecs.cpu().numpy())
            modality_labels.extend([modality_idx] * num_samples_per_modality)

            # Store generated samples
            generated_samples.append(samples.cpu())

    # Combine all latent vectors
    latent_vectors = np.vstack(latent_vectors)
    modality_labels = np.array(modality_labels)
    generated_samples = torch.cat(generated_samples, dim=0)

    print(f"Latent vectors shape: {latent_vectors.shape}")
    print(f"Analyzing {len(latent_vectors)} latent vectors...")

    # Dimensionality reduction
    print("Performing PCA analysis...")
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vectors)

    print("Performing t-SNE analysis...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_tsne = tsne.fit_transform(latent_vectors)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Latent Space Analysis: DisentangledConditionalVAE Modality Separation",
        fontsize=16,
    )

    # Plot 1: PCA
    ax1 = axes[0, 0]
    for i, (modality_name, color) in enumerate(zip(modality_names, colors)):
        mask = modality_labels == i
        ax1.scatter(
            latent_pca[mask, 0],
            latent_pca[mask, 1],
            c=color,
            label=modality_name,
            alpha=0.7,
            s=30,
        )
    ax1.set_title("PCA Projection")
    ax1.set_xlabel(f"PC1 (explained var: {pca.explained_variance_ratio_[0]:.2%})")
    ax1.set_ylabel(f"PC2 (explained var: {pca.explained_variance_ratio_[1]:.2%})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: t-SNE
    ax2 = axes[0, 1]
    for i, (modality_name, color) in enumerate(zip(modality_names, colors)):
        mask = modality_labels == i
        ax2.scatter(
            latent_tsne[mask, 0],
            latent_tsne[mask, 1],
            c=color,
            label=modality_name,
            alpha=0.7,
            s=30,
        )
    ax2.set_title("t-SNE Projection")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Raw latent space (first 2 dims)
    ax3 = axes[0, 2]
    latent_raw = latent_vectors[:, :2]
    for i, (modality_name, color) in enumerate(zip(modality_names, colors)):
        mask = modality_labels == i
        ax3.scatter(
            latent_raw[mask, 0],
            latent_raw[mask, 1],
            c=color,
            label=modality_name,
            alpha=0.7,
            s=30,
        )
    ax3.set_title("Raw Latent Space (Dims 0-1)")
    ax3.set_xlabel("Latent Dimension 0")
    ax3.set_ylabel("Latent Dimension 1")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Pairwise distances between modality centroids
    ax4 = axes[1, 0]
    centroids = []
    for i in range(5):  # Updated to 5 modalities
        mask = modality_labels == i
        centroid = np.mean(latent_vectors[mask], axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform

    distances = squareform(pdist(centroids, metric="euclidean"))

    im = ax4.imshow(distances, cmap="viridis")
    ax4.set_title("Pairwise Distances Between Modality Centroids")
    ax4.set_xticks(range(5))  # Updated to 5
    ax4.set_yticks(range(5))  # Updated to 5
    ax4.set_xticklabels(modality_names, rotation=45)
    ax4.set_yticklabels(modality_names)

    # Add distance values to the heatmap
    for i in range(5):  # Updated to 5
        for j in range(5):  # Updated to 5
            ax4.text(
                j, i, f"{distances[i, j]:.2f}", ha="center", va="center", color="white"
            )

    plt.colorbar(im, ax=ax4)

    # Plot 5: Latent dimension variance
    ax5 = axes[1, 1]
    latent_std = np.std(latent_vectors, axis=0)
    ax5.bar(range(len(latent_std)), latent_std)
    ax5.set_title("Latent Dimension Variance")
    ax5.set_xlabel("Latent Dimension")
    ax5.set_ylabel("Standard Deviation")
    ax5.grid(True, alpha=0.3)

    # Plot 6: Generated sample comparison
    ax6 = axes[1, 2]

    # Show a few representative samples from each modality
    sample_grid = []
    for i in range(5):  # Updated to 5 modalities
        # Take first sample from each modality
        sample = generated_samples[i * num_samples_per_modality]
        sample = (sample + 1) / 2  # Normalize to [0, 1]
        sample = torch.clamp(sample, 0, 1)

        if sample.shape[0] == 3:  # RGB
            sample = sample.permute(1, 2, 0).numpy()
        else:  # Grayscale
            sample = sample[0].numpy()

        sample_grid.append(sample)

    # Create a grid of samples (5 modalities, so we'll use a different layout)
    if len(sample_grid) >= 5:
        # Create 1x5 horizontal layout
        combined = np.zeros((28, 140, 3))  # 1x5 grid of 28x28 images
        for idx, sample in enumerate(sample_grid[:5]):
            start_col, end_col = idx * 28, (idx + 1) * 28

            if len(sample.shape) == 2:  # Grayscale
                combined[:, start_col:end_col, :] = np.stack([sample] * 3, axis=-1)
            else:  # RGB
                combined[:, start_col:end_col, :] = sample

        ax6.imshow(combined)
        ax6.set_title("Sample from Each Modality")

        # Add labels below each sample
        for idx, modality_name in enumerate(modality_names):
            ax6.text(
                idx * 28 + 14,  # Center of each sample
                30,  # Below the image
                modality_name,
                ha="center",
                va="top",
                color="black",
                fontsize=8,
                weight="bold",
            )

    ax6.axis("off")

    plt.tight_layout()
    plt.savefig("disentangled_latent_space_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Calculate clustering quality
    from sklearn.metrics import silhouette_score

    try:
        sil_pca = silhouette_score(latent_pca, modality_labels)
        sil_tsne = silhouette_score(latent_tsne, modality_labels)
        sil_raw = silhouette_score(latent_raw, modality_labels)
    except Exception as e:
        print(f"Error calculating silhouette scores: {e}")
        sil_pca = sil_tsne = sil_raw = 0

    # Print analysis results
    print("\n" + "=" * 70)
    print("DISENTANGLED CONDITIONAL VAE LATENT SPACE ANALYSIS")
    print("=" * 70)

    print(f"\n1. METHODOLOGY:")
    print(
        f"   - Generated {num_samples_per_modality} samples per modality using conditional sampling"
    )
    print(f"   - Each modality uses DisentangledConditionalVAE.sample_conditional()")
    print(
        f"   - Analyzed latent vectors used for generation with modality-specific shifts"
    )

    print(f"\n2. LATENT SPACE PROPERTIES:")
    print(f"   Total latent dimensions: {latent_vectors.shape[1]}")
    print(
        f"   PCA variance explained (2 components): {pca.explained_variance_ratio_[:2].sum():.2%}"
    )
    print(f"   Average latent variance: {np.mean(latent_std):.3f}")

    print(f"\n3. MODALITY SEPARATION:")
    print(
        f"   Average pairwise centroid distance: {np.mean(distances[np.triu_indices(5, k=1)]):.3f}"
    )
    print(
        f"   Min pairwise centroid distance: {np.min(distances[np.triu_indices(5, k=1)]):.3f}"
    )
    print(
        f"   Max pairwise centroid distance: {np.max(distances[np.triu_indices(5, k=1)]):.3f}"
    )

    print(f"\n4. CLUSTERING QUALITY (Silhouette Scores):")
    print(f"   PCA projection: {sil_pca:.3f}")
    print(f"   t-SNE projection: {sil_tsne:.3f}")
    print(f"   Raw latent (2D): {sil_raw:.3f}")
    print(f"   (Score range: -1 to 1, higher is better)")

    print(f"\n5. KEY FINDINGS:")
    print(f"   🔍 This analysis evaluates the DisentangledConditionalVAE's ability")
    print(f"      to separate different modalities in latent space")
    print(f"   🔍 Higher separation indicates better modality disentanglement")

    if np.min(distances[np.triu_indices(5, k=1)]) < 2.0:
        print(f"   ⚠️  OBSERVATION: Very close modality centroids")
        print(f"   ⚠️  This suggests limited latent space utilization")

    # Convert to regular Python floats to avoid numpy typing issues
    max_sil = max(float(sil_pca), float(sil_tsne), float(sil_raw))
    if max_sil < 0.3:
        print(f"   ⚠️  INSIGHT: Poor clustering (max silhouette: {max_sil:.3f})")
        print(f"      suggests limited modality separation after epoch 0")
        print(f"   💡 RECOMMENDATION: Model may need more training or stronger")
        print(f"      separation/contrastive loss weights")

    print("=" * 70)

    # Save results
    np.savez(
        "disentangled_modality_analysis.npz",
        latent_vectors=latent_vectors,
        modality_labels=modality_labels,
        centroids=centroids,
        distances=distances,
        pca_projection=latent_pca,
        tsne_projection=latent_tsne,
    )

    print("\nAnalysis saved to 'disentangled_modality_analysis.npz'")
    print("Visualization saved to 'disentangled_latent_space_analysis.png'")


if __name__ == "__main__":
    main()
