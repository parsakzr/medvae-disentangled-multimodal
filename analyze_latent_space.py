#!/usr/bin/env python3
"""
Analyze the latent space of the multi-modal conditional VAE to see modality separation.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from src.models import ConditionalVAE
from src.data import MedMNISTDataModule


def main():
    # Create model with same config as multi-modal quick training
    print("Creating Conditional VAE model...")
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
        num_modalities=4,  # chest, oct, path, derm
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

    # Create data module to get real samples
    print("Loading data...")
    datamodule = MedMNISTDataModule(
        dataset_names=["chestmnist", "octmnist", "pathmnist", "dermamnist"],
        batch_size=64,
        num_workers=2,
        size=28,
        as_rgb=True,
        root="./data",
        normalize=True,
        augment_train=False,
    )
    datamodule.setup("fit")

    # Get validation data
    val_loader = datamodule.val_dataloader()

    # Define modalities
    modality_names = ["ChestMNIST", "OCTMNIST", "PathMNIST", "DermaMNIST"]
    colors = ["blue", "red", "green", "orange"]

    # Encode samples from each modality to analyze latent space
    print("Encoding samples to latent space...")

    latent_vectors = []
    modality_labels = []
    original_labels = []

    max_samples_per_modality = 200  # Limit for visualization
    samples_collected = {i: 0 for i in range(4)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if all(
                count >= max_samples_per_modality
                for count in samples_collected.values()
            ):
                break

            images, labels, modalities = batch
            images = images.to(device)

            # Create dummy condition (zeros) since encode doesn't use it properly yet
            dummy_condition = torch.zeros(images.shape[0], 4, device=device)

            # Encode to get latent representations
            mu, logvar = model.encode(images, dummy_condition)

            # Use mean of posterior for analysis (not sampled z)
            latent_mean = mu.view(mu.shape[0], -1)  # Flatten spatial dimensions

            # Process each sample in the batch
            for i in range(images.shape[0]):
                # Find which modality this sample belongs to
                modality_vector = modalities[i]
                modality_idx = torch.argmax(modality_vector).item()

                if samples_collected[modality_idx] < max_samples_per_modality:
                    latent_vectors.append(latent_mean[i].cpu().numpy())
                    modality_labels.append(modality_idx)
                    original_labels.append(labels[i].cpu().numpy())
                    samples_collected[modality_idx] += 1

    print(f"Collected samples per modality: {samples_collected}")

    # Convert to numpy arrays
    latent_vectors = np.array(latent_vectors)
    modality_labels = np.array(modality_labels)

    print(f"Latent vectors shape: {latent_vectors.shape}")
    print(f"Latent space dimensionality: {latent_vectors.shape[1]}")

    # Analyze latent space with different dimensionality reduction techniques

    # 1. PCA Analysis
    print("Performing PCA analysis...")
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vectors)

    # 2. t-SNE Analysis
    print("Performing t-SNE analysis...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_tsne = tsne.fit_transform(latent_vectors)

    # 3. Raw latent space (first 2 dimensions)
    latent_raw = latent_vectors[:, :2]

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Latent Space Analysis: Multi-Modal VAE", fontsize=16)

    # Plot 1: PCA
    ax1 = axes[0, 0]
    for i, (modality_name, color) in enumerate(zip(modality_names, colors)):
        mask = modality_labels == i
        ax1.scatter(
            latent_pca[mask, 0],
            latent_pca[mask, 1],
            c=color,
            label=modality_name,
            alpha=0.6,
            s=20,
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
            alpha=0.6,
            s=20,
        )
    ax2.set_title("t-SNE Projection")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Raw latent space (first 2 dims)
    ax3 = axes[0, 2]
    for i, (modality_name, color) in enumerate(zip(modality_names, colors)):
        mask = modality_labels == i
        ax3.scatter(
            latent_raw[mask, 0],
            latent_raw[mask, 1],
            c=color,
            label=modality_name,
            alpha=0.6,
            s=20,
        )
    ax3.set_title("Raw Latent Space (Dims 0-1)")
    ax3.set_xlabel("Latent Dimension 0")
    ax3.set_ylabel("Latent Dimension 1")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Pairwise distances between modality centroids
    ax4 = axes[1, 0]
    centroids = []
    for i in range(4):
        mask = modality_labels == i
        if np.any(mask):
            centroid = np.mean(latent_vectors[mask], axis=0)
            centroids.append(centroid)
        else:
            centroids.append(np.zeros(latent_vectors.shape[1]))

    centroids = np.array(centroids)

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform

    distances = squareform(pdist(centroids, metric="euclidean"))

    im = ax4.imshow(distances, cmap="viridis")
    ax4.set_title("Pairwise Distances Between Modality Centroids")
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xticklabels(modality_names, rotation=45)
    ax4.set_yticklabels(modality_names)

    # Add distance values to the heatmap
    for i in range(4):
        for j in range(4):
            ax4.text(
                j, i, f"{distances[i, j]:.2f}", ha="center", va="center", color="white"
            )

    plt.colorbar(im, ax=ax4)

    # Plot 5: Latent space variance analysis
    ax5 = axes[1, 1]
    latent_std = np.std(latent_vectors, axis=0)
    ax5.bar(range(len(latent_std)), latent_std)
    ax5.set_title("Latent Dimension Variance")
    ax5.set_xlabel("Latent Dimension")
    ax5.set_ylabel("Standard Deviation")
    ax5.grid(True, alpha=0.3)

    # Plot 6: Modality separation metrics
    ax6 = axes[1, 2]

    # Calculate silhouette score for modality separation
    from sklearn.metrics import silhouette_score

    try:
        sil_pca = silhouette_score(latent_pca, modality_labels)
        sil_tsne = silhouette_score(latent_tsne, modality_labels)
        sil_raw = silhouette_score(latent_raw, modality_labels)

        methods = ["PCA", "t-SNE", "Raw (2D)"]
        scores = [sil_pca, sil_tsne, sil_raw]

        bars = ax6.bar(methods, scores, color=["skyblue", "lightcoral", "lightgreen"])
        ax6.set_title("Modality Separation Quality\n(Silhouette Score)")
        ax6.set_ylabel("Silhouette Score")
        ax6.set_ylim(-1, 1)
        ax6.grid(True, alpha=0.3)

        # Add score values on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

    except Exception as e:
        ax6.text(
            0.5,
            0.5,
            f"Error calculating\nsilhouette scores:\n{str(e)}",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )
        ax6.set_title("Separation Quality (Error)")

    plt.tight_layout()
    plt.savefig("latent_space_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print quantitative analysis
    print("\n" + "=" * 60)
    print("LATENT SPACE ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\n1. DATA SUMMARY:")
    print(f"   Total samples analyzed: {len(latent_vectors)}")
    for i, name in enumerate(modality_names):
        count = np.sum(modality_labels == i)
        print(f"   {name}: {count} samples")

    print(f"\n2. DIMENSIONALITY:")
    print(f"   Latent space dimensions: {latent_vectors.shape[1]}")
    print(
        f"   PCA variance explained (2 components): {pca.explained_variance_ratio_[:2].sum():.2%}"
    )

    print(f"\n3. MODALITY SEPARATION:")
    print(
        f"   Average pairwise centroid distance: {np.mean(distances[np.triu_indices(4, k=1)]):.3f}"
    )
    print(
        f"   Min pairwise centroid distance: {np.min(distances[np.triu_indices(4, k=1)]):.3f}"
    )
    print(
        f"   Max pairwise centroid distance: {np.max(distances[np.triu_indices(4, k=1)]):.3f}"
    )

    try:
        print(f"\n4. CLUSTERING QUALITY (Silhouette Scores):")
        print(f"   PCA projection: {sil_pca:.3f}")
        print(f"   t-SNE projection: {sil_tsne:.3f}")
        print(f"   Raw latent (2D): {sil_raw:.3f}")
        print(f"   (Score range: -1 to 1, higher is better)")
    except:
        print(f"\n4. CLUSTERING QUALITY: Could not compute")

    print(f"\n5. INTERPRETATION:")
    if np.min(distances[np.triu_indices(4, k=1)]) < 2.0:
        print("   ⚠️  LOW SEPARATION: Modalities are very close in latent space")
        print("   ⚠️  This explains why samples from different modalities look similar")

    try:
        if max(sil_pca, sil_tsne, sil_raw) < 0.3:
            print(
                "   ⚠️  POOR CLUSTERING: Low silhouette scores indicate mixed modalities"
            )
        elif max(sil_pca, sil_tsne, sil_raw) > 0.5:
            print(
                "   ✅ GOOD CLUSTERING: High silhouette scores indicate separated modalities"
            )
        else:
            print("   ⚡ MODERATE CLUSTERING: Some separation but room for improvement")
    except:
        pass

    print("=" * 60)

    # Save detailed analysis
    analysis_results = {
        "latent_vectors": latent_vectors,
        "modality_labels": modality_labels,
        "centroids": centroids,
        "distances": distances,
        "pca_projection": latent_pca,
        "tsne_projection": latent_tsne,
    }

    np.savez("latent_space_analysis.npz", **analysis_results)
    print("\nDetailed results saved to 'latent_space_analysis.npz'")
    print("Visualization saved to 'latent_space_analysis.png'")


if __name__ == "__main__":
    main()
