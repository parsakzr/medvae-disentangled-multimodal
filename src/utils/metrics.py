"""
Metrics for evaluating VAE performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr


def compute_reconstruction_metrics(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.

    Args:
        original: Original images [B, C, H, W]
        reconstructed: Reconstructed images [B, C, H, W]

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # MSE Loss
    mse = F.mse_loss(reconstructed, original, reduction="mean")
    metrics["mse"] = mse.item()

    # MAE Loss
    mae = F.l1_loss(reconstructed, original, reduction="mean")
    metrics["mae"] = mae.item()

    # PSNR
    psnr_value = psnr(reconstructed, original, data_range=1.0)
    metrics["psnr"] = psnr_value.item()

    # SSIM
    ssim_value = ssim(reconstructed, original, data_range=1.0)
    metrics["ssim"] = ssim_value.item()

    return metrics


def compute_kl_metrics(mean: torch.Tensor, logvar: torch.Tensor) -> Dict[str, float]:
    """
    Compute KL divergence metrics.

    Args:
        mean: Latent means [B, D]
        logvar: Latent log variances [B, D]

    Returns:
        Dictionary of KL metrics
    """
    # KL divergence per dimension
    kl_per_dim = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)

    # Total KL
    kl_total = kl_per_dim.sum()

    # KL per sample
    kl_per_sample = kl_per_dim.sum(dim=1)

    return {
        "kl_total": kl_total.item(),
        "kl_mean": kl_per_sample.mean().item(),
        "kl_std": kl_per_sample.std().item(),
        "kl_per_dim_mean": kl_per_dim.mean(dim=0).mean().item(),
    }


def compute_latent_metrics(latents: torch.Tensor) -> Dict[str, float]:
    """
    Compute latent space metrics.

    Args:
        latents: Latent representations [B, D] or [B, D, H, W]

    Returns:
        Dictionary of latent metrics
    """
    if latents.dim() > 2:
        latents = latents.flatten(1)

    # Mean and std of latent activations
    latent_mean = latents.mean(dim=0)
    latent_std = latents.std(dim=0)

    # Sparsity (fraction of near-zero activations)
    threshold = 0.1
    sparsity = (latents.abs() < threshold).float().mean()

    return {
        "latent_mean_avg": latent_mean.mean().item(),
        "latent_std_avg": latent_std.mean().item(),
        "latent_sparsity": sparsity.item(),
    }


def compute_fid_score(
    real_features: torch.Tensor, fake_features: torch.Tensor
) -> float:
    """
    Compute Fréchet Inception Distance (FID).

    Args:
        real_features: Features from real images [N, D]
        fake_features: Features from generated images [N, D]

    Returns:
        FID score
    """
    # Convert to numpy
    real_features = real_features.cpu().numpy()
    fake_features = fake_features.cpu().numpy()

    # Compute means and covariances
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    # Compute FID
    diff = mu1 - mu2
    covmean = np.sqrt(sigma1.dot(sigma2))

    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return float(fid)


def compute_disentanglement_metrics(
    latents: torch.Tensor, factors: torch.Tensor
) -> Dict[str, float]:
    """
    Compute disentanglement metrics (requires known factors).

    Args:
        latents: Latent representations [B, D]
        factors: Ground truth factors [B, F]

    Returns:
        Dictionary of disentanglement metrics
    """
    if latents.dim() > 2:
        latents = latents.flatten(1)

    latents_np = latents.cpu().numpy()
    factors_np = factors.cpu().numpy()

    # Mutual Information Gap (MIG)
    mig_score = compute_mig(latents_np, factors_np)

    # Beta-VAE metric
    beta_vae_score = compute_beta_vae_metric(latents_np, factors_np)

    return {
        "mig": mig_score,
        "beta_vae_metric": beta_vae_score,
    }


def compute_mig(latents: np.ndarray, factors: np.ndarray) -> float:
    """Compute Mutual Information Gap (MIG)."""
    # Simplified MIG computation
    # In practice, you would need to discretize latents and compute MI
    from sklearn.feature_selection import mutual_info_regression

    mig_scores = []
    for f in range(factors.shape[1]):
        mi_scores = []
        for z in range(latents.shape[1]):
            mi = mutual_info_regression(latents[:, z : z + 1], factors[:, f])
            mi_scores.append(mi[0])

        mi_scores = np.array(mi_scores)
        if len(mi_scores) > 1:
            mig_score = (
                mi_scores.max() - mi_scores[np.argsort(mi_scores)[-2]]
            ) / mi_scores.max()
        else:
            mig_score = 0.0
        mig_scores.append(mig_score)

    return np.mean(mig_scores)


def compute_beta_vae_metric(latents: np.ndarray, factors: np.ndarray) -> float:
    """Compute Beta-VAE disentanglement metric."""
    # Simplified implementation
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    scores = []
    for f in range(factors.shape[1]):
        X_train, X_test, y_train, y_test = train_test_split(
            latents, factors[:, f], test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)

    return np.mean(scores)


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    task_type: str = "multiclass",
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes
        task_type: "binary", "multiclass", or "multilabel"

    Returns:
        Dictionary of classification metrics
    """
    if task_type == "multilabel":
        # For multilabel, use sigmoid and threshold
        preds = torch.sigmoid(predictions) > 0.5
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        return {
            "accuracy": accuracy_score(targets_np, preds_np),
            "f1_macro": f1_score(targets_np, preds_np, average="macro"),
            "f1_micro": f1_score(targets_np, preds_np, average="micro"),
            "precision": precision_score(targets_np, preds_np, average="macro"),
            "recall": recall_score(targets_np, preds_np, average="macro"),
        }
    else:
        # For multiclass, use argmax
        if predictions.dim() > 1 and predictions.shape[1] > 1:
            preds = predictions.argmax(dim=1)
        else:
            preds = (torch.sigmoid(predictions) > 0.5).long().squeeze()

        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        avg_type = "binary" if num_classes == 2 else "macro"

        return {
            "accuracy": accuracy_score(targets_np, preds_np),
            "f1": f1_score(targets_np, preds_np, average=avg_type),
            "precision": precision_score(targets_np, preds_np, average=avg_type),
            "recall": recall_score(targets_np, preds_np, average=avg_type),
        }
