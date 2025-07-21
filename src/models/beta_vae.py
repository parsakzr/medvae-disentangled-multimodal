"""
Beta-VAE implementation with controllable beta parameter.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict

from .base_vae import BaseVAE


class BetaVAE(BaseVAE):
    """Beta-VAE for disentangled representation learning."""

    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        prior: Normal,
        posterior: Normal,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute Beta-VAE loss with weighted KL term."""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction="mean")

        # KL divergence with beta weighting
        kl_loss = kl_divergence(posterior, prior).mean()

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "weighted_kl_loss": self.beta * kl_loss,
        }
