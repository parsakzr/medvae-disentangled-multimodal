"""
Base VAE implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, Tuple, Optional

from .encoder_decoder import Encoder, Decoder


class BaseVAE(nn.Module):
    """Base Variational Autoencoder implementation."""

    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 128,
        hidden_channels: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: list = [16],
        dropout: float = 0.0,
        resolution: int = 224,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        double_z: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels

        # Calculate encoder output resolution
        self.encoder_out_res = resolution // (2 ** (len(ch_mult) - 1))

        # Encoder
        self.encoder = Encoder(
            ch=hidden_channels,
            out_ch=input_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=True,
            in_channels=input_channels,
            resolution=resolution,
            z_channels=latent_dim,
            double_z=double_z,  # Output both mean and logvar
            use_linear_attn=use_linear_attn,
            attn_type=attn_type,
        )

        # Decoder
        self.decoder = Decoder(
            ch=hidden_channels,
            out_ch=input_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=True,
            in_channels=input_channels,
            resolution=resolution,
            z_channels=latent_dim,
            use_linear_attn=use_linear_attn,
            attn_type=attn_type,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        # Split into mean and logvar
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self, x: torch.Tensor, return_latents: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE."""
        # Encode
        mean, logvar = self.encode(x)

        # Sample
        z = self.reparameterize(mean, logvar)

        # Decode
        reconstruction = self.decode(z)

        # Create distributions for loss computation
        prior = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
        posterior = Normal(mean, torch.exp(0.5 * logvar))

        outputs = {
            "reconstruction": reconstruction,
            "mean": mean,
            "logvar": logvar,
            "z": z,
            "prior": prior,
            "posterior": posterior,
        }

        if return_latents:
            outputs["latents"] = z

        return outputs

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from prior."""
        z = torch.randn(
            num_samples,
            self.latent_dim,
            self.encoder_out_res,
            self.encoder_out_res,
            device=device,
        )
        return self.decode(z)

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        prior: Normal,
        posterior: Normal,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss components."""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction="mean")

        # KL divergence
        kl_loss = kl_divergence(posterior, prior).mean()

        # Total loss
        total_loss = recon_loss + kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
