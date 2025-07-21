"""
Conditional VAE implementation with medical modality conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, Optional, List

from .base_vae import BaseVAE


class ConditionalVAE(BaseVAE):
    """Conditional VAE with medical modality conditioning."""

    def __init__(
        self,
        modalities: List[str] = None,
        condition_dim: int = None,
        condition_method: str = "concat",  # "concat", "inject", "film"
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set up modalities
        if modalities is None:
            modalities = [
                "chest_xray",
                "pathology",
                "oct",
                "pneumonia",
                "dermatoscope",
                "blood_cell",
                "tissue",
                "retina",
                "breast_ultrasound",
                "abdominal_ct_a",
                "abdominal_ct_c",
                "abdominal_ct_s",
            ]

        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.condition_dim = condition_dim or self.num_modalities
        self.condition_method = condition_method

        # Condition embedding layers
        if condition_method == "concat":
            # Modify encoder input channels to include condition
            self._setup_concat_conditioning()
        elif condition_method == "inject":
            # Add condition injection layers
            self._setup_inject_conditioning()
        elif condition_method == "film":
            # Feature-wise Linear Modulation
            self._setup_film_conditioning()

    def _setup_concat_conditioning(self):
        """Set up conditioning by concatenating to input."""
        # Create a new encoder with modified input channels
        original_encoder = self.encoder

        # Create condition projection layer
        self.condition_proj = nn.Sequential(
            nn.Linear(self.condition_dim, self.input_channels * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (self.input_channels, 8, 8)),
        )

        # Modify encoder to accept concatenated input
        self.encoder.conv_in = nn.Conv2d(
            self.input_channels * 2,  # Original + condition map
            original_encoder.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def _setup_inject_conditioning(self):
        """Set up conditioning by injecting into encoder layers."""
        # Add condition embedding
        self.condition_embedding = nn.Sequential(
            nn.Linear(self.condition_dim, 512), nn.ReLU(), nn.Linear(512, 512)
        )

        # Modify encoder blocks to accept condition
        # This would require more extensive modifications to the encoder architecture
        pass

    def _setup_film_conditioning(self):
        """Set up Feature-wise Linear Modulation conditioning."""
        # Add FiLM layers for each encoder block
        self.film_layers = nn.ModuleList()
        for i in range(len(self.encoder.down)):
            film_layer = FiLMLayer(
                condition_dim=self.condition_dim, feature_dim=self.encoder.ch * (2**i)
            )
            self.film_layers.append(film_layer)

    def encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """Encode condition vector."""
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        return condition

    def create_condition_map(
        self, condition: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Create spatial condition map."""
        batch_size = condition.shape[0]
        condition_map = self.condition_proj(condition)

        # Resize to match input spatial dimensions
        condition_map = F.interpolate(
            condition_map, size=(height, width), mode="bilinear", align_corners=False
        )

        return condition_map

    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> tuple:
        """Conditional encoding."""
        if self.condition_method == "concat":
            # Create condition map and concatenate
            condition_map = self.create_condition_map(condition, x.shape[2], x.shape[3])
            x_cond = torch.cat([x, condition_map], dim=1)
            return super().encode(x_cond)
        else:
            # For other methods, encode condition and pass through modified encoder
            encoded_condition = self.encode_condition(condition)
            # TODO: Implement other conditioning methods
            return super().encode(x)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor, return_latents: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through conditional VAE."""
        # Encode with condition
        mean, logvar = self.encode(x, condition)

        # Sample
        z = self.reparameterize(mean, logvar)

        # Decode (decoder remains unconditional for now)
        reconstruction = self.decode(z)

        # Create distributions
        prior = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
        posterior = Normal(mean, torch.exp(0.5 * logvar))

        outputs = {
            "reconstruction": reconstruction,
            "mean": mean,
            "logvar": logvar,
            "z": z,
            "prior": prior,
            "posterior": posterior,
            "condition": condition,
        }

        if return_latents:
            outputs["latents"] = z

        return outputs

    def conditional_sample(
        self, num_samples: int, condition: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Sample conditioned on modality."""
        z = torch.randn(
            num_samples,
            self.latent_dim,
            self.encoder_out_res,
            self.encoder_out_res,
            device=device,
        )
        # For now, decoder is unconditional
        # TODO: Add conditional decoding
        return self.decode(z)

    def get_modality_condition(self, modality: str) -> torch.Tensor:
        """Get one-hot condition vector for modality."""
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}")

        condition = torch.zeros(self.num_modalities)
        condition[self.modalities.index(modality)] = 1.0
        return condition


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.scale_transform = nn.Linear(condition_dim, feature_dim)
        self.shift_transform = nn.Linear(condition_dim, feature_dim)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning."""
        scale = self.scale_transform(condition).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift_transform(condition).unsqueeze(-1).unsqueeze(-1)
        return features * scale + shift
