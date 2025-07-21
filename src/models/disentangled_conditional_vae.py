#!/usr/bin/env python3
"""
Improved Conditional VAE with explicit modality separation in latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple
from src.models.base_vae import BaseVAE


class DisentangledConditionalVAE(BaseVAE):
    """
    Conditional VAE with explicit modality disentanglement in latent space.

    Key improvements:
    1. Partitioned latent space: [z_shared, z_modality_specific]
    2. Modality contrastive loss
    3. Modality-specific encoders/decoders
    4. Regularization for modality separation
    """

    def __init__(
        self,
        num_modalities: int = 4,
        shared_latent_dim: int = 8,
        modality_latent_dim: int = 8,
        modality_separation_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        resolution: int = 28,  # Explicitly handle resolution
        **kwargs,
    ):
        # Store our specific parameters before calling super
        self.num_modalities = num_modalities
        self.shared_latent_dim = shared_latent_dim
        self.modality_latent_dim = modality_latent_dim
        self.modality_separation_weight = modality_separation_weight
        self.contrastive_weight = contrastive_weight

        # Modify total latent_dim for partitioned space
        total_latent_dim = shared_latent_dim + modality_latent_dim
        kwargs["latent_dim"] = total_latent_dim
        kwargs["resolution"] = resolution

        # Call parent constructor
        super().__init__(**kwargs)

        # Store resolution for our use
        self.resolution = resolution

        # Now we can safely access self.resolution and other BaseVAE attributes

        # Modality embedding for conditioning
        self.modality_embedding = nn.Embedding(num_modalities, 64)

        # Modality-specific projection layers for encoder
        # Note: We'll compute the actual projection size dynamically in create_modality_condition_map
        self.modality_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 512),  # Fixed size, we'll adapt dynamically
                )
                for _ in range(num_modalities)
            ]
        )

        # Modify encoder input to accept modality conditioning
        # We'll store the original conv_in and replace it dynamically in encode()
        self.original_conv_in = self.encoder.conv_in

        # Modality-specific decoder heads
        self.modality_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.input_channels, self.input_channels, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(self.input_channels, self.input_channels, 3, 1, 1),
                )
                for _ in range(num_modalities)
            ]
        )

    def create_modality_condition_map(
        self, modality_indices: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Create spatial modality condition map."""
        batch_size = modality_indices.shape[0]

        # Get modality embeddings
        modality_embeds = self.modality_embedding(modality_indices)  # [B, 64]

        # Calculate target size for projection
        target_size = self.input_channels * height * width

        # Project to spatial dimensions with adaptive linear layer
        condition_maps = []
        for i in range(batch_size):
            modality_idx = modality_indices[i].item()

            # Get base projection (fixed size 512)
            base_proj = self.modality_projections[modality_idx](
                modality_embeds[i : i + 1]
            )  # [1, 512]

            # Adapt to target size
            if base_proj.shape[1] != target_size:
                # Use adaptive pooling or linear layer to match target size
                if base_proj.shape[1] > target_size:
                    # Downsample
                    adapted_proj = F.adaptive_avg_pool1d(
                        base_proj.unsqueeze(1), target_size
                    ).squeeze(1)
                else:
                    # Upsample by repeating and truncating
                    repeat_factor = (target_size // base_proj.shape[1]) + 1
                    repeated = base_proj.repeat(1, repeat_factor)
                    adapted_proj = repeated[:, :target_size]
            else:
                adapted_proj = base_proj

            condition_map = adapted_proj.view(
                self.input_channels, height, width
            )  # [C, H, W]
            condition_maps.append(condition_map)

        return torch.stack(condition_maps, dim=0)  # [B, C, H, W]

    def encode(
        self, x: torch.Tensor, modality_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with modality conditioning."""
        # Create modality condition map
        condition_map = self.create_modality_condition_map(
            modality_indices, x.shape[2], x.shape[3]
        )

        # Concatenate input with condition
        conditioned_input = torch.cat([x, condition_map], dim=1)

        # Dynamically adjust conv_in if needed
        input_channels = conditioned_input.shape[1]
        if self.encoder.conv_in.in_channels != input_channels:
            # Create new conv_in with correct input channels
            original_conv = self.original_conv_in
            self.encoder.conv_in = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                device=original_conv.weight.device,
                dtype=original_conv.weight.dtype,
            )
            # Initialize weights (simple approach)
            with torch.no_grad():
                # Copy original weights for the first input_channels
                original_weight = original_conv.weight
                new_weight = torch.zeros_like(self.encoder.conv_in.weight)

                # Handle different numbers of input channels
                min_channels = min(original_weight.shape[1], input_channels)
                new_weight[:, :min_channels] = original_weight[:, :min_channels]

                # Initialize additional channels (if any) with small random values
                if input_channels > original_weight.shape[1]:
                    remaining_channels = input_channels - original_weight.shape[1]
                    new_weight[:, -remaining_channels:] = (
                        torch.randn_like(new_weight[:, -remaining_channels:]) * 0.01
                    )

                self.encoder.conv_in.weight.copy_(new_weight)
                if original_conv.bias is not None:
                    self.encoder.conv_in.bias.copy_(original_conv.bias)

        # Encode
        mu, logvar = super().encode(conditioned_input)

        return mu, logvar

    def partition_latent(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Partition latent vector into shared and modality-specific parts."""
        batch_size = z.shape[0]
        z_flat = z.view(batch_size, -1)

        z_shared = z_flat[:, : self.shared_latent_dim]
        z_modality = z_flat[
            :,
            self.shared_latent_dim : self.shared_latent_dim + self.modality_latent_dim,
        ]

        return z_shared, z_modality

    def reconstruct_latent(
        self, z_shared: torch.Tensor, z_modality: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct full latent vector from partitioned components."""
        # Pad to full latent_dim if needed
        remaining_dim = (
            self.latent_dim - self.shared_latent_dim - self.modality_latent_dim
        )
        if remaining_dim > 0:
            padding = torch.zeros(
                z_shared.shape[0], remaining_dim, device=z_shared.device
            )
            z_full = torch.cat([z_shared, z_modality, padding], dim=1)
        else:
            z_full = torch.cat([z_shared, z_modality], dim=1)

        # Reshape to spatial format
        spatial_size = int((z_full.shape[1] / (self.encoder_out_res**2)) ** 0.5)
        if spatial_size**2 * (self.encoder_out_res**2) != z_full.shape[1]:
            # If dimensions don't match perfectly, use the encoder output resolution
            z_reshaped = z_full.view(
                z_full.shape[0], -1, self.encoder_out_res, self.encoder_out_res
            )
        else:
            z_reshaped = z_full.view(
                z_full.shape[0],
                spatial_size,
                self.encoder_out_res,
                self.encoder_out_res,
            )

        return z_reshaped

    def decode(
        self, z: torch.Tensor, modality_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode with optional modality-specific processing."""
        # Standard decode
        reconstruction = super().decode(z)

        # Apply modality-specific processing if modality provided
        if modality_indices is not None:
            batch_size = reconstruction.shape[0]
            processed_recons = []

            for i in range(batch_size):
                modality_idx = modality_indices[i].item()
                modality_decoder = self.modality_decoders[modality_idx]
                processed_recon = modality_decoder(reconstruction[i : i + 1])
                processed_recons.append(processed_recon)

            reconstruction = torch.cat(processed_recons, dim=0)

        return reconstruction

    def modality_separation_loss(
        self, z: torch.Tensor, modality_indices: torch.Tensor
    ) -> torch.Tensor:
        """Encourage different modalities to use different latent regions."""
        z_shared, z_modality = self.partition_latent(z)

        # Compute modality centroids
        unique_modalities = torch.unique(modality_indices)
        centroids = []

        for modality in unique_modalities:
            mask = modality_indices == modality
            if mask.sum() > 0:
                centroid = z_modality[mask].mean(dim=0)
                centroids.append(centroid)

        if len(centroids) < 2:
            return torch.tensor(0.0, device=z.device)

        centroids = torch.stack(centroids)

        # if device is MPS, use alternative distance computation
        if z.device.type == "mps":
            # Maximize distances between centroids (MPS-compatible implementation)
            # Instead of torch.pdist, compute pairwise distances manually
            n_centroids = centroids.shape[0]
            distances = []
            for i in range(n_centroids):
                for j in range(i + 1, n_centroids):
                    dist = torch.norm(centroids[i] - centroids[j], p=2)
                    distances.append(dist)
        else:
            # Use pdist for other devices
            distances = torch.pdist(centroids, p=2)

        if len(distances) == 0:
            return torch.tensor(0.0, device=z.device)

        distances = torch.stack(distances)
        separation_loss = -distances.mean()  # Negative to maximize distances

        return separation_loss

    def contrastive_loss(
        self, z: torch.Tensor, modality_indices: torch.Tensor, temperature: float = 0.1
    ) -> torch.Tensor:
        """Contrastive loss to cluster same modalities and separate different ones."""
        z_shared, z_modality = self.partition_latent(z)

        # Normalize for contrastive learning
        z_norm = F.normalize(z_modality, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(z_norm, z_norm.t()) / temperature

        # Create positive pairs mask (same modality)
        modality_mask = modality_indices.unsqueeze(0) == modality_indices.unsqueeze(1)
        modality_mask.fill_diagonal_(False)  # Remove self-comparisons

        # Contrastive loss
        exp_sim = torch.exp(similarity_matrix)

        # Positive similarities
        pos_sim = exp_sim * modality_mask.float()

        # All similarities (for normalization)
        all_sim = exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)

        # Contrastive loss
        contrastive_loss = -torch.log(pos_sim.sum(dim=1) / all_sim.squeeze() + 1e-8)
        contrastive_loss = contrastive_loss[
            pos_sim.sum(dim=1) > 0
        ]  # Only for samples with positive pairs

        return (
            contrastive_loss.mean()
            if len(contrastive_loss) > 0
            else torch.tensor(0.0, device=z.device)
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_indices: torch.Tensor,
        return_latents: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with modality conditioning and disentanglement losses."""
        # Encode with modality conditioning
        mu, logvar = self.encode(x, modality_indices)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode with modality-specific processing
        reconstruction = self.decode(z, modality_indices)

        # Compute additional losses
        separation_loss = self.modality_separation_loss(z, modality_indices)
        contrastive_loss_val = self.contrastive_loss(z, modality_indices)

        # Create distributions for loss computation (match BaseVAE)
        prior = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        posterior = Normal(mu, torch.exp(0.5 * logvar))

        # Create output dictionary
        output = {
            "reconstruction": reconstruction,
            "mean": mu,  # Lightning module expects 'mean'
            "logvar": logvar,
            "mu": mu,  # Keep both for compatibility
            "z": z,
            "prior": prior,
            "posterior": posterior,
            "separation_loss": separation_loss,
            "contrastive_loss": contrastive_loss_val,
        }

        if return_latents:
            z_shared, z_modality = self.partition_latent(z)
            output.update(
                {
                    "z_shared": z_shared,
                    "z_modality": z_modality,
                }
            )

        return output

    def sample_conditional(
        self, num_samples: int, modality_indices: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Sample from specific modalities."""
        # Use the same approach as the parent BaseVAE.sample() method
        # Sample from the latent space with the same dimensions as used in training
        z = torch.randn(
            num_samples,
            self.latent_dim,
            self.encoder_out_res,
            self.encoder_out_res,
            device=device,
        )

        # Apply modality-specific modifications to the latent representation
        # This provides conditional generation based on modality
        with torch.no_grad():
            for i, modality_idx in enumerate(modality_indices):
                # Add modality-specific bias/shift to encourage different generations
                # Use a deterministic shift based on modality index
                modality_shift = (
                    modality_idx.float() - 2.0
                ) * 0.3  # Center around 0 for modality 2
                z[i] = z[i] + modality_shift

        # Decode with modality conditioning
        return self.decode(z, modality_indices)


class DisentangledVAELoss(nn.Module):
    """Loss function for disentangled conditional VAE."""

    def __init__(
        self,
        recon_loss_type: str = "mse",
        kl_weight: float = 1.0,
        recon_weight: float = 1.0,
        separation_weight: float = 0.1,
        contrastive_weight: float = 0.05,
    ):
        super().__init__()
        self.recon_loss_type = recon_loss_type
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.separation_weight = separation_weight
        self.contrastive_weight = contrastive_weight

        if recon_loss_type == "mse":
            self.recon_criterion = nn.MSELoss()
        elif recon_loss_type == "l1":
            self.recon_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown reconstruction loss: {recon_loss_type}")

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss with disentanglement terms."""
        reconstruction = outputs["reconstruction"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        separation_loss = outputs["separation_loss"]
        contrastive_loss = outputs["contrastive_loss"]

        # Reconstruction loss
        recon_loss = self.recon_criterion(reconstruction, targets)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / targets.numel()  # Normalize by number of elements

        # Total loss
        total_loss = (
            self.recon_weight * recon_loss
            + self.kl_weight * kl_loss
            + self.separation_weight * separation_loss
            + self.contrastive_weight * contrastive_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "separation_loss": separation_loss,
            "contrastive_loss": contrastive_loss,
        }
