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
    3. Modality-specific encoders/decoders with proper channel handling
    4. Regularization for modality separation
    """

    def __init__(
        self,
        num_modalities: int = 5,
        shared_latent_dim: int = 8,
        modality_latent_dim: int = 8,
        modality_separation_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        resolution: int = 28,
        hidden_channels: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: list = [16],
        dropout: float = 0.0,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
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
        
        # Define channel requirements for each modality
        self.modality_channels = self._get_modality_channel_map()
        
        # Use max channels for base VAE - we'll handle modality-specific processing separately
        max_channels = max(self.modality_channels.values())
        
        kwargs.update({
            "latent_dim": total_latent_dim,
            "resolution": resolution,
            "input_channels": max_channels,  # Base VAE uses max channels
            "hidden_channels": hidden_channels,
            "ch_mult": ch_mult,
            "num_res_blocks": num_res_blocks,
            "attn_resolutions": attn_resolutions,
            "dropout": dropout,
            "use_linear_attn": use_linear_attn,
            "attn_type": attn_type,
        })

        # Call parent constructor
        super().__init__(**kwargs)

        # Store resolution for our use
        self.resolution = resolution

        # Create modality-specific input projection layers
        self.modality_input_projectors = nn.ModuleDict()
        for modality_idx, channels in self.modality_channels.items():
            if channels != max_channels:
                # Create projection layer to convert to max_channels
                self.modality_input_projectors[str(modality_idx)] = nn.Conv2d(
                    channels, max_channels, kernel_size=1, padding=0
                )

        # Create modality-specific output projection layers  
        self.modality_output_projectors = nn.ModuleDict()
        for modality_idx, channels in self.modality_channels.items():
            if channels != max_channels:
                # Create projection layer to convert from max_channels back to modality channels
                self.modality_output_projectors[str(modality_idx)] = nn.Conv2d(
                    max_channels, channels, kernel_size=1, padding=0
                )

        # Modality embedding for conditioning
        self.modality_embedding = nn.Embedding(num_modalities, 64)

        # Modality-specific decoder heads for final processing
        self.modality_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(max_channels, max_channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(max_channels, max_channels, 3, 1, 1),
            )
            for _ in range(num_modalities)
        ])
    
    def _get_modality_channel_map(self) -> Dict[int, int]:
        """Define channel requirements for each modality."""
        # Map modality indices to channel counts
        # Based on the modality mapping in the data module
        return {
            0: 1,  # chestmnist - grayscale X-ray
            1: 3,  # pathmnist - color pathology  
            2: 3,  # octmnist - color OCT
            3: 1,  # pneumoniamnist - grayscale X-ray
            4: 3,  # dermamnist - color dermatoscope
        }

    def encode(
        self, x: torch.Tensor, modality_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with modality-specific channel handling."""
        batch_size = x.shape[0]
        processed_inputs = []
        
        # Add input validation
        if torch.isnan(x).any():
            print("Warning: NaN detected in input tensor x")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # Process each sample according to its modality
        for i in range(batch_size):
            sample = x[i:i+1]  # Keep batch dimension
            modality_idx = int(modality_indices[i].item())
            
            # Ensure modality_idx is within bounds
            if modality_idx >= len(self.modality_channels):
                print(f"Warning: modality_idx {modality_idx} is out of range for modality_channels. Using modality {len(self.modality_channels)-1}")
                modality_idx = len(self.modality_channels) - 1
            
            # Get the expected number of channels for this modality
            expected_channels = self.modality_channels[modality_idx]
            
            # Extract only the relevant channels (remove padding if any)
            if sample.shape[1] > expected_channels:
                sample = sample[:, :expected_channels, :, :]
            
            # Project input channels if needed
            projector_key = str(modality_idx)
            if projector_key in self.modality_input_projectors:
                sample = self.modality_input_projectors[projector_key](sample)
                
                # Check for NaN after projection
                if torch.isnan(sample).any():
                    print(f"Warning: NaN detected after projection for modality {modality_idx}")
                    sample = torch.where(torch.isnan(sample), torch.zeros_like(sample), sample)
            
            processed_inputs.append(sample)
        
        # Concatenate processed inputs
        processed_x = torch.cat(processed_inputs, dim=0)
        
        # Final input validation
        if torch.isnan(processed_x).any():
            print("Warning: NaN detected in processed input, replacing with zeros")
            processed_x = torch.where(torch.isnan(processed_x), torch.zeros_like(processed_x), processed_x)
        
        # Encode using base VAE
        mu, logvar = super().encode(processed_x)
        
        # Check encoder output for NaN
        if torch.isnan(mu).any():
            print("Warning: NaN detected in encoder mu output")
            mu = torch.where(torch.isnan(mu), torch.zeros_like(mu), mu)
            
        if torch.isnan(logvar).any():
            print("Warning: NaN detected in encoder logvar output")
            logvar = torch.where(torch.isnan(logvar), torch.zeros_like(logvar), logvar)
        
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
        """Decode with modality-specific channel handling."""
        # Standard decode - outputs in max_channels format
        reconstruction = super().decode(z)
        
        if modality_indices is not None:
            batch_size = reconstruction.shape[0]
            processed_outputs = []
            max_output_channels = 0

            # First pass: process all samples and find max channels
            temp_outputs = []
            for i in range(batch_size):
                sample = reconstruction[i:i+1]  # Keep batch dimension
                modality_idx = int(modality_indices[i].item())

                # Ensure modality_idx is within bounds
                if modality_idx >= len(self.modality_decoders):
                    # If modality index is out of range, use the last available decoder
                    print(f"Warning: modality_idx {modality_idx} is out of range. Using modality {len(self.modality_decoders)-1}")
                    modality_idx = len(self.modality_decoders) - 1

                # Apply modality-specific processing
                modality_decoder = self.modality_decoders[modality_idx]
                processed_sample = modality_decoder(sample)

                # Project back to modality-specific channels if needed
                projector_key = str(modality_idx)
                if projector_key in self.modality_output_projectors:
                    processed_sample = self.modality_output_projectors[projector_key](processed_sample)

                temp_outputs.append(processed_sample)
                max_output_channels = max(max_output_channels, processed_sample.shape[1])

            # Second pass: pad to max channels and concatenate
            for processed_sample in temp_outputs:
                if processed_sample.shape[1] < max_output_channels:
                    # Pad with zeros to match max channels
                    padding_shape = (
                        processed_sample.shape[0],
                        max_output_channels - processed_sample.shape[1],
                        *processed_sample.shape[2:]
                    )
                    padding = torch.zeros(padding_shape, dtype=processed_sample.dtype, device=processed_sample.device)
                    processed_sample = torch.cat([processed_sample, padding], dim=1)
                
                processed_outputs.append(processed_sample)

            reconstruction = torch.cat(processed_outputs, dim=0)

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
            
            if len(distances) == 0:
                return torch.tensor(0.0, device=z.device)
            
            distances = torch.stack(distances)
        else:
            # Use pdist for other devices
            distances = torch.pdist(centroids, p=2)
            
            if distances.numel() == 0:
                return torch.tensor(0.0, device=z.device)
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

        # Add numerical stability checks to prevent NaN values
        # Check for NaN values and replace with zeros if found
        if torch.isnan(mu).any():
            print("Warning: NaN detected in mu, replacing with zeros")
            mu = torch.where(torch.isnan(mu), torch.zeros_like(mu), mu)
        
        if torch.isnan(logvar).any():
            print("Warning: NaN detected in logvar, replacing with zeros")
            logvar = torch.where(torch.isnan(logvar), torch.zeros_like(logvar), logvar)
        
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        # Clamp mu to reasonable range
        mu = torch.clamp(mu, min=-10.0, max=10.0)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode with modality-specific processing
        reconstruction = self.decode(z, modality_indices)

        # Compute additional losses
        separation_loss = self.modality_separation_loss(z, modality_indices)
        contrastive_loss_val = self.contrastive_loss(z, modality_indices)

        # Create distributions for loss computation (match BaseVAE)
        # Ensure standard deviation is positive and stable
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-6, max=10.0)  # Prevent zero or extreme std
        
        prior = Normal(torch.zeros_like(mu), torch.ones_like(std))
        posterior = Normal(mu, std)

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

        # KL divergence with numerical stability
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / targets.numel()  # Normalize by number of elements
        
        # Add numerical stability checks
        if torch.isnan(recon_loss).any() or torch.isinf(recon_loss).any():
            print("Warning: NaN/Inf detected in reconstruction loss, replacing with zero")
            recon_loss = torch.tensor(0.0, device=reconstruction.device, dtype=recon_loss.dtype)
            
        if torch.isnan(kl_loss).any() or torch.isinf(kl_loss).any():
            print("Warning: NaN/Inf detected in KL loss, replacing with zero")
            kl_loss = torch.tensor(0.0, device=mu.device, dtype=kl_loss.dtype)
            
        if torch.isnan(separation_loss).any() or torch.isinf(separation_loss).any():
            print("Warning: NaN/Inf detected in separation loss, replacing with zero")
            separation_loss = torch.tensor(0.0, device=separation_loss.device, dtype=separation_loss.dtype)
            
        if torch.isnan(contrastive_loss).any() or torch.isinf(contrastive_loss).any():
            print("Warning: NaN/Inf detected in contrastive loss, replacing with zero")
            contrastive_loss = torch.tensor(0.0, device=contrastive_loss.device, dtype=contrastive_loss.dtype)

        # Total loss
        total_loss = (
            self.recon_weight * recon_loss
            + self.kl_weight * kl_loss
            + self.separation_weight * separation_loss
            + self.contrastive_weight * contrastive_loss
        )
        
        # Final check on total loss
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print("Warning: NaN/Inf detected in total loss, replacing with large value")
            total_loss = torch.tensor(1e6, device=total_loss.device, dtype=total_loss.dtype)

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "separation_loss": separation_loss,
            "contrastive_loss": contrastive_loss,
        }
