"""
VAE loss functions inspired by Stanford MedVAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, Optional, Any
import lpips
import open_clip
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

from ..models.discriminator import NLayerDiscriminator


class VAELoss(nn.Module):
    """Standard VAE loss with reconstruction and KL terms."""

    def __init__(
        self,
        recon_loss_type: str = "mse",  # "mse", "l1", "bce"
        kl_weight: float = 1.0,
        recon_weight: float = 1.0,
    ):
        super().__init__()
        self.recon_loss_type = recon_loss_type
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        posteriors: Normal,
        priors: Normal,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss."""
        # Reconstruction loss
        if self.recon_loss_type == "mse":
            recon_loss = F.mse_loss(reconstructions, inputs, reduction="mean")
        elif self.recon_loss_type == "l1":
            recon_loss = F.l1_loss(reconstructions, inputs, reduction="mean")
        elif self.recon_loss_type == "bce":
            recon_loss = F.binary_cross_entropy_with_logits(
                reconstructions, inputs, reduction="mean"
            )
        else:
            raise ValueError(
                f"Unknown reconstruction loss type: {self.recon_loss_type}"
            )

        # KL divergence
        kl_loss = kl_divergence(posteriors, priors).mean()

        # Total loss
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity loss."""

    def __init__(self, net: str = "alex", use_gpu: bool = True):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)
        if use_gpu and torch.cuda.is_available():
            self.lpips = self.lpips.cuda()

        # Freeze LPIPS parameters
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(
        self, inputs: torch.Tensor, reconstructions: torch.Tensor
    ) -> torch.Tensor:
        """Compute LPIPS loss."""
        # Ensure inputs are in [-1, 1] range for LPIPS
        inputs = inputs * 2.0 - 1.0
        reconstructions = reconstructions * 2.0 - 1.0

        # Convert grayscale to RGB if needed
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        if reconstructions.shape[1] == 1:
            reconstructions = reconstructions.repeat(1, 3, 1, 1)

        return self.lpips(inputs, reconstructions).mean()


class BiomedCLIPLoss(nn.Module):
    """
    BiomedCLIP-inspired loss for medical image reconstruction.
    Simplified version without actual BiomedCLIP model.
    """

    def __init__(self, compute_rec_loss: bool = True, compute_lat_loss: bool = False):
        super().__init__()

        # Use a simpler vision model as placeholder
        try:
            self.clip, _, _ = open_clip.create_model_and_transforms(
                model_name="ViT-B-32", pretrained="openai"
            )
        except:
            # Fallback to a simple CNN encoder if OpenCLIP is not available
            self.clip = SimpleCLIPEncoder()

        # Freeze parameters
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()

        # Image preprocessing
        self.transform = Compose(
            [
                Resize(size=224, interpolation=3, max_size=None, antialias=True),
                CenterCrop(size=(224, 224)),
                Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        self.compute_rec_loss = compute_rec_loss
        self.compute_lat_loss = compute_lat_loss

    def forward(
        self,
        img: torch.Tensor,
        rec: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute BiomedCLIP-inspired loss."""
        # Normalize image to [0, 1]
        img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)

        # Convert grayscale to RGB
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)

        # Transform and encode original image
        img_transformed = self.transform(img)
        img_features = self.encode_image(img_transformed)

        total_loss = torch.tensor(0.0, device=img.device)

        if self.compute_rec_loss and rec is not None:
            # Process reconstruction
            rec = torch.clamp((rec + 1.0) / 2.0, min=0.0, max=1.0)
            if rec.shape[1] == 1:
                rec = rec.expand(-1, 3, -1, -1)

            rec_transformed = self.transform(rec)
            rec_features = self.encode_image(rec_transformed)

            # Compute feature difference
            rec_loss = ((img_features - rec_features) ** 2).sum(1).mean()
            total_loss += rec_loss

        if self.compute_lat_loss and latent is not None:
            # Process latent (simplified)
            latent = latent / 4.6  # Normalization factor
            latent = latent.mean(1, keepdim=True)  # Pool across channels
            latent = F.interpolate(latent, size=(224, 224), mode="bilinear")
            latent = latent.expand(-1, 3, -1, -1)

            latent_features = self.encode_image(latent)
            lat_loss = ((img_features - latent_features) ** 2).sum(1).mean()
            total_loss += lat_loss

        return total_loss

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image using vision model."""
        if hasattr(self.clip, "encode_image"):
            return self.clip.encode_image(x)
        else:
            return self.clip(x)


class SimpleCLIPEncoder(nn.Module):
    """Simple CNN encoder as CLIP fallback."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LPIPSWithDiscriminator(nn.Module):
    """
    Combined loss with LPIPS, discriminator, and optional BiomedCLIP.
    Inspired by Stanford MedVAE loss implementation.
    """

    def __init__(
        self,
        discriminator_factor: float = 1.0,
        perceptual_factor: float = 1.0,
        kl_factor: float = 1.0,
        discriminator_iter_start: int = 50001,
        use_biomedclip_loss: bool = False,
        biomedclip_factor: float = 1.0,
        discriminator_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Loss factors
        self.discriminator_factor = discriminator_factor
        self.perceptual_factor = perceptual_factor
        self.kl_factor = kl_factor
        self.biomedclip_factor = biomedclip_factor
        self.discriminator_iter_start = discriminator_iter_start
        self.use_biomedclip_loss = use_biomedclip_loss

        # Perceptual loss
        self.perceptual_loss = LPIPSLoss()

        # Discriminator
        if discriminator_config is None:
            discriminator_config = {"input_nc": 3, "ndf": 64, "n_layers": 3}
        self.discriminator = NLayerDiscriminator(**discriminator_config)

        # BiomedCLIP loss
        if self.use_biomedclip_loss:
            self.biomed_clip_loss = BiomedCLIPLoss(
                compute_rec_loss=True, compute_lat_loss=False
            )

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        latent: torch.Tensor,
        posteriors: Normal,
        optimizer_idx: int,
        global_step: int,
        last_layer: nn.Module,
        split: str = "train",
        **kwargs,
    ) -> tuple:
        """
        Compute loss for generator or discriminator.

        Args:
            optimizer_idx: 0 for generator, 1 for discriminator
        """
        bsz = inputs.shape[0]

        if optimizer_idx == 0:
            # Generator loss

            # Perceptual loss
            p_loss = self.perceptual_loss(inputs, reconstructions)

            # BiomedCLIP loss
            bc_loss = torch.tensor(0.0, device=inputs.device)
            if self.use_biomedclip_loss:
                bc_loss = self.biomed_clip_loss(
                    inputs.contiguous(), rec=reconstructions.contiguous(), latent=None
                )
                bc_loss = bc_loss.sum() / bsz

            # KL regularization loss
            kl_loss = posteriors.kl()
            kl_loss = kl_loss.sum() / bsz

            # Generator loss (adversarial)
            d_valid = 0 if global_step < self.discriminator_iter_start else 1
            d_weight = torch.tensor(0.0, device=inputs.device)

            if d_valid:
                # Convert grayscale to RGB for discriminator
                if reconstructions.shape[1] == 1:
                    reconstructions_rgb = reconstructions.repeat(1, 3, 1, 1)
                else:
                    reconstructions_rgb = reconstructions

                logits_fake = self.discriminator(reconstructions_rgb.contiguous())
                g_loss = -torch.mean(logits_fake)

                # Gradient penalty for stability
                try:
                    d_weight = self.calculate_adaptive_weight(
                        p_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)

                d_weight = d_weight * self.discriminator_factor
            else:
                g_loss = torch.tensor(0.0, device=inputs.device)

            # Total generator loss
            loss = (
                self.perceptual_factor * p_loss
                + self.kl_factor * kl_loss
                + d_weight * g_loss
            )

            if self.use_biomedclip_loss:
                loss += self.biomedclip_factor * bc_loss

            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/g_loss": g_loss.detach().mean(),
            }

            if self.use_biomedclip_loss:
                log[f"{split}/bc_loss"] = bc_loss.detach().mean()

            return loss, log

        elif optimizer_idx == 1:
            # Discriminator loss
            d_valid = 0 if global_step < self.discriminator_iter_start else 1

            if d_valid:
                # Convert grayscale to RGB
                if inputs.shape[1] == 1:
                    inputs_rgb = inputs.repeat(1, 3, 1, 1)
                    reconstructions_rgb = reconstructions.repeat(1, 3, 1, 1)
                else:
                    inputs_rgb = inputs
                    reconstructions_rgb = reconstructions

                logits_real = self.discriminator(inputs_rgb.contiguous().detach())
                logits_fake = self.discriminator(
                    reconstructions_rgb.contiguous().detach()
                )

                d_loss = 0.5 * (
                    torch.mean(F.relu(1.0 - logits_real))
                    + torch.mean(F.relu(1.0 + logits_fake))
                )
            else:
                d_loss = torch.tensor(0.0, device=inputs.device)

            log = {f"{split}/d_loss": d_loss.detach().mean()}

            return d_loss, log

    def calculate_adaptive_weight(
        self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: nn.Module
    ) -> torch.Tensor:
        """Calculate adaptive weight for adversarial loss."""
        nll_grads = torch.autograd.grad(nll_loss, last_layer.weight, retain_graph=True)[
            0
        ]
        g_grads = torch.autograd.grad(g_loss, last_layer.weight, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        return d_weight
