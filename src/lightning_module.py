"""
Lightning module for VAE training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from typing import Dict, Any, Optional, Tuple
import wandb

from src.models import BaseVAE, BetaVAE, ConditionalVAE, DisentangledConditionalVAE
from src.losses import VAELoss, LPIPSLoss, BiomedCLIPLoss, LPIPSWithDiscriminator
from src.models.disentangled_conditional_vae import DisentangledVAELoss
from src.utils import compute_reconstruction_metrics, compute_kl_metrics, get_scheduler


class VAELightningModule(L.LightningModule):
    """Lightning module for VAE training."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.loss_config = loss_config

        # Setup loss function
        self._setup_loss()

        # Enable manual optimization only for discriminator losses
        if hasattr(self.criterion, "discriminator"):
            self.automatic_optimization = False
        else:
            self.automatic_optimization = True

    def _setup_loss(self):
        """Setup loss function based on configuration."""
        loss_type = self.loss_config.get("type", "vae")

        if loss_type == "vae":
            self.criterion = VAELoss(
                recon_loss_type=self.loss_config.get("recon_loss_type", "mse"),
                kl_weight=self.loss_config.get("kl_weight", 1.0),
                recon_weight=self.loss_config.get("recon_weight", 1.0),
            )
        elif loss_type == "disentangled_vae":
            self.criterion = DisentangledVAELoss(
                recon_loss_type=self.loss_config.get("recon_loss_type", "mse"),
                kl_weight=self.loss_config.get("kl_weight", 1.0),
                recon_weight=self.loss_config.get("recon_weight", 1.0),
                separation_weight=self.loss_config.get("separation_weight", 0.1),
                contrastive_weight=self.loss_config.get("contrastive_weight", 0.05),
            )
        elif loss_type == "lpips":
            self.criterion = LPIPSLoss()
        elif loss_type == "biomedclip":
            self.criterion = BiomedCLIPLoss(
                compute_rec_loss=True, compute_lat_loss=False
            )
        elif loss_type == "lpips_discriminator":
            self.criterion = LPIPSWithDiscriminator(
                discriminator_factor=self.loss_config.get("discriminator_factor", 1.0),
                perceptual_factor=self.loss_config.get("perceptual_factor", 1.0),
                kl_factor=self.loss_config.get("kl_factor", 1.0),
                discriminator_iter_start=self.loss_config.get(
                    "discriminator_iter_start", 50001
                ),
                use_biomedclip_loss=self.loss_config.get("use_biomedclip_loss", False),
                biomedclip_factor=self.loss_config.get("biomedclip_factor", 1.0),
                discriminator_config=self.loss_config.get("discriminator", {}),
            )
            self.use_discriminator = True
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.use_discriminator = loss_type == "lpips_discriminator"

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """Forward pass."""
        if (
            isinstance(self.model, (ConditionalVAE, DisentangledConditionalVAE))
            and condition is not None
        ):
            return self.model(x, condition)
        else:
            return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """Training step."""
        if len(batch) == 3:
            x, labels, modality = batch
        else:
            x, labels = batch
            modality = None

        # Forward pass
        if (
            isinstance(self.model, (ConditionalVAE, DisentangledConditionalVAE))
            and modality is not None
        ):
            if isinstance(self.model, DisentangledConditionalVAE):
                # Convert one-hot modality to indices for DisentangledConditionalVAE
                modality_indices = torch.argmax(modality, dim=1)
                outputs = self.model(x, modality_indices)
            else:
                outputs = self.model(x, modality)
        else:
            outputs = self.model(x)

        # Compute loss
        if self.use_discriminator:
            # Dual optimization for discriminator
            opt_g, opt_d = self.optimizers()

            # Generator step
            loss_g, log_g = self.criterion(
                inputs=x,
                reconstructions=outputs["reconstruction"],
                latent=outputs["z"],
                posteriors=outputs["posterior"],
                optimizer_idx=0,
                global_step=self.global_step,
                last_layer=self.model.decoder.conv_out,
                split="train",
            )

            # Backward generator
            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

            # Discriminator step
            loss_d, log_d = self.criterion(
                inputs=x,
                reconstructions=outputs["reconstruction"].detach(),
                latent=outputs["z"].detach(),
                posteriors=outputs["posterior"],
                optimizer_idx=1,
                global_step=self.global_step,
                last_layer=None,
                split="train",
            )

            # Backward discriminator
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()

            # Log combined
            log_dict = {**log_g, **log_d}
            self.log_dict(
                log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
            )

            return loss_g

        else:
            # Check if this is a disentangled VAE loss
            if hasattr(self.criterion, "separation_weight"):
                # DisentangledVAELoss expects different parameters
                loss_dict = self.criterion(outputs, x)
            else:
                # Standard VAE loss
                loss_dict = self.criterion(
                    inputs=x,
                    reconstructions=outputs["reconstruction"],
                    posteriors=outputs["posterior"],
                    priors=outputs["prior"],
                )

            loss = loss_dict["loss"]

            # Log losses
            for key, value in loss_dict.items():
                self.log(
                    f"train/{key}",
                    value,
                    prog_bar=True,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                )

            return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """Validation step."""
        if len(batch) == 3:
            x, labels, modality = batch
        else:
            x, labels = batch
            modality = None

        # Forward pass
        if (
            isinstance(self.model, (ConditionalVAE, DisentangledConditionalVAE))
            and modality is not None
        ):
            if isinstance(self.model, DisentangledConditionalVAE):
                # Convert one-hot modality to indices for DisentangledConditionalVAE
                modality_indices = torch.argmax(modality, dim=1)
                outputs = self.model(x, modality_indices)
            else:
                outputs = self.model(x, modality)
        else:
            outputs = self.model(x)

        # Compute metrics
        recon_metrics = compute_reconstruction_metrics(x, outputs["reconstruction"])
        kl_metrics = compute_kl_metrics(outputs["mean"], outputs["logvar"])

        # Log metrics
        for key, value in recon_metrics.items():
            self.log(f"val/{key}", value, prog_bar=False, logger=True, on_epoch=True)

        for key, value in kl_metrics.items():
            self.log(f"val/{key}", value, prog_bar=False, logger=True, on_epoch=True)

        # Compute total loss for monitoring
        if hasattr(self.criterion, "__call__"):
            if self.use_discriminator:
                loss, _ = self.criterion(
                    inputs=x,
                    reconstructions=outputs["reconstruction"],
                    latent=outputs["z"],
                    posteriors=outputs["posterior"],
                    optimizer_idx=0,
                    global_step=self.global_step,
                    last_layer=self.model.decoder.conv_out,
                    split="val",
                )
            else:
                loss_dict = self.criterion(
                    inputs=x,
                    reconstructions=outputs["reconstruction"],
                    posteriors=outputs["posterior"],
                    priors=outputs["prior"],
                )
                loss = loss_dict["loss"]
        else:
            # Fallback to MSE
            loss = nn.functional.mse_loss(outputs["reconstruction"], x)

        self.log("val/loss", loss, prog_bar=True, logger=True, on_epoch=True)

        return outputs

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Main optimizer (generator/vae)
        if self.optimizer_config["type"] == "adam":
            optimizer_g = optim.Adam(
                self.model.parameters(),
                lr=self.optimizer_config["lr"],
                weight_decay=self.optimizer_config.get("weight_decay", 0),
                betas=self.optimizer_config.get("betas", [0.9, 0.999]),
            )
        elif self.optimizer_config["type"] == "adamw":
            optimizer_g = optim.AdamW(
                self.model.parameters(),
                lr=self.optimizer_config["lr"],
                weight_decay=self.optimizer_config.get("weight_decay", 1e-4),
                betas=self.optimizer_config.get("betas", [0.9, 0.999]),
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_config['type']}")

        optimizers = [optimizer_g]
        schedulers = []

        # Add scheduler for generator
        scheduler_g = get_scheduler(optimizer_g, self.scheduler_config)
        if scheduler_g is not None:
            schedulers.append(
                {
                    "scheduler": scheduler_g,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            )

        # Add discriminator optimizer if needed
        if self.use_discriminator:
            optimizer_d = optim.Adam(
                self.criterion.discriminator.parameters(),
                lr=self.optimizer_config["lr"]
                * 0.5,  # Usually slower for discriminator
                betas=[0.5, 0.999],
            )
            optimizers.append(optimizer_d)

            # Discriminator scheduler
            scheduler_d = get_scheduler(optimizer_d, self.scheduler_config)
            if scheduler_d is not None:
                schedulers.append(
                    {
                        "scheduler": scheduler_d,
                        "monitor": "val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    }
                )

        if schedulers:
            return optimizers, schedulers
        else:
            return optimizers

    def on_validation_epoch_end(self):
        """Log sample images at end of validation."""
        if self.current_epoch % 10 == 0:  # Every 10 epochs
            self._log_sample_images()

    def _log_sample_images(self):
        """Log sample reconstructions and generations to wandb."""
        if not hasattr(self.logger, "experiment") or self.logger.experiment is None:
            return

        # Get a batch from validation
        val_dataloader = self.trainer.datamodule.val_dataloader()
        batch = next(iter(val_dataloader))

        if len(batch) == 3:
            x, labels, modality = batch
        else:
            x, labels = batch
            modality = None

        x = x[:8].to(self.device)  # Take first 8 samples
        if modality is not None:
            modality = modality[:8].to(self.device)

        with torch.no_grad():
            # Reconstructions
            if (
                isinstance(self.model, (ConditionalVAE, DisentangledConditionalVAE))
                and modality is not None
            ):
                if isinstance(self.model, DisentangledConditionalVAE):
                    # Convert one-hot modality to indices for DisentangledConditionalVAE
                    modality_indices = torch.argmax(modality, dim=1)
                    outputs = self.model(x, modality_indices)
                else:
                    outputs = self.model(x, modality)
            else:
                outputs = self.model(x)

            reconstructions = outputs["reconstruction"]

            # Samples from prior
            samples = self.model.sample(8, self.device)

        # Convert to wandb images
        original_images = [wandb.Image(img.cpu()) for img in x]
        recon_images = [wandb.Image(img.cpu()) for img in reconstructions]
        sample_images = [wandb.Image(img.cpu()) for img in samples]

        # Log based on logger type
        if hasattr(self.logger, "experiment") and hasattr(
            self.logger.experiment, "log"
        ):
            # WandB logger
            self.logger.experiment.log(
                {
                    "original_images": original_images,
                    "reconstructed_images": recon_images,
                    "generated_samples": sample_images,
                    "epoch": self.current_epoch,
                }
            )
        else:
            # TensorBoard or other logger - just log metrics
            print(f"Epoch {self.current_epoch}: Generated {len(sample_images)} samples")
