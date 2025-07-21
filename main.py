"""
Main training script for MedMNIST Conditional VAE.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import torch

from src.lightning_module import VAELightningModule
from src.utils.training_utils import set_random_seed, count_parameters


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function."""

    # Set random seed
    set_random_seed(cfg.seed)

    # Print configuration
    print("Training Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Initialize model
    model = hydra.utils.instantiate(cfg.model)
    print(f"Model initialized: {model.__class__.__name__}")

    # Count parameters
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count}")

    # Initialize data module
    datamodule = hydra.utils.instantiate(cfg.data)

    # Initialize lightning module
    lightning_module = VAELightningModule(
        model=model,
        optimizer_config=cfg.training.optimizer,
        scheduler_config=cfg.training.scheduler,
        loss_config=cfg.training.loss,
    )

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    if cfg.get("checkpointing"):
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            filename=f"{cfg.experiment_name}-{{epoch:02d}}-{{val/loss:.3f}}",
            monitor=cfg.checkpointing.monitor,
            mode=cfg.checkpointing.mode,
            save_top_k=cfg.checkpointing.save_top_k,
            save_last=cfg.checkpointing.save_last,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.get("early_stopping") and cfg.early_stopping.enabled:
        early_stopping_callback = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            mode=cfg.early_stopping.mode,
            verbose=True,
        )
        callbacks.append(early_stopping_callback)

    # Setup logger
    logger = None
    if cfg.get("wandb") and cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            save_dir=cfg.log_dir,
        )
        # Log hyperparameters
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
    )

    # Train model
    trainer.fit(lightning_module, datamodule=datamodule)

    # Test model
    if hasattr(datamodule, "test_dataloader"):
        trainer.test(lightning_module, datamodule=datamodule)

    print("Training completed!")

    # Save final model
    if cfg.get("checkpointing"):
        final_model_path = os.path.join(
            cfg.checkpoint_dir, f"{cfg.experiment_name}_final.ckpt"
        )
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    train()
