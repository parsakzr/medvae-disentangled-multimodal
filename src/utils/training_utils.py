"""
Training utilities and helper functions.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional
import numpy as np


def get_scheduler(
    optimizer: optim.Optimizer, scheduler_config: Dict[str, Any]
) -> Optional[_LRScheduler]:
    """
    Get learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Scheduler configuration
    """
    scheduler_type = scheduler_config.get("type", "none")

    if scheduler_type == "none":
        return None
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 30),
            gamma=scheduler_config.get("gamma", 0.1),
        )
    elif scheduler_type == "multistep":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config.get("milestones", [50, 100]),
            gamma=scheduler_config.get("gamma", 0.1),
        )
    elif scheduler_type == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=scheduler_config.get("gamma", 0.95)
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 100),
            eta_min=scheduler_config.get("eta_min", 0),
        )
    elif scheduler_type == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 10),
            threshold=scheduler_config.get("threshold", 1e-4),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class EarlyStopping:
    """Early stopping utility."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max" - direction of improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
        elif mode == "max":
            self.is_better = lambda score, best: score > best + min_delta
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric score

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Improvement detected, resetting counter")
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("EarlyStopping: Stopping training")

        return self.early_stop


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gradient_penalty(
    discriminator: torch.nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.

    Args:
        discriminator: Discriminator network
        real_samples: Real data samples
        fake_samples: Generated samples
        device: Device to run on

    Returns:
        Gradient penalty loss
    """
    batch_size = real_samples.shape[0]

    # Random interpolation factor
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    # Interpolated samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)

    # Discriminator output on interpolated samples
    d_interpolated = discriminator(interpolated)

    # Gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the gradient norm of model parameters."""
    total_norm = 0.0
    param_count = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1

    if param_count > 0:
        total_norm = total_norm ** (1.0 / 2)

    return total_norm


def exponential_moving_average(
    model: torch.nn.Module, ema_model: torch.nn.Module, decay: float = 0.999
):
    """Update exponential moving average of model parameters."""
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1 - decay)
