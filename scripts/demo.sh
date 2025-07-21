#!/bin/bash

# Quick demo script to test the setup with uv

echo "MedMNIST VAE Quick Demo"
echo "======================"

echo "Setting up environment with uv..."
# Install dependencies
uv sync

echo "Downloading sample data..."
uv run python -c "
import medmnist
from medmnist import ChestMNIST
# Download a small sample to test
dataset = ChestMNIST(split='train', download=True, size=28, root='./data')
print(f'Downloaded {len(dataset)} training samples')
"

echo "Testing basic model..."
uv run python -c "
from src.models import BaseVAE
import torch

# Test model initialization - much smaller model
model = BaseVAE(
    input_channels=1, 
    latent_dim=16, 
    hidden_channels=32,
    ch_mult=[1, 2, 4],
    num_res_blocks=1,
    attn_resolutions=[],
    resolution=28
)
print(f'Model initialized with {sum(p.numel() for p in model.parameters())} parameters')

# Test forward pass
x = torch.randn(4, 1, 28, 28)
outputs = model(x)
print(f'Forward pass successful: {outputs[\"reconstruction\"].shape}')
"

echo "Quick training test (5 epochs)..."
uv run python main.py \
    experiment=chest_base_vae_quick \
    wandb.enabled=false

echo "Demo completed successfully!"
echo ""
echo "Available quick experiments:"
echo "  uv run python main.py experiment=chest_base_vae_quick"
echo "  uv run python main.py experiment=chest_beta_vae_quick" 
echo "  uv run python main.py experiment=multi_modal_cvae_quick"
echo ""
echo "For full training: uv run python main.py experiment=chest_base_vae"
