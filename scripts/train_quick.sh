#!/bin/bash

# Quick training scripts for lightweight VAE experiments using uv

echo "MedMNIST Conditional VAE - Quick Training"
echo "========================================"

echo "Available quick experiments (5-8 epochs each):"
echo ""

# Quick Base VAE on ChestMNIST (28x28, ~200k params)
echo "1. Quick Base VAE on ChestMNIST (28x28, ~200k params)..."
uv run python main.py experiment=chest_base_vae_quick wandb.enabled=false

# Quick Beta-VAE on ChestMNIST
echo "2. Quick Beta-VAE on ChestMNIST..."
uv run python main.py experiment=chest_beta_vae_quick wandb.enabled=false

# Quick Conditional VAE on multiple modalities
echo "3. Quick Conditional VAE on multiple modalities..."
uv run python main.py experiment=multi_modal_cvae_quick wandb.enabled=false

echo ""
echo "All quick training experiments completed!"
echo "Total time should be under 30 minutes on modern hardware."
