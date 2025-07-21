#!/bin/bash

# Training scripts for different VAE experiments using uv

echo "MedMNIST Conditional VAE Training Scripts"
echo "========================================"

# Base VAE on ChestMNIST
echo "1. Training Base VAE on ChestMNIST..."
uv run train experiment=chest_base_vae

# Beta-VAE on PathMNIST  
# echo "2. Training Beta-VAE on PathMNIST..."
# uv run train experiment=path_beta_vae

# Conditional VAE on multiple modalities
echo "3. Training Conditional VAE on multiple modalities..."
uv run train experiment=multi_modal_cvae

echo "All training experiments completed!"
