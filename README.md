# MedMNIST Conditional VAE

A comprehensive Conditional Variational Autoencoder (CVAE) implementation for medical imaging using MedMNIST datasets, inspired by Stanford MedVAE techniques.

## Features

- **Multiple VAE Variants**: Base VAE, Beta-VAE, and Conditional VAE implementations
- **Advanced CNN Architecture**: ResNet-style encoder/decoder with attention mechanisms
- **Medical Modality Conditioning**: One-hot encoding for different medical imaging modalities (X-ray, Pathology, OCT, etc.)
- **MedMNIST Integration**: Support for all 2D MedMNIST datasets
- **Hydra Configuration**: Flexible experiment management and hyperparameter tuning
- **Advanced Loss Functions**: Including LPIPS, BiomedCLIP-inspired losses, and KL regularization
- **Lightning Training**: Efficient training with PyTorch Lightning
- **Comprehensive Evaluation**: Reconstruction quality metrics and realistic sample generation

## Supported MedMNIST Datasets (2D Only)

- **ChestMNIST**: Chest X-Ray (Multi-Label, 14 classes)
- **PathMNIST**: Colon Pathology (Multi-Class, 9 classes)  
- **OCTMNIST**: Retinal OCT (Multi-Class, 4 classes)
- **PneumoniaMNIST**: Chest X-Ray (Binary, 2 classes)
- **DermaMNIST**: Dermatoscope (Multi-Class, 7 classes)
- **BloodMNIST**: Blood Cell Microscope (Multi-Class, 8 classes)
- **TissueMNIST**: Kidney Cortex Microscope (Multi-Class, 8 classes)
- **RetinaMNIST**: Fundus Camera (Ordinal Regression, 5 classes)
- **BreastMNIST**: Breast Ultrasound (Binary, 2 classes)
- **OrganAMNIST**: Abdominal CT (Multi-Class, 11 classes)
- **OrganCMNIST**: Abdominal CT (Multi-Class, 11 classes)
- **OrganSMNIST**: Abdominal CT (Multi-Class, 11 classes)

## Quick Start

```bash
# Install dependencies with uv
uv sync

# Train Base VAE on ChestMNIST
uv run train experiment=chest_base_vae

# Train Beta-VAE on PathMNIST
uv run train experiment=path_beta_vae

# Train Conditional VAE on multiple modalities
uv run train experiment=multi_modal_cvae

# Generate samples
uv run generate --model_path=logs/checkpoints/best.ckpt --num_samples=64

# Evaluate model
uv run evaluate --model_path=logs/checkpoints/best.ckpt --config_path=configs/data/chest_xray.yaml
```

## Architecture

The model uses a ResNet-style encoder-decoder architecture with:

- Attention mechanisms at multiple resolutions
- Progressive downsampling/upsampling
- Skip connections for better reconstruction
- Modality-specific conditioning layers

## Configuration

All experiments are managed through Hydra configs in `configs/`. Key configuration groups:

- `model/`: VAE architecture variants
- `data/`: Dataset configurations
- `training/`: Training hyperparameters
- `experiment/`: Complete experiment setups

## Research Techniques

Inspired by Stanford MedVAE, this implementation includes:

- Advanced perceptual losses (LPIPS)
- BiomedCLIP-inspired feature matching
- Adversarial training components
- Multi-scale attention mechanisms
- Progressive training strategies

<!-- ## Citation

```bibtex
@software{medmnist_conditional_vae,
  title={MedMNIST Conditional VAE},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/medmnist-conditional-vae}
}
``` -->

## Acknowledgments

- [Stanford MedVAE](https://github.com/StanfordMIMI/MedVAE) for architectural inspiration
- [MedMNIST](https://medmnist.com/) for the standardized medical imaging datasets
- [Hydra](https://hydra.cc/) for configuration management
