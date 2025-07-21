"""
MedMNIST data module and dataset classes.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any
import lightning as L
from torchvision import transforms
import medmnist
from medmnist import INFO


class MedMNISTDataset(Dataset):
    """MedMNIST dataset wrapper."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        size: int = 224,
        as_rgb: bool = True,
        root: str = "./data",
    ):
        """
        Initialize MedMNIST dataset.

        Args:
            dataset_name: Name of MedMNIST dataset (e.g., 'chestmnist', 'pathmnist')
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply
            download: Whether to download if not present
            size: Image size (28, 64, 128, or 224)
            as_rgb: Convert grayscale to RGB
            root: Root directory for data
        """
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.size = size
        self.as_rgb = as_rgb
        self.root = root

        # Get dataset info
        if self.dataset_name not in INFO:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.info = INFO[self.dataset_name]
        self.n_channels = self.info["n_channels"]
        self.n_classes = len(self.info["label"])  # Calculate from label dictionary
        self.task = self.info["task"]

        # Get dataset class
        DataClass = getattr(medmnist, self.info["python_class"])

        # Load dataset
        self.dataset = DataClass(
            split=split,
            transform=None,  # We'll handle transforms separately
            download=download,
            size=size,
            root=root,
        )

        self.transform = transform

        # Create modality mapping for conditioning
        self.modality_map = self._create_modality_map()
        self.modality_idx = self.modality_map[self.dataset_name]

    def _create_modality_map(self) -> Dict[str, int]:
        """Create mapping from dataset names to modality indices."""
        modalities = [
            "chestmnist",  # Chest X-Ray
            "pathmnist",  # Colon Pathology
            "octmnist",  # Retinal OCT
            "pneumoniamnist",  # Chest X-Ray (Pneumonia)
            "dermamnist",  # Dermatoscope
            "bloodmnist",  # Blood Cell Microscope
            "tissuemnist",  # Kidney Cortex Microscope
            "retinamnist",  # Fundus Camera
            "breastmnist",  # Breast Ultrasound
            "organamnist",  # Abdominal CT (Axial)
            "organcmnist",  # Abdominal CT (Coronal)
            "organsmnist",  # Abdominal CT (Sagittal)
        ]
        return {name: idx for idx, name in enumerate(modalities)}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.

        Returns:
            image: Preprocessed image tensor (always same channel count)
            label: Standardized label tensor (always 1D)
            modality: One-hot encoded modality vector
        """
        image, label = self.dataset[idx]

        # Convert PIL image to tensor if needed
        if hasattr(image, "mode"):
            image = transforms.ToTensor()(image)

        # Handle channel normalization for multi-modal training
        # Force all images to have consistent channels based on as_rgb setting
        if self.as_rgb:
            # Convert everything to RGB (3 channels)
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            # If already 3 channels, keep as is
        else:
            # Convert everything to grayscale (1 channel)
            if image.shape[0] == 3:
                # Convert RGB to grayscale using standard weights
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
                image = gray.unsqueeze(0)
            # If already 1 channel, keep as is

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Standardize label format for multi-modal training
        # Convert to tensor and ensure consistent shape
        label_tensor = torch.tensor(label).long()

        # Handle different label formats across datasets
        if label_tensor.dim() == 0:
            # Scalar label -> make it 1D
            label_tensor = label_tensor.unsqueeze(0)
        elif label_tensor.dim() > 1:
            # Multi-dimensional label -> flatten to 1D
            label_tensor = label_tensor.flatten()

        # For conditional VAE, we mainly need modality info, so we can use a dummy label
        # or convert multi-labels to a single representative value
        if label_tensor.shape[0] > 1:
            # For multi-label, take the first positive label or sum all labels
            if label_tensor.sum() > 0:
                label_tensor = torch.tensor([label_tensor.argmax().item()]).long()
            else:
                label_tensor = torch.tensor([0]).long()

        # Create modality one-hot vector
        modality = torch.zeros(len(self.modality_map))
        modality[self.modality_idx] = 1.0

        return image, label_tensor, modality


class MedMNISTDataModule(L.LightningDataModule):
    """Lightning data module for MedMNIST."""

    def __init__(
        self,
        dataset_names: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        size: int = 224,
        as_rgb: bool = True,
        root: str = "./data",
        normalize: bool = True,
        augment_train: bool = True,
        **kwargs,
    ):
        """
        Initialize MedMNIST data module.

        Args:
            dataset_names: List of dataset names to load
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            size: Image size
            as_rgb: Convert to RGB
            root: Data root directory
            normalize: Whether to normalize images to [-1, 1]
            augment_train: Whether to apply augmentations to training data
        """
        super().__init__()
        self.dataset_names = [name.lower() for name in dataset_names]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.as_rgb = as_rgb
        self.root = root
        self.normalize = normalize
        self.augment_train = augment_train

        # Setup transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup data transforms."""
        # Base transforms
        base_transforms = []

        if self.size != 28:
            base_transforms.append(transforms.Resize((self.size, self.size)))

        # Training augmentations
        train_transforms = base_transforms.copy()
        if self.augment_train:
            train_transforms.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ]
            )

        # Normalization
        if self.normalize:
            # Normalize to [-1, 1] for VAE training
            if self.as_rgb:
                # RGB normalization
                train_transforms.append(
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                )
                base_transforms.append(
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                )
            else:
                # Grayscale normalization
                train_transforms.append(transforms.Normalize(mean=[0.5], std=[0.5]))
                base_transforms.append(transforms.Normalize(mean=[0.5], std=[0.5]))

        self.train_transform = transforms.Compose(train_transforms)
        self.val_transform = transforms.Compose(base_transforms)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            # Combine all datasets for training
            train_datasets = []
            val_datasets = []

            for dataset_name in self.dataset_names:
                train_dataset = MedMNISTDataset(
                    dataset_name=dataset_name,
                    split="train",
                    transform=self.train_transform,
                    size=self.size,
                    as_rgb=self.as_rgb,
                    root=self.root,
                )
                train_datasets.append(train_dataset)

                val_dataset = MedMNISTDataset(
                    dataset_name=dataset_name,
                    split="val",
                    transform=self.val_transform,
                    size=self.size,
                    as_rgb=self.as_rgb,
                    root=self.root,
                )
                val_datasets.append(val_dataset)

            # Concatenate datasets
            self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            self.val_dataset = torch.utils.data.ConcatDataset(val_datasets)

        if stage == "test" or stage is None:
            test_datasets = []
            for dataset_name in self.dataset_names:
                test_dataset = MedMNISTDataset(
                    dataset_name=dataset_name,
                    split="test",
                    transform=self.val_transform,
                    size=self.size,
                    as_rgb=self.as_rgb,
                    root=self.root,
                )
                test_datasets.append(test_dataset)

            self.test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about loaded datasets."""
        info = {}
        for dataset_name in self.dataset_names:
            if dataset_name in INFO:
                info[dataset_name] = INFO[dataset_name]
        return info
