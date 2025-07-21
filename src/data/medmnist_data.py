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


def mixed_modality_collate_fn(batch):
    """
    Custom collate function that handles mixed channel dimensions.
    
    Since the DisentangledConditionalVAE model processes each sample individually
    with its own input projector, we need to handle mixed channel batches carefully.
    We'll create a batch where each sample keeps its original channel count,
    and let the model handle the projection.
    """
    # Separate the batch components
    images, labels, modalities, modality_indices = zip(*batch)
    
    # For mixed channel batches, we'll store images as a list of tensors
    # and let the model handle them individually in its encode method
    # 
    # But PyTorch DataLoader expects tensor outputs, so we need to create
    # same-sized tensors. We'll group by channel count.
    
    # Group by channel count
    channel_groups = {}
    for i, img in enumerate(images):
        channels = img.shape[0]
        if channels not in channel_groups:
            channel_groups[channels] = []
        channel_groups[channels].append(i)
    
    # If all images have the same channel count, use normal collation
    if len(channel_groups) == 1:
        images_tensor = torch.stack(images)
        labels_tensor = torch.stack(labels)
        modalities_tensor = torch.stack(modalities) 
        modality_indices_tensor = torch.stack(modality_indices)
        return images_tensor, labels_tensor, modalities_tensor, modality_indices_tensor
    
    # For mixed channels, we need to pad to the same size for PyTorch batching
    # The model will handle the channel projection internally
    max_channels = max(img.shape[0] for img in images)
    
    padded_images = []
    for img in images:
        if img.shape[0] < max_channels:
            # Pad with zeros to make tensors the same size
            # The model's input projectors will handle the actual channel processing
            padding_shape = (max_channels - img.shape[0], *img.shape[1:])
            padding = torch.zeros(padding_shape, dtype=img.dtype)
            padded_img = torch.cat([img, padding], dim=0)
        else:
            padded_img = img
        padded_images.append(padded_img)
    
    # Stack all components
    images_tensor = torch.stack(padded_images)
    labels_tensor = torch.stack(labels)
    modalities_tensor = torch.stack(modalities)
    modality_indices_tensor = torch.stack(modality_indices)
    
    return images_tensor, labels_tensor, modalities_tensor, modality_indices_tensor


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
        
        # Determine target channels for this modality
        self.target_channels = self._get_modality_channels()

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
    
    def _get_modality_channels(self) -> int:
        """Get the natural number of channels for this modality."""
        # Define natural channel counts for each modality type
        grayscale_modalities = {
            "chestmnist",      # X-Ray should stay grayscale
            "pneumoniamnist",  # X-Ray should stay grayscale
            "organamnist",     # CT scans are grayscale
            "organcmnist",     # CT scans are grayscale 
            "organsmnist",     # CT scans are grayscale
        }
        
        rgb_modalities = {
            "pathmnist",       # Pathology images are naturally color
            "dermamnist",      # Dermatoscope images are naturally color
            "retinamnist",     # Fundus camera images are naturally color
            "bloodmnist",      # Blood microscopy can be color
            "tissuemnist",     # Tissue microscopy can be color
            "octmnist",        # OCT images can be color/pseudo-color
            "breastmnist",     # Ultrasound can be pseudo-color
        }
        
        if self.dataset_name in grayscale_modalities:
            return 1
        elif self.dataset_name in rgb_modalities:
            return 3
        else:
            # Default to the dataset's natural channel count
            return self.n_channels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.

        Returns:
            image: Preprocessed image tensor (modality-specific channels)
            label: Standardized label tensor (always 1D)
            modality: One-hot encoded modality vector
            modality_idx: Modality index as tensor
        """
        image, label = self.dataset[idx]

        # Convert PIL image to tensor if needed
        if hasattr(image, "mode"):
            image = transforms.ToTensor()(image)

        # Handle channel conversion based on modality requirements
        current_channels = image.shape[0]
        target_channels = self.target_channels

        if target_channels == 1:  # Convert to grayscale
            if current_channels == 3:
                # Convert RGB to grayscale using standard weights
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
                image = gray.unsqueeze(0)
            # If already 1 channel, keep as is
        elif target_channels == 3:  # Convert to RGB
            if current_channels == 1:
                image = image.repeat(3, 1, 1)
            # If already 3 channels, keep as is

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
        
        # Create modality index tensor
        modality_idx_tensor = torch.tensor(self.modality_idx).long()

        return image, label_tensor, modality, modality_idx_tensor


class MedMNISTDataModule(L.LightningDataModule):
    """Lightning data module for MedMNIST."""

    def __init__(
        self,
        dataset_names: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        size: int = 224,
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
            root: Data root directory
            normalize: Whether to normalize images to [-1, 1]
            augment_train: Whether to apply augmentations to training data
        """
        super().__init__()
        self.dataset_names = [name.lower() for name in dataset_names]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.root = root
        self.normalize = normalize
        self.augment_train = augment_train

        # Setup transforms
        self._setup_transforms()
        
        # Store modality channel information
        self._setup_modality_info()
    
    def _setup_modality_info(self):
        """Setup modality information including channel counts."""
        # Create a dummy dataset instance to get modality info
        dummy_dataset = MedMNISTDataset(
            dataset_name=self.dataset_names[0],
            split="train",
            transform=None,
            size=self.size,
            root=self.root,
        )
        
        # Get modality info
        self.modality_map = dummy_dataset.modality_map
        self.num_modalities = len(self.modality_map)
        
        # Get channel info for each modality
        self.modality_channels = {}
        for dataset_name in self.dataset_names:
            print(f"Loading modality info for {dataset_name}...")
            temp_dataset = MedMNISTDataset(
                dataset_name=dataset_name,
                split="train", 
                transform=None,
                size=self.size,
                root=self.root,
            )
            self.modality_channels[dataset_name] = temp_dataset.target_channels
            print(f"  {dataset_name}: {temp_dataset.target_channels} channels")

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

        # Note: Normalization will be applied per-modality since different modalities 
        # have different channel counts. We'll handle this in the dataset __getitem__ method.
        
        self.train_transform_base = transforms.Compose(train_transforms)
        self.val_transform_base = transforms.Compose(base_transforms)
        
        # We'll create modality-specific normalizations in _get_modality_transform
    
    def _get_modality_transform(self, dataset_name: str, train: bool = True) -> transforms.Compose:
        """Get modality-specific transform including normalization."""
        # Get channel count for this modality
        temp_dataset = MedMNISTDataset(
            dataset_name=dataset_name,
            split="train",
            transform=None,
            size=self.size,
            root=self.root,
        )
        channels = temp_dataset.target_channels
        
        # Start with base transforms
        if train:
            transform_list = list(self.train_transform_base.transforms)
        else:
            transform_list = list(self.val_transform_base.transforms)
        
        # Add normalization based on channel count
        if self.normalize:
            if channels == 1:
                # Grayscale normalization
                transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
            elif channels == 3:
                # RGB normalization  
                transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
        return transforms.Compose(transform_list)

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
                    transform=self._get_modality_transform(dataset_name, train=True),
                    size=self.size,
                    root=self.root,
                )
                train_datasets.append(train_dataset)

                val_dataset = MedMNISTDataset(
                    dataset_name=dataset_name,
                    split="val",
                    transform=self._get_modality_transform(dataset_name, train=False),
                    size=self.size,
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
                    transform=self._get_modality_transform(dataset_name, train=False),
                    size=self.size,
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
            collate_fn=mixed_modality_collate_fn,
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
            collate_fn=mixed_modality_collate_fn,
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
            collate_fn=mixed_modality_collate_fn,
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about loaded datasets."""
        info = {}
        for dataset_name in self.dataset_names:
            if dataset_name in INFO:
                info[dataset_name] = INFO[dataset_name]
        return info
