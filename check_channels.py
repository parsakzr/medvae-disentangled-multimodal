#!/usr/bin/env python3
"""
Check MedMNIST dataset channel information.
"""

import medmnist
from medmnist import INFO


def check_dataset_channels():
    """Check which datasets are naturally grayscale vs RGB."""
    grayscale_datasets = []
    rgb_datasets = []

    for dataset_name, info in INFO.items():
        n_channels = info["n_channels"]
        if n_channels == 1:
            grayscale_datasets.append(dataset_name)
        elif n_channels == 3:
            rgb_datasets.append(dataset_name)
        else:
            print(f"Unusual: {dataset_name} has {n_channels} channels")

    print("GRAYSCALE DATASETS (1 channel):")
    for name in sorted(grayscale_datasets):
        print(f"  - {name}: {INFO[name]['label']}")

    print("\nRGB DATASETS (3 channels):")
    for name in sorted(rgb_datasets):
        print(f"  - {name}: {INFO[name]['label']}")


if __name__ == "__main__":
    check_dataset_channels()
