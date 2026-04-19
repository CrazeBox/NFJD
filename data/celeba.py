from __future__ import annotations

import os
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CelebA


def make_celeba(
    num_clients: int,
    root: str = "data/celeba",
    download: bool = True,
    iid: bool = True,
    num_tasks: int = 4,
    val_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    """
    Create federated CelebA dataset.
    
    Args:
        num_clients: Number of clients
        root: Root directory for dataset
        download: Whether to download the dataset
        iid: Whether to use IID split
        num_tasks: Number of tasks (attributes) to use
        val_ratio: Validation set ratio
        seed: Random seed
    
    Returns:
        Tuple of (train_datasets, val_datasets, test_datasets)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Load CelebA dataset
    train_dataset = CelebA(
        root=root,
        split="train",
        transform=transform,
        download=download,
    )
    
    val_dataset = CelebA(
        root=root,
        split="valid",
        transform=transform,
        download=download,
    )
    
    test_dataset = CelebA(
        root=root,
        split="test",
        transform=transform,
        download=download,
    )
    
    # Get attributes
    attributes = train_dataset.attr_names
    selected_attributes = attributes[:num_tasks]
    print(f"Selected attributes: {selected_attributes}")
    
    # Create federated splits
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    if iid:
        # IID split
        for i in range(num_clients):
            # Create client-specific datasets
            client_train = CelebAClientDataset(train_dataset, i, num_clients, selected_attributes)
            client_val = CelebAClientDataset(val_dataset, i, num_clients, selected_attributes)
            client_test = CelebAClientDataset(test_dataset, i, num_clients, selected_attributes)
            
            train_datasets.append(client_train)
            val_datasets.append(client_val)
            test_datasets.append(client_test)
    else:
        # Non-IID split (by attributes)
        for i in range(num_clients):
            # Create client-specific datasets with attribute-based split
            client_train = CelebANonIIDDataset(train_dataset, i, num_clients, selected_attributes)
            client_val = CelebANonIIDDataset(val_dataset, i, num_clients, selected_attributes)
            client_test = CelebANonIIDDataset(test_dataset, i, num_clients, selected_attributes)
            
            train_datasets.append(client_train)
            val_datasets.append(client_val)
            test_datasets.append(client_test)
    
    return train_datasets, val_datasets, test_datasets


class CelebAClientDataset(Dataset):
    """CelebA dataset for a specific client (IID split)."""
    
    def __init__(self, base_dataset: CelebA, client_id: int, num_clients: int, selected_attributes: List[str]):
        self.base_dataset = base_dataset
        self.client_id = client_id
        self.num_clients = num_clients
        self.selected_attributes = selected_attributes
        self.attribute_indices = [self.base_dataset.attr_names.index(attr) for attr in selected_attributes]
        
        # Create client-specific indices
        total_len = len(base_dataset)
        self.indices = list(range(client_id, total_len, num_clients))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        img, attrs = self.base_dataset[data_idx]
        # Extract selected attributes
        selected_attrs = attrs[self.attribute_indices]
        # Convert attributes to float
        selected_attrs = selected_attrs.float()
        return img, selected_attrs


class CelebANonIIDDataset(Dataset):
    """CelebA dataset for a specific client (Non-IID split by attributes)."""
    
    def __init__(self, base_dataset: CelebA, client_id: int, num_clients: int, selected_attributes: List[str]):
        self.base_dataset = base_dataset
        self.client_id = client_id
        self.num_clients = num_clients
        self.selected_attributes = selected_attributes
        self.attribute_indices = [self.base_dataset.attr_names.index(attr) for attr in selected_attributes]
        
        # Create non-IID split based on attributes
        self.indices = []
        for i in range(len(base_dataset)):
            _, attrs = base_dataset[i]
            # Use the first selected attribute for splitting
            attr_value = attrs[self.attribute_indices[0]].item()
            if attr_value % num_clients == client_id:
                self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        img, attrs = self.base_dataset[data_idx]
        # Extract selected attributes
        selected_attrs = attrs[self.attribute_indices]
        # Convert attributes to float
        selected_attrs = selected_attrs.float()
        return img, selected_attrs