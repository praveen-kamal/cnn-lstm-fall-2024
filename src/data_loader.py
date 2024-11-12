# data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class AWIDDataset(Dataset):
    def __init__(self, data_path, labels_path):
        # Load preprocessed data and labels
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to PyTorch tensors and return
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def get_dataloaders(batch_size, data_dir):
    # Paths to data and labels
    train_data_path = os.path.join(data_dir, "train_data.npy")
    train_labels_path = os.path.join(data_dir, "train_labels.npy")
    test_data_path = os.path.join(data_dir, "test_data.npy")
    test_labels_path = os.path.join(data_dir, "test_labels.npy")

    # Initialize datasets and dataloaders
    train_dataset = AWIDDataset(train_data_path, train_labels_path)
    test_dataset = AWIDDataset(test_data_path, test_labels_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
