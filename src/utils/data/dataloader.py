import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import h5py

def get_loaders(dataset, batch_size, split_ratios, seed=42):
    # Preload all labels once from h5 file (assuming dataset.data_dir exists)
    with h5py.File(dataset.data_dir, 'r') as f:
        all_labels = f['ans'][:]
    all_labels_tensor = torch.from_numpy(all_labels).long()
    
    num_samples = len(dataset)
    train_len = int(num_samples * split_ratios[0])
    val_len = int(num_samples * split_ratios[1])
    test_len = num_samples - train_len - val_len
    
    # Generate random shuffle indices for splitting
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]
    
    # Get labels for training subset only
    train_labels = all_labels_tensor[train_indices]
    
    # Calculate class counts and weights for WeightedRandomSampler
    unique_classes = torch.unique(train_labels)
    class_sample_count = torch.tensor([(train_labels == t).sum() for t in unique_classes], dtype=torch.float)
    class_weights = 1. / class_sample_count
    # Create weight for each sample
    samples_weight = torch.tensor([class_weights[torch.where(unique_classes == label)[0][0]] for label in train_labels], dtype=torch.double)
    
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
    
    # Create subsets from original dataset using indices
    train_set = Subset(dataset, train_indices.tolist())
    val_set = Subset(dataset, val_indices.tolist())
    test_set = Subset(dataset, test_indices.tolist())
    
    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=15, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
    
    return train_loader, val_loader, test_loader





