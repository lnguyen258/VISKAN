import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch.nn.functional as F


class Galaxy10_SDSS_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        with h5py.File(self.data_dir, 'r') as f:
            self.length = f['images'].shape[0]
        

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        with h5py.File( self.data_dir, 'r') as f:
            data = np.array(f['images'][idx])
            label = np.array(f['ans'][idx])

        if self.transform:
            data = self.transform(data)
        else:
            data = torch.from_numpy(data)

        label = torch.from_numpy(label).long()
        num_classes=10
        label = F.one_hot(label, num_classes=num_classes).squeeze()

        return data, label



