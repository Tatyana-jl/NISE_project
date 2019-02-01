import torch
import numpy as np
from torch.utils.data import Dataset


class DataSigns(Dataset):
    def __init__(self):
        data = np.loadtxt('/home/NISE/DataSigns.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.len = data.shape[0]
        min_max = np.amax(data[:, 1:], axis=1) - np.amin(data[:, 1:], axis=1)
        images_rows = data[:, 1:] - np.amin(data[:, 1:], axis=1).reshape(-1, 1)
        images_rows = images_rows/min_max.reshape(-1, 1)
        img_size = int(np.sqrt(images_rows.shape[1]))
        self.images = torch.from_numpy(images_rows.reshape((self.len, img_size, img_size)))
        self.labels = torch.from_numpy(data[:, 0])
        self.labels[np.where(self.labels>9)] -= 1
        
    def __getitem__(self, idx):
        return self.images[idx, :, :], self.labels[idx]

    def __len__(self):
        return self.len
    
    