import torch
import numpy as np
from torch.utils.data import Dataset


class DataSigns(Dataset):
    def __init__(self):
        data = np.loadtxt('/home/NISE/DataSigns.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.labels = data[:, 0]
        data_img = np.squeeze(data[np.where(self.labels<6), 1:])
        self.len = data_img.shape[0]
        self.labels = torch.from_numpy(self.labels[np.where(self.labels<6)])
        
        min_max = np.amax(data_img, axis=1) - np.amin(data_img, axis=1)
        images_rows = data_img - np.amin(data_img, axis=1).reshape(-1, 1)
        images_rows = images_rows/min_max.reshape(-1, 1)
        img_size = int(np.sqrt(images_rows.shape[1]))
        self.images = torch.from_numpy(images_rows.reshape((self.len, img_size, img_size)))
#         self.labels[np.where(self.labels>9)] -= 1
        
    def __getitem__(self, idx):
        return self.images[idx, :, :], self.labels[idx]

    def __len__(self):
        return self.len
    
    