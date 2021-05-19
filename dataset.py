import os

import numpy as np
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform

        self.path_to_train_files = os.path.join(root_dir, 'train')
        self.path_to_val_files = os.path.join(root_dir, 'val')

        train_path, _, train_files = list(os.walk(self.path_to_train_files))[0]
        train_files = [os.path.join(train_path, file) for file in train_files]
        val_path, _, val_files = list(os.walk(self.path_to_val_files))[0]
        val_files = [os.path.join(val_path, file) for file in val_files]

        self.files = train_files + val_files
        self.files = self.files[:-16] if is_train else self.files[-16:]

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = np.array(img)

        x, y = np.split(img, 2, axis=1)

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        return y, x

    def __len__(self):
        return len(self.files)
