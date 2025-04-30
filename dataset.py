import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np


class CaptchaDataset(Dataset):
    def __init__(self, image_dir, label_csv, char_to_idx, img_width=128, img_height=64):
        self.image_dir = image_dir
        self.data = pd.read_csv(label_csv)
        self.char_to_idx = char_to_idx
        self.img_width = img_width
        self.img_height = img_height

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        label = [self.char_to_idx[c] for c in row["label"]]
        return torch.FloatTensor(img), torch.LongTensor(label)
