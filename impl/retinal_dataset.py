import os
from pathlib import Path

from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = "data/retinal-disease-classification"

class RetinalDataset(Dataset):
    def __init__(self, split="train", transform=None, num_rows=-1):
        self.transform = transform
        self.num_rows = num_rows

        paths = self._split_paths(split)

        root_dir = Path(ROOT_DIR)
        split_dir = root_dir / paths[0]

        self.images_dir = split_dir / paths[1]

        csv_file_path = split_dir / paths[2]
        self.labels_df = pd.read_csv(csv_file_path)

        self._load_split()

    def _split_paths(self, split: str):
        if split == "train":
            return (
                "Training_Set/Training_Set",
                "Training",
                "RFMiD_Training_Labels.csv"
            )

    def _load_split(self):
        images = []
        labels = []

        n = self.num_rows
        if n == -1:
            n = len(self.labels_df)

        for i, row in tqdm(self.labels_df.iterrows(), total=n):
            if i >= n:
                break

            # img_id = self.labels_df.loc[idx, "ID"]
            # label = self.labels_df.loc[idx, "Disease_Risk"]
            img_id = row["ID"]
            label = row["Disease_Risk"]
            
            img_path = str(self.images_dir / f"{img_id}.png")
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)

            images.append(image)
            labels.append(label)

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.images[idx], self.labels[idx]

        