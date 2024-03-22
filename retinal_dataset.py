import os
from pathlib import Path

from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = "data/retinal-disease-classification"

class RetinalDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.transform = transform
        paths = self._split_paths(split)

        root_dir = Path(ROOT_DIR)
        split_dir = root_dir / paths[0]

        self.images_dir = split_dir / paths[1]

        csv_file_path = split_dir / paths[2]
        self.labels_df = pd.read_csv(csv_file_path)

    def _split_paths(self, split: str):
        if split == "train":
            return (
                "Training_Set/Training_Set",
                "Training",
                "RFMiD_Training_Labels.csv"
            )


    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.labels_df.loc[idx, "ID"]
        label = self.labels_df.loc[idx, "Disease_Risk"]
        
        img_path = str(self.images_dir / f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, label