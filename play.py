"""
import os
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = "data/retinal-disease-classification"

class RetinalDataset(Dataset):
    def __init__(self, split="train", transform=None):
        paths = self._split_paths(split)

        root_dir = Path(ROOT_DIR)
        split_dir = root_dir / paths[0]
        self._images_dir = split_dir / paths[1]

        csv_file_path = split_dir / paths[2]

        self.labels_df = pd.read_csv(csv_file_path)
        self.transform = transform

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
        
        # Get image ID and label
        img_id = self.labels_df.iloc[idx, "ID"]
        label = self.labels_df.iloc[idx, "Disease_Risk"]
        
        # Load image
        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = RetinalDataset(transform=transform)
# train_dataloader = DataLoader(custom_dataset, batch_size=16, shuffle=True)

"""
