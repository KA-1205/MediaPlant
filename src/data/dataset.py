import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class MediPlantDataset(Dataset):
    """
    Expects a CSV with columns: [image_path, label]
    image_path: full or relative path to image file
    label: integer class index
    """

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label
