from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class LFWPairsDataset(Dataset):
    def __init__(self, lfw_pairs, transform=None):
        self.pairs = lfw_pairs.pairs
        self.labels = lfw_pairs.target
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_array = self.pairs[idx, 0]
        img2_array = self.pairs[idx, 1]

        # Ensure arrays are in the correct shape and dtype
        if img1_array.dtype != np.uint8:
            img1_array = img1_array.astype(np.uint8)
        if img2_array.dtype != np.uint8:
            img2_array = img2_array.astype(np.uint8)

        # If the shape is not (H, W, 3) for color images, adjust it
        if len(img1_array.shape) == 2:  # Grayscale image case
            img1_array = np.stack([img1_array] * 3, axis=-1)
        if len(img2_array.shape) == 2:
            img2_array = np.stack([img2_array] * 3, axis=-1)

        # Convert numpy arrays to PIL Images
        img1 = Image.fromarray(img1_array)
        img2 = Image.fromarray(img2_array)
        label = self.labels[idx]

        # Apply transformations if any
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label