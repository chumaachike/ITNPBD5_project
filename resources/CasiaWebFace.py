import numpy as np
np.bool = bool
from torch.utils.data import Dataset
import mxnet as mx
from PIL import Image

class CASIAWebFaceDataset(Dataset):
    def __init__(self, rec_file, idx_file, transform=None):
        self.rec_file = rec_file
        self.idx_file = idx_file
        self.transform = transform
        self.recordio = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')
        self.header, _ = mx.recordio.unpack(self.recordio.read_idx(0))

        # Preprocess to filter out invalid records
        self.valid_indices = self._filter_valid_indices()

        # Compute unique labels
        self.unique_labels = self._get_unique_labels()

        # Store expected labels for valid indices (optional)
        self.expected_labels = self._map_indices_to_labels()

    def _filter_valid_indices(self):
        valid_indices = []
        for idx in range(int(self.header.label[0])):  # Iterate through all indices
            record = self.recordio.read_idx(idx + 1)
            header, img = mx.recordio.unpack(record)
            if img is not None and len(img) > 0:
                valid_indices.append(idx)
        return valid_indices

    def _get_unique_labels(self):
        labels = set()
        for idx in self.valid_indices:
            record = self.recordio.read_idx(idx + 1)
            header, _ = mx.recordio.unpack(record)
            labels.add(int(header.label))
        return labels

    def _map_indices_to_labels(self):
        """Map each valid index to its corresponding label"""
        index_to_label = {}
        for idx in self.valid_indices:
            record = self.recordio.read_idx(idx + 1)
            header, _ = mx.recordio.unpack(record)
            index_to_label[idx] = int(header.label)
        return index_to_label

    def _is_valid_label_for_index(self, index, label):
        """Check if the label for a given index is correct"""
        expected_label = self.expected_labels.get(index)
        return expected_label == label

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        record = self.recordio.read_idx(valid_idx + 1)
        header, img = mx.recordio.unpack(record)

        img = mx.image.imdecode(img).asnumpy()

        # Convert to PIL Image
        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        label = int(header.label)

        # Sanity check: Ensure label corresponds to the correct image
        if not self._is_valid_label_for_index(valid_idx, label):
            raise ValueError(f"Label mismatch at index {valid_idx}")

        return img, label

    def get_num_unique_labels(self):
        return len(self.unique_labels)
