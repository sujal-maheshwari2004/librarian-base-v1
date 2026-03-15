import numpy as np
import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):

    def __init__(self, path, seq_len):

        self.seq_len = seq_len

        self.data = np.memmap(
            path,
            dtype=np.uint16,
            mode="r"
        )

        # FIXED: prevent overflow
        self.num_sequences = len(self.data) // seq_len - 1

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):

        start = idx * self.seq_len
        end = start + self.seq_len + 1

        tokens = self.data[start:end]

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y
