import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class MemmapDataset(Dataset):
    def __init__(self, data_file, block_size, stride=1):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Binary tensor stream '{data_file}' not found. Please run prepare_data.py first to tokenize the datasets.")
            
        # Advanced NLP dataloading: memory map allows us to stream infinite GBs of tokens 
        # without overflowing RAM, jumping to precise bytes instantly.
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.stride = stride
        total_samples = (len(self.data) - self.block_size - 1) // self.stride
        print(f"Loaded high-performance binary NLP stream: {data_file} ({len(self.data):,} tokens, {total_samples:,} samples with stride={stride})")

    def __len__(self):
        return (len(self.data) - self.block_size - 1) // self.stride

    def __getitem__(self, idx):
        # Map strided index back to actual token position
        actual_idx = idx * self.stride
        chunk = self.data[actual_idx : actual_idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

def get_dataloader(data_file, block_size, batch_size, stride=1, shuffle=True):
    dataset = MemmapDataset(data_file, block_size, stride=stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)
