import os
import requests
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TextPreprocessor:
    def __init__(self, input_path, data_dir, url=None):
        self.input_path = input_path
        self.data_dir = data_dir
        self.url = url or 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        self.meta_path = os.path.join(data_dir, 'meta.pkl')
        self.train_bin = os.path.join(data_dir, 'train.bin')
        self.val_bin = os.path.join(data_dir, 'val.bin')

    def download(self):
        dir_path = os.path.dirname(self.input_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(self.input_path):
            with open(self.input_path, 'w') as f:
                f.write(requests.get(self.url).text)

    def preprocess(self, val_split=0.1):
        with open(self.input_path, 'r') as f:
            data = f.read()
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        char_to_id = {ch: i for i, ch in enumerate(chars)}
        id_to_char = {i: ch for i, ch in enumerate(chars)}
        n = len(data)
        train_data = data[:int(n * (1 - val_split))]
        val_data = data[int(n * (1 - val_split)) :]
        train_ids = np.array([char_to_id[c] for c in train_data], dtype=np.uint16)
        val_ids = np.array([char_to_id[c] for c in val_data], dtype=np.uint16)
        os.makedirs(self.data_dir, exist_ok=True)
        train_ids.tofile(self.train_bin)
        val_ids.tofile(self.val_bin)
        meta = {'vocab_size': vocab_size, 'itos': id_to_char, 'stoi': char_to_id}
        with open(self.meta_path, 'wb') as f:
            pickle.dump(meta, f)
        return meta

    def load_meta(self):
        with open(self.meta_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def encode(s, stoi):
        return [stoi[c] for c in s]

    @staticmethod
    def decode(l, itos):
        return ''.join([itos[i] for i in l])

class CharDataset(Dataset):
    def __init__(self, bin_path, block_size):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.block_size].copy()).long()
        y = torch.from_numpy(self.data[idx+1:idx+self.block_size+1].copy()).long()
        return x, y

class CharDataModule:
    def __init__(self, data_dir, batch_size, block_size):
        self.train_bin = os.path.join(data_dir, 'train.bin')
        self.val_bin = os.path.join(data_dir, 'val.bin')
        self.batch_size = batch_size
        self.block_size = block_size
    def train_dataloader(self):
        return DataLoader(CharDataset(self.train_bin, self.block_size), batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(CharDataset(self.val_bin, self.block_size), batch_size=self.batch_size, shuffle=False) 