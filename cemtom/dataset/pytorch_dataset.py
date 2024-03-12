import torch
from cemtom._base import get_data
from cemtom.preprocessing import Preprocessor
from cemtom.dataset import fetch_dataset, Dictionary

from torch.utils.data import DataLoader, Dataset as PyDataset


class DatasetBoW(PyDataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


