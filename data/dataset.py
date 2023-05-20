import os
import pandas as pd
import torch
import logging
from torch.utils.data import Dataset
from typing import Callable, Tuple


class DCMDataset(Dataset):
    def __init__(self, dicom_path: str, label_path: str, collate_fn: Callable[[str, str], Tuple[torch.Tensor, pd.Series]]):
        super().__init__()
        self.dicom_path = os.path.join(dicom_path)
        self.label_path = label_path
        self.label = pd.read_csv(self.label_path)
        self.collate = collate_fn
        logging.info("Dataset loaded from: " + self.dicom_path)
        logging.info("Total Dataset length: " + str(len(self)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.collate(self.label.loc[index], self.dicom_path)

    def describe(self):
        return self.label.describe()
