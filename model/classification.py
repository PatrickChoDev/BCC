import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import CheckPointed, Layer, BCCModel


class MetaClassifier(BCCModel):
    def __init__(self) -> None:
        super(MetaClassifier,self).__init__()
        self.input_size = 4
        self.output_size = 1
        self.linear1 = nn.Sequential(CheckPointed(
            Layer(nn.Linear(512, 24),nn.ReLU(), nn.Linear(24, 1))))

    @property
    def name(self):
        return f"meta-{self.input_size}-{self.output_size}"

    def forward(self, x):
        return self.linear1(x)
