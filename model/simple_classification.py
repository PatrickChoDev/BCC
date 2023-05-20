import torch
import torch.nn as nn
import torch.nn.functional as F

class DicomClassifier(nn.Module):
    def __init__(self,):
        super(DicomClassifier, self).__init__()
        self.lconv1 = nn.Conv2d(4, 8, 2)
        self.lbn1 = nn.BatchNorm2d(8)
        self.lconv2 = nn.Conv2d(8, 8, 2)
        self.lbn2 = nn.BatchNorm2d(8)
        self.lconv3 = nn.Conv2d(8, 8, 2)
        self.lbn3 = nn.BatchNorm2d(8)
        self.lflatten = nn.Flatten()
        self.lfc1 = nn.LazyLinear(128)
        self.lbnfc1 = nn.BatchNorm1d(128)
        self.lfc2 = nn.Linear(128, 32)
        self.lbnfc2 = nn.BatchNorm1d(32)
        self.ldropout = nn.Dropout(0.5)
        self.lhead = nn.Linear(32, 1)

    def forward(self,x):
        lconv1 = F.relu(self.lbn1(self.lconv1(x)))
        lconv2 = F.relu(self.lbn2(self.lconv2(lconv1)))
        lconv3 = F.relu(self.lbn3(self.lconv3(lconv2)))
        lflat = self.lflatten(lconv3)
        lfc1 = F.relu(self.lbnfc1(self.lfc1(lflat)))
        lfc2 = F.relu(self.lbnfc2(self.lfc2(lfc1)))
        lfc3 = F.relu(self.ldropout(lfc2))
        pre = self.lhead(lfc3)
        return pre