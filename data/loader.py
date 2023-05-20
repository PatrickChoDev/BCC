import os
from pydicom import dcmread
import pandas as pd
import torch
from torchvision.transforms import Resize, Compose, Normalize, CenterCrop


def dataset_collate(label: pd.Series, root_path: str):
    dirname = os.path.join(root_path, ("benign" if label['FinalSubjectId'].startswith(
        "N") else "cancer"), label['FinalSubjectId'])
    composer = Compose([Resize(224), CenterCrop(224), Normalize(0.5, 0.5)])
    data = []
    sides = ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']
    for annotate in sides:
        dcm = torch.Tensor(dcmread(os.path.join(
            dirname, "{}-{}.dcm".format(label['subjectId'], annotate))).pixel_array).unsqueeze(0)
        data.append(composer(dcm))
    label = torch.Tensor([0 if label['FinalSubjectId'].startswith('N') else 1])
    return torch.stack(data).squeeze(1), label
