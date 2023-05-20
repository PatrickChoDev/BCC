from torch.utils.data import Dataset
import torch
from data.loader import DicomLoader, LabelLoader


class DCMDataset(Dataset):
    def __init__(self, dicom_loader: DicomLoader, label_loader: LabelLoader) -> None:
        super().__init__()
        self.__dicom_loader = dicom_loader
        self.__label_loader = label_loader
        if self.__dicom_loader.__len__() != self.__label_loader.__len__():
            raise Exception(
                'DicomLoader and LabelLoader must have the same length')

    def __len__(self):
        return self.__label_loader.__len__()

    def __getitem__(self, idx):
        return self.__dicom_loader.loadId(self.__label_loader.__getitem__(idx)), self.__label_loader.__getitem__(idx)


class ClassificationDataset(Dataset):
    def __init__(self, dicom_loader: DicomLoader, label_loader: LabelLoader) -> None:
        super().__init__()
        self.__dicom_loader = dicom_loader
        self.__label_loader = label_loader
        if self.__dicom_loader.__len__() != self.__label_loader.__len__():
            raise Exception(
                'DicomLoader and LabelLoader must have the same length')

    def __len__(self):
        return self.__label_loader.__len__()

    def __getitem__(self, idx):
        label = self.__label_loader.__getitem__(idx)
        dcm = self.__dicom_loader.__getitem__(idx)
        x = [None,None]
        x[0] = torch.stack(
            [dcm['dcm']['L']['MLO'], dcm['dcm']['L']['CC']], dim=1)
        x[1] = torch.stack(
            [dcm['dcm']['R']['MLO'], dcm['dcm']['R']['CC']], dim=1)
        y = [torch.tensor(label.cancerL).float(),torch.tensor(label.cancerR).float()]
        print(x[0].shape, x[1].shape, label.FinalSubjectId,
              label.subjectId, label.cancerL, label.cancerR)
        return x,y