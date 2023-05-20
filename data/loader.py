import os
import json
import pandas as pd
import pydicom as dc
from torch.utils.data import Dataset
from torch import Tensor
import logging


class DicomLoader():
    def __init__(self, dcm_root_path: str, label_path, dicom_dir=['benign', 'cancer'], dicom_series='FinalSubjectId', dicom_prefixes=['N', 'C']):
        self.__dcm_root_path = dcm_root_path
        self.__label_path = label_path
        self.__dicom_dir = dicom_dir
        self.__dicom_series = dicom_series
        self.__dicom_prefixes = dicom_prefixes
        self.reload()

    def reload(self):
        self.label = pd.read_csv(self.__label_path)

    def loadId(self, finalSubjectId: str):
        dir_path = os.path.join(self.__dcm_root_path, (self.__dicom_dir[0] if finalSubjectId.startswith(
            self.__dicom_prefixes[0]) else self.__dicom_dir[1]), finalSubjectId)
        logging.debug("Loading dicom from :", dir_path)
        data_store = {'dcm': {'L': {}, 'R': {}}, 'json': {'L': {}, 'R': {}}}
        for file in os.listdir(dir_path):
            if file.endswith(".dcm"):
                path = os.path.join(dir_path, file)
                attr = file.split('.')[0].split('-')[1:]
                dcm = dc.dcmread(path)
                data_store['dcm'][attr[0]][attr[1]] = Tensor(dcm.pixel_array)
            elif file.endswith(".json"):
                path = os.path.join(dir_path, file)
                attr = file.split('.')[0].split('-')[1:]
                data_store['json'][attr[0]][attr[1]
                                            ] = json.load(open(path, 'r'))
        return data_store

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.loadId(self.label.loc[idx][self.__dicom_series])


class LabelLoader():
    def __init__(self, label_file_path):
        self.__label_file_path = label_file_path
        self.__loadLabel()

    def __len__(self):
        return len(self.data)

    def __loadLabel(self):
        self.data = pd.read_csv(self.__label_file_path)

    def __getitem__(self, idx):
        return self.data.loc[idx]

    def getTargetLabels(self):
        return self.data.columns

    def removeLabels(self, remove_columns=None, inverse=False):
        if inverse:
            self.data.drop(
                columns=[x for x in self.data.columns if x in remove_columns], inplace=True)
        self.data.drop(
            columns=[x for x in self.data.columns if x not in remove_columns], inplace=True)
