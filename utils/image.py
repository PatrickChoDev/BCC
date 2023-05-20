import pydicom
import matplotlib.pyplot as plt

def read_dicom(path):
    return pydicom.dcmread(path)

def plot_dicom(dcm):
    plt.imshow(dcm.pixel_array, cmap=plt.cm.bone)
    plt.show()

def plot_dicom_from_path(path):
    plot_dicom(read_dicom(path))