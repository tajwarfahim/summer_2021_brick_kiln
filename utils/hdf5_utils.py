import numpy as np
import h5py as h5
import os


def get_all_hdf5_files_in_a_directory(dir_path):
    all_files = []
    if os.path.isdir(dir_path):
        for f in os.listdir(dir_path):
            object_path = os.path.join(dir_path, f)
            if os.path.isfile(object_path) and h5.is_hdf5(object_path):
                all_files.append(object_path)

    all_files.sort()
    return all_files


def get_common_keys(file_1, file_2):
    file_1_keys = set(file_1.keys())
    common_keys = []
    for file_2_key in file_2.keys():
        if file_2_key in file_1_keys:
            common_keys.append(file_2_key)

    return common_keys


def open_hdf5_file(filepath):
    if not h5.is_hdf5(filepath):
        raise ValueError(filepath + " is not a valid hdf5 file.")
    print("File path: ", filepath)

    file = h5.File(filepath, 'r')
    return file


def retrieve_datasets_from_hdf5_file(hdf5_file, keys):
    datasets = {}
    len = None
    for key in keys:
        datasets[key] = np.array(hdf5_file[key])
        if len is None:
            len = datasets[key].shape[0]
        elif len != datasets[key].shape[0]:
            raise ValueError("Different datasets in same HDF5 file have different shapes.")

    return len, datasets
