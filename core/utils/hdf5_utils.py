import numpy as np
import h5py as h5
import os
import glob


def get_all_hdf5_files_in_a_directory(dir_path):
    valid_files = []
    if os.path.isdir(dir_path):
        for f in os.listdir(dir_path):
            object_path = os.path.join(dir_path, f)
            if os.path.isfile(object_path) and h5.is_hdf5(object_path):
                valid_files.append(object_path)

    valid_files.sort()
    return valid_files


def get_all_hdf5_files_from_regex(regex, verbose=False):
    valid_file_names = get_all_hdf5_filenames_from_regex(regex=regex, verbose=verbose)

    hdf5_files = []
    for filepath in valid_file_names:
        hdf5_file = open_hdf5_file(filepath=filepath)
        hdf5_files.append(hdf5_file)

    return hdf5_files


def get_all_hdf5_filenames_from_regex(regex, verbose=False):
    valid_files = []
    possible_files = glob.glob(regex, recursive=True)
    for file_name in possible_files:
        if os.path.isfile(file_name) and h5.is_hdf5(file_name):
            valid_files.append(file_name)

    valid_files.sort()

    if verbose:
        print("\nRegex: ", regex)
        for file_name in valid_files:
            print(file_name)
        print("")

    return valid_files


def get_common_keys(list_of_files):
    if len(list_of_files) == 0:
        return set()

    common_keys = set(list_of_files[0].keys())
    for i in range(1, len(list_of_files)):
        set_of_keys = set(list_of_files[i].keys())
        common_keys = common_keys.intersection(set_of_keys)

    return common_keys


def open_hdf5_file(filepath):
    if not h5.is_hdf5(filepath):
        raise ValueError(filepath + " is not a valid hdf5 file.")
    print("File path: ", filepath)

    file = h5.File(filepath, "r")
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
