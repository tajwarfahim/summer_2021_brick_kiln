import numpy as np
import pandas as pd
import h5py as h5
import argparse as ap
from hashlib import sha256


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


def compare_indexwise_data(filepath_1, filepath_2, index_1, index_2):
    file_1 = open_hdf5_file(filepath = filepath_1)
    file_2 = open_hdf5_file(filepath = filepath_2)

    common_keys = get_common_keys(file_1 = file_1, file_2 = file_2)
    print("\n Common keys between two files: ", common_keys, "\n")

    for key in common_keys:
        file_1_all_data = np.array(file_1[key])
        file_2_all_data = np.array(file_2[key])

        if index_1 < 0  or index_1 >= file_1_all_data.shape[0]:
            raise ValueError("Invalid value for index 1")
        if index_2 < 0 or index_2 >= file_2_all_data.shape[0]:
            raise ValueError("Invalid value for index 2")

        file_1_data = file_1_all_data[index_1]
        file_2_data = file_2_all_data[index_2]

        if not np.array_equal(a1 = file_1_data, a2 = file_2_data):
            print("\n Data in this key value between two HDF5 file are different: ", key, "\n")

        else:
            print("\n Data in this key value between two HDF5 file are the same: ", key, "\n")

    file_1.close()
    file_2.close()


def debug_hashing(filepath):
    file = open_hdf5_file(filepath = filepath)
    assert "images" in file.keys()
    images = np.array(file["images"])

    # first we check if two same functions produce the same hash
    image1, image2 = images[0], images[0]
    print("Image shape: ", image1.shape)
    image1_hash = sha256(image1).hexdigest()
    image2_hash = sha256(image2).hexdigest()

    if image1_hash == image2_hash:
        print("Same images produce same hash string.")
    else:
        print("Same images do not produce same hash string, abort!")

    # check if we change the RGB value of an image slightly, that changes the hash function
    assert len(image1.shape) == 3
    image2[-1, -1, -1] += 1e-4
    changed_image2_hash = sha256(image2).hexdigest()

    if image2_hash == changed_image2_hash:
        print("Changing RGB value slightly did not change the hash function.")
    else:
        print("Changing RGB value slightly did change the hash function.")


def compare_images_in_two_hdf5_files(filepath_1, filepath_2):
    file_1 = open_hdf5_file(filepath = filepath_1)
    file_2 = open_hdf5_file(filepath = filepath_2)

    assert "images" in file_1.keys() and "images" in file_2.keys()
    images_1 = np.array(file_1["images"])
    images_2 = np.array(file_2["images"])

    cache_1 = set()
    for index in range(images_1.shape[0]):
        image = images_1[index]
        hash_string = sha256(image).hexdigest()
        if hash_string in cache_1:
            print("there are duplicate images in the first hdf5 file.")
        cache_1.add(hash_string)

    cache_2 = set()
    for index in range(images_2.shape[0]):
        image = images_2[index]
        hash_string = sha256(image).hexdigest()
        if hash_string in cache_2:
            print("there are duplicate images in the first hdf5 file.")
        cache_2.add(hash_string)

    num_duplicates = 0
    for hash_string in cache_1:
        if hash_string in cache_2:
            num_duplicates += 1

    print("Number of duplicates between 1st and 2nd file:", num_duplicates)
