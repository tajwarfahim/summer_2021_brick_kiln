import numpy as np
import h5py as h5
from hashlib import sha256
import os


def get_all_hdf5_files_in_a_directory(dir_path):
    all_files = []
    if os.path.isdir(dir_path):
        for f in os.listdir(dir_path):
            object_path = os.path.join(dir_path, f)
            if os.path.isfile(object_path) and h5.is_hdf5(object_path):
                all_files.append(object_path)

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


def hash_datasets_of_a_single_file(num_datapoints, datasets):
    hash_strings = []
    for index in range(num_datapoints):
        encoder = None
        for key in datasets:
            data_point = datasets[key][index]
            if encoder is None:
                encoder = sha256(data_point)
            else:
                encoder.update(data_point)

        hash_strings.append(encoder.hexdigest())

    if len(set(hash_strings)) != len(hash_strings):
        raise ValueError("There are duplicates within a single file.")

    return hash_strings


def compare_two_hdf5_files(filepath_1, filepath_2):
    file_1 = open_hdf5_file(filepath_1)
    file_2 = open_hdf5_file(filepath_2)

    common_keys = get_common_keys(file_1 = file_1, file_2 = file_2)
    print("\n Common keys between two files: ", common_keys, "\n")

    num_datapoints_1, datasets_1 = retrieve_datasets_from_hdf5_file(
        hdf5_file = file_1, keys = common_keys
    )
    num_datapoints_2, datasets_2 = retrieve_datasets_from_hdf5_file(
        hdf5_file = file_2, keys = common_keys
    )

    file_1.close()
    file_2.close()

    hash_strings_1 = hash_datasets_of_a_single_file(
        num_datapoints = num_datapoints_1, datasets = datasets_1
    )

    hash_strings_2 = hash_datasets_of_a_single_file(
        num_datapoints = num_datapoints_2, datasets = datasets_2
    )

    duplicates_between_files = []
    for i in range(num_datapoints_1):
        for j in range(num_datapoints_2):
            if hash_strings_1[i] == hash_strings_2[j]:
                duplicates_between_files.append(i)

    print(
        "\nThere have been ", len(duplicates_between_files),
        " duplicates between the two files.\n"
    )

    return datasets_1, datasets_2, duplicates_between_files


def remove_duplicates(target_path, source_path, dedupped_file_path):
    datasets_target, datasets_source, duplicates_between_files = compare_two_hdf5_files(
        filepath_1=filepath_1,
        filepath_2=filepath_2,
    )

    dedupped_hdf5_file = h5.File(dedupped_file_path, 'w')
    for key in datasets_target:
        particular_dataset = datasets_target[key]
        assert isinstance(particular_dataset, np.ndarray)

        new_dataset = []
        for i in range(particular_dataset.shape[0]):
            if i not in duplicates_between_files:
                new_dataset.append(particular_dataset[i])

        new_dataset = np.array(new_dataset)
        dedupped_hdf5_file.create_dataset(key, data=new_dataset)

    dedupped_hdf5_file.close()
