# import from general packages
import numpy as np
from hashlib import sha256
import os
from collections import defaultdict
import h5py as h5
import math


# import from our packages
from ..utils.hdf5_utils import (
    get_all_hdf5_files_from_regex,
    get_common_keys,
    open_hdf5_file,
    retrieve_datasets_from_hdf5_file,
)


def hash_datasets(list_of_files, keys):
    hash_strings = set()

    for file in list_of_files:
        num_datapoints = get_num_datapoints(datasets=file)

        for i in range(num_datapoints):
            hash_encoder = None
            for key in keys:
                if hash_encoder is None:
                    hash_encoder = sha256(file[key][i])
                else:
                    hash_encoder.update(file[key][i])

            hash_strings.add(hash_encoder.hexdigest())

    return hash_strings


def compare_datapoint(file_1, file_2, index_1, index_2, common_keys):
    is_duplicate = True
    for key in common_keys:
        dset_1 = file_1[key]
        dset_2 = file_2[key]

        datapoint_1 = dset_1[index_1]
        datapoint_2 = dset_2[index_2]

        if not np.array_equal(datapoint_1, datapoint_2):
            is_duplicate = False
            break

    return is_duplicate


def put_duplicates_at_the_start_of_dsets(dsets_map, duplicate_indices, keys):
    new_dsets_map = defaultdict(list)
    for key in keys:
        for index in duplicate_indices:
            new_dsets_map[key].append(dsets_map[key][index])

    num_datapoints = get_num_datapoints(datasets=dsets_map)
    for key in keys:
        for index in range(num_datapoints):
            if index not in duplicate_indices:
                new_dsets_map[key].append(dsets_map[key][index])

    datasets = {}
    for key in keys:
        datasets[key] = np.array(new_dsets_map[key])

    return datasets


def convert_hdf5_file_to_map(file, keys):
    dset_map = {}
    for key in keys:
        dset_map[key] = np.array(file[key])

    return dset_map


def analyze_duplicates_between_two_files(source_file_name, target_file_name, save_dir):
    assert h5.is_hdf5(source_file_name) and h5.is_hdf5(target_file_name)

    print("Source filename: ", source_file_name)
    print("Target filename: ", target_file_name)

    source_file = h5.File(source_file_name, 'r')
    target_file = h5.File(target_file_name, 'r')

    common_keys = get_common_keys(list_of_files=[source_file, target_file])
    print("\nCommon keys between all hdf5 files: ", common_keys, "\n")

    source_num_datapoints = get_num_datapoints(datasets=source_file)
    target_num_datapoints = get_num_datapoints(datasets=target_file)

    source_dsets = convert_hdf5_file_to_map(file=source_file, keys=common_keys)
    target_dsets = convert_hdf5_file_to_map(file=target_file, keys=common_keys)

    source_file.close()
    target_file.close()

    source_file_duplicate_indices = []
    target_file_duplicate_indices = []

    for i in range(source_num_datapoints):
        for j in range(target_num_datapoints):
            if compare_datapoint(source_dsets, target_dsets, i, j, common_keys):
                source_file_duplicate_indices.append(i)
                target_file_duplicate_indices.append(j)

    print("Number of duplicates: ", len(source_file_duplicate_indices))

    source_dsets_modified = put_duplicates_at_the_start_of_dsets(
        dsets_map=source_dsets,
        duplicate_indices=source_file_duplicate_indices,
        keys=common_keys,
    )

    target_dsets_modified = put_duplicates_at_the_start_of_dsets(
        dsets_map=target_dsets,
        duplicate_indices=target_file_duplicate_indices,
        keys=common_keys,
    )

    divide_and_save_dataset(
        datasets=source_dsets_modified,
        save_dir=os.path.join(save_dir, "delta+1"),
        chunk_size=0,
    )

    divide_and_save_dataset(
        datasets=target_dsets_modified,
        save_dir=os.path.join(save_dir, "positives"),
        chunk_size=0,
    )


def remove_duplicates_between_two_list_of_files(source_regex, target_regex, save_dir):
    source_hdf5_files = get_all_hdf5_files_from_regex(regex=source_regex, verbose=True)
    target_hdf5_files = get_all_hdf5_files_from_regex(regex=target_regex, verbose=True)

    common_keys = get_common_keys(list_of_files=source_hdf5_files + target_hdf5_files)
    print("\nCommon keys between all hdf5 files: ", common_keys, "\n")

    hash_strings_from_source_files = hash_datasets(
        list_of_files=source_hdf5_files,
        keys=common_keys,
    )

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for file_index in range(len(target_hdf5_files)):
        file_name = "task_" + str(file_index) + ".hdf5"
        file_path = os.path.join(save_dir, file_name)
        file = h5.File(file_path, "w")

        target_hdf5_file = target_hdf5_files[file_index]
        num_datapoints = get_num_datapoints(datasets=target_hdf5_file)
        dsets = defaultdict(list)

        for i in range(num_datapoints):
            hash_encoder = None
            for key in common_keys:
                if hash_encoder is None:
                    hash_encoder = sha256(target_hdf5_file[key][i])
                else:
                    hash_encoder.update(target_hdf5_file[key][i])

            hash_string = hash_encoder.hexdigest()
            if hash_string not in hash_strings_from_source_files:
                for key in common_keys:
                    dsets[key].append(target_hdf5_file[key][i])

        num_unique_elements = None
        for key in common_keys:
            dataset = np.array(dsets[key])
            if num_unique_elements is None:
                num_unique_elements = dataset.shape[0]
            file.create_dataset(key, data=dataset)

        print("\nNum total elements: ", num_datapoints)
        print("Num unique elements: ", num_unique_elements, "\n")

    for hdf5_file_to_close in source_hdf5_files + target_hdf5_files:
        hdf5_file_to_close.close()


def remove_duplicates_from_single_list_of_files(regex, save_dir, chunk_size):
    hdf5_files = get_all_hdf5_files_from_regex(regex=regex)

    common_keys = get_common_keys(list_of_files=hdf5_files)
    print("\nCommon keys between all hdf5 files: ", common_keys, "\n")

    dedupped_datasets = defaultdict(list)
    hash_of_datasets = set()
    num_total_elements = 0

    for i in range(len(hdf5_files)):
        print("Index of file being processed: ", i)

        file = hdf5_files[i]
        num_datapoints = get_num_datapoints(datasets=file)
        dsets = {}
        for key in common_keys:
            dsets[key] = np.array(file[key])

        num_total_elements += num_datapoints

        for i in range(num_datapoints):
            hash_encoder = None
            for key in common_keys:
                if hash_encoder is None:
                    hash_encoder = sha256(dsets[key][i])
                else:
                    hash_encoder.update(dsets[key][i])

            hash_value = hash_encoder.hexdigest()

            if hash_value not in hash_of_datasets:
                hash_of_datasets.add(hash_value)

                for key in common_keys:
                    dedupped_datasets[key].append(dsets[key][i])

        file.close()

    for key in common_keys:
        dedupped_datasets[key] = np.array(dedupped_datasets[key])
        print("key:", key, " shape unique elements: ", dedupped_datasets[key].shape)

    print("\nNumber of total elements (including duplicates): ", num_total_elements, "\n")

    divide_and_save_dataset(datasets=dedupped_datasets, save_dir=save_dir, chunk_size=chunk_size)


def get_num_datapoints(datasets):
    num_datapoints = None
    for key in datasets.keys():
        if num_datapoints is None:
            num_datapoints = datasets[key].shape[0]
        elif num_datapoints != datasets[key].shape[0]:
            raise ValueError("Incompatible dataset.")

    return num_datapoints


def divide_and_save_dataset(datasets, save_dir, chunk_size):
    num_datapoints = get_num_datapoints(datasets=datasets)

    if chunk_size == 0:
        chunk_size = int(input("Enter the size of each individual chunk: "))

    assert chunk_size > 0
    num_chunks = math.ceil(num_datapoints / chunk_size)

    assert save_dir is not None and isinstance(save_dir, str)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i in range(num_chunks):
        file_name = "task_" + str(i) + ".hdf5"
        file_path = os.path.join(save_dir, file_name)
        file = h5.File(file_path, "w")

        for key in datasets:
            chunk_dataset = datasets[key][i * chunk_size : (i + 1) * chunk_size]
            file.create_dataset(key, data=chunk_dataset)

        file.close()
