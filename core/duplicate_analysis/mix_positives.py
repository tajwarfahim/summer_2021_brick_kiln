# imports from packages
import os
import numpy as np
import h5py as h5
import argparse
import configargparse
from math import ceil


# import from our packages
from ..utils.hdf5_utils import (
    get_all_hdf5_files_from_regex,
    get_common_keys,
    open_hdf5_file,
    retrieve_datasets_from_hdf5_file,
)
from .check_duplicates_utils import (
    get_num_datapoints,
    divide_and_save_dataset,
)


def load_all_files(regex):
    files = get_all_hdf5_files_from_regex(regex=regex)
    common_keys = get_common_keys(list_of_files=files)

    dsets = {}
    for key in common_keys:
        for file in neg_files:
            if key not in dsets:
                dsets[key] = [np.array(file[key])]
            else:
                dsets[key].append(np.array(file[key]))

    for file in files:
        file.close()

    for key in common_keys:
        dsets[key] = np.concatenate(dsets[key])
        print("Key: ", key, "dataset shape: ", dsets[key].shape)

    print("")

    return dsets


def randomly_sample_dataset(dsets, num_sample):
    num_datapoints = get_num_datapoints(datasets=dsets)
    assert num_sample <= num_datapoints

    random_indices = np.random.choice(a=num_datapoints, size=num_sample, replace=False)

    random_dsets = {}
    for key in dsets:
        random_dsets[key] = dsets[key][random_indices]

    for key in random_dsets:
        print("Key: ", key, "dataset shape: ", random_dsets[key].shape)

    print("")

    return random_dsets


def choose_true_positives(positives_dsets):
    true_labels = positives_dsets['labels']

    indices = []
    for i in range(true_labels.shape[0]):
        if true_labels[i] == 1:
            indices.append(i)

    indices = np.array(indices, dtype=np.int32)
    print("Num true positives: ", indices.shape[0])

    true_positives_dsets = {}
    for key in positives_dsets:
        true_positives_dsets[key] = positives_dsets[key][indices]

    print("True positives: ")
    for key in true_positives_dsets:
        print("Key: ", key, "dataset shape: ", true_positives_dsets[key].shape)

    return true_positives_dsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        parents=[parser],
        add_help=False,
    )

    parser.add_argument("--config", is_config_file=True)

    parser.add_argument("--neg_regex", type=str)
    parser.add_argument("--pos_regex", type=str)
    parser.add_argument("--save_dir", type=str)

    parser.add_argument("--negative_samples", type=int)
    parser.add_argument("--positive_samples", type=int)

    parser.add_argument("--num_chunks", type=int)

    args=parser.parse_args()
    print(parser.format_values())

    return args


def create_mixture_dsets(positive_dsets, negative_dsets, num_chunks, common_keys):
    mixture_dsets = {}

    size_positive_chunk = ceil(get_num_datapoints(datasets=positive_dsets) / num_chunks)
    size_negative_chunk = ceil(get_num_datapoints(datasets=negative_dsets) / num_chunks)

    for i in range(num_chunks):
        for key in common_keys:
            if key not in mixture_dsets:
                mixture_dsets[key] = [positive_dsets[key][i * size_positive_chunk : (i + 1) * size_positive_chunk]]
            mixture_dsets[key].append(negative_dsets[key][i * size_negative_chunk : (i + 1) * size_negative_chunk])

    print("Final (unchunked) mixture dataset shapes: ")
    for key in common_keys:
        print("Key: ", key, "dataset shape: ", mixture_dsets[key].shape)
    print("")

    return mixture_dsets, (size_negative_chunk + size_positive_chunk)


def run_script():
    args = parse_args()

    negative_dsets = load_all_files(regex=args.neg_regex)

    positive_dsets = load_all_files(regex=args.pos_regex)
    true_positives_dsets = choose_true_positives(positive_dsets=positive_dsets)

    positive_random_sample = randomly_sample_dataset(
        dsets=true_positives_dsets, num_sample=args.positive_samples,
    )
    negative_random_sample = randomly_sample_dataset(
        dsets= negative_dsets, num_sample=args.negative_samples,
    )

    common_keys = set(positive_random_sample.keys()).intersection(set(negative_random_sample.keys()))
    print("\nCommon keys: ", common_keys, "\n")

    mixture_dset, chunk_size = create_mixture_dsets(
        positive_dsets=positive_random_sample,
        negative_dsets=negative_random_sample,
        num_chunks=args.num_chunks,
        common_keys=common_keys,
    )

    divide_and_save_dataset(
        datasets=mixture_dsets,
        save_dir=args.save_dir,
        chunk_size=chunk_size,
    )

if __name__ == "__main__":
    run_script()
