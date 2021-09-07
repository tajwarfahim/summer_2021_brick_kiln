# imports from packages
import os
import numpy as np
import h5py as h5
import argparse
import configargparse
from math import ceil
from collections import defaultdict


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
        for file in files:
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

    print("")
    for key in random_dsets:
        print("Key: ", key, "dataset shape: ", random_dsets[key].shape)

    print("")

    return random_dsets


def choose_true_positives(positive_dsets):
    true_labels = positive_dsets['labels']

    indices = []
    for i in range(true_labels.shape[0]):
        if true_labels[i] == 1:
            indices.append(i)

    indices = np.array(indices, dtype=np.int32)
    print("\nNum true positives: ", indices.shape[0])

    true_positives_dsets = {}
    for key in positive_dsets:
        true_positives_dsets[key] = positive_dsets[key][indices]

    print("True positives: ")
    for key in true_positives_dsets:
        print("Key: ", key, "dataset shape: ", true_positives_dsets[key].shape)

    print("")

    return true_positives_dsets


def create_mixture_dsets(positive_dsets, negative_dsets, common_keys):
    num_positive_examples = get_num_datapoints(datasets=positive_dsets)
    num_negative_examples = get_num_datapoints(datasets=negative_dsets)

    print("\nNum positive examples: ", num_positive_examples)
    print("Num negative examples: ", num_negative_examples, "\n")

    assert num_negative_examples % num_positive_examples == 0
    assert num_negative_examples >= num_positive_examples
    neg_pos_ratio = int(num_negative_examples / num_positive_examples)
    num_chunks = num_positive_examples

    mixture_dsets = defaultdict(list)
    for i in range(num_chunks):
        for key in common_keys:
            mixture_dsets[key].append(positive_dsets[key][i])
            mixture_dsets[key].extend(negative_dsets[key][i * neg_pos_ratio: (i + 1) * neg_pos_ratio])


    print("Final (unchunked) mixture dataset shapes: ")
    for key in common_keys:
        mixture_dsets[key] = np.array(mixture_dsets[key])
        print("Key: ", key, "dataset shape: ", mixture_dsets[key].shape)
    print("")

    return mixture_dsets


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

    mixture_dsets = create_mixture_dsets(
        positive_dsets=positive_random_sample,
        negative_dsets=negative_random_sample,
        common_keys=common_keys,
    )

    mixture_dsets_size = get_num_datapoints(datasets=mixture_dsets)
    assert mixture_dsets_size % args.num_chunks == 0
    chunk_size = int(mixture_dsets_size / args.num_chunks)

    divide_and_save_dataset(
        datasets=mixture_dsets,
        save_dir=args.save_dir,
        chunk_size=chunk_size,
    )

if __name__ == "__main__":
    run_script()
