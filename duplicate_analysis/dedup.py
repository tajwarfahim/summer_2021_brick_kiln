# imports from packages
import os
import numpy as np
import h5py as h5
import argparse

# imports from our code
import check_duplicates_utils as lib


def parse_script_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str)
    args = parser.parse_args()

    return args


def check_duplicates_in_same_directory(target_dir):
    all_files = lib.get_all_hdf5_files_in_a_directory(dir_path = target_dir)

    for i in range(len(all_files) - 1):
        for j in range(i + 1, len(all_files)):
            lib.compare_two_hdf5_files(
                filepath_1 = all_files[i],
                filepath_2 = all_files[j],
            )


def run_script():
    args = parse_script_arguments()
    check_duplicates_in_same_directory(target_dir=args.target_dir)

if __name__ == "__main__":
    run_script()
