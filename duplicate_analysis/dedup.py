# imports from packages
import os
import numpy as np
import h5py as h5
import argparse

# imports from our code
import sys
from .check_duplicates_utils import *


def parse_script_arguments():
    parser = argparse.ArgumentParser()

    # task option
    parser.add_argument("--check_single_directory", action="store_true")
    parser.add_argument("--compare_two_directories", action="store_true")
    parser.add_argument("--remove_duplicates", action="store_true")

    # file path option
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--source_dir", type=str)
    parser.add_argument("--dedupped_dir", type=str)
    args = parser.parse_args()

    return args


def check_duplicates_in_same_directory(target_dir):
    all_files = get_all_hdf5_files_in_a_directory(dir_path=target_dir)

    print("\n Directory:", target_dir)
    print("Dataset files in directory: ", all_files)
    print("Checking if there are duplicates between any two of these files.\n")

    for i in range(len(all_files) - 1):
        for j in range(i + 1, len(all_files)):
            compare_two_hdf5_files(
                filepath_1 = all_files[i],
                filepath_2 = all_files[j],
            )


def compare_two_different_directories(source_dir, target_dir, dedupped_dir, should_remove_duplicates):
    target_files = get_all_hdf5_files_in_a_directory(dir_path=target_dir)
    source_files = get_all_hdf5_files_in_a_directory(dir_path=source_dir)

    if not os.path.isdir(dedupped_dir):
        os.makedirs(dedupped_dir)

    print("\n Target directory:", target_dir)
    print("Dataset files in directory: ", target_files)
    print("\nSource directory:", source_dir)
    print("Dataset files in directory: ", source_files)
    print("\nChecking if there are duplicates between any two of these files.\n")

    for i in range(len(target_files)):
        target_file_name = target_files[i].split("/")[-1]
        dedupped_file_path = os.path.join(dedupped_dir, target_file_name)
        for j in range(len(source_files)):
            if target_files[i] != source_files[j]:
                if j == 0:
                    target_path = target_files[i]
                else:
                    target_path = dedupped_file_path

                if not should_remove_duplicates:
                    compare_two_hdf5_files_numpy(
                        filepath_1=target_path,
                        filepath_2=source_files[j],
                    )

                else:
                    remove_duplicates(
                        target_path=target_path,
                        source_path=source_files[j],
                        dedupped_file_path=dedupped_file_path,
                    )


def run_script():
    args = parse_script_arguments()

    if args.check_single_directory:
        check_duplicates_in_same_directory(target_dir=args.target_dir)

    elif args.compare_two_directories:
        compare_two_different_directories(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            dedupped_dir=args.dedupped_dir,
            should_remove_duplicates=args.remove_duplicates,
        )

if __name__ == "__main__":
    run_script()
