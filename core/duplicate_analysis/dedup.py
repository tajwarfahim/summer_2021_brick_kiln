# imports from packages
import os
import numpy as np
import h5py as h5
import argparse
import configargparse

# imports from our code
from .check_duplicates_utils import (
    remove_duplicates_between_two_list_of_files,
    remove_duplicates_from_single_list_of_files,
    analyze_duplicates_between_two_files,
)


def parse_script_arguments():
    parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        parents=[parser],
        add_help=False,
    )

    # config argparse
    parser.add_argument("--config", is_config_file=True)

    # task option
    parser.add_argument("--check_single_regex", action="store_true")
    parser.add_argument("--check_double_regex", action="store_true")
    parser.add_argument("--analyze", action="store_true")

    # file path option
    parser.add_argument("--target_regex", type=str)
    parser.add_argument("--source_regex", type=str)
    parser.add_argument("--dedupped_dir", type=str)
    parser.add_argument("--regex", type=str)

    # number of chunks
    parser.add_argument("--chunk_size", type=int)

    args = parser.parse_args()
    print(parser.format_values())

    return args


def validate_script_arguments(args):
    # at least one option is true
    assert args.check_single_regex or args.check_double_regex
    # but not both options
    assert not args.check_single_regex or not args.check_double_regex


def run_script():
    args = parse_script_arguments()
    validate_script_arguments(args=args)

    if args.check_single_regex:
        remove_duplicates_from_single_list_of_files(
            regex=args.regex,
            save_dir=args.dedupped_dir,
            chunk_size=args.chunk_size,
        )

    elif args.analyze:
        analyze_duplicates_between_two_files(
            source_file_name=args.source_regex,
            target_file_name=args.target_regex,
            save_dir=args.dedupped_dir,
        )

    elif args.check_double_regex:
        remove_duplicates_between_two_list_of_files(
            source_regex=args.source_regex,
            target_regex=args.target_regex,
            save_dir=args.dedupped_dir,
        )


if __name__ == "__main__":
    run_script()
