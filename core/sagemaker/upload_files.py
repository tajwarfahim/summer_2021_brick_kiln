# general imports
import argparse
import configargparse
import glob
import os

# imports from our packages
from .sagemaker_utils import upload_files_to_bucket

def parse_script_arguments():
    parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        parents=[parser],
        add_help=False,
    )

    # config argparse
    parser.add_argument("--config", is_config_file=True)

    # file name options
    parser.add_argument("--filename_regex", type=str)

    # s3 bucket options
    parser.add_argument("--bucket_name", type=str)
    parser.add_argument("--bucket_key_prefix", type=str)

    parser.add_argument("--verbose", action="store_true")

    # parse and print args
    args = parser.parse_args()
    print(parser.format_values())

    return args

def get_all_filenames(regex, verbose):
    potential_filenames = glob.glob(regex, recursive=True)
    file_names = []
    for file_name in potential_filenames:
        if os.path.isfile(file_name):
            file_names.append(file_name)

    if verbose:
        print("\nNumber of files: ", len(file_names), "\n")
        for i in range(len(file_names)):
            print("Index: ", i, " filename: ", file_names[i])
        print("")

    return file_names

def run_script():
    args = parse_script_arguments()

    files_to_upload = get_all_filenames(
        regex=args.filename_regex,
        verbose=args.verbose,
    )
    upload_files_to_bucket(
        file_names=files_to_upload,
        bucket_name=args.bucket_name,
        bucket_key_prefix=args.bucket_key_prefix,
    )

if __name__ == "__main__":
    run_script()
