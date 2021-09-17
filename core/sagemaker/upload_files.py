# general imports
import argparse
import configargparse
import glob
import os
import time

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

    # task options
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--bucket_name", type=str)
    parser.add_argument("--verbose", action="store_true")

    # parse and print args
    args = parser.parse_args()
    print(parser.format_values())

    return args


def get_all_directory_elements(directory, content_type, verbose):
    if content_type == "file":
        check_function = os.path.isfile
    elif content_type == "directory":
        check_function = os.path.isdir
    else:
        raise ValueError("Given content type not supported.")

    elements = []
    for potential_element in os.listdir(directory):
        path_to_element = os.path.join(directory, potential_element)
        if check_function(path_to_element):
            elements.append(path_to_element)

    elements.sort()

    if verbose:
        print("\nDirectory name: ", directory)
        print("Number of elements: ", len(elements), "\n")
        for i in range(len(elements)):
            print("Index: ", i, "element: ", elements[i])

        print("")

    return elements


def run_script():
    args = parse_script_arguments()

    subdirs = get_all_directory_elements(
        directory=args.img_dir,
        content_type="directory",
        verbose=args.verbose,
    )

    for subdir in subdirs:
        start = time.time()

        images_to_upload = get_all_directory_elements(
            directory=subdir,
            content_type="file",
            verbose=args.verbose,
        )

        bucket_key_prefix = os.path.join(images_to_upload[0].split("/")[-2], "input")

        upload_files_to_bucket(
            file_names=images_to_upload,
            bucket_name=args.bucket_name,
            bucket_key_prefix=bucket_key_prefix,
        )

        end = time.time()

        print("\nTime took to upload all ", len(images_to_upload), "files: ", end - start, "seconds.")


if __name__ == "__main__":
    run_script()
