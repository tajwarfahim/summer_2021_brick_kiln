# import from packages
from PIL import Image
import argparse
import configargparse
import os
import numpy as np

# import from our scripts
from ..utils.hdf5_utils import (
    open_hdf5_file,
    get_all_hdf5_filenames_from_regex,
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

    # paths to files and directories
    parser.add_argument("--regex", type=str)
    parser.add_argument("--image_dir", type=str)

    args = parser.parse_args()

    print(parser.format_values())

    return args


def retrieve_images(hdf5_file):
    # retrieve images and close file
    images = np.array(hdf5_file["images"])

    # reorder axes to 1000, 64, 64, 3
    images = np.transpose(images[:, 1:4, :, :], axes=[0, 2, 3, 1])
    images = images[:, :, :, ::-1]  # B, G, R --> R, G, B
    # Normalize values to be between 0 and 255
    images *= 255.0 / np.max(images)
    images = np.clip(images, 0, 255).astype(np.uint8)

    return images


def save_images_as_png(images, image_dir, sub_dir):
    if not isinstance(images, np.ndarray):
        raise ValueError("Given images array does not have proper format.")
    if len(images.shape) != 4:
        raise ValueError("Given images array does not have proper shape.")

    image_dir = os.path.join(image_dir, sub_dir)
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    for img_indx in range(images.shape[0]):
        # create image path
        img_path = os.path.join(image_dir, sub_dir + "_" + f"image_{img_indx}.jpeg")

        # retreive images
        img = images[img_indx]
        pil_img = Image.fromarray(img).convert("RGB")

        # save image to the aforementioned path
        pil_img.save(img_path)


def run_script():
    args = parse_script_arguments()

    hdf5_file_names = get_all_hdf5_filenames_from_regex(regex=args.regex, verbose=True)

    for i in range(len(hdf5_file_names)):
        hdf5_file = open_hdf5_file(filepath=hdf5_file_names[i])
        images = retrieve_images(hdf5_file=hdf5_file)
        hdf5_file.close()

        main_file_name = hdf5_file_names[i].split("/")[-1]
        assert main_file_name.endswith(".hdf5")
        sub_dir = main_file_name[0 : len(main_file_name) - 5]

        save_images_as_png(images=images, image_dir=args.image_dir, sub_dir=sub_dir)


if __name__ == "__main__":
    run_script()
