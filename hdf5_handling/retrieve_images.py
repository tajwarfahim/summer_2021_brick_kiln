# import from packages
from PIL import Image
import imageio
import argparse
import configargparse
import os

# import from our scripts
from ..utils.hdf5_utils import open_hdf5_file


def parse_script_arguments():
    parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        parents=[parser],
        add_help=False,
    )

    # paths to files and directories
    parser.add_argument("--hdf5_filepath", type=str)
    parser.add_argument("--image_dir", type=str)

    args = parser.parse_args()

    return args


def retrieve_images(hdf5_filepath):
    hdf5_file = open_hdf5_file(filepath=hdf5_filepath)
    if "images" not in hdf5_file.keys():
        raise ValueError("Given HDF5 file does not have correct format.")

    # retrieve images and close file
    images = hdf5_file["images"]
    hdf5_file.close()

    # reorder axes to 1000, 64, 64, 3
    images = np.transpose(images[:, 1:4, :, :], axes=[0, 2, 3, 1])
    images = images[:, :, :, ::-1]  # B, G, R --> R, G, B
    # Normalize values to be between 0 and 255
    images *= 255.0 / np.max(images)
    images = np.clip(images, 0, 255).astype(np.uint8)

    return images


def save_images_as_png(images, image_dir):
    if not isinstance(images, np.ndarray):
        raise ValueError("Given images array does not have proper format.")
    if len(images.shape) != 4:
        raise ValueError("Given images array does not have proper shape.")

    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    for img_indx in range(images.shape[0]):
        # create image path
        img_path = os.path.join(image_dir, f"image_{img_indx}.jpeg")

        # retreive images
        img = images[img_indx]
        pil_img = Image.fromarray(img).convert("RGB")

        # save image to the aforementioned path
        pil_img.save(img_path)


def run_script():
    args = parse_script_arguments()
    images = retrieve_images(hdf5_filepath=args.hdf5_filepath)
    save_images_as_png(images=images, image_dir=args.image_dir)


if __name__ == "__main__":
    run_script()
