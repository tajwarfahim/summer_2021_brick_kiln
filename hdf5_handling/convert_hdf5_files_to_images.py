# import from packages
from PIL import Image
import imageio
import argparse
import os

# import from our scripts
from ..utils.hdf5_utils import open_hdf5_file

def parse_script_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    return args


def retrieve_images(hdf5_filepath):
    hdf5_file = open_hdf5_file(filepath=hdf5_filepath)
    if 'images' not in hdf5_file.keys():
        raise ValueError("Given HDF5 file does not have correct format.")

    # reorder axes to 1000, 64, 64, 3
    images = np.transpose(hdf5_file['images'][:, 1:4, :, :], axes=[0, 2, 3, 1])
    images = images[:, :, :, ::-1]  # B, G, R --> R, G, B
    # Normalize values to be between 0 and 255
    images *= 255. / np.max(images)
    images = np.clip(images, 0, 255).astype(np.uint8)

    hdf5_file.close()
    return images
