import h5py
import io
import numpy as np
import os
import pandas as pd

from base64 import encodebytes
from flask import Flask
from flask import render_template, send_from_directory, jsonify, request, send_file
from PIL import Image, ImageOps


app = Flask(__name__)

TEMPLATES_AUTO_RELOAD = True

LOW_INDEX = 0
HIGH_INDEX = 25


def create_examples():
    indices = [i for i in range(LOW_INDEX, HIGH_INDEX)]
    examples = {
        #'/atlas/u/jihyeonlee/handlabeling/delta+1/jihyeon/examples_0_new.hdf5': indices,
        #'/atlas/u/jihyeonlee/handlabeling/delta+1/jihyeon/examples_1_new.hdf5': indices,
        #'/atlas/u/jihyeonlee/handlabeling/delta+1/jihyeon/examples_3_new.hdf5': indices,
        #'/atlas/u/jihyeonlee/handlabeling/delta-1/jihyeon/examples_1_new.hdf5': indices,
        "/atlas/u/jihyeonlee/handlabeling/positives/examples_0_new.hdf5": indices,
        "/atlas/u/jihyeonlee/handlabeling/positives/examples_1_new.hdf5": indices,
        "/atlas/u/jihyeonlee/handlabeling/positives/examples_3_new.hdf5": indices,
        "/atlas/u/jihyeonlee/handlabeling/positives/examples_4_new.hdf5": indices,
        "/atlas/u/jihyeonlee/handlabeling/positives/examples_5_new.hdf5": indices,
        "/atlas/u/jihyeonlee/handlabeling/positives/examples_6_new.hdf5": indices,
        "/atlas/u/jihyeonlee/handlabeling/positives/examples_7_new.hdf5": indices,
    }

    return examples


def encode_images(imgs):
    encoded_imgs = []
    for np_img in imgs:
        pil_img = Image.fromarray(np_img)
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, "JPEG", quality=100)
        # byte_arr.seek(0)
        encoded_img = encodebytes(byte_arr.getvalue()).decode("ascii")  # encode as base64
        encoded_img_tag = f'<img src="data:image/jpg;base64,{encoded_img}" height="256"/>'
        encoded_imgs.append(encoded_img_tag)
    return encoded_imgs


def load_data(hdf5_filename, indexes):
    global images
    global labels
    global bounds
    global total
    # Read in the current data
    with h5py.File(hdf5_filename, "r") as f:
        print(f.keys())
        images = np.transpose(f["images"][:, 1:4, :, :], axes=[0, 2, 3, 1])  # reorder axes to 1000, 64, 64, 3
        images = images[:, :, :, ::-1]  # B, G, R --> R, G, B
        # Normalize values to be between 0 and 255
        # images *= 255. / (0.8 * np.max(images))
        images *= 255.0 / np.max(images)
        images = np.clip(images, 0, 255).astype(np.uint8)
        labels = f["labels"][:]
        bounds = f["bounds"][:]
        total = images.shape[0]
    return {
        "images": encode_images(images[indexes]),
        "labels": labels[indexes],
        "lon": bounds[indexes, 0],
        "lat": bounds[indexes, 3],
    }


@app.context_processor
def inject_dict_for_all_templates():
    examples = create_examples()
    examples_data = {}
    for hdf5_file in examples.keys():
        examples_data[hdf5_file] = load_data(hdf5_file, examples[hdf5_file])

    return {"examples": examples_data}


@app.route("/")
def hello():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
