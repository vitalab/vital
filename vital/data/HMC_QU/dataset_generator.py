import argparse
import os
from pathlib import Path
from typing import List
import cv2
import h5py
import torch.nn as nn
import scipy.io
from matplotlib import pyplot as plt
from PIL.Image import LINEAR
from PIL import Image
import skvideo.io
import skvideo
import pandas as pd
import pprint
import numpy as np
from skimage import color
from sklearn.model_selection import train_test_split
from vital.data.camus.config import View, CamusTags, img_save_options, seg_save_options
from vital.data.config import Subset
from vital.utils.image.transform import resize_image
from scipy import misc
import math
import re

import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

AVAILABLE_GT_SYMBOL = 'ü'
AVAILABLE_GT_ROW = 'LV Wall Ground-truth Segmentation Masks'
DATASET_INFO_FILENAME = 'HMC_QU.xlsx'
IMG_RELATIVE_PATH = Path('HMC-QU Echos/HMC-QU Echos')
GT_RELATIVE_PATH = Path('LV Ground-truth Segmentation Masks')
target_image_size = (256, 256)


#
# def find_lines(img: np.ndarray):
#     img_shape = img.shape
#
#     x_max = img_shape[0]
#     y_max = img_shape[1]
#
#     theta_max = 1.0 * math.pi
#     theta_min = 0.0
#
#     r_min = 0.0
#     r_max = math.hypot(x_max, y_max)
#
#     r_dim = 200
#     theta_dim = 300
#
#     hough_space = np.zeros((r_dim, theta_dim))
#
#     for x in range(x_max):
#         for y in range(y_max):
#             if img[x, y, 0] == 255: continue
#             for itheta in range(theta_dim):
#                 theta = 1.0 * itheta * theta_max / theta_dim
#                 r = x * math.cos(theta) + y * math.sin(theta)
#                 ir = int(r_dim * (1.0 * r) / r_max)
#                 hough_space[ir, itheta] = hough_space[ir, itheta] + 1
#
#     neighborhood_size = 20
#     threshold = 140
#
#     data_max = filters.maximum_filter(hough_space, neighborhood_size)
#     maxima = (hough_space == data_max)
#
#     data_min = filters.minimum_filter(hough_space, neighborhood_size)
#     diff = ((data_max - data_min) > threshold)
#     maxima[diff == 0] = 0
#
#     labeled, num_objects = ndimage.label(maxima)
#     slices = ndimage.find_objects(labeled)
#
#     x, y = [], []
#     for dy, dx in slices:
#         x_center = (dx.start + dx.stop - 1) / 2
#         x.append(x_center)
#         y_center = (dy.start + dy.stop - 1) / 2
#         y.append(y_center)
#
#     print(x)
#     print(y)
#
#     line_index = 1
#
#     for i, j in zip(y, x):
#
#         r = round((1.0 * i * r_max) / r_dim, 1)
#         theta = round((1.0 * j * theta_max) / theta_dim, 1)
#
#         fig, ax = plt.subplots()
#
#         ax.imshow(img)
#
#         ax.autoscale(False)
#
#         px = []
#         py = []
#         for i in range(-y_max - 40, y_max + 40, 1):
#             px.append(math.cos(-theta) * i - math.sin(-theta) * r)
#             py.append(math.sin(-theta) * i + math.cos(-theta) * r)
#
#         ax.plot(px, py, linewidth=10)
#         plt.show()
#         line_index = line_index + 1
#
# def find_lines(img):
#     plt.imshow(img)
#     plt.show()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#
#     plt.imshow(edges)
#     plt.show()
#
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 125)
#
#     print(lines.shape)
#
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#     ax.autoscale(False)
#
#     for line in lines:
#         for rho, theta in line:
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
#             ax.plot([x1, x2], [y1, y2], linewidth=2)
#
#     plt.show()


def generate_dataset(path: Path, name: Path, seed: int, test_size: float, val_size: float):
    """Generate the h5 dataset.

    Args:
        path: path to the raw data.
        name: Name of the output file.
    """
    dataset_info = pd.read_excel(path / DATASET_INFO_FILENAME, index_col='ECHO')
    columns = ['SEG1', 'SEG2', 'SEG3', 'SEG5', 'SEG6', 'SEG7',
               'Reference Frame', 'End of Cycle', 'LV Wall Ground-truth Segmentation Masks']
    dataset_info.columns = columns
    dataset_info = dataset_info[dataset_info[AVAILABLE_GT_ROW] == AVAILABLE_GT_SYMBOL]  # Only keep samples with GT
    patient_list = list(dataset_info.index.values)

    train_patients, test_patients = train_test_split(patient_list, test_size=test_size, random_state=seed)
    train_patients, val_patients = train_test_split(train_patients, test_size=val_size, random_state=seed)

    with h5py.File(name, "w") as h5f:
        # List all the samples by vendor.

        # Create training sets
        print("Training...")
        print(len(train_patients))
        train_group = h5f.create_group(Subset.TRAIN.value)
        generate_set(train_patients, path, dataset_info, train_group)

        print("Validation...")
        print(len(val_patients))
        val_group = h5f.create_group(Subset.VAL.value)
        generate_set(val_patients, path, dataset_info, val_group)

        print("Testing...")
        print(len(test_patients))
        test_group = h5f.create_group(Subset.TEST.value)
        generate_set(test_patients, path, dataset_info, test_group)


def generate_set(patient_list: List[str], path: Path, dataset_info: pd.DataFrame, group: h5py.Group):
    for patient in patient_list:
        img = skvideo.io.vread(str(path / IMG_RELATIVE_PATH / (patient + '.avi')))
        gt = scipy.io.loadmat(str(path / GT_RELATIVE_PATH / ('Mask_' + patient + '.mat')))['predicted']

        reference_frame = dataset_info.loc[patient]['Reference Frame']
        end_of_cycle = dataset_info.loc[patient]['End of Cycle']
        img = img[reference_frame - 1:end_of_cycle]  # Subtract one as frames are from Matlab format (index starts at 1)

        img = np.array(
            [np.array(color.rgb2gray(resize_image(x, target_image_size, resample=LINEAR))) for x in img]
        )
        gt_proc = np.array([resize_image(y, target_image_size) for y in gt])

        patient_group = group.create_group(re.split('_|\s', patient)[0])
        patient_view_group = patient_group.create_group(View.A4C)
        patient_view_group.create_dataset(name=CamusTags.img_proc, data=img, **img_save_options)
        patient_view_group.create_dataset(name=CamusTags.gt_proc, data=gt_proc, **seg_save_options)
        patient_view_group.create_dataset(name=CamusTags.gt, data=gt, **seg_save_options)


def main():
    """Main function where we define the argument for the script."""
    parser = argparse.ArgumentParser(
        description=(
            "Script to create the HMC_QU dataset "
            "hdf5 file from the directory. "
            "The given directory need to have two "
            "directory inside, 'training' and 'testing'."
        )
    )
    parser.add_argument("--path", type=Path, required=True, help="Path of the HMC_QU downlaoed dataset unziped folder.")
    parser.add_argument("--name", type=Path, default=Path('HMC_QU.h5'), help="Name of the generated hdf5 file.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random ")
    parser.add_argument("--test_size", type=float, default=0.25, help="Size of test set")
    parser.add_argument("--val_size", type=float, default=0.1, help="Size of validation set")

    args = parser.parse_args()

    generate_dataset(args.path, args.name, args.seed, args.test_size, args.val_size)


if __name__ == "__main__":
    main()
