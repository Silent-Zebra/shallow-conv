from random import choice

import numpy as np
import torch
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt


def generate_patches(image_tensor, patch_size, stride):
    # These are min edges for the top left most patch
    # For now, taking a total of 9 non-overlapping patches
    min_top_start = 0
    image_height = image_tensor.shape[1]
    max_top_start = image_height - patch_size - 2 * stride
    top_start = np.random.randint(min_top_start, max_top_start)

    min_left_start = 0
    image_width = image_tensor.shape[2]
    max_left_start = image_width - patch_size - 2 * stride
    left_start = np.random.randint(min_left_start, max_left_start)

    patches = []
    for i in range(3):
        for j in range(3):
            left_edge = left_start + j * stride
            top_edge = top_start + i * stride
            patch = image_tensor[:, top_edge:top_edge+patch_size, left_edge:left_edge+patch_size]
            patches.append(patch)

    return patches


def generate_random_patch(image_tensor, patch_size):
    image_height = image_tensor.shape[1]
    top_edge = np.random.randint(0, image_height - patch_size)
    image_width = image_tensor.shape[2]
    left_edge = np.random.randint(0, image_width - patch_size)
    patch = image_tensor[:, top_edge:top_edge + patch_size,
            left_edge:left_edge + patch_size]
    return patch


def generate_training_example(patches):
    image = patches[4]
    # patches[4] exctracts the center of the 9 patches
    # target = sum(patches) - patches[4]
    # target = target / 8

    # random selection
    selection = choice([j for j in range(9) if j != 4])

    target = patches[selection]
    # target right now is an average value of the 8 surrounding patches
    return image, target


def visualize_image(image):
    visual = make_lupton_rgb(image[0], image[1], image[2], stretch=1)

    plt.imshow(visual)
    plt.show()


def normalize_01(tensor):
    return tensor / torch.max(torch.abs(tensor)) / 2 + 0.5
