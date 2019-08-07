# Adapted from https://github.com/adambielski/siamese-triplet

from random import choice

import numpy as np
import torch
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt

from itertools import combinations


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


def pdist(vectors):
    # print(vectors)
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in list(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                # Calculate loss on the embeddings
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=hardest_negative,
                                           cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=random_hard_negative,
                                           cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=lambda x: semihard_negative(x, margin),
                                           cpu=cpu)


