# Adapted from https://github.com/adambielski/siamese-triplet
import random

import numpy as np
import torch
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt

from itertools import combinations

import pickle


def generate_random_patch(image_tensor, patch_size):
    image_height = image_tensor.shape[1]
    top_edge = np.random.randint(0, image_height - patch_size)
    image_width = image_tensor.shape[2]
    left_edge = np.random.randint(0, image_width - patch_size)
    patch = image_tensor[:, top_edge:top_edge + patch_size,
            left_edge:left_edge + patch_size]
    return patch


def visualize_image(image):
    visual = make_lupton_rgb(image[0], image[1], image[2], stretch=1)
    plt.figure(figsize=(1,1))
    plt.imshow(visual)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_image_visualization(image, filename):
    with open(filename, 'ab') as file:
        pickle.dump(image, file)


def visualize_image_from_file(filename):
    with open(filename, 'rb') as file:
        while True:
            try:
                image = pickle.load(file)
                visualize_image(image)
            except (EOFError):
                break



def visualize_image2(image):
    visual = make_lupton_rgb(image[0], image[1], image[2], stretch=1)
    plt.figure(figsize=(1,1))
    plt.imshow(visual)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_image_from_file2(filename):
    with open(filename, 'rb') as file:
      images = []
      while True:
        try:
          images.append(pickle.load(file))
        except (EOFError):
          break

    # fig, axes = plt.subplots(nrows=12, ncols=12,figsize = (8,8))
    fig, axes = plt.subplots(nrows=12, ncols=12,figsize = (16,16))


    for i, ax in enumerate(axes.flatten()):
      if i < len(images):
        image = images[i]
        visual = make_lupton_rgb(image[0], image[1], image[2], stretch=1)
        ax.imshow(visual)
      ax.axis('off')

    fig.tight_layout()
    plt.show()


def normalize_01(tensor):
    return tensor / torch.max(torch.abs(tensor)) / 2 + 0.5


def pdist(vectors):
    a0 = vectors
    a1 = torch.t(vectors)
    a = -2 * vectors.mm(torch.t(vectors))
    b0 = vectors.pow(2)
    b = vectors.pow(2).sum(dim=1).view(1, -1)
    c = vectors.pow(2).sum(dim=1).view(-1, 1)
    d = a + b
    d1 = a + c
    d2 = a+b+c

    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def pdist_2(vectors):
    #Matrix mult/work from scratch what you think makes sense for the semihard mining
    pass


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

def random_selection(loss_values):
    hard_negatives = np.where(loss_values > -np.inf)[0]
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

        last_label = None

        for label in set(list(labels.flatten())):
            # Ensure only one iteration through each downsampled image to avoid
            # sample correlation
            # if label == last_label:
            #     continue
            # else:
            #     last_label = label

            # label_mask = (labels == label)
            # label_indices = np.where(label_mask)[0]

            # negative_indices = np.where(np.logical_not(label_mask))[0]
            # anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            # anchor_positives = np.array(anchor_positives)

            # ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]

            # # Select a random pair instead of doing all positive combinations
            # rand_index = random.randint(0, len(ap_distances) - 1)

            # anchor_positive = anchor_positives[rand_index]
            # ap_distance = ap_distances[rand_index]

            # # Calculate loss on the embeddings
            # loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
            # loss_values = loss_values.data.cpu().numpy()
            # hard_negative = self.negative_selection_fn(loss_values)
            # if hard_negative is not None:
            #     hard_negative = negative_indices[hard_negative]
            #     triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])



            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
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

def RandomTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=random_selection,
                                           cpu=cpu)

def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=lambda x: semihard_negative(x, margin),
                                           cpu=cpu)


