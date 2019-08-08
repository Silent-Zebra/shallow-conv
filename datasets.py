import numpy as np
import utils

from torch.utils.data import Dataset


class DownsampledCIFAR100(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, CIFAR100_dataset, downsampled_size):
        self.dataset = CIFAR100_dataset
        self.downsampled_size = downsampled_size

    def __getitem__(self, index):
        downsampled_img = utils.generate_random_patch(self.dataset[index][0], self.downsampled_size)
        return downsampled_img, []

    def __len__(self):
        return len(self.dataset)
