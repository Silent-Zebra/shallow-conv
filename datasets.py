import numpy as np
import utils

from torch.utils.data import Dataset

class TripletCIFAR100(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, CIFAR100_dataset, patch_size):
        self.dataset = CIFAR100_dataset
        self.patch_size = patch_size


    def __getitem__(self, index):
        patches = utils.generate_patches(self.dataset[index][0], self.patch_size, stride=4)
        img1, img2 = utils.generate_training_example(patches)

        negative_index = index
        while negative_index == index:
            negative_index = np.random.randint(0, len(self))
        img3 = utils.generate_random_patch(self.dataset[negative_index][0], self.patch_size)

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)
