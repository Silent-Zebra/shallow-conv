import numpy as np
import utils

from torch.utils.data import Dataset

class FirstLayerProcessedCIFAR100(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, CIFAR100_dataset, frozen_conv_network):
        self.dataset = CIFAR100_dataset
        self.frozen_conv_network = frozen_conv_network

    def __getitem__(self, index):
        embedding = self.frozen_conv_network(self.dataset[index][0].unsqueeze(dim=0))
        return embedding, self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)
