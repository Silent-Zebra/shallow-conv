# Adapted from https://github.com/adambielski/siamese-triplet

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmbeddingNet(nn.Module):
    def __init__(self):
        # TODO make these inputable/editable parameters
        super(EmbeddingNet, self).__init__()
        input_depth = 3
        layer1_stride = 4
        layer1_kernel_size = 8
        layer1_output_channels = 64
        self.num_filters = layer1_output_channels
        layer1_padding = 0

        self.convnet = nn.Sequential(
            nn.Conv2d(input_depth, layer1_output_channels, layer1_kernel_size,
                      layer1_stride, layer1_padding),
            # nn.LeakyReLU()
        )

    def forward(self, x):
        output = self.convnet(x)
        return output.squeeze()

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class OnlineTripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(OnlineTripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        return self.embedding_net(x)

    def get_embedding(self, x):
        return self.embedding_net(x)
