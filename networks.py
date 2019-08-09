# Adapted from https://github.com/adambielski/siamese-triplet

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassifierCNN(nn.Module):
    def __init__(self, input_size, input_depth, layer1_stride, layer1_kernel_size,
                 layer1_output_channels, layer1_padding, output_size):
        super(ClassifierCNN, self).__init__()

        self.num_filters = layer1_output_channels

        maxpool_size = 2
        maxpool_stride = maxpool_size

        layer1_output_size = int((int(input_size) - layer1_kernel_size + 2*layer1_padding) / layer1_stride + 1)
        # after maxpool
        layer1_output_size = int((layer1_output_size - (maxpool_size - 1) - 1)/maxpool_stride) + 1

        layer2_kernel_size = 3
        layer2_stride = 1
        layer2_padding = 0

        layer2_output_size = int((int(layer1_output_size) - layer2_kernel_size + 2*layer2_padding) / layer2_stride + 1)
        # after maxpool
        layer2_output_size = int(
            (layer2_output_size - (maxpool_size - 1) - 1) / maxpool_stride) + 1

        self.convnet = nn.Sequential(
            nn.Conv2d(input_depth, layer1_output_channels, layer1_kernel_size,
                      layer1_stride, layer1_padding),
            nn.MaxPool2d(maxpool_size, maxpool_stride),
            nn.BatchNorm2d(layer1_output_channels),
            nn.LeakyReLU(),
            nn.Conv2d(layer1_output_channels, layer1_output_channels,
                      layer2_kernel_size,
                      layer2_stride, layer2_padding),
            nn.MaxPool2d(maxpool_size, maxpool_stride),
            nn.BatchNorm2d(layer1_output_channels),
            nn.LeakyReLU(),
        )
        # layer3_kernel_size = 3
        # layer3_stride = 1
        # layer3_padding = 0
        #
        # layer3_output_size = int((int(
        #     layer2_output_size) - layer3_kernel_size + 2 * layer3_padding) / layer3_stride + 1)
        #
        # self.convnet = nn.Sequential(
        #     nn.Conv2d(input_depth, layer1_output_channels, layer1_kernel_size,
        #               layer1_stride, layer1_padding),
        #     nn.MaxPool2d(maxpool_size, maxpool_stride),
        #     nn.BatchNorm2d(layer1_output_channels),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(layer1_output_channels, layer1_output_channels, layer2_kernel_size,
        #               layer2_stride, layer2_padding),
        #     nn.MaxPool2d(maxpool_size, maxpool_stride),
        #     nn.BatchNorm2d(layer1_output_channels),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(layer1_output_channels, layer1_output_channels,
        #               layer3_kernel_size,
        #               layer3_stride, layer3_padding),
        #     nn.BatchNorm2d(layer1_output_channels),
        #     nn.LeakyReLU(),
        # )

        hidden_units = 512

        self.fc = nn.Sequential(
            nn.Linear(layer2_output_size**2 * layer1_output_channels, hidden_units),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, output_size),
            nn.Softmax(dim=1)
        )

        self.output_size = output_size


    def forward(self, x):
        output = self.convnet(x)
        output = output.view(-1, output.shape[1] * output.shape[2]**2)
        output = self.fc(output)
        return output


class EmbeddingNet(nn.Module):
    def __init__(self, input_depth, layer1_stride, layer1_kernel_size,
                 layer1_output_channels, layer1_padding=0):
        super(EmbeddingNet, self).__init__()

        self.num_filters = layer1_output_channels

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
