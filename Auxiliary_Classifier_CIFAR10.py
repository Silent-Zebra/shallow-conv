# Adapted from https://github.com/adambielski/siamese-triplet

# --- HYPERPARAMETERS ---

# image size to downsample to
downsampled_size = 8

batch_size = 512

# margin for triplet loss function
margin = 2.

n_epochs = 100
# log every x batches
log_interval = 10

patch_size = 7
patch_stride = 1

# Convnet hyperparameters
lr = 1e-3
input_depth = 3
layer1_stride = 1
layer1_kernel_size = 6
layer1_output_channels = 256
layer1_padding = 0
use_relu = True

# Right now must use, fix later
use_pooling = True

visualize_filter = True

# Number of examples to visualize and see how the network embeds
visualize_model_working = 0


from torchvision.datasets import CIFAR10
from torchvision import transforms
import utils
import torch
import torch.nn as nn


train_dataset = CIFAR10('./data/CIFAR10', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]))
test_dataset = CIFAR10('./data/CIFAR10', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]))

cuda = torch.cuda.is_available()

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

inputs, classes = next(iter(train_loader))

input_size = inputs.shape[2]
output_size = 10

from torch.optim import lr_scheduler
from trainer import fit, fit_classifier, fit_aux_classifier
from networks import AuxiliaryClassifierCNN
import torch.optim as optim
from networks import EmbeddingNet, TripletNet, OnlineTripletNet, ConvEmbeddingNet, EmbeddingNetWithPooling
from losses import TripletLoss, OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, \
    RandomNegativeTripletSelector, SemihardNegativeTripletSelector, RandomTripletSelector

if use_pooling:
    embedding_net = EmbeddingNetWithPooling(input_depth=input_depth,
                                            layer1_stride=layer1_stride,
                                            layer1_kernel_size=layer1_kernel_size,
                                            layer1_output_channels=layer1_output_channels,
                                            layer1_padding=layer1_padding,
                                            use_relu=use_relu)
else:
    embedding_net = EmbeddingNet(input_depth=input_depth,
                                 layer1_stride=layer1_stride,
                                 layer1_kernel_size=layer1_kernel_size,
                                 layer1_output_channels=layer1_output_channels,
                                 layer1_padding=layer1_padding,
                                 use_relu=use_relu)
conv_embedding_net = ConvEmbeddingNet(embedding_net=embedding_net, patch_size=patch_size,
                         patch_stride=patch_stride, input_size=downsampled_size)

model = AuxiliaryClassifierCNN(embedding_net, input_size=input_size, output_size=output_size)

if cuda:
    model.cuda()

#
# optimizer = optim.Adam(model.parameters(), lr=lr)
#
# # learning rate decay over epochs
# scheduler = optim.lr_scheduler.StepLR(optimizer, n_epochs // 1.5, gamma=0.1)
#
# fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler,
#     n_epochs, cuda, log_interval, visualize_workings=visualize_model_working,
#     val_loss_fn=val_loss_fn)
#
# if visualize_filter:
#     if use_pooling:
#         visualization_filename = "visualization_unsupervised_1l"
#     else:
#         visualization_filename = "visualization_unsupervised_1l_nopool"
#     # Reset
#     open(visualization_filename, 'w').close()
#
#     for filter in list(model.embedding_net.convnet.parameters())[0]:
#         filter = utils.normalize_01(filter)
#         utils.save_image_visualization(filter.detach().cpu().numpy(),


classifier_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

aux_loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
aux_val_loss_fn = OnlineTripletLoss(margin, RandomTripletSelector(margin))
# aux_val_loss_fn = OnlineTripletLoss(margin, RandomTripletSelector(margin))

# learning rate decay over epochs
scheduler = lr_scheduler.StepLR(optimizer, n_epochs // 1.5, gamma=0.1, last_epoch=-1)

fit_aux_classifier(train_loader, test_loader, model, classifier_loss_fn, aux_loss_fn, aux_val_loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, visualize_workings=visualize_model_working)

if visualize_filter:
    filename = "visualization_CIFAR10_aux"
    # Reset
    open(filename, 'w').close()

    for filter in list(model.convnet.parameters())[0]:
        filter = utils.normalize_01(filter)
        utils.save_image_visualization(filter.detach().cpu().numpy(), filename=filename)

