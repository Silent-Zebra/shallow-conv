# Adapted from https://github.com/adambielski/siamese-triplet

# --- HYPERPARAMETERS ---

# image size to downsample to
downsampled_size = 5

batch_size = 512

# margin for triplet loss function
margin = 2.

n_epochs = 100
# log every x batches
log_interval = 10

patch_size = 4
patch_stride = 1

# Convnet hyperparameters
lr = 1e-3
input_depth = 3
layer1_stride = 1
layer1_kernel_size = 3
layer1_output_channels = 256
layer1_padding = 0
use_relu = True

frozen_layer1_stride = 1
frozen_layer1_kernel_size = 6
frozen_layer1_output_channels = 256
frozen_layer1_padding = 0
frozen_use_relu = use_relu

# frozen_patch_size = 7
# frozen_patch_stride = 1
# frozen_downsampled_size = 8

use_pooling = True

# *2 for maxpool
if use_pooling:
    crop_size = downsampled_size * 2 + (frozen_layer1_kernel_size - 1)
else:
    crop_size = downsampled_size + (frozen_layer1_kernel_size - 1)

visualize_filter = True

# Number of examples to visualize and see how the network embeds
visualize_model_working = 0


from torchvision.datasets import CIFAR100
from torchvision import transforms
import utils
import torch
from datasets import FirstLayerProcessedCIFAR100

train_dataset = CIFAR100('./data/CIFAR100', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(crop_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]))
test_dataset = CIFAR100('./data/CIFAR100', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(crop_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                            ]))


cuda = torch.cuda.is_available()

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}


import torch.optim as optim
from trainer import fit
from networks import EmbeddingNet, TripletNet, OnlineTripletNet, ConvEmbeddingNet, EmbeddingNetWithPooling
from losses import TripletLoss, OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, \
    RandomNegativeTripletSelector, SemihardNegativeTripletSelector, RandomTripletSelector

if use_pooling:
    embedding_net = EmbeddingNetWithPooling(input_depth=layer1_output_channels,
                                 layer1_stride=layer1_stride,
                                 layer1_kernel_size=layer1_kernel_size,
                                 layer1_output_channels=layer1_output_channels,
                                 layer1_padding=layer1_padding,
                                 use_relu=use_relu)
else:
    embedding_net = EmbeddingNet(input_depth=layer1_output_channels,
                                 layer1_stride=layer1_stride,
                                 layer1_kernel_size=layer1_kernel_size,
                                 layer1_output_channels=layer1_output_channels,
                                 layer1_padding=layer1_padding,
                                 use_relu=use_relu)
model = ConvEmbeddingNet(embedding_net=embedding_net, patch_size=patch_size,
                         patch_stride=patch_stride, input_size=downsampled_size)
if cuda:
    model.cuda()

frozen_embedding_net = EmbeddingNet(input_depth=input_depth,
                                 layer1_stride=frozen_layer1_stride,
                                 layer1_kernel_size=frozen_layer1_kernel_size,
                                 layer1_output_channels=frozen_layer1_output_channels,
                                 layer1_padding=frozen_layer1_padding,
                                 use_relu=frozen_use_relu)


# Load 1 conv layer
frozen_embedding_net.convnet[0].load_state_dict(torch.load("model_unsupervised_1l.pt", map_location="cpu"))

# Freeze weights of that layer
for param in frozen_embedding_net.convnet[0].parameters():
    param.requires_grad = False


train_dataset = FirstLayerProcessedCIFAR100(train_dataset, frozen_embedding_net)

test_dataset = FirstLayerProcessedCIFAR100(test_dataset, frozen_embedding_net)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
val_loss_fn = OnlineTripletLoss(margin, RandomTripletSelector(margin))

optimizer = optim.Adam(model.parameters(), lr=lr)

# learning rate decay over epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, n_epochs // 1.5, gamma=0.1)

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler,
    n_epochs, cuda, log_interval, visualize_workings=visualize_model_working, val_loss_fn=val_loss_fn)


if visualize_filter:
    if use_pooling:
        visualization_filename = "visualization_unsupervised_2ndl"
    else:
        visualization_filename = "visualization_unsupervised_2ndl_nopool"
    # Reset
    open(visualization_filename, 'w').close()

    for filter in list(model.embedding_net.convnet.parameters())[0]:
        filter = utils.normalize_01(filter)
        utils.save_image_visualization(filter.detach().cpu().numpy(),
                                       filename=visualization_filename)
if use_pooling:
    filename = 'model_unsupervised_2ndl.pt'
else:
    filename = 'model_unsupervised_2ndl_nopool.pt'

torch.save(model.embedding_net.convnet[0].state_dict(), filename)
