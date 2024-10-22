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

use_pooling = False

visualize_filter = True

# Number of examples to visualize and see how the network embeds
visualize_model_working = 0


from torchvision.datasets import CIFAR100
from torchvision import transforms
import utils
import torch

train_dataset = CIFAR100('./data/CIFAR100', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(downsampled_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]))
test_dataset = CIFAR100('./data/CIFAR100', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(downsampled_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                            ]))

cuda = torch.cuda.is_available()

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


import torch.optim as optim
from trainer import fit
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
model = ConvEmbeddingNet(embedding_net=embedding_net, patch_size=patch_size,
                         patch_stride=patch_stride, input_size=downsampled_size)
if cuda:
    model.cuda()

loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
val_loss_fn = OnlineTripletLoss(margin, RandomTripletSelector(margin))

optimizer = optim.Adam(model.parameters(), lr=lr)

# learning rate decay over epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, n_epochs // 1.5, gamma=0.1)

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler,
    n_epochs, cuda, log_interval, visualize_workings=visualize_model_working, val_loss_fn=val_loss_fn)


if visualize_filter:
    if use_pooling:
        visualization_filename = "visualization_unsupervised_1l"
    else:
        visualization_filename = "visualization_unsupervised_1l_nopool"
    # Reset
    open(visualization_filename, 'w').close()

    for filter in list(model.embedding_net.convnet.parameters())[0]:
        filter = utils.normalize_01(filter)
        utils.save_image_visualization(filter.detach().cpu().numpy(),
                                       filename=visualization_filename)
if use_pooling:
    filename = 'model_unsupervised_1l.pt'
else:
    filename = 'model_unsupervised_1l_nopool.pt'

torch.save(model.embedding_net.convnet[0].state_dict(), filename)
