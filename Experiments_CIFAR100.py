# Adapted from https://github.com/adambielski/siamese-triplet

from torchvision.datasets import CIFAR100
from torchvision import transforms
import utils

train_dataset = CIFAR100('./data/CIFAR100', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 # transforms.Normalize((mean,), (std,))
                             ]))
test_dataset = CIFAR100('./data/CIFAR100', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((mean,), (std,))
                            ]))

import torch
from torch.optim import lr_scheduler
import torch.optim as optim

from trainer import fit
cuda = torch.cuda.is_available()

from datasets import TripletCIFAR100

patch_size = 8

triplet_train_dataset = TripletCIFAR100(train_dataset, patch_size) # Returns triplets of images
triplet_test_dataset = TripletCIFAR100(test_dataset, patch_size)
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss

margin = 1.
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 1
log_interval = 50

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


visualize_filter = True
if visualize_filter:
    for filter in list(model.embedding_net.convnet.parameters())[0]:
        filter = utils.normalize_01(filter)
        utils.visualize_image(filter.detach().numpy())
