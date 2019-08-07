# Adapted from https://github.com/adambielski/siamese-triplet

from torchvision.datasets import CIFAR100
from torchvision import transforms
import utils
from datasets import DownsampledCIFAR100



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

# from datasets import TripletCIFAR100

downsampled_size = 16

# triplet_train_dataset = TripletCIFAR100(train_dataset, patch_size) # Returns triplets of images
# triplet_test_dataset = TripletCIFAR100(test_dataset, patch_size)
downsampled_train_dataset = DownsampledCIFAR100(train_dataset, downsampled_size=downsampled_size) # Returns triplets of images
downsampled_test_dataset = DownsampledCIFAR100(test_dataset, downsampled_size=downsampled_size)
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(downsampled_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(downsampled_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, TripletNet, OnlineTripletNet
from losses import TripletLoss, OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, \
    RandomNegativeTripletSelector, SemihardNegativeTripletSelector


margin = 1.
embedding_net = EmbeddingNet()
model = OnlineTripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 1
log_interval = 20

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)



fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


visualize_filter = True
if visualize_filter:
    for filter in list(model.embedding_net.convnet.parameters())[0]:
        filter = utils.normalize_01(filter)
        utils.visualize_image(filter.detach().numpy())
