# Adapted from https://github.com/adambielski/siamese-triplet

# --- HYPERPARAMETERS ---

# image size to downsample to
import pickle

downsampled_size = 9

batch_size = 128

# margin for triplet loss function
margin = 1.

n_epochs = 1200
# log every x batches
log_interval = 20

# Convnet hyperparameters
lr = 1e-3
input_depth = 3
layer1_stride = 1
layer1_kernel_size = 8
layer1_output_channels = 64
layer1_padding = 0

visualize_filter = True

# Number of examples to visualize and see how the network embeds
visualize_model_working = 0


from torchvision.datasets import CIFAR100
from torchvision import transforms
import utils
from datasets import DownsampledCIFAR100
import torch


train_dataset = CIFAR100('./data/CIFAR100', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                             ]))
test_dataset = CIFAR100('./data/CIFAR100', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]))

downsampled_train_dataset = DownsampledCIFAR100(train_dataset, downsampled_size=downsampled_size)
downsampled_test_dataset = DownsampledCIFAR100(test_dataset, downsampled_size=downsampled_size)

cuda = torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(downsampled_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(downsampled_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit
from networks import EmbeddingNet, TripletNet, OnlineTripletNet
from losses import TripletLoss, OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, \
    RandomNegativeTripletSelector, SemihardNegativeTripletSelector

embedding_net = EmbeddingNet(input_depth=input_depth,
                             layer1_stride=layer1_stride,
                             layer1_kernel_size=layer1_kernel_size,
                             layer1_output_channels=layer1_output_channels,
                             layer1_padding=layer1_padding)
model = OnlineTripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
optimizer = optim.Adam(model.parameters(), lr=lr)

# learning rate decay over epochs
scheduler = lr_scheduler.StepLR(optimizer, 1000, gamma=0.1, last_epoch=-1)

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, visualize_workings=visualize_model_working)

if visualize_filter:
    visualization_filename = "visualization_unsupervised"
    # Reset
    open(visualization_filename, 'w').close()

    for filter in list(model.embedding_net.convnet.parameters())[0]:
        filter = utils.normalize_01(filter)
        utils.save_image_visualization(filter.detach().cpu().numpy(),
                                       filename=visualization_filename)

    # utils.visualize_image_from_file(filename)

model_filename = "model_unsupervised"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
