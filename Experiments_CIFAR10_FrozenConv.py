# Adapted from https://github.com/adambielski/siamese-triplet

# --- HYPERPARAMETERS ---

# image size to downsample to
# downsampled_size = 16

random_features = True

batch_size = 128

# margin for triplet loss function
# margin = 1.

n_epochs = 500
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


from torchvision.datasets import CIFAR10
from torchvision import transforms
import utils
# from datasets import DownsampledCIFAR100
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

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit, fit_classifier
from networks import EmbeddingNet, TripletNet, OnlineTripletNet, ClassifierCNN


inputs, classes = next(iter(train_loader))

input_size = inputs.shape[2]
output_size = 100

model = ClassifierCNN(input_size=input_size, input_depth=input_depth,
                        layer1_stride=layer1_stride,
                        layer1_kernel_size=layer1_kernel_size,
                        layer1_output_channels=layer1_output_channels,
                        layer1_padding=layer1_padding,
                        output_size=output_size)

# Load 1 conv layer
if not random_features:
    model.convnet[0].load_state_dict(torch.load("model_unsupervised.pt", map_location="cpu"))
# Freeze weights of that layer
for param in model.convnet[0].parameters():
    param.requires_grad = False

if cuda:
    model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# learning rate decay over epochs
scheduler = lr_scheduler.StepLR(optimizer, 500, gamma=0.1, last_epoch=-1)

fit_classifier(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, visualize_workings=visualize_model_working)