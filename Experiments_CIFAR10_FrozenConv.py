# Adapted from https://github.com/adambielski/siamese-triplet

# --- HYPERPARAMETERS ---

# Test against a random feature baseline
random_features = False
# Load the supervised filters instead of unsupervised
load_supervised = False
load_one_layer = True
# Freeze layers means no fine-tuning
freeze_layers = False

batch_size = 128

n_epochs = 100
# log every x batches
log_interval = 100

# Convnet hyperparameters
lr = 1e-3
input_depth = 3
layer1_stride = 1
layer1_kernel_size = 6
layer1_output_channels = 256
layer1_padding = 0

layer2_output_channels = 512

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


from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit, fit_classifier
from networks import EmbeddingNet, TripletNet, OnlineTripletNet, ClassifierCNN


inputs, classes = next(iter(train_loader))

input_size = inputs.shape[2]
output_size = 10

model = ClassifierCNN(input_size=input_size, input_depth=input_depth,
                      layer1_stride=layer1_stride,
                      layer1_kernel_size=layer1_kernel_size,
                      layer1_output_channels=layer1_output_channels,
                      layer1_padding=layer1_padding,
                      output_size=output_size,
                      layer2_output_channels=layer2_output_channels)


def load_trained_model(trained_model, model):
    conv_layers = []
    for layer in trained_model.convnet:
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers.append(layer)
    i = 0
    for layer in model.convnet:
        if isinstance(layer, torch.nn.Conv2d):
            layer.load_state_dict(conv_layers[i].state_dict())
            i += 1


# Load 1 conv layer
if not random_features:
    if load_supervised:

        trained_model = torch.load("model_supervised.pt", map_location="cpu")
        if load_one_layer:
            model.convnet[0].load_state_dict(trained_model.convnet[0].state_dict())
        else:
            load_trained_model(trained_model, model)

    else:
        if load_one_layer:
            model.convnet[0].load_state_dict(torch.load("model_unsupervised_1l.pt", map_location="cpu"))
        else:
            i = 0
            for layer in model.convnet:
                if isinstance(layer, torch.nn.Conv2d):
                    if i == 0:
                        layer.load_state_dict(torch.load("model_unsupervised_1l.pt", map_location="cpu"))
                    elif i == 1:
                        layer.load_state_dict(torch.load("model_unsupervised_2ndl.pt", map_location="cpu"))

                    i += 1


# Freeze weights of that layer
if freeze_layers:
    if load_one_layer:
        for param in model.convnet[0].parameters():
            param.requires_grad = False
    else:
        for layer in model.convnet:
            if isinstance(layer, torch.nn.Conv2d):
                for param in layer.parameters():
                    param.requires_grad = False

if cuda:
    model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# learning rate decay over epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, n_epochs // 1.5, gamma=0.1)

fit_classifier(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, visualize_workings=visualize_model_working)

