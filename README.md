# Layer-Wise Contrastive Unsupervised Representation Learning

This repository contains code for the extended abstract published in RUCS <link>.

To learn the unsupervised filters, run Experiments_CIFAR100_1Layer.py. 

To train a classifier from scratch (for supervised transfer), run Classifier_CIFAR100.py. 

To then transfer and test learned filters, run Experiments_CIFAR10_FrozenConv.py.

Hyperparameters are within the respective files above; network structure hyperparameters are in networks.py.
