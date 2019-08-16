# Source: https://github.com/adambielski/siamese-triplet
import time

import torch
import numpy as np
import utils
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, visualize_workings=0, val_loss_fn=None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process targets output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    if val_loss_fn is None:
      val_loss_fn = loss_fn

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, visualize_workings)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, val_loss_fn, cuda, metrics, visualize_workings)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def reshape_outputs_and_create_labels(outputs):
    outputs = outputs.view(
        (outputs.shape[0], outputs.shape[1],
         outputs.shape[2] * outputs.shape[3])).permute(0, 2, 1).contiguous()

    end = outputs.shape[0] * outputs.shape[1]
    targets = torch.arange(0, end=end) / outputs.shape[1]
    targets = targets.view((outputs.shape[0], outputs.shape[1], 1))
    targets = targets.view(-1, 1)

    outputs = outputs.view(-1, outputs.shape[2])

    return outputs, targets


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval,
                metrics, visualize_workings, with_labels=False):

    # TODO combine duplicate code in test and train epochs

    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    accuracy_record = []
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(train_loader):

        targets = targets if len(targets) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)

        optimizer.zero_grad()
        outputs = model(*data)

        if not with_labels:
            outputs, targets = reshape_outputs_and_create_labels(outputs)

        if cuda:
            if targets is not None:
                targets = targets.cuda()


        if with_labels:
            predictions = torch.argmax(outputs, dim=1)

            # correct = predictions==torch.argmax(targets, dim=1)
            correct = predictions==targets

            accuracy = float(sum(correct)) / float(len(correct))

            accuracy_record.append(accuracy)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if targets is not None:
            targets = (targets,)
            loss_inputs += targets

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, targets, loss_outputs)

        # for parameter in model.convnet[0].parameters():
        #     print(parameter)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            if with_labels:
                print("Average Train Accuracy: " + str(np.mean(accuracy_record)))

            print(message)
            losses = []

            visualize_difference(model, data, visualize_workings)

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics, visualize_workings,
               with_labels=False):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        accuracy_record = []

        for batch_idx, (data, targets) in enumerate(val_loader):
            targets = targets if len(targets) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)

            outputs = model(*data)

            if not with_labels:
                outputs, targets = reshape_outputs_and_create_labels(outputs)

            if cuda:
                if targets is not None:
                    targets = targets.cuda()

            if with_labels:
                predictions = torch.argmax(outputs, dim=1)

                correct = predictions == targets

                accuracy = float(sum(correct)) / float(len(correct))

                accuracy_record.append(accuracy)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if targets is not None:
                targets = (targets,)
                loss_inputs += targets

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()


            for metric in metrics:
                metric(outputs, targets, loss_outputs)

    if with_labels:
        return val_loss, metrics, np.mean(accuracy_record)
    else:
        return val_loss, metrics


def reshape_conv_embedding(embedding1):
    embedding1 = embedding1.view(-1, embedding1.shape[1] ** 2)
    embedding1 = embedding1.permute(1, 0)
    return embedding1


def get_cosine_loss_individual(model, image, image_2, verbose=True):
    # For testing only
    embedding1 = model(image.unsqueeze(dim=0))
    embedding1 = reshape_conv_embedding(embedding1)
    embedding2 = model(image_2.unsqueeze(dim=0))
    embedding2 = reshape_conv_embedding(embedding2)

    if verbose:
        print("Image1 First Patch Embedding:")
        print(embedding1[0])
        print("Image1 Second Patch Embedding:")
        print(embedding1[1])
        print("Image2 First Patch Embedding:")
        print(embedding2[0])
        print("Image2 Second Patch Embedding:")
        print(embedding2[1])
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        print("Cosine Similarity Image1 Patch 1 and Patch 2: " + str(cos(embedding1[0], embedding1[1]).item()))
        print("Cosine Similarity Image2 Patch 1 and Patch 2: " + str(cos(embedding2[0], embedding2[1]).item()))
        print("Cosine Similarity Image1 Patch 1 and Image2 Patch 1: " + str(cos(embedding1[0], embedding2[0]).item()))


def visualize_difference(model, data, visualize_workings=0):
    # For testing only
    input_visualize = visualize_workings
    for i in range(input_visualize):
        img1 = data[0][i]
        img2 = data[0][i + input_visualize]
        utils.visualize_image(img1)
        utils.visualize_image(img2)
        print(model(img1.unsqueeze(dim=0)).shape)
        get_cosine_loss_individual(model, data[0][i],
                                   data[0][i + input_visualize],
                                   verbose=True)



def fit_classifier(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, visualize_workings=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process targets output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    writer = SummaryWriter()

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer,
                                          cuda, log_interval, metrics, visualize_workings, with_labels=True)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics, accuracy = test_epoch(val_loader, model, loss_fn, cuda, metrics,
                                       visualize_workings, with_labels=True)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)

        print("Validation Average Accuracy: " + str(accuracy))

        writer.add_scalar("Validation Average Accuracy", accuracy, epoch)

    writer.close()
