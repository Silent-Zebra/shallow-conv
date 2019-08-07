# Source: https://github.com/adambielski/siamese-triplet

import torch
import numpy as np
import utils
import torch.nn.functional as F

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def reshape_outputs_and_create_labels(outputs):
    outputs = outputs.view(
        (outputs.shape[0], outputs.shape[1],
         outputs.shape[2] ** 2)).permute(0, 2, 1).contiguous()

    # print(outputs.shape)

    end = outputs.shape[0] * outputs.shape[1]
    target = torch.arange(0, end=end) / outputs.shape[1]
    target = target.view((outputs.shape[0], outputs.shape[1], 1))

    # print(target.shape)
    # print(target)

    target = target.view(-1, 1)

    outputs = outputs.view(-1, outputs.shape[2])

    return outputs, target


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)

        # print(target)
        # print(data[0].shape)

        optimizer.zero_grad()
        outputs = model(*data)

        # # print(outputs.shape)
        #
        # outputs = outputs.view((outputs.shape[0], outputs.shape[1], outputs.shape[2]**2)).permute(0, 2, 1).contiguous()
        #
        # # print(outputs.shape)
        #
        # end = outputs.shape[0] * outputs.shape[1]
        # target = torch.arange(0, end=end) / outputs.shape[1]
        # target = target.view((outputs.shape[0], outputs.shape[1], 1))
        #
        # # print(target.shape)
        # # print(target)
        #
        #
        # target = target.view(-1, 1)
        #
        # outputs = outputs.view(-1, outputs.shape[2])
        # # print(outputs.shape)
        # # print(target.shape)

        outputs, target = reshape_outputs_and_create_labels(outputs)

        if cuda:
            if target is not None:
                target = target.cuda()


        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        # print(loss_inputs)

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []


            visualize_difference(model, data)


        # early_stop = 20000
        # if batch_idx* len(data[0]) > early_stop:
        #     print("early stop")
        #     total_loss /= (batch_idx + 1)
        #     return total_loss, metrics



    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)

            outputs = model(*data)

            outputs, target = reshape_outputs_and_create_labels(outputs)

            if cuda:
                if target is not None:
                    target = target.cuda()

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

            early_stop = 5000
            if batch_idx * len(data[0]) > early_stop:
                visualize_difference(model, data)

                print("early stop")
                return val_loss, metrics

    return val_loss, metrics

def reshape_conv_embedding(embedding1):
    embedding1 = embedding1.view(-1, embedding1.shape[1] ** 2)
    embedding1 = embedding1.permute(1, 0)
    return embedding1


def get_cosine_loss_individual(model, image, image_2, verbose=True):
    # For testing only
    num_filters = model.embedding_net.num_filters
    embedding1 = model(image.unsqueeze(dim=0))
    embedding1 = reshape_conv_embedding(embedding1)
    print(embedding1.shape)
    # embedding1 =
    # view(1, num_filters)
    embedding2 = model(image_2.unsqueeze(dim=0))
    embedding2 = reshape_conv_embedding(embedding2)

    if verbose:
        print(embedding1[0])
        print(embedding1[1])
        print(embedding2[0])
        print(embedding2[1])
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        print(cos(embedding1[0], embedding1[1]))
        print(cos(embedding2[0], embedding2[1]))
        print(cos(embedding1[0], embedding2[0]))


def visualize_difference(model, data):
    # For testing only
    input_visualize = 0
    for i in range(input_visualize):
        img1 = data[0][i]
        img2 = data[0][i + input_visualize]
        utils.visualize_image(img1)
        utils.visualize_image(img2)
        print(model(img1.unsqueeze(dim=0)).shape)
        get_cosine_loss_individual(model, data[0][i],
                                   data[0][i + input_visualize],
                                   verbose=True)
