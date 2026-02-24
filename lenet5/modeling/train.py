import math
import collections

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import lenet5.dataset as dataset

Hyper = collections.namedtuple('Hyper', 'name seed model optimizer lr optimizer_kwargs nepochs batch_size random_order', defaults=(1, False))
Result = collections.namedtuple('Result', 'hyper model epochs train_losses test_losses train_accuracies test_accuracies')

class Trainer:
    def __init__(self, training: dataset.Dataset, test: dataset.Dataset,
                 loss_fn = None, one_hot: bool = True, num_classes: int = 10):
        self.training = training
        self.test = test
        self.loss_fn = loss_fn or (lambda pred, target: F.mse_loss(pred, target) * self.num_classes / 2)
        self.one_hot = one_hot
        self.num_classes = num_classes

    def run_scenario(self, hyper: Hyper):
        print(f"Running Pytorch scenario {hyper.name}: model={hyper.model.__name__} seed={hyper.seed} lr={hyper.lr} nepochs={hyper.nepochs}")
        torch.random.manual_seed(hyper.seed)
        model = hyper.model()
        optimizer = hyper.optimizer(model.parameters(), lr=hyper.lr, **hyper.optimizer_kwargs)
        training_features = self.training.features * 2 - 1
        test_features = self.test.features * 2 - 1
        if self.one_hot:
            training_targets = torch.nn.functional.one_hot(self.training.labels, num_classes=self.num_classes).float()*2 - 1
            test_targets = torch.nn.functional.one_hot(self.test.labels, num_classes=self.num_classes).float()*2 - 1
        else:
            training_targets = self.training.labels
            test_targets = self.test.labels

        epochs = []
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        def batch(features_batch, labels_batch):
            optimizer.zero_grad()
            pred = model.forward(features_batch)
            loss = self.loss_fn(pred, labels_batch)
            loss.backward()
            optimizer.step()

        def log_metrics(epoch):
            epochs.append(epoch)

            pred = model.forward(training_features)
            train_losses.append(self.loss_fn(pred, training_targets).item())
            train_accuracies.append((pred.argmax(1)==self.training.labels).sum().item() * 100 / pred.shape[0])

            pred = model.forward(test_features)
            test_losses.append(self.loss_fn(pred, test_targets).item())
            test_accuracies.append((pred.argmax(1)==self.test.labels).sum().item() * 100 / pred.shape[0])

        log_metrics(0)

        for epoch in tqdm(range(hyper.nepochs)):
            order = torch.randperm(training_features.shape[0]) if hyper.random_order else torch.arange(training_features.shape[0])
            training_features_chunks = training_features[order].split(hyper.batch_size)
            training_targets_chunks = training_targets[order].split(hyper.batch_size)
            for i in range(len(training_features_chunks)):
                batch(training_features_chunks[i], training_targets_chunks[i])

            log_metrics(epoch+1)

        return Result(hyper=hyper, model=model, epochs=epochs,
                      train_losses=train_losses, test_losses=test_losses,
                      train_accuracies=train_accuracies, test_accuracies=test_accuracies)