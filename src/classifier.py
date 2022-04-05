from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
from torch.utils.data import WeightedRandomSampler


from tqdm import tqdm
import os
import math
import pandas as pd
from collections import Counter

from torch.utils.tensorboard import SummaryWriter

## Base Class for classification
from imgloader import PlankDataset
from configuration import config_deeplearing
from models import (
    CNNet,
    ResNet34,
    ResNet50,
    ResNet50T,
    PerceptronNet,
    LinearNet,
    ResNet152,
)
from utils import f1_torch

my_config = config_deeplearing()


## INPUTS
PATH = my_config["PATH"]["DATA_PATH"]
TRAIN_FOLDER = my_config["PATH"]["TRAIN_FOLDER"]
TEST_FOLDER = my_config["PATH"]["TEST_FOLDER"]
CHECKPOINT_PATH = my_config["PATH"]["CHECKPOINT_PATH"]
TSBOARD_PATH = my_config["PATH"]["TSBOARD_PATH"]
BATCH_SIZE = int(my_config["PARAMETERS"]["BATCH_SIZE"])
VAL_SIZE_PROP = float(my_config["PARAMETERS"]["VAL_SIZE_PROP"])
NUM_WORKER = int(my_config["PARAMETERS"]["NUM_WORKER"])
LR = float(my_config["PARAMETERS"]["LR"])
NUMBER_EPOCHS = int(my_config["PARAMETERS"]["NUMBER_EPOCHS"])
CROP_SIZE = int(my_config["PARAMETERS"]["CROP_SIZE"])
NUMBER_OUTPUTS = int(my_config["PARAMETERS"]["NUMBER_OUTPUTS"])
PATIENCE = int(my_config["PARAMETERS"]["PATIENCE"])
DELTA = int(my_config["PARAMETERS"]["DELTA"])
UNFREEZE = int(my_config["PARAMETERS"]["UNFREEZE"])
## GLOBALS
NUMBER_INPUTS = 3 * CROP_SIZE * CROP_SIZE
TRAIN_PATH = PATH + TRAIN_FOLDER
TEST_PATH = PATH + TEST_FOLDER


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(
        self,
        patience: int = PATIENCE,
        delta: int = DELTA,
        path: str = CHECKPOINT_PATH + "checkpoint_test.pt",
    ):
        """[Constructor]

        Args:
            patience (int, optional): [value for patience]. Defaults to PATIENCE.
            delta (int, optional): [value for delta]. Defaults to DELTA.
            path (str, optional): [path to save chackpoints]. Defaults to CHECKPOINT_PATH+'checkpoint_test.pt'.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        """[Call]

        Args:
            val_loss ([float]): [Value of the loss]
            model ([type]): [model used]
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)  # Save model
            # reset counter if validation loss improves
            self.counter = 0

    def save_checkpoint(self, model):
        """[Save the best result of the pytoch model during the current training]

        Args:
            model ([type]): [pytorch model to save]
        """
        torch.save(model.state_dict(), self.path)


class Classifier:
    def __init__(
        self,
        data_dir: str,
        num_classes: int,
        device,
        Transform=None,
        sample=True,
        loss_weights=False,
        batch_size=BATCH_SIZE,
        lr=LR,
        stop_early=True,
        freeze_backbone=True,
        pretrained=True,
        f1_loss=False,
    ):
        """[Constructor]

        Args:
            data_dir ([str]): [Path to the data]
            num_classes ([int]): [Number of classes of the problem]
            device ([type]): [Pytorch device (cpu or GPU)]
            Transform ([type], optional): [Transform to apply to the dataset]. Defaults to None.
            sample (bool, optional): [If you want to use sampling]. Defaults to False.
            loss_weights (bool, optional): [If you want to take into account loss weight]. Defaults to False.
            batch_size ([int], optional): [Size of batch]. Defaults to BATCH_SIZE.
            lr ([type], optional): [Learning rate for the optimizer]. Defaults to LR.
            stop_early (bool, optional): [If you want to use early stopping : stop the training if there is no evolution after a given number of training]. Defaults to True.
            freeze_backbone (bool, optional): [If you want to freeze the parameters of a pre-trained model when you use transfert learning]. Defaults to True.
            pretrained (bool, optional): [If you want to use a pre-trained model when you are doing transfert learning]. Defaults to True.
        """
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.device = device
        self.sample = sample
        self.loss_weights = loss_weights
        self.batch_size = batch_size
        self.lr = lr
        self.stop_early = stop_early
        self.freeze_backbone = freeze_backbone
        self.Transform = Transform
        self.pretrained = pretrained
        self.name = ""
        self.f1 = f1_loss

    def load_data(self):
        """[Load the dataset]

        Returns:
            [type]: [The train loader and the validation loader]
        """
        dataset = PlankDataset(self.data_dir, transform=self.Transform)
        val_size = int(len(dataset) * VAL_SIZE_PROP)
        train_size = len(dataset) - val_size
        trainset, valset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        if self.sample:
            y_train_indices, y_val_indices = trainset.indices, valset.indices

            y_train = [dataset.data.targets[i] for i in y_train_indices]
            y_val = [dataset.data.targets[i] for i in y_val_indices]

            class_sample_count_train = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
            )
            class_sample_count_val = np.array(
                [len(np.where(y_val == t)[0]) for t in np.unique(y_val)]
            )

            weight_train = 1.0 / class_sample_count_train
            weight_val = 1.0 / class_sample_count_val

            samples_weight_train = np.array([weight_train[t] for t in y_train])
            samples_weight_val = np.array([weight_val[t] for t in y_val])

            samples_weight_train = torch.from_numpy(samples_weight_train)
            samples_weight_val = torch.from_numpy(samples_weight_val)

            train_sampler = WeightedRandomSampler(
                samples_weight_train.type("torch.DoubleTensor"),
                len(samples_weight_train),
            )
            val_sampler = WeightedRandomSampler(
                samples_weight_val.type("torch.DoubleTensor"), len(samples_weight_val)
            )

            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=NUM_WORKER,
                pin_memory=True,
                sampler=train_sampler,
            )
            val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=NUM_WORKER,
                pin_memory=True,
                sampler=val_sampler,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=self.batch_size,
                num_workers=NUM_WORKER,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                valset,
                batch_size=self.batch_size,
                num_workers=NUM_WORKER,
                pin_memory=True,
            )

        return train_loader, val_loader

    def load_model(self, model_type="cnn"):
        """[Load a model (e.g. resnet50, cnn, mlp, p...)]

        Args:
            model_type (str, optional): [description]. Defaults to 'cnn'.
        """
        self.name = model_type
        # Load the right model from models.py
        if model_type == "resnet50":
            self.model = ResNet50(
                num_classes=self.num_classes,
                freeze_backbone=self.freeze_backbone,
                pretrained=self.pretrained,
            )

        elif model_type == "R50":
            self.model = ResNet50T(NUMBER_OUTPUTS)

        elif model_type == "resnet34":
            self.model = ResNet34(
                num_classes=self.num_classes,
                freeze_backbone=self.freeze_backbone,
                pretrained=self.pretrained,
            )

        elif model_type == "resnet152":
            self.model = ResNet152(
                num_classes=self.num_classes,
                freeze_backbone=self.freeze_backbone,
                pretrained=self.pretrained,
            )

        elif model_type == "cnn":
            self.model = CNNet(self.num_classes, CROP_SIZE)

        elif model_type == "mlp":
            self.model = LinearNet(NUMBER_INPUTS, self.num_classes)

        elif model_type == "p":
            self.model = PerceptronNet(NUMBER_INPUTS, self.num_classes)

        else:  # default
            self.model = CNNet(self.num_classes, CROP_SIZE)

        # Apply the model to the device
        self.model = self.model.to(self.device)
        # Apply Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        # Consider loss weight
        if self.loss_weights:
            class_count = Counter(self.train_classes)
            class_weights = torch.Tensor(
                [
                    len(self.train_classes) / c
                    for c in pd.Series(class_count).sort_index().values
                ]
            )
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        if self.f1 == True:
            self.criterion = f1_torch
            self.criterion.requires_grad = True

    def fit_one_epoch(self, train_loader, epoch, num_epochs):
        """[Method for fitting one epoch]

        Args:
            train_loader ([type]): [the train loader]
            epoch ([type]): [number of the current epoch]
            num_epochs ([type]): [total number of epochs]
        """

        # Lists for losses and accuracies
        train_losses = []
        train_acc = []
        # Turn model into train mode
        self.model.train()
        # Train...
        if self.f1 == True:
            for i, (images, targets) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss.requires_grad = True
                loss.backward()
                self.optimizer.step()

                self.optimizer.zero_grad()

                train_losses.append(loss.item())

                # Calculate accuracy for the current epoch
                predictions = torch.argmax(logits, dim=1)
                num_correct = sum(predictions.eq(targets))
                running_train_acc = float(num_correct) / float(images.shape[0])
                train_acc.append(running_train_acc)
        else:
            for i, (images, targets) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)
                # loss.requires_grad = True
                loss.backward()
                self.optimizer.step()

                self.optimizer.zero_grad()

                train_losses.append(loss.item())

                # Calculate accuracy for the current epoch
                predictions = torch.argmax(logits, dim=1)
                num_correct = sum(predictions.eq(targets))
                running_train_acc = float(num_correct) / float(images.shape[0])
                train_acc.append(running_train_acc)

        train_loss = torch.tensor(train_losses).mean()
        print(f"Epoch n° {epoch}/{num_epochs-1}")
        print(f"Training Loss: {train_loss:.2f}")
        return train_loss

    def val_one_epoch(self, val_loader):
        """[Method for validating one epoch]

        Args:
            val_loader ([type]): [the validation loader]
        """
        # Lists for losses and accuracies
        val_losses = []
        val_accs = []
        # Turn model into eval mode
        self.model.eval()
        # Disabling gradient descent for evaluation
        with torch.no_grad():
            for (images, targets) in tqdm(val_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)
                val_losses.append(loss.item())

                predictions = torch.argmax(logits, dim=1)
                num_correct = sum(predictions.eq(targets))
                # num_not_correct = int(sum(predictions.not_equal(targets)))
                running_val_acc = float(num_correct) / float(images.shape[0])
                val_accs.append(running_val_acc)

            self.val_loss = torch.tensor(val_losses).mean()
            val_acc = torch.tensor(val_accs).mean()  # Average acc per batch

            print(f"Validation loss: {self.val_loss:.2f}")
            print(f"Validation accuracy: {val_acc:.2f}")
        return val_acc

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs=NUMBER_EPOCHS,
        unfreeze_after=UNFREEZE,
        checkpoint_dir=CHECKPOINT_PATH,
    ):
        """[Method to fit a model : apply training et validation for a given number of epochs]

        Args:
            train_loader ([type]): [description]
            val_loader ([type]): [description]
            num_epochs ([int], optional): [number of epochs for fitting]. Defaults to NUMBER_EPOCHS.
            unfreeze_after ([int], optional): [parameter for unfreezing parameters of a pre-trained model when doing transfert learining]. Defaults to UNFREEZE.
            checkpoint_dir ([str], optional): [path to the chackpoint folder]. Defaults to CHECKPOINT_PATH.
        """

        # Start a tensorboard for monitoring
        if not os.path.exists(TSBOARD_PATH):
            # Create Tensorboard directory if not exists
            os.mkdir(TSBOARD_PATH)

        # Compute the number of the tensorboard exist from the kind of the model
        ltb = os.listdir(TSBOARD_PATH)
        n = 0
        path_to_tsboard = TSBOARD_PATH
        for tb in ltb:
            if tb.startswith(self.name):
                n += 1
        path_to_tsboard += self.name + "_" + str(n)
        os.mkdir(path_to_tsboard)
        # Instanciate a summary writer
        TensorboardWriter = SummaryWriter(log_dir=path_to_tsboard)

        # If one considers Early Stopping
        if self.stop_early:
            path = checkpoint_dir
            # Check if output path exits
            if not os.path.exists(path):
                print("Creating output path : ", path)
                os.makedirs(path)
            # Compute the number of the cp prediction from the kind of the model
            lcp = os.listdir(path)
            n = 0
            for cp in lcp:
                if cp.startswith(self.name):
                    n += 1
            path += self.name + "_" + str(n) + ".pt"
            early_stopping = EarlyStopping(patience=PATIENCE, path=path)

        # Fit... for num_epochs epochs
        for epoch in range(num_epochs):
            if self.freeze_backbone:
                if epoch == unfreeze_after:  # Unfreeze grad after x epochs
                    for param in self.model.parameters():
                        param.requires_grad = True
            train_loss = self.fit_one_epoch(train_loader, epoch, num_epochs)
            val_acc = self.val_one_epoch(val_loader)
            if self.stop_early:
                early_stopping(self.val_loss, self.model)
                if early_stopping.early_stop:
                    print("Early Stopping")
                    print(f"Best validation loss: {early_stopping.best_score}")
                    break
            # Add metrics to the tensorbaord
            TensorboardWriter.add_scalar("eval loss", float(self.val_loss), epoch)
            TensorboardWriter.add_scalar("train loss", float(train_loss), epoch)
            TensorboardWriter.add_scalar("val acc", float(val_acc), epoch)
