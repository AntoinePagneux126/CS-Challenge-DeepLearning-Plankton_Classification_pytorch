import torch
import torch.nn as nn
import torch.nn.init
import torchvision
import numpy as np


class PerceptronNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(PerceptronNet, self).__init__()
        self.input_size = input_size
        self.classifier = nn.Linear(self.input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


class LinearNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(LinearNet, self).__init__()

        self.layer_1 = nn.Linear(input_size, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


class CNNet(nn.Module):
    def __init__(self, num_classes: int, crop_size: int):
        super(CNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        probe_tensor = torch.zeros((4, 3, crop_size, crop_size))
        dummy_output = self.net(probe_tensor)
        out_size = np.prod(dummy_output.shape[1:])

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(out_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size()[0], -1)
        return self.classifier(x)


class ResNet34(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone=False, pretrained=True):
        super(ResNet34, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=pretrained)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        # print("Out features number: ",self.model.fc.in_features)
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features, out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone=False, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        # print("Out features number: ",self.model.fc.in_features)
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features, out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)


class ResNet50T(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ResNet152(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False, pretrained=True):
        super(ResNet152, self).__init__()
        self.model = torchvision.models.resnet152(pretrained=pretrained)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        # print("Out features number: ",self.model.fc.in_features)
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features, out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)
