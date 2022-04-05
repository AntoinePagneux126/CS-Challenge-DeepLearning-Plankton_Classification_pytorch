"""[Class image data loader implementation]

    Raises:
        Exception: [if transformation is not in the right format]
        Exception: [if index to get item is not a positive int]
        Exception: [if index to get item is not a positive int]
        Exception: [data_part must be a str which represents the path from data folder to desired data set]
        Exception: [data_part must be a str which represents the path from data folder to desired data set]
        Exception: [plot_title must be a str which is the title of the histogram]
        Exception: [data_part must be a str which represent the path from data folder to desired data set]

    Returns:
        [type]: [description]
 """
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch
import torchvision
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, dataloader
import random

# import seaborn as sns


## INPUTS
BATCH_SIZE = 2
VAL_SIZE_PROP = 0.2
PATH = "deepchallenge-cs/data/"
TRAIN_FOLDER = "train_4"
TEST_FOLDER = "test_4"
NUM_WORKER = 0
LR = 1e-3
NUMBER_EPOCHS = 1
CROP_SIZE = 224
NUMBER_INPUTS = 3 * CROP_SIZE * CROP_SIZE
NUMBER_OUTPUTS = 5

## GLOBALS
TRAIN_PATH = PATH + TRAIN_FOLDER
TEST_PATH = PATH + TEST_FOLDER


torch.manual_seed(17)


class ToTensor:
    """[Transform numpy data inputs to torch.tensor]
    """

    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MyRotationTransform:
    """[Rotate by one of the given angles.]"""

    def __init__(self, angles):
        self._angles = angles

    def __call__(self, x):
        angle = random.choice(self._angles)
        return TF.rotate(x, angle)


class MyResizeTransform:
    """[Rotate by one of the given angles.]"""

    def __init__(self, size):
        self._size = size

    def __call__(self):
        return transforms.Resize(self._size, interpolation="bilinear")


class PlankDataset(Dataset):
    """[Planton dataset is a class to create a DataLoader from Kaggle Dataset]

    Args:
        Dataset ([heritage from torch.utils.dataset]): [class Dataset]
    """

    def __init__(
        self,
        path=TRAIN_PATH,
        transform=torchvision.transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    ):
        """[Object attributes definition]

        Args:
            transform ([transforms], optional): [transforms or composed transform applied to
            every images of the dataset]. Defaults to None.
        """
        super(Dataset, self).__init__()

        if not isinstance(transform, (torchvision.transforms.transforms.Compose)):
            raise Exception(
                " transform must be None, torchvision.transforms.transforms.Compose"
            )

        # TODO add is not isinstance path or doesn't exist

        self.path = path
        self.data = datasets.ImageFolder(os.path.join(self.path), transforms.ToTensor())
        self.transform = transform

    def __getitem__(self, index):
        """[get a sample from dataset]

        Args:
            index ([int]): [int index position of the desired sample]

        Returns:
            [tuple(torch.tensor,torch.tensor)]: [image, class encoded]
        """

        if not isinstance(index, (int, float)) or index < 0:
            raise Exception(" index must be an float or int greater than 0")
        sample, target = self.data[index]

        return self.transform(sample), target

    def __len__(self):
        """[get the lenght of the whole dataset]

        Returns:
            [int]: [length of the whole dataset]
        """
        return len(self.data)

    def imshow(self, index):
        """[Display the image corresponding to the index in the training set ]

        Args:
            index ([int]): [index of the image, should be inferior to self.n_train ]
        """
        if not isinstance(index, (int, float)) or index < 0:
            raise Exception(" index must be an float or int greater than 0")

        X_i, y_i = self.data[index]
        inp = X_i.numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        plt.title(f"Index : {index} ; classe : {y_i}")
        plt.show()

    def get_class_distribution(self):
        """[Return the number of examples per class in the dataset]

        Args:
            data_part (str, optional): [path to the desired dataset train or test from data folder]. Defaults to "train/imgs".

        Returns:
            [dict]: [number of element per class]
        """
        idx2class = {v: k for k, v in self.data.class_to_idx.items()}
        count_dict = {k: 0 for k, v in self.data.class_to_idx.items()}
        for _, label_id in self.data:
            label = idx2class[label_id]
            count_dict[label] += 1
        return count_dict

    def plot_from_dict(self, plot_title, **kwargs):
        """[Use a dict to plot an histogram of dataset distribution]

        Args:
            plot_title ([str]): [title of the figure]
            data_part (str, optional): [path to the desired dataset train or test from data folder]. Defaults to "train/imgs".

        Returns:
            [matplotlib.axes._subplots.AxesSubplot]: [Histogram figure of dataset's distribution]
        """
        if not isinstance(plot_title, str):
            raise Exception(
                " plot_title must be a str which is the title of the histogram"
            )
        return sns.barplot(
            data=pd.DataFrame.from_dict([self.get_class_distribution()]).melt(),
            x="variable",
            y="value",
            hue="variable",
            **kwargs,
        ).set_title(plot_title)

    def plot_distrib(self, data_part="train"):
        """[display the distribution histogram of the desired datas set part]

        Args:
            data_part (str, optional): [path to the desired dataset train or test from data folder]. Defaults to "train/imgs".
        """
        if not isinstance(data_part, str):
            raise Exception(
                " data_part must be a str which represent the path from data folder to desired data set"
            )
        self.plot_from_dict(plot_title=f"Distribution of {data_part} data set")
        plt.show()

    def sampler_making(self, batch_size=128):
        """[return a DataLoader that gives balanced batch]

        Args:
            batch_size (int, optional): [batch size]. Defaults to 128.

        Returns:
            [torch.utils.data.dataloader.DataLoader ]: [Balanced DataLoader]
        """
        class_sample_count = np.array(list(self.get_class_distribution().values()))
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in self.data.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler


class PlankDatasetTest(Dataset):
    """[Planton dataset is a class to create a DataLoader from Kaggle Dataset]

    Args:
        Dataset ([heritage from torch.utils.dataset]): [class Dataset]
    """

    def __init__(
        self,
        path=TEST_PATH,
        transform=torchvision.transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    ):
        """[Object attributes definition]

        Args:
            transform ([transforms], optional): [transforms or composed transform applied to
            every images of the dataset]. Defaults to None.
        """
        if not isinstance(transform, (torchvision.transforms.transforms.Compose)):
            raise Exception(
                " transform must be None, torchvision.transforms.transforms.Compose"
            )

        super(Dataset, self).__init__()
        self.path = path

        self.data = datasets.ImageFolder(os.path.join(self.path), transforms.ToTensor())

        self.transform = transform

    def __getitem__(self, index):
        """[get a sample from dataset]

        Args:
            index ([int]): [int index position of the desired sample]

        Returns:
            [tuple(torch.tensor,torch.tensor)]: [image, class encoded]
        """
        if not isinstance(index, (int, float)) or index < 0:
            raise Exception(" index must be an float or int greater than 0")
        sample = self.data[index][0]

        return self.transform(sample)

    def __len__(self):
        """[get the lenght of the whole dataset]

        Returns:
            [int]: [length of the whole dataset]
        """
        return len(self.data)

    def imshow(self, index):
        """[Display the image corresponding to the index in the training set ]

        Args:
            index ([int]): [index of the image, should be inferior to self.n_train ]
        """
        if not isinstance(index, (int, float)) or index < 0:
            raise Exception(" index must be an float or int greater than 0")
        X_i = self.data[index]
        inp = X_i.numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        plt.title(f"Index : {index}")
        plt.show()


############################################
################### Test ###################
############################################
if __name__ == "__main__":

    ## Train loader
    t = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        MyRotationTransform(angles=list(np.linspace(-15, 15, 60))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    composed = torchvision.transforms.Compose(t)
    """
    dataset=PlankDataset(transform=composed)

    valid_ratio=0.2
    nb_train = int((1.0 - valid_ratio)*len(dataset))
    nb_test= int((valid_ratio)*len(dataset))
    train,test = torch.utils.data.random_split(dataset,[nb_train,nb_test])
    print(test,train)
    print(train,test)
    print(f"before doing anything len : {len(dataset)}")
    first_data=dataset[0]
    features,labels=first_data
    #print("features : " , features)
    print("features' shape : " , features.shape)
    print(f"label : {labels}")
    print(type(features),type(labels))
    print(type(len(dataset)),len(dataset))
    print(dataset.get_class_distribution())
    #dataset.imshow(2)
    #dataset.plot_distrib()

    sampler = dataset.sampler_making(BATCH_SIZE)
    dataloader= DataLoader(dataset, BATCH_SIZE, sampler=sampler)

    print(dataloader)

    for i, (data, target) in enumerate(dataloader):
        print("batch index {}, 0/1: {}/{}/{}/{}/{}".format(
            i,
            len(np.where(target.numpy() == 0)[0]),
            len(np.where(target.numpy() == 1)[0]),
            len(np.where(target.numpy() == 2)[0]),
            len(np.where(target.numpy() == 3)[0]),
            len(np.where(target.numpy() == 4)[0]),))

    for img,target in dataset :
                print(f"target : {img.shape}")
    """

    ## Test loader
    t = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        MyRotationTransform(angles=list(np.linspace(-15, 15, 60))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    composed = torchvision.transforms.Compose(t)
    dataset = PlankDatasetTest(transform=composed)
    for img in dataset:
        print(f"target : {img.shape}")
    print("train")
    dataset = PlankDataset(transform=composed)
    for img, label in dataset:
        print(f"target : {img.shape}")
