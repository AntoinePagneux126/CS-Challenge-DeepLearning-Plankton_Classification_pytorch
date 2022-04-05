import sys, os, inspect

#############################################
#####   Import gestion across project   #####
#############################################

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import src
from src.imgloader import PlankDataset, PlankDatasetTest
import torch, torchvision

import pytest

#############################################
#####   Testing every single function   #####
#############################################


def test_loader():

    composed = torchvision.transforms.Compose([])
    dataset = PlankDataset(transform=composed)
    first_data = dataset[0]
    features, labels = first_data

    assert isinstance(features, torch.tensor)
    assert isinstance(labels, torch.tensor)
    assert isinstance(dataset.imshow(2), None)
    assert isinstance(dataset.plot_distrib(), None)

    with pytest.raises(Exception) as execinfo:
        PlankDataset(transform=3)
        assert (
            str(execinfo.value)
            == " transform must be None, torchvision.transforms.transforms.Compose"
        )
    with pytest.raises(Exception) as execinfo:
        dataset.get_class_distribution(2)
        assert (
            str(execinfo.value)
            == " data_part must be a str which represents the path from data folder to desired data set"
        )
    with pytest.raises(Exception) as execinfo:
        dataset.plot_from_dict(2)
        assert (
            str(execinfo.value)
            == "  plot_title must be a str which is the title of the histogram"
        )
    with pytest.raises(Exception) as execinfo:
        dataset.imshow(-2)
        assert str(execinfo.value) == " index must be an float or int greater than 0"
    with pytest.raises(Exception) as execinfo:
        dataset.plot_distrib(3)
        assert (
            str(execinfo.value)
            == " data_part must be a str which represent the path from data folder to desired data set"
        )

    composed = torchvision.transforms.Compose([])
    dataset = PlankDatasetTest(transform=composed)
    first_data = dataset[0]
    features = first_data

    assert isinstance(features, torch.tensor)
    assert isinstance(dataset.imshow(2), None)

    with pytest.raises(Exception) as execinfo:
        PlankDatasetTest(transform=3)
        assert (
            str(execinfo.value)
            == " transform must be None, torchvision.transforms.transforms.Compose"
        )
    with pytest.raises(Exception) as execinfo:
        dataset.imshow(-2)
        assert str(execinfo.value) == " index must be an float or int greater than 0"
