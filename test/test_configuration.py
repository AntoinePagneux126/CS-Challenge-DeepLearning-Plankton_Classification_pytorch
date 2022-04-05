import sys, os, inspect

#############################################
#####   Import gestion across project   #####
#############################################

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import src
from src.configuration import *

import pytest

#############################################
#####   Testing every single function   #####
#############################################
def test_config_deep_learning():
    config = config_deeplearing()
    assert isinstance(config["PATH"]["PATH"], str)
    assert isinstance(config["PARAMETERS"]["BATCHSIZE"], str)
