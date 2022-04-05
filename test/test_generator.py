import sys, os, inspect

#############################################
#####   Import gestion across project   #####
#############################################

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import src
from src.generator_csv import *

import pytest

#############################################
#####   Testing every single function   #####
#############################################
def test_csv_generator():
    csv_generator({"None": None}, "test")
    assert os.path.exists("../outputs/test.csv")


def test_csv_to_dict():
    dico = csv_to_dic("../outputs/test.csv")
    assert isinstance(dico, dict)
