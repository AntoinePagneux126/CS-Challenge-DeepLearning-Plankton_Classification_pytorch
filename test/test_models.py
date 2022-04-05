import sys, os, inspect

#############################################
#####   Import gestion across project   #####
#############################################

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import src
from src.models import *

import pytest

#############################################
#####   Testing every single function   #####
#############################################


def test_PerceptronNet():
    with pytest.raises(Exception) as execinfo:
        model = PerceptronNet(6.0, 7)
        assert (
            str(execinfo.value)
            == "empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of"
        )

    with pytest.raises(Exception) as execinfo:
        model = PerceptronNet(6, 7.0)
        assert (
            str(execinfo.value)
            == "empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of"
        )


def test_LinearNet():
    with pytest.raises(Exception) as execinfo:
        model = LinearNet(6.0, 7)
        assert (
            str(execinfo.value)
            == "empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of"
        )

    with pytest.raises(Exception) as execinfo:
        model = LinearNet(6, 7.0)
        assert (
            str(execinfo.value)
            == "empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of"
        )


def test_CNNet():
    with pytest.raises(Exception) as execinfo:
        model = CNNet(6.0, 7)
        assert (
            str(execinfo.value)
            == "empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of"
        )

    with pytest.raises(Exception) as execinfo:
        model = CNNet(6, 7.0)
        assert (
            str(execinfo.value)
            == "empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of"
        )


def test_ResNet34():
    with pytest.raises(Exception) as execinfo:
        model = ResNet34(6.0)
        assert (
            str(execinfo.value)
            == "empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of"
        )


def test_ResNet50():
    with pytest.raises(Exception) as execinfo:
        model = ResNet50(6.0)
        assert (
            str(execinfo.value)
            == "empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of"
        )


def test_ResNet50T():
    with pytest.raises(Exception) as execinfo:
        model = ResNet50T(6.0)
        assert (
            str(execinfo.value)
            == "empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of"
        )


if __name__ == "__main__":
    test_PerceptronNet()
    test_LinearNet()
    test_CNNet()
    test_ResNet34()
    test_ResNet50()
    test_ResNet50T()
