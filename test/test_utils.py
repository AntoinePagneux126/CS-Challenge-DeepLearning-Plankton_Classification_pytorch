import sys, os, inspect

#############################################
#####   Import gestion across project   #####
#############################################

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import src
from src.utils import *

import pytest

#############################################
#####   Testing every single function   #####
#############################################


def test_ES_class():
    ES = EarlyStopping()
    assert ES.patience == 3
    assert ES.min_delta == 0


def test_f1():
    assert isinstance(f1([0], [1]), np.floating)
    assert len(f1([0], [1])) == 1
    with pytest.raises(Exception) as execinfo:
        f1([0, 0], [1])
        assert str(execinfo.value) == "y_true and y_pred must have the same length"
    with pytest.raises(Exception) as execinfo:
        f1(0, 1)
        assert str(execinfo.value) == "y_true and y_pred must be list or np.ndarray"


def test_f1_loss():
    assert isinstance(f1_loss([0], [1]), np.floating)
    assert len(f1_loss([0], [1])) == 1
    with pytest.raises(Exception) as execinfo:
        f1_loss([0, 0], [1])
        assert str(execinfo.value) == "y_true and y_pred must have the same length"
    with pytest.raises(Exception) as execinfo:
        f1_loss(0, 1)
        assert str(execinfo.value) == "y_true and y_pred must be list or np.ndarray"


def test_f1_torch():
    y_t, y_p = torch.empty((2, 3)), torch.empty((3, 3))
    with pytest.raises(Exception) as execinfo:
        f1_torch(y_t, y_p)
        assert str(execinfo.value) == "y_true and y_pred must have the same length"


def test_generate_unique_log_path():
    assert isinstance(generate_unique_logpath("testing", "test_coverage"), str)
    assert (
        generate_unique_logpath("testing", "test_coverage") == "testing/test_coverage_0"
    )
