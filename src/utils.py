import torch
import numpy as np
from sklearn.metrics import f1_score
import os


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=3, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


epsilon = 1e-7


def f1(y_true, y_pred):
    """[F1-macro adapted to be C1]

    Args:
        y_true ([np.ndarray]): [array of true labels ]
        y_pred ([np.ndarray]): [array of predicted labels]

    Returns:
        [float]: [F1_Macro]
    """
    if len(y_true) != len(y_pred):
        raise Exception("y_true and y_pred must have the same length")
    if not isinstance(y_true, (list, np.ndarray)) or not isinstance(
        y_pred, (list, np.ndarray)
    ):
        raise Exception("y_true and y_pred must be list or np.ndarray")

    y_pred = np.round(y_pred)
    tp = np.sum(np.cast(y_true * y_pred, "float"), axis=0)
    tn = np.sum(np.cast((1 - y_true) * (1 - y_pred), "float"), axis=0)
    fp = np.sum(np.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = np.sum(np.cast(y_true * (1 - y_pred), "float"), axis=0)

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)
    return np.mean(f1)


def f1_loss(y_true, y_pred):
    """[F1-macro adapted to be C1 and useable as loss function]

    Args:
        y_true ([np.ndarray]): [array of true labels ]
        y_pred ([np.ndarray]): [array of predicted labels]

    Returns:
        [float]: [1- F1_Macro]
    """
    if len(y_true) != len(y_pred):
        raise Exception("y_true and y_pred must have the same length")
    if not isinstance(y_true, (list, np.ndarray)) or not isinstance(
        y_pred, (list, np.ndarray)
    ):
        raise Exception("y_true and y_pred must be list or np.ndarray")

    tp = np.sum(np.array(y_true * y_pred, dtype="float"), axis=0)
    tn = np.sum(np.array((1 - y_true) * (1 - y_pred), dtype="float"), axis=0)
    fp = np.sum(np.array((1 - y_true) * y_pred, dtype="float"), axis=0)
    fn = np.sum(np.array(y_true * (1 - y_pred), dtype="float"), axis=0)

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)
    return 1 - np.mean(f1)


def f1_torch(y_pred, y_true):
    if y_true.size()[0] != y_pred.size()[0]:
        raise Exception("y_true and y_pred must have the same length")

    epsilon = 1e-6
    _, y_pred = torch.max(y_pred, 1)

    tp = torch.sum((y_true * y_pred).float(), dim=0)
    tn = torch.sum(((1 - y_true) * (1 - y_pred)).float(), dim=0)
    fp = torch.sum(((1 - y_true) * y_pred).float(), dim=0)
    fn = torch.sum((y_true * (1 - y_pred)).float(), dim=0)

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    f1 = 2 * p * r / (p + r + epsilon)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return torch.tensor(1 - f1_score(y_true.cpu(), y_pred.cpu(), average="macro"))


def generate_unique_logpath(logdir: str, raw_run_name: str):
    """[Generate unique logpath for tensorboard log at each train]

    Args:
        logdir ([str]): [dir path which host all logs]
        raw_run_name ([str]): [name of the current training]

    Returns:
        [str]: [path corresponding to the host folder]
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


print(generate_unique_logpath("cov", "1"))
