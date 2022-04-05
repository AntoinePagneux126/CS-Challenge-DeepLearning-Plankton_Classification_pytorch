import torch
import torch.nn.init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

import os
import argparse

# Utils
from mailsender import send_csv_by_mail
from generator_csv import csv_generator, correct_dic
from imgloader import MyRotationTransform
from configuration import config_deeplearing
from classifier import Classifier
from models import (
    CNNet,
    PerceptronNet,
    LinearNet,
    ResNet152,
    ResNet50,
    ResNet50T,
    ResNet34,
)

# exemple to run:   python3 main.py train --model=cnn
#                   python3 main.py train --model=resnet50
#                   python3 main.py test --model=cnn --number=1

## INPUTS
my_config = config_deeplearing()
parser = argparse.ArgumentParser()

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
UNFREEZE = int(my_config["PARAMETERS"]["UNFREEZE"])
EARLY_STOP = True if (my_config["PARAMETERS"]["EARLY_STOP"] == "True") else False
LOSS_WEIGHT = True if (my_config["PARAMETERS"]["LOSS_WEIGHT"] == "True") else False
BEST_MODEL = my_config["MODELS"]["BEST_MODEL"]
BEST_MODEL_NUM = my_config["MODELS"]["BEST_MODEL_NUM"]

## GLOBALS
NUMBER_INPUTS = 3 * CROP_SIZE * CROP_SIZE
TRAIN_PATH = PATH + TRAIN_FOLDER
TEST_PATH = PATH + TEST_FOLDER

## Argparse
parser.add_argument(
    "mode", type=str, help="Enter the mode you want to use (train/test)"
)
parser.add_argument(
    "--train_path",
    type=str,
    help="Enter the path of the train set",
    required=False,
    default=TRAIN_PATH,
)
parser.add_argument(
    "--test_path",
    type=str,
    help="Enter the path of the test set",
    required=False,
    default=TEST_PATH,
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Enter the path of the checkpoint of the best model",
    required=False,
    default=CHECKPOINT_PATH + BEST_MODEL + "_" + BEST_MODEL_NUM + ".pt",
)
parser.add_argument(
    "--model",
    type=str,
    help="Enter the model you want to use",
    required=False,
    default=BEST_MODEL,
)
parser.add_argument(
    "--number",
    type=int,
    help="Enter the number of the model you want to test",
    required=False,
    default=-1,
)
parser.add_argument(
    "--pretrained",
    type=bool,
    help="Boolean to impose pretraining",
    required=False,
    default=True,
)
parser.add_argument(
    "--f1_loss",
    type=bool,
    help="Boolean to choose F1_loss or not",
    required=False,
    default=False,
)


TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        MyRotationTransform(angles=list(np.linspace(-15, 15, 60))),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        MyRotationTransform(angles=list(np.linspace(-15, 15, 60))),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if __name__ == "__main__":

    # Read instructions from argparse
    args = parser.parse_args()
    mode = args.mode
    model_type = args.model
    number = int(args.number)
    pretrained = bool(args.pretrained)
    f1_loss = bool(args.f1_loss)
    train_path = args.train_path
    test_path = args.test_path
    cp_path = args.checkpoint_path

    # Select the Device
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Print
    print("-------------")
    print("You will {} a {} on {}.".format(mode, model_type, device))
    print("-------------")

    # Training a model
    if mode == "train":
        MyClassifier = Classifier(
            data_dir=train_path,
            num_classes=NUMBER_OUTPUTS,
            device=device,
            sample=True,
            Transform=TRANSFORM,
            stop_early=EARLY_STOP,
            loss_weights=LOSS_WEIGHT,
            pretrained=pretrained,
            f1_loss=f1_loss,
        )

        print("Loading data: trainset...")
        train_loader, val_loader = MyClassifier.load_data()
        print("OK")

        print("Loading classifier...")
        MyClassifier.load_model(model_type=model_type)
        print("OK")

        print("Start fitting...")
        MyClassifier.fit(
            num_epochs=NUMBER_EPOCHS,
            unfreeze_after=UNFREEZE,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        print("OK")

    # Testing a model
    print("-------------")
    print("You will test a {} on {}.".format(model_type, device))
    print("-------------")

    print("Loading classifier...")
    # Create an instance of the model
    model = CNNet(NUMBER_OUTPUTS, CROP_SIZE)
    if model_type == "resnet50":
        print("resnet50")
        model = ResNet50(NUMBER_OUTPUTS, freeze_backbone=False)

    elif model_type == "resnet34":
        print("resnet34")
        model = ResNet34(NUMBER_OUTPUTS, freeze_backbone=False)

    elif model_type == "resnet152":
        print("resnet152")
        model = ResNet152(NUMBER_OUTPUTS, freeze_backbone=False)

    elif model_type == "R50":
        print("R50")
        model = ResNet50T(NUMBER_OUTPUTS)

    elif model_type == "cnn":
        print("cnn")
        model = CNNet(NUMBER_OUTPUTS, CROP_SIZE)

    elif model_type == "mlp":
        print("mlp")
        model = LinearNet(NUMBER_INPUTS, NUMBER_OUTPUTS)

    elif model_type == "p":
        print("p")
        model = PerceptronNet(NUMBER_INPUTS, NUMBER_OUTPUTS)

    else:  # default case : train a cnn
        print("Default model loaded")
        model = CNNet(NUMBER_OUTPUTS, CROP_SIZE)

    # Apply of training to the model
    model = model.to(device)
    # Find the model to load
    # If one wants to load an old model:
    if mode == "test" and number != -1:
        if cp_path != CHECKPOINT_PATH + BEST_MODEL + "_" + BEST_MODEL_NUM + ".pt":
            checkpoint = torch.load(cp_path)
        elif model_type == BEST_MODEL:
            checkpoint = torch.load(
                CHECKPOINT_PATH + BEST_MODEL + "_" + BEST_MODEL_NUM + ".pt"
            )
        else:
            checkpoint = torch.load(
                CHECKPOINT_PATH + model_type + "_" + str(number) + ".pt"
            )
    else:
        lcp = os.listdir(CHECKPOINT_PATH)
        n = 0
        for cp in lcp:
            if cp.startswith(model_type):
                n += 1
        # Then load the so called model
        checkpoint = torch.load(CHECKPOINT_PATH + model_type + "_" + str(n - 1) + ".pt")
    model.load_state_dict(checkpoint)
    # Turn model into eval mode
    model.eval()
    print("OK")

    # Load testset
    print("Loading data: trainset...")
    testset = torchvision.datasets.ImageFolder(test_path, transform=TRANSFORM_TEST)
    test_loader = DataLoader(
        testset, batch_size=1, num_workers=NUM_WORKER, shuffle=False, pin_memory=True
    )
    print("OK")

    # Start predictions
    print("Start fitting...")
    dict_img = {}
    with torch.no_grad():
        i = 0
        model.eval()
        for data in tqdm(test_loader):
            image = data[0].to(device)

            outputs = model(image)
            predicted = torch.argmax(outputs.data, 1)

            name = str(i) + ".jpg"
            # print(predicted)
            dict_img[name] = str(int(predicted))
            i += 1

    dict_img = correct_dic(
        dict_img, test_loader, testset
    )  # To apply classes corrections
    print("OK")

    print("Generate CSV file and send it by mail...")
    csv_generator(dict_img, model_type)  # Generate a csv file
    # send_csv_by_mail() # Send the result by mail
