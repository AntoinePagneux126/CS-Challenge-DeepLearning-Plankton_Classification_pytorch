# Deep Learning plankton Challenge : École CentraleSupelec 2021-2022

![CentraleSupelec Logo](https://www.centralesupelec.fr/sites/all/themes/cs_theme/medias/common/images/intro/logo_nouveau.jpg)


## Authors
* Matthieu Briet : matthieu.briet@student-cs.fr
* Tanguy Colleville : tanguy.colleville@student-cs.fr
* Antoine Pagneux : antoine.pagneux@student-cs.fr

## Useful links
* Our Workspace : [Here](https://tanguycolleville.notion.site/DEEP-LEARNING-2022-Challenge-f389c01ee85a4d7f88df8b67794bafc8)
* Our Documentation : [Here](https://www.overleaf.com/read/bsqfrdjvsjft)
* Kaggle home page : [Here](https://www.kaggle.com/c/3md4040-2022-challenge/)
* Our video link : [Here](https://youtu.be/5eNh8_mltwQ)

## Summary
  - [Authors ](#authors-)
  - [Useful links](#Useful-links)
  - [Summary](#summary)
  - [Introduction](#introduction)
  - [Our approach](#our--approach)
  - [Architecture & overview](#architecture--overview)
  - [Model & stuffs](#model--stuffs)
  - [Conclusion](#conclusion)

 ## Introduction
  This kaggle contest  takes part in the Deep Learning courses 3MD4040 as evaluation. This challenge is about a images classification problem with 86 exclusive classes, strongly unbalanced with images of varying sizes. We have to solve is a real problem that [biologists](https://anr.fr/fr/detail/call/challenge-ia-biodiv-recherche-en-intelligence-artificielle-dans-le-champ-de-la-biodiversite/) are facing as plankton are strong indicators on the quality of the marine ecosystem.

 ## Our approach
Our intellectual path way to get these perfomance : 0.7, is detailed in our overleaf report [Here](https://www.overleaf.com/read/bsqfrdjvsjft).

 ## Architecture & overview
```
.
├── LICENSE
├── README.md
├── config
│   └── config.ini
├── data
│   ├── test
│   ├── test_4
│   ├── train
│   └── train_4
├── docs
│   ├── Makefile
│   ├── _build
│   ├── _static
│   ├── _templates
│   ├── classifier.rst
│   ├── conf.py
│   ├── configuration.rst
│   ├── generator_csv.rst
│   ├── imgloader.rst
│   ├── index.rst
│   ├── mailsender.rst
│   ├── main.rst
│   ├── make.bat
│   ├── models.rst
│   ├── modules.rst
│   └── utils.rst
├── logs
│   ├── resnet50_0
│   ├── resnet50_1
│   ├── resnet50_10
│   ├── resnet50_11
│   ├── resnet50_2
│   ├── resnet50_3
│   ├── resnet50_4
│   ├── resnet50_5
│   ├── resnet50_6
│   ├── resnet50_7
│   ├── resnet50_8
│   └── resnet50_9
├── logslurms
│   └── empty.txt
├── models
│   ├── cnn_0.pt
│   └── resnet50_0.pt
├── outputs
│   └── tanTL_0.csv
├── requirements.txt
├── src
│   ├── __pycache__
│   ├── classifier.py
│   ├── configuration.py
│   ├── generator_csv.py
│   ├── imgloader.py
│   ├── mailsender.py
│   ├── main.py
│   ├── models.py
│   └── utils.py
└── test
    ├── test_classifier.py
    ├── test_configuration.py
    ├── test_generator.py
    ├── test_imgloader.py
    ├── test_mailsender.py
    ├── test_main.py
    ├── test_models.py
    └── test_utils.py
```
 ## Run model
First be sure of what you have written your config/config.ini file.
* If you really have huge GPU go ahead it is quite straightforward :

  Example to train a CNN ``` python3 main.py train --model=cnn --f1_loss=True```

  Example to test our 8th ResNet50 ``` python3 main.py test --model=resnet50 --number=8```

  Example to train our best model on another dataset ```python3 main.py train --train_path=PATH_TO_TRAINING_SET```

  Example to test a trained model saved in a specific path on a specific test set ```python3 main.py test --checkpoint_path=PATH_TO_CHECKPOINT --test_path=PATH_TO_TEST_SET```

there is numerous flag :
** mode (train or test)

** model

** number

** f1_loss

** pretrained

** train_path

** test_path

** checkpoint_path


* Otherwise, please use cluster as we have at our disposal at Centralesupelec and use the script src/jobBeru4s.batch. Go in src folder in your terminal and ```sbatch jobBeru4s.batch```. You will see a job id, typically 4263, which refers to your job. By doing ```squeue -u gpusdi1_XX``` you can be sure that your job is running or in the queue. Additionnaly, in ```deepchallenge-cs/logslurm/slurm4263.out``` and in ```deepchallenge-cs/logslurm/slurm4263.err ``` you will see respectively output and err. Finnaly the CSV is automatically sended to you, if you mentioned your email adress in the ```config/config.ini``` file.

## Code documentation
* You can access to python code documentation in `docs/_build/hmtl/index.html`
* You can access to python coverage report in `coverage/index.html`
* You can access to python pylint report in `docs/pylint_report.txt`

 ## Conclusion
 Since the problem was complicated, we have learned so much about modeling, optimizer, weigth initialisation choice and so on. The lecture was just an appetizer to go further on interesting point go get a better score with our model for this kaggle challenge. Moreover, it gives us the opportunity to use PyTorch.
