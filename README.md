# Term-Project: Classifying Images in the SVHN Dataset

This repository contains code and resources for the Phase 1 and 2 Term Project of the Pattern Recognition course. 

## Data Set

The Street View House Number dataset used in this project can be downloaded from [here](http://ufldl.stanford.edu/housenumbers/). It is divided into a train and test set.

## Phase 1

The project report can be found in `project_1.ipynb`, which contains explanations, code, and results related to the different tasks and components of the project phase 1.

### Project Components

The project is organized into several code files, each dedicated to one of the four tasks.

- `utils.py`: This file contains utility functions and functions for checking the dataset, which are used for task 1.

- `feature_extraction.py`: Here, you can find code for implementing the five chosen feature extraction techniques as part of task 2.

- `classification.py`: This file contains code related to the implementation of Linear Discriminant Classifiers, as specified in task 3.

- `metric_learning.py`: This file is contains the code related to metric learning, which is part of task 4. However, please note that the metric learning framework is implemented in `project_1.ipynb`.


## Phase 2

The project report can be found in `project_2.ipynb`, which contains explanations, code, and results related to the different tasks and components of the project phase 2.

### Project Components

The project is organized into several code files.

- `random_forest.py`: This file contains code related to training, tuning and testing the decision tree and random forest.

- `model.py`: This file contains code related to the architecture of the Vision Transformer. The code was adopted from this [repo](https://github.com/s-chh/PyTorch-Vision-Transformer-ViT-MNIST/blob/main/data_loader.py).

- `solver.py`: This file contains code related to the training and testing of the Vision Transformer.