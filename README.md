## Table of Contents

  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Model Architectures](#model-architectures)
  - [Techniques Used](#techniques-used)
  - [Installation](#installation)
  - [Acknowledgements](#acknowledgements)

## Introduction

This project aims to enhance the classification accuracy of glitches in gravitational wave signals in LIGO data by leveraging advanced deep learning models. Through a comprehensive literature review on Gravity Spy research, which consisted of implementation of multi-view fusion techniques, attention modules, and label smoothing, various convolutional nueral network archiectures were implemented and evaluated in this project including VGG16, VGG19, and MobileNet.

## Dataset

- **Training and Validation Dataset**: Custom dataset with categorized images.
- **Test Dataset**: Same as above or a separate test set.

The dataset should be organized into subdirectories for each class.

## Model Architectures

- **VGG16**: Standard VGG16 architecture with additional custom blocks.
- **VGG19**: Standard VGG19 architecture with additional custom blocks.
- **MobileNet**: Standard MobileNet architecture with additional custom blocks.

## Techniques Used

- **Label Smoothing**: Custom loss function to smooth labels during training.
- **Regularization**: L2 regularization and Dropout for better generalization.
- **Early Stopping**: Prevents overfitting by monitoring validation loss.
- **Model Checkpointing**: Saves the best model based on validation loss.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Acknowledgements

- TensorFlow and Keras for providing the deep learning framework.
- [Kaggle](https://www.kaggle.com/) for providing datasets.
