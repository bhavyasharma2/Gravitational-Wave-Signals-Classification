# Gravitational Wave Signals Classification

## Introduction

This project focuses on improving the classification accuracy of glitches in gravitational wave signals from LIGO data using deep learning models. Conducted as part of an ML research internship at Spartificial, the project builds upon prior research from the Gravity Spy initiative. We evaluated the effectiveness of label smoothing, attention modules, and different deep learning architectures, including VGG16, VGG19, and MobileNet.

## Dataset

- **Training and Validation Dataset**: Custom dataset with categorized spectrogram images of gravitational wave glitches.
- **Test Dataset**: A separate test set.
- **Organization**: The dataset is structured into subdirectories, each corresponding to a different glitch class (22 classes in total).

The dataset should be organized into subdirectories for each class.

## Model Architectures

We implemented and evaluated the following deep learning models:

- **VGG16**: Standard VGG16 architecture with additional custom blocks and preprocessing techniques.
- **VGG19**: Enhanced version of VGG16 with an attention module and label smoothing.
- **MobileNet**: Lightweight and efficient architecture, optimized with label smoothing but without an attention module or multi-view fusion techniques.

## Techniques Used

Several deep learning and optimization techniques were employed to enhance model performance:

- **Label Smoothing**: Used as a regularization technique to improve generalization and prevent overconfidence in predictions.
- **Attention Mechanisms**: Implemented in VGG19 to highlight important features in spectrogram images.
- **Regularization**: Applied L2 regularization and dropout layers to mitigate overfitting.
- **Early Stopping**: Monitored validation loss to halt training when overfitting was detected.
- **Model Checkpointing**: Saved the best-performing model based on validation accuracy.

## Key Findings

Through extensive experimentation, we observed the following:

- **VGG16 with Attention & Label Smoothing**: Achieved an accuracy of 56%.
- **VGG19 with Attention & Label Smoothing**: Accuracy improved to 72%.
- **MobileNet with Label Smoothing (without Attention & Multi-view Fusion)**: Outperformed other models with an accuracy of 89%, making it the most effective solution for this classification task.

These results underscore the effectiveness of preprocessing techniques like label smoothing in improving classification accuracy. The Gravity Spy dataset, comprising 22 glitch categories, provided a comprehensive evaluation platform for the models.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/bhavyasharma2/Gravitational-Wave-Signals-Classification.git
    cd Gravitational-Wave-Signals-Classification
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Acknowledgements

- TensorFlow and Keras for providing the deep learning framework.
- [Kaggle](https://www.kaggle.com/) for providing datasets.
- Gravity Spy Initiative: For pioneering research in glitch classification.
