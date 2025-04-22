# Gravitational Wave Signals Classification

## Introduction

This project focuses on improving the classification accuracy of glitches in gravitational wave signals from LIGO data using deep learning models. Conducted as part of an ML research internship at Spartificial, the project builds upon prior research from the Gravity Spy initiative. We evaluated the effectiveness of label smoothing, attention modules, and different deep learning architectures, including VGG16, VGG19, and MobileNet.

## Setup Instructions

To set up your development environment and run this project locally, follow the steps below:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/bhavyasharma2/Gravitational-Wave-Signals-Classification.git
   cd Gravitational-Wave-Signals-Classification
   ```

2. **Download the Dataset:**
   - Go to the Kaggle dataset page: [Gravity Spy Gravitational Waves Dataset](https://www.kaggle.com/datasets/tentotheminus9/gravity-spy-gravitational-waves).
   - - Click on the "Download" button to get the dataset as a `.zip` file.
   - Visit [this link](https://zenodo.org/records/5649212) and download `L1_O3b.csv`. 

3. **Extract the Dataset:**
   - After downloading, extract the contents of the `.zip` file.

4. **Place the Dataset in the Project:**
   - Move the extracted folder and `L1_O3b.csv` file into the `data/` directory in the project root.

5. **Create and Activate a Virtual Environment:**
   - **For Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **For macOS/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

6. **Install the Required Dependencies:**
   - Upgrade pip:
     ```bash
     pip install --upgrade pip
     ```
   - Install dependencies from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

7. **Run the Project:**
   - Make sure the dataset is extracted in the correct directory (`data/gravitational_waves_dataset`).
   - Then run:
     ```bash
     python MobileNet_Model.py
     python VGG_Models.py
     ```


## Dataset

This notebook uses the **Gravity Spy Gravitational Waves** dataset from Kaggle and **L1_O3b** data from Zenodo records. To replicate this project, please follow the steps below to download and set up the dataset:

The dataset contains the following:

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

## Acknowledgements

- TensorFlow and Keras for providing the deep learning framework.
- [Kaggle](https://www.kaggle.com/) for providing datasets.
- Gravity Spy Initiative: For pioneering research in glitch classification.
