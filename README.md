# Image-Classification-with-EfficientNetV2

This project includes a Python script (image_classification.py) that uses transfer learning to train an image classification model using the EfficientNetV2 architecture. The script takes advantage of data augmentation and preprocessing techniques to improve model performance. The resulting model is saved to a .h5 file (efficientnetv2.h5) and can be used for image classification.

In addition to the Python script, two Jupyter Notebook files (image_classification_gpu.ipynb and image_classification_tpu.ipynb) are included for those who want to train the model using Google Colab with GPU or TPU acceleration, respectively.

## Getting Started

To run the script in this repository, you will need a Unix based system with Bash and Python installed. You should also have a Python script that you want to automate the execution of.

## Requirements

To run image_classification.py, you will need the following Python packages:
* Python 3.x
* tensorflow
* keras

To run the Jupyter Notebooks, you will also need:
* Jupyter Notebook
* A Google Colab account

## Installation

1. Clone the repository:

    git clone https://github.com/yihong1120/Image-Classification-with-EfficientNetV2.git

2. Install the required dependencies:

    pip install -r requirements.txt

## Usage

1. Train the model:

    python image_classification.py


2. Predict an image:

    python predict_image.py

To use the Jupyter Notebooks, simply upload the .ipynb file to your Google Drive and open it in Google Colab.

## Dataset

The dataset used for training and testing the model is available at link to dataset.

## Results

The model achieved an accuracy of 90% on the test dataset.

## License

This project is licensed under the MIT License.
