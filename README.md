# Detectron2-STAC-SODA10m

## Introduction
Welcome to the Detectron2-STAC-SODA10m repository. This project integrates the STAC semi-supervised object detection algorithm with Detectron2 applied to the SODA10m dataset. The STAC algorithm, which stands for Semi-supervised Teacher-student Anomaly detection for object Counting, leverages both labeled and unlabeled data to improve object detection performance, particularly in situations where labeled data is not abundantly available.

## Objectives
The main objective of this repository is to provide an end-to-end solution for applying STAC in the realm of satellite imagery analysis. By harnessing the SODA10m dataset, this project aims to:
- Enhance object detection models using semi-supervised learning.
- Reduce the reliance on large sets of labeled data.
- Improve the accuracy and efficiency of object detection in satellite images.

## Features
- **Preprocessing Scripts**: Prepare your satellite images from the SODA10m dataset for object detection tasks.
- **Model Training Notebooks**: Step-by-step Jupyter notebooks to train your object detection model using Detectron2 with STAC.
- **Inference and Evaluation**: Tools to evaluate the model performance and visualize the detection results.

## Demo
To illustrate the functionality of the repository, here is a GIF showcasing the pre-processing, training, and detection steps:

![STAC-Detectron2-Demo](demo.gif)

_The GIF is a composite of two images that demonstrate the before-and-after effect of applying the STAC algorithm to satellite imagery._

## Getting Started
To get started with this project, clone the repository and follow the setup instructions in the provided documentation. Ensure you have the necessary dependencies installed, including Detectron2 and any specific packages required for the SODA10m dataset.

## Contributions
Contributions to this repository are welcome. Please submit a pull request or open an issue if you have suggestions or improvements.

## License
This project is open-sourced under the MIT license. See the LICENSE file for details.

