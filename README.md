# Chess Piece Image Classification Model

## Overview
This repository contains code for a chess piece image classification model built using deep learning techniques. The model is trained to classify images of chess pieces into their respective categories, such as pawn, knight, bishop, rook, queen, and king.

## Requirements
- Python (>=3.6)
- PyTorch
- torchvision

## Dataset
The model requires a dataset of labeled chess piece images for training and evaluation. The dataset should be organized into directories, with each directory representing a different class (e.g., 'pawn', 'knight', etc.).
Breakdown of training, test and validation data (60%, 20%, 20% respectively):
Train
Bishop = 52
King = 46
Knight = 64
Pawn = 64
Queen = 47
Rook = 61 

Validate
Bishop = 17
King = 15
Knight = 21
Pawn = 21
Queen = 15
Rook = 20 

Test
Bishop = 18
King = 15
Knight = 21
Pawn = 22
Queen = 16
Rook = 21 

## Steps
1. **Data Preparation:** Prepare the dataset and organize it into appropriate directories.
2. **Model Training:** Run the `train.py` script to train the model. Adjust hyperparameters and model architecture as needed.
3. **Model Evaluation:** Evaluate the trained model using the `evaluate.py` script on a separate test dataset.
4. **Inference:** Use the trained model for inference on new chess piece images using the `infer.py` script.


## Model Evaluation
The model's performance can be evaluated using metrics such as accuracy. Additionally, visual inspection of classification results can provide insights into the model's strengths and weaknesses.


## Acknowledgments
- This project was inspired by the need for automated chess piece recognition in various applications.
- We acknowledge the contributors to PyTorch and torchvision for providing the necessary tools and libraries for building deep learning models.
