# HemoMorphNet – Blood Cell Morphology Classification System
Blood Cell Detection Using InceptionV3 (Deep Learning Project)

## Project Overview

This project focuses on automatic detection and classification of blood cells using a Deep Learning–based CNN model.
We use InceptionV3, a powerful convolutional neural network pretrained on ImageNet, and fine-tune it to classify 8 types of blood cells from microscopic images.

The goal is to assist medical analysis and automate blood cell identification for faster diagnostics.

## Project Objectives

* Build a deep-learning classifier using CNN + Transfer Learning
* Train and fine-tune InceptionV3 for high accuracy
* Evaluate performance using:
   * Accuracy & Loss graphs
   * Confusion matrix
   * Classification report
   * ROC curves

## Dataset (8 Classes)

The dataset contains the following 8 blood cell classes:
+ EOSINOPHIL
+ LYMPHOCYTE
+ MONOCYTE
+ NEUTROPHIL
+ BASOPHIL
+ NORMAL RBC (Red Blood Cells)
+ IMMATURE GRANULOCYTE
+ PLATELET

Project Files (Google Drive Link)
The complete dataset, trained models, and other large resources are available here:
+ Google Drive Link: https://drive.google.com/drive/folders/1DlxIaTSyhhdNGPdGsLXDdAV8KEspdnp2?usp=sharing
(Some files were too large to upload to GitHub, so they are stored in Google Drive for view.)

### What CNN model does this project use?

This project uses a Convolutional Neural Network architecture based on InceptionV3, which is a deep CNN model consisting of:
* Multiple convolution layers
* Batch Normalization
* Multi-scale Inception blocks
* Auxiliary classifiers
* Global Average Pooling
* Dense classification layers

### Why InceptionV3?

* Extracts multi-scale features
* Efficient & high-performing
* Reduces computational cost with Inception modules
* Pretrained on ImageNet → Faster convergence
* Works great for medical imaging tasks

### Custom layers added on top of InceptionV3
After removing the top classifier, we added:
+ GlobalAveragePooling2D
+ BatchNormalization
+ Dense(256, activation='relu')
+ Dropout(0.4)
+ Dense(8, activation='softmax') ← final layer for 8 classes

* Two-stage training approach:
  Stage 1 (Feature Extraction):
  + Base model frozen
  + Train only top layers
  + 25 epochs

  Stage 2 (Fine-tuning):
  * Unfreeze last 50 layers
  * Smaller learning rate
  * 10 epochs
Both stages generate separate training histories.

## Training Pipeline

1. Load dataset using ImageDataGenerator
2. Build InceptionV3-based model
3. Compile with Adam optimizer
4. Train in two stages
5. Save checkpoint and final .keras model
   Model paths used:
* checkpoint.keras
* final_trained_model.keras

## Evaluation Metrics

The evaluation notebook contains:
1. Confusion Matrix - Visualizes misclassifications between classes.
2. Classification Report - Includes: Precision, Recall, F1-score, Support
3. ROC Curve (One-vs-Rest) - Using roc_curve & auc for each class.
4. Accuracy Graph - Shows overfitting/underfitting trends.
5. Loss Graph - Shows optimization behavior.

## Visualizations Included
Sample Output Images:
+ Confusion Matrix
+ ROC Curves
+ Accuracy - Loss Graphs
+ Random predictions with labels
These visualizations provide insights into model performance.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib & Seaborn
- scikit-learn
- Jupyter Notebook

## How to Run the Project
  
1. Install dependencies
   * pip install tensorflow keras numpy matplotlib seaborn scikit-learn
2. Prepare dataset
   * Place the dataset in the directory.
3. Train the model
   * Run the inceptionv3_training.ipynb notebook.
4. Evaluate the model
   * Run the evaluation.ipynb notebook.

## Applications

- Automated blood image analysis
- Preprocessing for medical diagnosis
- Supporting laboratory workflows
- Training medical AI systems

  
