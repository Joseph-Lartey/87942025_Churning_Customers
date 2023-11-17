# 87942025_Churning_Customers

# Customer Churn Prediction

This project aims to predict customer churn in a telecommunications company using machine learning and neural networks. 
The provided code includes data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Machine Learning Model](#machine-learning-model)
  - [Neural Network Model](#neural-network-model)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Model Deployment with Streamlit](#model-deployment-with-streamlit)

## Introduction

Customer churn is a critical metric for telecommunications companies. This project utilizes machine learning and neural networks to predict customer churn based on various features. 
The provided code includes data loading, preprocessing, feature selection, model training (Random Forest and Neural Network), and a Streamlit app for deployment.

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Machine Learning model training (Random Forest Classifier)
- Neural Network model training (using TensorFlow and Keras)
- Hyperparameter tuning using Grid Search
- Model evaluation (accuracy, AUC score, confusion matrix)
- Deployment with Streamlit

## Getting Started

### Prerequisites

Before running the code, make sure you have the following prerequisites:

- Python (version 3.6 or later)
- Required Python packages: pandas, numpy, scikit-learn, matplotlib, seaborn, tensorflow, streamlit


## Usage

### Data Preprocessing

The dataset used for this project is stored in `datasets/CustomerChurn_dataset.csv`. The Jupyter notebook (`churning.ipynb`) demonstrates the data preprocessing steps, including handling missing values, label encoding, and standardization.

### Exploratory Data Analysis

Explore the dataset and understand its characteristics using the provided code. Visualizations such as count plots and heatmaps help analyze the distribution of features and their correlation with the target variable.

### Machine Learning Model

Train a Random Forest Classifier using the processed data. The notebook includes feature selection using Recursive Feature Elimination with Cross-Validation (RFECV) and evaluation metrics like accuracy.

### Neural Network Model

Utilize TensorFlow and Keras to create a neural network model. Train the model and evaluate its performance on the test set.

### Hyperparameter Tuning

Perform hyperparameter tuning using Grid Search with scikeras for the neural network model. Save the best model for deployment.

### Model Deployment with Streamlit

The Streamlit app (`app.py`) allows users to input customer information and predicts churn using the deployed model. To run the app, use the following command:
streamlit run app.py

### Deployment Video Link


