# PCA + ANN Face Recognition System

A machine learning based face recognition system that combines Principal Component Analysis (PCA) and an Artificial Neural Network (ANN) to recognize enrolled individuals from facial images.

## 📌 Project Overview

This project implements a face recognition pipeline using PCA for dimensionality reduction and feature extraction, followed by an Artificial Neural Network for classification.

The system converts facial images into numerical feature vectors, extracts important facial patterns using eigenfaces, generates compact face signatures, and uses an ANN classifier to identify the corresponding person.

An additional imposter detection experiment is also performed using unknown faces that are not part of the enrolled dataset.

## 🎯 Objectives

- Load and preprocess a facial image dataset.
- Convert facial images into numerical representations.
- Calculate the mean face.
- Perform PCA-based dimensionality reduction.
- Generate eigenfaces.
- Generate compact face signatures.
- Train an ANN classifier.
- Evaluate recognition accuracy for different values of `k`.
- Visualize accuracy using an Accuracy vs K graph.
- Evaluate classification using a confusion matrix and classification report.
- Test unknown/imposter face detection using a distance-based threshold.

## 🗂️ Dataset

The project uses a face dataset containing:

- **450 facial images**
- **9 enrolled persons**
- **270 training images**
- **180 testing images**

The facial images are organized into separate folders according to the person.

## 🧠 Methodology

The complete pipeline is:

```text
Face Dataset
     ↓
Image Preprocessing
     ↓
Face Database Creation
     ↓
Train/Test Split
     ↓
Mean Face
     ↓
Mean-Zero Data
     ↓
Surrogate Covariance Matrix
     ↓
Eigenvalues & Eigenvectors
     ↓
Eigenfaces
     ↓
Face Signatures
     ↓
ANN Classifier
     ↓
Face Recognition
     ↓
Imposter Detection
