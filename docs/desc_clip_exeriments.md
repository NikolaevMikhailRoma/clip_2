# CLIP Model Experiments for Brand Recognition

## Overview

This project explores the use of the CLIP (Contrastive Language-Image Pre-training) model for brand recognition tasks, specifically focusing on automobile brands. The experiments are designed to test the model's ability to identify and classify different car brands from images using various approaches.

## Dataset

The dataset consists of images of different car brands. The data is organized in a folder structure where each subfolder represents a specific brand. This flexible structure allows for easy addition or modification of brands in the dataset.

## Experiments

The project includes several different approaches to brand recognition:

### 1. Zero-Shot Learning (zero_shot.py)

This experiment tests the CLIP model's ability to recognize brands without any specific training on the task.

- **Method**: The model is given two text prompts for each image: "this is not a [brand] brand" and "this is a [brand] brand". It then decides which statement is more likely.
- **Results**: 
  - Accuracy: 65%

### 2. Few-Shot Learning (few_shots.py)

This experiment tests the model's performance when given a small number of examples to learn from.

- **Method**: Two approaches are implemented:
  1. Binary classification: Determining whether an image belongs to a specific car brand or not
  2. Multi-class classification across all brands
- **Results**:
  - Binary classification accuracy: 100% (The model consistently identifies that the image is of a car brand)
  - Multi-class classification: The model consistently selects the same brand for all images, indicating poor performance without fine-tuning

### 3. Fine-Tuning with Custom Layers (fine_tune.py)

This experiment involves adding custom layers to the CLIP model and fine-tuning it on the specific brand recognition task.

- **Method**: 
  1. Stage 1: Train only the custom layers
  2. Stage 2: Fine-tune the entire model including CLIP layers
- **Results**: 
  - After Stage 1 (custom layers only): 82% accuracy for classification task
  - Further accuracy improvements require fine-tuning the entire model, not just the last layers
  - Fine-tuning only the last layers does not yield significant improvements

### 4. Machine Learning on CLIP Features (clip_ml.py)

This experiment uses the CLIP model as a feature extractor and applies traditional machine learning models to the extracted features.

- **Method**: 
  1. Extract features using CLIP
  2. Train various ML models on these features:
     - SVM (Support Vector Machine)
     - LightGBM
     - KNN (K-Nearest Neighbors) [commented out in code]
     - Random Forest [commented out in code]
     - XGBoost [commented out in code]
  3. Evaluate models using cross-validation and a test set
- **Results**: 
  - Accuracy: 75-80%
  - Training is very fast compared to fine-tuning approaches

## Key Findings

1. Zero-shot learning shows moderate success (65% accuracy) in brand recognition tasks.
2. Few-shot learning excels at binary classification (is/is not a car brand) but struggles with multi-class brand classification.
3. Fine-tuning the CLIP model with custom layers allows for significant improvement in classification accuracy (up to 82%), but requires fine-tuning the entire model for best results.
4. Using CLIP as a feature extractor and applying traditional ML models offers a good balance between accuracy (75-80%) and training speed.
5. The choice of approach depends on the specific requirements of the task: speed, accuracy, or ability to generalize to new brands.

## Conclusion

The CLIP model demonstrates versatility in brand recognition tasks, with each approach offering different trade-offs. Zero-shot and few-shot learning show promise for quick deployment and generalization. Fine-tuning provides the highest accuracy but requires more computational resources. The ML approach offers a good balance of accuracy and training speed. The best approach depends on the specific use case and available resources.

## Future Work

1. Expand the dataset with more diverse brand images
2. Experiment with different text prompts for zero-shot and few-shot learning
3. Optimize hyperparameters for fine-tuning and ML models
4. Explore ensemble methods combining different approaches
5. Investigate the model's performance on brands from different industries
6. Conduct a thorough comparison of all ML models (including those currently commented out) to identify the best performer for this specific task