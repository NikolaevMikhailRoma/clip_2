import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from PIL import Image

from config import device, GLOBAL_RAND_STATE
from dataloader import (
    dataset, train_dataloader, val_dataloader, test_dataloader
)


def extract_clip_features(model: CLIPModel, dataloader: DataLoader, device: torch.device, split_name: str) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Extract features from CLIP model for a given dataloader.

    Args:
        model (CLIPModel): The CLIP model.
        dataloader (DataLoader): The data loader containing images and labels.
        device (torch.device): The device to run the model on.
        split_name (str): The name of the data split (e.g., "train", "validation", "test").

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - features: numpy array of shape (n_samples, n_features)
            - labels: numpy array of shape (n_samples,) containing original label indices
    """
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm(dataloader, desc=f"Extracting CLIP features for {split_name}"):
            images = images.to(device)
            image_features = model.get_image_features(images)
            features.append(image_features.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.vstack(features), np.concatenate(labels)


def encode_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Encode labels using one-hot encoding.

    Args:
        labels (np.ndarray): Array of label indices of shape (n_samples,).
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: One-hot encoded labels of shape (n_samples, num_classes).
    """
    encoder = OneHotEncoder(sparse_output=False, categories=[range(num_classes)])
    return encoder.fit_transform(labels.reshape(-1, 1))


def train_ml_models(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, num_classes: int) -> \
Dict[str, object]:
    """
    Train various ML models with cross-validation.

    Args:
        X_train (np.ndarray): Training features of shape (n_samples, n_features).
        y_train (np.ndarray): Training labels of shape (n_samples,).
        X_val (np.ndarray): Validation features of shape (n_samples, n_features).
        y_val (np.ndarray): Validation labels of shape (n_samples,).
        num_classes (int): Total number of classes.

    Returns:
        Dict[str, object]: A dictionary of trained models.
    """
    models = {
        # 'KNN': KNeighborsClassifier(),
        # 'RandomForest': RandomForestClassifier(n_estimators=100, random_state=GLOBAL_RAND_STATE),
        'SVM': SVC(kernel='rbf', probability=True, random_state=GLOBAL_RAND_STATE),
        # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=GLOBAL_RAND_STATE),
        'LightGBM': LGBMClassifier(random_state=GLOBAL_RAND_STATE),
    }

    trained_models = {}

    # Encode labels
    y_train_encoded = encode_labels(y_train, num_classes)
    y_val_encoded = encode_labels(y_val, num_classes)

    for name, model in models.items():
        print(f"Training and cross-validating {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"{name} Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Train on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        correct_predictions = np.sum(y_val_pred == y_val)
        total_predictions = len(y_val)
        print(f"{name} Validation Accuracy: {val_accuracy:.4f}")
        print(
            f"{name} Validation Correct Predictions: {correct_predictions}/{total_predictions} ({100 * correct_predictions / total_predictions:.2f}%)")

        # Calculate accuracy based on one-hot encoded labels
        y_val_pred_encoded = encode_labels(y_val_pred, num_classes)
        val_accuracy_ohe = np.mean(np.all(y_val_pred_encoded == y_val_encoded, axis=1))
        print(f"{name} Validation Accuracy (One-Hot Encoded): {val_accuracy_ohe:.4f}")

    return trained_models


def evaluate_models(models: Dict[str, object], X_test: np.ndarray, y_test: np.ndarray, num_classes: int) -> Tuple[
    str, float]:
    """
    Evaluate trained models on test data and return the best model.

    Args:
        models (Dict[str, object]): A dictionary of trained models.
        X_test (np.ndarray): Test features of shape (n_samples, n_features).
        y_test (np.ndarray): Test labels of shape (n_samples,).
        num_classes (int): Total number of classes.

    Returns:
        Tuple[str, float]: A tuple containing the name of the best model and its accuracy.
    """
    best_accuracy = 0
    best_model_name = ''

    # Encode test labels
    y_test_encoded = encode_labels(y_test, num_classes)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        correct_predictions = np.sum(y_pred == y_test)
        total_predictions = len(y_test)
        print(f"{name} Test Accuracy: {accuracy:.4f}")
        print(
            f"{name} Test Correct Predictions: {correct_predictions}/{total_predictions} ({100 * correct_predictions / total_predictions:.2f}%)")

        # Calculate accuracy based on one-hot encoded labels
        y_pred_encoded = encode_labels(y_pred, num_classes)
        accuracy_ohe = np.mean(np.all(y_pred_encoded == y_test_encoded, axis=1))
        print(f"{name} Test Accuracy (One-Hot Encoded): {accuracy_ohe:.4f}")

        if accuracy_ohe > best_accuracy:
            best_accuracy = accuracy_ohe
            best_model_name = name

    return best_model_name, best_accuracy


def save_model(model: object, clip_model: CLIPModel, class_names: List[str], path: str) -> None:
    """
    Save the trained model along with CLIP model and class names.

    Args:
        model (object): The trained ML model.
        clip_model (CLIPModel): The CLIP model.
        class_names (List[str]): List of class names.
        path (str): Path to save the model.
    """
    torch.save({
        'ml_model': model,
        'clip_model_state_dict': clip_model.state_dict(),
        'clip_model_name': clip_model.name_or_path,
        'class_names': class_names
    }, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: torch.device) -> Tuple[object, CLIPModel, List[str]]:
    """
    Load a saved model from a file.

    Args:
        path (str): Path to the saved model.
        device (torch.device): The device to load the model on.

    Returns:
        Tuple[object, CLIPModel, List[str]]: A tuple containing the ML model, CLIP model, and class names.
    """
    checkpoint = torch.load(path, map_location=device)
    clip_model = CLIPModel.from_pretrained(checkpoint['clip_model_name']).to(device)
    clip_model.load_state_dict(checkpoint['clip_model_state_dict'])
    ml_model = checkpoint['ml_model']
    class_names = checkpoint['class_names']
    print(f"Model loaded from {path}")
    return ml_model, clip_model, class_names


def train_clip_ml(clip_model: CLIPModel, train_dataloader: DataLoader, val_dataloader: DataLoader,
                  test_dataloader: DataLoader, device: torch.device,
                  model_save_path: str = 'best_clip_ml_model.pth') -> None:
    """
    Train ML models on CLIP features.

    Args:
        clip_model (CLIPModel): The CLIP model.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        test_dataloader (DataLoader): DataLoader for test data.
        device (torch.device): The device to run the model on.
        model_save_path (str, optional): Path to save the best model. Defaults to 'best_clip_ml_model.pth'.
    """
    # Extract features
    X_train, y_train = extract_clip_features(clip_model, train_dataloader, device, "train")
    X_val, y_val = extract_clip_features(clip_model, val_dataloader, device, "validation")
    X_test, y_test = extract_clip_features(clip_model, test_dataloader, device, "test")

    num_classes = len(dataset.classes)

    # Train ML models
    trained_models = train_ml_models(X_train, y_train, X_val, y_val, num_classes)

    # Evaluate models
    best_model_name, best_accuracy = evaluate_models(trained_models, X_test, y_test, num_classes)
    print(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")

    # Save the best model
    save_model(trained_models[best_model_name], clip_model, dataset.classes, model_save_path)


def get_or_train_model(model_path: str, device: torch.device) -> Tuple[object, CLIPModel, List[str]]:
    """
    Get a trained model or train a new one if it doesn't exist.

    Args:
        model_path (str): Path to the model file.
        device (torch.device): The device to run the model on.

    Returns:
        Tuple[object, CLIPModel, List[str]]: A tuple containing the ML model, CLIP model, and class names.
    """
    if os.path.exists(model_path):
        return load_model(model_path, device)
    else:
        print(f"Model not found at {model_path}. Training new model...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        train_clip_ml(clip_model, train_dataloader, val_dataloader, test_dataloader, device, model_path)
        return load_model(model_path, device)


def classify_image(ml_model: object, clip_model: CLIPModel, processor: CLIPProcessor,
                   image_path: str, class_names: List[str], device: torch.device) -> str:
    """
    Classify a single image using the trained ML model and CLIP.

    Args:
        ml_model (object): The trained ML model.
        clip_model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.
        image_path (str): Path to the image file.
        class_names (List[str]): List of class names.
        device (torch.device): The device to run the model on.

    Returns:
        str: The predicted class name.
    """
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_features = clip_model.get_image_features(inputs.pixel_values)

    # Use the ML model for prediction
    predicted_class_idx = ml_model.predict(image_features.cpu().numpy())[0]
    predicted_class = class_names[predicted_class_idx]

    return predicted_class


if __name__ == "__main__":
    model_path = '../best_clip_ml_model.pth'

    # Get the model (load existing or train new)
    ml_model, clip_model, class_names = get_or_train_model(model_path, device)

    # Initialize CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Example usage for image classification
    image_paths = ['/Users/admin/projects/model_clip/data/cars/Benz/2023-Mercedes-AMG-C43-sedan-9.jpg',
                   '/Users/admin/projects/model_clip/data/cars/Ford mustang/2024-ford-mustang-exterior-112-1663170333.jpg',
                   '/Users/admin/projects/model_clip/data/cars/Maserati/2022_maserati_quattroporte_sedan_trofeo_fq_oem_1_1280.jpg',
                   '/Users/admin/projects/model_clip/data/cars/Benz/2023_Mercedes_Benz_C-Class_SEO1.jpg',
                   '/Users/admin/projects/model_clip/data/cars/Benz/front-left-side-472.jpg',
                   ]

    for image_path in image_paths:
        predicted_class = classify_image(ml_model, clip_model, processor, image_path, class_names, device)
        print(f"Image: {image_path}")
        print(f"Predicted class: {predicted_class}")