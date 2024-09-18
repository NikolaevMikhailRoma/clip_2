import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from typing import List, Tuple
import time
from PIL import Image

from config import global_epochs, device
from dataloader import (
    dataset, train_dataloader, val_dataloader, test_dataloader
)
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CustomCLIPModel(nn.Module):
    """Custom CLIP model with additional layers for fine-tuning."""

    def __init__(self, clip_model: CLIPModel):
        super(CustomCLIPModel, self).__init__()
        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Get the output dimension of CLIP's image encoder
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = self.clip_model.get_image_features(dummy_image)
            clip_output_dim = dummy_output.shape[1]

        # Add custom layers
        self.custom_layers = nn.Sequential(
            nn.Linear(clip_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(dataset.classes))
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(image)
        return self.custom_layers(image_features)

    def unfreeze_clip(self):
        """Unfreeze CLIP model parameters for fine-tuning."""
        for param in self.clip_model.parameters():
            param.requires_grad = True

    def freeze_clip(self):
        """Unfreeze CLIP model parameters for fine-tuning."""
        for param in self.clip_model.parameters():
            param.requires_grad = False


def train_stage1(
        model: CustomCLIPModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = global_epochs
) -> CustomCLIPModel:
    """Train the custom layers of the CLIP model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.custom_layers.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True, min_lr=1e-10)

    best_val_accuracy = 0.0
    start_time = time.time()
    patience = 100
    counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        train_total_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']

        train_progress_bar = tqdm(train_dataloader, desc=f"Stage 1 Training Epoch {epoch + 1}/{num_epochs}")

        for images, labels in train_progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_total_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

            train_accuracy = 100 * train_correct / train_total
            train_avg_loss = train_total_loss / (train_progress_bar.n + 1)
            train_progress_bar.set_postfix({
                'Train Accuracy': f'{train_accuracy:.2f}%',
                'Loss': f'{train_avg_loss:.4f}',
                "Current learning rate": f'{current_lr:.2e}'
            })

        # Validation phase
        val_accuracy, val_avg_loss = evaluate(model, val_dataloader, criterion, 'Validation', epoch + 1, num_epochs)

        print(
            f'\nStage 1 Epoch {epoch + 1} - Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, 'best_custom_clip_model_stage1.pth')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")
            counter = 0
        else:
            counter += 1

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    total_time = time.time() - start_time
    print(f'Total Stage 1 training time: {total_time:.2f} seconds')

    return model


def train_stage2(
        model: CustomCLIPModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = global_epochs
) -> CustomCLIPModel:
    """Fine-tune the entire CLIP model including custom layers."""
    model.unfreeze_clip()  # Unfreeze CLIP model parameters

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # Lower learning rate for fine-tuning
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True, min_lr=1e-10)

    best_val_accuracy = 0.0
    start_time = time.time()
    patience = 100
    counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        train_total_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']

        train_progress_bar = tqdm(train_dataloader, desc=f"Stage 2 Training Epoch {epoch + 1}/{num_epochs}")

        for images, labels in train_progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_total_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

            train_accuracy = 100 * train_correct / train_total
            train_avg_loss = train_total_loss / (train_progress_bar.n + 1)
            train_progress_bar.set_postfix({
                'Train Accuracy': f'{train_accuracy:.2f}%',
                'Loss': f'{train_avg_loss:.4f}',
                "Current learning rate": f'{current_lr:.2e}'
            })

        # Validation phase
        val_accuracy, val_avg_loss = evaluate(model, val_dataloader, criterion, 'Validation', epoch + 1, num_epochs)

        print(
            f'\nStage 2 Epoch {epoch + 1} - Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, 'models/best_custom_clip_model_stage2.pth')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")
            counter = 0
        else:
            counter += 1

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    total_time = time.time() - start_time
    print(f'Total Stage 2 training time: {total_time:.2f} seconds')

    return model


def evaluate(
        model: CustomCLIPModel,
        dataloader: DataLoader,
        criterion: nn.Module,
        phase: str,
        epoch: int,
        num_epochs: int
) -> Tuple[float, float]:
    """Evaluate the model on the given dataloader."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"{phase} Epoch {epoch}/{num_epochs}")

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            accuracy = 100 * correct / total
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({
                f'{phase} Accuracy': f'{accuracy:.2f}%',
                'Loss': f'{avg_loss:.4f}'
            })

    return accuracy, avg_loss


def test(model: CustomCLIPModel, test_dataloader: DataLoader) -> None:
    """Test the model on the test dataset."""
    criterion = nn.CrossEntropyLoss()
    test_accuracy, test_avg_loss = evaluate(model, test_dataloader, criterion, 'Test', 1, 1)
    print(f'Test Accuracy: {test_accuracy:.2f}%, Test Average Loss: {test_avg_loss:.4f}')


def save_model(model: CustomCLIPModel, path: str) -> None:
    """Save the model to a file."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'clip_model_name': model.clip_model.name_or_path,
        'num_classes': len(dataset.classes),
        'class_names': dataset.classes
    }, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: torch.device) -> Tuple[CustomCLIPModel, List[str]]:
    """Load a saved model from a file."""
    checkpoint = torch.load(path, map_location=device)
    clip_model = CLIPModel.from_pretrained(checkpoint['clip_model_name']).to(device)
    model = CustomCLIPModel(clip_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model, checkpoint['class_names']


def classify_image(model: CustomCLIPModel, processor: CLIPProcessor, image_path: str, class_names: List[str]) -> str:
    """Classify a single image using the trained model."""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(inputs.pixel_values)

    predicted_class_idx = outputs.argmax(1).item()
    predicted_class = class_names[predicted_class_idx]

    return predicted_class


def fine_tune(
        clip_model: CLIPModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: torch.device,
        model_save_path: str = 'best_custom_clip_model_stage2.pth'
) -> None:
    """Two-stage fine-tuning of the CLIP model with custom layers."""
    # Stage 1: Train custom layers
    custom_model = CustomCLIPModel(clip_model).to(device)
    custom_model = train_stage1(custom_model, train_dataloader, val_dataloader)

    # Test after Stage 1
    print("Testing after Stage 1:")
    test(custom_model, test_dataloader)

    # Stage 2: Fine-tune entire model
    custom_model = train_stage2(custom_model, train_dataloader, val_dataloader)

    # Save the final model
    save_model(custom_model, model_save_path)

    # Test the best model
    print("Testing after Stage 2:")
    best_model, _ = load_model(model_save_path, device)
    test(best_model, test_dataloader)


def get_or_train_model(model_path: str, device: torch.device) -> Tuple[CustomCLIPModel, List[str]]:
    """Get a trained model or train a new one if it doesn't exist."""
    if os.path.exists(model_path):
        return load_model(model_path, device)
    else:
        print(f"Model not found at {model_path}. Training new model...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        fine_tune(clip_model, train_dataloader, val_dataloader, test_dataloader, device, model_path)
        return load_model(model_path, device)


if __name__ == "__main__":
    model_path = 'models/best_custom_clip_model_stage2.pth'

    get_new_model = True
    if get_new_model:
        if os.path.exists(model_path):
            os.remove(model_path)

    # Get the model (load existing or train new)
    model, class_names = get_or_train_model(model_path, device)

    # Initialize CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Example usage for image classification
    image_paths = [
        '/Users/admin/projects/model_clip/data/cars/Benz/2023-Mercedes-AMG-C43-sedan-9.jpg',
        '/Users/admin/projects/model_clip/data/cars/Ford mustang/2024-ford-mustang-exterior-112-1663170333.jpg',
        '/Users/admin/projects/model_clip/data/cars/Maserati/2022_maserati_quattroporte_sedan_trofeo_fq_oem_1_1280.jpg',
        '/Users/admin/projects/model_clip/data/cars/Benz/2023_Mercedes_Benz_C-Class_SEO1.jpg',
        '/Users/admin/projects/model_clip/data/cars/Benz/front-left-side-472.jpg',
    ]

    for image_path in image_paths:
        # Classify the image
        predicted_class = classify_image(model, processor, image_path, class_names)
        print(f"Image: {image_path}")
        print(f"Predicted class: {predicted_class}")