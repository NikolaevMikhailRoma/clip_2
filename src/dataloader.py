import os
from PIL import Image
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import RandomHorizontalFlip, ColorJitter, RandomRotation, Normalize
from config import *
# from src.config import *


# Function to validate and clean images
def validate_and_clean_images(local_data_folder: str) -> None:
    """
    Validate and clean images by removing invalid files.

    Args:
        local_data_folder (str): Path to the local data folder containing images.
    """
    allowed_extensions = ('.jpeg', '.jpg', '.png')

    for root, _, files in os.walk(local_data_folder):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file has an allowed extension
            if file.lower().endswith(allowed_extensions):
                try:
                    img = Image.open(file_path)
                    # Convert the image to RGB if it's not
                    if img.mode != 'RGB':
                        print(f"Removing {file_path}: Invalid image format")
                        os.remove(file_path)
                except Exception as e:
                    print(f"Removing {file_path}: Invalid image format")
                    os.remove(file_path)
            else:
                print(f"Removing {file_path}: Invalid file extension")
                os.remove(file_path)


# Function to get image paths and labels from directory
def get_image_paths_n_labels(root_dir: str) -> tuple[list[str], list[str], list[str]]:
    """
    Get image file paths and corresponding labels from a root directory.

    Args:
        root_dir (str): Path to the root directory containing images categorized by labels.

    Returns:
        tuple[list[str], list[str], list[str]]: A tuple containing image paths, labels, and unique label names.
    """
    image_paths = []
    labels = []
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                image_paths.append(img_path)
                labels.append(label)
    unique_labels = list(set(labels))
    return image_paths, labels, unique_labels


# Function to split data into training, validation, and test indices
def get_data_indices(data: ImageFolder,
                     test_proportion: float = global_test_proportion,
                     val_proportion: float = global_test_proportion,
                     random_state: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training, validation, and test indices.

    Args:
        data (np.ndarray): Input dataset.
        test_proportion (float): Proportion of data for the test set.
        val_proportion (float): Proportion of data for the validation set.
        random_state (int | None): Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Indices for training, validation, and test sets.
    """
    # Get dataset length
    data_len = len(data)

    # Calculate sizes for test and validation sets
    test_size = int(data_len * test_proportion)
    val_size = int(data_len * val_proportion)

    # Generate array of indices
    indices = np.arange(data_len)

    # Shuffle indices if random state is provided
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(indices)

    # Split indices
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    return train_indices, val_indices, test_indices


# Image transformations for data augmentation
train_transform = Compose([
    Resize((224, 224)),  # Resize image
    RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, and saturation
    RandomRotation(degrees=15),  # Random rotation
    ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (if needed)
])

# Image transformations for validation and test (no augmentation)
test_transform = Compose([
    Resize((224, 224)),  # Resize image
    ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (if needed)
])


# Load datasets with transformations
dataset = ImageFolder(data_folder, transform=test_transform)
dataset_augment = ImageFolder(data_folder, transform=train_transform)

# Split dataset into train, validation, and test sets
train_indices, val_indices, test_indices = get_data_indices(dataset, random_state=GLOBAL_RAND_STATE)

# Create subsets for each dataset split
datasets = {
    'train': Subset(dataset_augment, train_indices),
    'val': Subset(dataset, val_indices),
    'test': Subset(dataset, test_indices)
}

# Create data loaders for each dataset split
train_dataloader = DataLoader(datasets['train'],
                              # num_workers=4,
                              batch_size=global_batch_size, shuffle=True)
val_dataloader = DataLoader(datasets['val'], batch_size=global_batch_size, shuffle=True)
test_dataloader = DataLoader(datasets['test'], batch_size=global_batch_size, shuffle=True)

if __name__ == '__main__':

    # Display an image and its label
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze().permute(1, 2, 0)
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")

