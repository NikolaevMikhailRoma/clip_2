from config import *
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader, Dataset
from dataloader import get_image_paths_n_labels, test_dataloader, dataset, datasets
import time
from tqdm import tqdm
from typing import Tuple, List


def few_shot_test_batch(model: CLIPModel, processor: CLIPProcessor, test_dataloader: torch.utils.data.DataLoader,
                         device: torch.device) -> Tuple[int, int]:
    """
    Run a zero-shot test on the model with batch processing.

    Args:
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): The device to run the model on.

    Returns:
        Tuple[int, int]: A tuple containing the number of correct answers and total answers.
    """
    correct_predictions: int = 0
    total_predictions: int = 0
    start_time: float = time.time()

    model.eval()
    with torch.no_grad():
        for images, label_indices in tqdm(test_dataloader, desc="Processing batches"):
            batch_size = images.size(0)

            # Generate text pairs for each image in the batch
            text_pairs: List[List[str]] = [
                [f"this is not a {dataset.classes[idx]} brand", f"this is a {dataset.classes[idx]} brand"]
                for idx in label_indices
            ]

            # text_pairs = [f"this is not a {dataset.classes[idx]} brand" for idx in label_indices ]
            # [text_pairs.append(f"this is a {dataset.classes[idx]} brand") for idx in label_indices]

            # Flatten the list of text pairs
            flattened_texts: List[str] = [text for pair in text_pairs for text in pair]

            # Process inputs
            inputs = processor(text=flattened_texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)

            # Get model outputs
            outputs = model(**inputs)

            # Prepare logits for each image
            logits_per_image = []
            for i, logits in enumerate(outputs.logits_per_image):
                logits_per_image.append(logits[i*2 : i*2 + 2])
                # logits_per_image.append(logits[[i, i + 64]])
                # logits_per_image.append(logits)

            # Calculate predictions and update counters
            for i, logits in enumerate(logits_per_image):
                probabilities = logits.softmax(dim=0)
                total_predictions += 1

                if probabilities.argmax(dim=0).item() == 1:
                # if probabilities.argmax(dim=0).item() == i*2 + 1:
                    correct_predictions += 1

    # Calculate accuracy
    accuracy = 100 * correct_predictions / total_predictions
    print(f'Percent of correct answers: {accuracy:.2f}%')

    # Calculate and print total execution time
    end_time: float = time.time()
    total_time: float = end_time - start_time
    print(f'Total execution time: {total_time:.2f} seconds')

    return correct_predictions, total_predictions


def few_shot_test_2_batch(model: CLIPModel, processor: CLIPProcessor, test_dataloader: torch.utils.data.DataLoader,
                         device: torch.device) -> Tuple[int, int]:
    """
    Run a zero-shot test on the model with batch processing.

    Args:
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): The device to run the model on.

    Returns:
        Tuple[int, int]: A tuple containing the number of correct answers and total answers.
    """
    correct_predictions: int = 0
    total_predictions: int = 0
    start_time: float = time.time()

    # Generate text pairs for each image in the batch
    text_pairs_dataset = [f"this is a {label} brand" for label in dataset.classes] ####
    # text_pairs_dataset = [f"{label}" for label in dataset.classes]
    text_pairs_dataset.append("this is unknown brand")


    model.eval()
    with torch.no_grad():
        for images, label_indices in tqdm(test_dataloader, desc="Processing batches"):
            batch_size = images.size(0)

            # Process inputs
            inputs = processor(text=text_pairs_dataset, images=images, return_tensors="pt", padding=True, truncation=True).to(device)

            # Get model outputs
            outputs = model(**inputs)

            # Prepare logits for each image
            logits_per_image = outputs.logits_per_image.softmax(dim=1)
            true_logits = torch.zeros_like(logits_per_image)
            error = abs(logits_per_image - true_logits)
            for i, idx in enumerate(label_indices):
                true_logits[i][idx] = 1

            total_predictions += batch_size
            correct_predictions += sum(logits_per_image.argmax(1) == label_indices.to(device))

    # Calculate accuracy
    accuracy = 100 * correct_predictions / total_predictions
    print(f'Percent of correct answers: {accuracy:.2f}%')

    # Calculate and print total execution time
    end_time: float = time.time()
    total_time: float = end_time - start_time
    print(f'Total execution time: {total_time:.2f} seconds')

    return correct_predictions, total_predictions


if __name__ == "__main__":

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # При наличии любого автомобильного бренда, будет выбираться любой автомобильный бренд в лейбле да\нет - 100%-98%
    # в датасете с автомобилями
    # При сравнении всех брендов автомобилей будет выбираться только одна марка!
    few_shot_test_batch(model, processor, test_dataloader, device)  # time: 17 sec

    # с данным подходом будет выбираться только одна марка автомобиля всегда
    few_shot_test_2_batch(model, processor, test_dataloader, device)  # 4.68%, time: 17 sec