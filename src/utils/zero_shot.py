from config import *
from dataloader import (test_dataloader,
                        dataset,
                        datasets,
                        )
import random
import time
import torch
from typing import Tuple, List
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
def zero_shot_example(model: CLIPModel, processor: CLIPProcessor) -> None:
    """
    Run a single zero-shot classification example.

    Args:
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.
    """
    # Create a text description (two variants: incorrect and correct)
    random_number: int = random.randint(0, len(datasets['test']))
    img, idx_label = datasets['test'].__getitem__(random_number)
    label: str = dataset.classes[idx_label]
    texts: List[str] = [f"this is not a {label} brand", f"this is a {label} brand"]

    # Process input data
    inputs: dict = processor(text=texts, images=img, return_tensors="pt", do_rescale=False,
                             padding=True).to(device)

    # Forward pass
    outputs: torch.Tensor = model(**inputs)

    # Evaluate maximum logits
    logits_per_image: torch.Tensor = outputs.logits_per_image  # shape (1, 2)
    probs: torch.Tensor = logits_per_image.softmax(dim=1)  # shape (1, 2)

    if probs.argmax(dim=1).item() == 1:
        print(f'Correct prediction for {label}')
    else:
        print(f'Incorrect prediction for {label}')

def zero_shot_test(model: CLIPModel, processor: CLIPProcessor) -> Tuple[int, int]:
    """
    Run a zero-shot test on the model with progress bar and timing.

    Args:
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.

    Returns:
        Tuple[int, int]: A tuple containing the number of correct answers and total answers.
    """
    right_answers: int = 0
    count_answers: int = 0
    total_time: float = 0

    # Wrap the outer loop with tqdm for a progress bar
    for imgs, idx_labels in tqdm(test_dataloader, desc="Processing batches"):
        batch_start_time: float = time.time()

        for i in range(len(imgs)):
            label: str = dataset.classes[idx_labels[i]]
            texts: List[str] = [f"this is not a {label} brand", f"this is a {label} brand"]

            inputs: dict = processor(text=texts, images=imgs[i], return_tensors="pt", do_rescale=False,
                                     padding=True).to(device)

            outputs: torch.Tensor = model(**inputs)

            logits_per_image: torch.Tensor = outputs.logits_per_image
            probs: torch.Tensor = logits_per_image.softmax(dim=1)

            count_answers += 1

            if probs.argmax(dim=1).item() == 1:
                right_answers += 1

        batch_end_time: float = time.time()
        batch_time: float = batch_end_time - batch_start_time
        total_time += batch_time

        # Calculate average time per example in the batch
        avg_time_per_example: float = batch_time / len(imgs)

    # Calculate and print overall results
    overall_avg_time: float = total_time / count_answers
    print(f'Percent of right answers: {100 * right_answers / count_answers:.2f}%')
    print(f'\nOverall average time per example: {overall_avg_time:.4f} seconds')

    return right_answers, count_answers





def zero_shot_test_batch(model: CLIPModel, processor: CLIPProcessor, test_dataloader: torch.utils.data.DataLoader,
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
    right_answers: int = 0
    count_answers: int = 0

    model.eval()
    with torch.no_grad():
        for imgs, idx_labels in tqdm(test_dataloader, desc="Processing batches"):
            batch_size = imgs.size(0)

            # Generate text pairs for each image in the batch
            text_pairs: List[List[str]] = [
                [f"this is not a {dataset.classes[idx]} brand", f"this is a {dataset.classes[idx]} brand"]
                for idx in idx_labels
            ]

            # Flatten the list of text pairs
            texts: List[str] = [text for pair in text_pairs for text in pair]

            # Process inputs
            inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True, truncation=True).to(device)

            # Get model outputs
            outputs = model(**inputs)

            # Prepare logits

            prep_per_image = []
            for i, logits in enumerate(outputs.logits_per_image):
                prep_per_image.append(logits[i*2 : i*2 +2])

            for t in prep_per_image:
                t = t.softmax(dim=0)
                count_answers += 1

                if prep_per_image[0].softmax(dim=0).argmax(dim=0).item()==1:
                    right_answers += 1

    accuracy = 100 * right_answers / count_answers
    print(f'Percent of right answers: {accuracy:.2f}%')

    return right_answers, count_answers





if __name__ == "__main__":
    model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor: CLIPProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Uncomment the following line to run a single example
    zero_shot_example(model, processor) # Expected accuracy: ~65.82%, time: 93 sec

    # При наличии любого автомобильного бренда, будет выбираться он в датасете с автомобилями 100%-98%
    # При сравнении всех автомобилей будет выбираться только одна марка
    zero_shot_test_batch(model, processor, test_dataloader, device)  # time: 17 sec



