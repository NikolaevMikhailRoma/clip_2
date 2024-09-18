# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# from typing import List
# import os
#
# from src.config import config
# from src.fine_tune import CustomCLIPModel, fine_tune
# from src.dataloader import train_dataloader, val_dataloader, test_dataloader, dataset
# from src.logger import predictor_logger as logger
#
#
# class Predictor:
#     def __init__(self, model_path: str):
#         self.device = torch.device(config.DEVICE)
#         logger.info(f"Using device: {self.device}")
#         self.model, self.class_names = self.load_model(model_path)
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         logger.info("Predictor initialized successfully")
#
#     def load_model(self, model_path: str) -> tuple[CustomCLIPModel, List[str]]:
#         """Load the trained model and class names."""
#         if not os.path.exists(model_path):
#             logger.warning(f"Model not found at {model_path}. Training new model...")
#             clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
#             custom_model = CustomCLIPModel(clip_model)
#             fine_tune(custom_model, train_dataloader, val_dataloader, test_dataloader, model_path)
#
#         logger.info(f"Loading model from {model_path}")
#         checkpoint = torch.load(model_path, map_location=self.device)
#         clip_model = CLIPModel.from_pretrained(checkpoint['clip_model_name']).to(self.device)
#         model = CustomCLIPModel(clip_model)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.to(self.device)
#         model.eval()
#         logger.info("Model loaded successfully")
#         return model, checkpoint['class_names']
#
#     def preprocess_image(self, image_path: str) -> torch.Tensor:
#         """Preprocess the input image."""
#         logger.debug(f"Preprocessing image: {image_path}")
#         image = Image.open(image_path).convert('RGB')
#         inputs = self.processor(images=image, return_tensors="pt", padding=True)
#         return inputs.pixel_values.to(self.device)
#
#     def predict(self, image_path: str) -> str:
#         """Predict the class of the input image."""
#         logger.info(f"Predicting class for image: {image_path}")
#         preprocessed_image = self.preprocess_image(image_path)
#
#         with torch.no_grad():
#             outputs = self.model(preprocessed_image)
#
#         predicted_class_idx = outputs.argmax(1).item()
#         predicted_class = self.class_names[predicted_class_idx]
#         logger.info(f"Predicted class: {predicted_class}")
#         return predicted_class
#

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List
import os

from src.config import config
from src.fine_tune import CustomCLIPModel, fine_tune
from src.dataloader import train_dataloader, val_dataloader, test_dataloader, dataset
from src.logger import predictor_logger as logger


class Predictor:
    def __init__(self, model_path: str):
        self.device = torch.device(config.DEVICE)
        logger.info(f"Using device: {self.device}")
        self.model, self.class_names = self.load_model(model_path)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("Predictor initialized successfully")

    def load_model(self, model_path: str) -> tuple[CustomCLIPModel, List[str]]:
        """Load the trained model and class names."""
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Training new model...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            custom_model = CustomCLIPModel(clip_model)
            fine_tune(custom_model, train_dataloader, val_dataloader, test_dataloader, model_path)

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        clip_model = CLIPModel.from_pretrained(checkpoint['clip_model_name']).to(self.device)
        model = CustomCLIPModel(clip_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        logger.info("Model loaded successfully")
        return model, checkpoint['class_names']

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess the input image."""
        logger.debug(f"Preprocessing image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return inputs.pixel_values.to(self.device)

    def predict(self, image_path: str) -> str:
        """Predict the class of the input image."""
        logger.info(f"Predicting class for image: {image_path}")
        preprocessed_image = self.preprocess_image(image_path)

        with torch.no_grad():
            outputs = self.model(preprocessed_image)

        predicted_class_idx = outputs.argmax(1).item()
        predicted_class = self.class_names[predicted_class_idx]
        logger.info(f"Predicted class: {predicted_class}")
        return predicted_class

if __name__ == "__main__":
    # Test the Predictor
    model_path = config.MODEL_PATH
    predictor = Predictor(model_path)

    # Test images
    test_images = ['/Users/admin/projects/model_clip/data/cars/Benz/2023-Mercedes-AMG-C43-sedan-9.jpg',
                   '/Users/admin/projects/model_clip/data/cars/Ford mustang/2024-ford-mustang-exterior-112-1663170333.jpg',
                   '/Users/admin/projects/model_clip/data/cars/Maserati/2022_maserati_quattroporte_sedan_trofeo_fq_oem_1_1280.jpg',
                   '/Users/admin/projects/model_clip/data/cars/Benz/2023_Mercedes_Benz_C-Class_SEO1.jpg',
                   '/Users/admin/projects/model_clip/data/cars/Benz/front-left-side-472.jpg',
                   ]


    for image_path in test_images:
        if os.path.exists(image_path):
            predicted_class = predictor.predict(image_path)
            logger.info(f"Image: {image_path}")
            logger.info(f"Predicted class: {predicted_class}")
        else:
            logger.warning(f"Image not found: {image_path}")
        logger.info("---")