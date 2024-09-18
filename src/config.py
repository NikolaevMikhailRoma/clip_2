# import torch
#
# # Configuration parameters
# # batch_size = 32
# data_folder = '/Users/admin/projects/model_clip/data/cars'  # replace with the path to your data folder
# GLOBAL_RAND_STATE = 42
# # Get cpu, gpu or mps device for training.
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")
#
# # device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
# augment_factor = 0
# global_test_proportion = 0.15
# global_batch_size = 2**9
# global_epochs = 1000
# # global_epochs = 10
#
# # Training mode: 'clip+nn' for the current approach, 'clip+ml' for the new ML approach
# # training_mode = 'clip+ml'
#
# import torch
#
# # Путь к сохраненной модели
# MODEL_PATH = "path/to/your/fine_tuned_model"
#
# # Устройство для выполнения вычислений
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Другие конфигурационные параметры...
#
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import requests
# from PIL import Image
# from io import BytesIO
# from transformers import CLIPProcessor, CLIPModel
# import torch
# from typing import Dict
# from src.config import MODEL_PATH, DEVICE
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # Load the model and processor
# model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#
# import torch
#
# # Существующие параметры
# data_folder = '/Users/admin/projects/model_clip/data/cars'
# GLOBAL_RAND_STATE = 42
# # device = ...  # оставьте существующую логику выбора устройства
#
# # Новые параметры
# MODEL_PATH = "path/to/your/fine_tuned_model"  # укажите правильный путь
# API_HOST = "0.0.0.0"
# API_PORT = 8000

# Остальные параметры оставьте без изменений

# import os
# from dotenv import load_dotenv
# import torch
#
#
# # Load environment variables from .env file
# load_dotenv()
#
# class BaseConfig:
#     # Common configurations
#     DATA_FOLDER = '/Users/admin/projects/model_clip/data/cars'
#     GLOBAL_RAND_STATE = 42
#     GLOBAL_BATCH_SIZE = 512
#     GLOBAL_EPOCHS = 1000
#     TEST_PROPORTION = 0.15
#     VAL_PROPORTION = 0.15
#
# class DevelopmentConfig(BaseConfig):
#     # Development-specific configurations
#     DEBUG = True
#     MODEL_PATH = './models/best_custom_clip_model_stage2.pth'
#     API_HOST = 'localhost'
#     API_PORT = 8000
#
# class TestingConfig(BaseConfig):
#     # Testing-specific configurations
#     DEBUG = True
#     MODEL_PATH = './models/best_custom_clip_model_stage2.pth'
#     API_HOST = 'localhost'
#     API_PORT = 8001
#
# class ProductionConfig(BaseConfig):
#     # Production-specific configurations
#     DEBUG = False
#     MODEL_PATH = './models/best_custom_clip_model_stage2.pth'
#     API_HOST = '0.0.0.0'
#     API_PORT = 80
#
# # Determine the current environment
# env = os.getenv('ENVIRONMENT', 'development').lower()
#
# # Export the appropriate configuration
# if env == 'production':
#     config = ProductionConfig()
# elif env == 'testing':
#     config = TestingConfig()
# else:
#     config = DevelopmentConfig()
#
# # Device configuration
# config.DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#
# print(f"Using {config.DEVICE} device")
# print(f"Current environment: {env}")

import os
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

# Define the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BaseConfig:
    # Common configurations
    # DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
    DATA_FOLDER = os.path.join('/Users/admin/projects/model_clip/data/cars')

    GLOBAL_RAND_STATE = 42
    GLOBAL_BATCH_SIZE = 512
    GLOBAL_EPOCHS = 1000
    TEST_PROPORTION = 0.15
    VAL_PROPORTION = 0.15

class DevelopmentConfig(BaseConfig):
    # Development-specific configurations
    DEBUG = True
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'best_custom_clip_model_stage2.pth')
    API_HOST = 'localhost'
    API_PORT = 8000

class TestingConfig(BaseConfig):
    # Testing-specific configurations
    DEBUG = True
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'test_model.pth')
    API_HOST = 'localhost'
    API_PORT = 8001

class ProductionConfig(BaseConfig):
    # Production-specific configurations
    DEBUG = False
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'best_custom_clip_model_stage2.pth')
    API_HOST = '0.0.0.0'
    API_PORT = 80

# Determine the current environment
env = os.getenv('ENVIRONMENT', 'development').lower()

# Export the appropriate configuration
if env == 'production':
    config = ProductionConfig()
elif env == 'testing':
    config = TestingConfig()
else:
    config = DevelopmentConfig()

# Device configuration
config.DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using {config.DEVICE} device")
print(f"Current environment: {env}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Model path: {config.MODEL_PATH}")