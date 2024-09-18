import torch

# Configuration parameters
# batch_size = 32
data_folder = '/Users/admin/projects/model_clip/data/cars'  # replace with the path to your data folder
GLOBAL_RAND_STATE = 42
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
augment_factor = 0
global_test_proportion = 0.15
global_batch_size = 2**9
global_epochs = 1000
# global_epochs = 10

# Training mode: 'clip+nn' for the current approach, 'clip+ml' for the new ML approach
# training_mode = 'clip+ml'

import torch

# Путь к сохраненной модели
MODEL_PATH = "path/to/your/fine_tuned_model"

# Устройство для выполнения вычислений
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Другие конфигурационные параметры...

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import torch
from typing import Dict
from src.config import MODEL_PATH, DEVICE

# Initialize FastAPI app
app = FastAPI()

# Load the model and processor
model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

import torch

# Существующие параметры
data_folder = '/Users/admin/projects/model_clip/data/cars'
GLOBAL_RAND_STATE = 42
# device = ...  # оставьте существующую логику выбора устройства

# Новые параметры
MODEL_PATH = "path/to/your/fine_tuned_model"  # укажите правильный путь
API_HOST = "0.0.0.0"
API_PORT = 8000

# Остальные параметры оставьте без изменений