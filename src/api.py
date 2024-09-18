from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
from src.predictor import Predictor
from src.config import config
import os

app = FastAPI()

# Initialize the predictor
model_path = config.MODEL_PATH
predictor = Predictor(model_path)


class ImageURL(BaseModel):
    url: str


@app.post("/predict/")
async def predict_image(image_url: ImageURL):
    try:
        # Download the image
        response = requests.get(image_url.url)
        image = Image.open(BytesIO(response.content)).convert('RGB')

        # Save the image temporarily
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        # Make prediction
        predicted_class = predictor.predict(temp_image_path)

        # Remove the temporary image
        os.remove(temp_image_path)

        return {"predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to the Brand Recognition API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)