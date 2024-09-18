# CLIP Model for Brand Recognition

## Overview

This project implements a brand recognition system using a fine-tuned CLIP (Contrastive Language-Image Pre-training) model. It is designed to identify and classify different car brands from images. The project now includes a FastAPI-based API for easy integration and usage.

## Features

- Fine-tuned CLIP model for car brand recognition
- Custom layers added to the CLIP model for improved performance
- Predictor class for easy image classification
- FastAPI-based API for remote image classification
- Comprehensive test suite
- Docker support for easy deployment

## Project Structure

```
model_clip_2/
├── src/
│   ├── config.py
│   ├── dataloader.py
│   ├── fine_tune.py
│   ├── predictor.py
│   ├── api.py
│   └── models/
│       └── best_custom_clip_model_stage2.pth
├── tests/
│   ├── test_predictor.py
│   └── test_data/
│       ├── test_image.jpg
│       ├── test_image_1.jpg
│       ├── test_image_2.jpg
│       └── test_image_3.jpg
├── data/
│   └── [your dataset here]
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your_username/model_clip_2.git
   cd model_clip_2
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model:

```
python src/fine_tune.py
```

### Running the API

To start the FastAPI server:

```
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Using the API

You can use the API to classify images by sending a POST request to the `/predict/` endpoint. Here's an example using Python:

```python
import requests

url = "http://localhost:8000/predict/"
data = {"url": "https://example.com/path/to/car_image.jpg"}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(f"Predicted class: {result['predicted_class']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

Replace `"https://example.com/path/to/car_image.jpg"` with the URL of the image you want to classify.

### Running Tests

To run tests:

```
python -m unittest discover tests
```

## Docker

To build and run the Docker container:

1. Build the Docker image:
   ```
   docker build -t clip-brand-recognition .
   ```

2. Run the container:
   ```
   docker run -p 8000:80 clip-brand-recognition
   ```

This will start the API server inside a Docker container, accessible at `http://localhost:8000`.

## Configuration

You can modify the configuration settings in `src/config.py`. This includes paths, batch sizes, and other hyperparameters.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Copy n enjoy]

## Contact

[NikMih] - [dog@dontknow.com]

Project Link: https://github.com/NikolaevMikhailRoma/clip_2.git