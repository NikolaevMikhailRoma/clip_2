import unittest
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from src.predictor import Predictor
from src.config import config, PROJECT_ROOT

class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method is called once before any tests in this class are run
        cls.model_path = config.MODEL_PATH
        cls.predictor = Predictor(cls.model_path)
        cls.test_image_path = os.path.join(PROJECT_ROOT, 'test_image.jpg')

    def test_predictor_initialization(self):
        # Test if the Predictor is initialized correctly
        self.assertIsInstance(self.predictor, Predictor)
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.class_names)

    def test_preprocess_image(self):
        # Test image preprocessing
        if os.path.exists(self.test_image_path):
            preprocessed = self.predictor.preprocess_image(self.test_image_path)
            self.assertIsInstance(preprocessed, torch.Tensor)
            self.assertEqual(preprocessed.shape[1:], (3, 224, 224))
        else:
            self.skipTest(f"Test image not found at {self.test_image_path}")

    def test_predict(self):
        # Test prediction
        if os.path.exists(self.test_image_path):
            prediction = self.predictor.predict(self.test_image_path)
            self.assertIsInstance(prediction, str)
            self.assertIn(prediction, self.predictor.class_names)
        else:
            self.skipTest(f"Test image not found at {self.test_image_path}")

    def test_invalid_image_path(self):
        # Test handling of invalid image path
        with self.assertRaises(FileNotFoundError):
            self.predictor.predict('non_existent_image.jpg')

if __name__ == '__main__':
    unittest.main()