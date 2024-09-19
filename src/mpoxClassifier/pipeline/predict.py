import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model_path = os.path.join("artifacts", "training", "model.keras")  # Ensure this matches your model file extension
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = load_model(model_path)

        # Load and preprocess the image
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Image file not found at {self.filename}")

        test_image = image.load_img(self.filename, target_size=(456, 456))  # Ensure this size matches your model's expected input
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)  # Expand dimensions to match input shape
        test_image = test_image / 255.0  # Normalize the image to match the model's input requirements

        # Predict the class probabilities
        result = model.predict(test_image)

        # Get the index of the highest probability
        predicted_class_index = np.argmax(result, axis=1)[0]

        # Map the index to the corresponding class label
        class_labels = ['Chickenpox', 'Cowpox', 'Healthy', 'HFMD', 'Measles', 'Monkeypox']
        prediction = class_labels[predicted_class_index]

        return [{"image": prediction}]
