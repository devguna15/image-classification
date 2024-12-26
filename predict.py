import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImagePredictor:
    def __init__(self, model_path: str):
        try:
            self.model = tf.keras.models.load_model(model_path)
        except IOError:
            raise IOError(f"Failed to load model from {model_path}")

    def predict(self, image_path: str):
        try:
            # Load and preprocess the image
            image = load_img(image_path, target_size=(64, 64))
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Predict
            predictions = self.model.predict(image_array)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            return {"class": int(predicted_class), "confidence": float(confidence)}
        except Exception as e:
            raise ValueError(f"Error during prediction: {e}")
