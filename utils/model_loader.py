"""
Model loading and prediction utilities for the Brain Tumor Classifier.
Supports both binary (tumor/normal) and multi-class (4-class) models.
"""

import os
import numpy as np
from PIL import Image

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras

# Class definitions
BINARY_CLASSES = ['Normal', 'Tumor']
MULTI_CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

_model = None
_class_names = None


def load_model_from_path(model_path):
    """Load a Keras model and auto-detect class configuration."""
    global _model, _class_names

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _model = keras.models.load_model(model_path)

    # Auto-detect: binary (1 output) vs multi-class (N outputs)
    output_shape = _model.output_shape
    num_outputs = output_shape[-1] if isinstance(output_shape[-1], int) else 1

    if num_outputs == 1:
        _class_names = BINARY_CLASSES
    elif num_outputs == 4:
        _class_names = MULTI_CLASSES
    else:
        _class_names = [f"Class {i}" for i in range(num_outputs)]

    print(f"[Model] Loaded: {model_path}")
    print(f"[Model] Output units: {num_outputs} → Classes: {_class_names}")

    return _model


def get_model():
    """Return the loaded model."""
    return _model


def get_class_names():
    """Return class names based on the loaded model."""
    return _class_names if _class_names else BINARY_CLASSES


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for ResNet50 prediction.

    Args:
        image_path: Path to the image file
        target_size: Tuple (height, width) for resizing

    Returns:
        img_array: Preprocessed array of shape (1, 224, 224, 3)
        original_img: Original resized image as numpy array
    """
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(target_size)
    original_img = np.array(img_resized)

    img_array = original_img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_img


def predict(image_path):
    """
    Run prediction on an image.

    Args:
        image_path: Path to the image file

    Returns:
        dict with keys:
            - predicted_class: str
            - confidence: float (0-100)
            - all_probabilities: dict {class_name: probability}
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model_from_path() first.")

    img_array, original_img = preprocess_image(image_path)
    predictions = _model.predict(img_array, verbose=0)

    class_names = get_class_names()

    if predictions.shape[-1] == 1:
        # Binary classification (sigmoid output)
        prob = float(predictions[0][0])
        probabilities = {
            class_names[0]: round((1 - prob) * 100, 2),
            class_names[1]: round(prob * 100, 2)
        }
        # Standard 0.5 threshold for binary sigmoid classification
        predicted_class = class_names[1] if prob >= 0.5 else class_names[0]
        confidence = max(prob, 1 - prob) * 100
    else:
        # Multi-class (softmax output)
        probs = predictions[0]
        # Calculate percentages first
        probabilities = {
            class_names[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(class_names))
        }
        
        # Get the highest probability class
        pred_idx = int(np.argmax(probs))
        predicted_class = class_names[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        
        # Custom Threshold Logic: Only classify as Tumor if confidence > 70%
        # Check if the predicted class is actually a tumor type (not "No Tumor" or "Normal")
        is_tumor_class = 'no' not in predicted_class.lower() and 'normal' not in predicted_class.lower()

        if is_tumor_class and confidence < 70:
             # Find a "No Tumor" or "Normal" class to fallback to
             normal_indices = [i for i, name in enumerate(class_names) if 'no' in name.lower() or 'normal' in name.lower()]
             
             if normal_indices:
                 fallback_idx = normal_indices[0]
                 predicted_class = class_names[fallback_idx]
                 # Use the probability of the Normal class, or if it's very low, maybe just show the low tumor confidence?
                 # Let's use the probability of the Normal class to be consistent.
                 confidence = float(probs[fallback_idx]) * 100

    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2),
        'all_probabilities': probabilities,
        'img_array': img_array,
        'original_img': original_img
    }
