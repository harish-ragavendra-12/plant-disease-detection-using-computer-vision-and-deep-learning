"""
==========================================================
Plant Disease Detection Using Computer Vision & Deep Learning

File: prediction.py

Description:
Loads the trained model and predicts the disease
class for a single plant leaf image.

==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json

from config import (
    IMAGE_SIZE,
    MODEL_FILE,
    CLASS_NAMES_FILE,
)


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():
    """
    Load the trained model.
    """

    return tf.keras.models.load_model(MODEL_FILE)

# ==========================================================
# LOAD CLASS NAMES
# ==========================================================

def load_class_names():
    """
    Load disease class names.
    """

    with open(CLASS_NAMES_FILE, "r") as file:

        class_names = json.load(file)

    return class_names


# ==========================================================
# PREPROCESS IMAGE
# ==========================================================

def preprocess_image(image_source):
    """
    Load and preprocess a single image.

    Parameters
    ----------
    image_source : str, Path or UploadedFile

    Returns
    -------
    image
        PIL image.

    image_array
        Preprocessed image.
    """

    image = tf.keras.utils.load_img(
        image_source,
        target_size=IMAGE_SIZE,
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image, image_array


# ==========================================================
# PREDICT DISEASE
# ==========================================================

def predict_disease(image_path):
    """
    Predict the disease class for a plant leaf image.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image.

    Returns
    -------
    predicted_class : str
        Predicted disease class.

    confidence : float
        Prediction confidence (%).
    """

    # Load model and class names
    model = load_model()
    class_names = load_class_names()

    # Preprocess image
    image, image_array = preprocess_image(image_path)

    # Perform prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions)

    confidence = float(np.max(predictions) * 100)

    predicted_class = class_names[predicted_index]

    # Display prediction
    print("=" * 60)
    print("PLANT DISEASE PREDICTION")
    print("=" * 60)
    print(f"Predicted Disease : {predicted_class}")
    print(f"Confidence Score  : {confidence:.2f}%")

    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    plt.title(
        f"{predicted_class}\nConfidence: {confidence:.2f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()

    return predicted_class, confidence, predictions


# ==========================================================
# MAIN
# ==========================================================

def main():
    """
    Test prediction using a sample image.
    """

    sample_image = Path(
        "data/raw/PlantVillage/Tomato_healthy/0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL 0555.JPG"
    )

    if sample_image.exists():
        predict_disease(sample_image)
    else:
        print("Sample image not found.")
        print("Update the sample image path in prediction.py")


if __name__ == "__main__":
    main()
