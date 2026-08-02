"""
==========================================================
Plant Disease Detection Using Computer Vision & Deep Learning

File: evaluation.py

Description:
Evaluates the trained MobileNetV2 model and generates
evaluation metrics and visualizations.

==========================================================
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from config import (
    MODEL_FILE,
    HISTORY_FILE,
    ACCURACY_FIGURE,
    LOSS_FIGURE,
    CONFUSION_MATRIX_FIGURE,
)

from preprocessing import preprocess_dataset


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_trained_model():
    """
    Load trained model.
    """

    model = tf.keras.models.load_model(MODEL_FILE)

    return model


# ==========================================================
# LOAD TRAINING HISTORY
# ==========================================================

def load_training_history():
    """
    Load training history.
    """

    with open(HISTORY_FILE, "r") as file:

        history = json.load(file)

    return history


# ==========================================================
# PLOT ACCURACY
# ==========================================================

def plot_accuracy(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history["accuracy"], label="Training Accuracy")

    plt.plot(history["val_accuracy"], label="Validation Accuracy")

    plt.title("Training vs Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(ACCURACY_FIGURE, dpi=300)

    plt.show()


# ==========================================================
# PLOT LOSS
# ==========================================================

def plot_loss(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history["loss"], label="Training Loss")

    plt.plot(history["val_loss"], label="Validation Loss")

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(LOSS_FIGURE, dpi=300)

    plt.show()


# ==========================================================
# MODEL EVALUATION
# ==========================================================

def evaluate_model():

    train_dataset, validation_dataset, class_names = preprocess_dataset()

    model = load_trained_model()

    loss, accuracy = model.evaluate(validation_dataset)

    print("=" * 60)

    print("MODEL EVALUATION")

    print("=" * 60)

    print(f"Validation Loss      : {loss:.4f}")

    print(f"Validation Accuracy  : {accuracy:.4f}")

    predictions = model.predict(validation_dataset)

    predicted_labels = np.argmax(predictions, axis=1)

    true_labels = np.concatenate(

        [labels.numpy() for _, labels in validation_dataset]

    )

    print("\nClassification Report\n")

    print(

        classification_report(

            true_labels,

            predicted_labels,

            target_names=class_names,

        )

    )

    cm = confusion_matrix(

        true_labels,

        predicted_labels,

    )

    plt.figure(figsize=(12, 10))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=class_names,

        yticklabels=class_names,

    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(

        CONFUSION_MATRIX_FIGURE,

        dpi=300,

    )

    plt.show()


# ==========================================================
# PERFORM EVALUATION
# ==========================================================

def perform_evaluation():

    history = load_training_history()

    plot_accuracy(history)

    plot_loss(history)

    evaluate_model()


# ==========================================================
# MAIN
# ==========================================================

def main():

    perform_evaluation()


if __name__ == "__main__":
    main()
