"""
==========================================================
Plant Disease Detection Using Computer Vision & Deep Learning

File: model_training.py

Description:
Builds, trains, and saves the MobileNetV2 transfer learning
model along with the training history.

==========================================================
"""

import json
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import (
    IMAGE_SIZE,
    LEARNING_RATE,
    EPOCHS,
    MODEL_FILE,
    HISTORY_FILE,
    CLASS_NAMES_FILE,
)

from preprocessing import preprocess_dataset


# ==========================================================
# BUILD MODEL
# ==========================================================

def build_model(num_classes):
    """
    Build MobileNetV2 Transfer Learning Model.

    Parameters
    ----------
    num_classes : int
        Number of disease classes.

    Returns
    -------
    tensorflow.keras.Model
        Compiled MobileNetV2 model.
    """

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SIZE + (3,),
    )

    # Freeze pretrained layers
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.30)(x)

    output = Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=output,
    )

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model():
    """
    Train the MobileNetV2 model.

    Returns
    -------
    model : tensorflow.keras.Model
        Trained model.

    history : tensorflow.keras.callbacks.History
        Training history.
    """

    train_dataset, validation_dataset, class_names = preprocess_dataset()

    model = build_model(
        len(class_names),
    )

    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    model.summary()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        verbose=1,
    )

    # Create models directory if it doesn't exist
    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save training history
    with open(HISTORY_FILE, "w") as file:
        json.dump(history.history, file, indent=4)

    print(f"\nTraining history saved to: {HISTORY_FILE}")

    with open(CLASS_NAMES_FILE, "w") as file:
        json.dump(class_names, file, indent=4)

    print(f"Class names saved to: {CLASS_NAMES_FILE}")

    # Save trained model
    model.save(MODEL_FILE)

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)
    print(f"Model saved to: {MODEL_FILE}")

    return model, history


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():
    """
    Execute model training.
    """

    train_model()


if __name__ == "__main__":
    main()
