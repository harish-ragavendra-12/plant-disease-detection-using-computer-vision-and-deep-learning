"""
==========================================================
Plant Disease Detection Using Computer Vision & Deep Learning

File: model_training.py

Description:
Builds, trains and saves the MobileNetV2 transfer learning
model.

==========================================================
"""

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
)

from preprocessing import preprocess_dataset


# ==========================================================
# BUILD MODEL
# ==========================================================

def build_model(num_classes):
    """
    Build MobileNetV2 Transfer Learning Model.
    """

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SIZE + (3,),
    )

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
    Train MobileNetV2 model.
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

    MODEL_FILE.parent.mkdir(
        exist_ok=True,
    )

    model.save(
        MODEL_FILE,
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)
    print(f"Model saved to: {MODEL_FILE}")

    return model, history


# ==========================================================
# MAIN
# ==========================================================

def main():

    train_model()


if __name__ == "__main__":
    main()
