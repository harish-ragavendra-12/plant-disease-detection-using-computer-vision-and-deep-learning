"""
==========================================================
Plant Disease Detection Using Computer Vision & Deep Learning

File: preprocessing.py

Description:
Performs image preprocessing including normalization,
data augmentation, caching, and prefetching.

==========================================================
"""

import tensorflow as tf

from load_data import load_dataset


# ==========================================================
# DATA AUGMENTATION
# ==========================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])


# ==========================================================
# NORMALIZATION
# ==========================================================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)


# ==========================================================
# PREPROCESS DATASET
# ==========================================================

def preprocess_dataset():
    """
    Preprocess the PlantVillage dataset.

    Returns
    -------
    train_dataset : tf.data.Dataset

    validation_dataset : tf.data.Dataset

    class_names : list
    """

    train_dataset, validation_dataset, class_names = load_dataset()

    # Normalize images
    train_dataset = train_dataset.map(
        lambda images, labels: (
            normalization_layer(images),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    validation_dataset = validation_dataset.map(
        lambda images, labels: (
            normalization_layer(images),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Data augmentation (Training only)
    train_dataset = train_dataset.map(
        lambda images, labels: (
            data_augmentation(images, training=True),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Cache & Prefetch
    train_dataset = (
        train_dataset
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_dataset = (
        validation_dataset
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, validation_dataset, class_names


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():
    """
    Test preprocessing pipeline.
    """

    train_dataset, validation_dataset, class_names = preprocess_dataset()

    print("=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Training Batches   : {len(train_dataset)}")
    print(f"Validation Batches : {len(validation_dataset)}")
    print(f"Number of Classes  : {len(class_names)}")


if __name__ == "__main__":
    main()
