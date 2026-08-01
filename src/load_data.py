"""
==========================================================
Plant Disease Detection Using Computer Vision & Deep Learning

File: data_loader.py

Description:
Loads the PlantVillage dataset and creates TensorFlow
training and validation datasets.

==========================================================
"""

import tensorflow as tf

from config import (
    DATASET_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    RANDOM_STATE,
)


def load_dataset():
    """
    Load PlantVillage dataset and split into training
    and validation datasets.

    Returns
    -------
    train_dataset : tf.data.Dataset
        Training dataset.

    validation_dataset : tf.data.Dataset
        Validation dataset.

    class_names : list
        List of disease class names.
    """

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=RANDOM_STATE,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=RANDOM_STATE,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_dataset.class_names

    return train_dataset, validation_dataset, class_names


def get_class_count(class_names):
    """
    Display total number of classes.

    Parameters
    ----------
    class_names : list

    Returns
    -------
    int
    """

    return len(class_names)


def dataset_information(train_dataset, validation_dataset, class_names):
    """
    Print dataset information.
    """

    print("=" * 60)
    print("PLANT DISEASE DATASET INFORMATION")
    print("=" * 60)

    print(f"Number of Classes      : {len(class_names)}")
    print(f"Training Batches       : {len(train_dataset)}")
    print(f"Validation Batches     : {len(validation_dataset)}")
    print(f"Image Size             : {IMAGE_SIZE}")
    print(f"Batch Size             : {BATCH_SIZE}")

    print("\nDisease Classes\n")

    for index, disease in enumerate(class_names, start=1):
        print(f"{index}. {disease}")


def main():
    """
    Main function.
    """

    train_dataset, validation_dataset, class_names = load_dataset()

    dataset_information(
        train_dataset,
        validation_dataset,
        class_names,
    )


if __name__ == "__main__":
    main()
