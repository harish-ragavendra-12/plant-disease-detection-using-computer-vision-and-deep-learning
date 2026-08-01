"""
==========================================================
Plant Disease Detection Using Computer Vision & Deep Learning

File: eda.py

Description:
Performs Exploratory Data Analysis (EDA) on the
PlantVillage image dataset.

The generated visualizations are displayed and
saved inside the figures directory.

==========================================================
"""

import matplotlib.pyplot as plt
from collections import Counter

from config import FIGURES_DIR
from preprocessing import preprocess_dataset


# ==========================================================
# DATASET SUMMARY
# ==========================================================

def dataset_summary(train_dataset, validation_dataset, class_names):
    """
    Display dataset information.
    """

    print("=" * 60)
    print("PLANT DISEASE DATASET SUMMARY")
    print("=" * 60)

    print(f"Number of Classes      : {len(class_names)}")
    print(f"Training Batches       : {len(train_dataset)}")
    print(f"Validation Batches     : {len(validation_dataset)}")


# ==========================================================
# DISPLAY SAMPLE IMAGES
# ==========================================================

def display_sample_images(train_dataset, class_names):
    """
    Display and save sample images.
    """

    plt.figure(figsize=(12, 10))

    for images, labels in train_dataset.take(1):

        for i in range(9):

            plt.subplot(3, 3, i + 1)

            plt.imshow(images[i].numpy())

            plt.title(class_names[labels[i]])

            plt.axis("off")

    plt.suptitle("Sample Plant Leaf Images", fontsize=16)

    plt.tight_layout()

    output_file = FIGURES_DIR / "sample_images.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    print(f"Saved: {output_file}")

    plt.show()


# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

def plot_class_distribution(train_dataset, class_names):
    """
    Plot and save class distribution.
    """

    labels = []

    for _, batch_labels in train_dataset:

        labels.extend(batch_labels.numpy())

    counts = Counter(labels)

    plt.figure(figsize=(14, 6))

    plt.bar(
        range(len(class_names)),
        [counts[i] for i in range(len(class_names))]
    )

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=90
    )

    plt.xlabel("Disease Class")

    plt.ylabel("Number of Images")

    plt.title("Class Distribution")

    plt.tight_layout()

    output_file = FIGURES_DIR / "class_distribution.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    print(f"Saved: {output_file}")

    plt.show()


# ==========================================================
# AUGMENTED IMAGES
# ==========================================================

def visualize_augmented_images(train_dataset):
    """
    Display and save augmented images.
    """

    plt.figure(figsize=(12, 10))

    for images, _ in train_dataset.take(1):

        for i in range(9):

            plt.subplot(3, 3, i + 1)

            plt.imshow(images[i])

            plt.axis("off")

    plt.suptitle("Augmented Plant Leaf Images", fontsize=16)

    plt.tight_layout()

    output_file = FIGURES_DIR / "augmented_images.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    print(f"Saved: {output_file}")

    plt.show()


# ==========================================================
# PERFORM EDA
# ==========================================================

def perform_eda():
    """
    Execute complete EDA pipeline.
    """

    train_dataset, validation_dataset, class_names = preprocess_dataset()

    dataset_summary(
        train_dataset,
        validation_dataset,
        class_names,
    )

    display_sample_images(
        train_dataset,
        class_names,
    )

    plot_class_distribution(
        train_dataset,
        class_names,
    )

    visualize_augmented_images(
        train_dataset,
    )


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():

    perform_eda()


if __name__ == "__main__":
    main()
