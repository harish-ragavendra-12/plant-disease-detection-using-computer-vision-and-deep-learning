from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# DATA DIRECTORIES
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# PlantVillage Dataset Directory
DATASET_DIR = RAW_DATA_DIR / "PlantVillage"

# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

MODELS_DIR = PROJECT_ROOT / "models"

FIGURES_DIR = PROJECT_ROOT / "figures"

CONFUSION_MATRIX_FIGURE = FIGURES_DIR / "confusion_matrix.png"

ACCURACY_FIGURE = FIGURES_DIR / "accuracy_curve.png"

LOSS_FIGURE = FIGURES_DIR / "loss_curve.png"

# ==========================================================
# MODEL PARAMETERS
# ==========================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

RANDOM_STATE = 42

VALIDATION_SPLIT = 0.20

EPOCHS = 10

LEARNING_RATE = 0.0001

# ==========================================================
# MODEL FILE
# ==========================================================

MODEL_FILE = MODELS_DIR / "plant_disease_classifier.keras"

HISTORY_FILE = MODELS_DIR / "training_history.json"

CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"