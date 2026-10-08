from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_DIR = PROJECT_ROOT / "training"
VALIDATION_DIR = PROJECT_ROOT / "validation"
PROCESSED_DIR = PROJECT_ROOT / "processed"
PROCESSED_TRAINING_DIR = PROCESSED_DIR / "training"
PROCESSED_VALIDATION_DIR = PROCESSED_DIR / "validation"

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "pose_detection_model.h5"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 50

CLASS_NAMES = ("jumping", "sitting")
# flow_from_directory uses alphabetical order: jumping=0, sitting=1 for binary with class_mode='binary'

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
