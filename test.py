import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from src.config import PROJECT_ROOT, TRAINING_DIR, VALIDATION_DIR


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Training dir exists: {TRAINING_DIR.is_dir()}")
    print(f"Validation dir exists: {VALIDATION_DIR.is_dir()}")
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))


if __name__ == "__main__":
    main()
