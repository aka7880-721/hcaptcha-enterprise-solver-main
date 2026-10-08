import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MODEL_PATH,
    IMAGE_SIZE,
    MODELS_DIR,
    PROCESSED_TRAINING_DIR,
    PROCESSED_VALIDATION_DIR,
    TRAINING_DIR,
    VALIDATION_DIR,
)
from src.model import build_pose_classifier


def _pick_data_dir(processed: Path, raw: Path) -> Path:
    if processed.is_dir() and any(processed.iterdir()):
        return processed
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pose classifier CNN.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Train on training/ instead of processed/training/ even if processed exists.",
    )
    args = parser.parse_args()

    train_dir = TRAINING_DIR if args.raw else _pick_data_dir(PROCESSED_TRAINING_DIR, TRAINING_DIR)
    val_dir = VALIDATION_DIR if args.raw else _pick_data_dir(PROCESSED_VALIDATION_DIR, VALIDATION_DIR)

    if not train_dir.is_dir():
        raise SystemExit(f"Training directory not found: {train_dir}")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        class_mode="binary",
    )

    validation_generator = None
    if val_dir.is_dir() and any(val_dir.iterdir()):
        validation_generator = val_datagen.flow_from_directory(
            str(val_dir),
            target_size=IMAGE_SIZE,
            batch_size=args.batch_size,
            class_mode="binary",
            shuffle=False,
        )

    model = build_pose_classifier()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            str(DEFAULT_MODEL_PATH),
            monitor="val_accuracy" if validation_generator else "accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss" if validation_generator else "loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    steps = max(1, math.ceil(train_generator.samples / args.batch_size))
    val_steps = (
        max(1, math.ceil(validation_generator.samples / args.batch_size))
        if validation_generator
        else None
    )

    print(f"Training data: {train_dir}")
    if validation_generator:
        print(f"Validation data: {val_dir}")
    print(f"Classes (binary 0/1): {train_generator.class_indices}")

    model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    model.save(DEFAULT_MODEL_PATH)
    print(f"Saved model to {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
