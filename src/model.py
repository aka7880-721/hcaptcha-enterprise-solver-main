from tensorflow.keras import layers, models

from src.config import IMAGE_SIZE


def build_pose_classifier() -> models.Sequential:
    height, width = IMAGE_SIZE
    return models.Sequential(
        [
            layers.Input(shape=(height, width, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
