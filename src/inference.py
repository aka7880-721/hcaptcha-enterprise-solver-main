from pathlib import Path
from typing import Optional

from tensorflow.keras.models import load_model

from src.config import CLASS_NAMES, DEFAULT_MODEL_PATH, IMAGE_EXTENSIONS
from src.preprocess import preprocess_image_path


def load_classifier(model_path: Optional[Path] = None):
    path = model_path or DEFAULT_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Run train.py first (or place a .h5 file there)."
        )
    return load_model(path)


def predict_proba_sitting(model, image_path: Path, blur_radius: float = 1.0) -> float:
    batch = preprocess_image_path(image_path, blur_radius=blur_radius)
    proba = model.predict(batch, verbose=0)
    return float(proba[0][0])


def predict_label(
    model, image_path: Path, blur_radius: float = 1.0
) -> tuple[str, float, float]:
    p_sitting = predict_proba_sitting(model, image_path, blur_radius=blur_radius)
    p_jumping = 1.0 - p_sitting
    label = "sitting" if p_sitting >= 0.5 else "jumping"
    return label, p_sitting, p_jumping


def select_matching_tiles(
    model,
    tiles_dir: Path,
    target_class: str,
    threshold: float = 0.5,
    blur_radius: float = 1.0,
) -> list[tuple[int, Path, float]]:
    target = target_class.strip().lower()
    if target not in CLASS_NAMES:
        raise ValueError(f"target_class must be one of {CLASS_NAMES}, got {target!r}")

    tiles = sorted(
        p for p in tiles_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected: list[tuple[int, Path, float]] = []
    for index, tile_path in enumerate(tiles):
        p_sitting = predict_proba_sitting(model, tile_path, blur_radius=blur_radius)
        score = p_sitting if target == "sitting" else (1.0 - p_sitting)
        if score >= threshold:
            selected.append((index, tile_path, score))
    return selected
