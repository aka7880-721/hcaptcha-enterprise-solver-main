from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.config import IMAGE_SIZE


def preprocess_pil_image(img: Image.Image, blur_radius: float = 1.0) -> Image.Image:
    rgb = img.convert("RGB")
    if blur_radius > 0:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return rgb.resize(IMAGE_SIZE)


def preprocess_image_path(path: Path, blur_radius: float = 1.0) -> np.ndarray:
    with Image.open(path) as img:
        processed = preprocess_pil_image(img, blur_radius=blur_radius)
    array = np.array(processed, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def save_preprocessed_copy(src_path: Path, dest_path: Path, blur_radius: float = 1.0) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        processed = preprocess_pil_image(img, blur_radius=blur_radius)
        processed.save(dest_path)
