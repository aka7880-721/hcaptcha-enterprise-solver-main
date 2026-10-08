"""Copy training/validation images into processed/ with RGB resize + Gaussian blur."""

import argparse
from pathlib import Path

from src.config import (
    IMAGE_EXTENSIONS,
    PROCESSED_TRAINING_DIR,
    PROCESSED_VALIDATION_DIR,
    TRAINING_DIR,
    VALIDATION_DIR,
)
from src.preprocess import save_preprocessed_copy


def _iter_images(folder: Path):
    if not folder.is_dir():
        return
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def process_tree(source_root: Path, dest_root: Path, blur_radius: float) -> int:
    count = 0
    for src in _iter_images(source_root):
        rel = src.relative_to(source_root)
        dest = dest_root / rel
        save_preprocessed_copy(src, dest, blur_radius=blur_radius)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preprocessed dataset under processed/.")
    parser.add_argument(
        "--blur",
        type=float,
        default=1.0,
        help="Gaussian blur radius (0 to disable blur).",
    )
    args = parser.parse_args()

    n_train = process_tree(TRAINING_DIR, PROCESSED_TRAINING_DIR, args.blur)
    n_val = process_tree(VALIDATION_DIR, PROCESSED_VALIDATION_DIR, args.blur)
    print(f"Wrote {n_train} training and {n_val} validation images under processed/")


if __name__ == "__main__":
    main()
