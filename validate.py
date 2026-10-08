import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    IMAGE_EXTENSIONS,
    PROCESSED_DIR,
    PROCESSED_VALIDATION_DIR,
    VALIDATION_DIR,
)
from src.inference import load_classifier, predict_label


def _blur_radius_for_dir(data_dir: Path) -> float:
    try:
        data_dir.resolve().relative_to(PROCESSED_DIR.resolve())
        return 0.0
    except ValueError:
        return 1.0


def _pick_val_dir(use_raw: bool) -> Path:
    if use_raw:
        return VALIDATION_DIR
    if PROCESSED_VALIDATION_DIR.is_dir() and any(PROCESSED_VALIDATION_DIR.iterdir()):
        return PROCESSED_VALIDATION_DIR
    return VALIDATION_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the model on validation images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--raw", action="store_true", help="Use validation/ instead of processed/.")
    args = parser.parse_args()

    val_dir = _pick_val_dir(args.raw)
    blur_radius = _blur_radius_for_dir(val_dir)
    model = load_classifier(args.model)

    total = 0
    correct = 0
    per_class = {name: {"total": 0, "correct": 0} for name in CLASS_NAMES}

    for expected in CLASS_NAMES:
        category_path = val_dir / expected
        if not category_path.is_dir():
            print(f"Missing folder: {category_path}")
            continue

        print(f"\n{expected}/")
        for img_path in sorted(category_path.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            predicted, p_sit, p_jump = predict_label(
                model, img_path, blur_radius=blur_radius
            )
            ok = predicted == expected
            total += 1
            correct += int(ok)
            per_class[expected]["total"] += 1
            per_class[expected]["correct"] += int(ok)
            mark = "OK" if ok else "MISS"
            print(
                f"  [{mark}] {img_path.name}: pred={predicted} "
                f"(sitting={p_sit:.2f}, jumping={p_jump:.2f})"
            )

    if total:
        print(f"\nAccuracy: {correct}/{total} ({100.0 * correct / total:.1f}%)")
        for name in CLASS_NAMES:
            stats = per_class[name]
            if stats["total"]:
                acc = 100.0 * stats["correct"] / stats["total"]
                print(f"  {name}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    else:
        print("No validation images found.")


if __name__ == "__main__":
    main()
