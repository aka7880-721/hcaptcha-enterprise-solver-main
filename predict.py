"""Run inference on one image or select hCaptcha-style tiles matching a target pose."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, DEFAULT_MODEL_PATH
from src.inference import load_classifier, predict_label, select_matching_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict pose or select matching captcha tiles.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a single image.")
    group.add_argument("--tiles-dir", type=Path, help="Folder of challenge tile images.")
    parser.add_argument(
        "--target",
        choices=CLASS_NAMES,
        help="Required with --tiles-dir: pose to select (e.g. sitting).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum confidence to include a tile when using --tiles-dir.",
    )
    args = parser.parse_args()

    model = load_classifier(args.model)

    if args.image:
        label, p_sit, p_jump = predict_label(model, args.image)
        print(f"File: {args.image}")
        print(f"Prediction: {label}")
        print(f"Confidence sitting: {p_sit:.4f}, jumping: {p_jump:.4f}")
        return

    if args.target is None:
        parser.error("--target is required when using --tiles-dir")

    selected = select_matching_tiles(
        model, args.tiles_dir, args.target, threshold=args.threshold
    )
    print(f"Target pose: {args.target} (threshold={args.threshold})")
    print(f"Selected {len(selected)} tile(s):")
    for index, path, score in selected:
        print(f"  index={index}  score={score:.4f}  file={path.name}")


if __name__ == "__main__":
    main()
