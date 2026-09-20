#!/usr/bin/env python3
"""
python run.py --image samples/resume_180.jpg
python run.py --image samples/resume_180.jpg -o samples/prediction_resume_180.json -v
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podder_task_foundation import MODE, Payload, ProcessExecutor, bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description="ADCS document classification inference")
    parser.add_argument("--image", required=True, help="Path to document image")
    parser.add_argument(
        "--config",
        default="config",
        help="Podder config directory (default: config)",
    )
    parser.add_argument("-o", "--output", default=None, help="Write prediction JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    executor = ProcessExecutor(
        config_path=args.config,
        mode=MODE.CONSOLE,
        verbose=args.verbose,
        debug_mode=args.debug,
    )
    bootstrap(executor.context)

    request = Payload()
    request.add_file(image_path, name="image")

    result = executor.execute(None, input_payload=request)

    prediction = result.get_data("prediction") or {}
    orientation = result.get_data("orientation") or {}
    ocr = result.get_data("ocr") or {}
    payload_out = {
        "image_path": str(image_path.resolve()),
        "orientation": orientation,
        "ocr": {
            "n_words": ocr.get("n_words"),
            "width": ocr.get("width"),
            "height": ocr.get("height"),
            "box_space": ocr.get("box_space"),
        },
        "prediction": prediction,
    }

    text = json.dumps(payload_out, indent=2, ensure_ascii=False)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    main()
