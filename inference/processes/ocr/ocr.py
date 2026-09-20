"""OCR and bounding-box helpers for LayoutLMv3."""

from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any, Sequence

import pytesseract
from PIL import Image
from pytesseract import Output
from pytesseract.pytesseract import TesseractError


def normalize_box(box: Sequence[int] | Sequence[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, int(1000 * float(x0) / max(width, 1)))),
        max(0, min(1000, int(1000 * float(y0) / max(height, 1)))),
        max(0, min(1000, int(1000 * float(x1) / max(width, 1)))),
        max(0, min(1000, int(1000 * float(y1) / max(height, 1)))),
    ]


def denormalize_box(
    box: Sequence[int] | Sequence[float], width: int, height: int
) -> list[int]:
    x0, y0, x1, y1 = box
    return [
        max(0, min(width, int(float(x0) / 1000.0 * width))),
        max(0, min(height, int(float(y0) / 1000.0 * height))),
        max(0, min(width, int(float(x1) / 1000.0 * width))),
        max(0, min(height, int(float(y1) / 1000.0 * height))),
    ]


def normalize_boxes(
    boxes: Sequence[Sequence[int] | Sequence[float]], width: int, height: int
) -> list[list[int]]:
    return [normalize_box(b, width, height) for b in boxes]


def denormalize_boxes(
    boxes: Sequence[Sequence[int] | Sequence[float]], width: int, height: int
) -> list[list[int]]:
    return [denormalize_box(b, width, height) for b in boxes]


def boxes_look_normalized(boxes: Sequence[Sequence[int] | Sequence[float]]) -> bool:
    if not boxes:
        return True
    return max(float(v) for box in boxes for v in box) <= 1000.0


def _prepare_tesseract_image(
    image: Image.Image,
    dpi: int = 200,
    max_side: int = 1280,
) -> tuple[Image.Image, float, float]:
    rgb = image.convert("RGB")
    orig_w, orig_h = rgb.size
    w, h = orig_w, orig_h

    if min(w, h) < 64:
        scale = max(2, int(64 / max(min(w, h), 1)))
        w, h = w * scale, h * scale
        rgb = rgb.resize((w, h), Image.BICUBIC)

    long_side = max(w, h)  # speedup
    if long_side > max_side:
        ratio = max_side / float(long_side)
        w, h = max(1, int(w * ratio)), max(1, int(h * ratio))
        rgb = rgb.resize((w, h), Image.BILINEAR)

    if hasattr(rgb, "info"):
        rgb.info = dict(rgb.info or {})
        rgb.info["dpi"] = (int(dpi), int(dpi))

    sx = orig_w / max(w, 1)
    sy = orig_h / max(h, 1)
    return rgb, sx, sy


def run_ocr(
    image: Image.Image,
    lang: str = "eng",
    min_conf: float = 0.0,
    normalize: bool = False,
    dpi: int = 200,
    max_side: int = 1280,
    fast: bool = True,
) -> tuple[list[str], list[list[int]]]:
    rgb, sx, sy = _prepare_tesseract_image(image, dpi=dpi, max_side=max_side)
    width, height = rgb.size
    orig_w, orig_h = image.convert("RGB").size

    if fast:
        configs = (
            f"--oem 1 --psm 6 -c user_defined_dpi={dpi}",
            f"--oem 1 --psm 4 -c user_defined_dpi={dpi}",
        )
    else:
        configs = (
            f"--psm 6 -c user_defined_dpi={dpi}",
            f"--psm 4 -c user_defined_dpi={dpi}",
            f"--psm 3 -c user_defined_dpi={dpi}",
            f"--psm 11 -c user_defined_dpi={dpi}",
        )

    data = None
    last_err: Exception | None = None
    for cfg in configs:
        try:
            data = pytesseract.image_to_data(
                rgb, lang=lang, output_type=Output.DICT, config=cfg
            )
            break
        except TesseractError as exc:
            last_err = exc
            continue
        except Exception as exc:
            last_err = exc
            continue

    if data is None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            rgb.save(tmp_path, format="PNG", dpi=(dpi, dpi))
            data = pytesseract.image_to_data(
                str(tmp_path),
                lang=lang,
                output_type=Output.DICT,
                config=f"--oem 1 --psm 6 -c user_defined_dpi={dpi}",
            )
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            warnings.warn(
                f"Tesseract failed for image size {rgb.size}: {last_err or exc}",
                stacklevel=2,
            )
            return [], []

    words: list[str] = []
    boxes: list[list[int]] = []
    n = len(data["text"])

    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < min_conf:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if w <= 0 or h <= 0:
            continue
        box = [
            int(x * sx),
            int(y * sy),
            int((x + w) * sx),
            int((y + h) * sy),
        ]
        if normalize:
            box = normalize_box(box, orig_w, orig_h)
        words.append(text)
        boxes.append(box)
    return words, boxes


class OCRCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def make_key(sample_id: str, image: Image.Image) -> str:
        buf = image.convert("RGB").tobytes()
        digest = hashlib.sha1(buf).hexdigest()[:16]
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample_id)
        return f"{safe_id}_{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set(self, key: str, words: list[str], boxes: list[list[int]]) -> None:
        path = self._path(key)
        path.write_text(json.dumps({"words": words, "boxes": boxes}))
