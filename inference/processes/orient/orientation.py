from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import pytesseract
from PIL import Image
from pytesseract import Output

OrientationBackend = Literal["osd", "ocr_score", "auto", "none"]


@dataclass
class OrientationResult:
    rotate_degrees: int  # clockwise rotation applied to correct the page
    detected_orientation: int
    confidence: float
    backend: str
    script: str = ""
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def apply_rotation(image: Image.Image, degrees_clockwise: int) -> Image.Image:
    """Rotate an image clockwise"""
    degrees_clockwise = int(degrees_clockwise) % 360
    if degrees_clockwise == 0:
        return image.convert("RGB")
    # conter-clockwise with white canvas
    return image.convert("RGB").rotate(-degrees_clockwise, expand=True, fillcolor=(255, 255, 255))


def detect_orientation_osd(
    image: Image.Image,
    min_characters_to_try: int = 20,
) -> OrientationResult:
    rgb = image.convert("RGB")
    config = f"--psm 0 -c min_characters_to_try={int(min_characters_to_try)}"
    try:
        osd = pytesseract.image_to_osd(rgb, output_type=Output.DICT, config=config)
        rotate = int(osd.get("rotate", 0)) % 360
        return OrientationResult(
            rotate_degrees=rotate,
            detected_orientation=int(osd.get("orientation", 0)),
            confidence=float(osd.get("orientation_conf", 0.0)),
            backend="osd",
            script=str(osd.get("script", "")),
            raw=dict(osd),
        )
    except Exception as exc:
        return OrientationResult(
            rotate_degrees=0,
            detected_orientation=0,
            confidence=0.0,
            backend="osd_failed",
            script="",
            raw={"error": str(exc)},
        )


def _ocr_readable_score(image: Image.Image) -> float:
    text = pytesseract.image_to_string(image.convert("RGB"), config="--psm 6") or ""
    return float(sum(ch.isalnum() for ch in text))


def detect_orientation_ocr_score(image: Image.Image) -> OrientationResult:
    rgb = image.convert("RGB")
    best_rot, best_score = 0, -1.0
    scores: dict[str, float] = {}
    for rot in (0, 90, 180, 270):
        candidate = apply_rotation(rgb, rot)
        score = _ocr_readable_score(candidate)
        scores[str(rot)] = score
        if score > best_score:
            best_rot, best_score = rot, score
    return OrientationResult(
        rotate_degrees=best_rot,
        detected_orientation=(360 - best_rot) % 360,
        confidence=float(best_score),
        backend="ocr_score",
        raw={"scores": scores},
    )


def correct_orientation(
    image: Image.Image,
    backend: OrientationBackend = "auto",
    min_osd_confidence: float = 1.5,
    min_characters_to_try: int = 20,
    validate_with_ocr_score: bool = True,
) -> tuple[Image.Image, OrientationResult]:
    if backend == "none":
        return image.convert("RGB"), OrientationResult(
            rotate_degrees=0,
            detected_orientation=0,
            confidence=0.0,
            backend="none",
        )

    if backend == "ocr_score":
        result = detect_orientation_ocr_score(image)
    else:
        osd = detect_orientation_osd(image, min_characters_to_try=min_characters_to_try)
        if osd.backend == "osd_failed" or osd.confidence < min_osd_confidence:
            if backend == "auto":
                result = detect_orientation_ocr_score(image)
                result.raw = {"osd": osd.to_dict(), "ocr_score": result.raw}
                result.backend = "auto:ocr_score"
            else:
                result = OrientationResult(
                    rotate_degrees=0,
                    detected_orientation=osd.detected_orientation,
                    confidence=osd.confidence,
                    backend="osd_skipped_low_conf",
                    script=osd.script,
                    raw=osd.raw,
                )
        else:
            result = osd
            result.backend = "auto:osd" if backend == "auto" else "osd"

    if validate_with_ocr_score and result.rotate_degrees % 360 != 0:
        base_score = _ocr_readable_score(image)
        rotated = apply_rotation(image, result.rotate_degrees)
        rot_score = _ocr_readable_score(rotated)
        raw = dict(result.raw or {})
        raw["ocr_validate"] = {
            "base_score": base_score,
            "rotated_score": rot_score,
            "proposed_rotate": result.rotate_degrees,
        }
        if rot_score + 5.0 < base_score:
            result = OrientationResult(
                rotate_degrees=0,
                detected_orientation=result.detected_orientation,
                confidence=result.confidence,
                backend=f"{result.backend}+rejected_by_ocr_score",
                script=result.script,
                raw=raw,
            )
            return image.convert("RGB"), result
        result.raw = raw
        return rotated, result

    corrected = apply_rotation(image, result.rotate_degrees)
    return corrected, result
