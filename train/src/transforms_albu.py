"""Albumentations pipelines for document images + OCR boxes (pixel coords).

Supports albumentations 1.x and 2.4+ parameter renames via signature probing.
"""

from __future__ import annotations

import inspect
from typing import Any

import albumentations as A
import numpy as np


def _accepts(cls, *names: str) -> bool:
    params = inspect.signature(cls.__init__).parameters
    return any(n in params for n in names)


def _affine(
    *,
    rotation_degrees: float,
    translate_frac: float,
    scale_range: tuple[float, float],
    apply_prob: float,
):
    kwargs: dict[str, Any] = dict(
        rotate=(-rotation_degrees, rotation_degrees),
        translate_percent={
            "x": (-translate_frac, translate_frac),
            "y": (-translate_frac, translate_frac),
        },
        scale=(scale_range[0], scale_range[1]),
        fit_output=False,
        p=apply_prob,
    )
    if _accepts(A.Affine, "fill"):
        kwargs["fill"] = 255
    elif _accepts(A.Affine, "cval"):
        kwargs["cval"] = 255
    return A.Affine(**kwargs)


def _brightness_contrast(brightness: float, contrast: float, p: float = 0.7):
    if _accepts(A.RandomBrightnessContrast, "brightness_range"):
        return A.RandomBrightnessContrast(
            brightness_range=(-brightness, brightness),
            contrast_range=(-contrast, contrast),
            p=p,
        )
    return A.RandomBrightnessContrast(
        brightness_limit=brightness,
        contrast_limit=contrast,
        p=p,
    )


def _image_compression(p: float = 0.3):
    if _accepts(A.ImageCompression, "quality_range"):
        return A.ImageCompression(quality_range=(50, 95), p=p)
    return A.ImageCompression(quality_lower=50, quality_upper=95, p=p)


def _gauss_noise(var_limit: float = 15.0, p: float = 0.4):
    if _accepts(A.GaussNoise, "std_range"):
        return A.GaussNoise(std_range=(0.0, max(var_limit / 255.0, 1e-3)), p=p)
    if _accepts(A.GaussNoise, "var_limit"):
        return A.GaussNoise(var_limit=(0.0, var_limit), p=p)
    return A.GaussNoise(p=p)


def _gaussian_blur(p: float = 0.15):
    if _accepts(A.GaussianBlur, "blur_range"):
        return A.GaussianBlur(blur_range=(3, 3), sigma_range=(0.5, 1.5), p=p)
    if _accepts(A.GaussianBlur, "blur_limit"):
        return A.GaussianBlur(blur_limit=(3, 3), p=p)
    return A.GaussianBlur(p=p)


def _bbox_params(*, min_visibility: float, clip: bool = True) -> A.BboxParams:
    if _accepts(A.BboxParams, "coord_format"):
        return A.BboxParams(
            coord_format="pascal_voc",
            label_fields=["box_labels"],
            min_visibility=min_visibility,
            clip_bboxes_on_input=clip,
        )
    return A.BboxParams(
        format="pascal_voc",
        label_fields=["box_labels"],
        min_visibility=min_visibility,
        clip=clip,
    )


def build_train_transforms(
    *,
    rotation_degrees: float = 5.0,
    translate_frac: float = 0.02,
    scale_range: tuple[float, float] = (0.95, 1.05),
    brightness: float = 0.15,
    contrast: float = 0.2,
    gauss_noise_var: float = 15.0,
    apply_prob: float = 0.8,
) -> A.Compose:
    """Geometric + photometric aug that keeps Pascal-VOC boxes in sync."""
    return A.Compose(
        [
            _affine(
                rotation_degrees=rotation_degrees,
                translate_frac=translate_frac,
                scale_range=scale_range,
                apply_prob=apply_prob,
            ),
            _brightness_contrast(brightness, contrast, p=0.7),
            _gauss_noise(var_limit=gauss_noise_var, p=0.4),
            _gaussian_blur(p=0.15),
            _image_compression(p=0.3),
        ],
        bbox_params=_bbox_params(min_visibility=0.1, clip=True),
    )


def build_eval_transforms() -> A.Compose | None:
    # No-op pipeline; returning None avoids albumentations 2.x warning
    # ("Got processor for bboxes, but no transform to process it").
    return None


def apply_transforms(
    image: np.ndarray,
    boxes: list[list[float]],
    words: list[str],
    transform: A.Compose | None,
) -> tuple[np.ndarray, list[list[float]], list[str]]:
    if transform is None:
        return image, boxes, words
    if not boxes:
        out = transform(image=image, bboxes=[], box_labels=[])
        return out["image"], [], []

    box_labels = list(range(len(boxes)))
    out = transform(image=image, bboxes=boxes, box_labels=box_labels)
    keep_idx = [int(i) for i in out["box_labels"]]
    new_boxes = [list(map(float, b)) for b in out["bboxes"]]
    new_words = [words[i] for i in keep_idx]
    return out["image"], new_boxes, new_words


def transforms_from_config(cfg: dict[str, Any] | None, train: bool) -> A.Compose | None:
    cfg = cfg or {}
    if not train or not cfg.get("enabled", True):
        return build_eval_transforms()
    scale = cfg.get("scale_range", [0.95, 1.05])
    return build_train_transforms(
        rotation_degrees=float(cfg.get("rotation_degrees", 5.0)),
        translate_frac=float(cfg.get("translate_frac", 0.02)),
        scale_range=(float(scale[0]), float(scale[1])),
        brightness=float(cfg.get("brightness", 0.15)),
        contrast=float(cfg.get("contrast", 0.2)),
        gauss_noise_var=float(cfg.get("gauss_noise_var", 15.0)),
        apply_prob=float(cfg.get("apply_prob", 0.8)),
    )
