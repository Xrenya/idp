from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np


def build_train_transforms(
    *,
    rotation_degrees: float = 5.0,
    translate_frac: float = 0.02,
    scale_range: tuple[float, float] = (0.95, 1.05),
    brightness: float = 0.15,
    contrast: float = 0.2,
    gauss_noise_std_range: tuple[float, float] = (0.0, 0.06),
    apply_prob: float = 0.8,
) -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                scale=scale_range,
                translate_percent={
                    "x": (-translate_frac, translate_frac),
                    "y": (-translate_frac, translate_frac),
                },
                rotate=(-rotation_degrees, rotation_degrees),
                fit_output=False,
                fill=255,
                p=apply_prob,
            ),
            A.RandomBrightnessContrast(
                brightness_range=(-brightness, brightness),
                contrast_range=(-contrast, contrast),
                brightness_by_max=False,
                p=0.7,
            ),
            A.GaussNoise(
                std_range=gauss_noise_std_range,
                mean_range=(0.0, 0.0),
                per_channel=False,
                p=0.4,
            ),
            A.GaussianBlur(
                blur_range=(3, 3),
                sigma_range=(0.5, 1.5),
                p=0.15,
            ),
            A.ImageCompression(
                compression_type="jpeg",
                quality_range=(50, 95),
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="pascal_voc",
            label_fields=["box_labels"],
            min_visibility=0.1,
            clip_bboxes_on_input=True,
        ),
    )


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
        return None
    scale = cfg.get("scale_range", [0.95, 1.05])
    noise_std = cfg.get("gauss_noise_std_range", [0.0, 0.06])
    return build_train_transforms(
        rotation_degrees=float(cfg.get("rotation_degrees", 5.0)),
        translate_frac=float(cfg.get("translate_frac", 0.02)),
        scale_range=(float(scale[0]), float(scale[1])),
        brightness=float(cfg.get("brightness", 0.15)),
        contrast=float(cfg.get("contrast", 0.2)),
        gauss_noise_std_range=(float(noise_std[0]), float(noise_std[1])),
        apply_prob=float(cfg.get("apply_prob", 0.8)),
    )
