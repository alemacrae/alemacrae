"""Utility to segment plants from images using color-based segmentation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def segment_plants(image_path: str | Path, save_mask: bool = False) -> np.ndarray:
    """Segment green vegetation from an image.

    Parameters
    ----------
    image_path:
        Path to the input image.
    save_mask:
        If ``True``, the binary mask will be saved alongside the image
        with suffix ``"_mask.png"``.

    Returns
    -------
    np.ndarray
        Binary mask where vegetation pixels are ``255`` and the rest ``0``.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV ranges for green vegetation; may need tuning for specific cameras.
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up noise: remove small objects and fill holes.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    if save_mask:
        out_path = Path(image_path).with_name(Path(image_path).stem + "_mask.png")
        cv2.imwrite(str(out_path), mask)

    return mask


if __name__ == "__main__":  # pragma: no cover - manual invocation example
    import argparse

    parser = argparse.ArgumentParser(description="Segment plants from an image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--save", action="store_true", help="Save mask to disk")
    args = parser.parse_args()

    mask = segment_plants(args.image, save_mask=args.save)
    print(f"Mask computed with shape {mask.shape}")
