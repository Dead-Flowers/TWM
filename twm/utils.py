from typing import Sequence

import cv2

from .types import ImageType


def resize_image_320x320(image: ImageType) -> ImageType:
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio != 1.0:
        if aspect_ratio > 1.0:
            new_height = int(width / aspect_ratio)
            start_row = int((height - new_height) / 2)
            end_row = start_row + new_height
            image = image[start_row:end_row, :]
        else:
            new_width = int(height * aspect_ratio)
            start_col = int((width - new_width) / 2)
            end_col = start_col + new_width
            image = image[:, start_col:end_col]

    return cv2.resize(image, (320, 320))


def bgr_to_grayscale(image: ImageType) -> ImageType:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def min_bounding_box(
    points: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int]]:
    xs, ys = zip(*points)
    return (min(xs), min(ys)), (max(xs), max(ys))
