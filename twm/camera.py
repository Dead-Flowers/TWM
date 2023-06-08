from contextlib import contextmanager
from typing import Any, Generator

import cv2

from .types import ImageType


@contextmanager
def video_capture(camera_id: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_id)

    try:
        yield capture
    finally:
        capture.release()


def frame_generator(capture: cv2.VideoCapture) -> Generator[ImageType, Any, None]:
    ret, frame = capture.read()

    while ret:
        yield frame
        ret, frame = capture.read()
