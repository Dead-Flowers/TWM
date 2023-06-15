from typing import Any, Generator

import cv2

from .types import ImageType


class FrameReader:
    def __init__(self, camera_id_or_path: int | str) -> None:
        self.capture = cv2.VideoCapture(camera_id_or_path)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.capture.release()

    def __iter__(self) -> Generator[ImageType, Any, None]:
        ret, frame = self.capture.read()

        while ret:
            yield frame
            ret, frame = self.capture.read()

    def __del__(self) -> None:
        self.capture.release()
