from contextlib import contextmanager
from threading import Lock
from typing import Any, Generator, Generic, Sequence, TypeVar

import cv2

from .types import BoundingBox, ImageType

T = TypeVar("T")


def bgr_to_grayscale(image: ImageType) -> ImageType:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def min_bounding_box(points: Sequence[tuple[int, int]]) -> BoundingBox:
    xs, ys = zip(*points)
    return BoundingBox(x1=min(xs), x2=max(xs), y1=min(ys), y2=max(ys))


class Mutex(Generic[T]):
    def __init__(self, value: T) -> None:
        self.__value = value
        self.__lock = Lock()

    @contextmanager
    def lock(self) -> Generator[T, Any, None]:
        self.__lock.acquire()
        try:
            yield self.__value
        finally:
            self.__lock.release()
