from dataclasses import dataclass
from enum import Enum

import numpy as np


class Gesture(Enum):
    SCROLL_UP = "scroll-up"
    SCROLL_DOWN = "scroll-down"
    CURSOR_MODE = "cursor-mode"
    LEFT_MOUSE_BTN = "left-mouse-btn"
    RIGHT_MOUSE_BTN = "right-mouse-btn"


class GazeDirection(Enum):
    CENTER = (0, 0)
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


@dataclass
class BoundingBox:
    __slots__ = ("x1", "x2", "y1", "y2")

    x1: int
    x2: int
    y1: int
    y2: int


ImageType = np.ndarray[int, np.dtype[np.uint8]]
FaceLandmarks = list[tuple[int, int]]
Detections = list[tuple[int, BoundingBox]]
