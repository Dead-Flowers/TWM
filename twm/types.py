from dataclasses import dataclass, field
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
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


@dataclass
class BoundingBox:
    __slots__ = ("x1", "x2", "y1", "y2")

    x1: int
    x2: int
    y1: int
    y2: int


@dataclass
class GazeAreas:
    center_box: tuple[float, float, float, float]
    diagonal_down: tuple[float, float]
    diagonal_up: tuple[float, float]

    def check_box(self, x: float, y: float) -> bool:
        return (
            self.center_box[0] <= x <= self.center_box[1]
            and self.center_box[2] <= y <= self.center_box[3]
        )

    def check_diagonal_down(self, x: float, y: float) -> bool:
        return y >= self.diagonal_down[0] * x + self.diagonal_down[1]

    def check_diagonal_up(self, x: float, y: float) -> bool:
        return y >= self.diagonal_up[0] * x + self.diagonal_up[1]


ImageType = np.ndarray[int, np.dtype[np.uint8]]
Detections = list[tuple[int, BoundingBox]]


@dataclass
class DebugInfo:
    gaze_ratios: tuple[float, float] | None = None
    detections: Detections = field(default_factory=list)
    image_with_pupils: ImageType | None = None
