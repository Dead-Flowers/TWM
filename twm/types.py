from enum import Enum

import numpy as np
from expression import Option


class Gesture(Enum):
    SCROLL_UP = "scroll-up"
    SCROLL_DOWN = "scroll-down"
    CURSOR_MODE = "cursor-mode"
    LEFT_MOUSE_BTN = "left-mouse-btn"
    RIGHT_MOUSE_BTN = "right-mouse-btn"


class CursorDirection(Enum):
    CENTER = (0, 0)
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


ImageType = np.ndarray[int, np.dtype[np.uint8]]
FaceLandmarks = Option[list[tuple[int, int]]]
