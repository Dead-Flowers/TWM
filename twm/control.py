import math
from collections import Counter, deque

import pyautogui

from .types import GazeDirection, Gesture


class InputController:
    def __init__(
        self,
        cursor_speed: float = 10,
        scroll_speed: float = 2,
        gesture_smoothing_window: int = 10,
    ) -> None:
        self.cursor_speed = cursor_speed
        self.scroll_speed = scroll_speed
        self.x_fract = 0
        self.y_fract = 0
        self.scroll_fract = 0
        self.last_gesture = None
        self.gesture_window: deque[Gesture] = deque(maxlen=gesture_smoothing_window)

    def __call__(
        self,
        gesture: Gesture | None,
        direction: GazeDirection | None,
        delta_time: float,
    ) -> None:
        if gesture is None:
            return

        self.gesture_window.append(gesture)
        most_common_gesture = Counter(self.gesture_window).most_common(1)[0][0]

        match most_common_gesture:
            case Gesture.CURSOR_MODE if direction:
                x, y = direction.value
                x_amount, self.x_fract = self._step(
                    self.x_fract, x * self.cursor_speed, delta_time
                )
                y_amount, self.y_fract = self._step(
                    self.y_fract, y * self.cursor_speed, delta_time
                )
                pyautogui.moveRel(x_amount, y_amount)
            case Gesture.SCROLL_UP:
                amount, self.scroll_fract = self._step(
                    self.scroll_fract, self.scroll_speed, delta_time
                )
                pyautogui.scroll(amount)
            case Gesture.SCROLL_DOWN:
                amount, self.scroll_fract = self._step(
                    self.scroll_fract, -self.scroll_speed, delta_time
                )
                pyautogui.scroll(amount)
            case Gesture.LEFT_MOUSE_BTN if self.last_gesture != most_common_gesture:
                pyautogui.click(button=pyautogui.PRIMARY)
            case Gesture.RIGHT_MOUSE_BTN if self.last_gesture != most_common_gesture:
                pyautogui.click(button=pyautogui.SECONDARY)

        self.last_gesture = most_common_gesture

    @staticmethod
    def _step(curr_fract: float, speed: float, delta_time: float) -> tuple[int, float]:
        curr_fract += speed * delta_time
        amount, new_fract = divmod(curr_fract, math.copysign(1, curr_fract))
        return int(math.copysign(amount, curr_fract)), new_fract
