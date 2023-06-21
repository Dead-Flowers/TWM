import time

import cv2

from .control import InputController
from .gaze import GazeDirectionPredictor
from .gesture import FAKE_GESTURE_KEYS, GestureDetector
from .gui import display_debug_window
from .types import DebugInfo
from .video import FrameReader


def main() -> None:
    gaze_predictor = GazeDirectionPredictor()
    gesture_detector = GestureDetector()
    input_controller = InputController()
    last_key = None
    curr_time = last_time = time.time()

    with FrameReader(0) as reader:
        for image in reader:
            last_time = curr_time
            curr_time = time.time()
            debug_info = DebugInfo()
            debug_info.image_with_pupils = image

            direction = gaze_predictor(image, debug_info)
            gesture = gesture_detector(image, debug_info)
            gesture = gesture or FAKE_GESTURE_KEYS.get(last_key, None)
            input_controller(gesture, direction, curr_time - last_time)

            display_debug_window(
                debug_info.image_with_pupils, gesture, direction, debug_info
            )

            last_key = chr(cv2.waitKey(delay=1) & 0xFF)

            if last_key == "q":
                break


if __name__ == "__main__":
    main()
