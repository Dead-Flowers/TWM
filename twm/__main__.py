import time
from argparse import ArgumentParser

import cv2

from .control import InputController
from .gaze import GazeDirectionPredictor
from .gesture import FAKE_GESTURE_KEYS, GestureDetector
from .gui import display_debug_window, start_gui
from .types import DebugInfo
from .utils import equalize_histogram
from .video import FrameReader


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-n", "--no-debug", dest="debug", action="store_false")
    subparsers = parser.add_subparsers(dest="action")
    gui_parser = subparsers.add_parser("gui")

    return parser


def main_loop(debug: bool = False) -> None:
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

            eq_image = equalize_histogram(image)
            direction = gaze_predictor(eq_image, debug_info)
            gesture = gesture_detector(image, debug_info)
            gesture = gesture or FAKE_GESTURE_KEYS.get(last_key, None)
            input_controller(gesture, direction, curr_time - last_time)

            if debug:
                display_debug_window(image, gesture, direction, debug_info)
                cv2.imshow("TWM - Debug", image)

            last_key = chr(cv2.waitKey(delay=1) & 0xFF)

            if last_key == "q":
                break


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    match args.action:
        case "gui":
            start_gui()
            main_loop(args.debug)
        case _:
            main_loop(args.debug)


if __name__ == "__main__":
    main()
