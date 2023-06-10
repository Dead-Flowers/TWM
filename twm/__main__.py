from argparse import ArgumentParser

import cv2

from .camera import frame_generator, video_capture
from .gaze import GazeDirectionPredictor
from .gesture import GestureDetector
from .gui import display_debug_window, start_gui
from .utils import bgr_to_grayscale


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    subparsers = parser.add_subparsers(dest="action")
    gui_parser = subparsers.add_parser("gui")

    return parser


def main_loop(debug: bool = False) -> None:
    gaze_predictor = GazeDirectionPredictor()
    gesture_detector = GestureDetector()

    with video_capture(0) as capture:
        for image in frame_generator(capture):
            landmarks = gaze_predictor.detect_landmarks(bgr_to_grayscale(image))
            detections = gesture_detector.detect_objects(image)
            if debug:
                display_debug_window(image, landmarks, detections)

            if cv2.waitKey(delay=1) & 0xFF == ord("q"):
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
