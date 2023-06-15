from argparse import ArgumentParser

import cv2

from .gaze import GazeDirectionPredictor
from .gesture import GestureDetector
from .gui import display_debug_window, start_gui
from .utils import equalize_histogram
from .video import FrameReader


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    subparsers = parser.add_subparsers(dest="action")
    gui_parser = subparsers.add_parser("gui")

    return parser


def main_loop(debug: bool = False) -> None:
    gaze_predictor = GazeDirectionPredictor()
    gesture_detector = GestureDetector()

    with FrameReader(0) as reader:
        for image in reader:
            eq_image = equalize_histogram(image)
            direction = gaze_predictor(eq_image)
            ratios = gaze_predictor.gaze_ratios()
            detections = gesture_detector.detect_objects(image)

            if debug:
                display_debug_window(image, detections, direction, ratios)
                cv2.imshow("TWM - Debug", image)

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
