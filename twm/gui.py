import cv2

from .types import DebugInfo, Detections, GazeDirection, Gesture, ImageType


def display_gesture(image: ImageType, gesture: Gesture):
    cv2.putText(
        image,
        gesture.name,
        (image.shape[1] - 300, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (255, 0, 0),
        1,
    )


def display_gaze_direction(image: ImageType, direction: GazeDirection):
    cv2.putText(
        image, direction.name, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1
    )


def display_gaze_ratios(
    image: ImageType, horizontal_ratio: float, vertical_ratio: float
):
    cv2.putText(
        image,
        f"{horizontal_ratio:.2f} {vertical_ratio:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (255, 0, 0),
        1,
    )

    cv2.circle(
        image,
        (
            int(image.shape[1] * (horizontal_ratio + 0.5)),
            int(image.shape[0] * (vertical_ratio + 0.5)),
        ),
        5,
        (0, 255, 0),
        -1,
    )


def display_detections(image: ImageType, detections: Detections) -> None:
    for detection in detections:
        cv2.rectangle(
            image,
            (detection[1].x1, detection[1].x2, detection[1].y1, detection[1].y2),
            color=(255, 255, 255),
        )


def display_debug_window(
    image: ImageType,
    gesture: Gesture | None,
    direction: GazeDirection | None,
    debug_info: DebugInfo,
) -> None:
    if gesture:
        display_gesture(image, gesture)

    if direction:
        display_gaze_direction(image, direction)

    if debug_info.gaze_ratios is not None:
        display_gaze_ratios(image, *debug_info.gaze_ratios)

    display_detections(image, debug_info.detections)
    cv2.imshow("TWM - Debug", image)
