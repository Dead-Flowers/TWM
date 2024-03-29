import tensorflow as tf

from .constants import OBJECT_DETECTOR_MODEL
from .types import BoundingBox, DebugInfo, Detections, Gesture, ImageType

# TODO
GESTURE_CLASS_MAP = {
    2137: Gesture.SCROLL_UP,
    2138: Gesture.SCROLL_DOWN,
    2139: Gesture.CURSOR_MODE,
    2140: Gesture.LEFT_MOUSE_BTN,
    2141: Gesture.RIGHT_MOUSE_BTN,
}

FAKE_GESTURE_KEYS = {
    "w": Gesture.SCROLL_UP,
    "s": Gesture.SCROLL_DOWN,
    "a": Gesture.LEFT_MOUSE_BTN,
    "d": Gesture.RIGHT_MOUSE_BTN,
    "e": Gesture.CURSOR_MODE,
}


class GestureDetector:
    def __init__(self) -> None:
        self.model = tf.saved_model.load(OBJECT_DETECTOR_MODEL)

    def detect_objects(
        self, image: ImageType, confidence_threshold: float = 0.5
    ) -> Detections:
        detections = self.model(tf.convert_to_tensor(image)[tf.newaxis, ...])

        boxes = detections["detection_boxes"]
        classes = detections["detection_classes"]
        scores = detections["detection_scores"]

        width, height = image.shape[1], image.shape[0]

        results = []

        for i in range(scores.shape[1]):
            if scores[0, i] < confidence_threshold:
                continue

            ymin, xmin, ymax, xmax = boxes[0, i]
            x1, x2 = int(xmin * width), int(xmax * width)
            y1, y2 = int(ymin * height), int(ymax * height)

            bounding_box = BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2)
            detection_class = int(classes[0, i])
            results.append((detection_class, bounding_box))

        results.sort(key=lambda x: scores[0, x[0]], reverse=True)

        return results

    def __call__(
        self, image: ImageType, debug_info: DebugInfo | None = None
    ) -> Gesture | None:
        detections = self.detect_objects(image)

        if debug_info:
            debug_info.detections = detections

        return next(
            (
                GESTURE_CLASS_MAP[det[0]]
                for det in detections
                if det[0] in GESTURE_CLASS_MAP
            ),
            None,
        )
