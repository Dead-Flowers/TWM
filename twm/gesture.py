import tensorflow as tf

from .constants import OBJECT_DETECTOR_MODEL
from .types import BoundingBox, Detections, Gesture, ImageType

# TODO
GESTURE_CLASS_MAP = {
    0: Gesture.SCROLL_UP,
    1: Gesture.SCROLL_DOWN,
    2: Gesture.CURSOR_MODE,
    3: Gesture.LEFT_MOUSE_BTN,
    4: Gesture.RIGHT_MOUSE_BTN,
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

    def __call__(self, image: ImageType) -> Gesture | None:
        return next(
            (
                GESTURE_CLASS_MAP[det[0]]
                for det in self.detect_objects(image)
                if det[0] in GESTURE_CLASS_MAP
            ),
            None,
        )
