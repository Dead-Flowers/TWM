import dlib
from expression import Nothing, Some

from .constants import SHAPE_PREDICTOR_MODEL
from .types import FaceLandmarks, ImageType


class FaceDetector:
    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_MODEL)

    def __call__(self, image_grayscale: ImageType) -> FaceLandmarks:
        faces = self.detector(image_grayscale)

        if faces:
            landmarks = self.predictor(image=image_grayscale, box=faces[0])

            return Some(
                [(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)]
            )

        return Nothing
