import dlib
from expression import Nothing, Option, Some

from .constants import SHAPE_PREDICTOR_MODEL
from .types import FaceLandmarks, GazeDirection, ImageType
from .utils import min_bounding_box


class GazeDirectionPredictor:
    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_MODEL)

    def detect_landmarks(self, image_grayscale: ImageType) -> Option[FaceLandmarks]:
        faces = self.detector(image_grayscale)

        if faces:
            landmarks = self.predictor(image=image_grayscale, box=faces[0])

            return Some(
                [(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)]
            )

        return Nothing

    def extract_eyes(
        self, image: ImageType, landmarks: FaceLandmarks
    ) -> tuple[ImageType, ImageType]:
        left_eye_bb = min_bounding_box(landmarks[42:48])
        right_eye_bb = min_bounding_box(landmarks[36:42])

        return (
            image[left_eye_bb.y1 : left_eye_bb.y2, left_eye_bb.x1 : left_eye_bb.x2],
            image[right_eye_bb.y1 : right_eye_bb.y2, right_eye_bb.x1 : right_eye_bb.x2],
        )

    def predict_direction(
        self, left_eye: ImageType, right_eye: ImageType
    ) -> GazeDirection:
        # TODO
        raise NotImplementedError

    def __call__(self, image_grayscale: ImageType) -> Option[GazeDirection]:
        return (
            self.detect_landmarks(image_grayscale)
            .map(lambda landmarks: self.extract_eyes(image_grayscale, landmarks))
            .map(lambda eyes: self.predict_direction(*eyes))
        )
