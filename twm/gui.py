import cv2
from expression import Some

from .types import FaceLandmarks, ImageType


def display_landmarks(image: ImageType, landmarks: FaceLandmarks):
    match landmarks:
        case Some(landmarks_list):
            for landmark in landmarks_list:
                cv2.circle(
                    img=image,
                    center=landmark,
                    radius=3,
                    color=(0, 255, 0),
                    thickness=-1,
                )

    cv2.imshow("", image)
