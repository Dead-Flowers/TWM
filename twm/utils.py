import cv2

from .types import ImageType


def equalize_histogram(image: ImageType) -> ImageType:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
