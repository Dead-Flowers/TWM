import cv2
from expression.collections import seq

from .camera import frame_generator, video_capture
from .gaze import FaceDetector
from .gui import display_landmarks
from .utils import bgr_to_grayscale


def main():
    face_detector = FaceDetector()

    with video_capture(0) as capture:
        processed_images = (
            seq.of_iterable(frame_generator(capture))
            .map(lambda image: (image, bgr_to_grayscale(image)))
            .starmap(
                lambda image_bgr, image_grayscale: (
                    image_bgr,
                    face_detector(image_grayscale),
                )
            )
        )

        for image, landmarks in processed_images:
            display_landmarks(image, landmarks)

            if cv2.waitKey(delay=1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
