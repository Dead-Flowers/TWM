from gaze_tracking import GazeTracking

from .types import GazeDirection, ImageType


class GazeDirectionPredictor:
    def __init__(self) -> None:
        self._gaze_tracker = GazeTracking()

    def __call__(self, image: ImageType) -> GazeDirection:
        self._gaze_tracker.refresh(image)
        vertical_ratio = self._gaze_tracker.vertical_ratio()

        if self._gaze_tracker.is_right():
            return GazeDirection.RIGHT
        if self._gaze_tracker.is_left():
            return GazeDirection.LEFT
        if vertical_ratio is not None and vertical_ratio >= 0.65:
            return GazeDirection.UP
        if vertical_ratio is not None and vertical_ratio <= 0.35:
            return GazeDirection.DOWN

        return GazeDirection.CENTER

    def gaze_ratios(self) -> tuple[float, float] | None:
        if self._gaze_tracker.pupils_located:
            return (
                self._gaze_tracker.horizontal_ratio(),
                self._gaze_tracker.vertical_ratio(),
            )
