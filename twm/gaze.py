from gaze_tracking import GazeTracking

from .types import DebugInfo, GazeDirection, ImageType


class GazeDirectionPredictor:
    def __init__(self) -> None:
        self._gaze_tracker = GazeTracking()

    def __call__(
        self, image: ImageType, debug_info: DebugInfo | None = None
    ) -> GazeDirection | None:
        self._gaze_tracker.refresh(image)

        if self._gaze_tracker.pupils_located:
            horizontal_ratio = self._gaze_tracker.horizontal_ratio()
            vertical_ratio = self._gaze_tracker.vertical_ratio()

            if debug_info:
                debug_info.gaze_ratios = horizontal_ratio, vertical_ratio

            if horizontal_ratio >= 0.65:
                return GazeDirection.LEFT
            if horizontal_ratio <= 0.35:
                return GazeDirection.RIGHT
            if vertical_ratio >= 0.65:
                return GazeDirection.UP
            if vertical_ratio <= 0.35:
                return GazeDirection.DOWN
