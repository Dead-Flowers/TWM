from gaze_tracking import GazeTracking

from .types import DebugInfo, GazeAreas, GazeDirection, ImageType

DEFAULT_GAZE_AREAS = GazeAreas((-0.1, 0.1, -0.1, 0.1), (-1, 0), (1, 0))


class GazeDirectionPredictor:
    def __init__(
        self,
        gaze_areas: GazeAreas = DEFAULT_GAZE_AREAS,
        bias: tuple[float, float] = (-0.1, -0.3),
    ) -> None:
        self._areas = gaze_areas
        self._bias = bias
        self._gaze_tracker = GazeTracking()

    def __call__(
        self, image: ImageType, debug_info: DebugInfo | None = None
    ) -> GazeDirection | None:
        self._gaze_tracker.refresh(image)

        if self._gaze_tracker.pupils_located:
            x = self._gaze_tracker.horizontal_ratio() - 0.5 + self._bias[0]
            y = self._gaze_tracker.vertical_ratio() - 0.5 + self._bias[1]

            if debug_info:
                debug_info.gaze_ratios = x, y
                debug_info.image_with_pupils = self._gaze_tracker.annotated_frame()

            if self._areas.check_box(x, y):
                return GazeDirection.CENTER
            if self._areas.check_diagonal_down(
                x, y
            ) and not self._areas.check_diagonal_up(x, y):
                return GazeDirection.LEFT
            if not self._areas.check_diagonal_down(
                x, y
            ) and self._areas.check_diagonal_up(x, y):
                return GazeDirection.RIGHT
            if self._areas.check_diagonal_down(x, y) and self._areas.check_diagonal_up(
                x, y
            ):
                return GazeDirection.DOWN
            if not self._areas.check_diagonal_down(
                x, y
            ) and not self._areas.check_diagonal_up(x, y):
                return GazeDirection.UP
