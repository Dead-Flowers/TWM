import tkinter as tk
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Any, Generator, Generic, TypeVar

import cv2

from .types import Detections, GazeDirection, ImageType

T = TypeVar("T")


class Mutex(Generic[T]):
    def __init__(self, value: T) -> None:
        self.__value = value
        self.__lock = Lock()

    @contextmanager
    def lock(self) -> Generator[T, Any, None]:
        self.__lock.acquire()
        try:
            yield self.__value
        finally:
            self.__lock.release()


@dataclass
class Settings:
    sensitivity: float = 0.8
    cursor_speed: float = 10
    scroll_speed: float = 2


global_settings = Mutex(Settings())


def display_gaze_direction(image: ImageType, direction: GazeDirection):
    cv2.putText(
        image, direction.name, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2
    )


def display_gaze_ratios(
    image: ImageType, horizontal_ratio: float, vertical_ratio: float
):
    cv2.putText(
        image,
        f"{horizontal_ratio:.2f} {vertical_ratio:.2f}",
        (120, 120),
        cv2.FONT_HERSHEY_DUPLEX,
        2,
        (255, 0, 0),
        2,
    )


def display_detections(image: ImageType, detections: Detections) -> None:
    for detection in detections:
        cv2.rectangle(
            image,
            (detection[1].x1, detection[1].x2, detection[1].y1, detection[1].y2),
            color=(255, 255, 255),
        )


def display_debug_window(
    image: ImageType,
    detections: Detections,
    direction: GazeDirection,
    ratios: tuple[float, float] | None,
) -> None:
    display_gaze_direction(image, direction)

    if ratios is not None:
        display_gaze_ratios(image, *ratios)

    display_detections(image, detections)
    cv2.imshow("TWM - Debug", image)


def gui_main():
    root = tk.Tk()
    root.title("TWM")

    def update_settings():
        with global_settings.lock() as settings:
            settings.sensitivity = sensitivity_slider.get() / 100
            settings.cursor_speed = float(cursor_speed_entry.get())
            settings.scroll_speed = float(scroll_speed_entry.get())

    tk.Label(root, text="Detection sensitivity:").grid(row=0, column=0, sticky="w")
    sensitivity_slider = tk.Scale(
        root, from_=0, to=100, orient="horizontal", command=lambda _: update_settings()
    )
    with global_settings.lock() as settings:
        sensitivity_slider.set(settings.sensitivity * 100)
    sensitivity_slider.grid(row=0, column=1)

    tk.Label(root, text="Cursor speed:").grid(row=1, column=0, sticky="w")
    cursor_speed_entry = tk.Entry(root)
    with global_settings.lock() as settings:
        cursor_speed_entry.insert(0, str(settings.cursor_speed))
    cursor_speed_entry.grid(row=1, column=1)

    tk.Label(root, text="Scrolling speed:").grid(row=2, column=0, sticky="w")
    scroll_speed_entry = tk.Entry(root)
    with global_settings.lock() as settings:
        scroll_speed_entry.insert(0, str(settings.scroll_speed))
    scroll_speed_entry.grid(row=2, column=1)

    update_button = tk.Button(root, text="Update Settings", command=update_settings)
    update_button.grid(row=3, column=0, columnspan=2)

    root.mainloop()


def start_gui() -> Thread:
    thread = Thread(target=gui_main)
    thread.start()
    return thread
