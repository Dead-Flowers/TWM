import tkinter as tk
from dataclasses import dataclass
from threading import Thread

import cv2
from expression import Option, Some

from .types import Detections, FaceLandmarks, ImageType
from .utils import Mutex


@dataclass
class Settings:
    sensitivity: float = 0.8
    cursor_speed: float = 10
    scroll_speed: float = 2


global_settings = Mutex(Settings())


def display_landmarks(image: ImageType, landmarks: Option[FaceLandmarks]) -> None:
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


def display_detections(image: ImageType, detections: Detections) -> None:
    for detection in detections:
        cv2.rectangle(
            image,
            (detection[1].x1, detection[1].x2, detection[1].y1, detection[1].y2),
            color=(255, 255, 255),
        )


def display_debug_window(
    image: ImageType, landmarks: Option[FaceLandmarks], detections: Detections
) -> None:
    display_landmarks(image, landmarks)
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
