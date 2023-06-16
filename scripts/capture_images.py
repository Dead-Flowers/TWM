import os
import time
import cv2

IMG_PATH = "../tensorflow/workspace/images/"


os.makedirs(IMG_PATH, exist_ok=True)

labels = ['ok', 'rock', 'point_up', 'point_down', 'open']
num_images = 20

for label in labels:
    os.makedirs(os.path.join(IMG_PATH, label), exist_ok=True)
    capture = cv2.VideoCapture(0)
    print(f'Processing gesture: {label}')
    time.sleep(10)
    for i in range(num_images):
        _, frame = cap.read()
        print(f"Capturing {i+1}/{num_images}")
        path = os.path.join(IMG_PATH, label, f"{label}_{i}.jpg")
        cv2.imwrite(path, frame)
        cv2.imshow(f"{i+1}/{num_images}", frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()

