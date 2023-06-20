import os
import time
import cv2

IMG_PATH = "./tensorflow/workspace/images/"

os.makedirs(IMG_PATH, exist_ok=True)

labels = ['ok', 'rock', 'point_up', 'point_down', 'open']
num_images = 20

for label in labels:
    os.makedirs(os.path.join(IMG_PATH, label), exist_ok=True)
    capture = cv2.VideoCapture(0)
    print(f'Processing gesture: {label}')
    time.sleep(10)
    for i in range(num_images):
        start_time = time.time()
        while True:
            _, frame = capture.read()
            cv2.imshow("", frame)
            if time.time() - start_time > 2:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
        print(f"Capturing {i+1}/{num_images}")
        path = os.path.join(IMG_PATH, label, f"{label}_{i+60}.jpg")
        cv2.imwrite(path, frame)
    capture.release()