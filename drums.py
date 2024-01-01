import cv2
import numpy as np
import pygame

pygame.init()

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Read the first frame to initialize the background
ret, prev_frame = cap.read()

# Load the drum sounds
drum_sound_fast = pygame.mixer.Sound("drum.wav")

high_drum_sound = pygame.mixer.Sound("high.wav")

# Set the default speed and volume
default_speed = 44100  # Adjust as needed
default_volume = 0.5  # Adjust as needed
pygame.mixer.init(frequency=default_speed, size=-16, channels=1, buffer=512)
y_previous = 0
while True:
    ret, current_frame = cap.read()

    if not ret:
        break

    diff_frame = cv2.absdiff(current_frame, prev_frame)

    gray_diff = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)

    _, thresholded_diff = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

    thresholded_diff = cv2.morphologyEx(
        thresholded_diff, cv2.MORPH_OPEN, kernel=np.ones((5, 5))
    )

    contours, _ = cv2.findContours(
        thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        speed = y - (y - h)
        # speed = h if h > 0 else 0
        # print(speed)
        speed_threshold = 300
        if speed > speed_threshold and y + h > current_frame.shape[0] - 15:
            roi = current_frame[y : y + h, x : x + w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray_roi)
            texture_threshold = 500


            print(y_previous, y)
            if y_previous > y:
                print("Object moving downward at a high speed with low texture")

                speed_mapping = min((speed - speed_threshold) / 50, 1.0)
                drum_sound_speed = int(default_speed * (1.0 + speed_mapping))

                print(
                    f"Object moving at a high speed! Drum sound speed: {drum_sound_speed}"
                )

                adjusted_volume = min(default_volume + speed_mapping, 1.0)

                if drum_sound_speed > 80000:
                    pygame.mixer.Channel(0).set_volume(adjusted_volume)
                    pygame.mixer.Channel(0).play(high_drum_sound)
                else:
                    pygame.mixer.Channel(0).set_volume(adjusted_volume)
                    pygame.mixer.Channel(0).play(drum_sound_fast)

                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        y_previous = y

    cv2.imshow("Original", current_frame)
    cv2.imshow("Moving only", thresholded_diff)

    prev_frame = current_frame.copy()

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
