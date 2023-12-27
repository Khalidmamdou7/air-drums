import cv2
import numpy as np
import pygame

pygame.init()
# Open a video capture object
cap = cv2.VideoCapture(0)

# Read the first frame to initialize the background

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

ret, prev_frame = cap.read()

drum_sound = pygame.mixer.Sound("drum.wav")

while True:
    # Read a frame from the video
    ret, current_frame = cap.read()

    if not ret:
        break  # Break the loop if the video is finished

    # Calculate the absolute difference between the current and previous frames
    diff_frame = cv2.absdiff(current_frame, prev_frame)

    # Convert the difference to grayscale
    gray_diff = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to clarify the moving parts
    _, thresholded_diff = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

    # Morphological opening (erosion then dilation) to remove noise
    thresholded_diff = cv2.morphologyEx(
        thresholded_diff, cv2.MORPH_OPEN, kernel=np.ones((5, 5))
    )

    # Find contours in the thresholded difference
    contours, _ = cv2.findContours(
        thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Check each contour
    for contour in contours:
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate speed (change in y-coordinate)
        speed = y - (y - h)

        # Set a threshold for speed
        speed_threshold = 250

        # Check if the object is moving faster than the threshold
        if speed > speed_threshold and y + h > current_frame.shape[0] - 10:
            print("Object moving at a high speed!")

            roi = current_frame[y : y + h, x : x + w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            mean_intensity = np.mean(gray_roi)

            texture_threshold = 500

            if mean_intensity < texture_threshold:
                print("Object moving downward at a high speed with low texture!")
                drum_sound.play()

                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # faces = face_cascade.detectMultiScale(
            #     gray_roi, scaleFactor=1.3, minNeighbors=5
            # )

            # if len(faces) == 0:
            #     print("Object moving downward at a high speed!")
            #     drum_sound.play()

            #     # Draw a bounding box around the moving object
            #     cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the original and moving only frames
    cv2.imshow("Original", current_frame)
    cv2.imshow("Moving only", thresholded_diff)

    # Update the previous frame
    prev_frame = current_frame.copy()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
