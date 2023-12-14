import cv2
import numpy as np


# Open a video capture object
cap = cv2.VideoCapture(0)

# Read the first frame to initialize the background
ret, prev_frame = cap.read()

while True:
    # Read a frame from the video
    ret, current_frame = cap.read()

    if not ret:
        break  # Break the loop if the video is finished

    # Calculate the absolute difference between the current and previous frames
    diff_frame = cv2.absdiff(current_frame, prev_frame)

    # Convert the difference to grayscale
    gray_diff = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to clearify the moving parts 
    _, thresholded_diff = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

    # morphological opening (erosion then dilation)  to remove noise
    thresholded_diff = cv2.morphologyEx(thresholded_diff, cv2.MORPH_OPEN, kernel=np.ones((5, 5)))

    cv2.imshow('Original', current_frame)
    cv2.imshow('Moving only', thresholded_diff)

    # Update the previous frame 
    prev_frame = current_frame.copy()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

