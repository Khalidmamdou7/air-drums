import cv2
import numpy as np
import pygame


pygame.init()
# Open a video capture object
cap = cv2.VideoCapture(0)

# Read the first frame to initialize the background
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
    thresholded_diff = cv2.morphologyEx(thresholded_diff, cv2.MORPH_OPEN, kernel=np.ones((5, 5)))

    # Find contours in the thresholded difference
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check each contour
    for contour in contours:
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if the bottom edge of the object is close to the height of the frame
        if y+5 > current_frame.shape[0] - 1:  # You can adjust the threshold (10 in this case)
            # Print (hit) or perform any action here
            print("Object hit the bottom!")
            drum_sound.play()

    # Show the original and moving only frames
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

