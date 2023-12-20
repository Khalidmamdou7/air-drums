import cv2
import numpy as np
import pygame

pygame.init()

# Open a video capture object
cap = cv2.VideoCapture(0)

# Read the first frame to initialize the background
ret, prev_frame = cap.read()

drum_sound = pygame.mixer.Sound("drum.wav")

# Load drum image
drum = cv2.resize(cv2.imread("./drum.png"), (200, 100), interpolation=cv2.INTER_CUBIC)
drum_reversed = cv2.flip(drum, 1)


# Minimum aspect ratio for stick-like objects
min_aspect_ratio = 3.0

# Minimum distance from the bottom edge for stick detection
min_bottom_edge_threshold = 50

# Define the lower and upper HSV ranges for red sticks
red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])

while True:
    # Read a frame from the video
    ret, current_frame = cap.read()

    if not ret:
        break  # Break the loop if the video is finished

    x_position = (current_frame.shape[1] - drum.shape[1]) // 2 + 150
    y_position = (current_frame.shape[0] - drum.shape[0]) // 2 + 130

    current_frame[
        y_position : y_position + drum.shape[0], x_position : x_position + drum.shape[1]
    ] = cv2.addWeighted(
        drum,
        1,
        current_frame[
            y_position : y_position + drum.shape[0],
            x_position : x_position + drum.shape[1],
        ],
        1,
        0,
    )

    x_position_reversed = 50
    y_position_reversed = (current_frame.shape[0] - drum_reversed.shape[0]) // 2 + 130

    current_frame[
        y_position_reversed : y_position_reversed + drum_reversed.shape[0],
        x_position_reversed : x_position_reversed + drum_reversed.shape[1],
    ] = cv2.addWeighted(
        drum_reversed,
        1,
        current_frame[
            y_position_reversed : y_position_reversed + drum_reversed.shape[0],
            x_position_reversed : x_position_reversed + drum_reversed.shape[1],
        ],
        1,
        0,
    )

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

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

    # Create a mask for red sticks
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

    # Combine the motion mask and red stick mask
    final_mask = cv2.bitwise_and(thresholded_diff, red_mask)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Check each contour
    for contour in contours:
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the aspect ratio of the bounding box
        aspect_ratio = w / h

        # Check if the contour has a minimum aspect ratio and is close to the bottom edge
        if (
            aspect_ratio > min_aspect_ratio
            and y + h > current_frame.shape[0] - min_bottom_edge_threshold
        ):
            # Print (hit) or perform any action here
            print("Moving Red Stick detected!")
            drum_sound.play()

    # Show the original, moving only, and final frames
    cv2.imshow("Original", current_frame)
    cv2.imshow("Moving only", thresholded_diff)
    cv2.imshow("Final Mask", final_mask)

    # Update the previous frame
    prev_frame = current_frame.copy()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
