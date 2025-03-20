import cv2
import argparse
import numpy as np
import logging
from segmentation_utils import *  # Assuming this imports required utilities
import os

# Initialize logging
logging.basicConfig(filename='segmentation_errors.log', level=logging.ERROR)

# Initialize argument parser to allow users to provide a video path via command line
parser = argparse.ArgumentParser(description="Segmentation of a video file.")
parser.add_argument("--video", type=str, help="Path to the video file.")
args = parser.parse_args()

# Get the video path from command line argument or ask the user for input
video_path = args.video if args.video else input("Enter the video file path: ").strip()

# Load the video using the utility function
try:
    cap = load_video(video_path)  # Open the video file
except FileNotFoundError:
    logging.error(f"File not found: {video_path}")
    print("Error: Video file not found. Please check the path and try again.")
    exit(1)
except cv2.error as e:
    logging.error(f"OpenCV error: {str(e)}")
    print("Error: Unable to open video file. Ensure the file is not corrupted and is in a supported format.")
    exit(1)
except Exception as e:
    logging.error(f"Unexpected error: {str(e)}")
    print("Error: An unexpected error occurred while loading the video.")
    exit(1)

if not cap.isOpened():
    raise ValueError("Error opening video file.")

# Create display windows to show results
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)  # Ensure "Tracking" window is created
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)  # Window to display original frame

# Create trackbars for real-time tuning of segmentation parameters
def create_trackbars(window_name):
    # Trackbars to adjust lower and upper bounds of the color range
    cv2.createTrackbar("LH", window_name, 0, 179, nothing)  # Lower Hue
    cv2.createTrackbar("LS", window_name, 0, 255, nothing)  # Lower Saturation
    cv2.createTrackbar("LV", window_name, 0, 255, nothing)  # Lower Value
    cv2.createTrackbar("UH", window_name, 179, 179, nothing)  # Upper Hue
    cv2.createTrackbar("US", window_name, 255, 255, nothing)  # Upper Saturation
    cv2.createTrackbar("UV", window_name, 255, 255, nothing)  # Upper Value
    cv2.createTrackbar("Kernel Size", window_name, 1, 20, nothing)  # Kernel size for morphology

def nothing(x):
    pass  # Placeholder for the trackbar callback (does nothing)

# Create the trackbars
create_trackbars("Tracking")

# Ensure the trackbars and window are initialized properly before the loop
cv2.waitKey(1)  # This ensures the window and trackbars are fully initialized

cv2.resizeWindow("Tracking", 500, 10)

# Start an infinite loop for video processing
while True:

    try:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            print("End of video or unable to read frame. Exiting...")
            break  # Exit when video ends or an error occurs

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get the current values of the trackbars (lower and upper bounds for segmentation)
        lower, upper = get_trackbar_values("Tracking")

        # Handle hue wrapping for colors like red (where lower hue > upper hue)
        if upper[0] < lower[0]:  # Handle hue wrapping case
            mask1, result1 = apply_mask(hsv_frame, lower, np.array([179, upper[1], upper[2]]))
            mask2, result2 = apply_mask(hsv_frame, np.array([0, lower[1], lower[2]]), upper)
            mask = cv2.bitwise_or(mask1, mask2)  # Combine the two masks
            result = cv2.bitwise_or(result1, result2)  # Combine the results
        else:
            mask, result = apply_mask(hsv_frame, lower, upper)

        # Resize video frame to match desired width (e.g., width=512)
        frame = resize_with_aspect_ratio(frame, width=512)

        # Adjustable Morphology Parameters: Dynamically adjust kernel size using trackbars
        kernel_size = cv2.getTrackbarPos("Kernel Size", "Tracking")  # Trackbar to control kernel size
        kernel_size = max(kernel_size, 1)  # Ensure kernel size is at least 1x1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a kernel of specified size
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)  # Apply opening (dilation + erosion)

        # Display the original frame in the "Original" window
        cv2.imshow("Original", frame)

        # Display the mask and result side by side in the "Tracking" window
        display_results(frame=frame, mask=mask, result=result)

        # Wait for 1ms and check if the ESC key (27) is pressed to exit the loop
        if cv2.waitKey(1) == 27:
            break
    except cv2.error as e:
        logging.error(f"OpenCV error during processing: {str(e)}")
        print("Error: An error occurred during video processing. Check the log file for details.")
        break
    except Exception as e:
        logging.error(f"Unexpected error during processing: {str(e)}")
        print("Error: An unexpected error occurred during video processing. Check the log file for details.")
        
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        print("End of video or unable to read frame. Exiting...")
        break  # Exit when video ends or an error occurs

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the current values of the trackbars (lower and upper bounds for segmentation)
    lower, upper = get_trackbar_values("Tracking")

    # Handle hue wrapping for colors like red (where lower hue > upper hue)
    if upper[0] < lower[0]:  # Handle hue wrapping case
        mask1, result1 = apply_mask(hsv_frame, lower, np.array([179, upper[1], upper[2]]))
        mask2, result2 = apply_mask(hsv_frame, np.array([0, lower[1], lower[2]]), upper)
        mask = cv2.bitwise_or(mask1, mask2)  # Combine the two masks
        result = cv2.bitwise_or(result1, result2)  # Combine the results
    else:
        mask, result = apply_mask(hsv_frame, lower, upper)

    # Resize video frame to match desired width (e.g., width=512)
    frame = resize_with_aspect_ratio(frame, width=512)

    # Adjustable Morphology Parameters: Dynamically adjust kernel size using trackbars
    kernel_size = cv2.getTrackbarPos("Kernel Size", "Tracking")  # Trackbar to control kernel size
    kernel_size = max(kernel_size, 1)  # Ensure kernel size is at least 1x1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a kernel of specified size
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)  # Apply opening (dilation + erosion)

    # Convert mask to 3 channels before writing to video
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Get frame dimensions
    frame_height, frame_width = mask.shape[:2]

    # Resize mask and result to match the original frame dimensions
    mask_3ch = cv2.resize(mask_3ch, (frame_width, frame_height))
    result = cv2.resize(result, (frame_width, frame_height))

    # Display the original frame in the "Original" window
    cv2.imshow("Original", frame)

    # Display the mask and result side by side in the "Tracking" window
    display_results(frame=frame, mask=mask, result=result)

    # Wait for 1ms and check if the ESC key (27) or 's' key is pressed
    key = cv2.waitKey(1)
    if key == 27:  # ESC key

        break

# Release video capture and destroy all OpenCV windows
release_video(cap)
cv2.destroyAllWindows()
