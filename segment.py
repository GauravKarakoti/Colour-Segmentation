import cv2
import argparse
import numpy as np
import logging
from segmentation_utils import *  # Assuming this imports required utilities
import time

# Initialize logging
logging.basicConfig(filename='segmentation_errors.log', level=logging.ERROR)

# Initialize argument parser
parser = argparse.ArgumentParser(description="Segmentation of a video file.")
parser.add_argument("--video", type=str, help="Path to the video file.")
parser.add_argument("--output", type=str, default="output.avi", help="Path to save the output video.")
parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
args = parser.parse_args()

# Get the video path from command line argument or ask the user for input
video_path = args.video if args.video else input("Enter the video file path: ").strip()
output_path = args.output
fps = args.fps

# Load the video
try:
    cap = load_video(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")
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

# Create display windows
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

# Create trackbars
def create_trackbars(window_name):
    cv2.resizeWindow("Tracking", 500, 300)
    cv2.createTrackbar("LH", window_name, 0, 179, nothing)  # Lower Hue
    cv2.createTrackbar("LS", window_name, 0, 255, nothing)  # Lower Saturation
    cv2.createTrackbar("LV", window_name, 0, 255, nothing)  # Lower Value
    cv2.createTrackbar("UH", window_name, 179, 179, nothing)  # Upper Hue
    cv2.createTrackbar("US", window_name, 255, 255, nothing)  # Upper Saturation
    cv2.createTrackbar("UV", window_name, 255, 255, nothing)  # Upper Value
    cv2.createTrackbar("Kernel Size", window_name, 1, 30, nothing)

def nothing(x):
    pass

create_display_windows(input_type='video')

# Create the trackbars
create_trackbars("Tracking")

cv2.setWindowProperty("Tracking", cv2.WND_PROP_TOPMOST, 1)

# Get frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read the first frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit(1)

height, width = frame.shape[:2]
frame_height, frame_width = 512, 512

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec for AVI format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 3, frame_height))
frame_delay = 1 / fps  # Control FPS

# Initialize the paused flag
paused = False

# Start the video processing loop
while True:
    start_time = time.time()

    try:
        if not paused:
            ret, frame = cap.read()  # Read a frame from the video
            if not ret:
                print("End of video or unable to read frame. Exiting...")
                break  # Exit when video ends or an error occurs

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get the current values of the trackbars (lower and upper bounds for segmentation)
        lower, upper = get_trackbar_values("Tracking")

        kernel_size = cv2.getTrackbarPos("Kernel Size", "Tracking")  # Trackbar to control kernel size
        kernel_size = get_valid_kernel_size(kernel_size)  # Ensure kernel size is at least 1x1

        # Handle hue wrapping for colors like red (where lower hue > upper hue)
        if upper[0] < lower[0]:  # Handle hue wrapping case
            mask1, result1 = apply_mask(hsv_frame, lower, np.array([179, upper[1], upper[2]]),kernel_size=kernel_size)
            mask2, result2 = apply_mask(hsv_frame, np.array([0, lower[1], lower[2]]), upper,kernel_size=kernel_size)
            mask = cv2.bitwise_or(mask1, mask2)  # Combine the two masks
            result = cv2.bitwise_or(result1, result2)  # Combine the results
        else:
            mask, result = apply_mask(hsv_frame, lower, upper,kernel_size=kernel_size)

        # Resize video frame to match desired width (e.g., width=512)
        frame = cv2.resize(frame, (frame_width, frame_height))
        mask = cv2.resize(mask, (frame_width, frame_height))
        result = cv2.resize(result, (frame_width, frame_height))

        # Adjustable Morphology Parameters: Dynamically adjust kernel size using trackbars
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a kernel of specified size
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)  # Apply opening (dilation + erosion)

        # Display the original frame in the "Original" window
        cv2.imshow("Original", frame)

        # Display the mask and result side by side in the "Tracking" window
        display_results(frame=frame, mask=mask, result=result)

        # Convert mask to 3-channel image for saving
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Combine results into a single frame
        combined_output = np.hstack((frame, mask_3ch, result))
        
        # Write the combined frame to the output video
        out.write(combined_output)

        key = cv2.waitKey(10) & 0xFF

        # Exit on ESC key
        if key == 27:
            break
        elif key == 32:  # Toggle pause on spacebar press
            paused = not paused
            if paused:
                print("Video paused. Press SPACE to resume.")
            else:
                print("Video resumed.")

        # ---- FPS Control ----
        elapsed_time = time.time() - start_time
        time.sleep(max(0, frame_delay - elapsed_time))

    except cv2.error as e:
        logging.error(f"OpenCV error during processing: {str(e)}")
        print("Error: An error occurred during video processing. Check the log file for details.")
        break
    except Exception as e:
        logging.error(f"Unexpected error during processing: {str(e)}")
        print("Error: An unexpected error occurred during video processing. Check the log file for details.")
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()