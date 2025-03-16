import cv2  # OpenCV for video processing
import argparse  # Argument parsing module
from segmentation_utils import *  # Import custom segmentation utilities

# Initialize argument parser to allow users to provide a video path via command line
parser = argparse.ArgumentParser(description="Segmentation of a video file.")
parser.add_argument("--video", type=str, help="Path to the video file.")
args = parser.parse_args()

# Get the video path from command line argument or ask the user for input
video_path = args.video if args.video else input("Enter the video file path: ").strip()

# Try to load the video, handle errors if the file is not found or can't be opened
try:
    cap = load_video(video_path)  # Load video using a function from segmentation_utils
except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as e:
    print(f"Error: {e}")  # Print error message if unable to open video
    exit(1)  # Exit the program

# Create trackbars for real-time tuning of segmentation parameters
create_trackbars("Tracking")

# Create display windows to show results
create_display_windows("video")

# Start an infinite loop to process each frame
while True:
    # Check if the tracking window is closed; if so, exit the loop
    if cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
        print("Tracking window closed. Exiting...")
        break

    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Warning: Unable to read frame or video has ended.")
        break

    # Resize the frame to 512x512 for consistent processing
    frame = cv2.resize(frame, (512, 512))

    # Get the current values of the trackbars (lower and upper bounds for segmentation)
    lower_bound, upper_bound = get_trackbar_values("Tracking")
    
    # Handle hue wrapping if lower hue is greater than upper hue
    if isinstance(upper_bound, tuple):  # Handle hue wrapping case
        mask1, result1 = apply_mask(frame, lower_bound, upper_bound[0])
        mask2, result2 = apply_mask(frame, lower_bound, upper_bound[1])
        mask = cv2.bitwise_or(mask1, mask2)  # Combine the two masks
        result = cv2.bitwise_or(result1, result2)  # Combine the results
    else:
        mask, result = apply_mask(frame, lower_bound, upper_bound)
    
    # Display the original frame, mask, and result side by side
    display_results(frame=frame, mask=mask, result=result)
    
    # Wait for 1ms and check if the ESC key (27) is pressed to exit the loop
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object to free resources
release_video(cap)

# Destroy all OpenCV windows once the loop ends
cv2.destroyAllWindows()
