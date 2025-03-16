import cv2  # OpenCV library for image processing
import argparse  # Argument parsing module
from segmentation_utils import *  # Import custom segmentation utilities
import numpy as np

# Initialize argument parser to allow users to provide an image path via command line
parser = argparse.ArgumentParser(description="Segmentation of an image file.")
parser.add_argument("--image", type=str, help="Path to the image file.")
args = parser.parse_args()

# Get the image path from command line argument or ask the user for input
image_path = args.image if args.image else input("Enter the image file path: ").strip()

# Try to load the image, handle errors if the file is not found
try:
    img = load_image(image_path)  # Load image using a function from segmentation_utils
except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as e:
    print(f"Error: {e}")  # Print error message if file not found
    exit(1)  # Exit the program

# Create trackbars for real-time tuning of segmentation parameters
create_trackbars("Tracking")

# Create display windows to show results
create_display_windows("img")

# Start an infinite loop for interactive segmentation
while True:
    # Check if any window has been closed; if so, exit the loop
    if cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
        print("Tracking window closed. Exiting...")
        break

    # Get the current values of the trackbars (lower and upper bounds for segmentation)
    lower, upper = get_trackbar_values("Tracking")
    
    # Handle hue wrapping for colors like red (where lower hue > upper hue)
    if upper[0] < lower[0]:  # Handle hue wrapping case
        # Split into two ranges and apply masks separately
        mask1, result1 = apply_mask(img, lower, np.array([179, upper[0][1], upper[0][2]]))
        mask2, result2 = apply_mask(img, np.array([0, lower[0][1], lower[0][2]]), upper)
        mask = cv2.bitwise_or(mask1, mask2)  # Combine the two masks
        result = cv2.bitwise_or(result1, result2)  # Combine the results
    else:
        mask, result = apply_mask(img, lower, upper)
    
    # Resize video frames (if processing video) to match image dimensions (width=512)
    if isinstance(img, np.ndarray):
        img = resize_with_aspect_ratio(img, width=512)

    # Adjustable Morphology Parameters: Dynamically adjust kernel size using trackbars
    kernel_size = cv2.getTrackbarPos("Kernel Size", "Tracking")  # Trackbar to control kernel size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a kernel of specified size
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)  # Apply opening (dilation + erosion)
    
    # Display the original image, mask, and result side by side
    display_results(original=img, mask=mask, result=result)
    
    # Wait for 1ms and check if the ESC key (27) is pressed to exit the loop
    if cv2.waitKey(1) == 27:
        break

# Destroy all OpenCV windows once the loop ends
cv2.destroyAllWindows()
