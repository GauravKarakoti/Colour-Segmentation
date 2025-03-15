import cv2  # OpenCV library for image processing
import argparse  # Argument parsing module
from segmentation_utils import *  # Import custom segmentation utilities

# Initialize argument parser to allow users to provide an image path via command line
parser = argparse.ArgumentParser(description="Segmentation of an image file.")
parser.add_argument("--image", type=str, help="Path to the image file.")
args = parser.parse_args()

# Get the image path from command line argument or ask the user for input
image_path = args.image if args.image else input("Enter the image file path: ").strip()

# Try to load the image, handle errors if the file is not found
try:
    img = load_image(image_path)  # Load image using a function from segmentation_utils
except ( FileNotFoundError, ValueError, PermissionError, RuntimeError ) as e:
    print(e)  # Print error message if file not found
    exit(1)  # Exit the program

# Create trackbars for real-time tuning of segmentation parameters
create_trackbars("Tracking")

# Create display windows to show results
create_display_windows("img")

# Start an infinite loop for interactive segmentation
while True:
    # Check if the tracking window is closed; if so, exit the loop
    if cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
        print("Tracking window closed. Exiting...")
        break

    # Get the current values of the trackbars (lower and upper bounds for segmentation)
    lower, upper = get_trackbar_values("Tracking")
    
    # Apply segmentation mask using the obtained lower and upper bounds
    mask, result = apply_mask(img, lower, upper)
    
    # Display the original image, mask, and result side by side
    display_results(original=img, mask=mask, result=result)
    
    # Wait for 1ms and check if the ESC key (27) is pressed to exit the loop
    if cv2.waitKey(1) == 27:
        break

# Destroy all OpenCV windows once the loop ends
cv2.destroyAllWindows()
