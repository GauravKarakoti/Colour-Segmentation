import cv2
import argparse
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import logging
from segmentation_utils import *

# Initialize logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

# Initialize argument parser
parser = argparse.ArgumentParser(description="Segmentation of an image file.")
parser.add_argument("--image", type=str, help="Path to the image file.")
args = parser.parse_args()

# Get the image path from command line argument or ask user for input
image_path = args.image if args.image else input("Enter the image file path: ").strip()

# Try loading the image
try:
    img = load_image(image_path)  # Function from segmentation_utils
    if img is None:
        raise FileNotFoundError(f"Unable to open image: {image_path}")
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
except FileNotFoundError:
    logging.error(f"File not found: {image_path}")
    print("Error: File not found. Please check the file path and try again.")
    exit(1)
except ValueError as e:
    logging.error(f"Value error: {e}")
    print("Error: Invalid image format. Please provide a valid image file.")
    exit(1)
except PermissionError:
    logging.error(f"Permission denied: {image_path}")
    print("Error: Permission denied. Please check your file permissions.")
    exit(1)
except RuntimeError as e:
    logging.error(f"Runtime error: {e}")
    print("Error: Unable to load image. Please ensure the file is not corrupted and try again.")
    exit(1)
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    print("Error: An unexpected error occurred. Please check the error log for details.")
    exit(1)

# Create a named window FIRST
cv2.namedWindow("Tracking",cv2.WINDOW_NORMAL)

# Function to handle trackbar updates
def nothing(x):
    pass

# Create a blank image to ensure trackbars are visible
cv2.imshow("Tracking", np.zeros((100, 600, 3), np.uint8))

# Create trackbars
cv2.setWindowProperty("Tracking", cv2.WND_PROP_TOPMOST, 1)
cv2.createTrackbar("LH", "Tracking", 0, 179, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 179, 179, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
cv2.createTrackbar("K_Size", "Tracking", 1, 30, nothing)

create_display_windows(input_type='img')

def prompt_filename(title, default_name):
    root = tk.Tk()
    root.withdraw()  
    filename = simpledialog.askstring(title, f"Enter filename for {title} (without extension):", initialvalue=default_name)
    return filename if filename else default_name

# Main loop
while True:
    # Check if the "Tracking" window is still open
    if cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
        print("Tracking window closed. Exiting...")
        break

    # Get trackbar values
    lower = np.array([
        cv2.getTrackbarPos("LH", "Tracking"),
        cv2.getTrackbarPos("LS", "Tracking"),
        cv2.getTrackbarPos("LV", "Tracking")
    ])
    
    upper = np.array([
        cv2.getTrackbarPos("UH", "Tracking"),
        cv2.getTrackbarPos("US", "Tracking"),
        cv2.getTrackbarPos("UV", "Tracking")
    ])

    # Handle hue wrapping
    if upper[0] < lower[0]:
        mask1, result1 = apply_mask(hsv_img, lower, np.array([179, upper[1], upper[2]]))
        mask2, result2 = apply_mask(hsv_img, np.array([0, lower[1], lower[2]]), upper)
        mask = cv2.bitwise_or(mask1, mask2)
        result = cv2.bitwise_or(result1, result2)
    else:
        mask, result = apply_mask(hsv_img, lower, upper)

    # Resize image for display
    img_resized = resize_with_aspect_ratio(img, width=512)

    # Adjust kernel size
    kernel_size = get_valid_kernel_size(cv2.getTrackbarPos("K_Size", "Tracking"))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    # Display results
    display_results(original=img_resized, mask=mask, result=result)


    # Handle keypress events
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    elif key == ord('s'):  # 's' key to save mask and result
        mask_filename = prompt_filename("Mask Image", "mask") + ".png"
        result_filename = prompt_filename("Result Image", "result") + ".png"

        cv2.imwrite(mask_filename, mask)
        cv2.imwrite(result_filename, result)

        print(f"Mask saved as {mask_filename}")
        print(f"Result saved as {result_filename}")


# Clean up
cv2.destroyAllWindows()
