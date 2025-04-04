import cv2
import argparse
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import logging
from segmentation_utils import *
import time

# Initialize logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


# Initialize argument parser
parser = argparse.ArgumentParser(description="Segmentation of an image file.")
parser.add_argument("--image", type=str, help="Path to the image file.")
args = parser.parse_args()

# Get the image path from command line argument or ask user for input
image_path = args.image if args.image else input("Enter the image file path: ").strip()
if image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
    if image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
        image_path = os.path.join('videos', image_path)  # Ensure the video path is correct




# Try loading the image
try:
    if image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
        print(f"Attempting to load video from path: {image_path}")  # Debug logging for video path
        cap = load_video(image_path)  # Function from segmentation_utils


        # Additional code to handle video processing would go here
        exit(0)  # Placeholder for video processing
    else:
        img = load_image(image_path)  # Function from segmentation_utils

    # Check if the image was loaded successfully
    if img is None:
        raise FileNotFoundError(f"Unable to open image: {image_path}")

    # Convert the image to HSV color space upfront
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Convert back to BGR before displaying
    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
except FileNotFoundError:
    logging.error(f"File not found: {image_path}. Please check the file path and try again.")

    print("Error: File not found. Please check the file path and try again.")
    exit(1)
except ValueError as e:
    logging.error(f"Value error: {e}. Please provide a valid image file.")

    print("Error: Invalid image format. Please provide a valid image file.")
    exit(1)
except PermissionError:
    logging.error(f"Permission denied: {image_path}. Check your file permissions.")

    print("Error: Permission denied. Please check your file permissions.")
    exit(1)
except RuntimeError as e:
    logging.error(f"Runtime error: {e}. Ensure the file is not corrupted and try again.")

    print("Error: Unable to load image. Please ensure the file is not corrupted and try again.")
    exit(1)
except Exception as e:
    logging.error(f"Unexpected error: {e}. Check the error log for details.")

    print("Error: An unexpected error occurred. Please check the error log for details.")
    exit(1)

# Create a named window FIRST
cv2.namedWindow("Tracking",cv2.WINDOW_NORMAL)

# Function to handle trackbar updates
def nothing(x):
    pass

# Create a blank image to ensure trackbars are visible
cv2.imshow("Tracking", np.zeros((100, 600, 3), np.uint8))

cv2.imshow("Tracking", bgr_img)

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

def prompt_filename(title, default_name, valid_extensions):
    root = tk.Tk()
    root.withdraw()  
    filename = simpledialog.askstring(title, f"Enter filename for {title} (without extension):", initialvalue=default_name)
    
    # Handle case where user cancels or closes the dialog
    if not filename:
        print(f"Operation canceled. {title} will not be saved.")
        return None

    return validate_filename_with_extension(filename, default_name, valid_extensions)

# Frame rate control settings
frame_delay = 1 / 30  # Target ~30 FPS
prev_tick = cv2.getTickCount()

# Main loop
while True:
    current_tick = cv2.getTickCount()

    # Check if the "Tracking" window is still open
    if cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
        print("Tracking window closed. Exiting...")
        break

    # Get trackbar values
    lower = np.array([
        validate_numeric_input(min(cv2.getTrackbarPos("LH", "Tracking"), cv2.getTrackbarPos("UH", "Tracking")), 0, 179, 0),
        validate_numeric_input(cv2.getTrackbarPos("LS", "Tracking"), 0, 255, 50),
        validate_numeric_input(cv2.getTrackbarPos("LV", "Tracking"), 0, 255, 50)
    ])
    
    upper = np.array([
        validate_numeric_input(max(cv2.getTrackbarPos("LH", "Tracking"), cv2.getTrackbarPos("UH", "Tracking")), 0, 179, 179),
        validate_numeric_input(cv2.getTrackbarPos("US", "Tracking"), 0, 255, 255),
        validate_numeric_input(cv2.getTrackbarPos("UV", "Tracking"), 0, 255, 255)
    ])

    # Adjust kernel size
    kernel_size = validate_numeric_input(cv2.getTrackbarPos("K_Size", "Tracking"), 1, 30, 1)
    kernel_size = get_valid_kernel_size(kernel_size)  # Ensure kernel size is odd and valid

    # Handle hue wrapping
    if upper[0] < lower[0]:
        mask1, result1 = apply_mask(hsv_img, lower, np.array([179, upper[1], upper[2]]), hsv_converted=True, kernel_size=kernel_size, apply_morph=True)
        mask2, result2 = apply_mask(hsv_img, np.array([0, lower[1], lower[2]]), upper, hsv_converted=True, kernel_size=kernel_size, apply_morph=True)
        mask = cv2.bitwise_or(mask1, mask2)
        result = cv2.bitwise_or(result1, result2)
    else:
        mask, result = apply_mask(hsv_img, lower, upper, hsv_converted=True, kernel_size=kernel_size)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    # Resize image for display
    img_resized = resize_with_aspect_ratio(img, width=512)
    mask_resized = resize_with_aspect_ratio(mask, width=512)
    result_resized = resize_with_aspect_ratio(result, width=512)

    # Display results
    display_results(original=img_resized, mask=mask_resized, result=result_resized)

    # Handle keypress events
    elapsed_time = (current_tick - prev_tick) / cv2.getTickFrequency()
    wait_time = max(1, int((frame_delay - elapsed_time) * 1000))
    
    key = cv2.waitKey(wait_time) # Slight delay to reduce CPU usage
    if key == 27:  # ESC key 
        break
    elif key == ord('s'):  # 's' key to save mask and result
        mask_filename = prompt_filename("Mask Image", "mask", ["png", "jpg"])
        result_filename = prompt_filename("Result Image", "result", ["png", "jpg"])

        # Save only if filenames are provided
        if mask_filename and result_filename:
            # Convert result to BGR before saving
            result_bgr = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

            cv2.imwrite(mask_filename, mask)
            cv2.imwrite(result_filename, result_bgr)

            print(f"Mask saved as {mask_filename}")
            print(f"Result saved as {result_filename}")
        else:
            print("Save operation skipped.")

    prev_tick = current_tick