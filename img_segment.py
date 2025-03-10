import cv2
import numpy as np
import os
import argparse
from segmentation_utils import load_image, create_trackbars, get_trackbar_values, apply_mask, display_results, create_display_windows

def nothing(x):
    pass

parser = argparse.ArgumentParser(description="Segmentation of an image file.")
parser.add_argument("--image", type=str, help="Path to the image file.")
args = parser.parse_args()

image_path = args.image if args.image else input("Enter the image file path: ").strip()
img = load_image(image_path)

if img is None:
    exit()

# Create trackbars
create_trackbars("Tracking")

# Create all necessary windows
create_display_windows(type="img")

while True:
    if cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
        print("Tracking window closed. Exiting...")
        break

    lower, upper = get_trackbar_values("Tracking")
    mask, result = apply_mask(img, lower, upper)
    
    display_results(img, mask, result, None)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()