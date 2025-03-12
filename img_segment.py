import cv2
import argparse
from segmentation_utils import *

parser = argparse.ArgumentParser(description="Segmentation of an image file.")
parser.add_argument("--image", type=str, help="Path to the image file.")
args = parser.parse_args()

image_path = args.image if args.image else input("Enter the image file path: ").strip()
try:
    img = load_image(image_path)
except FileNotFoundError as e:
    print(e)
    exit(1)

# Create trackbars
create_trackbars("Tracking")

create_display_windows("img")

while True:
    if cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
        print("Tracking window closed. Exiting...")
        break

    lower, upper = get_trackbar_values("Tracking")
    mask, result = apply_mask(img, lower, upper)
    display_results(original=img, mask=mask, result=result)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()