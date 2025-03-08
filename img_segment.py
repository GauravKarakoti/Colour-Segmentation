import cv2
import numpy as np
import os
import argparse

def nothing(x):
    pass

parser = argparse.ArgumentParser(description="Segmentation of an image file.")
parser.add_argument("--image", type=str, help="Path to the image file.")
args = parser.parse_args()

image_path = args.image if args.image else input("Enter the image file path: ").strip()

if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found. Please check the path.")
    exit()

img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not open image file '{image_path}'.")
    exit()

img = cv2.resize(img, (512, 512))

cv2.namedWindow("Tracking")
cv2.namedWindow("Original")
cv2.namedWindow("Mask")
cv2.namedWindow("Result")

cv2.setWindowProperty("Tracking", cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty("Original", cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty("Mask", cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty("Result", cv2.WND_PROP_TOPMOST, 1)

cv2.createTrackbar("LH", "Tracking", 0, 179, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 225, 225, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")
    
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(img,l_b,u_b)
    res = cv2.bitwise_and(img, img, mask=mask)
    
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", res)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()