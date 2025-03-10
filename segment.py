import cv2
import numpy as np
import argparse
import os
def nothing(x):
    pass

parser = argparse.ArgumentParser(description="Segmentation of a video file.")
parser.add_argument("--video", type=str, help="Path to the video file.")
args = parser.parse_args()

video_path = args.video if args.video else input("Enter the video file path: ").strip()
if not os.path.exists(video_path):
    print(f"Error: File '{video_path}' not found. Please check the path.")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'.")
    exit()

cv2.namedWindow("Tracking")
cv2.namedWindow("Frame")
cv2.namedWindow("Mask")
cv2.namedWindow("Result")
cv2.setWindowProperty("Tracking", cv2.WND_PROP_TOPMOST, 1) 
cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty("Mask", cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty("Result", cv2.WND_PROP_TOPMOST, 1)

cv2.createTrackbar("LH", "Tracking", 0, 179, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 179, 179, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
while True:
    if cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
        print("Tracking window closed. Exiting...")
        break
    ret, img = cap.read()
    if not ret:
        print("Warning: Unable to read frame or video has ended.")
        break
    img = cv2.resize(img, (512, 512))
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
    cv2.imshow("Frame", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", res)
    key=cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()
