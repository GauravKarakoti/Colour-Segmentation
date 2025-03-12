import cv2
import argparse
from segmentation_utils import *

parser = argparse.ArgumentParser(description="Segmentation of a video file.")
parser.add_argument("--video", type=str, help="Path to the video file.")
args = parser.parse_args()

video_path = args.video if args.video else input("Enter the video file path: ").strip()
try:
    cap = load_video(video_path)
except Exception as e:
    print(f"Error: {e}")
    exit(1) 

create_trackbars("Tracking")

create_display_windows("video")

while True:
    if cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
        print("Tracking window closed. Exiting...")
        break

    ret, frame = cap.read()
    if not ret:
        print("Warning: Unable to read frame or video has ended.")
        break

    frame = cv2.resize(frame, (512, 512))
    lower_bound, upper_bound = get_trackbar_values("Tracking")
    mask, result = apply_mask(frame, lower_bound, upper_bound)

    display_results(frame=frame, mask=mask, result=result)

    if cv2.waitKey(1) == 27:
        break

release_video(cap)
cv2.destroyAllWindows()
